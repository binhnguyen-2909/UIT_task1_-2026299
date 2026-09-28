"""V29-A0.2 BGE-M3 late-interaction diagnostic.

This is deliberately limited to the 24 queries whose gold is absent from the
existing c80.  It runs after the dense gate, uses the fixed coarse pool
``existing c120 ∪ dense top500``, and caches token vectors only for documents
in that pool.  No fine-tuning or threshold search is performed.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
V27_EXPERIMENTS = SCRIPT_DIR.parent / "v27"
if str(V27_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(V27_EXPERIMENTS))

from v27_common import load_metadata, read_json, sha256_file, write_json, write_jsonl  # noqa: E402


def _stage1(path: Path, questions: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    payload = read_json(path)
    result = {str(row["query_id"]): [str(doc) for doc in candidates] for row, candidates in zip(payload["queries"], payload["stage1_candidates"], strict=True)}
    for qid, row in questions.items():
        if result[qid][:35] != list(row["c35"]):
            raise ValueError(f"{qid}: frozen c35 mismatch")
    return result


def _dense_pools(path: Path) -> dict[str, list[dict[str, Any]]]:
    return {str(row["qid"]): row["candidates"] for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}


def _focus_questions(questions: dict[str, dict[str, Any]], stage1: dict[str, list[str]]) -> list[str]:
    return [qid for qid, row in questions.items() if not set(row["references"]) & set(stage1[qid][:80])]


def _coarse_pools(focus: list[str], stage1: dict[str, list[str]], dense: dict[str, list[dict[str, Any]]], dense_cutoff: int) -> dict[str, list[str]]:
    return {qid: list(dict.fromkeys(stage1[qid][:120] + [item["doc_id"] for item in dense[qid][:dense_cutoff]])) for qid in focus}


def _select_chunks(path: Path, wanted: set[str], max_chunks: int) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            # Parsing all rows is necessary because the manifest is sorted by
            # doc id but is kept streaming to avoid another multi-GB object.
            row = json.loads(line)
            doc_id = str(row["doc_id"])
            if doc_id not in wanted:
                continue
            grouped[doc_id].append(row)
    selected: dict[str, list[dict[str, Any]]] = {}
    for doc_id in sorted(grouped, key=lambda value: (int(value) if value.isdigit() else 10**18, value)):
        rows = grouped[doc_id]
        if len(rows) <= max_chunks:
            selected[doc_id] = rows
            continue
        head = max(1, max_chunks // 4)
        tail = max(1, max_chunks // 4)
        middle = max_chunks - head - tail
        step = (len(rows) - head - tail) / max(middle, 1)
        indices = list(range(head))
        indices.extend(min(len(rows) - tail - 1, head + int(round((i + 0.5) * step))) for i in range(middle))
        indices.extend(range(len(rows) - tail, len(rows)))
        selected[doc_id] = [rows[index] for index in sorted(set(indices))]
    return selected


def _encode_token_vectors(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: list[str],
    *,
    max_length: int,
    batch_size: int,
    device: torch.device,
    linear: torch.nn.Module,
) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    model.eval()
    linear.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                hidden = model(**inputs, return_dict=True).last_hidden_state
                transformed = linear(hidden[:, 1:, :])
            mask = inputs["attention_mask"][:, 1:].bool()
            for index in range(len(batch)):
                result.append(transformed[index][mask[index]].float().cpu().numpy().astype(np.float16, copy=False))
            del inputs, hidden, transformed, mask
    return result


def _save_cache(path: Path, vectors: dict[str, list[np.ndarray]], chunk_rows: dict[str, list[dict[str, Any]]]) -> None:
    flat: list[np.ndarray] = []
    rows = []
    offset = 0
    for doc_id in sorted(vectors, key=lambda value: (int(value) if value.isdigit() else 10**18, value)):
        for chunk_index, vector in enumerate(vectors[doc_id]):
            start = offset
            flat.append(vector)
            offset += len(vector)
            rows.append({"doc_id": doc_id, "chunk_index": chunk_index, "start": start, "end": offset, "token_count": len(vector), "chunk_id": chunk_rows[doc_id][chunk_index]["chunk_id"]})
    matrix = np.concatenate(flat, axis=0) if flat else np.zeros((0, 1024), dtype=np.float16)
    np.save(path.with_suffix(".f16.npy"), matrix)
    write_json(path, {"kind": "legalir_v29_a0_multivector_cache", "schema_version": 1, "holdout_opened": False, "public_labels_used": False, "dtype": str(matrix.dtype), "shape": list(matrix.shape), "documents": len(vectors), "chunks": len(rows), "rows": rows})


def _load_cache(path: Path) -> dict[str, list[np.ndarray]]:
    metadata = read_json(path)
    matrix = np.load(path.with_suffix(".f16.npy"), mmap_mode="r")
    result: dict[str, list[np.ndarray]] = collections.defaultdict(list)
    for row in metadata["rows"]:
        result[str(row["doc_id"])].append(matrix[int(row["start"]) : int(row["end"])])
    return dict(result)


def _score_query(qvec: torch.Tensor, doc_chunks: dict[str, list[np.ndarray]], candidate_docs: list[str], device: torch.device, batch_size: int) -> dict[str, float]:
    # One score per chunk, then article/document score is max over its chunks.
    passage_vectors: list[np.ndarray] = []
    passage_docs: list[str] = []
    for doc_id in candidate_docs:
        for vector in doc_chunks.get(doc_id, []):
            passage_vectors.append(vector)
            passage_docs.append(doc_id)
    scores_by_doc: dict[str, float] = {doc_id: float("-inf") for doc_id in candidate_docs}
    with torch.inference_mode():
        for start in range(0, len(passage_vectors), batch_size):
            rows = passage_vectors[start : start + batch_size]
            max_tokens = max(len(row) for row in rows)
            padded = torch.zeros((len(rows), max_tokens, qvec.shape[-1]), dtype=torch.float16, device=device)
            mask = torch.zeros((len(rows), max_tokens), dtype=torch.bool, device=device)
            for index, row in enumerate(rows):
                value = torch.as_tensor(row, dtype=torch.float16, device=device)
                padded[index, : len(row)] = value
                mask[index, : len(row)] = True
            similarity = torch.einsum("qd,btd->bqt", qvec, padded)
            similarity = similarity.masked_fill(~mask[:, None, :], -torch.inf)
            chunk_scores = similarity.max(dim=-1).values.sum(dim=-1).float().cpu().numpy()
            for doc_id, score in zip(passage_docs[start : start + len(rows)], chunk_scores):
                scores_by_doc[doc_id] = max(scores_by_doc[doc_id], float(score))
            del padded, mask, similarity
    return scores_by_doc


def _oracle_curve(questions: dict[str, dict[str, Any]], stage1: dict[str, list[str]], ranked: dict[str, list[dict[str, Any]]], eval_qids: list[str], cutoffs: list[int]) -> dict[str, Any]:
    baseline = float(np.mean([len(set(questions[qid]["anchor"]) & set(questions[qid]["references"])) / max(len(questions[qid]["references"]), 1) for qid in eval_qids]))
    result = {}
    for cutoff in cutoffs:
        values = []
        unique = 0
        for qid in eval_qids:
            row = questions[qid]
            refs = set(row["references"])
            existing = set(stage1[qid][:80])
            candidates = [item["doc_id"] for item in ranked.get(qid, [])[:cutoff]]
            union = list(dict.fromkeys(stage1[qid][:80] + candidates))
            gold = [doc for doc in union[4:] if doc in refs]
            answer = list(row["anchor"][:4]) + ([gold[0]] if gold else [row["anchor"][4]])
            values.append(len(set(answer) & refs) / max(len(refs), 1))
            unique += len((set(candidates) & refs) - existing)
        result[str(cutoff)] = {"cutoff": cutoff, "unique_gold_not_in_existing_c80": unique, "locked_top4_oracle_recall": float(np.mean(values)), "locked_top4_oracle_gain_vs_baseline": float(np.mean(values) - baseline)}
    return {"evaluation_queries": len(eval_qids), "baseline_recall": baseline, "cutoffs": result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--colbert-linear", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--stage1-audit", type=Path, required=True)
    parser.add_argument("--dense-pools", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunks-per-doc", type=int, default=8)
    parser.add_argument("--passage-batch-size", type=int, default=32)
    parser.add_argument("--score-batch-size", type=int, default=128)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    questions = load_metadata(args.metadata)
    stage1 = _stage1(args.stage1_audit, questions)
    dense = _dense_pools(args.dense_pools)
    focus = _focus_questions(questions, stage1)
    coarse = _coarse_pools(focus, stage1, dense, 500)
    wanted = set().union(*(set(items) for items in coarse.values()))
    print(json.dumps({"focus_queries": len(focus), "coarse_unique_docs": len(wanted)}, ensure_ascii=False), flush=True)
    selected = {} if args.reuse_cache else _select_chunks(args.manifest, wanted, args.chunks_per_doc)
    flat_texts = [row["text"] for doc_id in sorted(selected, key=lambda value: (int(value) if value.isdigit() else 10**18, value)) for row in selected[doc_id]]
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model.resolve()), local_files_only=True)
    model = AutoModel.from_pretrained(str(args.model.resolve()), local_files_only=True, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32)
    model.to(device)
    linear = torch.nn.Linear(1024, 1024, bias=True)
    linear.load_state_dict(torch.load(args.colbert_linear, map_location="cpu"))
    linear.to(device)
    started = time.perf_counter()
    if args.reuse_cache:
        doc_chunks = _load_cache(out / "doc_token_cache.json")
    else:
        encoded = _encode_token_vectors(model, tokenizer, flat_texts, max_length=256, batch_size=args.passage_batch_size, device=device, linear=linear)
        doc_chunks: dict[str, list[np.ndarray]] = {}
        cursor = 0
        for doc_id in sorted(selected, key=lambda value: (int(value) if value.isdigit() else 10**18, value)):
            count = len(selected[doc_id])
            doc_chunks[doc_id] = encoded[cursor : cursor + count]
            cursor += count
        _save_cache(out / "doc_token_cache.json", doc_chunks, selected)
    query_vectors = _encode_token_vectors(model, tokenizer, [questions[qid]["query"] for qid in focus], max_length=128, batch_size=args.passage_batch_size, device=device, linear=linear)
    ranked: dict[str, list[dict[str, Any]]] = {}
    details = []
    for qid, qvector in zip(focus, query_vectors):
        qvec = torch.as_tensor(qvector, dtype=torch.float16, device=device)
        available_docs = [doc_id for doc_id in coarse[qid] if doc_id in doc_chunks]
        missing_docs = [doc_id for doc_id in coarse[qid] if doc_id not in doc_chunks]
        scores = _score_query(qvec, doc_chunks, available_docs, device, args.score_batch_size)
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        ranked[qid] = [{"doc_id": doc_id, "score": score, "rank": rank + 1, "is_raw_gold": doc_id in set(questions[qid]["references"]), "is_new_vs_c80": doc_id not in set(stage1[qid][:80])} for rank, (doc_id, score) in enumerate(ordered)]
        refs = set(questions[qid]["references"])
        details.append({"qid": qid, "gold": sorted(refs), "gold_ranks": {doc: next((item["rank"] for item in ranked[qid] if item["doc_id"] == doc), None) for doc in refs}, "top5": ranked[qid][:5], "top80_gold": [item["doc_id"] for item in ranked[qid][:80] if item["is_raw_gold"]], "unique_gold_top80_vs_c80": sorted({item["doc_id"] for item in ranked[qid][:80] if item["is_raw_gold"] and item["is_new_vs_c80"]}), "unscored_missing_context_docs": missing_docs})
    write_jsonl(out / "multivector_ranked_pools.jsonl", ({"qid": qid, "candidates": ranked[qid]} for qid in focus))
    recovery = {
        "kind": "legalir_v29_a0_multivector_recovery_report",
        "schema_version": 1,
        "split": "clean_complement_focus_absent_c80",
        "holdout_opened": False,
        "public_labels_used": False,
        "model": "BAAI/bge-m3",
        "focus_queries": len(focus),
        "coarse_pool": "existing c120 union dense top500",
        "documents_cached": len(doc_chunks),
        "chunks_cached": sum(len(value) for value in doc_chunks.values()),
        "cache_reused": bool(args.reuse_cache),
        "chunk_aggregation": "MaxSim per chunk then max over chunks per document",
        "max_query_tokens": 128,
        "max_passage_tokens": 256,
        "elapsed_encoding_seconds": time.perf_counter() - started,
        "details": details,
    }
    write_json(out / "recovery_report.json", recovery)
    unique_rows = [{"qid": row["qid"], "unique_gold_top80_vs_c80": row["unique_gold_top80_vs_c80"], "gold_ranks": row["gold_ranks"]} for row in details]
    unique_count = sum(len(row["unique_gold_top80_vs_c80"]) for row in unique_rows)
    write_json(out / "unique_recovery.json", {"kind": "legalir_v29_a0_multivector_unique_recovery", "schema_version": 1, "split": "clean_complement_focus_absent_c80", "holdout_opened": False, "public_labels_used": False, "focus_queries": len(focus), "unique_gold_recovered_top80_vs_c80": unique_count, "rows": unique_rows})
    oracle = _oracle_curve(questions, stage1, ranked, focus, [5, 20, 50, 80, 120])
    write_json(out / "oracle_union_curve.json", {"kind": "legalir_v29_a0_multivector_oracle_union_curve", "schema_version": 1, "split": "clean_complement_focus_only_for_multivector_rankings", "holdout_opened": False, "public_labels_used": False, **oracle})
    gate = {"unique_recovery_ge_8": unique_count >= 8, "oracle_gain_ge_0.006": oracle["cutoffs"]["80"]["locked_top4_oracle_gain_vs_baseline"] >= 0.006, "decision": "proceed_to_a1_stage2_rescore" if unique_count >= 8 and oracle["cutoffs"]["80"]["locked_top4_oracle_gain_vs_baseline"] >= 0.006 else "stop_a0_multivector_and_open_v29_b_or_v30"}
    recovery["gate"] = gate
    write_json(out / "recovery_report.json", recovery)
    print(json.dumps({"recovery": str((out / 'recovery_report.json').resolve()), "gate": gate}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
