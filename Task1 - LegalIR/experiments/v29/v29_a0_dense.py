"""V29-A0.1 BGE-M3 dense-only reach test on the clean complement.

The corpus manifest is hierarchical, but the first reach test samples a small
deterministic set of chunks per document.  This keeps the zero-shot dense
screen cheap and avoids building a whole-corpus ColBERT index before there is
evidence that dense retrieval can recover new gold documents.
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
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
V27_EXPERIMENTS = SCRIPT_DIR.parent / "v27"
if str(V27_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(V27_EXPERIMENTS))

from v27_common import load_metadata, read_json, sha256_file, write_json, write_jsonl  # noqa: E402


def _stage1_candidates(path: Path, questions: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    payload = read_json(path)
    result = {str(row["query_id"]): [str(doc) for doc in candidates] for row, candidates in zip(payload["queries"], payload["stage1_candidates"], strict=True)}
    if set(questions) - set(result):
        raise ValueError("stage1 audit does not cover current clean-complement metadata")
    for qid, row in questions.items():
        if result[qid][:35] != list(row["c35"]):
            raise ValueError(f"{qid}: Stage1 top35 does not match frozen c35")
    return result


def _sample_manifest(path: Path, max_chunks_per_doc: int) -> tuple[list[str], list[str], dict[str, int]]:
    """Keep head/tail/interior chunks without materializing the 2.4GB manifest."""

    texts: list[str] = []
    doc_ids: list[str] = []
    doc_counts: dict[str, int] = {}
    current_doc = None
    current_rows: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal current_rows, current_doc
        if current_doc is None:
            return
        count = len(current_rows)
        doc_counts[str(current_doc)] = count
        if count <= max_chunks_per_doc:
            selected = current_rows
        else:
            head = max(1, max_chunks_per_doc // 4)
            tail = max(1, max_chunks_per_doc // 4)
            interior_count = max_chunks_per_doc - head - tail
            step = (count - head - tail) / max(interior_count, 1)
            indices = list(range(head))
            indices.extend(min(count - tail - 1, head + int(round((i + 0.5) * step))) for i in range(interior_count))
            indices.extend(range(count - tail, count))
            selected = [current_rows[index] for index in sorted(set(indices))]
        for row in selected:
            texts.append(str(row["text"]))
            doc_ids.append(str(row["doc_id"]))
        current_rows = []

    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = str(row["doc_id"])
            if current_doc is None:
                current_doc = doc_id
            if doc_id != current_doc:
                flush()
                current_doc = doc_id
            current_rows.append(row)
    flush()
    return texts, doc_ids, doc_counts


def _benchmark(model: SentenceTransformer, texts: list[str], device: str, batch_sizes: list[int]) -> list[dict[str, Any]]:
    sample = texts[:1000]
    rows = []
    for batch_size in batch_sizes:
        if torch.cuda.is_available() and device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(torch.device(device))
        started = time.perf_counter()
        model.encode(sample, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        if torch.cuda.is_available() and device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        token_estimate = sum(max(1, len(text.split())) for text in sample)
        row = {
            "chunks": len(sample),
            "batch_size": batch_size,
            "elapsed_seconds": elapsed,
            "chunks_per_second": len(sample) / max(elapsed, 1e-9),
            "tokens_per_second_word_estimate": token_estimate / max(elapsed, 1e-9),
            "device": device,
        }
        if torch.cuda.is_available() and device.startswith("cuda"):
            row.update({
                "peak_memory_allocated_mb": torch.cuda.max_memory_allocated(torch.device(device)) / 1024**2,
                "peak_memory_reserved_mb": torch.cuda.max_memory_reserved(torch.device(device)) / 1024**2,
            })
        rows.append(row)
    return rows


def _encode_chunks(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    return np.asarray(model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True), dtype=np.float16)


def _dense_top_docs(
    model: SentenceTransformer,
    queries: list[str],
    chunk_vectors: np.ndarray,
    chunk_doc_ids: list[str],
    doc_ids: list[str],
    *,
    batch_size: int,
    device: str,
    top_k: int,
) -> dict[str, list[dict[str, Any]]]:
    device_obj = torch.device(device)
    vectors = torch.as_tensor(chunk_vectors, dtype=torch.float16, device=device_obj)
    doc_index = {doc_id: index for index, doc_id in enumerate(doc_ids)}
    chunk_doc_index = torch.tensor([doc_index[doc_id] for doc_id in chunk_doc_ids], dtype=torch.long, device=device_obj)
    result: dict[str, list[dict[str, Any]]] = {}
    for start in range(0, len(queries), batch_size):
        batch_queries = queries[start : start + batch_size]
        qvec = model.encode(batch_queries, batch_size=batch_size, show_progress_bar=False, convert_to_tensor=True, normalize_embeddings=True)
        if qvec.dtype != torch.float16:
            qvec = qvec.to(dtype=torch.float16)
        with torch.inference_mode():
            scores = qvec @ vectors.T
            doc_scores = torch.full((len(batch_queries), len(doc_ids)), -torch.inf, dtype=scores.dtype, device=device_obj)
            expanded_index = chunk_doc_index.unsqueeze(0).expand(len(batch_queries), -1)
            doc_scores.scatter_reduce_(1, expanded_index, scores, reduce="amax", include_self=True)
            values, indices = torch.topk(doc_scores, k=min(top_k, len(doc_ids)), dim=1)
        values_cpu = values.float().cpu().numpy()
        indices_cpu = indices.cpu().numpy()
        for offset, qid in enumerate(list(queries[start : start + len(batch_queries)])):
            result[str(qid)] = [{"doc_id": doc_ids[int(index)], "score": float(score), "rank": rank + 1} for rank, (index, score) in enumerate(zip(indices_cpu[offset], values_cpu[offset]))]
        del qvec, scores, doc_scores, expanded_index
    del vectors, chunk_doc_index
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def _query_dense_top_docs(model: SentenceTransformer, query_rows: list[tuple[str, str]], chunk_vectors: np.ndarray, chunk_doc_ids: list[str], doc_ids: list[str], *, batch_size: int, device: str, top_k: int) -> dict[str, list[dict[str, Any]]]:
    qids = [qid for qid, _ in query_rows]
    queries = [query for _, query in query_rows]
    # Keep qids separate from query texts; _dense_top_docs uses positional
    # outputs and is called here only after query embeddings are produced.
    device_obj = torch.device(device)
    vectors = torch.as_tensor(chunk_vectors, dtype=torch.float16, device=device_obj)
    doc_index = {doc_id: index for index, doc_id in enumerate(doc_ids)}
    chunk_doc_index = torch.tensor([doc_index[doc_id] for doc_id in chunk_doc_ids], dtype=torch.long, device=device_obj)
    result: dict[str, list[dict[str, Any]]] = {}
    for start in range(0, len(queries), batch_size):
        qvec = model.encode(queries[start : start + batch_size], batch_size=batch_size, show_progress_bar=False, convert_to_tensor=True, normalize_embeddings=True)
        qvec = qvec.to(dtype=torch.float16)
        with torch.inference_mode():
            scores = qvec @ vectors.T
            doc_scores = torch.full((qvec.shape[0], len(doc_ids)), -torch.inf, dtype=scores.dtype, device=device_obj)
            expanded_index = chunk_doc_index.unsqueeze(0).expand(qvec.shape[0], -1)
            doc_scores.scatter_reduce_(1, expanded_index, scores, reduce="amax", include_self=True)
            values, indices = torch.topk(doc_scores, k=min(top_k, len(doc_ids)), dim=1)
        values_cpu = values.float().cpu().numpy()
        indices_cpu = indices.cpu().numpy()
        for offset, qid in enumerate(qids[start : start + qvec.shape[0]]):
            result[qid] = [{"doc_id": doc_ids[int(index)], "score": float(score), "rank": rank + 1} for rank, (index, score) in enumerate(zip(indices_cpu[offset], values_cpu[offset]))]
        del qvec, scores, doc_scores, expanded_index
    del vectors, chunk_doc_index
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def _recall(answer: list[str], refs: set[str]) -> float:
    return len(set(answer) & refs) / max(len(refs), 1)


def _curve(questions: dict[str, dict[str, Any]], stage1: dict[str, list[str]], dense: dict[str, list[dict[str, Any]]], cutoffs: list[int]) -> dict[str, Any]:
    baseline = float(np.mean([_recall(row["anchor"], set(row["references"])) for row in questions.values()]))
    absent35 = [qid for qid, row in questions.items() if not set(row["references"]) & set(stage1[qid][:35])]
    absent80 = [qid for qid, row in questions.items() if not set(row["references"]) & set(stage1[qid][:80])]
    absent120 = [qid for qid, row in questions.items() if not set(row["references"]) & set(stage1[qid][:120])]
    result = {}
    for cutoff in cutoffs:
        dense_sets = {qid: {item["doc_id"] for item in dense[qid][:cutoff]} for qid in questions}
        union_rows = []
        dense_present = 0
        dense_absent35 = 0
        dense_absent80 = 0
        dense_absent120 = 0
        unique_gold_not_c120 = 0
        union_oracle = []
        for qid, row in questions.items():
            refs = set(row["references"])
            existing120 = set(stage1[qid][:120])
            dense_set = dense_sets[qid]
            dense_present += bool(refs & dense_set)
            dense_absent35 += bool(qid in absent35 and refs & dense_set)
            dense_absent80 += bool(qid in absent80 and refs & dense_set)
            dense_absent120 += bool(qid in absent120 and refs & dense_set)
            unique_gold_not_c120 += len((refs & dense_set) - existing120)
            union = list(dict.fromkeys(stage1[qid][:80] + [item["doc_id"] for item in dense[qid][:cutoff]]))
            gold_after_top4 = [doc for doc in union[4:] if doc in refs]
            answer = list(row["anchor"][:4]) + ([gold_after_top4[0]] if gold_after_top4 else [row["anchor"][4]])
            union_oracle.append(_recall(answer, refs))
            if refs & dense_set:
                union_rows.append(qid)
        result[str(cutoff)] = {
            "cutoff": cutoff,
            "dense_topk_present_qids": dense_present,
            "dense_topk_present_rate": dense_present / len(questions),
            "absent_c35_qids": len(absent35),
            "absent_c35_recovered_by_dense": dense_absent35,
            "absent_c35_recovery_rate": dense_absent35 / max(len(absent35), 1),
            "absent_c80_qids": len(absent80),
            "absent_c80_recovered_by_dense": dense_absent80,
            "absent_c120_qids": len(absent120),
            "absent_c120_recovered_by_dense": dense_absent120,
            "unique_gold_not_in_existing_c120": unique_gold_not_c120,
            "union_existing_c80_dense_oracle_recall": float(np.mean(union_oracle)),
            "union_existing_c80_dense_oracle_gain_vs_baseline": float(np.mean(union_oracle) - baseline),
            "dense_recovery_qids": union_rows,
        }
    return {
        "baseline_recall": baseline,
        "existing_absence": {"absent_c35": absent35, "absent_c80": absent80, "absent_c120": absent120},
        "cutoffs": result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--stage1-audit", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-output-dir", type=Path, required=True)
    parser.add_argument("--max-chunks-per-doc", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--benchmark-batches", default="4,8,12")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    args.gpu_output_dir.mkdir(parents=True, exist_ok=True)
    questions = load_metadata(args.metadata)
    stage1 = _stage1_candidates(args.stage1_audit, questions)
    print("sampling manifest", flush=True)
    texts, chunk_doc_ids, doc_counts = _sample_manifest(args.manifest, args.max_chunks_per_doc)
    doc_ids = sorted(doc_counts, key=lambda value: (int(value) if value.isdigit() else 10**18, value))
    print(json.dumps({"sampled_chunks": len(texts), "docs": len(doc_ids), "max_chunks_per_doc": args.max_chunks_per_doc}, ensure_ascii=False), flush=True)
    device = args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(str(args.model.resolve()), device=device, local_files_only=True)
    model.max_seq_length = 256
    benchmark = [] if args.skip_benchmark else _benchmark(model, texts, device, [int(value) for value in args.benchmark_batches.split(",") if value.strip()])
    write_json(args.gpu_output_dir / "efficiency.json", {
        "kind": "legalir_v29_gpu_efficiency",
        "schema_version": 1,
        "holdout_opened": False,
        "public_labels_used": False,
        "model": str(args.model.resolve()),
        "device": device,
        "query_max_tokens": 128,
        "passage_max_tokens": 256,
        "benchmark": benchmark,
        "selected_batch_size": args.batch_size,
        "selected_under_5_3gb_target": any(row.get("batch_size") == args.batch_size and row.get("peak_memory_allocated_mb", 0.0) < 5300.0 for row in benchmark),
        "model_config_sha256": sha256_file(args.model / "config.json"),
    })
    started = time.perf_counter()
    chunk_vectors = _encode_chunks(model, texts, args.batch_size)
    write_json(out / "dense_embedding_cache.json", {
        "kind": "legalir_v29_a0_dense_embedding_cache",
        "schema_version": 1,
        "holdout_opened": False,
        "public_labels_used": False,
        "dtype": str(chunk_vectors.dtype),
        "shape": list(chunk_vectors.shape),
        "sampled_chunks": len(texts),
        "max_chunks_per_doc": args.max_chunks_per_doc,
    })
    np.save(out / "dense_chunk_embeddings.f16.npy", chunk_vectors)
    query_rows = [(qid, row["query"]) for qid, row in questions.items()]
    dense = _query_dense_top_docs(model, query_rows, chunk_vectors, chunk_doc_ids, doc_ids, batch_size=args.query_batch_size, device=device, top_k=1000)
    write_jsonl(out / "dense_candidate_pools_top1000.jsonl", ({"qid": qid, "candidates": dense[qid]} for qid in questions))
    curve = _curve(questions, stage1, dense, [100, 200, 500, 1000])
    curve.update({
        "kind": "legalir_v29_a0_dense_reach_curve",
        "schema_version": 1,
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "model": "BAAI/bge-m3",
        "model_path": str(args.model.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "metadata_sha256": sha256_file(args.metadata),
        "stage1_audit_sha256": sha256_file(args.stage1_audit),
        "dense_candidate_view": {"sampled_chunks": len(texts), "docs": len(doc_ids), "max_chunks_per_doc": args.max_chunks_per_doc, "aggregation": "max_chunk_cosine_per_document"},
        "elapsed_dense_seconds": time.perf_counter() - started,
        "gate": {
            "top500_or_1000_has_at_least_8_unique_gold_not_in_existing_c120": any(curve["cutoffs"][str(k)]["unique_gold_not_in_existing_c120"] >= 8 for k in [500, 1000]),
            "decision": "proceed_to_a0_2_multivector" if any(curve["cutoffs"][str(k)]["unique_gold_not_in_existing_c120"] >= 8 for k in [500, 1000]) else "stop_dense_lane_before_whole_corpus_multivector",
        },
    })
    write_json(out / "reach_curve.json", curve)
    absent_rows = []
    absent35 = set(curve["existing_absence"]["absent_c35"])
    for qid in sorted(absent35, key=lambda value: (int(value) if value.isdigit() else 10**18, value)):
        refs = set(questions[qid]["references"])
        dense_ranks = {item["doc_id"]: item["rank"] for item in dense[qid] if item["doc_id"] in refs}
        existing_ranks = {doc: rank + 1 for rank, doc in enumerate(stage1[qid][:120]) if doc in refs}
        absent_rows.append({"qid": qid, "references": sorted(refs), "dense_gold_ranks": dense_ranks, "existing_gold_ranks_c120": existing_ranks, "recovered_at": {str(k): sorted(doc for doc, rank in dense_ranks.items() if rank <= k) for k in [100, 200, 500, 1000]}})
    write_json(out / "absent47_recovery.json", {
        "kind": "legalir_v29_a0_dense_absent47_recovery",
        "schema_version": 1,
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "absent_c35_qids": len(absent_rows),
        "rows": absent_rows,
    })
    print(json.dumps({"reach_curve": str((out / 'reach_curve.json').resolve()), "gate": curve["gate"], "elapsed_seconds": curve["elapsed_dense_seconds"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
