"""V29-A1D-1: Stage-2 diagnostic for dense challengers on absent-c35 qids.

The frozen sidecar contains exact c35 scores only.  The historical c36--c80
tail is therefore scored once and kept in a separate cache so that the union
rank is honest.  New dense documents are scored with the same local
CrossEncoder, three-passage hydration, max-passage aggregation, 768-token
limit, and fp16 contract used by the frozen Stage-2 audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
V27_EXPERIMENTS = SCRIPT_DIR.parent / "v27"
ARCHIVE_SCRIPTS = REPO_ROOT / "archive" / "2026-08-31_retired_experiments" / "Task1-scripts"
for path in (REPO_ROOT / "scripts", V27_EXPERIMENTS, ARCHIVE_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from batch_stage2_rerank import hydrate, load_passage_store, open_readonly  # noqa: E402
from v27_common import load_metadata, load_sidecar, read_json, sha256_file, write_json, write_jsonl  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402


def _numeric_key(value: str) -> tuple[int, str]:
    return (int(value), value) if str(value).isdigit() else (10**18, str(value))


def _stage1_candidates(path: Path, questions: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    payload = read_json(path)
    rows = payload.get("queries")
    candidates = payload.get("stage1_candidates")
    result = {str(row["query_id"]): [str(doc) for doc in docs] for row, docs in zip(rows, candidates, strict=True)}
    if set(questions) - set(result):
        raise ValueError("Stage1 audit misses current metadata qids")
    for qid, question in questions.items():
        if result[qid][:35] != list(question["c35"]):
            raise ValueError(f"{qid}: Stage1 top35 does not match frozen c35")
    return result


def _dense_pools(path: Path, qids: set[str]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("qid", ""))
            if qid not in qids:
                continue
            if qid in result:
                raise ValueError(f"duplicate dense qid {qid}")
            candidates = row.get("candidates")
            if not isinstance(candidates, list) or len(candidates) < 500:
                raise ValueError(f"{qid}: dense pool has fewer than 500 candidates")
            normalized = []
            seen: set[str] = set()
            for expected_rank, item in enumerate(candidates, 1):
                doc_id = str(item.get("doc_id", ""))
                rank = int(item.get("rank", -1))
                score = float(item.get("score", float("nan")))
                if not doc_id or doc_id in seen or rank != expected_rank or not math.isfinite(score):
                    raise ValueError(f"{qid}: malformed dense row at line {line_number}, rank {expected_rank}")
                seen.add(doc_id)
                normalized.append({"doc_id": doc_id, "rank": rank, "score": score})
            result[qid] = normalized
    if result.keys() != qids:
        raise ValueError(f"dense pool coverage mismatch: got={len(result)} expected={len(qids)}")
    return result


class DirectStage2:
    """The frozen scalar CE with an explicit, dynamic-padding inference loop."""

    def __init__(self, model_dir: Path, device: str, max_length: int, fp16: bool):
        self.device = torch.device(device)
        self.max_length = int(max_length)
        self.fp16 = bool(fp16 and self.device.type == "cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir.resolve()), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir.resolve()), local_files_only=True)
        if int(getattr(self.model.config, "num_labels", 1) or 1) != 1:
            raise ValueError("Stage2 model must emit one scalar logit")
        self.model.to(self.device)
        if self.fp16:
            self.model.half()
        self.model.eval()

    def score(self, pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
        values: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start : start + batch_size]
                encoded = self.tokenizer(
                    [query for query, _ in batch],
                    [document for _, document in batch],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device, non_blocking=True) for key, value in encoded.items()}
                logits = self.model(**encoded, return_dict=True).logits.reshape(-1).float()
                values.extend(float(value) for value in torch.sigmoid(logits).cpu().tolist())
        return values


def _score_pairs(reranker: DirectStage2, pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
    if not pairs:
        return []
    return reranker.score(pairs, batch_size)


def _benchmark(reranker: DirectStage2, pairs: list[tuple[str, str]], batch_sizes: list[int], device: str) -> list[dict[str, Any]]:
    # A short fixed sample keeps the diagnostic usable when another workload
    # occupies half of the GPU.  The production score pass is still run over
    # every requested query-passage pair below.
    sample = pairs[:16]
    rows = []
    for batch_size in batch_sizes:
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(torch.device(device))
            torch.cuda.reset_peak_memory_stats(torch.device(device))
        started = time.perf_counter()
        _score_pairs(reranker, sample, batch_size)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(torch.device(device))
        elapsed = time.perf_counter() - started
        row = {
            "pairs": len(sample),
            "batch_size": batch_size,
            "elapsed_seconds": elapsed,
            "pairs_per_second": len(sample) / max(elapsed, 1e-9),
            "device": device,
        }
        if device.startswith("cuda") and torch.cuda.is_available():
            row["peak_memory_allocated_mb"] = torch.cuda.max_memory_allocated(torch.device(device)) / 1024**2
            row["peak_memory_reserved_mb"] = torch.cuda.max_memory_reserved(torch.device(device)) / 1024**2
        rows.append(row)
    return rows


def _rank_union(doc_ids: list[str], scores: Mapping[str, float]) -> tuple[list[str], dict[str, int]]:
    ranked = sorted(
        dict.fromkeys(doc_ids),
        key=lambda doc_id: (-float(scores[doc_id]), doc_ids.index(doc_id), _numeric_key(doc_id)),
    )
    return ranked, {doc_id: rank for rank, doc_id in enumerate(ranked, 1)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--stage1-audit", type=Path, required=True)
    parser.add_argument("--dense-pools", type=Path, required=True)
    parser.add_argument("--a0-audit", type=Path, required=True)
    parser.add_argument("--stage2-sidecar", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--benchmark-batches", default="4,8,12")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--passages-per-doc", type=int, default=3)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gate-only", action="store_true", help="Score recovered gold challengers only; do not claim a complete c80 union")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_length <= 0 or args.passages_per_doc != 3:
        raise ValueError("batch-size/max-length must be positive and passages-per-doc must remain 3")
    started = time.perf_counter()
    questions = load_metadata(args.metadata)
    stage1 = _stage1_candidates(args.stage1_audit, questions)
    sidecar = load_sidecar(args.stage2_sidecar, questions)
    audit = read_json(args.a0_audit)
    k_star = int(audit.get("k_selection", {}).get("selected_k_star", 0))
    if k_star < 1 or k_star > 1000:
        raise ValueError("A0.1b audit has no valid selected K*")
    absent35 = {
        qid for qid, row in questions.items()
        if not set(row["references"]) & set(stage1[qid][:35])
    }
    dense = _dense_pools(args.dense_pools, absent35)
    existing_c80 = {qid: list(dict.fromkeys(stage1[qid][:80])) for qid in absent35}
    dense_new: dict[str, list[dict[str, Any]]] = {}
    old_tail: dict[str, list[str]] = {}
    all_scoring_docs: set[str] = set()
    for qid in sorted(absent35, key=_numeric_key):
        existing = set(existing_c80[qid])
        new = [item for item in dense[qid][:k_star] if str(item["doc_id"]) not in existing]
        if args.gate_only:
            references = set(questions[qid]["references"])
            new = [item for item in new if str(item["doc_id"]) in references]
        dense_new[qid] = new
        old_tail[qid] = [] if args.gate_only else [doc_id for doc_id in existing_c80[qid][35:] if (qid, doc_id) not in sidecar]
        all_scoring_docs.update(old_tail[qid])
        all_scoring_docs.update(str(item["doc_id"]) for item in new)
    print(json.dumps({"k_star": k_star, "absent_c35_qids": len(absent35), "new_dense_docs": sum(len(v) for v in dense_new.values()), "legacy_c36_c80_docs": sum(len(v) for v in old_tail.values())}), flush=True)

    device = args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    reranker = DirectStage2(
        args.model, device=device, max_length=args.max_length,
        fp16=bool(args.fp16 and device.startswith("cuda")),
    )
    connection = open_readonly(args.index)
    passage_store = load_passage_store(connection, sorted(all_scoring_docs, key=_numeric_key))
    connection.close()

    pair_rows: list[dict[str, Any]] = []
    pair_index: dict[tuple[str, str], list[int]] = {}
    for qid in sorted(absent35, key=_numeric_key):
        docs = list(dict.fromkeys([*old_tail[qid], *[str(item["doc_id"]) for item in dense_new[qid]]]))
        hydrated = {str(item["doc_id"]): item for item in hydrate(questions[qid]["query"], docs, args.passages_per_doc, passage_store)}
        for doc_id in docs:
            passages = [str(value) for value in hydrated.get(doc_id, {}).get("passages", []) if str(value).strip()]
            if not passages:
                continue
            indices = []
            for passage_index, passage in enumerate(passages):
                indices.append(len(pair_rows))
                pair_rows.append({"qid": qid, "doc_id": doc_id, "passage_index": passage_index, "query": questions[qid]["query"], "passage": passage})
            pair_index[(qid, doc_id)] = indices
    pairs = [(str(row["query"]), str(row["passage"])) for row in pair_rows]
    print(json.dumps({"scorable_query_doc_pairs": len(pair_index), "query_passage_pairs": len(pairs), "empty_query_doc_pairs": len(all_scoring_docs) - len({doc_id for _, doc_id in pair_index}), "device": device}), flush=True)
    benchmark = [] if args.gate_only else _benchmark(reranker, pairs, [int(value) for value in args.benchmark_batches.split(",") if value.strip()], device)
    flat_scores = _score_pairs(reranker, pairs, args.batch_size)
    score_by_doc: dict[tuple[str, str], float] = {}
    passage_scores: dict[tuple[str, str], list[float]] = {}
    for key, indices in pair_index.items():
        values = [float(flat_scores[index]) for index in indices]
        passage_scores[key] = values
        score_by_doc[key] = max(values)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    new_rows = []
    old_tail_rows = []
    dense_rank_by_qid = {qid: {str(item["doc_id"]): int(item["rank"]) for item in dense[qid]} for qid in absent35}
    dense_score_by_qid = {qid: {str(item["doc_id"]): float(item["score"]) for item in dense[qid]} for qid in absent35}
    for qid in sorted(absent35, key=_numeric_key):
        for item in dense_new[qid]:
            doc_id = str(item["doc_id"])
            key = (qid, doc_id)
            new_rows.append({
                "qid": qid,
                "doc_id": doc_id,
                "dense_rank": int(item["rank"]),
                "dense_score": float(item["score"]),
                "stage2_score": score_by_doc.get(key),
                "passage_scores": passage_scores.get(key, []),
                "score_source": "v29_a1d1_new_dense_document",
            })
        for doc_id in old_tail[qid]:
            key = (qid, doc_id)
            old_tail_rows.append({
                "qid": qid,
                "doc_id": doc_id,
                "old_candidate_rank": stage1[qid].index(doc_id) + 1,
                "stage2_score": score_by_doc.get(key),
                "passage_scores": passage_scores.get(key, []),
                "score_source": "v29_a1d1_missing_legacy_c36_c80_tail",
            })
    new_hash = write_jsonl(out / "dense_new_stage2_scores.jsonl", new_rows)
    old_hash = write_jsonl(out / "legacy_c36_c80_stage2_scores.jsonl", old_tail_rows)

    detail_rows = []
    for qid in sorted(absent35, key=_numeric_key):
        c80 = existing_c80[qid]
        if args.gate_only:
            # c36--c80 has no frozen score cache.  The gate only needs the
            # exact frozen slot5 score, so rank the c35 boundary plus recovered
            # gold challengers and label this rank scope explicitly.
            old_union = stage1[qid][:35]
            rank_scope = "frozen_c35_plus_recovered_gold_challengers"
        else:
            old_union = c80
            rank_scope = "existing_c80_plus_dense_top_k_star"
        union_docs = list(dict.fromkeys([*old_union, *[str(item["doc_id"]) for item in dense_new[qid]]]))
        scores: dict[str, float] = {}
        missing: list[str] = []
        for doc_id in union_docs:
            value = sidecar.get((qid, doc_id))
            if value is None:
                value = score_by_doc.get((qid, doc_id))
            if value is None or not math.isfinite(float(value)):
                missing.append(doc_id)
            else:
                scores[doc_id] = float(value)
        if missing:
            raise ValueError(f"{qid}: union has missing Stage2 scores for {len(missing)} docs")
        ranked, rank_map = _rank_union(union_docs, scores)
        old_slot5 = str(questions[qid]["anchor"][4])
        if old_slot5 not in scores:
            raise ValueError(f"{qid}: frozen old slot5 is absent from c80 union")
        old_slot5_score = float(scores[old_slot5])
        refs = set(questions[qid]["references"])
        for gold_docid in sorted(refs, key=_numeric_key):
            dense_rank = dense_rank_by_qid[qid].get(gold_docid)
            gold_is_dense_recovered = dense_rank is not None and dense_rank <= k_star
            gold_is_new = gold_docid not in set(c80)
            gold_score = scores.get(gold_docid)
            gold_rank = rank_map.get(gold_docid)
            beats = bool(gold_score is not None and gold_score > old_slot5_score)
            detail_rows.append({
                "qid": qid,
                "gold_docid": gold_docid,
                "gold_absent_c35": True,
                "gold_outside_existing_c80": gold_is_new,
                "gold_outside_existing_c120": gold_docid not in set(stage1[qid][:120]),
                "gold_dense_rank": dense_rank,
                "gold_dense_score": dense_score_by_qid[qid].get(gold_docid),
                "gold_recovered_dense": bool(gold_is_dense_recovered),
                "gold_recovered_new_dense": bool(gold_is_dense_recovered and gold_is_new),
                "old_slot5_doc": old_slot5,
                "old_slot5_stage2": old_slot5_score,
                "gold_stage2": gold_score,
                "gold_stage2_rank_union": gold_rank,
                "gold_stage2_rank_scope": rank_scope,
                "gold_would_enter_top5": bool(gold_rank is not None and gold_rank <= 5),
                "gold_beats_old_slot5": beats,
                "gold_would_win_slot5": beats,
                "fails_stage2": gold_score is None,
                "union_size": len(union_docs),
            })

    new_gold_rows = [row for row in detail_rows if row["gold_recovered_new_dense"]]
    c120_gold_rows = [row for row in new_gold_rows if row["gold_outside_existing_c120"]]
    def count(field: str, rows: list[dict[str, Any]]) -> int:
        return sum(bool(row[field]) for row in rows)

    wins = count("gold_would_win_slot5", c120_gold_rows)
    if wins >= 10:
        strength = "strong"
    elif wins >= 7:
        strength = "moderate"
    elif wins <= 4:
        strength = "weak"
    else:
        strength = "borderline"
    report = {
        "kind": "legalir_v29_a1d1_stage2_dense_challenger_diagnostic",
        "schema_version": 1,
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "selection_performed": False,
        "inputs": {
            "metadata": str(args.metadata.resolve()),
            "metadata_sha256": sha256_file(args.metadata),
            "stage1_audit": str(args.stage1_audit.resolve()),
            "stage1_audit_sha256": sha256_file(args.stage1_audit),
            "dense_pools": str(args.dense_pools.resolve()),
            "dense_pools_sha256": sha256_file(args.dense_pools),
            "a0_audit": str(args.a0_audit.resolve()),
            "a0_audit_sha256": sha256_file(args.a0_audit),
            "stage2_sidecar": str(args.stage2_sidecar.resolve()),
            "stage2_sidecar_sha256": sha256_file(args.stage2_sidecar),
            "index": str(args.index.resolve()),
            "index_sha256": sha256_file(args.index),
            "model": str(args.model.resolve()),
            "model_config_sha256": sha256_file(args.model / "config.json"),
        },
        "contract": {
            "k_star": k_star,
            "focus": "47 queries absent from frozen c35",
            "candidate_union": "existing c80 union dense top-K*",
            "new_dense_scored": "dense top-K* minus existing c80",
            "frozen_c35_scores_reused": True,
            "frozen_c80_scores_available": False,
            "legacy_c36_c80_tail_scored_to_complete_union": not args.gate_only,
            "gate_only": bool(args.gate_only),
            "full_c80_union_evaluation_completed": not args.gate_only,
            "reranker_model": str(args.model.resolve()),
            "max_length": args.max_length,
            "passages_per_doc": args.passages_per_doc,
            "passage_aggregation": "max",
            "fp16": bool(args.fp16 and device.startswith("cuda")),
        },
        "counts": {
            "focus_qids": len(absent35),
            "new_dense_documents_scored": len(new_rows),
            "legacy_c36_c80_documents_scored": len(old_tail_rows),
            "query_passage_pairs_scored": len(pairs),
            "empty_query_doc_pairs": len(all_scoring_docs) - len({doc_id for _, doc_id in pair_index}),
            "dense_recovered_gold_pairs_new_vs_c80": len(new_gold_rows),
            "dense_recovered_gold_pairs_outside_c120": len(c120_gold_rows),
        },
        "aggregate": {
            "recovered_dense_gold_new_vs_c80": len(new_gold_rows),
            "recovered_dense_gold_stage2_rank_le_5": count("gold_would_enter_top5", new_gold_rows),
            "recovered_dense_gold_stage2_rank_le_10": sum(row["gold_stage2_rank_union"] is not None and row["gold_stage2_rank_union"] <= 10 for row in new_gold_rows),
            "recovered_dense_gold_beats_old_slot5": count("gold_beats_old_slot5", new_gold_rows),
            "recovered_dense_gold_fails_stage2": count("fails_stage2", new_gold_rows),
            "outside_c120_recovered_dense_gold": len(c120_gold_rows),
            "outside_c120_stage2_rank_le_5": count("gold_would_enter_top5", c120_gold_rows),
            "outside_c120_stage2_rank_le_10": sum(row["gold_stage2_rank_union"] is not None and row["gold_stage2_rank_union"] <= 10 for row in c120_gold_rows),
            "outside_c120_gold_beats_old_slot5": wins,
            "outside_c120_gold_fails_stage2": count("fails_stage2", c120_gold_rows),
            "conditional_p_beats_slot5_given_new_dense_recovered": wins / max(len(c120_gold_rows), 1),
        },
        "gate": {
            "denominator_definition": "unique dense-recovered gold pairs outside existing c120 at K*",
            "denominator": len(c120_gold_rows),
            "strong_threshold": 10,
            "moderate_threshold": 7,
            "wins": wins,
            "strength": strength,
            "pass_for_full_oof": wins >= 7,
            "decision": "proceed_to_a1d2_full_oof" if wins >= 7 else "run_parameter_free_rrf_diagnostic_before_v30",
            "rank_scope": "frozen_c35_plus_recovered_gold_challengers" if args.gate_only else "existing_c80_plus_dense_top_k_star",
        },
        "stage2_efficiency": {
            "device": device,
            "benchmark": benchmark,
            "selected_batch_size": args.batch_size,
            "selected_pairs_per_second": next((row["pairs_per_second"] for row in benchmark if row["batch_size"] == args.batch_size), None),
        },
        "cache": {
            "dense_new_stage2_scores_jsonl": str((out / "dense_new_stage2_scores.jsonl").resolve()),
            "dense_new_stage2_scores_sha256": new_hash,
            "legacy_c36_c80_stage2_scores_jsonl": str((out / "legacy_c36_c80_stage2_scores.jsonl").resolve()),
            "legacy_c36_c80_stage2_scores_sha256": old_hash,
            "format": "JSONL equivalent; pyarrow/parquet not required",
        },
        "rows": detail_rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(out / "a1d1_report.json", report)
    write_json(out / "a1d1_gold_diagnostics.json", {"kind": "legalir_v29_a1d1_gold_diagnostics", "split": "clean_complement_only", "holdout_opened": False, "public_labels_used": False, "rows": detail_rows})
    print(json.dumps({"report": str((out / 'a1d1_report.json').resolve()), "k_star": k_star, "new_dense_gold": len(new_gold_rows), "outside_c120_gold": len(c120_gold_rows), "wins": wins, "strength": strength, "pass_for_full_oof": wins >= 7, "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
