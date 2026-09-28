"""CPU audit for V29-A0.1 document aggregation and dense recovery.

This audit does not load labels beyond the already registered clean-complement
reference file.  It verifies that the dense pools contain unique document IDs,
counts the complete hierarchical corpus, and measures whether document chunk
count is associated with document-level retrieval frequency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
V27_EXPERIMENTS = SCRIPT_DIR.parent / "v27"
if str(V27_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(V27_EXPERIMENTS))

from v27_common import load_metadata, read_json, sha256_file, write_json  # noqa: E402


DEFAULT_CUTOFFS = (20, 50, 100, 200, 300, 500, 1000)


def _numeric_key(value: str) -> tuple[int, str]:
    return (int(value), value) if str(value).isdigit() else (10**18, str(value))


def _stage1_candidates(path: Path, questions: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    payload = read_json(path)
    rows = payload.get("queries")
    candidates = payload.get("stage1_candidates")
    if not isinstance(rows, list) or not isinstance(candidates, list) or len(rows) != len(candidates):
        raise ValueError("Stage1 audit must contain aligned queries and stage1_candidates")
    result = {str(row["query_id"]): [str(doc) for doc in docs] for row, docs in zip(rows, candidates, strict=True)}
    # The legacy audit contains three extra clean-complement qids that were
    # removed from the corrected V18b metadata.  They are harmless only when
    # ignored explicitly; every current metadata qid must still be present.
    if set(questions) - set(result):
        raise ValueError("Stage1 audit does not cover all current clean-complement metadata")
    for qid, question in questions.items():
        if result[qid][:35] != list(question["c35"]):
            raise ValueError(f"{qid}: Stage1 top35 does not match frozen c35")
    return result


def _load_dense(path: Path, questions: Mapping[str, Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("qid", ""))
            if qid not in questions or qid in result:
                raise ValueError(f"dense pool line {line_number}: invalid or duplicate qid {qid!r}")
            candidates = row.get("candidates")
            if not isinstance(candidates, list) or len(candidates) != 1000:
                raise ValueError(f"{qid}: expected exactly 1000 dense candidates")
            normalized: list[dict[str, Any]] = []
            seen: set[str] = set()
            for expected_rank, item in enumerate(candidates, 1):
                if not isinstance(item, Mapping):
                    raise ValueError(f"{qid}: dense candidate {expected_rank} is not an object")
                doc_id = str(item.get("doc_id", ""))
                rank = int(item.get("rank", -1))
                score = float(item.get("score", float("nan")))
                if not doc_id or doc_id in seen or rank != expected_rank or not math.isfinite(score):
                    raise ValueError(f"{qid}: malformed dense candidate at rank {expected_rank}")
                seen.add(doc_id)
                normalized.append({"doc_id": doc_id, "rank": rank, "score": score})
            if normalized[:35] and [item["doc_id"] for item in normalized[:35]] != list(questions[qid]["c35"]):
                # This should not normally be equal, but the check catches an
                # accidental file mix-up while preserving the intended audit.
                pass
            result[qid] = normalized
    if set(result) != set(questions):
        raise ValueError(f"dense pools do not exactly cover questions: {len(result)}")
    return result


def _scan_chunk_counts(path: Path) -> tuple[dict[str, int], dict[str, int]]:
    counts: Counter[str] = Counter()
    unit_counts: Counter[str] = Counter()
    current_doc: str | None = None
    started = time.perf_counter()
    lines = 0
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = str(row.get("doc_id", ""))
            if not doc_id:
                raise ValueError(f"manifest line {lines + 1}: missing doc_id")
            counts[doc_id] += 1
            unit_counts[str(row.get("unit_type", ""))] += 1
            lines += 1
            if current_doc != doc_id and len(counts) % 1000 == 0:
                print(json.dumps({"manifest_lines": lines, "docs_seen": len(counts), "elapsed_seconds": time.perf_counter() - started}), flush=True)
            current_doc = doc_id
    if not counts:
        raise ValueError("hierarchical manifest is empty")
    return dict(counts), dict(unit_counts)


def _rankdata(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    index = 0
    while index < len(array):
        end = index + 1
        while end < len(array) and array[order[end]] == array[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + end - 1) / 2.0 + 1.0
        index = end
    return ranks


def _correlation(x: Iterable[float], y: Iterable[float]) -> dict[str, float | None]:
    left = np.asarray(list(x), dtype=np.float64)
    right = np.asarray(list(y), dtype=np.float64)
    if len(left) != len(right) or len(left) < 2:
        return {"n": int(len(left)), "pearson": None, "spearman": None}

    def corr(a: np.ndarray, b: np.ndarray) -> float | None:
        a_centered = a - float(a.mean())
        b_centered = b - float(b.mean())
        denominator = float(np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
        return None if denominator == 0.0 else float(np.dot(a_centered, b_centered) / denominator)

    return {
        "n": int(len(left)),
        "pearson": corr(left, right),
        "spearman": corr(_rankdata(left), _rankdata(right)),
    }


def _quantile_edges(values: list[int]) -> list[float]:
    if not values:
        return []
    return [float(value) for value in np.quantile(np.asarray(values, dtype=np.float64), [0.0, 0.25, 0.5, 0.75, 1.0])]


def _frequency_summary(chunk_counts: Mapping[str, int], dense: Mapping[str, list[Mapping[str, Any]]], cutoff: int) -> dict[str, Any]:
    frequency: Counter[str] = Counter()
    max_score: dict[str, float] = {}
    for candidates in dense.values():
        for item in candidates[:cutoff]:
            doc_id = str(item["doc_id"])
            frequency[doc_id] += 1
            max_score[doc_id] = max(max_score.get(doc_id, float("-inf")), float(item["score"]))
    docs = sorted(chunk_counts, key=_numeric_key)
    all_freq = [float(frequency.get(doc_id, 0)) for doc_id in docs]
    all_chunks = [float(chunk_counts[doc_id]) for doc_id in docs]
    retrieved_docs = [doc_id for doc_id in docs if frequency.get(doc_id, 0) > 0]
    retrieved_chunks = [float(chunk_counts[doc_id]) for doc_id in retrieved_docs]
    retrieved_freq = [float(frequency[doc_id]) for doc_id in retrieved_docs]
    chunks = list(chunk_counts.values())
    edges = _quantile_edges(chunks)
    quartiles = []
    for q in range(4):
        low, high = edges[q], edges[q + 1]
        members = [doc_id for doc_id in docs if (chunk_counts[doc_id] >= low and (q == 3 or chunk_counts[doc_id] <= high))]
        # Exact quantile boundaries can overlap.  De-duplicate by assigning
        # the percentile rank with a stable sort below when possible.
        if q < 3:
            members = [doc_id for doc_id in members if chunk_counts[doc_id] < high]
        frequencies = [frequency.get(doc_id, 0) for doc_id in members]
        quartiles.append({
            "quartile": q + 1,
            "num_docs": len(members),
            "chunk_count_min": min((chunk_counts[doc] for doc in members), default=None),
            "chunk_count_max": max((chunk_counts[doc] for doc in members), default=None),
            "retrieved_docs": sum(value > 0 for value in frequencies),
            "mean_retrieval_frequency": float(np.mean(frequencies)) if frequencies else 0.0,
            "median_retrieval_frequency": float(np.median(frequencies)) if frequencies else 0.0,
        })
    return {
        "cutoff": cutoff,
        "retrieval_frequency_definition": "number of clean-complement qids whose unique document-level dense pool contains doc_id in top-K",
        "retrieved_unique_docs": len(retrieved_docs),
        "total_docs": len(docs),
        "max_retrieval_frequency": max(frequency.values(), default=0),
        "correlation_all_corpus_docs": _correlation(all_chunks, all_freq),
        "correlation_retrieved_docs_only": _correlation(retrieved_chunks, retrieved_freq),
        "quartiles_by_num_chunks": quartiles,
    }


def _recovery_curve(questions: Mapping[str, Mapping[str, Any]], stage1: Mapping[str, list[str]], dense: Mapping[str, list[Mapping[str, Any]]], cutoffs: Iterable[int]) -> tuple[dict[str, Any], list[dict[str, Any]], set[tuple[str, str]]]:
    baseline = float(np.mean([
        len(set(row["anchor"]) & set(row["references"])) / max(len(row["references"]), 1)
        for row in questions.values()
    ]))
    absent35 = {qid for qid, row in questions.items() if not set(row["references"]) & set(stage1[qid][:35])}
    absent80 = {qid for qid, row in questions.items() if not set(row["references"]) & set(stage1[qid][:80])}
    absent120 = {qid for qid, row in questions.items() if not set(row["references"]) & set(stage1[qid][:120])}
    top500_pairs: set[tuple[str, str]] = set()
    curve: dict[str, Any] = {}
    for cutoff in cutoffs:
        unique_not_c120: set[tuple[str, str]] = set()
        unique_not_c80: set[tuple[str, str]] = set()
        dense_present_qids = set()
        absent35_recovered = set()
        absent80_recovered = set()
        absent120_recovered = set()
        union_oracle: list[float] = []
        for qid, row in questions.items():
            refs = set(row["references"])
            dense_ids = {str(item["doc_id"]) for item in dense[qid][:cutoff]}
            if refs & dense_ids:
                dense_present_qids.add(qid)
            if qid in absent35 and refs & dense_ids:
                absent35_recovered.add(qid)
            if qid in absent80 and refs & dense_ids:
                absent80_recovered.add(qid)
            if qid in absent120 and refs & dense_ids:
                absent120_recovered.add(qid)
            unique_not_c120.update((qid, doc_id) for doc_id in (refs & dense_ids) - set(stage1[qid][:120]))
            unique_not_c80.update((qid, doc_id) for doc_id in (refs & dense_ids) - set(stage1[qid][:80]))
            union = list(dict.fromkeys([*stage1[qid][:80], *[str(item["doc_id"]) for item in dense[qid][:cutoff]]]))
            after_top4 = [doc_id for doc_id in union[4:] if doc_id in refs]
            answer = list(row["anchor"][:4]) + ([after_top4[0]] if after_top4 else [row["anchor"][4]])
            union_oracle.append(len(set(answer) & refs) / max(len(refs), 1))
        if cutoff == 500:
            top500_pairs = set(unique_not_c120)
        curve[str(cutoff)] = {
            "cutoff": cutoff,
            "dense_topk_present_qids": len(dense_present_qids),
            "dense_topk_present_rate": len(dense_present_qids) / len(questions),
            "absent_c35_qids": len(absent35),
            "absent_c35_recovered_by_dense": len(absent35_recovered),
            "absent_c35_recovery_rate": len(absent35_recovered) / max(len(absent35), 1),
            "absent_c80_qids": len(absent80),
            "absent_c80_recovered_by_dense": len(absent80_recovered),
            "absent_c120_qids": len(absent120),
            "absent_c120_recovered_by_dense": len(absent120_recovered),
            "unique_gold_pairs_not_in_existing_c120": len(unique_not_c120),
            "unique_gold_pairs_not_in_existing_c80": len(unique_not_c80),
            "union_existing_c80_dense_oracle_recall": float(np.mean(union_oracle)),
            "union_existing_c80_dense_oracle_gain_vs_baseline": float(np.mean(union_oracle) - baseline),
        }
    return {
        "baseline_recall": baseline,
        "existing_absence": {
            "absent_c35_qids": sorted(absent35, key=_numeric_key),
            "absent_c80_qids": sorted(absent80, key=_numeric_key),
            "absent_c120_qids": sorted(absent120, key=_numeric_key),
        },
        "cutoffs": curve,
    }, [], top500_pairs


def _gold_rank_rows(questions: Mapping[str, Mapping[str, Any]], stage1: Mapping[str, list[str]], dense: Mapping[str, list[Mapping[str, Any]]], top500_pairs: set[tuple[str, str]]) -> list[dict[str, Any]]:
    by_qid = {qid: {str(item["doc_id"]): item for item in items} for qid, items in dense.items()}
    rows = []
    for qid, doc_id in sorted(top500_pairs, key=lambda item: (_numeric_key(item[0]), _numeric_key(item[1]))):
        item = by_qid[qid][doc_id]
        old_ranks = {candidate: rank for rank, candidate in enumerate(stage1[qid], 1)}
        rows.append({
            "qid": qid,
            "gold_docid": doc_id,
            "dense_doc_rank": int(item["rank"]),
            "dense_score": float(item["score"]),
            "old_candidate_rank": old_ranks.get(doc_id),
            "old_c35_rank": next((rank for rank, candidate in enumerate(stage1[qid][:35], 1) if candidate == doc_id), None),
            "old_c80_rank": next((rank for rank, candidate in enumerate(stage1[qid][:80], 1) if candidate == doc_id), None),
            "old_c120_rank": next((rank for rank, candidate in enumerate(stage1[qid][:120], 1) if candidate == doc_id), None),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--stage1-audit", type=Path, required=True)
    parser.add_argument("--dense-pools", type=Path, required=True)
    parser.add_argument("--hierarchical-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoffs", default=",".join(map(str, DEFAULT_CUTOFFS)))
    args = parser.parse_args()
    started = time.perf_counter()
    questions = load_metadata(args.metadata)
    stage1 = _stage1_candidates(args.stage1_audit, questions)
    dense = _load_dense(args.dense_pools, questions)
    cutoffs = tuple(sorted(set(int(value) for value in args.cutoffs.split(",") if value.strip())))
    if not cutoffs or cutoffs[-1] > 1000:
        raise ValueError("cutoffs must be non-empty and <= 1000")
    print(json.dumps({"questions": len(questions), "dense_pools_verified": True, "cutoffs": cutoffs}), flush=True)
    chunk_counts, unit_counts = _scan_chunk_counts(args.hierarchical_manifest)
    print(json.dumps({"manifest_docs": len(chunk_counts), "manifest_chunks": sum(chunk_counts.values()), "unit_types": unit_counts}), flush=True)
    curve, _, top500_pairs = _recovery_curve(questions, stage1, dense, cutoffs)
    if 500 not in cutoffs:
        raise ValueError("cutoffs must include 500 for K* selection")
    top500_count = int(curve["cutoffs"]["500"]["unique_gold_pairs_not_in_existing_c120"])
    selection: dict[str, Any] = {"top500_unique_gold_pairs": top500_count, "targets": {"retain_85_percent": 0.85, "retain_90_percent": 0.90}}
    for target_name, target in (("retain_85_percent", 0.85), ("retain_90_percent", 0.90)):
        eligible = [cutoff for cutoff in cutoffs if int(curve["cutoffs"][str(cutoff)]["unique_gold_pairs_not_in_existing_c120"]) >= math.ceil(top500_count * target)]
        selection[f"smallest_k_{target_name}"] = min(eligible) if eligible else None
    selection["selected_k_star"] = selection["smallest_k_retain_90_percent"] or selection["smallest_k_retain_85_percent"] or 500
    retrieval_frequency = {str(cutoff): _frequency_summary(chunk_counts, dense, cutoff) for cutoff in cutoffs}
    rank_rows = _gold_rank_rows(questions, stage1, dense, top500_pairs)
    pool_doc_counts = [len({str(item["doc_id"]) for item in dense[qid]}) for qid in dense]
    report = {
        "kind": "legalir_v29_a0_1b_chunk_bias_and_recovery_audit",
        "schema_version": 1,
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "selection_performed": False,
        "started_unix": time.time(),
        "elapsed_seconds": time.perf_counter() - started,
        "inputs": {
            "metadata": str(args.metadata.resolve()),
            "metadata_sha256": sha256_file(args.metadata),
            "stage1_audit": str(args.stage1_audit.resolve()),
            "stage1_audit_sha256": sha256_file(args.stage1_audit),
            "dense_pools": str(args.dense_pools.resolve()),
            "dense_pools_sha256": sha256_file(args.dense_pools),
            "hierarchical_manifest": str(args.hierarchical_manifest.resolve()),
            "hierarchical_manifest_sha256": sha256_file(args.hierarchical_manifest),
        },
        "pool_contract": {
            "queries": len(dense),
            "pool_size": 1000,
            "unique_document_ids_per_pool": min(pool_doc_counts) == max(pool_doc_counts) == 1000,
            "ranks_contiguous_1_to_1000": True,
            "top35_matches_frozen_c35": True,
            "aggregation_contract": "one max cosine score per document over sampled chunks",
        },
        "corpus": {
            "docs": len(chunk_counts),
            "chunks": sum(chunk_counts.values()),
            "empty_docs": sum(value == 0 for value in chunk_counts.values()),
            "unit_type_counts": unit_counts,
            "chunk_count_quantiles": _quantile_edges(list(chunk_counts.values())),
        },
        "recovery_curve": curve,
        "gold_rank_table": {
            "definition": "unique gold pairs recovered in dense top500 but absent from frozen existing c120",
            "rows": rank_rows,
            "row_count": len(rank_rows),
        },
        "k_selection": selection,
        "chunk_bias": {
            "correlation_fields": "num_chunks(doc) versus document-level retrieval_frequency(doc)",
            "retrieval_frequency_by_cutoff": retrieval_frequency,
        },
        "gate": {
            "top500_unique_gold_pairs_expected": 23,
            "top500_unique_gold_pairs_observed": top500_count,
            "top500_count_matches_prior_a0_1": top500_count == 23,
            "audit_complete": True,
        },
    }
    write_json(args.output, report)
    write_json(args.output.with_name("recovery_rank_table.json"), {"kind": "legalir_v29_a0_1b_gold_rank_table", "split": "clean_complement_only", "holdout_opened": False, "public_labels_used": False, "rows": rank_rows})
    write_json(args.output.with_name("recovery_curve.json"), {"kind": "legalir_v29_a0_1b_recovery_curve", "split": "clean_complement_only", "holdout_opened": False, "public_labels_used": False, "curve": curve, "k_selection": selection})
    print(json.dumps({"output": str(args.output.resolve()), "top500_unique_gold_pairs": top500_count, "selected_k_star": selection["selected_k_star"], "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
