"""CPU fielded-BM25 candidate expansion for V28-C1."""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from v27_common import load_metadata, read_json, sha256_file, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from sqlite_retriever import SQLiteFTSRetriever  # noqa: E402


def recall(answer: list[str], refs: set[str]) -> float:
    return len(set(answer) & refs) / max(len(refs), 1)


def fielded_candidates(retriever: SQLiteFTSRetriever, query: str, limit: int, *, include_phrase: bool = False) -> tuple[list[dict[str, Any]], dict[str, int]]:
    _, tokens = retriever._match_expression(query)
    focused = retriever._chunk_candidates(tokens) if tokens else []
    broad = retriever._or_bm25_candidates(tokens) if tokens else []
    citation = retriever._citation_candidates(query)
    branches = [(3.0, "focused_bm25", focused), (2.0, "broad_bm25", broad), (2.0, "legal_citation", citation)]
    if include_phrase:
        branches.append((1.5, "phrase_bm25", retriever._chunk_phrase_candidates(query)))
    scores: dict[str, float] = collections.defaultdict(float)
    best: dict[str, dict[str, Any]] = {}
    branch_counts = {}
    for weight, name, branch in branches:
        branch_counts[name] = len(branch)
        for rank, item in enumerate(branch, 1):
            doc = str(item["doc_id"])
            scores[doc] += float(weight) / (60.0 + rank)
            current = best.get(doc)
            if current is None or float(item.get("raw_fts_score", item.get("score", 0.0))) > float(current.get("raw_fts_score", current.get("score", 0.0))):
                best[doc] = dict(item)
    ranked = []
    for doc, score in scores.items():
        item = dict(best[doc])
        item["doc_id"] = doc
        item["fielded_score"] = float(score)
        ranked.append(item)
    ranked.sort(key=lambda item: (-float(item["fielded_score"]), -float(item.get("raw_fts_score", 0.0)), str(item["doc_id"])))
    return ranked[:limit], branch_counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--stage1-audit", type=Path, required=True)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutoffs", default="35,50,80,120")
    parser.add_argument("--retrieval-limit", type=int, default=120)
    parser.add_argument("--focus-absent-c35", action="store_true", help="Run only qids whose frozen c35 has no raw qrel gold")
    parser.add_argument("--include-phrase", action="store_true", help="Include the slower phrase FTS branch")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    questions = load_metadata(args.metadata)
    stage1 = read_json(args.stage1_audit)
    stage1_candidates = {str(query["query_id"]): [str(doc) for doc in candidates] for query, candidates in zip(stage1["queries"], stage1["stage1_candidates"], strict=True)}
    if set(questions) - set(stage1_candidates):
        raise ValueError("Stage-1 audit is missing current clean-complement qids")
    active_qids = [qid for qid, question in questions.items() if not (set(question["references"]) & set(question["c35"]))] if args.focus_absent_c35 else list(questions)
    cutoffs = sorted({int(value) for value in args.cutoffs.split(",")})
    retriever = SQLiteFTSRetriever(args.sqlite, use_reranker=False, top_k_initial=args.retrieval_limit, enable_dense=False, title_weight=5.0, passage_weight=1.0)
    rows = []
    branch_totals = collections.Counter()
    started = time.perf_counter()
    try:
        for index, qid in enumerate(active_qids, 1):
            question = questions[qid]
            fielded, branch_counts = fielded_candidates(retriever, question["query"], args.retrieval_limit, include_phrase=args.include_phrase)
            for key, value in branch_counts.items():
                branch_totals[key] += value
            base = list(question["c35"])
            # The frozen c35 ordering remains the first segment.  The recovery
            # tail is a label-free union of historical Stage-1 reach and the
            # fresh title/body/phrase/citation BM25 branches.
            union = list(dict.fromkeys(base + stage1_candidates[qid][35:] + [str(item["doc_id"]) for item in fielded]))
            rows.append({
                "qid": qid,
                "query": question["query"],
                "c35": base,
                "candidate_union": union[: args.retrieval_limit],
                "fielded_ranked": [{"doc_id": str(item["doc_id"]), "fielded_score": float(item["fielded_score"]), "method": str(item.get("method", ""))} for item in fielded],
                "stage1_source_count": len(stage1_candidates[qid]),
                "branch_counts": branch_counts,
            })
            if index % 10 == 0:
                print(f"C1 fielded BM25: {index}/{len(active_qids)}", flush=True)
    finally:
        retriever.close()
    per_cutoff = {}
    active_questions = {qid: questions[qid] for qid in active_qids}
    baseline = float(np.mean([recall(row["anchor"], row["references"]) for row in active_questions.values()])) if active_questions else 0.0
    for cutoff in cutoffs:
        pool_by_qid = {row["qid"]: row["candidate_union"][:cutoff] for row in rows}
        exist_qids = sum(bool(set(active_questions[qid]["references"]) & set(pool_by_qid[qid])) for qid in active_qids)
        absent_c35 = [qid for qid, question in active_questions.items() if not (set(question["references"]) & set(question["c35"]))]
        recovered = [qid for qid in absent_c35 if set(questions[qid]["references"]) & set(pool_by_qid[qid])]
        slot5_oracle = []
        for qid, question in active_questions.items():
            refs = set(question["references"])
            slot5 = next((doc for doc in pool_by_qid[qid][4:] if doc in refs), question["anchor"][4])
            slot5_oracle.append(recall(list(question["anchor"][:4]) + [slot5], refs))
        per_cutoff[str(cutoff)] = {
            "cutoff": cutoff,
            "exist_at_k_qids": exist_qids,
            "exist_at_k_rate": exist_qids / max(len(questions), 1),
            "gold_absent_c35": len(absent_c35),
            "absent_c35_recovered_at_k": len(recovered),
            "absent_c35_recovery_rate": len(recovered) / max(len(absent_c35), 1),
            "oracle_slot5_recall_locked_top4": float(np.mean(slot5_oracle)),
            "oracle_slot5_gain_vs_baseline": float(np.mean(slot5_oracle) - baseline),
            "recovered_qids": recovered,
        }
    report = {
        "kind": "legalir_v28_c1_fielded_bm25_candidate_expansion",
        "schema_version": 1,
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "queries": len(active_questions),
        "full_clean_complement_queries": len(questions),
        "focus_absent_c35": bool(args.focus_absent_c35),
        "baseline_recall": baseline,
        "cutoffs": per_cutoff,
        "retrieval": {
            "backend": "SQLiteFTS5 chunks title/body + broad BM25 + phrase + citation branches",
            "limit": args.retrieval_limit,
            "title_weight": 5.0,
            "passage_weight": 1.0,
            "branch_totals": dict(branch_totals),
            "sqlite_sha256": sha256_file(args.sqlite),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "gate": {
            "c80_recovers_ge_50_percent_absent_c35": per_cutoff.get("80", {}).get("absent_c35_recovery_rate", 0.0) >= 0.50,
            "c80_oracle_gain_ge_0.010": per_cutoff.get("80", {}).get("oracle_slot5_gain_vs_baseline", 0.0) >= 0.010,
            "decision": "fielded_bm25_candidate_lane_has_signal" if per_cutoff.get("80", {}).get("absent_c35_recovery_rate", 0.0) >= 0.50 and per_cutoff.get("80", {}).get("oracle_slot5_gain_vs_baseline", 0.0) >= 0.010 else "fielded_bm25_needs_refinement_or_colbert_diagnostic",
        },
        "label_policy": "qrels are used only after retrieval for clean-complement audit; retrieval construction is label-free",
    }
    write_json(out / "v28_c1_fielded_bm25_report.json", report)
    write_jsonl(out / "v28_c1_candidate_pools.jsonl", rows)
    print(json.dumps({"report": str((out / "v28_c1_fielded_bm25_report.json").resolve()), "cutoffs": per_cutoff, "gate": report["gate"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
