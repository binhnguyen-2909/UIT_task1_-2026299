"""CPU candidate-ceiling oracle curve for V28-C0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from v27_common import load_metadata, read_json, sha256_file, write_json, write_jsonl


def recall(answer: list[str], refs: set[str]) -> float:
    return len(set(answer) & refs) / max(len(refs), 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--stage1-audit", type=Path, required=True)
    parser.add_argument("--slot5-oracle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutoffs", default="35,50,80,120")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    questions = load_metadata(args.metadata)
    stage1 = read_json(args.stage1_audit)
    by_qid = {str(row["query_id"]): row for row in stage1["queries"]}
    candidate_by_qid = {str(row["query_id"]): [str(doc) for doc in row] for row in stage1["queries"] for row in []}
    # The audit stores the long candidate lists separately from its query
    # metadata, but both lists use the same deterministic ordering.
    stage1_candidates = {str(query["query_id"]): [str(doc) for doc in candidates] for query, candidates in zip(stage1["queries"], stage1["stage1_candidates"], strict=True)}
    if set(stage1_candidates) < set(questions):
        raise ValueError("Stage-1 audit does not cover all current clean-complement qids")
    cutoffs = sorted({int(value) for value in args.cutoffs.split(",")})
    if any(value <= 0 for value in cutoffs):
        raise ValueError("cutoffs must be positive")
    slot5_oracle = read_json(args.slot5_oracle)
    per_cutoff = {}
    per_query_rows = []
    previous_present: set[str] = set()
    baseline_recall = float(np.mean([recall(row["anchor"], row["references"]) for row in questions.values()]))
    for cutoff in cutoffs:
        present = set()
        recovered_rows = []
        oracle_recalls = []
        unrestricted_oracle_recalls = []
        for qid, question in questions.items():
            candidates = list(dict.fromkeys(stage1_candidates[qid][:cutoff]))
            refs = set(map(str, question["references"]))
            gold_here = sorted(refs & set(candidates))
            present.update(f"{qid}:{doc}" for doc in gold_here)
            before = set(stage1_candidates[qid][:35])
            newly = sorted(refs & set(candidates) - before) if cutoff > 35 else []
            if newly:
                recovered_rows.append({"qid": qid, "gold_recovered": newly})
            # Keep V16b/V27 top1–4 locked and use the first exposed gold in
            # slot5.  This is the relevant oracle for the actual policy.
            slot5 = next((doc for doc in candidates[4:] if doc in refs), question["anchor"][4])
            locked_answer = list(question["anchor"][:4]) + [slot5]
            oracle_recalls.append(recall(locked_answer, refs))
            unrestricted = list(dict.fromkeys(gold_here))[:5]
            unrestricted_oracle_recalls.append(recall(unrestricted, refs) if unrestricted else recall(question["anchor"], refs))
            if cutoff == cutoffs[0] or qid not in {row["qid"] for row in per_query_rows}:
                per_query_rows.append({"qid": qid, "gold": sorted(refs), "candidate_counts": {}})
            next(row for row in per_query_rows if row["qid"] == qid)["candidate_counts"][str(cutoff)] = {"gold_present": gold_here, "slot5_oracle_recall": oracle_recalls[-1], "unrestricted_oracle_recall": unrestricted_oracle_recalls[-1]}
        present_qids = sum(bool(set(map(str, questions[qid]["references"])) & set(stage1_candidates[qid][:cutoff])) for qid in questions)
        absent_c35 = [qid for qid, question in questions.items() if not (set(map(str, question["references"])) & set(stage1_candidates[qid][:35]))]
        absent_recovered = [qid for qid in absent_c35 if set(map(str, questions[qid]["references"])) & set(stage1_candidates[qid][:cutoff])]
        per_cutoff[str(cutoff)] = {
            "cutoff": cutoff,
            "queries": len(questions),
            "exist_at_k_qids": present_qids,
            "exist_at_k_rate": present_qids / max(len(questions), 1),
            "new_gold_recovered_vs_c35": len(recovered_rows),
            "gold_absent_c35": len(absent_c35),
            "absent_c35_recovered_at_k": len(absent_recovered),
            "absent_c35_recovery_rate": len(absent_recovered) / max(len(absent_c35), 1),
            "oracle_slot5_recall_locked_top4": float(np.mean(oracle_recalls)),
            "oracle_slot5_gain_vs_baseline": float(np.mean(oracle_recalls) - baseline_recall),
            "unrestricted_oracle_recall_at_5": float(np.mean(unrestricted_oracle_recalls)),
            "unrestricted_oracle_gain_vs_baseline": float(np.mean(unrestricted_oracle_recalls) - baseline_recall),
            "recovered_qids": recovered_rows,
        }
        previous_present = present
    summary = {
        "kind": "legalir_v28_c0_candidate_oracle_curve",
        "schema_version": 1,
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "metadata_qids": len(questions),
        "stage1_audit_questions": int(stage1["questions"]),
        "stage1_audit_sha256": sha256_file(args.stage1_audit),
        "baseline_recall": baseline_recall,
        "source_candidate_generation": "frozen clean-complement Stage-1 audit; no retrieval rerun",
        "cutoffs": per_cutoff,
        "gate": {
            "c80_recovers_ge_50_percent_absent_c35": per_cutoff.get("80", {}).get("absent_c35_recovery_rate", 0.0) >= 0.50,
            "c80_oracle_gain_ge_0.010": per_cutoff.get("80", {}).get("oracle_slot5_gain_vs_baseline", 0.0) >= 0.010,
            "decision": "c80_candidate_lane_is_worth_c1_fielded_bm25" if per_cutoff.get("80", {}).get("absent_c35_recovery_rate", 0.0) >= 0.50 and per_cutoff.get("80", {}).get("oracle_slot5_gain_vs_baseline", 0.0) >= 0.010 else "candidate_expansion_signal_is_weak_or_needs_c1_retrieval",
        },
        "slot5_oracle_reference": str(args.slot5_oracle.resolve()),
    }
    write_json(out / "v28_c0_candidate_oracle_curve.json", summary)
    write_jsonl(out / "v28_c0_candidate_oracle_per_query.jsonl", per_query_rows)
    print(json.dumps({"report": str((out / "v28_c0_candidate_oracle_curve.json").resolve()), "cutoffs": per_cutoff, "gate": summary["gate"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
