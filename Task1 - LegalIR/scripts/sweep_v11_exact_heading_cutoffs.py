"""Sweep v11 exact-heading rescue reach without rerunning neural models.

The sweep reuses completed v11 Stage-1/Stage-2 audits and the exact-heading
policy. It selects the threshold on fit only, reports holdout, and emits one
public JSON per candidate cutoff. It does not read v57 rank caches or model
outputs; v57 contributes only the algorithmic exact-heading idea.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .evaluate import official_metrics
    from .v11_exact_heading_rescue import (
        apply_policy,
        load_contexts,
        load_references,
        load_stage1,
        load_submission,
        longest_match_tokens,
        normalize,
    )
except ImportError:
    from evaluate import official_metrics
    from v11_exact_heading_rescue import (
        apply_policy,
        load_contexts,
        load_references,
        load_stage1,
        load_submission,
        longest_match_tokens,
        normalize,
    )


def precompute(
    queries: Sequence[Mapping[str, Any]],
    candidates: Sequence[Sequence[str]],
    anchor: Mapping[str, Mapping[str, Sequence[str]]],
    documents: Mapping[str, str],
    titles: Mapping[str, str],
    maximum_cutoff: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute all candidate match lengths once, then reuse prefixes."""
    rows: list[dict[str, Any]] = []
    proposals: list[dict[str, Any]] = []
    for index, item in enumerate(queries):
        qid = str(item["query_id"])
        question = str(item.get("query", item.get("question", "")))
        tokens = normalize(question).split()
        answers = [str(value) for value in anchor[qid]["answer"]]
        tail = answers[-1]
        old_size, old_phrase = longest_match_tokens(tokens, documents.get(tail, ""))
        ranked = list(dict.fromkeys(candidates[index]))[:maximum_cutoff]
        matches = []
        for rank, doc_id in enumerate(ranked, 1):
            if doc_id in answers:
                continue
            size, phrase = longest_match_tokens(tokens, documents.get(doc_id, ""))
            matches.append({
                "rank": rank,
                "doc_id": str(doc_id),
                "size": int(size),
                "phrase": phrase,
                "title": titles.get(str(doc_id), ""),
            })
        rows.append({
            "question_id": qid,
            "question": question,
            "answers": answers,
            "old_id": tail,
            "old_size": int(old_size),
            "old_phrase": old_phrase,
            "matches": matches,
        })
    return rows, proposals


def proposals_for_cutoff(rows: Sequence[Mapping[str, Any]], cutoff: int) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    for row in rows:
        eligible = [item for item in row["matches"] if int(item["rank"]) <= cutoff and int(item["size"]) > 0]
        if not eligible:
            continue
        best = max(eligible, key=lambda item: (int(item["size"]), -int(item["rank"]), str(item["doc_id"])))
        proposals.append({
            "question_id": str(row["question_id"]),
            "question": str(row["question"]),
            "old_id": str(row["old_id"]),
            "new_id": str(best["doc_id"]),
            "old_match_tokens": int(row["old_size"]),
            "old_match_phrase": str(row["old_phrase"]),
            "new_match_tokens": int(best["size"]),
            "new_match_phrase": str(best["phrase"]),
            "best_candidate_rank": int(best["rank"]),
            "old_title": "",
            "new_title": str(best["title"]),
        })
    return proposals


def evaluate_rows(
    rows: Sequence[Mapping[str, Any]],
    anchor: Mapping[str, Mapping[str, Sequence[str]]],
    references: Mapping[str, Mapping[str, Sequence[str]]],
    cutoff: int,
) -> dict[str, Any]:
    proposals = proposals_for_cutoff(rows, cutoff)
    baseline = official_metrics(anchor, references)
    policies: dict[str, Any] = {}
    for threshold in range(10, 5, -1):
        submission, accepted = apply_policy(anchor, proposals, threshold)
        metrics = official_metrics(submission, references)
        policies[str(threshold)] = {
            "changes": len(accepted),
            "metrics": metrics,
            "delta": {key: metrics[key] - baseline[key] for key in ("recall", "precision")},
        }
    chosen = max(
        range(10, 5, -1),
        key=lambda value: (
            policies[str(value)]["delta"]["recall"],
            -policies[str(value)]["changes"],
            value,
        ),
    )
    return {
        "candidate_cutoff": cutoff,
        "proposals": len(proposals),
        "baseline": baseline,
        "policies": policies,
        "chosen_threshold": chosen,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts-zip", type=Path, required=True)
    parser.add_argument("--fit-audit", type=Path, required=True)
    parser.add_argument("--fit-anchor", type=Path, required=True)
    parser.add_argument("--fit-reference", type=Path, required=True)
    parser.add_argument("--holdout-audit", type=Path, required=True)
    parser.add_argument("--holdout-anchor", type=Path, required=True)
    parser.add_argument("--holdout-reference", type=Path, required=True)
    parser.add_argument("--public-audit", type=Path, required=True)
    parser.add_argument("--public-anchor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutoffs", nargs="+", type=int, default=[8, 10, 12, 20, 35, 50, 80, 200])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    contexts = load_contexts(args.contexts_zip)
    documents, titles = contexts
    max_cutoff = max(args.cutoffs)
    split_args = [
        ("fit", args.fit_audit, args.fit_anchor, args.fit_reference),
        ("holdout", args.holdout_audit, args.holdout_anchor, args.holdout_reference),
    ]
    prepared: dict[str, tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]] = {}
    for name, audit_path, anchor_path, reference_path in split_args:
        queries, candidates = load_stage1(audit_path)
        anchor = load_submission(anchor_path)
        references = load_references(reference_path, anchor)
        rows, _ = precompute(queries, candidates, anchor, documents, titles, max_cutoff)
        prepared[name] = (rows, anchor, references)
    public_queries, public_candidates = load_stage1(args.public_audit)
    public_anchor = load_submission(args.public_anchor)
    public_rows, _ = precompute(public_queries, public_candidates, public_anchor, documents, titles, max_cutoff)

    report: dict[str, Any] = {
        "kind": "legalir_v11_exact_heading_cutoff_sweep",
        "policy_selected_on": "fit only",
        "cutoffs": args.cutoffs,
        "v57_prediction_used": False,
        "candidates": {},
    }
    for cutoff in args.cutoffs:
        fit_report = evaluate_rows(*prepared["fit"], cutoff)
        holdout_report = evaluate_rows(*prepared["holdout"], cutoff)
        chosen = int(fit_report["chosen_threshold"])
        public_proposals = proposals_for_cutoff(public_rows, cutoff)
        public_submission, accepted = apply_policy(public_anchor, public_proposals, chosen)
        public_path = args.output_dir / f"v11_exact_heading_cutoff{cutoff}_t{chosen}.json"
        public_path.write_text(json.dumps(public_submission, ensure_ascii=False, indent=2), encoding="utf-8")
        report["candidates"][str(cutoff)] = {
            "fit": fit_report,
            "holdout": holdout_report,
            "public": {"proposals": len(public_proposals), "changes": len(accepted), "submission": str(public_path.resolve())},
        }
        print(json.dumps({"cutoff": cutoff, "fit": fit_report["policies"][str(chosen)], "holdout": holdout_report["policies"][str(chosen)], "public_changes": len(accepted)}, ensure_ascii=True), flush=True)
    report["elapsed_seconds"] = time.perf_counter() - started
    output_report = args.output_dir / "v11_exact_heading_cutoff_sweep_report.json"
    output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(output_report.resolve()), "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
