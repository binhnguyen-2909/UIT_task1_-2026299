"""V31-P0 CPU-only complementarity and specialist-selection audit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _load_stage2(sidecar_path: Path, full_path: Path, qids: set[str], candidate_keys: set[tuple[str, str]]) -> tuple[dict[tuple[str, str], float], dict[str, int]]:
    scores: dict[tuple[str, str], float] = {}
    counts = {"sidecar_rows": 0, "full_rows": 0, "matched": 0}
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    for row in sidecar["rows"]:
        key = (str(row["qid"]), str(row["doc_id"]))
        if key in candidate_keys:
            scores[key] = float(row["ce_score"])
            counts["sidecar_rows"] += 1
    with full_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["qid"])
            if qid not in qids:
                continue
            key = (qid, str(row["doc_id"]))
            if key in candidate_keys:
                scores[key] = float(row["stage2_score"])
                counts["full_rows"] += 1
    counts["matched"] = len(scores)
    return scores, counts


def _numeric(docid: str) -> tuple[int, str]:
    return (int(docid), docid) if docid.isdigit() else (10**18, docid)


def _query_old(predictions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["qid"]): row for row in predictions}


def _metric_for_k(rows: list[dict[str, Any]], old: dict[str, dict[str, Any]], stage2: dict[tuple[str, str], float], k: int) -> dict[str, Any]:
    base_gold_50 = base_gold_100 = ft_gold_20 = ft_gold_50 = ft_gold_100 = 0
    union_gold_100 = unique_gold = added_vs_base100 = added_vs_base500 = 0
    queries_with_unique = 0
    best_challenger_gold = 0
    ft_top1_gold = ft_top3_gold = ft_top5_gold = 0
    swaps = rescue = destructive = neutral = 0
    stage2_choices = stage2_gold = stage2_beats_slot5 = 0
    stage2_relevant_swaps = 0
    stage2_destructive_swaps = 0
    stage2_neutral_swaps = 0
    coverage = 0
    for row in rows:
        qid = str(row["qid"])
        refs = {str(docid) for docid in row["references"]}
        candidates = list(row["candidates"])
        base_rank = {str(candidate["doc_id"]): int(candidate["dense_rank"]) for candidate in candidates}
        ft_order = sorted(candidates, key=lambda candidate: (int(candidate["ft_dense_rank"]), int(candidate["dense_rank"]), _numeric(str(candidate["doc_id"]))))
        ft_rank = {str(candidate["doc_id"]): int(candidate["ft_dense_rank"]) for candidate in candidates}
        base_gold_50 += sum(1 for docid in refs if base_rank.get(docid, 1001) <= 50)
        base_gold_100 += sum(1 for docid in refs if base_rank.get(docid, 1001) <= 100)
        ft_gold_20 += sum(1 for docid in refs if ft_rank.get(docid, 1001) <= 20)
        ft_gold_50 += sum(1 for docid in refs if ft_rank.get(docid, 1001) <= 50)
        ft_gold_100 += sum(1 for docid in refs if ft_rank.get(docid, 1001) <= 100)
        ft_top = ft_order[:k]
        ft_top_ids = {str(candidate["doc_id"]) for candidate in ft_top}
        unique_docs = {docid for docid in refs if docid in ft_top_ids and base_rank.get(docid, 1001) > 100}
        unique_gold += len(unique_docs)
        union_gold_100 += sum(1 for docid in refs if base_rank.get(docid, 1001) <= 100 or docid in ft_top_ids)
        added_vs_base100 += sum(1 for candidate in ft_top if int(candidate["dense_rank"]) > 100)
        added_vs_base500 += sum(1 for candidate in ft_top if int(candidate["dense_rank"]) > 500)
        if unique_docs:
            queries_with_unique += 1
        old_row = old.get(qid, {})
        old_answer = [str(docid) for docid in old_row.get("old_answer", [])]
        old_top14 = set(old_answer[:4])
        new_candidates = [candidate for candidate in ft_top if str(candidate["doc_id"]) not in old_top14]
        best_new = new_candidates[0] if new_candidates else None
        if best_new is not None and str(best_new["doc_id"]) in refs:
            best_challenger_gold += 1
        if any(str(candidate["doc_id"]) in refs for candidate in ft_order[:1]):
            ft_top1_gold += 1
        if any(str(candidate["doc_id"]) in refs for candidate in ft_order[:3]):
            ft_top3_gold += 1
        if any(str(candidate["doc_id"]) in refs for candidate in ft_order[:5]):
            ft_top5_gold += 1

        old_has_gold = bool(set(old_answer[:5]) & refs)
        new_stage2 = []
        for candidate in new_candidates:
            key = (qid, str(candidate["doc_id"]))
            if key in stage2 and math.isfinite(float(stage2[key])):
                new_stage2.append((float(stage2[key]), -int(candidate["ft_dense_rank"]), str(candidate["doc_id"])))
        if new_stage2:
            coverage += 1
            choice = max(new_stage2)
            choice_doc = choice[2]
            choice_score = choice[0]
            stage2_choices += 1
            stage2_gold += int(choice_doc in refs)
            old_slot5_score = old_row.get("old_slot5_stage2")
            if old_slot5_score is not None and math.isfinite(float(old_slot5_score)) and choice_score > float(old_slot5_score):
                swaps += 1
                old_answer_new = [*old_answer[:4], choice_doc]
                new_has_gold = bool(set(old_answer_new) & refs)
                rescue += int(not old_has_gold and new_has_gold)
                destructive += int(old_has_gold and not new_has_gold)
                neutral += int(old_has_gold == new_has_gold and not (not old_has_gold and new_has_gold))
                stage2_relevant_swaps += int(not old_has_gold and new_has_gold)
                stage2_destructive_swaps += int(old_has_gold and not new_has_gold)
                stage2_neutral_swaps += int(old_has_gold == new_has_gold and not (not old_has_gold and new_has_gold))
                stage2_beats_slot5 += 1
    query_count = len(rows)
    return {
        "k_ft": k,
        "queries": query_count,
        "base_gold_docs_at_50": base_gold_50,
        "base_gold_docs_at_100": base_gold_100,
        "ft_gold_docs_at_20": ft_gold_20,
        "ft_gold_docs_at_50": ft_gold_50,
        "ft_gold_docs_at_100": ft_gold_100,
        "unique_ft_gold_outside_base100": unique_gold,
        "union_gold_docs_at_100": union_gold_100,
        "union_gain_vs_base100": union_gold_100 - base_gold_100,
        "candidate_count_added_vs_base100": added_vs_base100,
        "candidate_count_added_vs_base500": added_vs_base500,
        "queries_with_unique_ft_gold": queries_with_unique,
        "best_specialist_challenger_gold": best_challenger_gold,
        "best_specialist_challenger_gold_rate_conditional": best_challenger_gold / max(queries_with_unique, 1),
        "ft_gold_is_top1_queries": ft_top1_gold,
        "ft_gold_is_top3_queries": ft_top3_gold,
        "ft_gold_is_top5_queries": ft_top5_gold,
        "stage2_coverage_queries": coverage,
        "stage2_best_choice_queries": stage2_choices,
        "stage2_best_choice_gold": stage2_gold,
        "stage2_best_choice_gold_precision": stage2_gold / max(stage2_choices, 1),
        "stage2_best_choice_beats_slot5": stage2_beats_slot5,
        "candidate_swaps": swaps,
        "rescue": rescue,
        "destructive": destructive,
        "neutral": neutral,
        "rescue_precision_against_destructive": rescue / max(rescue + destructive, 1),
        "best_rescue_destructive_le_2": rescue if destructive <= 2 else 0,
        "best_specialist_rate_vs_v29_baseline_5_of_31": (best_challenger_gold / max(queries_with_unique, 1)) / (5 / 31),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specialist-scores", type=Path, required=True)
    parser.add_argument("--old-predictions", type=Path, required=True)
    parser.add_argument("--stage2-sidecar", type=Path, required=True)
    parser.add_argument("--stage2-full", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(args.specialist_scores)
    old = _query_old(_read_jsonl(args.old_predictions))
    qids = {str(row["qid"]) for row in rows}
    candidate_keys = {(str(row["qid"]), str(candidate["doc_id"])) for row in rows for candidate in row["candidates"]}
    stage2, stage2_counts = _load_stage2(args.stage2_sidecar, args.stage2_full, qids, candidate_keys)
    results = [_metric_for_k(rows, old, stage2, k) for k in (10, 20, 50, 100)]
    max_unique = max((int(row["unique_ft_gold_outside_base100"]) for row in results), default=0)
    recommended = next((row["k_ft"] for row in results if int(row["unique_ft_gold_outside_base100"]) >= math.ceil(0.85 * max_unique)), None)
    complementarity_pass = any(int(row["unique_ft_gold_outside_base100"]) >= 8 or int(row["union_gain_vs_base100"]) >= 5 for row in results)
    rank_pass = any(float(row["best_specialist_challenger_gold_rate_conditional"]) >= 2 * (5 / 31) or float(row["best_specialist_challenger_gold_rate_conditional"]) >= (5 / 31) + 0.15 for row in results)
    selection_pass = any(int(row["best_rescue_destructive_le_2"]) > 0 and float(row["rescue_precision_against_destructive"]) >= 0.90 for row in results)
    report = {
        "kind": "legalir_v31_p0_cpu_specialist_union_audit",
        "schema_version": 1,
        "specialist_scores": str(args.specialist_scores.resolve()),
        "old_predictions": str(args.old_predictions.resolve()),
        "stage2_cache": {"sidecar": str(args.stage2_sidecar.resolve()), "full": str(args.stage2_full.resolve()), "matched_candidate_scores": len(stage2), **stage2_counts},
        "queries": len(rows),
        "baseline_v29_specialist_reference": {"new_gold_found": 31, "new_gold_best_challenger": 5, "rate": 5 / 31},
        "results": results,
        "recommended_k": recommended,
        "gate": {"complementarity": complementarity_pass, "rank_compression": rank_pass, "selection_signal": selection_pass, "pass": bool(complementarity_pass and rank_pass and selection_pass)},
    }
    out = args.output_dir / "v31_p0_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"report": str(out.resolve()), "gate": report["gate"], "recommended_k": recommended, "stage2_matches": len(stage2)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
