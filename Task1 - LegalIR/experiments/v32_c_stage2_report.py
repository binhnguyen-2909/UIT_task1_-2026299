"""Rebuild the V32-C report from completed Stage2 JSONL caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import sys
SCRIPT_DIR = Path(__file__).resolve().parent
V27_DIR = SCRIPT_DIR / "v27"
if str(V27_DIR) not in sys.path:
    sys.path.insert(0, str(V27_DIR))
from v27_common import load_metadata, sha256_file, write_json  # noqa: E402


def _load(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def score(row: dict[str, Any]) -> float | None:
        value = row.get("stage2_score", row.get("winner_stage2"))
        return None if value is None else float(value)

    def relevant(row: dict[str, Any]) -> bool:
        return bool(row.get("is_gold", row.get("winner_is_gold", False)))

    margins = sorted({float(row["margin"]) for row in rows if row.get("margin") is not None and score(row) is not None}, reverse=True)
    result = []
    for tau in margins:
        selected = [row for row in rows if row.get("margin") is not None and score(row) is not None and float(row["margin"]) >= tau]
        good = sum(relevant(row) for row in selected)
        result.append({"tau": tau, "selected": len(selected), "rescue": good, "destructive": len(selected) - good, "precision": good / max(len(selected), 1)})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--winners", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-fold", type=int, default=0)
    args = parser.parse_args()
    all_questions = load_metadata(args.metadata)
    questions = {qid: row for qid, row in all_questions.items() if args.probe_fold < 0 or int(qid) % 5 == args.probe_fold}
    qids = set(questions)
    scores = [row for row in _load(args.scores) if str(row["qid"]) in qids]
    winners = [row for row in _load(args.winners) if str(row["qid"]) in qids]
    by_qid: dict[str, list[dict[str, Any]]] = {}
    for row in scores:
        by_qid.setdefault(str(row["qid"]), []).append(row)
    winner_candidates = []
    for qid in questions:
        rows = [row for row in winners if str(row["qid"]) == qid and row.get("winner_doc") is not None]
        if rows:
            winner_candidates.append(max(rows, key=lambda row: (float(row["margin"]), str(row["view"]))))
    winner_gold = sum(bool(row.get("winner_is_gold")) for row in winners)
    query_winner_gold = sum(any(bool(row.get("winner_is_gold")) for row in winners if str(row["qid"]) == qid) for qid in questions)
    gold_recovered = sum(any(bool(row.get("is_gold")) for row in by_qid.get(qid, [])) for qid in questions)
    best_ranks = [min(int(row["view_rank"]) for row in by_qid[qid] if row.get("is_gold")) for qid in questions if any(row.get("is_gold") for row in by_qid.get(qid, []))]
    pareto = _pareto(winner_candidates)
    pocket = [row for row in pareto if float(row["precision"]) >= 0.90 and int(row["rescue"]) > 0 and int(row["destructive"]) <= 2]
    report = {
        "kind": "legalir_v32_c_view_local_stage2_probe",
        "schema_version": 1,
        "queries": len(questions),
        "views": ["Q1", "Q2"],
        "top_k": 20,
        "probe_fold": args.probe_fold,
        "stage2": {
            "score_rows": len(scores),
            "view_winners": len(winners),
            "winner_gold": winner_gold,
            "winner_gold_rate_per_view": winner_gold / max(len(winners), 1),
            "queries_with_any_gold_view_winner": query_winner_gold,
            "query_any_gold_view_winner_rate": query_winner_gold / max(len(questions), 1),
            "gold_recovered_in_view_shortlists": gold_recovered,
            "gold_recovered_rate": gold_recovered / max(len(questions), 1),
            "best_gold_view_rank_median": float(np.median(best_ranks)) if best_ranks else None,
        },
        "threshold_pareto": pareto,
        "gate": {
            "winner_gold_rate_ge_0_30": winner_gold / max(len(winners), 1) >= 0.30,
            "query_winner_gold_rate_ge_0_30": query_winner_gold / max(len(questions), 1) >= 0.30,
            "positive_margin_pocket_precision_ge_0_90_and_rescue": bool(pocket),
            "pass_probe_selection_signal": bool(pocket),
        },
        "inputs": {
            "metadata": str(args.metadata.resolve()),
            "metadata_sha256": sha256_file(args.metadata),
            "scores": str(args.scores.resolve()),
            "scores_sha256": sha256_file(args.scores),
            "winners": str(args.winners.resolve()),
            "winners_sha256": sha256_file(args.winners),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({"report": str(args.output.resolve()), "stage2": report["stage2"], "gate": report["gate"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
