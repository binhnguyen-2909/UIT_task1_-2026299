"""CPU-only nested margin calibration and funnel evaluation for V29-A1D."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
V27_EXPERIMENTS = SCRIPT_DIR.parent / "v27"
if str(V27_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(V27_EXPERIMENTS))

from v27_common import load_metadata, load_sidecar, read_json, sha256_file, write_json, write_jsonl  # noqa: E402


def _numeric_key(value: str) -> tuple[int, str]:
    return (int(value), value) if str(value).isdigit() else (10**18, str(value))


def _fold(qid: str, question: Mapping[str, Any]) -> int | str:
    value = question.get("fold")
    return int(value) if value is not None else int(qid) % 5


def _bootstrap_lb(gains: np.ndarray, seed: int = 42, repeats: int = 10000) -> float:
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, repeats, 500):
        size = min(500, repeats - start)
        indices = rng.integers(0, len(gains), size=(size, len(gains)))
        estimates.append(gains[indices].mean(axis=1))
    return float(np.quantile(np.concatenate(estimates), 0.025))


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    old = np.asarray([float(row["recall_old"]) for row in rows], dtype=np.float64)
    new = np.asarray([float(row["recall_new"]) for row in rows], dtype=np.float64)
    gains = new - old
    promotions = [row for row in rows if row["promoted"]]
    relevant = [row for row in promotions if row["promotion_is_relevant"]]
    fold_values = []
    for fold in sorted({str(row["fold"]) for row in rows}, key=_numeric_key):
        fold_rows = [row for row in rows if str(row["fold"]) == fold]
        fold_values.append({"fold": fold, "queries": len(fold_rows), "gain": float(np.mean([float(row["new_recall_gain"]) for row in fold_rows])), "promotions": sum(bool(row["promoted"]) for row in fold_rows), "destructive": sum(bool(row["promoted"] and not row["promotion_is_relevant"]) for row in fold_rows)})
    lb = _bootstrap_lb(gains)
    return {
        "queries": len(rows),
        "baseline_recall": float(np.mean(old)),
        "new_recall": float(np.mean(new)),
        "gain": float(np.mean(gains)),
        "robust_gain_min_fold": min((float(row["gain"]) for row in fold_values), default=0.0),
        "bootstrap_lb95_gain": lb,
        "promotions": len(promotions),
        "relevant_promotions": len(relevant),
        "destructive": len(promotions) - len(relevant),
        "neutral": sum(float(row["new_recall_gain"]) == 0.0 for row in rows),
        "promotion_precision": len(relevant) / max(len(promotions), 1),
        "top1_4_identical": all(bool(row["top1_4_identical"]) for row in rows),
        "empty_output": sum(not row["answer"] for row in rows),
        "folds": fold_values,
    }


def _choose_tau(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    margins = sorted({float(row["margin"]) for row in train_rows if math.isfinite(float(row["margin"]))})
    candidates = [float("inf"), *margins]
    evaluated = []
    for tau in candidates:
        promoted = [row for row in train_rows if float(row["margin"]) >= tau]
        relevant = [row for row in promoted if row["challenger_is_relevant"]]
        destructive = len(promoted) - len(relevant)
        precision = len(relevant) / max(len(promoted), 1)
        gain = float(np.mean([float(row["recall_gain_if_promoted"]) if row in promoted else 0.0 for row in train_rows]))
        if precision >= 0.90 and destructive <= 2:
            evaluated.append((gain, tau, precision, destructive, len(promoted)))
    if not evaluated:
        return {"tau": float("inf"), "selection": "conservative_abstain_no_train_threshold_met", "train_rows": len(train_rows)}
    # Maximize clean-complement training gain under the preregistered safety
    # constraints; highest tau breaks ties conservatively.
    gain, tau, precision, destructive, promotions = max(evaluated, key=lambda item: (item[0], -item[1]))
    return {"tau": tau, "selection": "max_train_gain_precision_ge_0.90_destructive_le_2", "train_rows": len(train_rows), "train_gain": gain, "train_precision": precision, "train_destructive": destructive, "train_promotions": promotions, "candidate_thresholds": len(candidates)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--stage1-audit", type=Path, required=True)
    parser.add_argument("--stage2-sidecar", type=Path, required=True)
    parser.add_argument("--frozen-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    questions = load_metadata(args.metadata)
    sidecar = load_sidecar(args.stage2_sidecar, questions)
    by_qid: dict[str, list[dict[str, Any]]] = {}
    with args.frozen_cache.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                by_qid.setdefault(str(row["qid"]), []).append(row)
    if set(by_qid) != set(questions):
        raise ValueError("frozen cache misses qids")
    base_rows = []
    for qid, question in questions.items():
        candidates = sorted(by_qid[qid], key=lambda row: (float(row["stage2_score"]), -int(row["dense_rank"])), reverse=True)
        challenger = candidates[0] if candidates else None
        old_slot5 = str(question["anchor"][4])
        incumbent = float(sidecar[(qid, old_slot5)])
        challenger_score = float(challenger["stage2_score"]) if challenger else float("-inf")
        margin = challenger_score - incumbent
        refs = set(question["references"])
        base_rows.append({
            "qid": qid,
            "fold": _fold(qid, question),
            "old_answer": list(question["anchor"]),
            "old_slot5_doc": old_slot5,
            "old_slot5_stage2": incumbent,
            "challenger_doc": challenger["docid"] if challenger else None,
            "challenger_stage2": challenger_score if challenger else None,
            "challenger_dense_rank": challenger["dense_rank"] if challenger else None,
            "margin": margin,
            "challenger_is_relevant": bool(challenger and challenger["docid"] in refs),
            "dense_new_gold_docs": [row["docid"] for row in candidates if row["docid"] in refs],
            "dense_new_gold_beats_slot5": [row["docid"] for row in candidates if row["docid"] in refs and float(row["stage2_score"]) > incumbent],
            "recall_old": len(set(question["anchor"]) & refs) / max(len(refs), 1),
            "recall_gain_if_promoted": (len(set(question["anchor"][:4] + ([challenger["docid"]] if challenger else [old_slot5])) & refs) / max(len(refs), 1) - len(set(question["anchor"]) & refs) / max(len(refs), 1)) if challenger else 0.0,
            "reference_count": len(refs),
            "dense_new_candidate_count": len(candidates),
        })
    folds = sorted({str(row["fold"]) for row in base_rows}, key=_numeric_key)
    calibrated_rows = []
    threshold_rows = []
    for held_fold in folds:
        train = [row for row in base_rows if str(row["fold"]) != held_fold]
        test = [row for row in base_rows if str(row["fold"]) == held_fold]
        selection = _choose_tau(train)
        tau = float(selection["tau"])
        threshold_rows.append({"held_out_fold": held_fold, **selection})
        for row in test:
            promoted = bool(row["challenger_doc"] is not None and float(row["margin"]) >= tau)
            answer = list(row["old_answer"][:4]) + ([row["challenger_doc"]] if promoted else [row["old_slot5_doc"]])
            refs = set(questions[row["qid"]]["references"])
            calibrated_rows.append({
                **row,
                "tau": tau,
                "promoted": promoted,
                "answer": answer,
                "top1_4_identical": answer[:4] == row["old_answer"][:4],
                "promotion_is_relevant": bool(promoted and row["challenger_is_relevant"]),
                "recall_new": len(set(answer) & refs) / max(len(refs), 1),
                "new_recall_gain": len(set(answer) & refs) / max(len(refs), 1) - row["recall_old"],
                "threshold_survived": promoted,
            })
    calibrated_rows.sort(key=lambda row: _numeric_key(str(row["qid"])))
    metrics = _metrics(calibrated_rows)
    funnel = {
        "new_gold_found": sum(len(row["dense_new_gold_docs"]) for row in base_rows),
        "new_gold_beats_slot5": sum(len(row["dense_new_gold_beats_slot5"]) for row in base_rows),
        "gold_survives_calibrated_threshold": sum(bool(row["challenger_is_relevant"] and row["threshold_survived"]) for row in calibrated_rows),
        "candidate_swaps": sum(bool(row["promoted"]) for row in calibrated_rows),
        "rescue": sum(bool(row["promotion_is_relevant"] and row["new_recall_gain"] > 0) for row in calibrated_rows),
        "destructive": sum(bool(row["promoted"] and not row["promotion_is_relevant"]) for row in calibrated_rows),
        "neutral": sum(float(row["new_recall_gain"]) == 0.0 for row in calibrated_rows),
    }
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "a1d_calibrated_predictions.jsonl", calibrated_rows)
    write_json(out / "a1d_nested_thresholds.json", {"kind": "legalir_v29_a1d_nested_thresholds", "split": "clean_complement_only", "holdout_opened": False, "public_labels_used": False, "folds": threshold_rows})
    write_json(out / "a1d_funnel.json", {"kind": "legalir_v29_a1d_funnel", "split": "clean_complement_only", "holdout_opened": False, "public_labels_used": False, "funnel": funnel})
    gate = {
        "gain_ge_0_010": metrics["gain"] >= 0.010,
        "robust_gain_ge_0_008": metrics["robust_gain_min_fold"] >= 0.008,
        "bootstrap_lb95_ge_0_003": metrics["bootstrap_lb95_gain"] >= 0.003,
        "precision_ge_0_90": metrics["promotion_precision"] >= 0.90,
        "destructive_le_2": metrics["destructive"] <= 2,
        "no_negative_regime": all(float(row["gain"]) >= 0.0 for row in metrics["folds"]),
        "top1_4_identical": metrics["top1_4_identical"],
        "empty_output_zero": metrics["empty_output"] == 0,
        "holdout_closed": True,
    }
    gate["pass"] = all(gate.values())
    report = {
        "kind": "legalir_v29_a1d_nested_calibrated_oof",
        "schema_version": 1,
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "selection_performed": True,
        "calibration": "nested leave-one-fold-out; threshold selected only on training folds under precision>=0.90 and destructive<=2",
        "inputs": {"metadata": str(args.metadata.resolve()), "metadata_sha256": sha256_file(args.metadata), "stage1_audit_sha256": sha256_file(args.stage1_audit), "stage2_sidecar_sha256": sha256_file(args.stage2_sidecar), "frozen_cache": str(args.frozen_cache.resolve()), "frozen_cache_sha256": sha256_file(args.frozen_cache)},
        "metrics": metrics,
        "funnel": funnel,
        "gate": gate,
        "artifacts": {"predictions": str((out / "a1d_calibrated_predictions.jsonl").resolve()), "thresholds": str((out / "a1d_nested_thresholds.json").resolve()), "funnel": str((out / "a1d_funnel.json").resolve())},
    }
    write_json(out / "a1d_nested_calibrated_report.json", report)
    print(json.dumps({"report": str((out / 'a1d_nested_calibrated_report.json').resolve()), "metrics": metrics, "funnel": funnel, "gate": gate}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
