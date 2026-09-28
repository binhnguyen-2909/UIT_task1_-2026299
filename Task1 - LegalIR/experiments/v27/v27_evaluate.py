"""CPU-only calibration, threshold selection, and V27 slot-5 evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from v27_common import (
    REGIMES,
    candidate_swap,
    label_for_doc,
    load_metadata,
    metric_summary,
    read_json,
    write_json,
    write_jsonl,
)


def load_pack_keys(path: Path) -> set[tuple[str, str]]:
    keys = set()
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                keys.add((str(row["qid"]), str(row["doc_id"])))
    return keys


def load_scores(path: Path) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["qid"]), str(row["doc_id"]))
            if key in result:
                raise ValueError(f"duplicate score {key}")
            result[key] = float(row.get("raw_logit", row.get("score")))
    return result


def score_path(path: Path, fold: int, role: str) -> Path:
    """Resolve a fold-specific score file without merging duplicate calibration qids."""
    if path.is_dir():
        candidates = (
            path / f"fold_{fold}_{role}_logits.jsonl",
            path / f"fold_{fold}" / f"fold_{fold}_{role}_logits.jsonl",
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(candidates[-1])
    return path


def fit_platt(scores: Mapping[tuple[str, str], float], questions: Mapping[str, Mapping[str, Any]], qids: Sequence[str], canonical: Mapping[str, str]) -> dict[str, Any]:
    x, y = [], []
    for qid in qids:
        for doc in questions[qid]["c35"]:
            if (qid, doc) not in scores:
                continue
            x.append([float(scores[(qid, doc)])])
            y.append(label_for_doc(questions[qid], doc, canonical))
    if len(set(y)) < 2:
        raise ValueError("Platt calibration needs both relevant and non-relevant labels")
    clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    clf.fit(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.int64))
    return {"kind": "platt_logistic", "coefficient": float(clf.coef_[0][0]), "intercept": float(clf.intercept_[0]), "samples": len(y), "positive_rate": float(np.mean(y))}


def apply_platt(raw: float, calibration: Mapping[str, Any]) -> float:
    z = float(calibration["coefficient"]) * float(raw) + float(calibration["intercept"])
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0))))


def calibrated_scores(scores: Mapping[tuple[str, str], float], qids: Sequence[str], calibration: Mapping[str, Any]) -> dict[tuple[str, str], float]:
    qid_set = set(map(str, qids))
    return {(qid, doc): apply_platt(raw, calibration) for (qid, doc), raw in scores.items() if qid in qid_set}


def evaluate_policy(questions: Mapping[str, Mapping[str, Any]], qids: Sequence[str], probabilities: Mapping[tuple[str, str], float], canonical: Mapping[str, str], *, tau_abs: float, tau_delta: float, fold_map: Mapping[str, int], empty_docs: set[tuple[str, str]] | None = None, bootstrap_samples: int = 10000) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline = {qid: float(len(set(questions[qid]["anchor"]) & questions[qid]["references"]) / max(len(questions[qid]["references"]), 1)) for qid in qids}
    details = []
    empty_docs = empty_docs or set()
    empty_docs = set(empty_docs) | {
        (qid, doc)
        for qid in qids
        for doc in questions[qid]["c35"]
        if (qid, doc) not in probabilities
    }
    for qid in qids:
        result = candidate_swap(questions[qid]["anchor"], qid, probabilities, questions, tau_abs=tau_abs, tau_delta=tau_delta, empty_docs=empty_docs)
        answer = list(result["answer"])
        recall = float(len(set(answer) & questions[qid]["references"]) / max(len(questions[qid]["references"]), 1))
        details.append({"qid": qid, "fold": int(fold_map[qid]), "answer": answer, "baseline_answer": list(questions[qid]["anchor"]), "recall": recall, "baseline_recall": baseline[qid], "delta": recall - baseline[qid], "changed": bool(result["changed"]), "challenger": result.get("challenger"), "reason": result.get("reason"), "top1_4_identical": answer[:4] == questions[qid]["anchor"][:4], "empty_body_output": bool(result.get("changed") and (qid, str(result.get("challenger"))) in empty_docs)})
    return metric_summary(details, baseline, bootstrap_samples=bootstrap_samples), details


def select_thresholds(questions: Mapping[str, Mapping[str, Any]], qids: Sequence[str], probabilities: Mapping[tuple[str, str], float], canonical: Mapping[str, str], fold_map: Mapping[str, int]) -> dict[str, Any]:
    candidates = []
    for tau_abs in np.linspace(0.50, 0.99, 50):
        for tau_delta in np.linspace(0.00, 0.50, 51):
            summary, details = evaluate_policy(questions, qids, probabilities, canonical, tau_abs=float(tau_abs), tau_delta=float(tau_delta), fold_map=fold_map, bootstrap_samples=0)
            valid = summary["rescue_precision"] >= 0.90 and summary["destructive"] <= 2
            key = (int(valid), summary["gain"], summary["net_query_equivalent"], summary["rescue_precision"], -summary["destructive"], -float(tau_abs), -float(tau_delta))
            candidates.append((key, float(tau_abs), float(tau_delta), summary))
    _, tau_abs, tau_delta, summary = max(candidates, key=lambda item: item[0])
    return {"tau_abs": tau_abs, "tau_delta": tau_delta, "calibration_summary": summary, "selection": "maximize gain subject to precision>=0.90 and destructive<=2; otherwise maximize safety key"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--canonical-map", type=Path, required=True)
    parser.add_argument("--calibration-scores", type=Path, required=True)
    parser.add_argument("--outer-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    questions = load_metadata(args.metadata)
    split = read_json(args.split)
    canonical = {}
    for line in args.canonical_map.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            canonical[str(row["doc_id"])] = str(row["canonical_doc_id"])
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    all_details = []
    fold_reports = []
    fold_thresholds = []
    for fold_row in split["folds"]:
        fold = int(fold_row["fold"])
        calibration_qids = [str(qid) for qid in fold_row["calibration_qids"]]
        outer_qids = [str(qid) for qid in fold_row["test_qids"]]
        calibration_scores = load_scores(score_path(args.calibration_scores, fold, "calibration"))
        outer_scores = load_scores(score_path(args.outer_scores, fold, "outer"))
        fold_map_cal = {qid: fold for qid in calibration_qids}
        fold_map_outer = {qid: fold for qid in outer_qids}
        platt = fit_platt(calibration_scores, questions, calibration_qids, canonical)
        cal_prob = calibrated_scores(calibration_scores, calibration_qids, platt)
        threshold = select_thresholds(questions, calibration_qids, cal_prob, canonical, fold_map_cal)
        outer_prob = calibrated_scores(outer_scores, outer_qids, platt)
        summary, details = evaluate_policy(questions, outer_qids, outer_prob, canonical, tau_abs=float(threshold["tau_abs"]), tau_delta=float(threshold["tau_delta"]), fold_map=fold_map_outer)
        for row in details:
            row["outer_fold"] = fold
            row["tau_abs"] = threshold["tau_abs"]
            row["tau_delta"] = threshold["tau_delta"]
        all_details.extend(details)
        fold_thresholds.append({"fold": fold, "tau_abs": threshold["tau_abs"], "tau_delta": threshold["tau_delta"], "calibration": platt})
        fold_reports.append({"fold": fold, "calibration_qids": len(calibration_qids), "outer_qids": len(outer_qids), "threshold": threshold, "outer": summary})

    baseline = {qid: float(len(set(questions[qid]["anchor"]) & questions[qid]["references"]) / max(len(questions[qid]["references"]), 1)) for qid in questions}
    overall = metric_summary(all_details, baseline)
    regime_reports = {}
    for regime in REGIMES:
        assignments = split["regime_folds"][regime]
        rows = []
        for row in all_details:
            clone = dict(row)
            clone["fold"] = int(assignments[str(row["qid"])])
            rows.append(clone)
        regime_reports[regime] = metric_summary(rows, baseline)
    report = {"kind": "legalir_v27_cpu_calibration_and_oof_report", "schema_version": 1, "split": "clean_complement_only", "holdout_opened": False, "public_labels_used": False, "overall": overall, "folds": fold_reports, "fold_thresholds": fold_thresholds, "regimes": regime_reports, "contract_checks": {"top1_4_byte_identical": all(bool(row["top1_4_identical"]) for row in all_details), "empty_body_output": sum(bool(row["empty_body_output"]) for row in all_details), "thresholds_fit_on_calibration_only": True}}
    write_json(out / "v27_oof_report.json", report)
    write_jsonl(out / "v27_oof_decisions.jsonl", all_details)
    print(json.dumps({"overall": overall, "regimes": {k: v["gain"] for k, v in regime_reports.items()}, "report": str((out / 'v27_oof_report.json').resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
