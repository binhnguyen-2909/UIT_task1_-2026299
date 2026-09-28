#!/usr/bin/env python3
"""CPU evaluator for R1's frozen E3B+Qwen c35-long fusion.

All expensive neural work is completed by ``r1_qwen_c35_cache.py`` and the
E3B fold-safe score exports.  This evaluator never loads a neural checkpoint:
it validates the two-seed C/V lineage, delegates lambda=0 to the original E3B
policy for exact parity, and selects only lambda 0/0.25/0.50 on C.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e3b_residual_score_gate as e3b_gate  # noqa: E402
from r1_contract import (  # noqa: E402
    EPSILON, R1_BASELINE_THRESHOLD, R1_DISAGREEMENT_WEIGHT, R1_KIND,
    R1_LAMBDAS, E3BFoldScores, apply_r1_policy, load_e3b_fold_scores,
)
from r1_contract import require_complement_file  # noqa: E402
from v27_a_common import (  # noqa: E402
    CANDIDATE_LIMIT, V27ContractError, build_regime_groups, group_folds,
    json_sha256, load_manifest, read_json, sha256_file, transition_metrics,
    write_json,
)
from qwen_c200_common import load_candidate_universe  # noqa: E402


EVAL_KIND = f"{R1_KIND}_nested_eval"
SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "standard", "stress"), default="preflight")
    parser.add_argument("--stage1", type=Path, required=True); parser.add_argument("--stage2", type=Path, required=True); parser.add_argument("--anchor", type=Path, required=True); parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--e1-manifest", type=Path, required=True); parser.add_argument("--e3b-standard-report", type=Path, required=True); parser.add_argument("--e3b-stress-report", type=Path, required=True)
    parser.add_argument("--qwen-cache-root", type=Path, required=True); parser.add_argument("--gate-source", type=Path, required=True); parser.add_argument("--gate-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--standard-report", type=Path); parser.add_argument("--checkpoint-path", type=Path); parser.add_argument("--report-path", type=Path); parser.add_argument("--resume", dest="resume", action="store_true"); parser.add_argument("--no-resume", dest="resume", action="store_false"); parser.set_defaults(resume=True); parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _safe(path: Path, label: str) -> None:
    require_complement_file(path, label)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip(): continue
            row = json.loads(line)
            if not isinstance(row, dict): raise V27ContractError(f"{path}:{number}: invalid JSONL row")
            rows.append(row)
    return rows


def _load_qwen_cache(root: Path, *, expected_qids: Sequence[str], c35_by_qid: Mapping[str, Sequence[str]]) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    actual_root = root / "r1_qwen_cache" if (root / "r1_qwen_cache").is_dir() else root
    metas = sorted((actual_root / "metadata").glob("*.json")); shards = sorted((actual_root / "shards").glob("*.jsonl"))
    if not metas or not shards: raise FileNotFoundError(f"R1 Qwen cache is incomplete: {actual_root}")
    lineage_sha: str | None = None; shard_ids: set[tuple[int, int]] = set(); metadata_paths: list[str] = []
    for path in metas:
        meta = read_json(path)
        if not isinstance(meta, Mapping) or meta.get("kind") != f"{R1_KIND}_qwen_cache": raise V27ContractError(f"Invalid R1 Qwen cache metadata: {path}")
        current = str(meta.get("lineage_sha256", "")); lineage_sha = current if lineage_sha is None else lineage_sha
        if current != lineage_sha: raise V27ContractError("R1 Qwen cache shards have different lineage")
        index, count = int(meta.get("shard_index", -1)), int(meta.get("shard_count", -1))
        if count < 1 or not 0 <= index < count or (index, count) in shard_ids: raise V27ContractError("Invalid/duplicate R1 Qwen cache shard metadata")
        shard_ids.add((index, count)); metadata_paths.append(str(path.resolve()))
    counts = {count for _, count in shard_ids}
    if len(counts) != 1 or len(shard_ids) != next(iter(counts)): raise V27ContractError("R1 Qwen cache is missing a declared shard")
    scores: dict[str, dict[str, dict[str, Any]]] = {}
    for path in shards:
        for row in _read_jsonl(path):
            qid, doc_id = str(row.get("qid", "")), str(row.get("doc_id", "")); value = row.get("yes_no_logit")
            if qid not in set(map(str, expected_qids)) or doc_id not in set(c35_by_qid[qid]) or row.get("lineage_sha256") != lineage_sha or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or doc_id in scores.get(qid, {}):
                raise V27ContractError(f"R1 Qwen cache row is stale, invalid or duplicate: {qid}/{doc_id}")
            expected_rank = list(c35_by_qid[qid]).index(doc_id) + 1
            if int(row.get("candidate_rank", -1)) != expected_rank: raise V27ContractError(f"{qid}/{doc_id}: R1 candidate rank drift")
            scores.setdefault(qid, {})[doc_id] = row
    for qid in map(str, expected_qids):
        if set(scores.get(qid, {})) != set(c35_by_qid[qid]): raise V27ContractError(f"{qid}: R1 Qwen cache is not complete c35")
    return scores, {"root": str(actual_root.resolve()), "lineage_sha256": lineage_sha, "metadata": metadata_paths, "shards": [str(path.resolve()) for path in shards], "qids": len(scores), "pairs": sum(len(value) for value in scores.values())}


def _load_gate(path: Path):
    _safe(path, "R1 promotion gate source")
    spec = importlib.util.spec_from_file_location("r1_bound_gate", path)
    if spec is None or spec.loader is None: raise V27ContractError("Cannot import R1 bound gate source")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); gate = getattr(module, "_gate", None)
    if not callable(gate): raise V27ContractError("R1 gate source does not expose _gate")
    return gate


def _subset(split: Any, qids: Sequence[str]) -> Any:
    positions = {str(qid): index for index, qid in enumerate(split.qids)}; indices = [positions[str(qid)] for qid in qids]
    class View: pass
    view = View(); view.qids = list(qids); view.questions = [split.questions[index] for index in indices]; view.candidates = [split.candidates[index] for index in indices]; view.anchor = [split.anchor[index] for index in indices]; view.references = [split.references[index] for index in indices]
    return view


def _metrics(split: Any, predictions: Mapping[str, Mapping[str, Sequence[str]]]) -> dict[str, Any]:
    return transition_metrics(split, predictions)


def _choose_lambda(split: Any, qids: Sequence[str], e2_by_seed: Sequence[Mapping[str, Mapping[str, Any]]], qwen_scores: Mapping[str, Mapping[str, Mapping[str, Any]]], c35_by_qid: Mapping[str, Sequence[str]]) -> tuple[float, list[dict[str, Any]]]:
    view = _subset(split, qids); trials: list[dict[str, Any]] = []
    for lam in R1_LAMBDAS:
        pred, _ = apply_r1_policy(view, list(range(len(qids))), e2_by_seed, qwen_scores, lam)
        metrics = _metrics(view, pred); trials.append({"lambda": lam, "metrics": {key: value for key, value in metrics.items() if key != "rows"}})
    eligible = [trial for trial in trials if float(trial["metrics"]["gain"]) > EPSILON]
    if not eligible: return 0.0, trials
    selected = max(eligible, key=lambda trial: (float(trial["metrics"]["gain"]), -int(trial["metrics"]["destructive_changes"]), -float(trial["lambda"])))
    return float(selected["lambda"]), trials


def _fold_run(split: Any, fold: E3BFoldScores, qwen_scores: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> tuple[dict[str, dict[str, list[str]]], dict[str, Any]]:
    c35_by_qid = {str(qid): [str(value) for value in split.candidates[index][:CANDIDATE_LIMIT]] for index, qid in enumerate(split.qids)}
    calibration_qids = list(fold.calibration_qids); validation_qids = list(fold.validation_qids)
    calibration_view = _subset(split, calibration_qids); validation_view = _subset(split, validation_qids)
    baseline_c, baseline_c_decisions = e3b_gate.apply_policy(calibration_view, list(range(len(calibration_qids))), fold.calibration_by_seed, {}, e3b_gate.E3BPolicy(memory_weight=0.0, threshold=R1_BASELINE_THRESHOLD))
    baseline_v, baseline_v_decisions = e3b_gate.apply_policy(validation_view, list(range(len(validation_qids))), fold.validation_by_seed, {}, e3b_gate.E3BPolicy(memory_weight=0.0, threshold=R1_BASELINE_THRESHOLD))
    lambda_selected, trace = _choose_lambda(split, calibration_qids, fold.calibration_by_seed, qwen_scores, c35_by_qid)
    selected_c, _ = apply_r1_policy(calibration_view, list(range(len(calibration_qids))), fold.calibration_by_seed, qwen_scores, lambda_selected)
    selected_v, selected_decisions = apply_r1_policy(validation_view, list(range(len(validation_qids))), fold.validation_by_seed, qwen_scores, lambda_selected)
    parity_pred, parity_decisions = apply_r1_policy(validation_view, list(range(len(validation_qids))), fold.validation_by_seed, qwen_scores, 0.0)
    parity = all(parity_pred[qid] == baseline_v[qid] and bool(parity_decisions[qid].get("original", {}).get("changed", False)) == bool(baseline_v_decisions[qid].get("changed", False)) for qid in validation_qids)
    if not parity: raise V27ContractError(f"R1 lambda=0 baseline parity failed in {fold.regime}/seed={fold.inner_seed}/fold={fold.outer_fold}")
    selected_c_metrics = _metrics(calibration_view, selected_c); baseline_c_metrics = _metrics(calibration_view, baseline_c); selected_v_metrics = _metrics(validation_view, selected_v); baseline_v_metrics = _metrics(validation_view, baseline_v)
    return selected_v, {"regime": fold.regime, "inner_seed": fold.inner_seed, "outer_fold": fold.outer_fold, "calibration_qids": calibration_qids, "validation_qids": validation_qids, "selected_lambda": lambda_selected, "lambda_trace": trace, "baseline_calibration": {key: value for key, value in baseline_c_metrics.items() if key != "rows"}, "selected_calibration": {key: value for key, value in selected_c_metrics.items() if key != "rows"}, "baseline_validation": {key: value for key, value in baseline_v_metrics.items() if key != "rows"}, "validation": {key: value for key, value in selected_v_metrics.items() if key != "rows"}, "baseline_parity": parity, "decisions": selected_decisions}


def _checkpoint(args: argparse.Namespace, lineage_sha: str) -> tuple[Path, dict[str, Any]]:
    path = args.checkpoint_path or args.output_dir / f"r1_{args.mode}_progress.json"
    if path.exists():
        if not args.resume: raise FileExistsError(f"R1 checkpoint exists with --no-resume: {path}")
        state = read_json(path)
        if not isinstance(state, Mapping) or state.get("kind") != f"{EVAL_KIND}_checkpoint" or state.get("lineage_sha256") != lineage_sha or state.get("mode") != args.mode: raise V27ContractError("R1 checkpoint lineage/mode mismatch")
        return path, dict(state)
    return path, {"kind": f"{EVAL_KIND}_checkpoint", "schema_version": SCHEMA_VERSION, "lineage_sha256": lineage_sha, "mode": args.mode, "completed_folds": {}, "complete": False}


def _historical_run(report: Mapping[str, Any], regime: str, inner_seed: int) -> Mapping[str, Any] | None:
    runs = report.get("regimes", {}).get(regime, []) if isinstance(report.get("regimes"), Mapping) else []
    for run in runs:
        if isinstance(run, Mapping) and int(run.get("inner_seed", -1)) == int(inner_seed): return run
    return None


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path, label in ((args.stage1, "Stage-1"), (args.stage2, "Stage-2"), (args.anchor, "V16b anchor"), (args.e1_manifest, "E1 manifest"), (args.e3b_standard_report, "E3B standard report"), (args.e3b_stress_report, "E3B stress report"), (args.gate_source, "gate source"), (args.gate_config, "gate config")): _safe(path, label)
    label_blind = load_candidate_universe(stage1=args.stage1, stage2=args.stage2, anchor=args.anchor, reference=None)
    c35_by_qid = {str(qid): [str(value) for value in label_blind.split.candidates[index][:CANDIDATE_LIMIT]] for index, qid in enumerate(label_blind.split.qids)}
    qwen_scores, qwen_info = _load_qwen_cache(args.qwen_cache_root, expected_qids=label_blind.split.qids, c35_by_qid=c35_by_qid)
    report_path = args.e3b_standard_report if args.mode in {"preflight", "standard"} else args.e3b_stress_report
    e3b_folds, e3b_info = load_e3b_fold_scores(report_path=report_path, expected_mode="standard" if args.mode in {"preflight", "standard"} else "stress", split=label_blind.split)
    lineage = {"kind": EVAL_KIND, "schema_version": SCHEMA_VERSION, "mode": args.mode, "stage1_sha256": sha256_file(args.stage1), "stage2_sha256": sha256_file(args.stage2), "anchor_sha256": sha256_file(args.anchor), "e1_manifest_sha256": sha256_file(args.e1_manifest), "e3b_report_sha256": sha256_file(report_path), "e3b_lineage_contract_sha256": e3b_info["lineage_contract_sha256"], "qwen_cache_lineage_sha256": qwen_info["lineage_sha256"], "gate_source_sha256": sha256_file(args.gate_source), "gate_config_sha256": sha256_file(args.gate_config), "lambdas": list(R1_LAMBDAS), "baseline": {"kind": "e3b_fixed_policy", "memory_weight": 0.0, "threshold": R1_BASELINE_THRESHOLD, "disagreement_weight": R1_DISAGREEMENT_WEIGHT}, "preserve_top4": True, "replace_only_slot5": True, "new_policy_holdout_validated": False, "holdout_opened": False, "public_labels_used": False}
    lineage_sha = json_sha256(lineage); args.output_dir.mkdir(parents=True, exist_ok=True); manifest_path = args.output_dir / f"r1_{args.mode}_label_blind_manifest.json"; write_json(manifest_path, lineage)
    if args.mode == "preflight": return {"kind": f"{EVAL_KIND}_preflight", "stage": "label_blind_preflight", "lineage": lineage, "lineage_sha256": lineage_sha, "e3b_regimes": {key: len(value) for key, value in e3b_folds.items()}, "qwen_cache": qwen_info, "labels_opened_after_manifest": False, "holdout_opened": False, "public_labels_used": False, "gpu_required": False, "next_action": "Run MODE=standard only after checking E3B C/V artifact coverage and lambda=0 parity prerequisites."}
    if args.mode == "stress":
        if args.standard_report is None: raise ValueError("R1 MODE=stress requires --standard-report")
        standard = read_json(args.standard_report)
        if not isinstance(standard, Mapping) or standard.get("promotion_to_stress_allowed") is not True or standard.get("lineage_sha256") != lineage_sha: raise V27ContractError("R1 stress blocked by missing/failing standard report with exact lineage")
    split = load_candidate_universe(stage1=args.stage1, stage2=args.stage2, anchor=args.anchor, reference=args.reference).split
    evidence = load_manifest(args.e1_manifest); groups = build_regime_groups(split, evidence)
    checkpoint_path, checkpoint = _checkpoint(args, lineage_sha); gate = _load_gate(args.gate_source)
    regimes = ("standard",) if args.mode == "standard" else ("law_source_family", "query_template_family", "near_duplicate_family")
    source_report = read_json(report_path); reports: dict[str, list[dict[str, Any]]] = {}
    for regime in regimes:
        regime_folds = list(e3b_folds.get(regime, []))
        seeds = sorted({int(item.inner_seed) for item in regime_folds})
        folds_by_seed = {seed: [item for item in regime_folds if int(item.inner_seed) == int(seed)] for seed in seeds}
        seed_runs: list[dict[str, Any]] = []
        for seed, folds in sorted(folds_by_seed.items()):
            oof: dict[str, dict[str, list[str]]] = {}; fold_reports: list[dict[str, Any]] = []
            for fold in sorted(folds, key=lambda item: item.outer_fold):
                key = f"{regime}|inner={seed}|fold={fold.outer_fold}"; saved = checkpoint["completed_folds"].get(key)
                if saved is not None:
                    report = saved["fold_report"]; predictions = saved["predictions"]
                else:
                    predictions, report = _fold_run(split, fold, qwen_scores); checkpoint["completed_folds"][key] = {"predictions": predictions, "fold_report": report}; write_json(checkpoint_path, checkpoint)
                oof.update(predictions); fold_reports.append(report)
            if set(oof) != set(map(str, split.qids)): raise V27ContractError(f"R1 {regime}/seed={seed} OOF does not cover complement")
            ordered = {str(qid): oof[str(qid)] for qid in split.qids}; metrics = transition_metrics(split, ordered); baseline_gain = float(np.mean([float(report["baseline_validation"]["gain"]) for report in fold_reports]))
            historical = _historical_run(source_report, regime, seed); historical_metrics = None if historical is None else historical.get("metrics")
            seed_runs.append({"inner_seed": seed, "folds": fold_reports, "metrics": {key: value for key, value in metrics.items() if key != "rows"}, "baseline_mean_fold_gain": baseline_gain, "historical_e3b_metrics": historical_metrics, "predictions": ordered})
        reports[regime] = seed_runs
    checkpoint["complete"] = True; write_json(checkpoint_path, checkpoint); gates = {regime: dict(gate(runs)) for regime, runs in reports.items()}; parity_all = all(bool(fold["baseline_parity"]) for runs in reports.values() for run in runs for fold in run["folds"]); not_below_history = all(run.get("historical_e3b_metrics") is None or float(run["metrics"]["recall"]) >= float(run["historical_e3b_metrics"].get("recall", -math.inf)) for runs in reports.values() for run in runs)
    standard_pass = bool(gates.get("standard", {}).get("pass") and parity_all and not_below_history); stress_pass = bool(args.mode == "stress" and all(value.get("pass") is True for value in gates.values()) and parity_all and not_below_history)
    return {"kind": EVAL_KIND, "schema_version": SCHEMA_VERSION, "status": "complement_oof_only", "mode": args.mode, "split": "complement", "lineage": lineage, "lineage_sha256": lineage_sha, "label_blind_manifest": str(manifest_path.resolve()), "labels_opened_after_manifest": True, "holdout_opened": False, "public_labels_used": False, "checkpoint": {"path": str(checkpoint_path.resolve()), "completed_folds": len(checkpoint["completed_folds"]), "resumable": True}, "regimes": reports, "gates": gates, "checks": {"baseline_parity_all": parity_all, "not_below_historical_e3b_oof": not_below_history, "stress_seen_during_design": True}, "promotion_to_stress_allowed": bool(args.mode == "standard" and standard_pass), "promotion_to_public_allowed": stress_pass, "new_policy_holdout_validated": False, "next_action": "Keep E5/E4 artifacts unchanged; only consider public inference after a separately bound R1 release contract and enough deadline budget."}


def main() -> int:
    args = parse_args(); output = _out = args.report_path or args.output_dir / f"r1_{args.mode}_report.json"
    if output.exists() and not args.force: raise FileExistsError(f"Refusing to overwrite R1 report: {output}")
    report = run(args); write_json(output, report); print(json.dumps({"output": str(output.resolve()), "mode": args.mode, "promotion_to_stress_allowed": report.get("promotion_to_stress_allowed", False), "promotion_to_public_allowed": report.get("promotion_to_public_allowed", False), "holdout_opened": False}, ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
