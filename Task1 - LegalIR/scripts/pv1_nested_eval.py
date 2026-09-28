#!/usr/bin/env python3
"""Nested, complement-only evaluation for frozen PV1 pair logits.

The verifier cache is created by :mod:`pv1_score_pairs` without labels.  This
module writes an immutable label-blind manifest before loading complement
qrels, selects one of three fixed abstention policies on C, and evaluates it
once on V.  It never opens holdout or public labels.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e3b_nested_mil_scores as e3b  # noqa: E402
from pv1_contract import KIND, PROMPT_SHA256, load_pair_manifest, load_universe  # noqa: E402
from v27_a_common import V27ContractError, build_regime_groups, group_folds, json_sha256, load_manifest, read_json, sha256_file, write_json  # noqa: E402


EVAL_KIND = f"{KIND}_nested_eval"
SCHEMA_VERSION = 1
INNER_SEEDS = (20_260_910, 20_260_911)
STANDARD = "standard"
STRESS = ("law_source_family", "query_template_family", "near_duplicate_family")
POLICIES = (
    {"mode": "threshold", "tau": 0.0},
    {"mode": "threshold", "tau": math.log(3.0)},
    {"mode": "abstain", "tau": None},
)
EPSILON = 1.0e-12


def recall_of(predicted: Sequence[str], gold: Sequence[str]) -> float:
    """Compatibility helper: |predicted intersect gold| / |gold|."""
    gold_list = [str(value) for value in gold]
    if not gold_list:
        raise ValueError("gold list must be non-empty")
    predicted_set = {str(value) for value in predicted}
    return float(len(predicted_set & set(gold_list)) / len(gold_list))


def delta_recall(anchor: Sequence[str], new_slot5: str, gold: Sequence[str]) -> tuple[float, str]:
    """Compatibility helper: delta recall when slot-5 is replaced, plus rescue/destructive/neutral label."""
    anchor_list = [str(value) for value in anchor]
    if len(anchor_list) != 5:
        raise ValueError("anchor must hold exactly five document IDs")
    candidate = str(new_slot5)
    gold_list = [str(value) for value in gold]
    before = recall_of(anchor_list, gold_list)
    after = recall_of(anchor_list[:4] + [candidate], gold_list)
    delta = float(after - before)
    incumbent_gold = anchor_list[4] in set(gold_list)
    challenger_gold = candidate in set(gold_list)
    if (not incumbent_gold) and challenger_gold:
        action = "rescue"
    elif incumbent_gold and (not challenger_gold):
        action = "destructive"
    else:
        action = "neutral"
    return delta, action


def select_policy(candidates: Mapping[str, Mapping[str, float]]) -> str:
    """Compatibility helper: highest gain wins, then fewest destructive, then fewest changes, else abstain."""
    best_key = "abstain"
    best_tuple: tuple[float, int, int] | None = None
    for key, stats in candidates.items():
        if key == "abstain":
            continue
        gain = float(stats.get("gain", 0.0))
        if gain <= EPSILON:
            continue
        destructive = int(stats.get("destructive", 0))
        changes = int(stats.get("changes", 0))
        current = (gain, -destructive, -changes)
        if best_tuple is None or current > best_tuple:
            best_tuple = current
            best_key = str(key)
    return best_key


def evaluate_predictions(predictions: Mapping[str, Sequence[str]], gold: Mapping[str, Sequence[str]], anchor: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    """Compatibility helper: count rescue/destructive/neutral/abstain and mean delta recall."""
    rescue = destructive = neutral_swaps = abstained = 0
    deltas: list[float] = []
    for qid, anchor_list in anchor.items():
        anchor_ids = [str(value) for value in anchor_list]
        predicted_ids = [str(value) for value in predictions[str(qid)]]
        gold_ids = [str(value) for value in gold[str(qid)]]
        if predicted_ids == anchor_ids:
            abstained += 1
            deltas.append(0.0)
            continue
        _delta, action = delta_recall(anchor_ids, predicted_ids[4], gold_ids)
        deltas.append(float(_delta))
        if action == "rescue":
            rescue += 1
        elif action == "destructive":
            destructive += 1
        else:
            neutral_swaps += 1
    queries = len(anchor)
    gain = float(sum(deltas) / len(deltas)) if deltas else 0.0
    return {
        "queries": int(queries),
        "rescue": int(rescue),
        "destructive": int(destructive),
        "neutral_swaps": int(neutral_swaps),
        "abstained": int(abstained),
        "gain": float(gain),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "standard", "stress"), default="preflight")
    parser.add_argument("--stage1", type=Path, required=True); parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True); parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--e1-manifest", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True); parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--promotion-gate-source", type=Path, required=True); parser.add_argument("--promotion-gate-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--standard-report", type=Path)
    parser.add_argument("--checkpoint-path", type=Path); parser.add_argument("--report-path", type=Path)
    parser.add_argument("--inner-seeds", nargs="+", type=int, default=list(INNER_SEEDS))
    parser.add_argument("--resume", dest="resume", action="store_true"); parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True); parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _safe_file(path: Path, label: str) -> None:
    normalized = str(path).replace("\\", "/").casefold()
    if any(term in normalized for term in ("holdout", "submission", "locked")):
        raise V27ContractError(f"PV1 evaluator cannot use {label} outside complement: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _out(args: argparse.Namespace) -> Path:
    return args.report_path or args.output_dir / f"pv1_{args.mode}_report.json"


def _checkpoint_path(args: argparse.Namespace) -> Path:
    return args.checkpoint_path or args.output_dir / f"pv1_{args.mode}_progress.json"


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return tuple(str(row.get(key, "")) for key in ("qid", "incumbent_id", "challenger_id", "orientation"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise V27ContractError(f"{path}:{number}: invalid JSONL") from exc
            if not isinstance(row, dict):
                raise V27ContractError(f"{path}:{number}: expected a JSON object")
            rows.append(row)
    return rows


def _load_scores(score_dir: Path, pair_rows: Sequence[Mapping[str, Any]], pair_manifest: Mapping[str, Any]) -> tuple[dict[tuple[str, str, str, str], dict[str, Any]], dict[str, Any]]:
    root = score_dir / "pv1_pair_scores" if (score_dir / "pv1_pair_scores").is_dir() else score_dir
    metas = sorted((root / "metadata").glob("*.json")); shards = sorted((root / "shards").glob("*.jsonl"))
    if not metas or not shards:
        raise FileNotFoundError(f"PV1 full score cache is incomplete under {root}")
    expected = {_row_key(row): row for row in pair_rows}; output: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    contracts: list[dict[str, Any]] = []
    shard_ids: set[tuple[int, int]] = set()
    for path in metas:
        meta = read_json(path)
        if not isinstance(meta, dict) or meta.get("kind") != f"{KIND}_score_cache" or meta.get("mode") != "cache":
            raise V27ContractError(f"Invalid/full PV1 cache metadata missing: {path}")
        contract = meta.get("run_contract")
        if not isinstance(contract, dict) or contract.get("pair_lineage_sha256") != pair_manifest.get("lineage_sha256"):
            raise V27ContractError("PV1 score cache differs from pair lineage")
        if contract.get("prompt_sha256") != PROMPT_SHA256 or meta.get("run_hash") != json_sha256(contract):
            raise V27ContractError("PV1 score metadata prompt/run hash mismatch")
        index, count = int(contract.get("shard_index", -1)), int(contract.get("shard_count", -1))
        if count < 1 or not 0 <= index < count or (index, count) in shard_ids:
            raise V27ContractError("PV1 score shard metadata is invalid or duplicate")
        shard_ids.add((index, count)); contracts.append(contract)
    counts = {count for _, count in shard_ids}
    if len(counts) != 1 or len(shard_ids) != next(iter(counts)):
        raise V27ContractError("PV1 cache does not contain every declared score shard")
    meta_by_index = {int(contract["shard_index"]): json_sha256(contract) for contract in contracts}
    for path in shards:
        for row in _read_jsonl(path):
            key = _row_key(row); run_hash = str(row.get("run_hash", ""))
            if key not in expected or key in output:
                raise V27ContractError(f"{path}: score cache key is unknown or duplicate")
            # The QID's declared shard owns both orientations of every pair.
            owner = int(int(__import__("hashlib").sha256(key[0].encode("utf-8")).hexdigest()[:16], 16) % next(iter(counts)))
            if run_hash != meta_by_index.get(owner):
                raise V27ContractError(f"{path}: score row belongs to a different model/prompt contract")
            if row.get("input_sha256") != expected[key].get("input_sha256"):
                raise V27ContractError(f"{path}: score row input text hash drift")
            logits = row.get("logits_abcd")
            if not isinstance(logits, list) or len(logits) != 4 or not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in logits):
                raise V27ContractError(f"{path}: invalid PV1 A/B/C/D logits")
            output[key] = row
    if set(output) != set(expected):
        raise V27ContractError(f"PV1 full score cache coverage mismatch: missing={len(set(expected) - set(output))}")
    return output, {"root": str(root.resolve()), "metadata": [str(path.resolve()) for path in metas], "shards": [str(path.resolve()) for path in shards], "contracts": contracts}


def _logsumexp(values: Sequence[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _utility(forward: Mapping[str, Any], reverse: Mapping[str, Any]) -> float:
    first = {label: float(value) for label, value in zip(("A", "B", "C", "D"), forward["logits_abcd"], strict=True)}
    second = {label: float(value) for label, value in zip(("A", "B", "C", "D"), reverse["logits_abcd"], strict=True)}
    if max(first, key=first.get) != "B" or max(second, key=second.get) != "A":
        return float("-inf")
    forward_margin = first["B"] - _logsumexp([first["A"], first["C"], first["D"]])
    reverse_margin = second["A"] - _logsumexp([second["B"], second["C"], second["D"]])
    return min(forward_margin, reverse_margin)


def _candidate_rows(qid: str, pair_manifest: Mapping[str, Any], scores: Mapping[tuple[str, str, str, str], Mapping[str, Any]]) -> list[dict[str, Any]]:
    check = pair_manifest["shortlist_checks"].get(qid)
    if not isinstance(check, dict):
        raise V27ContractError(f"PV1 pair manifest lacks QID {qid}")
    incumbent = str(check["incumbent_id"]); candidates: list[dict[str, Any]] = []
    for challenger in check.get("challengers", []):
        challenger = str(challenger)
        forward = scores.get((qid, incumbent, challenger, "incumbent_first")); reverse = scores.get((qid, incumbent, challenger, "challenger_first"))
        if forward is None or reverse is None:
            raise V27ContractError(f"{qid}/{challenger}: missing an orientation in PV1 cache")
        utility = _utility(forward, reverse)
        candidates.append({"doc_id": challenger, "f_rank": int(forward["f_rank"]), "f_score": float(forward["f_score"]), "utility": utility,
                           "forward_argmax": ("A", "B", "C", "D")[max(range(4), key=lambda i: float(forward["logits_abcd"][i]))],
                           "reverse_argmax": ("A", "B", "C", "D")[max(range(4), key=lambda i: float(reverse["logits_abcd"][i]))]})
    return candidates


def _predict(split: Any, *, indices: Sequence[int], pair_manifest: Mapping[str, Any], scores: Mapping[tuple[str, str, str, str], Mapping[str, Any]], policy: Mapping[str, Any], top_k: int = 3) -> tuple[dict[str, dict[str, list[str]]], dict[str, Any]]:
    predictions: dict[str, dict[str, list[str]]] = {}; decisions: dict[str, Any] = {}
    for index in indices:
        qid = str(split.qids[index]); anchor = [str(value) for value in split.anchor[index]]
        candidates = _candidate_rows(qid, pair_manifest, scores)[:top_k]
        ranked = sorted(candidates, key=lambda item: (-float(item["utility"]), int(item["f_rank"]), str(item["doc_id"])))
        selected = ranked[0] if ranked else None; changed = False
        if policy["mode"] == "threshold" and selected is not None and float(selected["utility"]) > float(policy["tau"]):
            answer = anchor[:4] + [str(selected["doc_id"])]; changed = True
        else:
            answer = anchor.copy()
        if len(answer) != 5 or len(set(answer)) != 5 or answer[:4] != anchor[:4]:
            raise V27ContractError(f"{qid}: PV1 violated frozen top-4/unique top-5 contract")
        predictions[qid] = {"answer": answer}
        decisions[qid] = {"changed": changed, "policy": dict(policy), "incumbent_id": anchor[4], "selected": selected, "candidates": candidates}
    return predictions, decisions


def _metrics(split: Any, predictions: Mapping[str, Mapping[str, Sequence[str]]]) -> dict[str, Any]:
    if split.references is None:
        raise V27ContractError("PV1 metrics require complement qrels")
    before_values: list[float] = []; after_values: list[float] = []; rows: list[dict[str, Any]] = []
    rescues = harms = changed = neutral = improved = harmed = 0
    for index, qid_raw in enumerate(split.qids):
        qid = str(qid_raw); anchor = [str(value) for value in split.anchor[index]]; answer = [str(value) for value in predictions[qid]["answer"]]
        gold = {str(value) for value in split.references[index]}; before = len(set(anchor) & gold) / len(gold); after = len(set(answer) & gold) / len(gold)
        delta = after - before; did_change = answer != anchor
        incumbent_gold, challenger_gold = anchor[4] in gold, answer[4] in gold
        if did_change:
            changed += 1
            if not incumbent_gold and challenger_gold: rescues += 1
            elif incumbent_gold and not challenger_gold: harms += 1
            else: neutral += 1
        improved += int(delta > EPSILON); harmed += int(delta < -EPSILON)
        before_values.append(before); after_values.append(after)
        rows.append({"qid": qid, "before": before, "after": after, "delta": delta, "changed": did_change,
                     "incumbent_gold": incumbent_gold, "challenger_gold": challenger_gold})
    gain = float(np.mean(after_values) - np.mean(before_values))
    denominator = rescues + harms
    return {"queries": len(rows), "recall": float(np.mean(after_values)), "anchor_recall": float(np.mean(before_values)), "gain": gain,
            "net_query_equivalents": float(sum(row["delta"] for row in rows)), "changed_queries": changed, "improved_queries": improved,
            "harmed_queries": harmed, "neutral_changed_queries": neutral, "successful_rescues": rescues, "destructive_changes": harms,
            "outcome_changing_precision": None if denominator == 0 else float(rescues / denominator),
            "rows": rows}


def _choose_policy(split: Any, calibration: Sequence[int], pair_manifest: Mapping[str, Any], scores: Mapping[tuple[str, str, str, str], Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trials: list[dict[str, Any]] = []
    for policy in POLICIES:
        predictions, _ = _predict(split, indices=calibration, pair_manifest=pair_manifest, scores=scores, policy=policy)
        metrics = _metrics(_subset(split, calibration), predictions)
        trials.append({"policy": dict(policy), "metrics": metrics})
    eligible = [trial for trial in trials if float(trial["metrics"]["gain"]) > EPSILON]
    if not eligible:
        return {"mode": "abstain", "tau": None}, trials
    # Higher tau is safer when all outcome metrics tie; abstain is not eligible
    # because it cannot have a positive gain.
    selected = max(eligible, key=lambda trial: (float(trial["metrics"]["gain"]), -int(trial["metrics"]["destructive_changes"]), -int(trial["metrics"]["changed_queries"]), float(trial["policy"]["tau"])))
    return dict(selected["policy"]), trials


def _subset(split: Any, indices: Sequence[int]) -> Any:
    # E3B's partition helper returns indices into the original split.  The
    # metric only needs these fields; keep it explicit instead of mutating it.
    class View:
        pass
    view = View(); view.qids = [split.qids[index] for index in indices]; view.questions = [split.questions[index] for index in indices]
    view.anchor = [split.anchor[index] for index in indices]; view.references = [split.references[index] for index in indices]
    return view


def _load_gate(path: Path) -> Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]]:
    _safe_file(path, "promotion-gate source")
    spec = importlib.util.spec_from_file_location("pv1_bound_gate", path)
    if spec is None or spec.loader is None:
        raise V27ContractError("Cannot load PV1 bound promotion-gate source")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    gate = getattr(module, "_gate", None)
    if not callable(gate):
        raise V27ContractError("Bound promotion-gate source does not expose _gate(runs)")
    return gate


def _lineage(args: argparse.Namespace, pair_manifest: Mapping[str, Any], score_info: Mapping[str, Any]) -> dict[str, Any]:
    for path, label in ((args.e1_manifest, "E1 manifest"), (args.promotion_gate_source, "promotion-gate source"), (args.promotion_gate_config, "promotion-gate config")):
        _safe_file(path, label)
    return {"kind": EVAL_KIND, "schema_version": SCHEMA_VERSION, "pair_lineage_sha256": pair_manifest["lineage_sha256"],
            "pair_manifest_sha256": sha256_file(args.pair_manifest), "pairs_sha256": sha256_file(args.pairs), "score_cache": dict(score_info),
            "e1_manifest_sha256": sha256_file(args.e1_manifest), "promotion_gate_source": str(args.promotion_gate_source.resolve()),
            "promotion_gate_source_sha256": sha256_file(args.promotion_gate_source), "promotion_gate_config": str(args.promotion_gate_config.resolve()),
            "promotion_gate_config_sha256": sha256_file(args.promotion_gate_config), "inner_seeds": [int(seed) for seed in args.inner_seeds],
            "policies": list(POLICIES), "strict_comparison": ">", "stress_seen_during_design": True,
            "holdout_opened": False, "public_labels_used": False}


def _checkpoint(args: argparse.Namespace, lineage_sha: str) -> tuple[Path, dict[str, Any]]:
    path = _checkpoint_path(args)
    if path.exists():
        if not args.resume: raise FileExistsError(f"PV1 checkpoint exists but --no-resume selected: {path}")
        state = read_json(path)
        if not isinstance(state, dict) or state.get("kind") != f"{EVAL_KIND}_checkpoint" or state.get("lineage_sha256") != lineage_sha or state.get("mode") != args.mode:
            raise V27ContractError("PV1 checkpoint differs from pair/gate/cache lineage")
        return path, state
    return path, {"kind": f"{EVAL_KIND}_checkpoint", "schema_version": SCHEMA_VERSION, "lineage_sha256": lineage_sha, "mode": args.mode, "completed_folds": {}, "complete": False}


def _validate_standard(path: Path, lineage_sha: str) -> None:
    report = read_json(path)
    if not isinstance(report, dict) or report.get("kind") != EVAL_KIND or report.get("mode") != "standard" or report.get("lineage_sha256") != lineage_sha or report.get("promotion_to_stress_allowed") is not True:
        raise V27ContractError("PV1 stress is blocked: standard report/gate/lineage does not pass")


def _run_fold(*, split: Any, groups: np.ndarray, outer_train: np.ndarray, validation: np.ndarray, inner_seed: int, outer_fold: int, pair_manifest: Mapping[str, Any], scores: Mapping[tuple[str, str, str, str], Mapping[str, Any]]) -> tuple[dict[str, dict[str, list[str]]], dict[str, Any]]:
    partitions, partition_metadata = e3b.make_partitions(split=split, groups=groups, outer_train=outer_train, validation=validation, inner_seed=inner_seed, outer_fold=outer_fold)
    policy, trace = _choose_policy(split, partitions["calibration"].tolist(), pair_manifest, scores)
    validation_predictions, decisions = _predict(split, indices=partitions["validation"].tolist(), pair_manifest=pair_manifest, scores=scores, policy=policy)
    validation_view = _subset(split, partitions["validation"].tolist()); metrics = _metrics(validation_view, validation_predictions)
    top1_predictions, _ = _predict(split, indices=partitions["validation"].tolist(), pair_manifest=pair_manifest, scores=scores, policy=policy, top_k=1)
    return validation_predictions, {"outer_fold": int(outer_fold), "partitions": partition_metadata, "selected_policy": policy,
                                    "calibration_policy_trace": trace, "validation": metrics,
                                    "counterfactual_f_top1": _metrics(validation_view, top1_predictions), "decisions": decisions}


def _run_seed(*, split: Any, groups: np.ndarray, regime: str, inner_seed: int, pair_manifest: Mapping[str, Any], scores: Mapping[tuple[str, str, str, str], Mapping[str, Any]], checkpoint: dict[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    oof: dict[str, dict[str, list[str]]] = {}; folds: list[dict[str, Any]] = []
    for outer_fold, (outer_train, validation) in enumerate(group_folds(groups)):
        key = f"{regime}|inner={inner_seed}|fold={outer_fold}"; validation_qids = [str(split.qids[index]) for index in validation]
        saved = checkpoint["completed_folds"].get(key)
        if saved is not None:
            if saved.get("validation_qids") != validation_qids or not isinstance(saved.get("predictions"), dict):
                raise V27ContractError("PV1 saved fold mismatches current partition")
            oof.update(saved["predictions"]); folds.append(saved["fold_report"]); continue
        predictions, report = _run_fold(split=split, groups=groups, outer_train=outer_train, validation=validation, inner_seed=inner_seed, outer_fold=outer_fold, pair_manifest=pair_manifest, scores=scores)
        oof.update(predictions); folds.append(report); checkpoint["completed_folds"][key] = {"validation_qids": validation_qids, "predictions": predictions, "fold_report": report}; write_json(checkpoint_path, checkpoint)
        print(json.dumps({"event": "pv1_fold_complete", "run": key, "selected_policy": report["selected_policy"], "validation": {key: value for key, value in report["validation"].items() if key != "rows"}}, ensure_ascii=False), flush=True)
    if set(oof) != set(map(str, split.qids)):
        raise V27ContractError(f"PV1 {regime}/seed={inner_seed} does not cover complement exactly")
    ordered = {str(qid): oof[str(qid)] for qid in split.qids}
    return {"inner_seed": int(inner_seed), "folds": folds, "metrics": _metrics(split, ordered), "predictions": ordered}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if len(set(args.inner_seeds)) != 2:
        raise V27ContractError("PV1 requires exactly the two locked inner seeds")
    pair_rows, pair_manifest = load_pair_manifest(args.pair_manifest, args.pairs)
    scores, score_info = _load_scores(args.score_dir, pair_rows, pair_manifest)
    lineage = _lineage(args, pair_manifest, score_info); lineage_sha = json_sha256(lineage)
    args.output_dir.mkdir(parents=True, exist_ok=True); manifest_path = args.output_dir / f"pv1_{args.mode}_label_blind_manifest.json"; write_json(manifest_path, lineage)
    if args.mode == "preflight":
        return {"kind": EVAL_KIND, "mode": "preflight", "lineage": lineage, "lineage_sha256": lineage_sha, "label_blind_manifest": str(manifest_path.resolve()), "labels_opened_after_manifest": False, "holdout_opened": False, "public_labels_used": False}
    if args.mode == "stress":
        if args.standard_report is None: raise ValueError("PV1 MODE=stress requires --standard-report")
        _validate_standard(args.standard_report, lineage_sha)
    gate = _load_gate(args.promotion_gate_source)
    split = load_universe(stage1=args.stage1, stage2=args.stage2, anchor=args.anchor, reference=args.reference).split
    evidence = load_manifest(args.e1_manifest); groups_by_regime = build_regime_groups(split, evidence)
    checkpoint_path, checkpoint = _checkpoint(args, lineage_sha)
    regimes = (STANDARD,) if args.mode == "standard" else STRESS
    runs = {regime: [_run_seed(split=split, groups=groups_by_regime[regime], regime=regime, inner_seed=int(seed), pair_manifest=pair_manifest, scores=scores, checkpoint=checkpoint, checkpoint_path=checkpoint_path) for seed in args.inner_seeds] for regime in regimes}
    checkpoint["complete"] = True; write_json(checkpoint_path, checkpoint)
    gates = {regime: dict(gate(value)) for regime, value in runs.items()}
    standard_pass = bool(gates.get(STANDARD, {}).get("pass"))
    stress_pass = bool(args.mode == "stress" and all(value.get("pass") is True for value in gates.values()))
    return {"kind": EVAL_KIND, "schema_version": SCHEMA_VERSION, "status": "nested_complement_oof_only", "mode": args.mode, "split": "complement",
            "lineage": lineage, "lineage_sha256": lineage_sha, "label_blind_manifest": str(manifest_path.resolve()), "labels_opened_after_manifest": True,
            "holdout_opened": False, "public_labels_used": False, "checkpoint": {"path": str(checkpoint_path.resolve()), "completed_folds": len(checkpoint["completed_folds"]), "resumable": True},
            "regimes": runs, "gates": gates, "promotion_to_stress_allowed": bool(args.mode == "standard" and standard_pass),
            "promotion_to_public_allowed": stress_pass, "promotion_to_e4_allowed": False,
            "next_action": "Finalize/public inference is blocked unless every bound strict stress gate passes; holdout stays sealed."}


def main() -> int:
    args = parse_args(); output = _out(args)
    if output.exists() and not args.force: raise FileExistsError(f"PV1 output exists: {output}")
    report = run(args); write_json(output, report)
    print(json.dumps({"output": str(output.resolve()), "mode": args.mode, "promotion_to_stress_allowed": report.get("promotion_to_stress_allowed", False), "promotion_to_public_allowed": report.get("promotion_to_public_allowed", False), "holdout_opened": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
