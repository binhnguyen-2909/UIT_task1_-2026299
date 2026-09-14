#!/usr/bin/env python3
"""Nested, complement-only evaluation of Qwen c35/c67/c200 score caches.

The expensive Qwen3 scoring happens beforehand in ``qwen_c200_cache.py``.
This program never loads the Qwen checkpoint: it evaluates the frozen cache
with two deliberately small selectors, each trained/calibrated inside the
outer-fold boundary.  It is research-only and cannot open holdout or create a
submission.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e3b_nested_mil_scores as e3b  # noqa: E402
from gpu_transition_metrics import transition_metrics_device  # noqa: E402
from qwen_c200_common import (  # noqa: E402
    LONG_MAX_TOKENS, POOL_SIZES, QWEN_BRANCH_KIND, SHORT_MAX_TOKENS,
    cap_cpu_threads, checkpoint_fingerprint, load_candidate_universe,
    oracle_slot5, pool_for, read_score_cache, require_complement_file, score_order,
    validate_cache_coverage,
)
from v27_a_common import (  # noqa: E402
    EDITABLE_SLOT, V27ContractError, build_regime_groups, group_folds, json_sha256,
    load_manifest, read_json, sha256_file, write_json,
)


KIND = "legalir_qwen_c200_nested_selector_research"
SCHEMA_VERSION = 1
INNER_SEEDS = (20260910, 20260911)
STANDARD_REGIME = "standard"
STRESS_REGIMES = ("law_source_family", "query_template_family", "near_duplicate_family")
VARIANTS = ("A_c35_short", "B_c35_long", "C_c67_long", "D_c200_long")
# New c67-shortlist8 report path (E/F). The legacy A/B/C/D VARIANTS tuple is
# frozen so existing standard/stress runs keep byte-identical behavior.
EVAL_VARIANTS_EF = ("E_c67_short", "F_shortlist8_long")
DEFAULT_EVAL_VARIANTS = ("A_c35_short", "E_c67_short", "F_shortlist8_long")
S1_THRESHOLDS = (0.0, 0.25, 0.50, 1.00)
S2_TOP_K = 3
S2_FEATURE_SCHEMA = (
    "qwen_candidate_minus_incumbent_logit",
    "qwen_candidate_within_pool_zscore",
    "stage1_candidate_rank_fraction",
    "stage1_incumbent_rank_fraction",
    "qwen_challenger_position_fraction",
)
S2_POLICY = {
    "top_k": S2_TOP_K,
    "frozen_top_k": EDITABLE_SLOT - 1,
    "editable_slot": EDITABLE_SLOT,
    "decision_floor": 0.0,
    "candidate_tie_break": "higher_predicted_delta_then_lower_qwen_challenger_position_then_doc_id",
    "incumbent_default": "keep_v16b_slot5",
}
S2_CHECKPOINT_KIND = "legalir_qwen_c200_s2_fit_checkpoint"
S2_CHECKPOINT_SCHEMA_VERSION = 1
S2_VERIFICATION_ATOL = 1.0e-6
S2_VERIFICATION_RTOL = 1.0e-5
FORBIDDEN_TERMS = ("holdout", "public", "submission", "locked")
EPSILON = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "standard", "stress"), default="preflight")
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--e1-manifest", type=Path, required=True)
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--short-cache-dir", type=Path, required=True)
    parser.add_argument("--long-cache-dir", type=Path, required=True)
    parser.add_argument("--e3b-standard-report", type=Path, required=True)
    parser.add_argument("--e3b-stress-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--standard-report", type=Path, help="required for MODE=stress")
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--inner-seeds", nargs="+", type=int, default=list(INNER_SEEDS))
    parser.add_argument("--s2-hidden", type=int, default=16)
    parser.add_argument("--s2-epochs", type=int, default=120)
    parser.add_argument("--s2-patience", type=int, default=12)
    parser.add_argument("--s2-learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _output_path(args: argparse.Namespace) -> Path:
    return args.report_path or args.output_dir / f"qwen_c200_{args.mode}_report.json"


def _checkpoint_path(args: argparse.Namespace) -> Path:
    return args.checkpoint_path or args.output_dir / f"qwen_c200_{args.mode}_progress.json"


def _safe_path(path: Path, label: str) -> None:
    require_complement_file(path, label)


def _cache_files(root: Path) -> tuple[list[Path], list[Path]]:
    score_paths = sorted((root / "shards").glob("*.jsonl"))
    meta_paths = sorted((root / "metadata").glob("*.json"))
    if not score_paths or not meta_paths:
        raise FileNotFoundError(f"Qwen cache directory is incomplete: {root}")
    return score_paths, meta_paths


def _load_cache_variant(
    *, universe: Any, cache_dir: Path, expected_lineage: Mapping[str, Any], variant: str,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    score_paths, meta_paths = _cache_files(cache_dir)
    expected_sha = json_sha256(expected_lineage)
    metas: list[dict[str, Any]] = []
    for path in meta_paths:
        meta = read_json(path)
        if not isinstance(meta, dict) or meta.get("kind") != "legalir_qwen_c200_score_cache":
            raise V27ContractError(f"Invalid Qwen cache metadata: {path}")
        if meta.get("lineage_sha256") != expected_sha or meta.get("lineage") != dict(expected_lineage):
            raise V27ContractError(f"Qwen cache metadata does not match frozen {variant} lineage: {path}")
        metas.append(meta)
    shard_ids = {(int(meta.get("shard_index", -1)), int(meta.get("shard_count", -1))) for meta in metas}
    if len(shard_ids) != len(metas) or any(count <= 0 or not 0 <= index < count for index, count in shard_ids):
        raise V27ContractError(f"Qwen {variant} cache shard metadata is invalid or duplicated")
    if len({count for _, count in shard_ids}) != 1 or len(shard_ids) != next(iter(shard_ids))[1]:
        raise V27ContractError(f"Qwen {variant} cache does not contain every shard exactly once")
    scores = read_score_cache(score_paths, expected_lineage_sha256=expected_sha)
    validate_cache_coverage(universe, scores, variant=variant)
    return scores, {
        "cache_dir": str(cache_dir.resolve()), "score_shards": [str(path.resolve()) for path in score_paths],
        "metadata_shards": [str(path.resolve()) for path in meta_paths], "lineage_sha256": expected_sha,
        "qids": len(scores), "pairs": sum(len(value) for value in scores.values()),
    }


def _lineage(args: argparse.Namespace, fingerprint: Mapping[str, str], cache_reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "kind": KIND, "schema_version": SCHEMA_VERSION, "split": "complement",
        "stage1_sha256": sha256_file(args.stage1), "stage2_sha256": sha256_file(args.stage2),
        "anchor_sha256": sha256_file(args.anchor), "reference_sha256": sha256_file(args.reference),
        "e1_manifest_sha256": sha256_file(args.e1_manifest),
        "qwen_model_fingerprint": dict(fingerprint),
        "short_cache_max_length": SHORT_MAX_TOKENS, "long_cache_max_length": LONG_MAX_TOKENS,
        "qwen_cache_lineages": {name: str(report["lineage_sha256"]) for name, report in sorted(cache_reports.items())},
        "variants": list(VARIANTS), "selector_contract": {
            "s1": {"kind": "qwen_incumbent_gap", "thresholds": list(S1_THRESHOLDS)},
            "s2": {"kind": "tiny_mlp_top3_delta", "top_k": S2_TOP_K, "hidden": int(args.s2_hidden),
                   "epochs": int(args.s2_epochs), "patience": int(args.s2_patience),
                   "learning_rate": float(args.s2_learning_rate), "feature_schema": list(S2_FEATURE_SCHEMA),
                   "policy": dict(S2_POLICY), "checkpoint_contract": S2_CHECKPOINT_KIND,
                   "decision_floor": 0.0},
            "frozen_top_k": EDITABLE_SLOT - 1, "default": "v16b_anchor_abstain",
            "outer_folds": 5, "inner_seeds": [int(value) for value in args.inner_seeds],
        },
        "scripts": {
            "evaluator": sha256_file(Path(__file__)),
            "common": sha256_file(Path(__file__).with_name("qwen_c200_common.py")),
            "cache_scorer": sha256_file(Path(__file__).with_name("qwen_c200_cache.py")),
            "gpu_metrics": sha256_file(Path(__file__).with_name("gpu_transition_metrics.py")),
            "e3b_partitions": sha256_file(Path(e3b.__file__)),
        },
        "e3b_prerequisites": {
            "standard_report_sha256": sha256_file(args.e3b_standard_report),
            "stress_report_sha256": sha256_file(args.e3b_stress_report),
        },
        "holdout_opened": False, "public_labels_used": False,
    }


def _preflight(args: argparse.Namespace) -> tuple[dict[str, Any], Any, dict[str, dict[str, dict[str, Any]]]]:
    cap_cpu_threads(args.cpu_threads)
    if len(set(args.inner_seeds)) != 2:
        raise ValueError("Qwen c200 nested evaluation requires exactly two inner seeds")
    if args.s2_hidden <= 0 or args.s2_epochs <= 0 or args.s2_patience <= 0 or args.s2_learning_rate <= 0.0:
        raise ValueError("Qwen c200 selector hyperparameters must be positive")
    for path, label in ((args.stage1, "Stage-1"), (args.stage2, "Stage-2"), (args.anchor, "V16b anchor"),
                        (args.reference, "complement reference"), (args.e1_manifest, "E1 evidence manifest"),
                        (args.e3b_standard_report, "E3B standard report"), (args.e3b_stress_report, "E3B stress report")):
        _safe_path(path, label)
    universe = load_candidate_universe(stage1=args.stage1, stage2=args.stage2, anchor=args.anchor, reference=None)
    fingerprint = checkpoint_fingerprint(args.qwen_model)
    expected = {
        "A_c35_short": {"kind": QWEN_BRANCH_KIND, "schema_version": 1, "split": "complement", "stage1_sha256": universe.stage1_sha256, "stage2_sha256": universe.stage2_sha256, "anchor_sha256": universe.anchor_sha256, "qwen_model": "Qwen/Qwen3-Reranker-0.6B", "qwen_checkpoint_fingerprint": dict(fingerprint), "variant": "A_c35_short", "pool_size": POOL_SIZES["A_c35_short"], "max_tokens": SHORT_MAX_TOKENS, "candidate_prefix_is_frozen_c35": True, "holdout_opened": False, "public_labels_used": False},
        "D_c200_long": {"kind": QWEN_BRANCH_KIND, "schema_version": 1, "split": "complement", "stage1_sha256": universe.stage1_sha256, "stage2_sha256": universe.stage2_sha256, "anchor_sha256": universe.anchor_sha256, "qwen_model": "Qwen/Qwen3-Reranker-0.6B", "qwen_checkpoint_fingerprint": dict(fingerprint), "variant": "D_c200_long", "pool_size": POOL_SIZES["D_c200_long"], "max_tokens": LONG_MAX_TOKENS, "candidate_prefix_is_frozen_c35": True, "holdout_opened": False, "public_labels_used": False},
    }
    # The contexts ZIP hash is intentionally copied from cache metadata below;
    # evaluator never needs to read raw documents and cannot accidentally alter
    # their source.  Other lineage fields are compared exactly.
    cache_scores: dict[str, dict[str, dict[str, Any]]] = {}
    cache_reports: dict[str, Any] = {}
    for primary, cache_dir in (("A_c35_short", args.short_cache_dir), ("D_c200_long", args.long_cache_dir)):
        _, metas = _cache_files(cache_dir)
        first = read_json(metas[0])
        if not isinstance(first, dict) or not isinstance(first.get("lineage"), dict):
            raise V27ContractError(f"Invalid Qwen cache metadata: {metas[0]}")
        actual = dict(first["lineage"])
        for key, value in expected[primary].items():
            if actual.get(key) != value:
                raise V27ContractError(f"Qwen {primary} cache preflight drift at {key}")
        expected[primary] = actual
        scores, report = _load_cache_variant(universe=universe, cache_dir=cache_dir, expected_lineage=actual, variant=primary)
        cache_scores[primary] = scores; cache_reports[primary] = report
    lineage = _lineage(args, fingerprint, cache_reports)
    report = {
        "kind": KIND, "schema_version": SCHEMA_VERSION, "stage": "label_blind_preflight", "split": "complement",
        "lineage": lineage, "lineage_sha256": json_sha256(lineage), "cache_validation": cache_reports,
        "queries": len(universe.split.qids), "pairs": {"A_c35_short": cache_reports["A_c35_short"]["pairs"], "D_c200_long": cache_reports["D_c200_long"]["pairs"]},
        "planned_configurations": [f"{variant}/{selector}" for variant in VARIANTS for selector in ("S1", "S2")],
        "model_budget": {"v16b_parameters_billion": 1.09, "qwen_parameters_billion": 0.60, "selector_parameters_billion_max": 0.001, "total_parameters_billion": 1.691, "under_4b": True},
        "labels_opened_after_manifest": False, "holdout_opened": False, "public_labels_used": False,
    }
    return report, universe, cache_scores


def _subset(split: Any, indices: Sequence[int]) -> Any:
    return e3b._subset(split, indices)


def _pool_scores(cache_scores: Mapping[str, Mapping[str, Mapping[str, Any]]], variant: str) -> Mapping[str, Mapping[str, Any]]:
    return cache_scores["A_c35_short"] if variant == "A_c35_short" else cache_scores["D_c200_long"]


def _ranked_challengers(split: Any, index: int, scores: Mapping[str, Mapping[str, Any]], variant: str) -> tuple[list[str], str, list[str]]:
    qid = str(split.qids[index]); anchor = [str(value) for value in split.anchor[index]]
    if len(anchor) != 5 or len(set(anchor)) != 5:
        raise V27ContractError(f"{qid}: V16b anchor must contain five unique IDs")
    pool = list(pool_for(split._qwen_universe, qid, variant))  # type: ignore[attr-defined]
    if not set(anchor).issubset(set(pool)):
        raise V27ContractError(f"{qid}: V16b incumbent escapes Qwen {variant} pool")
    ranked = score_order(scores, qid, pool)
    incumbent = anchor[EDITABLE_SLOT - 1]
    challengers = [doc_id for doc_id in ranked if doc_id not in set(anchor[: EDITABLE_SLOT - 1]) and doc_id != incumbent]
    return challengers, incumbent, anchor


def _answer(anchor: Sequence[str], replacement: str | None) -> list[str]:
    result = [str(value) for value in anchor]
    if replacement is not None:
        result[EDITABLE_SLOT - 1] = str(replacement)
    if len(result) != 5 or len(set(result)) != 5 or result[: EDITABLE_SLOT - 1] != list(anchor[: EDITABLE_SLOT - 1]):
        raise V27ContractError("Qwen selector violated frozen-top4/five-unique contract")
    return result


def _s1_predictions(split: Any, indices: Sequence[int], scores: Mapping[str, Mapping[str, Any]], variant: str, threshold: float | None) -> tuple[dict[str, dict[str, list[str]]], dict[str, Any]]:
    predictions: dict[str, dict[str, list[str]]] = {}; decisions: dict[str, Any] = {}
    for index in indices:
        qid = str(split.qids[index]); challengers, incumbent, anchor = _ranked_challengers(split, int(index), scores, variant)
        winner = challengers[0] if challengers else None
        gap = None if winner is None else float(scores[qid][winner]["yes_no_logit"]) - float(scores[qid][incumbent]["yes_no_logit"])
        replace = winner if threshold is not None and gap is not None and gap > float(threshold) + EPSILON else None
        predictions[qid] = {"answer": _answer(anchor, replace)}
        decisions[qid] = {"selector": "S1", "incumbent": incumbent, "candidate": winner, "score_gap": gap, "threshold": threshold, "changed": replace is not None}
    return predictions, decisions


def _s1_calibrate(split: Any, scores: Mapping[str, Mapping[str, Any]], variant: str, *, device: str) -> tuple[float | None, dict[str, Any], list[dict[str, Any]]]:
    trials: list[tuple[float | None, dict[str, Any]]] = []
    for threshold in (None, *S1_THRESHOLDS):
        predictions, _ = _s1_predictions(split, range(len(split.qids)), scores, variant, threshold)
        trials.append((threshold, _compact(transition_metrics_device(split, predictions, device=device))))
    eligible = [(threshold, metrics) for threshold, metrics in trials[1:] if float(metrics["gain"]) > EPSILON and int(metrics["successful_rescues"]) > int(metrics["destructive_changes"])]
    selected, selected_metrics = (None, trials[0][1]) if not eligible else max(eligible, key=lambda item: (float(item[1]["gain"]), -int(item[1]["destructive_changes"]), int(item[1]["successful_rescues"]), float(item[0] or 0.0)))
    return selected, selected_metrics, [{"threshold": threshold, "metrics": metrics, "eligible": threshold is not None and float(metrics["gain"]) > EPSILON and int(metrics["successful_rescues"]) > int(metrics["destructive_changes"])} for threshold, metrics in trials]


def _s2_features(split: Any, index: int, scores: Mapping[str, Mapping[str, Any]], variant: str, *, with_target: bool) -> list[dict[str, Any]]:
    qid = str(split.qids[index]); challengers, incumbent, anchor = _ranked_challengers(split, index, scores, variant)
    pool = list(pool_for(split._qwen_universe, qid, variant))  # type: ignore[attr-defined]
    values = np.asarray([float(scores[qid][doc_id]["yes_no_logit"]) for doc_id in pool], dtype=np.float32)
    scale = max(float(values.std()), 1.0e-6); incumbent_value = float(scores[qid][incumbent]["yes_no_logit"])
    incumbent_rank = pool.index(incumbent) + 1
    gold = set(map(str, split.references[index])) if with_target and split.references is not None else set()
    before = len(set(anchor) & gold) / len(gold) if gold else 0.0
    rows: list[dict[str, Any]] = []
    for position, doc_id in enumerate(challengers[:S2_TOP_K], 1):
        candidate_value = float(scores[qid][doc_id]["yes_no_logit"]); rank = pool.index(doc_id) + 1
        target = None
        if gold:
            target = len(set(_answer(anchor, doc_id)) & gold) / len(gold) - before
        rows.append({"qid": qid, "doc_id": doc_id, "candidate_position": position, "features": [candidate_value - incumbent_value, (candidate_value - float(values.mean())) / scale, float(rank) / len(pool), float(incumbent_rank) / len(pool), float(position) / S2_TOP_K], "target": target, "anchor": anchor, "incumbent": incumbent})
    return rows


def _new_s2_model(args: argparse.Namespace) -> Any:
    import torch

    return torch.nn.Sequential(
        torch.nn.Linear(len(S2_FEATURE_SCHEMA), int(args.s2_hidden)),
        torch.nn.ReLU(),
        torch.nn.Linear(int(args.s2_hidden), 1),
    ).to(args.device)


def _cpu_state_dict(model: Any) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _capture_rng_state() -> dict[str, Any]:
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all([value.to("cpu") for value in state["torch_cuda"]])


def _torch_load(path: Path) -> dict[str, Any]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before the weights_only parameter.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise V27ContractError(f"S2 checkpoint is not an object: {path}")
    return payload


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> str:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)
    return sha256_file(path)


def _atomic_backup(source: Path, backup: Path) -> str:
    backup.parent.mkdir(parents=True, exist_ok=True)
    temporary = backup.with_name(f".{backup.name}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, backup)
    source_hash, backup_hash = sha256_file(source), sha256_file(backup)
    if source_hash != backup_hash:
        raise V27ContractError("S2 selected-checkpoint backup hash differs from primary")
    return backup_hash


def _s2_fit_dir(args: argparse.Namespace, *, regime: str, inner_seed: int, outer_fold: int, variant: str) -> Path:
    return args.output_dir / "qwen_c200_s2_fits" / regime / f"inner_{inner_seed}" / f"fold_{outer_fold + 1}" / variant


def _s2_fit_contract(
    *, split: Any, a_fit: Sequence[int], a_stop: Sequence[int], variant: str, args: argparse.Namespace,
    seed: int, regime: str, inner_seed: int, outer_fold: int, lineage_sha256: str,
    partition_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "kind": S2_CHECKPOINT_KIND, "schema_version": S2_CHECKPOINT_SCHEMA_VERSION,
        "lineage_sha256": lineage_sha256, "regime": regime, "inner_seed": int(inner_seed),
        "outer_fold": int(outer_fold), "variant": variant, "seed": int(seed),
        "feature_schema": list(S2_FEATURE_SCHEMA), "architecture": {"input_features": len(S2_FEATURE_SCHEMA), "hidden": int(args.s2_hidden), "activation": "ReLU", "output": "delta_recall_at_5"},
        "hyperparameters": {"epochs": int(args.s2_epochs), "patience": int(args.s2_patience), "learning_rate": float(args.s2_learning_rate), "optimizer": "AdamW", "weight_decay": 1.0e-4, "loss": "SmoothL1", "scheduler": None},
        "policy": dict(S2_POLICY),
        "partitions": {"a_fit_qids": [str(split.qids[index]) for index in a_fit], "a_stop_qids": [str(split.qids[index]) for index in a_stop], "metadata": dict(partition_metadata)},
    }


def _checkpoint_payload(
    *, fit_contract: Mapping[str, Any], checkpoint_role: str, model: Any, optimiser: Any,
    mean: Any, std: Any, epoch: int, step: int, best_stop_loss: float, best_epoch: int, stale_epochs: int,
    verification_reference: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "kind": S2_CHECKPOINT_KIND, "schema_version": S2_CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_role": checkpoint_role, "fit_contract": dict(fit_contract),
        "fit_contract_sha256": json_sha256(fit_contract), "model_state": _cpu_state_dict(model),
        "optimizer_state": copy.deepcopy(optimiser.state_dict()), "scheduler_state": None,
        "scaler": {"mean": mean.detach().cpu().clone(), "std": std.detach().cpu().clone()},
        "training_cursor": {"epoch": int(epoch), "step": int(step), "best_stop_smooth_l1": float(best_stop_loss), "best_epoch": int(best_epoch), "stale_epochs": int(stale_epochs)},
        "rng_state": _capture_rng_state(),
        "verification_reference": dict(verification_reference),
    }


def _validate_s2_payload(payload: Mapping[str, Any], *, fit_contract: Mapping[str, Any], role: str) -> None:
    if payload.get("kind") != S2_CHECKPOINT_KIND or payload.get("schema_version") != S2_CHECKPOINT_SCHEMA_VERSION:
        raise V27ContractError("S2 checkpoint has unexpected kind/schema")
    if payload.get("checkpoint_role") != role or payload.get("fit_contract_sha256") != json_sha256(fit_contract) or payload.get("fit_contract") != dict(fit_contract):
        raise V27ContractError("S2 checkpoint provenance/configuration differs from current nested fit")
    if not isinstance(payload.get("model_state"), Mapping) or not isinstance(payload.get("optimizer_state"), Mapping) or not isinstance(payload.get("scaler"), Mapping) or not isinstance(payload.get("training_cursor"), Mapping) or not isinstance(payload.get("rng_state"), Mapping) or not isinstance(payload.get("verification_reference"), Mapping):
        raise V27ContractError("S2 checkpoint omits model, optimizer, scaler, cursor, RNG state, or verification reference")


def _model_data_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    scaler = payload["scaler"]
    return {"mean": scaler["mean"].cpu(), "std": scaler["std"].cpu(), "state": {key: value.cpu() for key, value in payload["model_state"].items()}}


def _model_values(model: Any, mean: Any, std: Any, rows: Sequence[Mapping[str, Any]], *, device: str) -> list[float]:
    import torch

    features = torch.tensor([row["features"] for row in rows], dtype=torch.float32, device=device)
    with torch.inference_mode():
        values = model((features - mean) / std).squeeze(1).detach().cpu().tolist()
    del features
    return [float(value) for value in values]


def _s2_values(model_data: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], *, args: argparse.Namespace) -> list[float]:
    import torch

    model = _new_s2_model(args)
    model.load_state_dict({key: value.to(args.device) for key, value in model_data["state"].items()}); model.eval()
    values = _model_values(model, model_data["mean"].to(args.device), model_data["std"].to(args.device), rows, device=args.device)
    del model; torch.cuda.empty_cache()
    return values


def _s2_decision_signature(rows: Sequence[Mapping[str, Any]], values: Sequence[float]) -> dict[str, str]:
    grouped: dict[str, list[tuple[Mapping[str, Any], float]]] = {}
    for row, value in zip(rows, values, strict=True):
        grouped.setdefault(str(row["qid"]), []).append((row, float(value)))
    decisions: dict[str, str] = {}
    for qid, candidates in grouped.items():
        winner, value = max(candidates, key=lambda item: (item[1], -int(item[0]["candidate_position"]), str(item[0]["doc_id"])))
        decisions[qid] = str(winner["doc_id"]) if value > float(S2_POLICY["decision_floor"]) else str(winner["incumbent"])
    return decisions


def _s2_verification_reference(model: Any, mean: Any, std: Any, rows: Sequence[Mapping[str, Any]], *, device: str) -> dict[str, Any]:
    values = _model_values(model, mean, std, rows, device=device)
    return {
        "row_keys": [{"qid": str(row["qid"]), "doc_id": str(row["doc_id"]), "candidate_position": int(row["candidate_position"])} for row in rows],
        "scores": values,
        "slot5_decisions": _s2_decision_signature(rows, values),
    }


def _verify_s2_checkpoint(
    *, path: Path, role: str, fit_contract: Mapping[str, Any], verification_rows: Sequence[Mapping[str, Any]], args: argparse.Namespace,
) -> dict[str, Any]:
    payload = _torch_load(path); _validate_s2_payload(payload, fit_contract=fit_contract, role=role)
    restored = _model_data_from_payload(payload)
    reference = payload["verification_reference"]
    expected_keys = [{"qid": str(row["qid"]), "doc_id": str(row["doc_id"]), "candidate_position": int(row["candidate_position"])} for row in verification_rows]
    if reference.get("row_keys") != expected_keys or not isinstance(reference.get("scores"), list) or not isinstance(reference.get("slot5_decisions"), Mapping):
        raise V27ContractError(f"S2 {role} verification reference differs from the fixed A-stop batch")
    expected_values = [float(value) for value in reference["scores"]]
    restored_values = _s2_values(restored, verification_rows, args=args)
    if not np.allclose(expected_values, restored_values, rtol=S2_VERIFICATION_RTOL, atol=S2_VERIFICATION_ATOL):
        maximum = float(np.max(np.abs(np.asarray(expected_values) - np.asarray(restored_values))))
        raise V27ContractError(f"S2 {role} reload score verification failed; maximum absolute error={maximum}")
    expected_decisions = {str(key): str(value) for key, value in reference["slot5_decisions"].items()}
    restored_decisions = _s2_decision_signature(verification_rows, restored_values)
    if expected_decisions != restored_decisions:
        raise V27ContractError(f"S2 {role} reload altered a slot-5 decision")
    return {"checkpoint": str(path.resolve()), "sha256": sha256_file(path), "role": role, "rows": len(verification_rows), "score_atol": S2_VERIFICATION_ATOL, "score_rtol": S2_VERIFICATION_RTOL, "max_abs_error": float(np.max(np.abs(np.asarray(expected_values) - np.asarray(restored_values)))) if expected_values else 0.0, "slot5_decisions_match": True, "pass": True}


def _fit_s2(
    split: Any, a_fit: Sequence[int], a_stop: Sequence[int], scores: Mapping[str, Mapping[str, Any]], variant: str, *,
    args: argparse.Namespace, seed: int, regime: str, inner_seed: int, outer_fold: int,
    lineage_sha256: str, partition_metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    train = [row for index in a_fit for row in _s2_features(split, int(index), scores, variant, with_target=True)]
    stop = [row for index in a_stop for row in _s2_features(split, int(index), scores, variant, with_target=True)]
    if not train or not stop:
        raise V27ContractError("S2 nested A-fit/A-stop split has no candidate rows")
    verification_rows = sorted(stop, key=lambda row: (str(row["qid"]), int(row["candidate_position"]), str(row["doc_id"])))[: min(32, len(stop))]
    fit_contract = _s2_fit_contract(split=split, a_fit=a_fit, a_stop=a_stop, variant=variant, args=args, seed=seed, regime=regime, inner_seed=inner_seed, outer_fold=outer_fold, lineage_sha256=lineage_sha256, partition_metadata=partition_metadata)
    fit_dir = _s2_fit_dir(args, regime=regime, inner_seed=inner_seed, outer_fold=outer_fold, variant=variant)
    last_path, selected_path, backup_path, complete_path = fit_dir / "last.pt", fit_dir / "selected.pt", fit_dir / "selected.backup.pt", fit_dir / "fit_complete.json"
    x_train = torch.tensor([row["features"] for row in train], dtype=torch.float32, device=args.device)
    y_train = torch.tensor([float(row["target"]) for row in train], dtype=torch.float32, device=args.device).unsqueeze(1)
    x_stop = torch.tensor([row["features"] for row in stop], dtype=torch.float32, device=args.device)
    y_stop = torch.tensor([float(row["target"]) for row in stop], dtype=torch.float32, device=args.device).unsqueeze(1)
    mean = x_train.mean(dim=0, keepdim=True); std = x_train.std(dim=0, keepdim=True).clamp_min(1.0e-6)
    torch.manual_seed(int(seed)); torch.cuda.manual_seed_all(int(seed))
    model = _new_s2_model(args)
    optimiser = torch.optim.AdamW(model.parameters(), lr=float(args.s2_learning_rate), weight_decay=1.0e-4)
    start_epoch = 1; best_loss = math.inf; best_epoch = 0; stale = 0; step = 0
    if last_path.is_file():
        last_payload = _torch_load(last_path); _validate_s2_payload(last_payload, fit_contract=fit_contract, role="last")
        model.load_state_dict({key: value.to(args.device) for key, value in last_payload["model_state"].items()})
        optimiser.load_state_dict(last_payload["optimizer_state"])
        saved_scaler = last_payload["scaler"]
        if not torch.allclose(mean.detach().cpu(), saved_scaler["mean"].cpu()) or not torch.allclose(std.detach().cpu(), saved_scaler["std"].cpu()):
            raise V27ContractError("S2 resumed scaler differs from the frozen A-fit feature scaler")
        cursor = last_payload["training_cursor"]
        start_epoch, step = int(cursor["epoch"]) + 1, int(cursor["step"])
        best_loss, best_epoch, stale = float(cursor["best_stop_smooth_l1"]), int(cursor["best_epoch"]), int(cursor["stale_epochs"])
        _restore_rng_state(last_payload["rng_state"])
    for epoch in range(start_epoch, int(args.s2_epochs) + 1):
        model.train(); optimiser.zero_grad(set_to_none=True)
        loss = torch.nn.functional.smooth_l1_loss(model((x_train - mean) / std), y_train)
        loss.backward(); optimiser.step(); step += 1
        model.eval()
        with torch.inference_mode():
            stop_loss = float(torch.nn.functional.smooth_l1_loss(model((x_stop - mean) / std), y_stop).detach().cpu())
        if stop_loss + 1.0e-9 < best_loss:
            best_loss, best_epoch, stale = stop_loss, epoch, 0
            selected_payload = _checkpoint_payload(
                fit_contract=fit_contract, checkpoint_role="selected_a_stop", model=model, optimiser=optimiser,
                mean=mean, std=std, epoch=epoch, step=step, best_stop_loss=best_loss, best_epoch=best_epoch,
                stale_epochs=stale, verification_reference=_s2_verification_reference(model, mean, std, verification_rows, device=args.device),
            )
            _atomic_torch_save(selected_payload, selected_path)
            _atomic_backup(selected_path, backup_path)
        else:
            stale += 1
        last_payload = _checkpoint_payload(
            fit_contract=fit_contract, checkpoint_role="last", model=model, optimiser=optimiser,
            mean=mean, std=std, epoch=epoch, step=step, best_stop_loss=best_loss, best_epoch=best_epoch,
            stale_epochs=stale, verification_reference=_s2_verification_reference(model, mean, std, verification_rows, device=args.device),
        )
        _atomic_torch_save(last_payload, last_path)
        print(json.dumps({"event": "qwen_s2_epoch_checkpoint", "regime": regime, "inner_seed": inner_seed, "outer_fold": outer_fold, "variant": variant, "epoch": epoch, "stop_smooth_l1": stop_loss, "best_stop_smooth_l1": best_loss}, ensure_ascii=False), flush=True)
        if stale >= int(args.s2_patience):
            break
    if not selected_path.is_file() or not last_path.is_file() or not backup_path.is_file():
        raise RuntimeError("S2 fit did not retain last/selected/backup checkpoints")
    selected_payload = _torch_load(selected_path); _validate_s2_payload(selected_payload, fit_contract=fit_contract, role="selected_a_stop")
    last_payload = _torch_load(last_path); _validate_s2_payload(last_payload, fit_contract=fit_contract, role="last")
    selected_data = _model_data_from_payload(selected_payload)
    rng_after_fit = _capture_rng_state()
    selected_verification = _verify_s2_checkpoint(path=selected_path, role="selected_a_stop", fit_contract=fit_contract, verification_rows=verification_rows, args=args)
    last_verification = _verify_s2_checkpoint(path=last_path, role="last", fit_contract=fit_contract, verification_rows=verification_rows, args=args)
    _restore_rng_state(rng_after_fit)
    backup_payload = _torch_load(backup_path); _validate_s2_payload(backup_payload, fit_contract=fit_contract, role="selected_a_stop")
    if sha256_file(selected_path) != sha256_file(backup_path):
        raise V27ContractError("S2 selected checkpoint backup no longer matches its primary")
    marker = {"kind": "legalir_qwen_c200_s2_fit_complete", "schema_version": 1, "fit_contract": fit_contract, "fit_contract_sha256": json_sha256(fit_contract), "last": last_verification, "selected": selected_verification, "selected_backup": {"path": str(backup_path.resolve()), "sha256": sha256_file(backup_path)}, "selection": "minimum A-stop SmoothL1 only; calibration C was not used to select a checkpoint", "complete": True}
    write_json(complete_path, marker)
    report = {"kind": "tiny_mlp_top3_delta", "fit_candidate_rows": len(train), "stop_candidate_rows": len(stop), "epochs_completed": int(last_payload["training_cursor"]["epoch"]), "best_stop_smooth_l1": float(selected_payload["training_cursor"]["best_stop_smooth_l1"]), "best_epoch": int(selected_payload["training_cursor"]["best_epoch"]), "hidden": int(args.s2_hidden), "learning_rate": float(args.s2_learning_rate), "seed": int(seed), "feature_schema": list(S2_FEATURE_SCHEMA), "policy": dict(S2_POLICY), "fit_contract_sha256": json_sha256(fit_contract), "artifacts": {"directory": str(fit_dir.resolve()), "last": last_verification, "selected": selected_verification, "selected_backup": marker["selected_backup"], "complete_marker": str(complete_path.resolve())}, "checkpoint_verified": True}
    del model, x_train, y_train, x_stop, y_stop
    torch.cuda.empty_cache()
    return selected_data, report


def _s2_predictions(split: Any, indices: Sequence[int], scores: Mapping[str, Mapping[str, Any]], variant: str, model_data: Mapping[str, Any], *, args: argparse.Namespace) -> tuple[dict[str, dict[str, list[str]]], dict[str, Any]]:
    import torch

    model = torch.nn.Sequential(torch.nn.Linear(5, int(args.s2_hidden)), torch.nn.ReLU(), torch.nn.Linear(int(args.s2_hidden), 1)).to(args.device)
    model.load_state_dict({key: value.to(args.device) for key, value in model_data["state"].items()}); model.eval()
    mean, std = model_data["mean"].to(args.device), model_data["std"].to(args.device)
    predictions: dict[str, dict[str, list[str]]] = {}; decisions: dict[str, Any] = {}
    with torch.inference_mode():
        for index in indices:
            rows = _s2_features(split, int(index), scores, variant, with_target=False)
            qid = str(split.qids[index]); anchor = [str(value) for value in split.anchor[index]]
            if rows:
                features = torch.tensor([row["features"] for row in rows], dtype=torch.float32, device=args.device)
                values = model((features - mean) / std).squeeze(1).detach().cpu().tolist()
                best = max(zip(rows, values, strict=True), key=lambda item: (float(item[1]), -int(item[0]["candidate_position"]), str(item[0]["doc_id"])))
                replacement = str(best[0]["doc_id"]) if float(best[1]) > 0.0 else None
                diagnostics = {"candidate": best[0]["doc_id"], "predicted_delta": float(best[1]), "candidate_count": len(rows)}
            else:
                replacement = None; diagnostics = {"candidate": None, "predicted_delta": None, "candidate_count": 0}
            predictions[qid] = {"answer": _answer(anchor, replacement)}
            decisions[qid] = {"selector": "S2", "incumbent": anchor[EDITABLE_SLOT - 1], "changed": replacement is not None, **diagnostics}
    del model; torch.cuda.empty_cache()
    return predictions, decisions


def _compact(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "rows"}


def _ranking_diagnostic(split: Any, indices: Sequence[int], scores: Mapping[str, Mapping[str, Any]], variant: str) -> dict[str, Any]:
    if split.references is None:
        raise V27ContractError("Qwen ranking diagnostic needs complement qrels")
    r1: list[float] = []; r3: list[float] = []
    for index in indices:
        qid = str(split.qids[index]); ranked = score_order(scores, qid, pool_for(split._qwen_universe, qid, variant))  # type: ignore[attr-defined]
        gold = set(map(str, split.references[index])); r1.append(len(set(ranked[:1]) & gold) / len(gold)); r3.append(len(set(ranked[:3]) & gold) / len(gold))
    return {"qwen_rank_recall_at_1": float(np.mean(r1)), "qwen_rank_recall_at_3": float(np.mean(r3)), "queries": len(r1)}


def _select_configuration(trials: Sequence[Mapping[str, Any]]) -> tuple[str | None, dict[str, Any], list[dict[str, Any]]]:
    eligible = [trial for trial in trials if float(trial["calibration_metrics"]["gain"]) > EPSILON and int(trial["calibration_metrics"]["successful_rescues"]) > int(trial["calibration_metrics"]["destructive_changes"])]
    selected = None if not eligible else max(eligible, key=lambda trial: (float(trial["calibration_metrics"]["gain"]), -int(trial["calibration_metrics"]["destructive_changes"]), int(trial["calibration_metrics"]["successful_rescues"]), str(trial["name"])))
    eligible_names = {str(trial["name"]) for trial in eligible}
    trace = [{key: value for key, value in trial.items() if key not in {"calibration_predictions", "validation_predictions", "validation_decisions"}} | {"eligible": str(trial["name"]) in eligible_names} for trial in trials]
    return (None if selected is None else str(selected["name"]), {} if selected is None else selected, trace)


def _run_fold(*, args: argparse.Namespace, split: Any, groups: np.ndarray, cache_scores: Mapping[str, Mapping[str, Mapping[str, Any]]], regime: str, inner_seed: int, outer_fold: int, outer_train: np.ndarray, validation: np.ndarray, lineage_sha256: str) -> tuple[dict[str, dict[str, list[str]]], dict[str, Any]]:
    partitions, partition_metadata = e3b.make_partitions(split=split, groups=groups, outer_train=outer_train, validation=validation, inner_seed=inner_seed, outer_fold=outer_fold)
    calibration = _subset(split, partitions["calibration"]); validation_split = _subset(split, partitions["validation"])
    # ``V27Split`` is intentionally a light mutable data carrier; retain the
    # label-blind candidate universe on derived partitions without copying its
    # 1,197 c200 lists into every nested fold.
    setattr(calibration, "_qwen_universe", split._qwen_universe)  # type: ignore[attr-defined]
    setattr(validation_split, "_qwen_universe", split._qwen_universe)  # type: ignore[attr-defined]
    trials: list[dict[str, Any]] = []
    for variant in VARIANTS:
        scores = _pool_scores(cache_scores, variant)
        threshold, s1_calibration, threshold_trace = _s1_calibrate(calibration, scores, variant, device=args.device)
        c_predictions, _ = _s1_predictions(calibration, range(len(calibration.qids)), scores, variant, threshold)
        v_predictions, v_decisions = _s1_predictions(validation_split, range(len(validation_split.qids)), scores, variant, threshold)
        trials.append({"name": f"{variant}/S1", "variant": variant, "selector": "S1", "selected_threshold": threshold, "threshold_trace": threshold_trace, "calibration_metrics": s1_calibration, "calibration_predictions": c_predictions, "validation_predictions": v_predictions, "validation_decisions": v_decisions, "fit": None})
        model_data, fit = _fit_s2(
            split, partitions["a_fit"], partitions["a_stop"], scores, variant,
            args=args, seed=int(inner_seed) + 1000 * (outer_fold + 1) + VARIANTS.index(variant),
            regime=regime, inner_seed=inner_seed, outer_fold=outer_fold,
            lineage_sha256=lineage_sha256, partition_metadata=partition_metadata,
        )
        c_predictions, _ = _s2_predictions(calibration, range(len(calibration.qids)), scores, variant, model_data, args=args)
        v_predictions, v_decisions = _s2_predictions(validation_split, range(len(validation_split.qids)), scores, variant, model_data, args=args)
        trials.append({"name": f"{variant}/S2", "variant": variant, "selector": "S2", "selected_threshold": 0.0, "threshold_trace": None, "calibration_metrics": _compact(transition_metrics_device(calibration, c_predictions, device=args.device)), "calibration_predictions": c_predictions, "validation_predictions": v_predictions, "validation_decisions": v_decisions, "fit": fit})
    selected_name, selected, trace = _select_configuration(trials)
    if selected_name is None:
        predictions = {str(qid): {"answer": [str(value) for value in validation_split.anchor[index]]} for index, qid in enumerate(validation_split.qids)}
        decisions = {str(qid): {"selector": "abstain", "changed": False} for qid in validation_split.qids}
        validation_metrics = _compact(transition_metrics_device(validation_split, predictions, device=args.device))
        selected_policy: dict[str, Any] = {"kind": "v16b_anchor_abstain"}
    else:
        predictions = selected["validation_predictions"]; decisions = selected["validation_decisions"]
        validation_metrics = _compact(transition_metrics_device(validation_split, predictions, device=args.device))
        selected_policy = {"name": selected_name, "variant": selected["variant"], "selector": selected["selector"], "threshold": selected["selected_threshold"]}
    pool_oracles = {variant: oracle_slot5(validation_split, {str(qid): pool_for(split._qwen_universe, str(qid), variant) for qid in validation_split.qids}) for variant in VARIANTS}  # type: ignore[attr-defined]
    ranking = {variant: _ranking_diagnostic(validation_split, range(len(validation_split.qids)), _pool_scores(cache_scores, variant), variant) for variant in VARIANTS}
    return predictions, {"outer_fold": int(outer_fold), "partitions": partition_metadata, "selected_configuration": selected_policy, "calibration_configuration_trace": trace, "validation": validation_metrics, "validation_pool_oracles": pool_oracles, "validation_qwen_ranking": ranking, "decisions": decisions}


def _checkpoint_load(args: argparse.Namespace, manifest: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _checkpoint_path(args)
    if path.exists():
        if not args.resume:
            raise FileExistsError(f"Refusing to reuse Qwen nested checkpoint without --resume: {path}")
        state = read_json(path)
        if not isinstance(state, dict) or state.get("kind") != f"{KIND}_checkpoint" or state.get("lineage_sha256") != manifest["lineage_sha256"]:
            raise V27ContractError("Qwen nested checkpoint does not match this frozen cache/code lineage")
        return path, state
    return path, {"kind": f"{KIND}_checkpoint", "schema_version": 1, "lineage_sha256": manifest["lineage_sha256"], "mode": args.mode, "completed_folds": {}, "complete": False}


def _validate_e3b_prerequisites(args: argparse.Namespace) -> dict[str, Any]:
    standard, stress = read_json(args.e3b_standard_report), read_json(args.e3b_stress_report)
    if not isinstance(standard, dict) or not isinstance(stress, dict) or standard.get("promotion_to_stress_allowed") is not True or stress.get("promotion_to_e4_allowed") is not True:
        raise V27ContractError("Qwen c200 branch requires completed, passing E3-B standard and stress reports")
    return {"e3b_standard_report": str(args.e3b_standard_report.resolve()), "e3b_stress_report": str(args.e3b_stress_report.resolve()), "standard_promotion_to_stress_allowed": True, "stress_promotion_to_e4_allowed": True}


def _validate_standard_gate(path: Path, *, lineage_sha256: str) -> None:
    report = read_json(path)
    if not isinstance(report, dict) or report.get("kind") != KIND or report.get("mode") != "standard" or report.get("lineage_sha256") != lineage_sha256 or report.get("promotion_to_stress_allowed") is not True:
        raise V27ContractError("Qwen standard report did not pass or differs from this cache/code lineage")


def _run_seed(*, args: argparse.Namespace, split: Any, groups: np.ndarray, cache_scores: Mapping[str, Mapping[str, Mapping[str, Any]]], regime: str, inner_seed: int, checkpoint_path: Path, checkpoint: dict[str, Any], lineage_sha256: str) -> dict[str, Any]:
    oof: dict[str, dict[str, list[str]]] = {}; reports: list[dict[str, Any]] = []
    for outer_fold, (outer_train, validation) in enumerate(group_folds(groups)):
        key = f"{regime}|inner={inner_seed}|fold={outer_fold}"
        qids = [str(split.qids[index]) for index in validation]
        saved = checkpoint["completed_folds"].get(key)
        if saved is not None:
            if saved.get("validation_qids") != qids or not isinstance(saved.get("predictions"), dict):
                raise V27ContractError("Qwen nested saved fold does not match current grouped validation")
            oof.update(saved["predictions"]); reports.append(saved["fold_report"])
            print(json.dumps({"event": "qwen_c200_fold_resumed", "run": key}, ensure_ascii=False), flush=True)
            continue
        print(json.dumps({"event": "qwen_c200_fold_start", "run": key, "validation_queries": len(validation)}, ensure_ascii=False), flush=True)
        predictions, report = _run_fold(args=args, split=split, groups=groups, cache_scores=cache_scores, regime=regime, inner_seed=inner_seed, outer_fold=outer_fold, outer_train=outer_train, validation=validation, lineage_sha256=lineage_sha256)
        oof.update(predictions); reports.append(report)
        checkpoint["completed_folds"][key] = {"validation_qids": qids, "predictions": predictions, "fold_report": report}
        write_json(checkpoint_path, checkpoint)
        print(json.dumps({"event": "qwen_c200_fold_complete", "run": key, "selected": report["selected_configuration"], "validation": report["validation"]}, ensure_ascii=False), flush=True)
    if set(oof) != set(map(str, split.qids)):
        raise V27ContractError(f"Qwen c200 {regime}/inner={inner_seed}: OOF does not cover complement exactly")
    ordered = {str(qid): oof[str(qid)] for qid in split.qids}
    return {"inner_seed": int(inner_seed), "folds": reports, "metrics": _compact(transition_metrics_device(split, ordered, device=args.device)), "predictions": ordered}


def _gate(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_seed = [float(run["metrics"]["gain"]) > EPSILON and int(run["metrics"]["successful_rescues"]) > int(run["metrics"]["destructive_changes"]) for run in runs]
    return {"per_inner_seed_pass": per_seed, "pass": bool(all(per_seed)), "minimum_gain": min(float(run["metrics"]["gain"]) for run in runs), "minimum_rescue_minus_destructive": min(int(run["metrics"]["successful_rescues"]) - int(run["metrics"]["destructive_changes"]) for run in runs)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest, label_blind_universe, cache_scores = _preflight(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / f"qwen_c200_{args.mode}_label_blind_manifest.json"
    write_json(manifest_path, manifest)
    if args.mode == "preflight":
        return manifest
    if args.mode == "stress":
        if args.standard_report is None:
            raise ValueError("MODE=stress requires --standard-report")
        _validate_standard_gate(args.standard_report, lineage_sha256=manifest["lineage_sha256"])
    prerequisites = _validate_e3b_prerequisites(args)
    # Only after label-blind cache manifest is immutable may qrels enter memory.
    universe = load_candidate_universe(stage1=args.stage1, stage2=args.stage2, anchor=args.anchor, reference=args.reference)
    split = universe.split
    setattr(split, "_qwen_universe", universe)
    evidence = load_manifest(args.e1_manifest)
    groups_by_regime = build_regime_groups(split, evidence)
    regimes = (STANDARD_REGIME,) if args.mode == "standard" else STRESS_REGIMES
    checkpoint_path, checkpoint = _checkpoint_load(args, manifest)
    reports = {regime: [_run_seed(args=args, split=split, groups=groups_by_regime[regime], cache_scores=cache_scores, regime=regime, inner_seed=int(seed), checkpoint_path=checkpoint_path, checkpoint=checkpoint, lineage_sha256=manifest["lineage_sha256"]) for seed in args.inner_seeds] for regime in regimes}
    checkpoint["complete"] = True; write_json(checkpoint_path, checkpoint)
    gates = {regime: _gate(runs) for regime, runs in reports.items()}
    return {"kind": KIND, "schema_version": SCHEMA_VERSION, "status": "nested_complement_oof_research_only", "mode": args.mode, "split": "complement", "lineage": manifest["lineage"], "lineage_sha256": manifest["lineage_sha256"], "label_blind_manifest": str(manifest_path.resolve()), "labels_opened_after_manifest": True, "holdout_opened": False, "public_labels_used": False, "e3b_prerequisites": prerequisites, "checkpoint": {"path": str(checkpoint_path.resolve()), "completed_folds": len(checkpoint["completed_folds"]), "resumable": True}, "regimes": reports, "gates": gates, "promotion_to_stress_allowed": bool(args.mode == "standard" and gates[STANDARD_REGIME]["pass"]), "promotion_to_e4_allowed": False, "next_action": "Research review only; E4/E5 remain sealed and are not opened by this branch."}


def main() -> int:
    args = parse_args(); output = _output_path(args)
    if output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite Qwen c200 output: {output}")
    report = run(args); write_json(output, report)
    print(json.dumps({"output": str(output.resolve()), "stage": report.get("stage", report.get("status")), "holdout_opened": False, "promotion_to_stress_allowed": report.get("promotion_to_stress_allowed", False), "promotion_to_e4_allowed": False}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
