"""OOF-select a query-normalized pointwise/listwise score ensemble.

The component score files must come from the same grouped-complement protocol.
Blend mode and weight are selected only on complement OOF predictions.  The
locked holdout and public tensors are opened after selection, and a submission
is emitted only when the ensemble beats the supplied holdout champion gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.model_selection import GroupKFold

try:
    from .v11_tiny_ltr import (
        apply_policy,
        build_leak_groups,
        evaluate_policy,
        load_audit_split,
        write_json,
    )
    from .v11_tiny_ltr_complement import clean_complement
except ImportError:
    from v11_tiny_ltr import (
        apply_policy,
        build_leak_groups,
        evaluate_policy,
        load_audit_split,
        write_json,
    )
    from v11_tiny_ltr_complement import clean_complement


def load_score_cache(
    path: Path,
    complement_qids: Sequence[str],
    holdout_qids: Sequence[str],
    public_qids: Sequence[str],
    candidate_limit: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata"].item()))
        if int(metadata.get("candidate_limit", -1)) != candidate_limit:
            raise ValueError(f"{path}: candidate limit mismatch")
        for key, expected in (
            ("complement_qids", complement_qids),
            ("holdout_qids", holdout_qids),
            ("public_qids", public_qids),
        ):
            if list(metadata.get(key, [])) != list(expected):
                raise ValueError(f"{path}: {key} mismatch")
        oof = cache["oof_scores"].copy()
        holdout = cache["holdout_scores"].copy()
        public = cache["public_scores"].copy()
    expected_shapes = (
        (len(complement_qids), candidate_limit),
        (len(holdout_qids), candidate_limit),
        (len(public_qids), candidate_limit),
    )
    for name, values, expected in zip(
        ("OOF", "holdout", "public"), (oof, holdout, public), expected_shapes,
        strict=True,
    ):
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError(f"{path}: invalid {name} score matrix {values.shape}")
    return metadata, oof, holdout, public


def load_feature_cache(
    path: Path,
    complement_questions: int,
    holdout_questions: int,
    public_questions: int,
    candidate_limit: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as cache:
        complement = cache["complement_matrix"].copy()
        holdout = cache["holdout_matrix"].copy()
        public = cache["public_matrix"].copy()
        names = [str(value) for value in cache["feature_names"]]
    expected_rows = (
        complement_questions * candidate_limit,
        holdout_questions * candidate_limit,
        public_questions * candidate_limit,
    )
    for label, matrix, rows in zip(
        ("complement", "holdout", "public"),
        (complement, holdout, public), expected_rows, strict=True,
    ):
        if matrix.shape[0] != rows or matrix.shape[1] != len(names):
            raise ValueError(f"feature cache {label} shape mismatch: {matrix.shape}")
    return complement, holdout, public, names


def normalize_scores(scores: np.ndarray, pool: int, mode: str) -> np.ndarray:
    output = np.zeros_like(scores, dtype=np.float32)
    active = scores[:, :pool].astype(np.float64)
    if mode == "zscore":
        mean = active.mean(axis=1, keepdims=True)
        scale = active.std(axis=1, keepdims=True)
        scale[scale < 1e-8] = 1.0
        output[:, :pool] = ((active - mean) / scale).astype(np.float32)
    elif mode == "rank":
        for row_index, row in enumerate(active):
            order = np.argsort(-row, kind="stable")
            values = np.empty(pool, dtype=np.float32)
            values[order] = np.linspace(1.0, 0.0, pool, dtype=np.float32)
            output[row_index, :pool] = values
    else:
        raise ValueError(f"Unknown score normalization: {mode}")
    return output


def fold_mapping(groups: np.ndarray) -> dict[int, int]:
    indices = np.arange(len(groups), dtype=np.int32)
    mapping: dict[int, int] = {}
    splitter = GroupKFold(n_splits=5)
    for fold, (_, validation) in enumerate(splitter.split(indices, groups=groups)):
        for local in validation:
            mapping[int(indices[local])] = fold
    return mapping


def blend_scores(
    pointwise: np.ndarray,
    listwise: np.ndarray,
    pool: int,
    mode: str,
    pointwise_weight: float,
) -> np.ndarray:
    point = normalize_scores(pointwise, pool, mode)
    listing = normalize_scores(listwise, pool, mode)
    return pointwise_weight * point + (1.0 - pointwise_weight) * listing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complement-stage1", type=Path, required=True)
    parser.add_argument("--complement-stage2", type=Path, required=True)
    parser.add_argument("--complement-reference", type=Path, required=True)
    parser.add_argument("--holdout-stage1", type=Path, required=True)
    parser.add_argument("--holdout-stage2", type=Path, required=True)
    parser.add_argument("--holdout-reference", type=Path, required=True)
    parser.add_argument("--public-stage1", type=Path, required=True)
    parser.add_argument("--public-stage2", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--pointwise-scores", type=Path, required=True)
    parser.add_argument("--listwise-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--submission-name", default="v11k_ensemble.json")
    parser.add_argument("--candidate-limit", type=int, default=80)
    parser.add_argument("--pool", type=int, default=35)
    parser.add_argument("--minimum-holdout-recall", type=float, required=True)
    args = parser.parse_args()
    if not 5 <= args.pool <= args.candidate_limit:
        raise ValueError("pool must be within [5, candidate-limit]")
    if Path(args.submission_name).name != args.submission_name:
        raise ValueError("submission-name must be a plain filename")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_complement = load_audit_split(
        "complement", args.complement_stage1, args.complement_stage2,
        args.complement_reference,
    )
    complement, excluded_exact_ids = clean_complement(
        raw_complement, args.complement_stage1
    )
    holdout = load_audit_split(
        "holdout", args.holdout_stage1, args.holdout_stage2,
        args.holdout_reference,
    )
    public = load_audit_split(
        "public", args.public_stage1, args.public_stage2, None
    )
    point_meta, point_oof, point_holdout, point_public = load_score_cache(
        args.pointwise_scores, complement.qids, holdout.qids, public.qids,
        args.candidate_limit,
    )
    list_meta, list_oof, list_holdout, list_public = load_score_cache(
        args.listwise_scores, complement.qids, holdout.qids, public.qids,
        args.candidate_limit,
    )
    complement_matrix, holdout_matrix, public_matrix, feature_names = load_feature_cache(
        args.feature_cache, len(complement.qids), len(holdout.qids),
        len(public.qids), args.candidate_limit,
    )
    indices = np.arange(len(complement.qids), dtype=np.int32)
    groups = build_leak_groups(complement)
    folds = fold_mapping(groups)
    policy = {"kind": "prefix", "pool": args.pool, "keep": 0, "exact_rescue": True}

    leaderboard: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_scores: np.ndarray | None = None
    for mode in ("zscore", "rank"):
        for pointwise_weight in np.linspace(0.0, 1.0, 21):
            blended = blend_scores(
                point_oof, list_oof, args.pool, mode, float(pointwise_weight)
            )
            predictions = apply_policy(
                complement, blended, indices, complement_matrix, feature_names,
                args.candidate_limit, policy,
            )
            evaluation = evaluate_policy(complement, predictions, indices, folds)
            row = {
                "mode": mode,
                "pointwise_weight": float(pointwise_weight),
                "listwise_weight": float(1.0 - pointwise_weight),
                "oof": evaluation,
            }
            leaderboard.append(row)
            key = (
                evaluation["robust_recall"], evaluation["recall"],
                evaluation["precision"], -evaluation["fold_recall_std"],
            )
            if best is None:
                best, best_scores = row, blended
            else:
                previous = best["oof"]
                previous_key = (
                    previous["robust_recall"], previous["recall"],
                    previous["precision"], -previous["fold_recall_std"],
                )
                if key > previous_key:
                    best, best_scores = row, blended
    assert best is not None and best_scores is not None
    leaderboard.sort(
        key=lambda row: (
            row["oof"]["robust_recall"], row["oof"]["recall"],
            row["oof"]["precision"],
        ),
        reverse=True,
    )
    print("ensemble configuration locked; opening holdout", flush=True)
    holdout_scores = blend_scores(
        point_holdout, list_holdout, args.pool, best["mode"],
        best["pointwise_weight"],
    )
    holdout_indices = np.arange(len(holdout.qids), dtype=np.int32)
    holdout_predictions = apply_policy(
        holdout, holdout_scores, holdout_indices, holdout_matrix, feature_names,
        args.candidate_limit, policy,
    )
    holdout_metrics = evaluate_policy(
        holdout, holdout_predictions, holdout_indices
    )
    promoted = float(holdout_metrics["recall"]) > args.minimum_holdout_recall
    public_output: Path | None = None
    public_changes_vs_pointwise = 0
    if promoted:
        public_scores = blend_scores(
            point_public, list_public, args.pool, best["mode"],
            best["pointwise_weight"],
        )
        public_indices = np.arange(len(public.qids), dtype=np.int32)
        public_predictions = apply_policy(
            public, public_scores, public_indices, public_matrix, feature_names,
            args.candidate_limit, policy,
        )
        pointwise_predictions = apply_policy(
            public,
            normalize_scores(point_public, args.pool, best["mode"]),
            public_indices, public_matrix, feature_names,
            args.candidate_limit, policy,
        )
        public_changes_vs_pointwise = sum(
            public_predictions[qid]["answer"]
            != pointwise_predictions[qid]["answer"]
            for qid in public.qids
        )
        public_output = args.output_dir / args.submission_name
        write_json(public_output, public_predictions)

    report = {
        "kind": "legalir_v11_oof_score_ensemble",
        "leakage_policy": {
            "selection": "blend mode/weight selected on grouped complement OOF only",
            "holdout": "opened after blend lock",
            "public_labels_used": False,
            "excluded_exact_memory_qids": excluded_exact_ids,
        },
        "candidate_limit": args.candidate_limit,
        "policy": policy,
        "pointwise_source": point_meta,
        "listwise_source": list_meta,
        "selected": best,
        "holdout": holdout_metrics,
        "promotion": {
            "minimum_holdout_recall_exclusive": args.minimum_holdout_recall,
            "promoted": promoted,
            "public_submission": str(public_output.resolve()) if public_output else None,
            "public_queries_changed_vs_pointwise": public_changes_vs_pointwise,
        },
        "leaderboard": leaderboard,
    }
    report_path = args.output_dir / "v11_score_ensemble_report.json"
    write_json(report_path, report)
    print(json.dumps({
        "report": str(report_path.resolve()),
        "selected": best,
        "holdout": holdout_metrics,
        "promotion": report["promotion"],
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
