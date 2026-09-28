"""Materialize the locked v11h holdout answers for a downstream gate.

The official v11h run writes only the public submission.  This utility
reconstructs its already-locked pointwise model/policy on the holdout using
the existing feature cache, without reopening model or policy selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .v11_tiny_ltr import (
        AuditSplit,
        apply_policy,
        load_audit_split,
        model_scores,
        train_model,
        write_json,
    )
except ImportError:
    from v11_tiny_ltr import (  # type: ignore
        AuditSplit,
        apply_policy,
        load_audit_split,
        model_scores,
        train_model,
        write_json,
    )


def subset_split(split: AuditSplit, indices: list[int]) -> AuditSplit:
    references = [split.references[index] for index in indices] if split.references is not None else None
    return AuditSplit(
        name=split.name,
        qids=[split.qids[index] for index in indices],
        questions=[split.questions[index] for index in indices],
        candidates=[split.candidates[index] for index in indices],
        baseline=[split.baseline[index] for index in indices],
        neural_scores=[split.neural_scores[index] for index in indices],
        result_rows=[split.result_rows[index] for index in indices],
        references=references,
        stage1_path=split.stage1_path,
    )


def clean_complement(split: AuditSplit) -> AuditSplit:
    if split.stage1_path is None:
        return split
    payload = json.loads(split.stage1_path.read_text(encoding="utf-8"))
    exact_ids = {
        str(row["query_id"])
        for row in payload["queries"]
        if bool(row.get("metadata", {}).get("exact_memory"))
    }
    keep = [index for index, qid in enumerate(split.qids) if qid not in exact_ids]
    return subset_split(split, keep)


def load_cache(path: Path) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata"].item()))
        return (
            metadata,
            cache["complement_matrix"].copy(),
            cache["complement_target"].copy(),
            cache["holdout_matrix"].copy(),
            [str(value) for value in cache["feature_names"]],
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complement-stage1", type=Path, required=True)
    parser.add_argument("--complement-stage2", type=Path, required=True)
    parser.add_argument("--complement-reference", type=Path, required=True)
    parser.add_argument("--holdout-stage1", type=Path, required=True)
    parser.add_argument("--holdout-stage2", type=Path, required=True)
    parser.add_argument("--holdout-reference", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    complement = clean_complement(load_audit_split(
        "complement", args.complement_stage1, args.complement_stage2,
        args.complement_reference,
    ))
    holdout = load_audit_split(
        "holdout", args.holdout_stage1, args.holdout_stage2, args.holdout_reference
    )
    metadata, complement_matrix, complement_target, holdout_matrix, feature_names = load_cache(
        args.feature_cache
    )
    if metadata.get("complement_qids") != complement.qids:
        raise ValueError("v11h cache complement qids do not match the clean complement split")
    if metadata.get("holdout_qids") != holdout.qids:
        raise ValueError("v11h cache holdout qids do not match the holdout split")
    if complement_target is None or len(complement_target) != complement_matrix.shape[0]:
        raise ValueError("v11h cache has no valid complement target")
    candidate_limit = 80
    if complement_matrix.shape[0] != len(complement.qids) * candidate_limit:
        raise ValueError("v11h complement cache has an unexpected row count")
    if holdout_matrix.shape[0] != len(holdout.qids) * candidate_limit:
        raise ValueError("v11h holdout cache has an unexpected row count")

    model = train_model(
        {"kind": "pointwise_logistic", "C": 1.0},
        complement_matrix,
        complement_target,
        np.arange(len(complement.qids), dtype=np.int32),
        candidate_limit,
    )
    scores = model_scores(model, holdout_matrix).reshape(
        len(holdout.qids), candidate_limit
    )
    policy = {"kind": "prefix", "pool": 35, "keep": 0, "exact_rescue": True}
    predictions = apply_policy(
        holdout,
        scores,
        np.arange(len(holdout.qids), dtype=np.int32),
        holdout_matrix,
        feature_names,
        candidate_limit,
        policy,
    )
    write_json(args.output, predictions)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "queries": len(predictions),
        "policy": policy,
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
