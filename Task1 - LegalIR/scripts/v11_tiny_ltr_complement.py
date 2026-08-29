"""Select tiny LTR on the clean complement split and test on v11 holdout.

This is the second, stricter LTR protocol.  Model and policy selection use
only grouped out-of-fold predictions from complement queries retrieved by an
index that does not contain those queries.  The original 1,000-question v11
holdout is opened only after locking the configuration.  Public inference is
written only when both selection and holdout beat their c35+exact anchors.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

try:
    from .v11_tiny_ltr import (
        AuditSplit,
        apply_policy,
        baseline_predictions,
        build_features,
        build_query_normalized_features,
        build_leak_groups,
        evaluate_policy,
        load_audit_split,
        load_documents,
        load_priors,
        model_scores,
        select_model_and_policy,
        train_model,
        write_json,
    )
except ImportError:
    from v11_tiny_ltr import (
        AuditSplit,
        apply_policy,
        baseline_predictions,
        build_features,
        build_query_normalized_features,
        build_leak_groups,
        evaluate_policy,
        load_audit_split,
        load_documents,
        load_priors,
        model_scores,
        select_model_and_policy,
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


def clean_complement(
    split: AuditSplit,
    stage1_path: Path,
) -> tuple[AuditSplit, list[str]]:
    audit = json.loads(stage1_path.read_text(encoding="utf-8"))
    exact_ids = {
        str(row["query_id"])
        for row in audit["queries"]
        if bool(row.get("metadata", {}).get("exact_memory"))
    }
    keep = [index for index, qid in enumerate(split.qids) if qid not in exact_ids]
    return subset_split(split, keep), sorted(exact_ids)


def feature_cache_signature(args: argparse.Namespace) -> str:
    paths = [
        args.contexts_zip, args.prior_index,
        args.complement_stage1, args.complement_stage2, args.complement_reference,
        args.holdout_stage1, args.holdout_stage2, args.holdout_reference,
        args.public_stage1, args.public_stage2,
    ]
    for stage1_path in (
        args.complement_stage1, args.holdout_stage1, args.public_stage1
    ):
        payload = json.loads(stage1_path.read_text(encoding="utf-8"))
        raw_source = payload.get("source_stage1_audit")
        if raw_source:
            source_path = Path(str(raw_source))
            if not source_path.is_absolute():
                source_path = stage1_path.parent / source_path
            paths.append(source_path)
    signature = []
    for path in paths:
        resolved = path.resolve()
        stat = resolved.stat()
        signature.append({
            "path": str(resolved), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
    return json.dumps(signature, ensure_ascii=False, sort_keys=True)


def normalized_cache_signature(raw_signature: str) -> list[dict[str, object]]:
    """Ignore the historical code-file stat; schema version guards features."""

    return [
        row for row in json.loads(raw_signature)
        if not str(row.get("path", "")).endswith("v11_tiny_ltr.py")
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts-zip", type=Path, required=True)
    parser.add_argument("--prior-index", type=Path, required=True)
    parser.add_argument("--complement-stage1", type=Path, required=True)
    parser.add_argument("--complement-stage2", type=Path, required=True)
    parser.add_argument("--complement-reference", type=Path, required=True)
    parser.add_argument("--holdout-stage1", type=Path, required=True)
    parser.add_argument("--holdout-stage2", type=Path, required=True)
    parser.add_argument("--holdout-reference", type=Path, required=True)
    parser.add_argument("--public-stage1", type=Path, required=True)
    parser.add_argument("--public-stage2", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-limit", type=int, default=80)
    parser.add_argument("--minimum-holdout-delta", type=float, default=0.0)
    parser.add_argument(
        "--feature-set", choices=("v1", "v2-query-normalized"), default="v1"
    )
    parser.add_argument(
        "--model-profile",
        choices=(
            "base", "pointwise-c-sweep", "listwise-v2",
            "pointwise-v2-winner", "listwise-v2-winner",
        ),
        default="base",
    )
    parser.add_argument(
        "--submission-name", default="v11f_tiny_ltr_complement.json"
    )
    parser.add_argument(
        "--feature-cache", type=Path,
        help="Optional validated .npz cache for feature matrices",
    )
    parser.add_argument(
        "--score-cache", type=Path,
        help="Optional .npz output with OOF/holdout/public model scores",
    )
    args = parser.parse_args()
    if args.candidate_limit < 80:
        raise ValueError("candidate-limit must be at least 80")
    if Path(args.submission_name).name != args.submission_name or not args.submission_name.endswith(".json"):
        raise ValueError("submission-name must be a plain .json filename")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    raw_complement = load_audit_split(
        "complement", args.complement_stage1, args.complement_stage2,
        args.complement_reference,
    )
    complement, excluded_exact_ids = clean_complement(raw_complement, args.complement_stage1)
    holdout = load_audit_split(
        "holdout", args.holdout_stage1, args.holdout_stage2,
        args.holdout_reference,
    )
    public = load_audit_split(
        "public", args.public_stage1, args.public_stage2, None
    )
    cache_signature = feature_cache_signature(args)
    cache_hit = False
    if args.feature_cache and args.feature_cache.exists():
        try:
            with np.load(args.feature_cache, allow_pickle=False) as cache:
                metadata = json.loads(str(cache["metadata"].item()))
                expected = {
                    "version": 1,
                    "feature_set": args.feature_set,
                    "candidate_limit": args.candidate_limit,
                    "complement_qids": complement.qids,
                    "holdout_qids": holdout.qids,
                    "public_qids": public.qids,
                }
                cached_signature = str(metadata.pop("signature", "[]"))
                if (
                    metadata != expected
                    or normalized_cache_signature(cached_signature)
                    != normalized_cache_signature(cache_signature)
                ):
                    raise ValueError("metadata/input fingerprint mismatch")
                complement_matrix = cache["complement_matrix"].copy()
                complement_target = cache["complement_target"].copy()
                holdout_matrix = cache["holdout_matrix"].copy()
                holdout_target = cache["holdout_target"].copy()
                public_matrix = cache["public_matrix"].copy()
                feature_names = [str(value) for value in cache["feature_names"]]
                holdout_names = list(feature_names)
                public_names = list(feature_names)
            cache_hit = True
            print(f"feature cache hit: {args.feature_cache}", flush=True)
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
            print(f"feature cache rejected ({error}); rebuilding", flush=True)
    if not cache_hit:
        needed_ids = {
            doc_id
            for split in (complement, holdout, public)
            for row in split.candidates
            for doc_id in row[:args.candidate_limit]
        }
        needed_ids.update(
            doc_id
            for split in (complement, holdout, public)
            for row in split.baseline for doc_id in row
        )
        priors = load_priors(args.prior_index)
        documents, document_frequency = load_documents(
            args.contexts_zip, needed_ids, priors
        )
        feature_builder = (
            build_features
            if args.feature_set == "v1"
            else build_query_normalized_features
        )
        complement_matrix, complement_target, feature_names = feature_builder(
            complement, documents, document_frequency, args.candidate_limit
        )
        holdout_matrix, holdout_target, holdout_names = feature_builder(
            holdout, documents, document_frequency, args.candidate_limit
        )
        public_matrix, _, public_names = feature_builder(
            public, documents, document_frequency, args.candidate_limit
        )
        if args.feature_cache:
            args.feature_cache.parent.mkdir(parents=True, exist_ok=True)
            metadata = {
                "version": 1,
                "feature_set": args.feature_set,
                "candidate_limit": args.candidate_limit,
                "signature": cache_signature,
                "complement_qids": complement.qids,
                "holdout_qids": holdout.qids,
                "public_qids": public.qids,
            }
            np.savez_compressed(
                args.feature_cache,
                metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
                complement_matrix=complement_matrix,
                complement_target=complement_target,
                holdout_matrix=holdout_matrix,
                holdout_target=holdout_target,
                public_matrix=public_matrix,
                feature_names=np.asarray(feature_names),
            )
            print(f"feature cache written: {args.feature_cache}", flush=True)
    if complement_target is None or holdout_target is None:
        raise RuntimeError("Labelled target missing")
    if feature_names != holdout_names or feature_names != public_names:
        raise RuntimeError("Feature schema mismatch")
    groups = build_leak_groups(complement)
    complement_indices = np.arange(len(complement.qids), dtype=np.int32)
    config, policy, leaderboard, oof_scores, fold_by_question = select_model_and_policy(
        complement, complement_matrix, complement_target, feature_names,
        complement_indices, groups, args.candidate_limit, args.model_profile,
    )
    complement_predictions = apply_policy(
        complement, oof_scores, complement_indices, complement_matrix,
        feature_names, args.candidate_limit, policy,
    )
    complement_ltr = evaluate_policy(
        complement, complement_predictions, complement_indices, fold_by_question
    )
    complement_baseline = evaluate_policy(
        complement,
        baseline_predictions(
            complement, complement_indices, complement_matrix, feature_names,
            args.candidate_limit, False,
        ),
        complement_indices,
    )
    complement_exact = evaluate_policy(
        complement,
        baseline_predictions(
            complement, complement_indices, complement_matrix, feature_names,
            args.candidate_limit, True,
        ),
        complement_indices,
    )
    print("complement configuration locked; opening original holdout once", flush=True)
    locked_model = train_model(
        config, complement_matrix, complement_target, complement_indices,
        args.candidate_limit,
    )
    holdout_indices = np.arange(len(holdout.qids), dtype=np.int32)
    holdout_values = model_scores(locked_model, holdout_matrix).reshape(
        len(holdout.qids), args.candidate_limit
    )
    holdout_predictions = apply_policy(
        holdout, holdout_values, holdout_indices, holdout_matrix, feature_names,
        args.candidate_limit, policy,
    )
    holdout_ltr = evaluate_policy(holdout, holdout_predictions, holdout_indices)
    holdout_baseline = evaluate_policy(
        holdout,
        baseline_predictions(
            holdout, holdout_indices, holdout_matrix, feature_names,
            args.candidate_limit, False,
        ),
        holdout_indices,
    )
    holdout_exact = evaluate_policy(
        holdout,
        baseline_predictions(
            holdout, holdout_indices, holdout_matrix, feature_names,
            args.candidate_limit, True,
        ),
        holdout_indices,
    )
    complement_delta = float(complement_ltr["recall"] - complement_exact["recall"])
    holdout_delta = float(holdout_ltr["recall"] - holdout_exact["recall"])
    promoted = complement_delta >= 0.0 and holdout_delta >= args.minimum_holdout_delta
    public_output: Path | None = None
    public_values: np.ndarray | None = None
    public_changes = 0
    if promoted:
        combined_matrix = np.vstack((complement_matrix, holdout_matrix))
        combined_target = np.concatenate((complement_target, holdout_target))
        combined_indices = np.arange(
            len(complement.qids) + len(holdout.qids), dtype=np.int32
        )
        final_model = train_model(
            config, combined_matrix, combined_target, combined_indices,
            args.candidate_limit,
        )
        public_values = model_scores(final_model, public_matrix).reshape(
            len(public.qids), args.candidate_limit
        )
        public_indices = np.arange(len(public.qids), dtype=np.int32)
        public_predictions = apply_policy(
            public, public_values, public_indices, public_matrix, feature_names,
            args.candidate_limit, policy,
        )
        public_output = args.output_dir / args.submission_name
        write_json(public_output, public_predictions)
        public_changes = sum(
            public_predictions[qid]["answer"] != public.baseline[index]
            for index, qid in enumerate(public.qids)
        )
    report = {
        "kind": "legalir_v11_tiny_ltr_complement_protocol",
        "leakage_policy": {
            "selection": "grouped OOF on complement index queries",
            "test": "original 1000-question v11 holdout after model/policy lock",
            "excluded_exact_memory_qids": excluded_exact_ids,
            "fit_exact_memory_audit_used": False,
            "public_labels_used": False,
            "v57_predictions_used": False,
            "document_prior_feature_used": False,
            "reason_document_prior_disabled": "fit-index priors contain complement labels",
        },
        "model_compliance": {
            "retrieval_model_billions": 0.33,
            "reranker_model_billions": 0.76,
            "tiny_ltr_parameters_upper_bound": 10000,
            "total_model_plan_billions_approx": 1.09,
            "under_4b": True,
        },
        "candidate_limit": args.candidate_limit,
        "feature_set": args.feature_set,
        "model_profile": args.model_profile,
        "feature_cache": {
            "path": str(args.feature_cache.resolve()) if args.feature_cache else None,
            "hit": cache_hit,
        },
        "score_cache": str(args.score_cache.resolve()) if args.score_cache else None,
        "features": feature_names,
        "selection_questions": len(complement.qids),
        "selection_groups": len(set(groups)),
        "locked_config": config,
        "locked_policy": policy,
        "complement_selection": {
            "baseline_c35": complement_baseline,
            "baseline_c35_exact_t10_c80": complement_exact,
            "oof_ltr": complement_ltr,
            "delta_vs_exact_recall": complement_delta,
        },
        "holdout_opened_once": {
            "baseline_c35": holdout_baseline,
            "baseline_c35_exact_t10_c80": holdout_exact,
            "locked_ltr": holdout_ltr,
            "delta_vs_exact_recall": holdout_delta,
        },
        "promotion": {
            "minimum_holdout_delta": args.minimum_holdout_delta,
            "promoted": promoted,
            "public_submission": str(public_output.resolve()) if public_output else None,
            "public_queries_changed_vs_c35": public_changes,
        },
        "model_leaderboard": leaderboard,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = args.output_dir / "v11_tiny_ltr_complement_report.json"
    write_json(report_path, report)
    if args.score_cache:
        args.score_cache.parent.mkdir(parents=True, exist_ok=True)
        score_metadata = {
            "version": 1,
            "feature_set": args.feature_set,
            "model_profile": args.model_profile,
            "candidate_limit": args.candidate_limit,
            "locked_config": config,
            "locked_policy": policy,
            "complement_qids": complement.qids,
            "holdout_qids": holdout.qids,
            "public_qids": public.qids,
        }
        np.savez_compressed(
            args.score_cache,
            metadata=np.asarray(json.dumps(score_metadata, ensure_ascii=False)),
            oof_scores=oof_scores,
            holdout_scores=holdout_values,
            public_scores=(
                public_values
                if public_values is not None
                else np.empty((0, args.candidate_limit), dtype=np.float32)
            ),
        )
        print(f"score cache written: {args.score_cache}", flush=True)
    print(json.dumps({
        "report": str(report_path.resolve()),
        "locked_config": config,
        "locked_policy": policy,
        "complement": report["complement_selection"],
        "holdout": report["holdout_opened_once"],
        "promotion": report["promotion"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
