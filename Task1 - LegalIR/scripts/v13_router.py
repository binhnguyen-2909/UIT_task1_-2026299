"""V13 conditional routing between the locked v11h anchor and v12.

The router is deliberately small and conditional.  It is trained with grouped
OOF predictions on the clean complement split, where both branches are
out-of-fold.  Thresholds are then locked before the original holdout is
evaluated.  The holdout cannot influence branch selection or threshold choice.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .evaluate import official_metrics
    from .v11_tiny_ltr import (
        apply_policy,
        build_leak_groups,
        load_audit_split,
        load_priors,
        read_json,
        write_json,
    )
    from .v11_tiny_ltr_complement import clean_complement
    from .v12_research import (
        ASPECT_FEATURE_NAMES,
        EVIDENCE_FEATURE_NAMES,
        TFIDF_FEATURE_NAMES,
        build_research_features,
        fit_ranker,
        fit_tfidf,
        load_chunk_evidence,
        load_documents_capped,
        make_predictions,
        make_ranker,
        parse_aspects,
        rank_scores,
    )
except ImportError:
    from evaluate import official_metrics
    from v11_tiny_ltr import apply_policy, build_leak_groups, load_audit_split, load_priors, read_json, write_json
    from v11_tiny_ltr_complement import clean_complement
    from v12_research import (
        ASPECT_FEATURE_NAMES,
        EVIDENCE_FEATURE_NAMES,
        TFIDF_FEATURE_NAMES,
        build_research_features,
        fit_ranker,
        fit_tfidf,
        load_chunk_evidence,
        load_documents_capped,
        make_predictions,
        make_ranker,
        parse_aspects,
        rank_scores,
    )


V11H_ROBUST_BASELINE = 0.9002195285172595
V11H_LOCKED_RECALL = 0.9188333333333334
ROUTER_FEATURE_NAMES = [
    "query_token_log_length", "query_number_count", "aspect_count",
    "has_negation", "has_modality", "has_condition", "v11h_margin",
    "v12_margin", "branch_top5_jaccard", "v12_unique_top5",
    "v12_aspect_max_min", "v12_aspect_cover70", "v12_aspect_cover80",
    "v12_aspect_weakest_margin", "v12_evidence_query_coverage",
    "v12_evidence_aspect_mean", "v12_evidence_aspect_min",
    "v12_tfidf_mean", "v12_tfidf_max",
]


def _references(split: Any, indices: Sequence[int]) -> dict[str, dict[str, list[str]]]:
    if split.references is None:
        raise ValueError("references unavailable")
    return {split.qids[index]: {"answer": sorted(split.references[index])} for index in indices}


def _metrics(split: Any, predictions: Mapping[str, Mapping[str, Sequence[str]]], indices: Sequence[int]) -> dict[str, Any]:
    return dict(official_metrics(dict(predictions), _references(split, indices)))


def _hit_quality(answer: Sequence[str], reference: set[str]) -> float:
    return len(set(answer) & reference) / max(len(reference), 1)


def branch_diagnostics(
    split: Any,
    v11h_predictions: Mapping[str, Mapping[str, Sequence[str]]],
    v12_predictions: Mapping[str, Mapping[str, Sequence[str]]],
    indices: Sequence[int],
) -> dict[str, Any]:
    if split.references is None:
        raise ValueError("references unavailable")
    counts = {"total": 0, "v11h_correct": 0, "v12_correct": 0, "v11h_only": 0,
              "v12_only": 0, "both_correct": 0, "both_wrong": 0}
    union_recall = 0.0
    v11h_gain = 0.0
    v12_gain = 0.0
    for index in indices:
        qid = split.qids[index]
        reference = split.references[index]
        left = set(v11h_predictions[qid]["answer"])
        right = set(v12_predictions[qid]["answer"])
        left_hit = bool(left & reference)
        right_hit = bool(right & reference)
        counts["total"] += 1
        counts["v11h_correct"] += int(left_hit)
        counts["v12_correct"] += int(right_hit)
        counts["v11h_only"] += int(left_hit and not right_hit)
        counts["v12_only"] += int(right_hit and not left_hit)
        counts["both_correct"] += int(left_hit and right_hit)
        counts["both_wrong"] += int(not left_hit and not right_hit)
        union_recall += _hit_quality(sorted(left | right), reference)
        v11h_gain += _hit_quality(v11h_predictions[qid]["answer"], reference)
        v12_gain += _hit_quality(v12_predictions[qid]["answer"], reference)
    total = max(counts["total"], 1)
    counts.update({
        "v11h_recall_unbounded": v11h_gain / total,
        "v12_recall_unbounded": v12_gain / total,
        "union_top5_plus_top5_recall": union_recall / total,
        "union_upside_vs_v11h": union_recall / total - v11h_gain / total,
        "v12_only_rate": counts["v12_only"] / total,
    })
    return counts


def _top5(score_row: np.ndarray, split: Any, index: int, limit: int) -> list[str]:
    candidates = list(dict.fromkeys(split.candidates[index][:limit]))
    order = np.argsort(-score_row, kind="stable")
    return [candidates[position] for position in order if position < len(candidates) and candidates[position]][:5]


def router_features(
    split: Any,
    v11h_scores: np.ndarray,
    v12_scores: np.ndarray,
    v12_matrix: np.ndarray,
    v12_feature_names: Sequence[str],
    candidate_limit: int,
) -> np.ndarray:
    columns = {name: index for index, name in enumerate(v12_feature_names)}
    needed = [
        "v12_aspect_max_min", "v12_aspect_cover70", "v12_aspect_cover80",
        "v12_aspect_weakest_margin", "v12_evidence_query_coverage",
        "v12_evidence_aspect_mean", "v12_evidence_aspect_min",
        "v12_tfidf_mean", "v12_tfidf_max",
    ]
    missing = [name for name in needed if name not in columns]
    if missing:
        raise ValueError(f"router feature schema missing: {missing}")
    rows: list[list[float]] = []
    for index, question in enumerate(split.questions):
        q_tokens = question.split()
        aspects = parse_aspects(question)
        v11_sorted = np.sort(v11h_scores[index])[::-1]
        v12_sorted = np.sort(v12_scores[index])[::-1]
        v11_top = set(_top5(v11h_scores[index], split, index, min(80, len(split.candidates[index]))))
        v12_top = set(_top5(v12_scores[index], split, index, candidate_limit))
        top_v12_position = int(np.argmax(v12_scores[index]))
        feature_row = v12_matrix[index * candidate_limit + top_v12_position]
        normalized = question.lower()
        rows.append([
            math.log1p(len(q_tokens)),
            float(sum(char.isdigit() for char in normalized)),
            len(aspects) / 6.0,
            float(any(value in normalized for value in ("khong", "cam", "ngoai tru", "tru truong hop"))),
            float(any(value in normalized for value in ("co duoc", "phai", "duoc phep", "can"))),
            float(any(value in normalized for value in ("neu", "khi", "trong truong hop", "truong hop"))),
            float(v11_sorted[0] - v11_sorted[min(4, len(v11_sorted) - 1)]),
            float(v12_sorted[0] - v12_sorted[min(4, len(v12_sorted) - 1)]),
            len(v11_top & v12_top) / max(len(v11_top | v12_top), 1),
            len(v12_top - v11_top) / 5.0,
            *[float(feature_row[columns[name]]) for name in needed],
        ])
    return np.asarray(rows, dtype=np.float32)


def grouped_router_oof(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[int, int]]:
    probabilities = np.full(len(target), np.nan, dtype=np.float32)
    fold_map: dict[int, int] = {}
    splitter = GroupKFold(n_splits=5)
    all_indices = np.arange(len(target), dtype=np.int32)
    for fold, (train_indices, valid_indices) in enumerate(splitter.split(all_indices, target, groups=groups)):
        model = Pipeline([
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=seed + fold)),
        ])
        model.fit(features[train_indices], target[train_indices])
        probabilities[valid_indices] = model.predict_proba(features[valid_indices])[:, 1]
        for index in valid_indices:
            fold_map[int(index)] = fold
    return probabilities, fold_map


def routed_predictions(
    split: Any,
    v11h_predictions: Mapping[str, Mapping[str, Sequence[str]]],
    v12_predictions: Mapping[str, Mapping[str, Sequence[str]]],
    probabilities: np.ndarray,
    threshold: float,
    indices: Sequence[int],
) -> dict[str, dict[str, list[str]]]:
    output: dict[str, dict[str, list[str]]] = {}
    for index in indices:
        qid = split.qids[index]
        use_v12 = float(probabilities[index]) >= threshold
        chosen = v12_predictions[qid] if use_v12 else v11h_predictions[qid]
        output[qid] = {"answer": list(chosen["answer"])}
    return output


def robust_recall(split: Any, predictions: Mapping[str, Mapping[str, Sequence[str]]], indices: Sequence[int], fold_map: Mapping[int, int]) -> dict[str, Any]:
    base = _metrics(split, predictions, indices)
    recalls = []
    for fold in sorted(set(fold_map[index] for index in indices)):
        fold_indices = [index for index in indices if fold_map[index] == fold]
        recalls.append(float(_metrics(split, {split.qids[index]: predictions[split.qids[index]] for index in fold_indices}, fold_indices)["recall"]))
    std = statistics.pstdev(recalls) if len(recalls) > 1 else 0.0
    base.update({"fold_recalls": recalls, "fold_recall_std": std, "robust_recall": float(base["recall"]) - 0.25 * std})
    return base


def load_v11h_anchor(
    split: Any,
    feature_cache_path: Path,
    score_cache_path: Path,
    policy: Mapping[str, Any],
    candidate_limit: int,
) -> tuple[dict[str, dict[str, list[str]]], np.ndarray, list[str]]:
    with np.load(feature_cache_path, allow_pickle=False) as feature_cache:
        feature_names = [str(value) for value in feature_cache["feature_names"]]
        matrix_key = "complement_matrix" if split.name == "complement" else "holdout_matrix"
        matrix = feature_cache[matrix_key].copy()
    with np.load(score_cache_path, allow_pickle=False) as score_cache:
        score_key = "oof_scores" if split.name == "complement" else "holdout_scores"
        scores = score_cache[score_key].copy()
    if len(scores) != len(split.qids):
        raise ValueError(f"v11h score cache row mismatch for {split.name}")
    predictions = apply_policy(
        split, scores, np.arange(len(split.qids), dtype=np.int32), matrix,
        feature_names, candidate_limit, policy,
    )
    return predictions, scores, feature_names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts-zip", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--complement-stage1", type=Path, required=True)
    parser.add_argument("--complement-stage2", type=Path, required=True)
    parser.add_argument("--complement-reference", type=Path, required=True)
    parser.add_argument("--holdout-stage1", type=Path, required=True)
    parser.add_argument("--holdout-stage2", type=Path, required=True)
    parser.add_argument("--holdout-reference", type=Path, required=True)
    parser.add_argument("--v11h-feature-cache", type=Path, required=True)
    parser.add_argument("--v11h-score-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-limit", type=int, default=35)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    raw_complement = load_audit_split(
        "complement", args.complement_stage1, args.complement_stage2, args.complement_reference,
    )
    complement, excluded_exact_ids = clean_complement(raw_complement, args.complement_stage1)
    holdout = load_audit_split(
        "holdout", args.holdout_stage1, args.holdout_stage2, args.holdout_reference,
    )
    v11h_policy = {"kind": "prefix", "pool": 35, "keep": 0, "exact_rescue": True}
    v11h_complement, v11h_complement_scores, _ = load_v11h_anchor(
        complement, args.v11h_feature_cache, args.v11h_score_cache, v11h_policy, 80,
    )
    v11h_holdout, v11h_holdout_scores, _ = load_v11h_anchor(
        holdout, args.v11h_feature_cache, args.v11h_score_cache, v11h_policy, 80,
    )
    needed_ids = {
        doc_id
        for split in (complement, holdout)
        for candidates in split.candidates
        for doc_id in candidates[: args.candidate_limit]
        if doc_id
    }
    needed_ids.update(
        doc_id
        for split in (complement, holdout)
        for answers in split.baseline
        for doc_id in answers
    )
    index_dir = args.index if args.index.is_dir() else args.index.parent
    priors = load_priors(index_dir / "legalir.sqlite3")
    documents, document_frequency = load_documents_capped(args.contexts_zip, needed_ids, priors)
    chunks = load_chunk_evidence(index_dir / "legalir.sqlite3", needed_ids)
    tfidf_bank = fit_tfidf(documents)

    complement_matrix, complement_target, feature_names = build_research_features(
        complement, documents, document_frequency, index_dir, args.model,
        args.candidate_limit, args.device, args.batch_size, tfidf_bank, chunks,
    )
    if complement_target is None:
        raise RuntimeError("complement target missing")
    groups = build_leak_groups(complement)
    complement_indices = np.arange(len(complement.qids), dtype=np.int32)
    v12_oof_scores = np.full((len(complement.qids), args.candidate_limit), np.nan, dtype=np.float32)
    v12_fold_map: dict[int, int] = {}
    splitter = GroupKFold(n_splits=5)
    for fold, (train_local, valid_local) in enumerate(splitter.split(complement_indices, complement_target.reshape(-1, args.candidate_limit)[:, 0], groups=groups)):
        train_questions = complement_indices[train_local]
        valid_questions = complement_indices[valid_local]
        model = fit_ranker(make_ranker(args.seed + fold), complement_matrix, complement_target, train_questions, args.candidate_limit)
        v12_oof_scores[valid_questions] = rank_scores(model, complement_matrix, valid_questions, args.candidate_limit)
        for question in valid_questions:
            v12_fold_map[int(question)] = fold
        print(f"v12 complement OOF fold {fold + 1}/5", flush=True)
    v12_complement_predictions = make_predictions(
        complement, complement_indices, v12_oof_scores, args.candidate_limit,
    )
    v11h_complement_metrics = robust_recall(complement, v11h_complement, complement_indices, v12_fold_map)
    v12_complement_metrics = robust_recall(complement, v12_complement_predictions, complement_indices, v12_fold_map)
    diagnostic_complement = branch_diagnostics(
        complement, v11h_complement, v12_complement_predictions, complement_indices,
    )
    router_x = router_features(
        complement, v11h_complement_scores[:, : args.candidate_limit], v12_oof_scores,
        complement_matrix, feature_names, args.candidate_limit,
    )
    router_target = np.asarray([
        int(_hit_quality(v12_complement_predictions[complement.qids[index]]["answer"], complement.references[index])
            > _hit_quality(v11h_complement[complement.qids[index]]["answer"], complement.references[index]))
        for index in complement_indices
    ], dtype=np.int8)
    positive_count = int(router_target.sum())
    router_gate = {"positive_v12_better_queries": positive_count, "total_queries": len(router_target)}
    router_oof_metrics: dict[str, Any] | None = None
    selected_threshold: float | None = None
    router_model: Pipeline | None = None
    holdout_metrics: dict[str, Any] | None = None
    holdout_diagnostic: dict[str, Any] | None = None
    if 0 < positive_count < len(router_target):
        router_probabilities, router_fold_map = grouped_router_oof(router_x, router_target, groups, args.seed + 500)
        threshold_rows = []
        for threshold in np.linspace(0.35, 0.95, 25):
            predictions = routed_predictions(
                complement, v11h_complement, v12_complement_predictions,
                router_probabilities, float(threshold), complement_indices,
            )
            evaluation = robust_recall(complement, predictions, complement_indices, router_fold_map)
            switched = int(np.sum(router_probabilities >= threshold))
            threshold_rows.append((evaluation["robust_recall"], evaluation["recall"], evaluation["precision"], -switched, float(threshold), evaluation))
        threshold_rows.sort(reverse=True)
        selected_threshold = float(threshold_rows[0][4])
        router_oof_metrics = dict(threshold_rows[0][5])
        router_oof_metrics["selected_threshold"] = selected_threshold
        router_oof_metrics["threshold_leaderboard"] = [
            {"threshold": row[4], "evaluation": row[5], "switched": -row[3]}
            for row in threshold_rows[:10]
        ]
        router_gate.update({
            "v11h_robust_baseline": V11H_ROBUST_BASELINE,
            "router_oof_robust": router_oof_metrics["robust_recall"],
            "robust_oof_wins_v11h": bool(router_oof_metrics["robust_recall"] > V11H_ROBUST_BASELINE),
        })
        if router_gate["robust_oof_wins_v11h"]:
            router_model = Pipeline([
                ("scale", StandardScaler()),
                ("logistic", LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=args.seed + 700)),
            ])
            router_model.fit(router_x, router_target)
            holdout_matrix, holdout_target, holdout_feature_names = build_research_features(
                holdout, documents, document_frequency, index_dir, args.model,
                args.candidate_limit, args.device, args.batch_size, tfidf_bank, chunks,
            )
            if holdout_feature_names != feature_names:
                raise RuntimeError("holdout v12 feature schema mismatch")
            holdout_model = fit_ranker(make_ranker(args.seed + 900), complement_matrix, complement_target, complement_indices, args.candidate_limit)
            v12_holdout_scores = rank_scores(holdout_model, holdout_matrix, np.arange(len(holdout.qids)), args.candidate_limit)
            v12_holdout_predictions = make_predictions(
                holdout, np.arange(len(holdout.qids)), v12_holdout_scores, args.candidate_limit,
            )
            holdout_x = router_features(
                holdout, v11h_holdout_scores[:, : args.candidate_limit], v12_holdout_scores,
                holdout_matrix, feature_names, args.candidate_limit,
            )
            holdout_probabilities = router_model.predict_proba(holdout_x)[:, 1]
            holdout_indices = np.arange(len(holdout.qids), dtype=np.int32)
            routed_holdout = routed_predictions(
                holdout, v11h_holdout, v12_holdout_predictions,
                np.asarray(holdout_probabilities), selected_threshold, holdout_indices,
            )
            holdout_metrics = {
                "v11h": _metrics(holdout, v11h_holdout, holdout_indices),
                "v12": _metrics(holdout, v12_holdout_predictions, holdout_indices),
                "v13_router": _metrics(holdout, routed_holdout, holdout_indices),
            }
            holdout_diagnostic = branch_diagnostics(
                holdout, v11h_holdout, v12_holdout_predictions, holdout_indices,
            )
            router_gate["holdout_recall"] = holdout_metrics["v13_router"]["recall"]
            router_gate["holdout_reaches_v11h"] = bool(holdout_gate := holdout_metrics["v13_router"]["recall"] >= V11H_LOCKED_RECALL)

    report = {
        "kind": "legalir_v13_conditional_v11h_v12_router",
        "candidate_limit": args.candidate_limit,
        "excluded_exact_memory_qids": excluded_exact_ids,
        "leakage_policy": {
            "router_training": "grouped OOF on clean complement only",
            "threshold_selection": "complement grouped OOF only",
            "holdout_used_for_selection": False,
            "public_labels_used": False,
            "v57_predictions_used": False,
        },
        "branches": {
            "v11h": {"policy": v11h_policy, "locked_recall": V11H_LOCKED_RECALL},
            "v12": {"features": feature_names},
        },
        "diagnostic_complement": diagnostic_complement,
        "v11h_complement_oof": v11h_complement_metrics,
        "v12_complement_oof": v12_complement_metrics,
        "router_gate": router_gate,
        "router_oof": router_oof_metrics,
        "diagnostic_holdout": holdout_diagnostic,
        "holdout": holdout_metrics,
        "fine_tune": {
            "started": False,
            "allowed_only_if_v13_holdout_recall_at_least": V11H_LOCKED_RECALL,
            "allowed": bool(holdout_metrics and holdout_metrics["v13_router"]["recall"] >= V11H_LOCKED_RECALL),
            "reason": "V13 router gate did not reach v11h" if not holdout_metrics or holdout_metrics["v13_router"]["recall"] < V11H_LOCKED_RECALL else "gate reached; separate explicit reranker fine-tune step required",
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = args.output_dir / "v13_router_report.json"
    write_json(report_path, report)
    print(json.dumps({
        "report": str(report_path.resolve()),
        "diagnostic_complement": diagnostic_complement,
        "v11h_complement_oof": v11h_complement_metrics,
        "v12_complement_oof": v12_complement_metrics,
        "router_gate": router_gate,
        "holdout": holdout_metrics,
        "fine_tune": report["fine_tune"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
