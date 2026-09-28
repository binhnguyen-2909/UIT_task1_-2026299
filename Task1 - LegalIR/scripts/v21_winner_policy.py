"""Evaluate V21-A's P3-GX + CATER-R policy with winner-level calibration.

The old P3 result used CE adapters whose hash folds were not isolated for the
four deployment regimes.  This evaluator consumes only the four replacement
P3-GX score matrices.  It then fits one tiny two-head utility model per regime
on the corresponding outer training groups, selects the challenger *before*
using any outcome label, and calibrates its gate on those selected winners.

There are no candidate-pair thresholds in this file.  Each deployment gate is
cross-fitted from complete consensus winners and the most conservative of the
four independently nested thresholds is required to swap V16b's slot 5.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


REGIMES = (
    "standard_grouped",
    "law_family_ood",
    "query_template_ood",
    "near_duplicate_ood",
)
MANIFEST_KIND = "legalir_v21_group_crossfit_manifest"
P3_KIND = "legalir_v21_p3_gx_group_crossfit"
KIND = "legalir_v21_p3_gx_winner_calibrated_policy"
FORBIDDEN_TERMS = ("holdout", "public", "locked", "submission")
SEED = 20_260_832


def _safe_clean_path(path: Path, label: str) -> None:
    normalized = str(path.resolve()).replace("\\", "/").casefold()
    if any(term in normalized for term in FORBIDDEN_TERMS):
        raise ValueError(f"{label} must be clean-complement-only: {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _safe_clean_path(path, label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _load_module(path: Path, module_name: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    active_scripts = Path(__file__).resolve().parent
    legacy_scripts = path.resolve().parent
    for candidate in (str(active_scripts), str(legacy_scripts)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def _load_manifest(path: Path, expected_metadata_sha256: str, qids: set[str]) -> dict[str, Any]:
    payload = _read_json(path, "group cross-fit manifest")
    if payload.get("kind") != MANIFEST_KIND:
        raise ValueError("not a V21 group cross-fit manifest")
    if payload.get("split") != "clean_complement_only" or payload.get("holdout_opened") is not False or payload.get("public_labels_used") is not False:
        raise ValueError("fold manifest does not attest unopened clean-complement execution")
    if payload.get("metadata_sha256") != expected_metadata_sha256:
        raise ValueError("fold manifest metadata hash differs from evaluator metadata")
    if set(str(value) for value in payload.get("qids", [])) != qids:
        raise ValueError("fold manifest query ids differ from metadata")
    if payload.get("all_registered_regimes_zero_overlap") is not True:
        raise ValueError("fold manifest did not pass its zero-overlap audit")
    return payload


def _fold_and_groups(manifest: Mapping[str, Any], regime: str, qids: set[str]) -> tuple[dict[str, int], dict[str, str], list[dict[str, Any]]]:
    raw = manifest.get("regimes", {}).get(regime)
    if not isinstance(raw, Mapping):
        raise ValueError(f"fold manifest lacks regime {regime}")
    assignments_raw = raw.get("group_assignments")
    folds_raw = raw.get("folds")
    if not isinstance(assignments_raw, Mapping) or not isinstance(folds_raw, list):
        raise ValueError(f"{regime}: malformed group assignments or folds")
    assignments = {str(qid): str(group) for qid, group in assignments_raw.items()}
    if set(assignments) != qids:
        raise ValueError(f"{regime}: group assignments do not cover metadata qids")
    fold_by_qid: dict[str, int] = {}
    normalized_folds: list[dict[str, Any]] = []
    for item in folds_raw:
        if not isinstance(item, Mapping):
            raise ValueError(f"{regime}: malformed fold")
        fold = int(item.get("fold", -1))
        train_qids = {str(value) for value in item.get("train_qids", [])}
        validation_qids = {str(value) for value in item.get("validation_qids", [])}
        if fold < 0 or not validation_qids or train_qids & validation_qids:
            raise ValueError(f"{regime}: malformed query partition for fold {fold}")
        if set(assignments[qid] for qid in train_qids) & set(assignments[qid] for qid in validation_qids):
            raise ValueError(f"{regime}: group leakage in fold {fold}")
        for qid in validation_qids:
            if qid in fold_by_qid:
                raise ValueError(f"{regime}: duplicate validation assignment for {qid}")
            fold_by_qid[qid] = fold
        normalized_folds.append({"fold": fold, "train_qids": train_qids, "validation_qids": validation_qids})
    if set(fold_by_qid) != qids:
        raise ValueError(f"{regime}: not every qid has exactly one validation fold")
    return fold_by_qid, assignments, sorted(normalized_folds, key=lambda item: item["fold"])


def _load_p3_scores(
    report_path: Path, questions: Sequence[Mapping[str, Any]], legacy: Any, *, metadata_sha256: str, fold_manifest_sha256: str,
) -> dict[str, dict[tuple[str, str], float]]:
    report = _read_json(report_path, "P3-GX report")
    if report.get("kind") != P3_KIND or report.get("split") != "clean_complement_only":
        raise ValueError("not a completed V21 P3-GX report")
    if report.get("holdout_opened") is not False or report.get("public_labels_used") is not False:
        raise ValueError("P3-GX report is not clean-complement-only")
    inputs = report.get("inputs")
    if not isinstance(inputs, Mapping) or inputs.get("metadata_sha256") != metadata_sha256 or inputs.get("fold_manifest_sha256") != fold_manifest_sha256:
        raise ValueError("P3-GX report provenance does not match this evaluation")
    regimes = report.get("regimes")
    if not isinstance(regimes, Mapping):
        raise ValueError("P3-GX report lacks regime outputs")
    result: dict[str, dict[tuple[str, str], float]] = {}
    for regime in REGIMES:
        item = regimes.get(regime)
        if not isinstance(item, Mapping):
            raise ValueError(f"P3-GX report lacks {regime}")
        score_path_raw = item.get("score_path")
        if not isinstance(score_path_raw, str) or not score_path_raw:
            raise ValueError(f"P3-GX {regime} has no completed OOF score matrix")
        score_path = Path(score_path_raw)
        _safe_clean_path(score_path, f"P3-GX {regime} scores")
        if not score_path.is_file():
            raise FileNotFoundError(score_path)
        expected_hash = item.get("score_sha256")
        if not isinstance(expected_hash, str) or expected_hash != _sha256_file(score_path):
            raise ValueError(f"P3-GX {regime} score hash does not match report")
        result[regime] = legacy._load_current(score_path, questions)
    return result


def _matrix(rows: Sequence[Any]) -> tuple[Any, list[str]]:
    import numpy as np

    if not rows:
        raise ValueError("cannot train a policy with no challenger rows")
    feature_names = sorted(rows[0].features)
    if any(set(row.features) != set(feature_names) for row in rows):
        raise ValueError("inconsistent CATER feature schema")
    return np.asarray([[float(row.features[name]) for name in feature_names] for row in rows], dtype=np.float64), feature_names


def _score_regime(
    rows: Sequence[Any], matrix: Any, labels: Sequence[int], fold_by_qid: Mapping[str, int], groups: Mapping[str, str], legacy: Any, *, seed: int,
) -> tuple[list[float], list[float], list[float], dict[str, Any]]:
    import numpy as np

    qids = [str(row.qid) for row in rows]
    folds = sorted(set(fold_by_qid.values()))
    utilities = np.full(len(rows), np.nan, dtype=np.float64)
    rescue_probs = np.full(len(rows), np.nan, dtype=np.float64)
    destructive_probs = np.full(len(rows), np.nan, dtype=np.float64)
    report: dict[str, Any] = {"outer_folds": {}, "calibrated_outer_models": 0}
    for fold in folds:
        validation = np.asarray([index for index, qid in enumerate(qids) if fold_by_qid[qid] == fold], dtype=np.int64)
        training = np.asarray([index for index, qid in enumerate(qids) if fold_by_qid[qid] != fold], dtype=np.int64)
        if not len(training) or not len(validation):
            raise ValueError(f"empty P3-GX outer partition at fold {fold}")
        train_groups = {groups[qids[index]] for index in training}
        validation_groups = {groups[qids[index]] for index in validation}
        if train_groups & validation_groups:
            raise ValueError(f"CATER group leakage in fold {fold}")
        model = legacy._fit_utility_model(
            matrix[training], [int(labels[index]) for index in training],
            [groups[qids[index]] for index in training], seed=seed + fold,
        )
        utility, rescue, destructive = model.predict(matrix[validation])
        utilities[validation] = utility
        rescue_probs[validation] = rescue
        destructive_probs[validation] = destructive
        calibrated = bool(model.rescue.calibrator is not None or model.destructive.calibrator is not None)
        report["calibrated_outer_models"] += int(calibrated)
        report["outer_folds"][str(fold)] = {
            "train_qids": len(set(qids[index] for index in training)),
            "validation_qids": len(set(qids[index] for index in validation)),
            "train_groups": len(train_groups),
            "validation_groups": len(validation_groups),
            "group_overlap": False,
            "binary_calibrator_present": calibrated,
        }
    if not np.isfinite(utilities).all() or not np.isfinite(rescue_probs).all() or not np.isfinite(destructive_probs).all():
        raise RuntimeError("P3-GX CATER OOF scoring did not cover every candidate")
    return utilities.tolist(), rescue_probs.tolist(), destructive_probs.tolist(), report


def _outcome_label(question: Mapping[str, Any], candidate_doc_id: str) -> int:
    references = {str(value) for value in question["references"]}
    tail = str(question["anchor_answer"][-1])
    if candidate_doc_id in references and tail not in references:
        return 1
    if tail in references and candidate_doc_id not in references:
        return 0
    return -1


def _consensus_winners(
    questions: Sequence[Mapping[str, Any]], rows_by_regime: Mapping[str, Sequence[Any]], utilities: Mapping[str, Sequence[float]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    winners: dict[str, dict[str, Any]] = {}
    traces: list[dict[str, Any]] = []
    by_regime_qid: dict[str, dict[str, list[int]]] = {}
    for regime in REGIMES:
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows_by_regime[regime]):
            grouped[str(row.qid)].append(index)
        by_regime_qid[regime] = grouped
    for question in questions:
        qid = str(question["qid"])
        reference_order = by_regime_qid[REGIMES[0]][qid]
        candidate_ids = [str(rows_by_regime[REGIMES[0]][index].candidate_doc_id) for index in reference_order]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"{qid}: duplicate CATER challenger row")
        rows_by_candidate: dict[str, dict[str, Any]] = {}
        per_head_winner: dict[str, str] = {}
        for regime in REGIMES:
            local_indices = by_regime_qid[regime].get(qid, [])
            local_docs = [str(rows_by_regime[regime][index].candidate_doc_id) for index in local_indices]
            if local_docs != candidate_ids:
                raise ValueError(f"{qid}: challenger order drift across {regime}")
            head_index = max(
                local_indices,
                key=lambda index: (float(utilities[regime][index]), -int(rows_by_regime[regime][index].candidate_rank), str(rows_by_regime[regime][index].candidate_doc_id)),
            )
            per_head_winner[regime] = str(rows_by_regime[regime][head_index].candidate_doc_id)
        for candidate in candidate_ids:
            values: dict[str, float] = {}
            rank: int | None = None
            for regime in REGIMES:
                index = next(index for index in by_regime_qid[regime][qid] if str(rows_by_regime[regime][index].candidate_doc_id) == candidate)
                values[regime] = float(utilities[regime][index])
                rank = int(rows_by_regime[regime][index].candidate_rank)
            ordered = [values[regime] for regime in REGIMES]
            median = float(statistics.median(ordered))
            mad = float(statistics.median(abs(value - median) for value in ordered))
            rows_by_candidate[candidate] = {
                "candidate_doc_id": candidate,
                "candidate_rank": int(rank if rank is not None else 999),
                "utilities": values,
                "winner_count": sum(doc_id == candidate for doc_id in per_head_winner.values()),
                "median_utility": median,
                "mad": mad,
                "lcb_utility": median - mad,
                "all_four_positive": all(value > 0.0 for value in ordered),
            }
        selected = max(
            rows_by_candidate.values(),
            key=lambda item: (
                int(item["winner_count"]), float(item["median_utility"]), float(item["lcb_utility"]),
                -int(item["candidate_rank"]), str(item["candidate_doc_id"]),
            ),
        )
        selected = {
            **selected,
            "qid": qid,
            "outcome_label": _outcome_label(question, str(selected["candidate_doc_id"])),
            "consensus_eligible": bool(selected["winner_count"] >= 3 and selected["all_four_positive"]),
            "head_winners": per_head_winner,
        }
        winners[qid] = selected
        traces.extend({"qid": qid, "selected": candidate == selected["candidate_doc_id"], **item} for candidate, item in rows_by_candidate.items())
    return winners, traces


def _select_winner_threshold(winners: Sequence[Mapping[str, Any]]) -> tuple[float, dict[str, Any]]:
    values = sorted({float(item["lcb_utility"]) for item in winners if bool(item["consensus_eligible"]) and math.isfinite(float(item["lcb_utility"])) and float(item["lcb_utility"]) > 0.0})
    thresholds = [0.0, *values, float("inf")]
    best_key = (float("-inf"), float("-inf"), float("-inf"), float("-inf"))
    best_threshold = float("inf")
    best_detail: dict[str, Any] = {}
    for threshold in thresholds:
        accepted = [item for item in winners if bool(item["consensus_eligible"]) and float(item["lcb_utility"]) >= threshold]
        rescues = sum(int(item["outcome_label"]) == 1 for item in accepted)
        destructive = sum(int(item["outcome_label"]) == 0 for item in accepted)
        neutral = len(accepted) - rescues - destructive
        net_utility = float(rescues - 3 * destructive)
        precision = rescues / (rescues + destructive) if rescues + destructive else 0.0
        key = (net_utility, precision, -float(destructive), float(threshold))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_detail = {
                "candidate_winners": len(winners), "eligible_winners": sum(bool(item["consensus_eligible"]) for item in winners),
                "accepted_winners": len(accepted), "rescues": rescues, "destructive": destructive,
                "neutral": neutral, "net_utility": net_utility, "intervention_precision": precision,
            }
    return best_threshold, best_detail


def _nested_winner_thresholds(
    winners: Mapping[str, Mapping[str, Any]], manifest: Mapping[str, Any], qids: set[str],
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    thresholds: dict[str, dict[str, float]] = {}
    audit: dict[str, Any] = {}
    for regime in REGIMES:
        _, assignments, folds = _fold_and_groups(manifest, regime, qids)
        per_qid: dict[str, float] = {}
        fold_audit: dict[str, Any] = {}
        for item in folds:
            train_qids = set(item["train_qids"])
            validation_qids = set(item["validation_qids"])
            if set(assignments[qid] for qid in train_qids) & set(assignments[qid] for qid in validation_qids):
                raise ValueError(f"{regime}: nested threshold group overlap in fold {item['fold']}")
            threshold, detail = _select_winner_threshold([winners[qid] for qid in sorted(train_qids)])
            per_qid.update({qid: threshold for qid in validation_qids})
            fold_audit[str(item["fold"])] = {
                "train_qids": len(train_qids), "validation_qids": len(validation_qids),
                "threshold": threshold if math.isfinite(threshold) else None,
                "explicit_abstain": not math.isfinite(threshold), "group_overlap": False, **detail,
            }
        if set(per_qid) != qids:
            raise ValueError(f"{regime}: nested threshold did not cover every qid")
        thresholds[regime] = per_qid
        audit[regime] = fold_audit
    return thresholds, audit


def _predictions(
    questions: Sequence[Mapping[str, Any]], winners: Mapping[str, Mapping[str, Any]], thresholds: Mapping[str, Mapping[str, float]],
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    predictions: dict[str, list[str]] = {}
    traces: list[dict[str, Any]] = []
    for question in questions:
        qid = str(question["qid"])
        winner = winners[qid]
        values = [float(thresholds[regime][qid]) for regime in REGIMES]
        nested_threshold = max(values)
        accepted = bool(winner["consensus_eligible"]) and float(winner["lcb_utility"]) >= nested_threshold
        anchor = [str(value) for value in question["anchor_answer"]]
        predictions[qid] = [*anchor[:4], str(winner["candidate_doc_id"])] if accepted else anchor
        traces.append({
            **winner,
            "nested_thresholds": {regime: value if math.isfinite(value) else None for regime, value in zip(REGIMES, values, strict=True)},
            "nested_threshold": nested_threshold if math.isfinite(nested_threshold) else None,
            "accepted": accepted,
            "top_four_frozen": True,
        })
    return predictions, traces


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    for path, label in (
        (args.metadata, "metadata"), (args.block_features, "block features"), (args.pack_scores, "E3B scores"),
        (args.fold_manifest, "fold manifest"), (args.p3_report, "P3-GX report"), (args.legacy_cater, "legacy CATER helpers"),
        (args.output, "output"),
    ):
        _safe_clean_path(path, label)
    if args.output.exists() and not args.force:
        raise FileExistsError(args.output)
    legacy = _load_module(args.legacy_cater, "v21_legacy_cater")
    metadata_payload = _read_json(args.metadata, "metadata")
    questions = legacy._metadata(metadata_payload)
    qids = {str(question["qid"]) for question in questions}
    metadata_hash = _sha256_file(args.metadata)
    manifest = _load_manifest(args.fold_manifest, metadata_hash, qids)
    manifest_hash = _sha256_file(args.fold_manifest)

    # Build all label-free rows before constructing rescue/destructive labels.
    blocks = legacy._load_blocks(args.block_features, questions)
    e3b_scores = legacy._load_scores(args.pack_scores, questions)
    adapted_scores = _load_p3_scores(args.p3_report, questions, legacy, metadata_sha256=metadata_hash, fold_manifest_sha256=manifest_hash)
    rows_by_regime = {regime: legacy._build_pair_rows(questions, blocks, e3b_scores, adapted_scores[regime], "adaptive_budget") for regime in REGIMES}
    matrices: dict[str, Any] = {}
    feature_names: dict[str, list[str]] = {}
    labels: list[int] | None = None
    for regime in REGIMES:
        matrices[regime], feature_names[regime] = _matrix(rows_by_regime[regime])
        candidate_labels = legacy._labels(rows_by_regime[regime], questions)
        if labels is None:
            labels = candidate_labels
        elif candidate_labels != labels:
            raise ValueError("outcome labels drift across identical challenger rows")
    if labels is None:
        raise RuntimeError("missing CATER labels")

    utilities: dict[str, list[float]] = {}
    rescue_probs: dict[str, list[float]] = {}
    destructive_probs: dict[str, list[float]] = {}
    crossfit: dict[str, Any] = {}
    group_maps: dict[str, dict[str, str]] = {}
    for offset, regime in enumerate(REGIMES):
        fold_by_qid, group_maps[regime], _ = _fold_and_groups(manifest, regime, qids)
        utility, rescue, destructive, audit = _score_regime(
            rows_by_regime[regime], matrices[regime], labels, fold_by_qid, group_maps[regime], legacy, seed=SEED + offset * 97,
        )
        utilities[regime], rescue_probs[regime], destructive_probs[regime], crossfit[regime] = utility, rescue, destructive, audit

    winners, candidate_traces = _consensus_winners(questions, rows_by_regime, utilities)
    thresholds, threshold_audit = _nested_winner_thresholds(winners, manifest, qids)
    predictions, winner_traces = _predictions(questions, winners, thresholds)
    baseline = {str(question["qid"]): [str(value) for value in question["anchor_answer"]] for question in questions}
    primary = legacy._regime_metrics(questions, predictions, baseline, group_maps["standard_grouped"])
    all_regimes = {regime: legacy._regime_metrics(questions, predictions, baseline, group_maps[regime]) for regime in REGIMES}
    bootstrap = legacy._bootstrap(predictions, baseline, questions, samples=args.bootstrap_samples, seed=SEED)
    top_four_frozen = all(predictions[str(question["qid"])][:4] == list(question["anchor_answer"][:4]) for question in questions)
    max_one_swap = all(
        prediction == baseline[qid] or prediction[:4] == baseline[qid][:4]
        for qid, prediction in predictions.items()
    )
    promotion = legacy._promotion(
        primary, bootstrap, all_regimes, top_four_frozen=top_four_frozen, max_one_swap=max_one_swap, data_safe=True,
    )
    finite_thresholds = {
        regime: sorted({float(value) for value in values.values() if math.isfinite(float(value))})
        for regime, values in thresholds.items()
    }
    report = {
        "kind": KIND, "schema_version": 1, "stage": "V21_A_complete_clean_complement",
        "split": "clean_complement_only", "holdout_opened": False, "public_labels_used": False,
        "selection_performed": False,
        "recipe": {
            "candidate_scope": "immutable V16b top-4 and slot-5 incumbent; rank-6-to-12 challengers only",
            "block_packs": "exact V18b query-conditioned BQB/BM25/MMR six configurations",
            "utility": "P(rescue) - 3 * P(destructive)",
            "calibration": "complete consensus challenger is selected before every nested OOF threshold is fit",
            "acceptance": "at least 3/4 challenger consensus, all four utilities positive, LCB utility above max of four nested thresholds",
        },
        "p3_gx": {"report": str(args.p3_report.resolve()), "report_sha256": _sha256_file(args.p3_report)},
        "features": {"variant": "adaptive_budget", "names": feature_names, "candidate_rows": len(labels)},
        "cross_fit": crossfit,
        "winner_level_thresholds": {"finite_values": finite_thresholds, "abstain_counts": {regime: sum(not math.isfinite(value) for value in values.values()) for regime, values in thresholds.items()}, "nested_audit": threshold_audit},
        "metrics_vs_v16b_anchor": primary, "regimes": all_regimes, "paired_bootstrap_vs_v16b": bootstrap,
        "promotion": promotion,
        "trace_summary": {
            "candidate_rows": len(candidate_traces), "winner_rows": len(winner_traces),
            "consensus_eligible_winners": sum(bool(item["consensus_eligible"]) for item in winners.values()),
            "accepted_queries": sum(bool(item["accepted"]) for item in winner_traces),
        },
        "winner_traces": winner_traces,
        "parameter_ledger": {
            "base_pipeline_parameters": 1_090_010_000, "shared_deployed_lora_upper_bound": 20_000_000,
            "four_tiny_heads_upper_bound": 1_000_000, "end_to_end_upper_bound": 1_111_010_000, "under_4b": True,
        },
        "provenance": {
            "metadata": str(args.metadata.resolve()), "metadata_sha256": metadata_hash,
            "block_features": str(args.block_features.resolve()), "block_features_sha256": _sha256_file(args.block_features),
            "pack_scores": str(args.pack_scores.resolve()), "pack_scores_sha256": _sha256_file(args.pack_scores),
            "fold_manifest": str(args.fold_manifest.resolve()), "fold_manifest_sha256": manifest_hash,
            "legacy_cater": str(args.legacy_cater.resolve()), "legacy_cater_sha256": _sha256_file(args.legacy_cater),
            "labels_used_only_after_label_free_feature_construction": True,
            "holdout_opened": False, "public_labels_used": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--block-features", type=Path, required=True)
    parser.add_argument("--pack-scores", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--p3-report", type=Path, required=True)
    parser.add_argument("--legacy-cater", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate(args)
    print(json.dumps({
        "output": str(args.output.resolve()), "promoted": report["promotion"]["promoted"],
        "recall_gain": report["metrics_vs_v16b_anchor"]["recall_gain"],
        "bootstrap_lower_95": report["paired_bootstrap_vs_v16b"]["lower_95"],
        "holdout_opened": False,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
