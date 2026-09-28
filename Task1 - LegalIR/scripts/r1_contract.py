#!/usr/bin/env python3
"""Contracts and frozen-data adapters for R1_C35_LONG_FUSION.

R1 consumes the two-seed, fold-safe E3B full-c35 score artifacts and a
label-blind Qwen long-score cache.  It never trains a model and it never
opens holdout/public labels.  The module is intentionally explicit about the
E3B artifact shape so final deployment weights cannot be silently substituted
for fold C/V scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e3b_residual_score_gate as e3b_gate  # noqa: E402
from qwen_c200_common import (  # noqa: E402
    INSTRUCTION, load_candidate_universe, pool_for, require_contexts_zip,
)
from v27_a_common import (  # noqa: E402
    CANDIDATE_LIMIT, V27ContractError, json_sha256, load_manifest, read_json,
    sha256_file, write_json,
)


R1_KIND = "legalir_r1_c35_long_fusion"
R1_SCHEMA_VERSION = 1
R1_VARIANT = "R1_C35_LONG"
QWEN_MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
QWEN_MAX_TOKENS = 4096
R1_LAMBDAS = (0.0, 0.25, 0.50)
R1_BASELINE_THRESHOLD = 0.50
R1_DISAGREEMENT_WEIGHT = 0.25
# Frozen upstream footprint from the locked V16b + E3B path. R1 adds only
# the frozen Qwen reranker; the runtime records the actual loaded count.
FROZEN_UPSTREAM_PARAMETERS = 2_610_010_000
PARAMETER_LIMIT = 4_000_000_000
EPSILON = 1.0e-12


@dataclass(frozen=True)
class E3BFoldScores:
    regime: str
    inner_seed: int
    outer_fold: int
    calibration_qids: tuple[str, ...]
    validation_qids: tuple[str, ...]
    calibration_by_seed: tuple[dict[str, dict[str, dict[str, Any]]], ...]
    validation_by_seed: tuple[dict[str, dict[str, dict[str, Any]]], ...]
    source_artifacts: tuple[dict[str, Any], ...]
    report_lineage_sha256: str


def require_complement_file(path: Path, label: str) -> None:
    normalized = str(path).replace("\\", "/").casefold()
    if any(term in normalized for term in ("holdout", "submission", "locked")):
        raise V27ContractError(f"R1 is complement-only; {label} points at sealed/output data: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


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
                raise V27ContractError(f"{path}:{number}: expected an object")
            rows.append(row)
    return rows


def qid_hash(qids: Sequence[str]) -> str:
    return json_sha256({"qids": [str(value) for value in qids]})


def _subset_qids(split: Any, qids: Sequence[str]) -> Any:
    from v27_a_common import V27Split

    positions = {str(qid): index for index, qid in enumerate(split.qids)}
    missing = [str(qid) for qid in qids if str(qid) not in positions]
    if missing:
        raise V27ContractError(f"R1 score artifact contains unknown QIDs: {missing[:5]}")
    indices = [positions[str(qid)] for qid in qids]
    return V27Split(
        split.name, list(qids), [split.questions[index] for index in indices],
        [split.candidates[index] for index in indices], [split.anchor[index] for index in indices],
        [split.neural_scores[index] for index in indices], [split.result_rows[index] for index in indices],
        [split.references[index] for index in indices] if split.references is not None else None,
        [split.source_candidates[index] for index in indices], [split.source_results[index] for index in indices],
    )


def _artifact_rows(path: Path, *, c35_by_qid: Mapping[str, Sequence[str]], qids: Sequence[str]) -> dict[str, dict[str, dict[str, Any]]]:
    rows = _read_jsonl(path)
    # Importing this validator keeps empty-body masks and the exact document
    # score semantics identical to E3B.
    parsed = e3b_gate.validate_full_c35_scores(rows, c35_by_qid=c35_by_qid, expected_qids=qids)
    return parsed


def _load_report(path: Path, *, expected_mode: str) -> dict[str, Any]:
    require_complement_file(path, f"E3B {expected_mode} report")
    report = read_json(path)
    if not isinstance(report, dict) or report.get("kind") != "legalir_e3b_nested_mil_full_c35_oof":
        raise V27ContractError(f"Unexpected E3B report kind: {path}")
    if report.get("mode") != expected_mode or report.get("split") != "complement":
        raise V27ContractError(f"E3B report is not the requested complement {expected_mode} report: {path}")
    if report.get("holdout_opened") is not False or report.get("public_labels_used") is not False:
        raise V27ContractError("R1 refuses an E3B prerequisite that opened holdout/public labels")
    return report


def load_e3b_fold_scores(*, report_path: Path, expected_mode: str, split: Any) -> tuple[dict[str, list[E3BFoldScores]], dict[str, Any]]:
    report = _load_report(report_path, expected_mode=expected_mode)
    regimes = report.get("regimes")
    if not isinstance(regimes, Mapping):
        raise V27ContractError("E3B report has no regimes")
    c35_by_qid = {str(qid): [str(value) for value in split.candidates[index][:CANDIDATE_LIMIT]] for index, qid in enumerate(split.qids)}
    output: dict[str, list[E3BFoldScores]] = {}
    for regime, runs in regimes.items():
        if not isinstance(runs, list) or len(runs) != 2:
            raise V27ContractError(f"E3B {regime}: expected exactly two inner-seed runs")
        fold_rows: list[E3BFoldScores] = []
        for run in runs:
            if not isinstance(run, Mapping) or not isinstance(run.get("folds"), list) or len(run["folds"]) != 5:
                raise V27ContractError(f"E3B {regime}: run lacks five fold reports")
            inner_seed = int(run.get("inner_seed"))
            for fold_index, fold in enumerate(run["folds"]):
                if not isinstance(fold, Mapping):
                    raise V27ContractError("E3B fold report is not an object")
                partitions = fold.get("partitions")
                if not isinstance(partitions, Mapping):
                    raise V27ContractError("E3B fold report has no partition manifest")
                c_qids = tuple(str(value) for value in partitions.get("calibration", {}).get("qids", []))
                v_qids = tuple(str(value) for value in partitions.get("validation", {}).get("qids", []))
                if not c_qids or not v_qids or set(c_qids) & set(v_qids):
                    raise V27ContractError(f"E3B {regime}/seed={inner_seed}/fold={fold_index}: invalid C/V QID boundary")
                artifacts = fold.get("score_artifacts")
                if not isinstance(artifacts, list) or len(artifacts) != 4:
                    raise V27ContractError(f"E3B {regime}/seed={inner_seed}/fold={fold_index}: expected two-seed C/V artifacts")
                by_seed: dict[int, dict[str, dict[str, dict[str, Any]]]] = {}
                artifact_records: list[dict[str, Any]] = []
                for raw in artifacts:
                    if not isinstance(raw, Mapping):
                        raise V27ContractError("E3B score artifact record is not an object")
                    score_path = Path(str(raw.get("score_path", "")))
                    meta_path = Path(str(raw.get("metadata_path", "")))
                    require_complement_file(score_path, "E3B score cache")
                    require_complement_file(meta_path, "E3B score metadata")
                    if raw.get("score_sha256") != sha256_file(score_path) or raw.get("metadata_sha256") != sha256_file(meta_path):
                        raise V27ContractError(f"E3B score artifact hash changed: {score_path}")
                    metadata = read_json(meta_path)
                    if not isinstance(metadata, Mapping):
                        raise V27ContractError(f"Invalid E3B score metadata: {meta_path}")
                    if metadata.get("lineage_contract_sha256") != report.get("lineage_contract_sha256"):
                        raise V27ContractError("E3B score artifact lineage differs from its report")
                    partition = str(metadata.get("partition", raw.get("partition", "")))
                    neural_seed = int(metadata.get("neural_seed", raw.get("neural_seed")))
                    if partition not in {"calibration", "validation"}:
                        raise V27ContractError(f"Unknown E3B score partition: {partition}")
                    qids = c_qids if partition == "calibration" else v_qids
                    parsed = _artifact_rows(score_path, c35_by_qid=c35_by_qid, qids=qids)
                    if neural_seed in by_seed and partition in by_seed[neural_seed]:
                        raise V27ContractError("Duplicate E3B seed/partition artifact")
                    by_seed.setdefault(neural_seed, {})[partition] = parsed
                    artifact_records.append({**dict(raw), "metadata": dict(metadata)})
                if set(by_seed) != {int(value.get("neural_seed")) for value in fold.get("neural_models", []) if isinstance(value, Mapping)}:
                    # Some historical reports omit neural_models in a compact
                    # report; the artifacts remain the source of truth then.
                    if len(by_seed) != 2:
                        raise V27ContractError("E3B fold does not contain two distinct neural seeds")
                if len(by_seed) != 2 or any(set(value) != {"calibration", "validation"} for value in by_seed.values()):
                    raise V27ContractError("E3B fold does not contain complete two-seed C/V scores")
                ordered_seeds = tuple(sorted(by_seed))
                fold_rows.append(E3BFoldScores(
                    regime=str(regime), inner_seed=inner_seed, outer_fold=fold_index,
                    calibration_qids=c_qids, validation_qids=v_qids,
                    calibration_by_seed=tuple(by_seed[seed]["calibration"] for seed in ordered_seeds),
                    validation_by_seed=tuple(by_seed[seed]["validation"] for seed in ordered_seeds),
                    source_artifacts=tuple(artifact_records),
                    report_lineage_sha256=str(report.get("lineage_contract_sha256")),
                ))
        output[str(regime)] = sorted(fold_rows, key=lambda row: (row.inner_seed, row.outer_fold))
    return output, {"path": str(report_path.resolve()), "sha256": sha256_file(report_path), "lineage_contract_sha256": report.get("lineage_contract_sha256"), "mode": expected_mode, "regimes": sorted(output)}


def qwen_lineage(*, stage1: Path, stage2: Path, anchor: Path, contexts_zip: Path, model_fingerprint: Mapping[str, str], model_revision: str, max_tokens: int = QWEN_MAX_TOKENS) -> dict[str, Any]:
    return {
        "kind": R1_KIND, "schema_version": R1_SCHEMA_VERSION, "split": "complement", "variant": R1_VARIANT,
        "stage1_sha256": sha256_file(stage1), "stage2_sha256": sha256_file(stage2), "anchor_sha256": sha256_file(anchor),
        "contexts_zip_sha256": sha256_file(contexts_zip), "qwen_model": QWEN_MODEL_ID,
        "qwen_checkpoint_fingerprint": dict(model_fingerprint), "model_revision": model_revision,
        "max_tokens": int(max_tokens), "pool_size": CANDIDATE_LIMIT, "instruction": INSTRUCTION,
        "candidate_prefix_is_frozen_c35": True, "yes_no_logit": "logit_yes_minus_logit_no",
        "holdout_opened": False, "public_labels_used": False,
    }


def normalise_qwen(values_by_doc: Mapping[str, Mapping[str, Any]], c35: Sequence[str]) -> tuple[dict[str, float], bool]:
    values: dict[str, float] = {}
    for doc_id in c35:
        raw = values_by_doc.get(str(doc_id), {}).get("yes_no_logit")
        if isinstance(raw, (int, float)) and math.isfinite(float(raw)):
            values[str(doc_id)] = float(raw)
    if not values:
        return {}, True
    array = np.asarray(list(values.values()), dtype=np.float64)
    median = float(np.median(array)); q75, q25 = np.percentile(array, [75.0, 25.0]); iqr = float(q75 - q25)
    if iqr <= EPSILON:
        spread = float(array.max() - array.min())
        if spread <= EPSILON:
            return {doc_id: 0.0 for doc_id in values}, True
        # A finite, pre-registered guard for a degenerate IQR.  It avoids an
        # epsilon explosion while retaining the direction of the raw score.
        scale = max(spread, 1.0)
        return {doc_id: (value - median) / scale for doc_id, value in values.items()}, True
    return {doc_id: (value - median) / iqr for doc_id, value in values.items()}, False


def utility_rows(*, qid: str, anchor: Sequence[str], c35: Sequence[str], e2_by_seed: Sequence[Mapping[str, Mapping[str, Any]]], qwen_by_doc: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(e2_by_seed) != 2:
        raise V27ContractError("R1 requires exactly two E2 score seeds")
    normalized_e2: list[dict[str, float]] = []
    for seed_scores in e2_by_seed:
        values: dict[str, float] = {}
        for doc_id in c35:
            raw = seed_scores.get(qid, {}).get(doc_id)
            value = e3b_gate._score_value(raw or {}, qid=qid, doc_id=doc_id) if raw is not None else None
            if value is not None:
                values[str(doc_id)] = value
        if values:
            array = np.asarray(list(values.values()), dtype=np.float64); median = float(np.median(array)); q75, q25 = np.percentile(array, [75.0, 25.0]); scale = max(float(q75 - q25), 1.0e-6)
            normalized_e2.append({doc_id: (value - median) / scale for doc_id, value in values.items()})
        else:
            normalized_e2.append({})
    normalized_f, degenerate = normalise_qwen(qwen_by_doc, c35)
    frozen = set(map(str, anchor[:4])); incumbent = str(anchor[4]); proposals: list[dict[str, Any]] = []
    if incumbent not in normalized_e2[0] or incumbent not in normalized_e2[1] or incumbent not in normalized_f:
        return [], {"qid": qid, "incumbent": incumbent, "qwen_iqr_degenerate": degenerate, "incumbent_scoreable": False}
    for doc_id in map(str, c35):
        if doc_id in frozen or doc_id == incumbent or doc_id not in normalized_e2[0] or doc_id not in normalized_e2[1] or doc_id not in normalized_f:
            continue
        gaps = [normalized_e2[seed][doc_id] - normalized_e2[seed][incumbent] for seed in range(2)]
        h_e = float(sum(gaps) / 2.0); penalty = R1_DISAGREEMENT_WEIGHT * abs(gaps[0] - gaps[1]); g_f = float(normalized_f[doc_id] - normalized_f[incumbent])
        proposals.append({"doc_id": doc_id, "e2_seed_gaps": gaps, "h_e": h_e, "p_e": penalty, "g_e": h_e - penalty, "g_f": g_f, "qwen_z": normalized_f[doc_id]})
    for row in proposals:
        row["utility_lambda_0"] = float(row["g_e"])
        row["utility_lambda_025"] = float(0.75 * row["h_e"] + 0.25 * row["g_f"] - row["p_e"])
        row["utility_lambda_050"] = float(0.50 * row["h_e"] + 0.50 * row["g_f"] - row["p_e"])
    return proposals, {"qid": qid, "incumbent": incumbent, "qwen_iqr_degenerate": degenerate, "incumbent_scoreable": True, "candidate_count": len(proposals)}


def apply_r1_policy(split: Any, indices: Sequence[int], e2_by_seed: Sequence[Mapping[str, Mapping[str, Any]]], qwen_scores: Mapping[str, Mapping[str, Mapping[str, Any]]], lam: float) -> tuple[dict[str, dict[str, list[str]]], dict[str, Any]]:
    if lam not in R1_LAMBDAS:
        raise V27ContractError(f"R1 lambda is not locked: {lam}")
    predictions: dict[str, dict[str, list[str]]] = {}; decisions: dict[str, Any] = {}
    for index in indices:
        qid = str(split.qids[index]); anchor = [str(value) for value in split.anchor[index]]; c35 = [str(value) for value in split.candidates[index][:CANDIDATE_LIMIT]]
        proposal_rows, diagnostic = utility_rows(qid=qid, anchor=anchor, c35=c35, e2_by_seed=e2_by_seed, qwen_by_doc=qwen_scores.get(qid, {}))
        if lam == 0.0:
            # The zero-weight arm delegates to the original policy function;
            # it is the baseline parity oracle, not a reimplementation.
            policy = e3b_gate.E3BPolicy(memory_weight=0.0, threshold=R1_BASELINE_THRESHOLD)
            predicted, original_decisions = e3b_gate.apply_policy(split, [index], e2_by_seed, {}, policy)
            qid_prediction = predicted[qid]; predictions[qid] = qid_prediction; decisions[qid] = {"lambda": 0.0, "baseline_parity_delegate": True, "r1_diagnostic": diagnostic, "original": original_decisions[qid]}
            continue
        utility_key = {0.0: "utility_lambda_0", 0.25: "utility_lambda_025", 0.5: "utility_lambda_050"}[float(lam)]
        # Keep exact c35 order as the tie-break after utility.
        ranked = sorted(proposal_rows, key=lambda row: (-float(row[utility_key]), c35.index(str(row["doc_id"])), str(row["doc_id"])))
        winner = ranked[0] if ranked else None; answer = anchor.copy()
        if winner is not None and float(winner[utility_key]) > R1_BASELINE_THRESHOLD:
            answer[4] = str(winner["doc_id"])
        if answer[:4] != anchor[:4] or len(answer) != 5 or len(set(answer)) != 5 or not set(answer).issubset(set(c35)):
            raise V27ContractError(f"{qid}: R1 output violates frozen top-4/c35 contract")
        predictions[qid] = {"answer": answer}; decisions[qid] = {"lambda": lam, "baseline_parity_delegate": False, "r1_diagnostic": diagnostic, "winner": winner, "changed": answer != anchor}
    return predictions, decisions


def build_preflight(*, stage1: Path, stage2: Path, anchor: Path, reference: Path, contexts_zip: Path, qwen_model: Path, model_revision: str, e3b_report: Path, e1_manifest: Path, gate_source: Path, gate_config: Path) -> dict[str, Any]:
    for path, label in ((stage1, "Stage-1"), (stage2, "Stage-2"), (anchor, "V16b anchor"), (reference, "complement reference"), (e3b_report, "E3B report"), (e1_manifest, "E1 manifest"), (gate_source, "gate source"), (gate_config, "gate config")):
        require_complement_file(path, label)
    require_contexts_zip(contexts_zip)
    # The contract preflight must remain label-blind.  `load_candidate_universe`
    # only needs the candidate/anchor files for E3B artifact validation; the
    # complement reference is deliberately hashed above and opened later by
    # the evaluator after this manifest exists.
    universe = load_candidate_universe(stage1=stage1, stage2=stage2, anchor=anchor, reference=None)
    if not qwen_model.is_dir():
        raise FileNotFoundError(f"Missing Qwen R1 model snapshot: {qwen_model}")
    from qwen_c200_common import checkpoint_fingerprint
    fingerprint = checkpoint_fingerprint(qwen_model)
    # Keep this stage label-blind.  Full complement qrels are opened only by
    # the evaluator after the resulting preflight manifest is written.
    e3b_reports, report_info = load_e3b_fold_scores(report_path=e3b_report, expected_mode="standard", split=universe.split)
    lineage = {"kind": R1_KIND, "schema_version": R1_SCHEMA_VERSION, "split": "complement", "stage1_sha256": sha256_file(stage1), "stage2_sha256": sha256_file(stage2), "anchor_sha256": sha256_file(anchor), "reference_sha256": sha256_file(reference), "contexts_zip_sha256": sha256_file(contexts_zip), "qwen_lineage": qwen_lineage(stage1=stage1, stage2=stage2, anchor=anchor, contexts_zip=contexts_zip, model_fingerprint=fingerprint, model_revision=model_revision), "e3b_report": report_info, "e1_manifest_sha256": sha256_file(e1_manifest), "gate_source_sha256": sha256_file(gate_source), "gate_config_sha256": sha256_file(gate_config), "lambdas": list(R1_LAMBDAS), "baseline_threshold": R1_BASELINE_THRESHOLD, "disagreement_weight": R1_DISAGREEMENT_WEIGHT, "preserve_top4": True, "replace_only_slot5": True, "new_policy_holdout_validated": False, "holdout_opened": False, "public_labels_used": False}
    return {"kind": f"{R1_KIND}_preflight", "schema_version": R1_SCHEMA_VERSION, "stage": "label_blind_preflight", "queries": len(universe.split.qids), "lineage": lineage, "lineage_sha256": json_sha256(lineage), "e3b_regimes": {key: len(value) for key, value in e3b_reports.items()}, "qwen_model_fingerprint": fingerprint, "parameter_budget": {"frozen_upstream_parameters": FROZEN_UPSTREAM_PARAMETERS, "qwen_estimate_parameters": 600_000_000, "estimated_combined_parameters": FROZEN_UPSTREAM_PARAMETERS + 600_000_000, "parameter_limit": PARAMETER_LIMIT, "under_4b_estimate": FROZEN_UPSTREAM_PARAMETERS + 600_000_000 < PARAMETER_LIMIT, "actual_count_checked_during_gpu_probe": True}, "gpu_required_for_qwen_cache": True, "planned_qwen_pairs": len(universe.split.qids) * CANDIDATE_LIMIT, "holdout_opened": False, "public_labels_used": False, "labels_opened_after_manifest": False}


def main() -> int:
    parser = argparse.ArgumentParser(description="Label-blind R1 contract preflight")
    parser.add_argument("--stage1", type=Path, required=True); parser.add_argument("--stage2", type=Path, required=True); parser.add_argument("--anchor", type=Path, required=True); parser.add_argument("--reference", type=Path, required=True); parser.add_argument("--contexts-zip", type=Path, required=True); parser.add_argument("--qwen-model", type=Path, required=True); parser.add_argument("--model-revision", required=True); parser.add_argument("--e3b-report", type=Path, required=True); parser.add_argument("--e1-manifest", type=Path, required=True); parser.add_argument("--gate-source", type=Path, required=True); parser.add_argument("--gate-config", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = build_preflight(stage1=args.stage1, stage2=args.stage2, anchor=args.anchor, reference=args.reference, contexts_zip=args.contexts_zip, qwen_model=args.qwen_model, model_revision=args.model_revision, e3b_report=args.e3b_report, e1_manifest=args.e1_manifest, gate_source=args.gate_source, gate_config=args.gate_config)
    output = args.output_dir / "r1_preflight_report.json"
    if output.exists() and not args.force: raise FileExistsError(f"R1 preflight exists: {output}")
    write_json(output, report)
    print(json.dumps({"output": str(output.resolve()), "stage": "label_blind_preflight", "gpu_required_for_qwen_cache": True, "holdout_opened": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
