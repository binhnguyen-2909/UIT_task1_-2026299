#!/usr/bin/env python3
"""Contracts and cache adapters for the hidden-state c67 reranker branch.

This branch freezes Qwen3-Reranker-0.6B and trains only a 1025-parameter
linear readout.  The module deliberately separates label-blind cache/fold
validation from the evaluator that opens complement qrels after its manifest
has been written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e3b_residual_score_gate as e3b_gate  # noqa: E402
from qwen_c200_common import (  # noqa: E402
    CANDIDATE_LIMIT, INSTRUCTION, POOL_SIZES, load_candidate_universe,
    require_contexts_zip,
)
from v27_a_common import (  # noqa: E402
    V27ContractError, json_sha256, read_json, sha256_file, write_json,
)


HSR_KIND = "legalir_hidden_state_reranker_c67"
HSR_SCHEMA_VERSION = 1
HSR_VARIANT = "C_c67_long"
QWEN_MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
MAX_TOKENS = 4096
HIDDEN_SIZE = 1024
HIDDEN_DTYPE = "float16"
REGULARISATIONS = (1.0e-4, 1.0e-3, 1.0e-2)
TAUS = (0.0, 1.0, 2.0)
HEAD_PARAMETERS = HIDDEN_SIZE + 1
FROZEN_PLAN_PARAMETERS = 3_205_786_512
QWEN_PLAN_PARAMETERS = 600_000_000
PARAMETER_LIMIT = 4_000_000_000
MIN_STANDARD_GAIN_VS_E3B = 0.003
EPSILON = 1.0e-12


@dataclass(frozen=True)
class E3BFold:
    regime: str
    inner_seed: int
    outer_fold: int
    train_qids: tuple[str, ...]
    calibration_qids: tuple[str, ...]
    validation_qids: tuple[str, ...]
    calibration_by_seed: tuple[dict[str, dict[str, dict[str, Any]]], ...]
    validation_by_seed: tuple[dict[str, dict[str, dict[str, Any]]], ...]
    selected_policy: e3b_gate.E3BPolicy | None
    reported_validation_metrics: Mapping[str, Any]
    reported_validation_decisions: Mapping[str, Mapping[str, Any]]
    source_artifacts: tuple[dict[str, Any], ...]
    report_lineage_contract_sha256: str


@dataclass(frozen=True)
class HiddenCache:
    scores: Mapping[str, Mapping[str, Mapping[str, Any]]]
    lineage_sha256: str
    metadata_paths: tuple[str, ...]
    feature_paths: tuple[str, ...]
    hidden_size: int
    hidden_dtype: str
    pairs: int
    lineage: Mapping[str, Any]
    parameter_ledger: Mapping[str, Any]


def require_complement_file(path: Path, label: str) -> None:
    normalized = str(path).replace("\\", "/").casefold()
    if any(term in normalized for term in ("holdout", "public", "submission", "locked")):
        raise V27ContractError(f"HSR is complement-only; {label} is sealed/public data: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise V27ContractError(f"{path}:{number}: invalid JSONL") from exc
            if not isinstance(value, dict):
                raise V27ContractError(f"{path}:{number}: expected JSON object")
            rows.append(value)
    return rows


def _load_report(path: Path, expected_mode: str) -> dict[str, Any]:
    require_complement_file(path, f"E3B {expected_mode} report")
    report = read_json(path)
    if not isinstance(report, Mapping) or report.get("kind") != "legalir_e3b_nested_mil_full_c35_oof":
        raise V27ContractError(f"Unexpected E3B report kind: {path}")
    if report.get("mode") != expected_mode or report.get("split") != "complement":
        raise V27ContractError(f"E3B report is not complement/{expected_mode}: {path}")
    if report.get("holdout_opened") is not False or report.get("public_labels_used") is not False:
        raise V27ContractError("HSR refuses an E3B prerequisite that opened holdout/public labels")
    return dict(report)


def _partition_qids(partitions: Mapping[str, Any], name: str) -> tuple[str, ...]:
    value = partitions.get(name)
    if not isinstance(value, Mapping) or not isinstance(value.get("qids"), list):
        raise V27ContractError(f"E3B partition {name} has no QID list")
    qids = tuple(str(item) for item in value["qids"])
    if not qids or len(set(qids)) != len(qids):
        raise V27ContractError(f"E3B partition {name} has duplicate/empty QIDs")
    return qids


def _artifact_rows(path: Path, *, c35_by_qid: Mapping[str, Sequence[str]], qids: Sequence[str]) -> dict[str, dict[str, dict[str, Any]]]:
    rows = _read_jsonl(path)
    return e3b_gate.validate_full_c35_scores(rows, c35_by_qid=c35_by_qid, expected_qids=qids)


def _selected_policy(fold: Mapping[str, Any]) -> e3b_gate.E3BPolicy | None:
    value = fold.get("selected_policy")
    if not isinstance(value, Mapping):
        raise V27ContractError("E3B fold has no selected policy")
    if value.get("kind") == "v16b_anchor_abstain":
        return None
    if value.get("kind") != "e3b_full_c35_mil_gap":
        raise V27ContractError(f"Unsupported E3B selected policy: {value.get('kind')}")
    try:
        return e3b_gate.E3BPolicy(memory_weight=float(value["memory_weight"]), threshold=float(value["threshold"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise V27ContractError("Malformed E3B selected policy") from exc


def load_e3b_folds(*, report_path: Path, expected_mode: str, split: Any) -> tuple[dict[str, list[E3BFold]], dict[str, Any]]:
    """Load and hash-check all two-seed C/V E3B artifacts for c67 reuse."""
    report = _load_report(report_path, expected_mode)
    regimes = report.get("regimes")
    if not isinstance(regimes, Mapping):
        raise V27ContractError("E3B report has no regimes")
    c35_by_qid = {str(qid): [str(doc) for doc in split.candidates[index][:CANDIDATE_LIMIT]] for index, qid in enumerate(split.qids)}
    all_qids = set(map(str, split.qids)); output: dict[str, list[E3BFold]] = {}
    for regime, runs in regimes.items():
        if not isinstance(runs, list) or len(runs) != 2:
            raise V27ContractError(f"E3B {regime}: expected two inner seeds")
        folds: list[E3BFold] = []
        for run in runs:
            if not isinstance(run, Mapping) or not isinstance(run.get("folds"), list) or len(run["folds"]) != 5:
                raise V27ContractError(f"E3B {regime}: expected five outer folds")
            inner_seed = int(run.get("inner_seed"))
            for outer_fold, fold in enumerate(run["folds"]):
                if not isinstance(fold, Mapping):
                    raise V27ContractError("E3B fold is not an object")
                partitions = fold.get("partitions")
                if not isinstance(partitions, Mapping):
                    raise V27ContractError("E3B fold lacks partition manifest")
                a_fit = _partition_qids(partitions, "a_fit")
                a_stop = _partition_qids(partitions, "a_stop")
                calibration = _partition_qids(partitions, "calibration")
                validation = _partition_qids(partitions, "validation")
                partition_sets = [set(a_fit), set(a_stop), set(calibration), set(validation)]
                if any(left & right for index, left in enumerate(partition_sets) for right in partition_sets[index + 1:]):
                    raise V27ContractError(f"E3B {regime}/seed={inner_seed}/fold={outer_fold}: partition overlap")
                if set().union(*partition_sets) != all_qids:
                    raise V27ContractError(f"E3B {regime}/seed={inner_seed}/fold={outer_fold}: partitions do not cover complement")
                artifacts = fold.get("score_artifacts")
                if not isinstance(artifacts, list) or len(artifacts) != 4:
                    raise V27ContractError(f"E3B {regime}/seed={inner_seed}/fold={outer_fold}: expected four C/V artifacts")
                by_seed: dict[int, dict[str, dict[str, dict[str, Any]]]] = {}
                records: list[dict[str, Any]] = []
                for raw in artifacts:
                    if not isinstance(raw, Mapping):
                        raise V27ContractError("E3B artifact record is not an object")
                    score_path = Path(str(raw.get("score_path", ""))); meta_path = Path(str(raw.get("metadata_path", "")))
                    require_complement_file(score_path, "E3B score artifact")
                    require_complement_file(meta_path, "E3B score metadata")
                    if raw.get("score_sha256") != sha256_file(score_path) or raw.get("metadata_sha256") != sha256_file(meta_path):
                        raise V27ContractError(f"E3B artifact hash changed: {score_path}")
                    metadata = read_json(meta_path)
                    if not isinstance(metadata, Mapping) or metadata.get("lineage_contract_sha256") != report.get("lineage_contract_sha256"):
                        raise V27ContractError("E3B artifact lineage differs from report")
                    partition = str(metadata.get("partition", raw.get("partition", "")))
                    neural_seed = int(metadata.get("neural_seed", raw.get("neural_seed")))
                    qids = calibration if partition == "calibration" else validation if partition == "validation" else ()
                    if not qids:
                        raise V27ContractError(f"Unknown E3B score partition: {partition}")
                    parsed = _artifact_rows(score_path, c35_by_qid=c35_by_qid, qids=qids)
                    if neural_seed in by_seed and partition in by_seed[neural_seed]:
                        raise V27ContractError("Duplicate E3B neural-seed/partition artifact")
                    by_seed.setdefault(neural_seed, {})[partition] = parsed
                    records.append({**dict(raw), "metadata": dict(metadata)})
                if len(by_seed) != 2 or any(set(value) != {"calibration", "validation"} for value in by_seed.values()):
                    raise V27ContractError("E3B fold lacks complete two-seed C/V scores")
                raw_decisions = fold.get("decisions")
                if not isinstance(raw_decisions, Mapping) or set(map(str, raw_decisions)) != set(validation):
                    raise V27ContractError(f"E3B {regime}/seed={inner_seed}/fold={outer_fold}: validation decisions differ from manifest")
                decisions: dict[str, dict[str, Any]] = {}
                for qid in validation:
                    raw = raw_decisions[qid]
                    if not isinstance(raw, Mapping) or not isinstance(raw.get("changed"), bool):
                        raise V27ContractError(f"E3B {regime}/seed={inner_seed}/fold={outer_fold}: malformed decision for {qid}")
                    changed = bool(raw["changed"]); replacement = raw.get("replacement")
                    if changed:
                        if not isinstance(replacement, str) or replacement not in c35_by_qid[qid]:
                            raise V27ContractError(f"E3B {regime}/seed={inner_seed}/fold={outer_fold}: invalid replacement for {qid}")
                    elif replacement is not None:
                        raise V27ContractError(f"E3B {regime}/seed={inner_seed}/fold={outer_fold}: unchanged decision has replacement for {qid}")
                    decisions[qid] = {"changed": changed, "replacement": replacement}
                seeds = sorted(by_seed)
                folds.append(E3BFold(
                    regime=str(regime), inner_seed=inner_seed, outer_fold=outer_fold,
                    train_qids=tuple(a_fit + a_stop), calibration_qids=calibration, validation_qids=validation,
                    calibration_by_seed=tuple(by_seed[seed]["calibration"] for seed in seeds),
                    validation_by_seed=tuple(by_seed[seed]["validation"] for seed in seeds),
                    selected_policy=_selected_policy(fold), reported_validation_metrics=dict(fold.get("validation", {})), reported_validation_decisions=decisions, source_artifacts=tuple(records),
                    report_lineage_contract_sha256=str(report.get("lineage_contract_sha256")),
                ))
        output[str(regime)] = sorted(folds, key=lambda item: (item.inner_seed, item.outer_fold))
    info = {"path": str(report_path.resolve()), "sha256": sha256_file(report_path), "lineage_contract_sha256": report.get("lineage_contract_sha256"), "mode": expected_mode, "regimes": sorted(output)}
    return output, info


def hidden_lineage(*, stage1: Path, stage2: Path, anchor: Path, contexts_zip: Path, model_fingerprint: Mapping[str, str], model_revision: str, hidden_dtype: str = HIDDEN_DTYPE) -> dict[str, Any]:
    return {
        "kind": HSR_KIND, "schema_version": HSR_SCHEMA_VERSION, "split": "complement",
        "variant": HSR_VARIANT, "stage1_sha256": sha256_file(stage1), "stage2_sha256": sha256_file(stage2),
        "anchor_sha256": sha256_file(anchor), "contexts_zip_sha256": sha256_file(contexts_zip),
        "qwen_model": QWEN_MODEL_ID, "qwen_checkpoint_fingerprint": dict(model_fingerprint),
        "model_revision": model_revision, "max_tokens": MAX_TOKENS, "pool_size": POOL_SIZES[HSR_VARIANT],
        "hidden_size": HIDDEN_SIZE, "hidden_dtype": hidden_dtype, "instruction": INSTRUCTION,
        "last_hidden_state_position": "last_non_padding_token", "native_score": "logit_yes_minus_logit_no",
        "head_formula": "z0 + dot(w, h_last / max(norm(h_last), 1e-8)) + b",
        "holdout_opened": False, "public_labels_used": False,
    }


def _path_from_metadata(meta_path: Path, value: Any, folder: str, suffix: str) -> Path:
    declared = Path(str(value or ""))
    if declared.is_file():
        return declared
    fallback = meta_path.parent.parent / folder / f"{meta_path.stem}{suffix}"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"Hidden cache artifact is missing: {declared or fallback}")


def load_hidden_cache(root: Path, *, expected_qids: Sequence[str], c67_by_qid: Mapping[str, Sequence[str]], expected_lineage_sha256: str | None = None) -> HiddenCache:
    actual_root = root / "hidden_state_cache" if (root / "hidden_state_cache").is_dir() else root
    metadata_paths = sorted((actual_root / "metadata").glob("*.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"Hidden-state cache metadata is missing: {actual_root}")
    scores: dict[str, dict[str, dict[str, Any]]] = {}; lineage_sha: str | None = None; feature_paths: list[str] = []
    for meta_path in metadata_paths:
        meta = read_json(meta_path)
        if not isinstance(meta, Mapping) or meta.get("kind") != f"{HSR_KIND}_cache":
            raise V27ContractError(f"Invalid hidden-state metadata: {meta_path}")
        current = str(meta.get("lineage_sha256", ""))
        if expected_lineage_sha256 is not None and current != expected_lineage_sha256:
            raise V27ContractError("Hidden-state cache lineage differs from preflight")
        if lineage_sha is None:
            lineage_sha = current
        elif lineage_sha != current:
            raise V27ContractError("Hidden-state cache shards have different lineage")
        if int(meta.get("hidden_size", -1)) != HIDDEN_SIZE:
            raise V27ContractError("Hidden-state cache hidden_size is not the locked 1024")
        feature_path = _path_from_metadata(meta_path, meta.get("feature_path"), "features", ".npy")
        rows_path = _path_from_metadata(meta_path, meta.get("rows_path"), "rows", ".jsonl")
        states = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        if states.ndim != 2 or states.shape[1] != HIDDEN_SIZE or not np.isfinite(states).all():
            raise V27ContractError(f"Invalid hidden-state array: {feature_path}")
        feature_paths.append(str(feature_path.resolve()))
        for row in _read_jsonl(rows_path):
            qid, doc_id = str(row.get("qid", "")), str(row.get("doc_id", ""))
            if qid not in set(map(str, expected_qids)) or doc_id not in set(c67_by_qid.get(qid, ())):
                raise V27ContractError(f"Hidden-state row escapes c67: {qid}/{doc_id}")
            expected_rank = list(c67_by_qid[qid]).index(doc_id) + 1
            if int(row.get("candidate_rank", -1)) != expected_rank:
                raise V27ContractError(f"{qid}/{doc_id}: hidden-state candidate rank drift")
            if row.get("lineage_sha256") != lineage_sha:
                raise V27ContractError(f"{qid}/{doc_id}: hidden-state row lineage drift")
            index = int(row.get("state_index", -1))
            native = row.get("native_yes_minus_no")
            if not 0 <= index < len(states) or not isinstance(native, (int, float)) or not math.isfinite(float(native)):
                raise V27ContractError(f"{qid}/{doc_id}: invalid hidden-state row")
            if not isinstance(row.get("evidence_valid"), bool):
                raise V27ContractError(f"{qid}/{doc_id}: evidence_valid is not boolean")
            if doc_id in scores.get(qid, {}):
                raise V27ContractError(f"Duplicate hidden-state row: {qid}/{doc_id}")
            scores.setdefault(qid, {})[doc_id] = {**row, "hidden": states[index]}
    if lineage_sha is None:
        raise V27ContractError("Hidden-state cache has no lineage")
    for qid in map(str, expected_qids):
        if set(scores.get(qid, {})) != set(map(str, c67_by_qid[qid])):
            raise V27ContractError(f"{qid}: hidden-state cache does not cover exact c67")
    first_meta = read_json(Path(metadata_paths[0]))
    parameter_ledger = dict(first_meta.get("parameter_ledger", {}))
    if not parameter_ledger or parameter_ledger.get("under_4b") is not True:
        raise V27ContractError("Hidden-state cache has no passing parameter ledger")
    return HiddenCache(scores=scores, lineage_sha256=lineage_sha, metadata_paths=tuple(str(path.resolve()) for path in metadata_paths), feature_paths=tuple(feature_paths), hidden_size=int(first_meta.get("hidden_size", HIDDEN_SIZE)), hidden_dtype=str(first_meta.get("hidden_dtype", HIDDEN_DTYPE)), pairs=sum(len(value) for value in scores.values()), lineage=dict(first_meta.get("lineage", {})), parameter_ledger=parameter_ledger)


def planned_parameter_budget() -> dict[str, Any]:
    total = FROZEN_PLAN_PARAMETERS + HEAD_PARAMETERS
    return {"frozen_plan_parameters": FROZEN_PLAN_PARAMETERS, "head_parameters": HEAD_PARAMETERS, "combined_parameters": total, "parameter_limit": PARAMETER_LIMIT, "under_4b_plan": total < PARAMETER_LIMIT, "actual_qwen_count_checked_by_exporter": True}


def build_preflight(*, stage1: Path, stage2: Path, anchor: Path, contexts_zip: Path, qwen_model: Path, model_revision: str, e3b_standard_report: Path, e3b_stress_report: Path, e1_manifest: Path, gate_source: Path, gate_config: Path) -> dict[str, Any]:
    for path, label in ((stage1, "Stage-1"), (stage2, "Stage-2"), (anchor, "V16b anchor"), (e3b_standard_report, "E3B standard report"), (e3b_stress_report, "E3B stress report"), (e1_manifest, "E1 manifest"), (gate_source, "gate source"), (gate_config, "gate config")):
        require_complement_file(path, label)
    require_contexts_zip(contexts_zip)
    universe = load_candidate_universe(stage1=stage1, stage2=stage2, anchor=anchor, reference=None)
    if not qwen_model.is_dir():
        raise FileNotFoundError(f"Missing Qwen snapshot: {qwen_model}")
    from qwen_c200_common import checkpoint_fingerprint
    fingerprint = checkpoint_fingerprint(qwen_model)
    standard_folds, standard_info = load_e3b_folds(report_path=e3b_standard_report, expected_mode="standard", split=universe.split)
    stress_folds, stress_info = load_e3b_folds(report_path=e3b_stress_report, expected_mode="stress", split=universe.split)
    required_stress = {"law_source_family", "query_template_family", "near_duplicate_family"}
    if not required_stress.issubset(stress_folds):
        raise V27ContractError(f"E3B stress report lacks required regimes: {sorted(required_stress - set(stress_folds))}")
    lineage = {"kind": HSR_KIND, "schema_version": HSR_SCHEMA_VERSION, "split": "complement", "stage1_sha256": sha256_file(stage1), "stage2_sha256": sha256_file(stage2), "anchor_sha256": sha256_file(anchor), "contexts_zip_sha256": sha256_file(contexts_zip), "qwen_lineage": hidden_lineage(stage1=stage1, stage2=stage2, anchor=anchor, contexts_zip=contexts_zip, model_fingerprint=fingerprint, model_revision=model_revision), "e3b_standard": standard_info, "e3b_stress": stress_info, "e1_manifest_sha256": sha256_file(e1_manifest), "gate_source_sha256": sha256_file(gate_source), "gate_config_sha256": sha256_file(gate_config), "regularisations": list(REGULARISATIONS), "taus": list(TAUS), "head_parameters": HEAD_PARAMETERS, "hidden_state_s2_reuse": False, "new_policy_holdout_validated": False, "holdout_opened": False, "public_labels_used": False}
    return {"kind": f"{HSR_KIND}_preflight", "schema_version": HSR_SCHEMA_VERSION, "stage": "label_blind_preflight", "queries": len(universe.split.qids), "planned_pairs": len(universe.split.qids) * POOL_SIZES[HSR_VARIANT], "lineage": lineage, "lineage_sha256": json_sha256(lineage), "experiment_lineage_sha256": json_sha256({key: value for key, value in lineage.items() if key not in {"mode"}}), "e3b_folds": {"standard": {key: len(value) for key, value in standard_folds.items()}, "stress": {key: len(value) for key, value in stress_folds.items()}}, "parameter_budget": planned_parameter_budget(), "gpu_required": True, "holdout_opened": False, "public_labels_used": False, "labels_opened_after_manifest": False}


def main() -> int:
    parser = argparse.ArgumentParser(description="Label-blind hidden-state c67 preflight")
    for name in ("stage1", "stage2", "anchor", "contexts_zip", "qwen_model", "e3b_standard_report", "e3b_stress_report", "e1_manifest", "gate_source", "gate_config"):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, required=True)
    parser.add_argument("--model-revision", required=True); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); report = build_preflight(stage1=args.stage1, stage2=args.stage2, anchor=args.anchor, contexts_zip=args.contexts_zip, qwen_model=args.qwen_model, model_revision=args.model_revision, e3b_standard_report=args.e3b_standard_report, e3b_stress_report=args.e3b_stress_report, e1_manifest=args.e1_manifest, gate_source=args.gate_source, gate_config=args.gate_config)
    output = args.output_dir / "hsr_preflight_report.json"
    if output.exists() and not args.force:
        raise FileExistsError(f"HSR preflight exists: {output}")
    write_json(output, report)
    print(json.dumps({"output": str(output.resolve()), "stage": report["stage"], "planned_pairs": report["planned_pairs"], "under_4b_plan": report["parameter_budget"]["under_4b_plan"], "holdout_opened": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
