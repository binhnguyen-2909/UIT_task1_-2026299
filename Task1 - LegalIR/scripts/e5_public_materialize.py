#!/usr/bin/env python3
"""E5: label-blind public materialization from the locked E3-B/E4 deployment.

E5 is authorized only when the one-time E4 locked-holdout parity passed with
``promotion_to_e5_allowed: true``.  This script never trains, never tunes a
threshold, never opens ``holdout_1000.json`` and never uses public labels.
It reuses the two frozen E4 deployment weights, scores the frozen public c35
with them, applies the fixed ``E3BPolicy(memory_weight=0.0, threshold=0.50)``
(slot-5 only, top-4 frozen, default abstain to the V16b public anchor), and
writes auditable public predictions plus a provenance report.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import e1_evidence_frozen_ce as e1  # noqa: E402
import e2_mil_finetune_oof as e2  # noqa: E402
from e3b_residual_score_gate import (  # noqa: E402
    E3BPolicy,
    apply_policy,
    validate_full_c35_scores,
)
from v27_a_common import (  # noqa: E402
    CANDIDATE_LIMIT,
    EDITABLE_SLOT,
    V27ContractError,
    json_sha256,
    load_manifest,
    load_relevance_model,
    load_split,
    read_json,
    sha256_file,
    write_json,
)


E5_KIND = "legalir_e5_public_materialization"
E5_SCHEMA_VERSION = 1
E5_LOCKED_POLICY = E3BPolicy(memory_weight=0.0, threshold=0.50)
E5_DEPLOYMENT_SEEDS = (20260908, 20260909)
E5_PARAMETER_COUNT = 2_610_010_000
E5_PARAMETER_LIMIT = 4_000_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "materialize"), default="preflight")
    parser.add_argument("--public-stage1", type=Path, required=True)
    parser.add_argument("--public-stage2", type=Path, required=True)
    parser.add_argument("--public-anchor", type=Path, required=True)
    parser.add_argument("--contexts-zip", type=Path, required=True)
    parser.add_argument("--bqb-model", type=Path, required=True)
    parser.add_argument("--reranker-model", type=Path, required=True)
    parser.add_argument("--e4-parity-report", type=Path, required=True)
    parser.add_argument("--e4-prepare-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--score-batch-size", type=int, default=1024)
    parser.add_argument("--bqb-batch-size", type=int, default=128)
    parser.add_argument("--cpu-threads", type=int, default=1)
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _configure_cpu_threads(threads: int) -> None:
    if threads < 1:
        raise ValueError("E5 --cpu-threads must be positive")
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_MAX_THREADS"] = str(threads)
    try:
        import torch

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        pass


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"E5 required {label} is missing: {path}")


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    _require_file(path, label)
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise V27ContractError(f"E5 {label} must be a JSON object: {path}")
    return dict(payload)


def _policy_payload() -> dict[str, Any]:
    return {
        "locked": True,
        "source": "E3-B fixed-policy diagnostic over all OOF views",
        **E5_LOCKED_POLICY.as_dict(),
    }


def validate_e4_authorization(report: Mapping[str, Any]) -> dict[str, Any]:
    promo = report.get("promotion", {})
    if not isinstance(promo, Mapping) or promo.get("promotion_to_e5_allowed") is not True:
        raise V27ContractError("E4 did not authorize E5 (promotion_to_e5_allowed != true)")
    locked = report.get("locked_policy", {})
    if not isinstance(locked, Mapping):
        raise V27ContractError("E4 report has no locked policy")
    if float(locked.get("threshold", -1)) != 0.50:
        raise V27ContractError("E4 locked policy threshold drift (expected 0.50)")
    if float(locked.get("memory_weight", -1)) != 0.0:
        raise V27ContractError("E4 locked policy memory_weight drift (expected 0.0)")
    if report.get("holdout_access_count") != 1:
        raise V27ContractError("E4 ledger access count unexpected (expected 1)")
    return {
        "promotion_to_e5_allowed": True,
        "locked_policy": dict(locked),
        "lineage_contract_sha256": report.get("lineage_contract_sha256"),
    }


def check_package_payload(payload: Mapping[str, Any], expected_qids: Sequence[str]) -> bool:
    expected = {str(qid) for qid in expected_qids}
    if set(map(str, payload.keys())) != expected:
        missing = sorted(expected - set(map(str, payload.keys())))
        extra = sorted(set(map(str, payload.keys())) - expected)
        raise V27ContractError(
            f"E5 QID mismatch vs public-official; missing={missing[:3]} extra={extra[:3]}"
        )
    for qid, row in payload.items():
        answer = row.get("answer") if isinstance(row, Mapping) else None
        if not isinstance(answer, list) or len(answer) != 5:
            raise V27ContractError(f"{qid}: E5 needs exactly five IDs")
        if len(set(map(str, answer))) != 5:
            raise V27ContractError(f"{qid}: E5 answers must be unique")
    return True


def validate_e5_predictions(split: Any, predictions: Mapping[str, Any]) -> None:
    expected = {str(qid) for qid in split.qids}
    if set(map(str, predictions.keys())) != expected:
        missing = sorted(expected - set(map(str, predictions.keys())))
        extra = sorted(set(map(str, predictions.keys())) - expected)
        raise V27ContractError(f"E5 prediction QIDs mismatch; missing={missing[:3]} extra={extra[:3]}")
    for index, qid in enumerate(split.qids):
        key = str(qid)
        row = predictions[key]
        answer = row.get("answer") if isinstance(row, Mapping) else None
        if not isinstance(answer, list) or len(answer) != 5 or len(set(map(str, answer))) != 5:
            raise V27ContractError(f"{key}: E5 needs five unique IDs")
        answer = [str(value) for value in answer]
        anchor = [str(value) for value in split.anchor[index]]
        c35 = set(map(str, split.candidates[index][:CANDIDATE_LIMIT]))
        if answer[: EDITABLE_SLOT - 1] != anchor[: EDITABLE_SLOT - 1]:
            raise V27ContractError(f"{key}: E5 mutated frozen V16b top-4")
        if not set(answer).issubset(c35):
            raise V27ContractError(f"{key}: E5 prediction escaped c35")


def build_e5_report_skeleton(
    e4_report: Mapping[str, Any],
    public_qids: Sequence[str],
    prediction_sha: str,
) -> dict[str, Any]:
    return {
        "kind": E5_KIND,
        "schema_version": E5_SCHEMA_VERSION,
        "stage": "public_materialization_complete",
        "locked_policy": _policy_payload(),
        "e4_lineage_contract_sha256": e4_report.get("lineage_contract_sha256"),
        "model_budget": {
            "pipeline_parameters": E5_PARAMETER_COUNT,
            "limit": E5_PARAMETER_LIMIT,
            "under_4b": bool(E5_PARAMETER_COUNT < E5_PARAMETER_LIMIT),
        },
        "public_queries": len(list(public_qids)),
        "predictions_sha256": prediction_sha,
        "holdout_labels_opened": False,
        "holdout_parity_opened": False,
        "public_labels_used": False,
        "v16b_fallback_retained": True,
    }


def _evidence_paths(output_dir: Path) -> tuple[Path, Path]:
    return (
        output_dir / "e5_public_evidence.jsonl",
        output_dir / "e5_public_evidence.meta.json",
    )


def _score_paths(output_dir: Path, seed: int) -> tuple[Path, Path]:
    path = output_dir / "e5_full_scores" / f"public_deployment_seed_{seed}.jsonl"
    return path, path.with_suffix(".meta.json")


def _prediction_path(output_dir: Path) -> Path:
    return output_dir / "e5_public_predictions.json"


def _preflight_path(output_dir: Path) -> Path:
    return output_dir / "e5_preflight.json"


def _report_path(output_dir: Path) -> Path:
    return output_dir / "e5_public_report.json"


def _evidence_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        contexts_zip=args.contexts_zip,
        stage1=args.public_stage1,
        stage2=args.public_stage2,
        anchor=args.public_anchor,
        bqb_model=args.bqb_model,
        reranker_model=args.reranker_model,
        proposal_device=args.device,
        bqb_batch_size=int(args.bqb_batch_size),
        force_manifest=False,
    )


def _deployment_weight_records(prepare_report: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    records = prepare_report.get("deployment_models", [])
    if not isinstance(records, list) or len(records) != 2:
        raise V27ContractError("E4 prepare report must carry exactly two deployment models")
    by_seed: dict[int, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise V27ContractError("E4 deployment record is not an object")
        seed = int(record.get("seed", -1))
        if seed not in E5_DEPLOYMENT_SEEDS:
            raise V27ContractError(f"E4 deployment seed {seed} is outside the frozen E5 contract")
        weights = record.get("weights", {})
        if not isinstance(weights, Mapping) or not weights.get("path"):
            raise V27ContractError(f"E4 deployment seed {seed} has no weight path")
        by_seed[seed] = dict(record)
    if set(by_seed) != set(E5_DEPLOYMENT_SEEDS):
        raise V27ContractError("E4 deployment seeds do not match the frozen E5 contract")
    return by_seed


def _validate_public_evidence(
    args: argparse.Namespace,
    split: Any,
    manifest_path: Path,
    meta_path: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    meta = _read_json_object(meta_path, "E5 public evidence metadata")
    if meta.get("kind") != e1.E1_KIND or meta.get("split") != "public":
        raise V27ContractError("E5 public evidence has an unexpected kind or split")
    if meta.get("reference_loaded") is not False or meta.get("holdout_opened") is not False:
        raise V27ContractError("E5 public evidence is not label-blind")
    for key, path in (
        ("stage1_sha256", args.public_stage1),
        ("stage2_sha256", args.public_stage2),
        ("anchor_sha256", args.public_anchor),
    ):
        if meta.get(key) != sha256_file(path):
            raise V27ContractError(f"E5 public evidence provenance drift for {key}")
    manifest = load_manifest(manifest_path)
    e2._validate_manifest(split, manifest)
    return manifest


def _materialize_public_evidence(
    args: argparse.Namespace, split: Any
) -> tuple[Path, Path, dict[tuple[str, str], dict[str, Any]]]:
    manifest_path, meta_path = _evidence_paths(args.output_dir)
    if manifest_path.exists() or meta_path.exists():
        if not manifest_path.is_file() or not meta_path.is_file():
            raise V27ContractError("E5 evidence manifest/meta must either both exist or both be absent")
        return manifest_path, meta_path, _validate_public_evidence(args, split, manifest_path, meta_path)
    e1.materialize_manifest(
        _evidence_args(args), manifest_path, meta_path, split_name="public"
    )
    return manifest_path, meta_path, _validate_public_evidence(args, split, manifest_path, meta_path)


def _load_deployment_model(args: argparse.Namespace, record: Mapping[str, Any]) -> tuple[Any, Any]:
    weights = record.get("weights", {})
    path = Path(str(weights.get("path", "")))
    if not path.is_file() or sha256_file(path) != weights.get("sha256"):
        raise V27ContractError("E5 deployment checkpoint is missing or its file hash changed")
    # NOTE (E5 dtype fix): deployment weights are FP32 master weights
    # (model_state_sha256 hashes dtype+bytes; E4 parity scored the fresh
    # FP32 fit model). Loading the base in FP16 on CUDA then verifying
    # always mismatches, so verify in FP32 and score in FP32 like E4.
    model, tokenizer = load_relevance_model(
        args.reranker_model,
        args.device,
        train=False,
        fp16=False,
        gradient_checkpointing=False,
    )
    try:
        import torch

        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        model.load_state_dict(state, strict=True)
    finally:
        pass
    if e2.model_state_sha256(model) != weights.get("state_sha256"):
        raise V27ContractError("E5 deployment checkpoint state hash changed after reload")
    return model, tokenizer


def _public_c35(split: Any) -> dict[str, list[str]]:
    return {
        str(qid): [str(doc_id) for doc_id in split.candidates[index][:CANDIDATE_LIMIT]]
        for index, qid in enumerate(split.qids)
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            sink.write("\n")
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise V27ContractError(f"{path}:{line_number}: expected JSON object")
            rows.append(dict(row))
    return rows


def _score_public_seed(
    *,
    args: argparse.Namespace,
    split: Any,
    manifest: Mapping[tuple[str, str], Mapping[str, Any]],
    model: Any,
    tokenizer: Any,
    seed: int,
    weights: Mapping[str, Any],
    e4_lineage_sha256: Any,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    path, meta_path = _score_paths(args.output_dir, seed)
    c35_by_qid = _public_c35(split)
    if path.is_file() or meta_path.is_file():
        if not path.is_file() or not meta_path.is_file():
            raise V27ContractError("E5 score JSONL/meta must either both exist or both be absent")
        meta = _read_json_object(meta_path, "E5 full-score metadata")
        if (
            meta.get("e4_lineage_contract_sha256") != e4_lineage_sha256
            or meta.get("deployment_seed") != int(seed)
            or meta.get("weights") != dict(weights)
            or meta.get("score_sha256") != sha256_file(path)
        ):
            raise V27ContractError("E5 cached full-c35 scores do not match deployment lineage/weights")
        rows = _read_jsonl(path)
        parsed = validate_full_c35_scores(rows, c35_by_qid=c35_by_qid, expected_qids=list(c35_by_qid))
        return parsed, meta
    rows = e2.score_full_c35(
        split,
        list(range(len(split.qids))),
        manifest,
        model,
        tokenizer,
        device=args.device,
        batch_size=int(args.score_batch_size),
    )
    materialized = [
        {**row, "deployment_seed": int(seed), "split": "public",
         "e4_lineage_contract_sha256": e4_lineage_sha256}
        for row in rows
    ]
    _write_jsonl(path, materialized)
    parsed = validate_full_c35_scores(materialized, c35_by_qid=c35_by_qid, expected_qids=list(c35_by_qid))
    meta = {
        "kind": "legalir_e5_full_c35_scores",
        "schema_version": 1,
        "split": "public",
        "deployment_seed": int(seed),
        "score_path": str(path.resolve()),
        "score_sha256": sha256_file(path),
        "score_rows": len(materialized),
        "score_qids_sha256": json_sha256({"qids": list(c35_by_qid)}),
        "e4_lineage_contract_sha256": e4_lineage_sha256,
        "weights": dict(weights),
        "holdout_labels_opened": False,
        "public_labels_used": False,
    }
    write_json(meta_path, meta)
    return parsed, meta


def preflight(args: argparse.Namespace) -> tuple[dict[str, Any], Any]:
    for path, label in (
        (args.contexts_zip, "contexts ZIP"),
        (args.public_stage1, "public Stage-1"),
        (args.public_stage2, "public Stage-2"),
        (args.public_anchor, "public V16b anchor"),
        (args.e4_parity_report, "E4 parity report"),
        (args.e4_prepare_report, "E4 prepare report"),
    ):
        _require_file(path, label)
    if not args.bqb_model.is_dir() or not args.reranker_model.is_dir():
        raise FileNotFoundError("E5 requires local BQB and Vietnamese reranker checkpoints")
    if E5_PARAMETER_COUNT >= E5_PARAMETER_LIMIT:
        raise V27ContractError("E5 pipeline exceeds the 4B model budget")
    e4_report = _read_json_object(args.e4_parity_report, "E4 parity report")
    authorization = validate_e4_authorization(e4_report)
    prepare_report = _read_json_object(args.e4_prepare_report, "E4 prepare report")
    weight_records = _deployment_weight_records(prepare_report)
    for seed, record in weight_records.items():
        weights = record.get("weights", {})
        path = Path(str(weights.get("path", "")))
        if not path.is_file() or sha256_file(path) != weights.get("sha256"):
            raise V27ContractError(f"E5 deployment seed {seed} checkpoint hash mismatch")
    # Label-blind by construction: reference is None, public labels never loaded.
    split = load_split("public", args.public_stage1, args.public_stage2, args.public_anchor, None)
    if len(split.qids) != 1000:
        raise V27ContractError(f"E5 expects exactly 1,000 public QIDs, found {len(split.qids)}")
    return {
        "kind": E5_KIND,
        "schema_version": E5_SCHEMA_VERSION,
        "stage": "e5_label_blind_preflight",
        "locked_policy": _policy_payload(),
        "e4_authorization": authorization,
        "public_queries": len(split.qids),
        "deployment_seeds": list(E5_DEPLOYMENT_SEEDS),
        "model_budget": {
            "pipeline_parameters": E5_PARAMETER_COUNT,
            "limit": E5_PARAMETER_LIMIT,
            "under_4b": True,
        },
        "holdout_labels_opened": False,
        "holdout_parity_opened": False,
        "public_labels_used": False,
        "public_predictions_materialized": False,
    }, split


def materialize(args: argparse.Namespace, preflight_report: Mapping[str, Any], split: Any) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    e4_report = _read_json_object(args.e4_parity_report, "E4 parity report")
    authorization = validate_e4_authorization(e4_report)
    prepare_report = _read_json_object(args.e4_prepare_report, "E4 prepare report")
    weight_records = _deployment_weight_records(prepare_report)
    evidence_path, evidence_meta_path, manifest = _materialize_public_evidence(args, split)
    per_seed: list[dict[str, dict[str, dict[str, Any]]]] = []
    score_artifacts: list[dict[str, Any]] = []
    for seed in E5_DEPLOYMENT_SEEDS:
        model = tokenizer = None
        try:
            record = weight_records[seed]
            model, tokenizer = _load_deployment_model(args, record)
            scores, meta = _score_public_seed(
                args=args,
                split=split,
                manifest=manifest,
                model=model,
                tokenizer=tokenizer,
                seed=seed,
                weights=record["weights"],
                e4_lineage_sha256=authorization["lineage_contract_sha256"],
            )
            per_seed.append(scores)
            score_artifacts.append(meta)
        finally:
            del model, tokenizer
            gc.collect()
            if args.device == "cuda":
                try:
                    import torch

                    torch.cuda.empty_cache()
                except ImportError:
                    pass
    predictions, decisions = apply_policy(
        split, list(range(len(split.qids))), per_seed, {}, E5_LOCKED_POLICY
    )
    payload = {str(qid): {"answer": [str(v) for v in row["answer"]]} for qid, row in predictions.items()}
    validate_e5_predictions(split, payload)
    prediction_path = _prediction_path(args.output_dir)
    if prediction_path.exists():
        existing = _read_json_object(prediction_path, "E5 public predictions")
        validate_e5_predictions(split, existing)
        if existing != payload:
            raise V27ContractError("E5 existing public predictions differ from the locked policy output")
    else:
        write_json(prediction_path, payload)
    prediction_sha256 = sha256_file(prediction_path)
    report = build_e5_report_skeleton(e4_report, split.qids, prediction_sha256)
    changed = sum(1 for row in decisions.values() if bool(row.get("changed")))
    report.update({
        "e4_authorization": authorization,
        "e4_parity_report": str(args.e4_parity_report.resolve()),
        "e4_parity_report_sha256": sha256_file(args.e4_parity_report),
        "e4_prepare_report": str(args.e4_prepare_report.resolve()),
        "e4_prepare_report_sha256": sha256_file(args.e4_prepare_report),
        "public_inputs": {
            "stage1_sha256": sha256_file(args.public_stage1),
            "stage2_sha256": sha256_file(args.public_stage2),
            "anchor_sha256": sha256_file(args.public_anchor),
            "contexts_zip_sha256": sha256_file(args.contexts_zip),
        },
        "public_evidence": {
            "path": str(evidence_path.resolve()),
            "sha256": sha256_file(evidence_path),
            "metadata_path": str(evidence_meta_path.resolve()),
            "metadata_sha256": sha256_file(evidence_meta_path),
        },
        "full_c35_score_artifacts": score_artifacts,
        "predictions": str(prediction_path.resolve()),
        "changed_queries": changed,
        "created_at": _utc_now(),
    })
    write_json(_report_path(args.output_dir), report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    _configure_cpu_threads(int(args.cpu_threads))
    preflight_report, split = preflight(args)
    if args.mode == "preflight":
        return preflight_report
    return materialize(args, preflight_report, split)


def main() -> int:
    args = parse_args()
    output = _preflight_path(args.output_dir) if args.mode == "preflight" else _report_path(args.output_dir)
    report = run(args)
    write_json(output, report)
    print(json.dumps({
        "output": str(output.resolve()),
        "stage": report.get("stage"),
        "public_queries": report.get("public_queries"),
        "changed_queries": report.get("changed_queries"),
        "holdout_labels_opened": report.get("holdout_labels_opened", False),
        "public_labels_used": report.get("public_labels_used", False),
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
