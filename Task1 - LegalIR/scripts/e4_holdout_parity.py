#!/usr/bin/env python3
"""E4: one-time locked-holdout parity for the fixed E3-B deployment policy.

The GPU ``prepare`` phase is intentionally label-blind: it creates the
holdout evidence pack, fits exactly two final E2 students from the sealed
complement data, scores every frozen c35 row, and emits slot-5 predictions.
Only the separate CPU ``parity`` phase may open the holdout reference.  The
parity access ledger makes a second evaluation an explicit hard failure.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import e1_evidence_frozen_ce as e1  # noqa: E402
import e2_mil_finetune_oof as e2  # noqa: E402
from e3b_residual_score_gate import (  # noqa: E402
    E3BPolicy,
    apply_policy,
    compact_metrics,
    validate_full_c35_scores,
)
from gpu_autoprobe import GIB, run_auto_probe  # noqa: E402
from gpu_transition_metrics import transition_metrics_device  # noqa: E402
from v27_a_common import (  # noqa: E402
    CANDIDATE_LIMIT,
    EDITABLE_SLOT,
    V27ContractError,
    atomic_torch_save,
    json_sha256,
    load_manifest,
    load_relevance_model,
    load_split,
    read_json,
    sha256_file,
    transition_metrics,
    verify_torch_state_dict,
    write_json,
)


E4_KIND = "legalir_e4_locked_holdout_parity"
E4_SCHEMA_VERSION = 1
E4_PREPARE_CHECKPOINT_KIND = "legalir_e4_deployment_prepare_checkpoint"
E4_DEPLOYMENT_SEEDS = (20260908, 20260909)
E4_LOCKED_POLICY = E3BPolicy(memory_weight=0.0, threshold=0.50)
E4_PARAMETER_COUNT = 2_610_010_000
E4_PARAMETER_LIMIT = 4_000_000_000
EPSILON = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "prepare", "parity"), default="preflight")
    parser.add_argument("--e3b-standard-report", type=Path, required=True)
    parser.add_argument("--e3b-stress-report", type=Path, required=True)
    parser.add_argument("--contexts-zip", type=Path, required=True)
    parser.add_argument("--complement-stage1", type=Path, required=True)
    parser.add_argument("--complement-stage2", type=Path, required=True)
    parser.add_argument("--complement-anchor", type=Path, required=True)
    parser.add_argument("--complement-reference", type=Path, required=True)
    parser.add_argument("--complement-e1-manifest", type=Path, required=True)
    parser.add_argument("--complement-e1-manifest-meta", type=Path, required=True)
    parser.add_argument("--complement-e1-frozen-scores", type=Path, required=True)
    parser.add_argument("--holdout-stage1", type=Path, required=True)
    parser.add_argument("--holdout-stage2", type=Path, required=True)
    parser.add_argument("--holdout-anchor", type=Path, required=True)
    parser.add_argument("--holdout-reference", type=Path, required=True)
    parser.add_argument("--bqb-model", type=Path, required=True)
    parser.add_argument("--reranker-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument("--max-negatives", type=int, default=8)
    parser.add_argument("--bqb-batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.set_defaults(gradient_checkpointing=False)
    parser.add_argument("--auto-probe", dest="auto_probe", action="store_true")
    parser.add_argument("--no-auto-probe", dest="auto_probe", action="store_false")
    parser.set_defaults(auto_probe=True)
    parser.add_argument("--runtime-contract", type=Path)
    parser.add_argument("--micro-batch-candidates", nargs="+", type=int, default=[16])
    parser.add_argument("--score-batch-candidates", nargs="+", type=int, default=[1024, 512, 256, 128, 64, 32, 16])
    parser.add_argument("--vram-cap-gib", type=float)
    parser.add_argument("--cpu-threads", type=int, default=1, help="cap host BLAS/OpenMP workers while CUDA prepares/parity runs")
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _configure_cpu_threads(args: argparse.Namespace) -> None:
    """Bound CPU numerical pools; evidence/tokenization I/O is inherently host-side."""

    threads = int(getattr(args, "cpu_threads", 1))
    if threads < 1:
        raise ValueError("E4 --cpu-threads must be positive")
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
        raise FileNotFoundError(f"E4 required {label} is missing: {path}")


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    _require_file(path, label)
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise V27ContractError(f"E4 {label} must be a JSON object: {path}")
    return dict(payload)


def _policy_payload() -> dict[str, Any]:
    return {
        "locked": True,
        "source": "E3-B fixed-policy diagnostic over all OOF views",
        **E4_LOCKED_POLICY.as_dict(),
    }


def _validate_e3_report(payload: Mapping[str, Any], *, path: Path, mode: str) -> None:
    if payload.get("kind") != "legalir_e3b_nested_mil_full_c35_oof":
        raise V27ContractError(f"E4 requires an E3-B report, not: {path}")
    if payload.get("mode") != mode:
        raise V27ContractError(f"E4 expected E3-B {mode} report: {path}")
    if payload.get("holdout_opened") is not False or payload.get("public_labels_used") is not False:
        raise V27ContractError(f"E3-B report already opened holdout/public: {path}")
    if not isinstance(payload.get("lineage_contract"), dict) or not isinstance(payload.get("lineage_contract_sha256"), str):
        raise V27ContractError(f"E3-B report has no valid lineage contract: {path}")


def validate_e3b_authorization(standard_path: Path, stress_path: Path) -> dict[str, Any]:
    """Accept only a matching completed E3-B standard+stress lineage."""

    standard = _read_json_object(standard_path, "E3-B standard report")
    stress = _read_json_object(stress_path, "E3-B stress report")
    _validate_e3_report(standard, path=standard_path, mode="standard")
    _validate_e3_report(stress, path=stress_path, mode="stress")
    if standard.get("promotion_to_stress_allowed") is not True:
        raise V27ContractError("E3-B standard gate did not authorize stress")
    if stress.get("promotion_to_e4_allowed") is not True:
        raise V27ContractError("E3-B stress gate did not authorize E4")
    if standard["lineage_contract_sha256"] != stress["lineage_contract_sha256"]:
        raise V27ContractError("E3-B standard and stress reports have different lineages")
    e3b = standard["lineage_contract"].get("e3b")
    if not isinstance(e3b, dict) or tuple(e3b.get("neural_seeds", ())) != E4_DEPLOYMENT_SEEDS:
        raise V27ContractError("E3-B deployment seeds do not match the frozen E4 contract")
    return {
        "standard_report": str(standard_path.resolve()),
        "standard_report_sha256": sha256_file(standard_path),
        "stress_report": str(stress_path.resolve()),
        "stress_report_sha256": sha256_file(stress_path),
        "e3b_lineage_contract_sha256": standard["lineage_contract_sha256"],
        "e3b_lineage_contract": dict(standard["lineage_contract"]),
    }


def _e2_args(args: argparse.Namespace) -> argparse.Namespace:
    """Return the exact E2 argument subset for final complement refits."""

    return argparse.Namespace(
        stage1=args.complement_stage1,
        stage2=args.complement_stage2,
        anchor=args.complement_anchor,
        reference=args.complement_reference,
        e1_manifest=args.complement_e1_manifest,
        e1_manifest_meta=args.complement_e1_manifest_meta,
        e1_frozen_scores=args.complement_e1_frozen_scores,
        reranker_model=args.reranker_model,
        output_dir=args.output_dir,
        epochs=int(args.epochs),
        micro_batch=int(args.micro_batch),
        gradient_accumulation=int(args.gradient_accumulation),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        score_batch_size=int(args.score_batch_size),
        max_negatives=int(args.max_negatives),
        device=args.device,
        gradient_checkpointing=bool(args.gradient_checkpointing),
    )


def _evidence_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        contexts_zip=args.contexts_zip,
        stage1=args.holdout_stage1,
        stage2=args.holdout_stage2,
        anchor=args.holdout_anchor,
        bqb_model=args.bqb_model,
        reranker_model=args.reranker_model,
        proposal_device=args.device,
        bqb_batch_size=int(args.bqb_batch_size),
        force_manifest=False,
    )


def _prepare_manifest_path(args: argparse.Namespace) -> Path:
    return args.output_dir / "e4_prepare_label_blind_manifest.json"


def _prepare_report_path(args: argparse.Namespace) -> Path:
    return args.output_dir / "e4_prepare_report.json"


def _parity_report_path(args: argparse.Namespace) -> Path:
    return args.output_dir / "e4_holdout_parity_report.json"


def _access_ledger_path(args: argparse.Namespace) -> Path:
    return args.output_dir / "e4_holdout_access_ledger.json"


def _prepare_checkpoint_path(args: argparse.Namespace) -> Path:
    return args.output_dir / "e4_prepare_progress.json"


def _holdout_evidence_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    return (
        args.output_dir / "e4_holdout_evidence.jsonl",
        args.output_dir / "e4_holdout_evidence.meta.json",
    )


def _prediction_path(args: argparse.Namespace) -> Path:
    return args.output_dir / "e4_holdout_predictions_locked.json"


def _score_path(args: argparse.Namespace, seed: int) -> Path:
    return args.output_dir / "e4_full_scores" / f"holdout_deployment_seed_{seed}.jsonl"


def _weight_path(args: argparse.Namespace, seed: int) -> Path:
    return args.output_dir / "e4_deployment_models" / f"e2_deployment_seed_{seed}.pt"


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


def _holdout_c35(split: Any) -> dict[str, list[str]]:
    return {
        str(qid): [str(doc_id) for doc_id in split.candidates[index][:CANDIDATE_LIMIT]]
        for index, qid in enumerate(split.qids)
    }


def _validate_e4_predictions(split: Any, predictions: Mapping[str, Any]) -> None:
    expected = {str(qid) for qid in split.qids}
    if set(predictions) != expected:
        missing = sorted(expected - set(predictions))
        extra = sorted(set(predictions) - expected)
        raise V27ContractError(f"E4 prediction QIDs mismatch; missing={missing[:3]} extra={extra[:3]}")
    for index, qid in enumerate(split.qids):
        key = str(qid)
        row = predictions[key]
        answer = row.get("answer") if isinstance(row, Mapping) else None
        if not isinstance(answer, list) or len(answer) != 5 or len(set(map(str, answer))) != 5:
            raise V27ContractError(f"{key}: E4 needs five unique IDs")
        answer = [str(value) for value in answer]
        anchor = [str(value) for value in split.anchor[index]]
        c35 = set(map(str, split.candidates[index][:CANDIDATE_LIMIT]))
        if answer[: EDITABLE_SLOT - 1] != anchor[: EDITABLE_SLOT - 1]:
            raise V27ContractError(f"{key}: E4 mutated locked V16b top-4")
        if not set(answer).issubset(c35):
            raise V27ContractError(f"{key}: E4 prediction escaped c35")


def _e4_lineage(
    args: argparse.Namespace,
    *,
    authorization: Mapping[str, Any],
    holdout_split: Any,
) -> dict[str, Any]:
    e2_args = _e2_args(args)
    e1_meta = e2._e1_manifest_contract(e2_args)
    reranker = e2.model_checkpoint_fingerprint(args.reranker_model)
    bqb = e1._checkpoint_fingerprint(args.bqb_model, ("config.json", "modules.json", "model.safetensors"))
    if E4_PARAMETER_COUNT >= E4_PARAMETER_LIMIT:
        raise V27ContractError("E4 pipeline exceeds the 4B model budget")
    return {
        "kind": E4_KIND,
        "schema_version": E4_SCHEMA_VERSION,
        "e3b_authorization": dict(authorization),
        "locked_policy": _policy_payload(),
        "deployment": {
            "seeds": list(E4_DEPLOYMENT_SEEDS),
            "training_split": "all sealed complement queries",
            "early_stopping": "disabled; fixed two-epoch final refit",
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "gradient_accumulation": int(args.gradient_accumulation),
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "max_negatives": int(args.max_negatives),
            "max_length": e2.E2_MAX_LENGTH,
            "temperature": e2.E2_TEMPERATURE,
            "loss_weights": e2.E2_LOSS_WEIGHTS,
        },
        "model_budget": {
            "pipeline_parameters": E4_PARAMETER_COUNT,
            "limit": E4_PARAMETER_LIMIT,
            "under_4b": True,
        },
        "complement": {
            "stage1_sha256": sha256_file(args.complement_stage1),
            "stage2_sha256": sha256_file(args.complement_stage2),
            "anchor_sha256": sha256_file(args.complement_anchor),
            "reference_sha256": sha256_file(args.complement_reference),
            "e1_manifest_sha256": sha256_file(args.complement_e1_manifest),
            "e1_manifest_meta_sha256": sha256_file(args.complement_e1_manifest_meta),
            "e1_frozen_scores_sha256": sha256_file(args.complement_e1_frozen_scores),
            "e1_provenance": e1_meta,
        },
        "holdout_label_blind_inputs": {
            "stage1_sha256": sha256_file(args.holdout_stage1),
            "stage2_sha256": sha256_file(args.holdout_stage2),
            "anchor_sha256": sha256_file(args.holdout_anchor),
            "queries": len(holdout_split.qids),
            "candidate_limit": CANDIDATE_LIMIT,
            "reference_path": str(args.holdout_reference.resolve()),
            "reference_sha256": None,
        },
        "models": {
            "reranker_checkpoint_fingerprint": reranker,
            "bqb_checkpoint_fingerprint": bqb,
            "contexts_zip_sha256": sha256_file(args.contexts_zip),
        },
        "scripts": {
            "e1": sha256_file(Path(e1.__file__)),
            "e2": sha256_file(Path(e2.__file__)),
            "e3b_gate": sha256_file(Path(__file__).with_name("e3b_residual_score_gate.py")),
            "autoprobe": sha256_file(Path(__file__).with_name("gpu_autoprobe.py")),
            "e4": sha256_file(Path(__file__)),
        },
    }


def preflight(args: argparse.Namespace) -> tuple[dict[str, Any], Any]:
    if args.epochs != 2:
        raise V27ContractError("E4 deployment contract fixes E2 training to exactly two epochs")
    if args.gradient_accumulation <= 0 or args.max_negatives <= 0:
        raise ValueError("E4 gradient accumulation and max negatives must be positive")
    if args.bqb_batch_size <= 0 or any(value <= 0 for value in (*args.micro_batch_candidates, *args.score_batch_candidates)):
        raise ValueError("E4 batches must be positive")
    for path, label in (
        (args.contexts_zip, "contexts ZIP"),
        (args.complement_stage1, "complement Stage-1"),
        (args.complement_stage2, "complement Stage-2"),
        (args.complement_anchor, "complement V16b anchor"),
        (args.complement_reference, "complement reference"),
        (args.complement_e1_manifest, "complement E1 manifest"),
        (args.complement_e1_manifest_meta, "complement E1 metadata"),
        (args.complement_e1_frozen_scores, "complement E1 frozen scores"),
        (args.holdout_stage1, "holdout Stage-1"),
        (args.holdout_stage2, "holdout Stage-2"),
        (args.holdout_anchor, "holdout V16b anchor"),
        (args.holdout_reference, "holdout reference"),
    ):
        _require_file(path, label)
    if not args.bqb_model.is_dir() or not args.reranker_model.is_dir():
        raise FileNotFoundError("E4 requires local BQB and Vietnamese reranker checkpoints")
    authorization = validate_e3b_authorization(args.e3b_standard_report, args.e3b_stress_report)
    # ``None`` is crucial: this validates c35/anchor identity without opening qrels.
    holdout_split = load_split("holdout", args.holdout_stage1, args.holdout_stage2, args.holdout_anchor, None)
    if len(holdout_split.qids) != 1000:
        raise V27ContractError(f"E4 expects exactly 1,000 holdout QIDs, found {len(holdout_split.qids)}")
    lineage = _e4_lineage(args, authorization=authorization, holdout_split=holdout_split)
    return {
        "kind": E4_KIND,
        "schema_version": E4_SCHEMA_VERSION,
        "stage": "label_blind_preflight",
        "lineage_contract": lineage,
        "lineage_contract_sha256": json_sha256(lineage),
        "holdout_candidate_data_opened": True,
        "holdout_labels_opened": False,
        "holdout_parity_opened": False,
        "public_labels_used": False,
        "public_predictions_materialized": False,
    }, holdout_split


def _validate_holdout_evidence(
    args: argparse.Namespace,
    split: Any,
    manifest_path: Path,
    meta_path: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    meta = _read_json_object(meta_path, "E4 holdout evidence metadata")
    if meta.get("kind") != e1.E1_KIND or meta.get("split") != "holdout":
        raise V27ContractError("E4 holdout evidence has an unexpected kind or split")
    if meta.get("reference_loaded") is not False or meta.get("holdout_opened") is not False:
        raise V27ContractError("E4 holdout evidence is not label-blind")
    expected_hashes = {
        "stage1_sha256": sha256_file(args.holdout_stage1),
        "stage2_sha256": sha256_file(args.holdout_stage2),
        "anchor_sha256": sha256_file(args.holdout_anchor),
    }
    for key, expected in expected_hashes.items():
        if meta.get(key) != expected:
            raise V27ContractError(f"E4 holdout evidence provenance drift for {key}")
    manifest = load_manifest(manifest_path)
    e2._validate_manifest(split, manifest)
    return manifest


def _materialize_holdout_evidence(args: argparse.Namespace, split: Any) -> tuple[Path, Path, dict[tuple[str, str], dict[str, Any]]]:
    manifest_path, meta_path = _holdout_evidence_paths(args)
    if manifest_path.exists() or meta_path.exists():
        if not manifest_path.is_file() or not meta_path.is_file():
            raise V27ContractError("E4 holdout evidence manifest/meta must either both exist or both be absent")
        return manifest_path, meta_path, _validate_holdout_evidence(args, split, manifest_path, meta_path)
    e1.materialize_manifest(
        _evidence_args(args), manifest_path, meta_path, split_name="holdout",
    )
    return manifest_path, meta_path, _validate_holdout_evidence(args, split, manifest_path, meta_path)


def _load_or_probe_runtime(args: argparse.Namespace, e2_args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    def checked_payload(path: Path) -> tuple[dict[str, Any], Path]:
        payload = _read_json_object(path, "E4 runtime contract")
        if payload.get("kind") != "legalir_e3b_gpu_runtime_contract":
            raise V27ContractError("E4 runtime contract must come from the audited E3-B auto-probe")
        runtime = payload.get("runtime")
        if not isinstance(runtime, dict) or payload.get("runtime_sha256") != json_sha256(runtime):
            raise V27ContractError("E4 runtime contract has an invalid runtime hash")
        return dict(runtime), path

    if args.runtime_contract is not None:
        return checked_payload(args.runtime_contract)
    path = args.output_dir / "e4_runtime_autoprobe.json"
    if path.exists():
        return checked_payload(path)
    if not args.auto_probe:
        raise V27ContractError("E4 requires a runtime contract or --auto-probe before final deployment fits")
    cap = None if args.vram_cap_gib is None else int(float(args.vram_cap_gib) * GIB)
    payload = run_auto_probe(
        e2_args=e2_args,
        output_path=path,
        micro_batches=args.micro_batch_candidates,
        score_batches=args.score_batch_candidates,
        vram_cap_bytes=cap,
    )
    return dict(payload["runtime"]), path


def _apply_runtime(e2_args: argparse.Namespace, runtime: Mapping[str, Any]) -> None:
    required = {
        "micro_batch", "score_batch_size", "gradient_accumulation",
        "gradient_checkpointing", "max_length", "precision", "optimizer",
    }
    if not required.issubset(runtime):
        raise V27ContractError(f"E4 runtime contract misses {sorted(required - set(runtime))}")
    if (
        int(runtime["gradient_accumulation"]) != int(e2_args.gradient_accumulation)
        or bool(runtime["gradient_checkpointing"]) != bool(e2_args.gradient_checkpointing)
        or int(runtime["max_length"]) != e2.E2_MAX_LENGTH
        or runtime["precision"] != "fp32_master_weights_fp16_autocast"
        or runtime["optimizer"] != "AdamW"
    ):
        raise V27ContractError("E4 runtime contract changes the frozen E2 learning configuration")
    e2_args.micro_batch = int(runtime["micro_batch"])
    e2_args.score_batch_size = int(runtime["score_batch_size"])
    if e2_args.micro_batch <= 0 or e2_args.score_batch_size <= 0:
        raise V27ContractError("E4 runtime contract contains non-positive batch sizes")


def _checkpoint_contract(preflight_report: Mapping[str, Any], runtime: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "lineage_contract_sha256": preflight_report["lineage_contract_sha256"],
        "runtime": dict(runtime),
        "runtime_sha256": json_sha256(runtime),
        "deployment_seeds": list(E4_DEPLOYMENT_SEEDS),
        "locked_policy": _policy_payload(),
    }


def _load_or_create_prepare_checkpoint(
    args: argparse.Namespace,
    preflight_report: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    path = _prepare_checkpoint_path(args)
    contract = _checkpoint_contract(preflight_report, runtime)
    contract_sha256 = json_sha256(contract)
    if path.exists() and args.resume:
        state = _read_json_object(path, "E4 prepare checkpoint")
        if state.get("kind") != E4_PREPARE_CHECKPOINT_KIND or state.get("contract_sha256") != contract_sha256:
            raise V27ContractError("E4 prepare checkpoint has a different data/code/runtime contract")
        if not isinstance(state.get("models"), dict):
            raise V27ContractError("E4 prepare checkpoint has no deployment-model map")
        return path, state
    if path.exists() and not args.resume:
        raise FileExistsError("E4 prepare checkpoint exists; use --resume or move the complete E4 output directory")
    state = {
        "kind": E4_PREPARE_CHECKPOINT_KIND,
        "schema_version": 1,
        "contract": contract,
        "contract_sha256": contract_sha256,
        "models": {},
        "complete": False,
    }
    write_json(path, state)
    return path, state


def _save_state_dict(model: Any, path: Path) -> dict[str, Any]:
    state_sha = e2.model_state_sha256(model)
    file_sha = atomic_torch_save(dict(model.state_dict()), path)
    verify_torch_state_dict(path)
    return {
        "path": str(path.resolve()),
        "sha256": file_sha,
        "state_sha256": state_sha,
    }


def _load_state_dict(model: Any, path: Path) -> None:
    import torch

    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older server torch builds
        state = torch.load(path, map_location="cpu")
    model.load_state_dict(state, strict=True)


def _model_from_checkpoint(args: argparse.Namespace, metadata: Mapping[str, Any]) -> tuple[Any, Any]:
    path = Path(str(metadata.get("path", "")))
    if not path.is_file() or sha256_file(path) != metadata.get("sha256"):
        raise V27ContractError("E4 deployment checkpoint is missing or its file hash changed")
    model, tokenizer = load_relevance_model(
        args.reranker_model,
        args.device,
        train=False,
        fp16=args.device == "cuda",
        gradient_checkpointing=bool(args.gradient_checkpointing),
    )
    _load_state_dict(model, path)
    if e2.model_state_sha256(model) != metadata.get("state_sha256"):
        raise V27ContractError("E4 deployment checkpoint state hash changed after reload")
    return model, tokenizer


def _fit_or_load_deployment_model(
    *,
    args: argparse.Namespace,
    e2_args: argparse.Namespace,
    data: e2.E2Data,
    seed: int,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    key = str(seed)
    saved = checkpoint["models"].get(key)
    if isinstance(saved, dict) and isinstance(saved.get("weights"), dict):
        model, tokenizer = _model_from_checkpoint(args, saved["weights"])
        return model, tokenizer, dict(saved)
    all_qids = [str(qid) for qid in data.split.qids]
    model, tokenizer, training = e2.fit_e2_model(
        e2_args,
        data.examples,
        data.split,
        data.manifest,
        all_qids,
        [],
        int(seed),
        f"e4_deployment;seed={seed};fit=all_complement",
    )
    record = {
        "seed": int(seed),
        "fit_queries": len(all_qids),
        "stop_queries": 0,
        "training": training,
        "weights": _save_state_dict(model, _weight_path(args, seed)),
    }
    checkpoint["models"][key] = record
    write_json(checkpoint_path, checkpoint)
    return model, tokenizer, record


def _score_artifact(
    *,
    args: argparse.Namespace,
    split: Any,
    manifest: Mapping[tuple[str, str], Mapping[str, Any]],
    model: Any,
    tokenizer: Any,
    seed: int,
    weights: Mapping[str, Any],
    lineage_sha256: str,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    path = _score_path(args, seed)
    meta_path = path.with_suffix(".meta.json")
    c35_by_qid = _holdout_c35(split)
    if path.is_file() or meta_path.is_file():
        if not path.is_file() or not meta_path.is_file():
            raise V27ContractError("E4 score JSONL/meta must either both exist or both be absent")
        meta = _read_json_object(meta_path, "E4 full-score metadata")
        if (
            meta.get("lineage_contract_sha256") != lineage_sha256
            or meta.get("deployment_seed") != int(seed)
            or meta.get("weights") != dict(weights)
            or meta.get("score_sha256") != sha256_file(path)
        ):
            raise V27ContractError("E4 cached full-c35 scores do not match deployment lineage/weights")
        rows = _read_jsonl(path)
        parsed = validate_full_c35_scores(rows, c35_by_qid=c35_by_qid, expected_qids=list(c35_by_qid))
        meta["metadata_path"] = str(meta_path.resolve())
        meta["metadata_sha256"] = sha256_file(meta_path)
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
        {
            **row,
            "deployment_seed": int(seed),
            "split": "holdout",
            "lineage_contract_sha256": lineage_sha256,
        }
        for row in rows
    ]
    _write_jsonl(path, materialized)
    parsed = validate_full_c35_scores(materialized, c35_by_qid=c35_by_qid, expected_qids=list(c35_by_qid))
    meta = {
        "kind": "legalir_e4_full_c35_scores",
        "schema_version": 1,
        "split": "holdout",
        "deployment_seed": int(seed),
        "score_path": str(path.resolve()),
        "score_sha256": sha256_file(path),
        "score_rows": len(materialized),
        "score_qids_sha256": json_sha256({"qids": list(c35_by_qid)}),
        "lineage_contract_sha256": lineage_sha256,
        "weights": dict(weights),
        "holdout_labels_opened": False,
        "public_labels_used": False,
    }
    write_json(meta_path, meta)
    meta["metadata_path"] = str(meta_path.resolve())
    meta["metadata_sha256"] = sha256_file(meta_path)
    return parsed, meta


def _prediction_payload(predictions: Mapping[str, Mapping[str, Sequence[str]]]) -> dict[str, dict[str, list[str]]]:
    return {
        str(qid): {"answer": [str(value) for value in row["answer"]]}
        for qid, row in predictions.items()
    }


def _write_or_verify_predictions(
    args: argparse.Namespace,
    split: Any,
    predictions: Mapping[str, Mapping[str, Sequence[str]]],
) -> tuple[Path, str]:
    path = _prediction_path(args)
    payload = _prediction_payload(predictions)
    _validate_e4_predictions(split, payload)
    if path.exists():
        existing = _read_json_object(path, "E4 prepared holdout predictions")
        _validate_e4_predictions(split, existing)
        if existing != payload:
            raise V27ContractError("E4 existing holdout predictions differ from the locked policy output")
    else:
        write_json(path, payload)
    return path, sha256_file(path)


def prepare(args: argparse.Namespace, preflight_report: Mapping[str, Any], holdout_split: Any) -> dict[str, Any]:
    if _access_ledger_path(args).exists() or _parity_report_path(args).exists():
        raise V27ContractError("E4 holdout parity has already been opened; prepare artifacts are immutable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _prepare_manifest_path(args)
    write_json(manifest_path, preflight_report)
    evidence_path, evidence_meta_path, holdout_manifest = _materialize_holdout_evidence(args, holdout_split)
    e2_args = _e2_args(args)
    runtime, runtime_path = _load_or_probe_runtime(args, e2_args)
    _apply_runtime(e2_args, runtime)
    # Complement labels are allowed only after E4's label-blind manifest is sealed.
    data = e2.load_e2_data(e2_args)
    if len(data.split.qids) != 1197:
        raise V27ContractError(f"E4 expected the sealed 1,197-query complement, found {len(data.split.qids)}")
    checkpoint_path, checkpoint = _load_or_create_prepare_checkpoint(args, preflight_report, runtime)
    per_seed: list[dict[str, dict[str, dict[str, Any]]]] = []
    model_records: list[dict[str, Any]] = []
    score_artifacts: list[dict[str, Any]] = []
    for seed in E4_DEPLOYMENT_SEEDS:
        model = tokenizer = None
        try:
            model, tokenizer, record = _fit_or_load_deployment_model(
                args=args,
                e2_args=e2_args,
                data=data,
                seed=seed,
                checkpoint_path=checkpoint_path,
                checkpoint=checkpoint,
            )
            scores, score_meta = _score_artifact(
                args=args,
                split=holdout_split,
                manifest=holdout_manifest,
                model=model,
                tokenizer=tokenizer,
                seed=seed,
                weights=record["weights"],
                lineage_sha256=preflight_report["lineage_contract_sha256"],
            )
            per_seed.append(scores)
            model_records.append(record)
            score_artifacts.append(score_meta)
        finally:
            del model, tokenizer
            gc.collect()
            if args.device == "cuda":
                try:
                    import torch

                    torch.cuda.empty_cache()
                except ImportError:  # pragma: no cover - defensive cleanup
                    pass
    predictions, decisions = apply_policy(
        holdout_split,
        list(range(len(holdout_split.qids))),
        per_seed,
        {},
        E4_LOCKED_POLICY,
    )
    prediction_path, prediction_sha256 = _write_or_verify_predictions(args, holdout_split, predictions)
    checkpoint["complete"] = True
    checkpoint["prediction_sha256"] = prediction_sha256
    write_json(checkpoint_path, checkpoint)
    return {
        "kind": E4_KIND,
        "schema_version": E4_SCHEMA_VERSION,
        "stage": "prepared_label_blind_holdout_predictions",
        "lineage_contract": preflight_report["lineage_contract"],
        "lineage_contract_sha256": preflight_report["lineage_contract_sha256"],
        "locked_policy": _policy_payload(),
        "runtime": runtime,
        "runtime_contract": str(runtime_path.resolve()),
        "holdout_evidence": {
            "path": str(evidence_path.resolve()),
            "sha256": sha256_file(evidence_path),
            "metadata_path": str(evidence_meta_path.resolve()),
            "metadata_sha256": sha256_file(evidence_meta_path),
        },
        "deployment_models": model_records,
        "full_c35_score_artifacts": score_artifacts,
        "predictions": str(prediction_path.resolve()),
        "predictions_sha256": prediction_sha256,
        "prepared_decisions": {
            "queries": len(decisions),
            "changed_queries": sum(bool(row.get("changed")) for row in decisions.values()),
        },
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "complete": True,
            "resumable": True,
        },
        "holdout_candidate_data_opened": True,
        "holdout_labels_opened": False,
        "holdout_parity_opened": False,
        "public_labels_used": False,
        "public_predictions_materialized": False,
    }


def _load_prepared_predictions(args: argparse.Namespace, preflight_report: Mapping[str, Any], holdout_split: Any) -> tuple[dict[str, Any], dict[str, dict[str, list[str]]]]:
    report = _read_json_object(_prepare_report_path(args), "E4 prepare report")
    if (
        report.get("kind") != E4_KIND
        or report.get("stage") != "prepared_label_blind_holdout_predictions"
        or report.get("lineage_contract_sha256") != preflight_report["lineage_contract_sha256"]
        or report.get("locked_policy") != _policy_payload()
        or report.get("holdout_labels_opened") is not False
    ):
        raise V27ContractError("E4 prepare report is not compatible with this one-time parity contract")
    prediction_path = Path(str(report.get("predictions", "")))
    predictions = _read_json_object(prediction_path, "E4 prepared predictions")
    if report.get("predictions_sha256") != sha256_file(prediction_path):
        raise V27ContractError("E4 prepared prediction hash changed after the label-blind phase")
    _validate_e4_predictions(holdout_split, predictions)
    return report, predictions


def _open_holdout_ledger(args: argparse.Namespace, preflight_report: Mapping[str, Any], prepare_report: Mapping[str, Any]) -> Path:
    ledger_path = _access_ledger_path(args)
    report_path = _parity_report_path(args)
    if ledger_path.exists() or report_path.exists():
        raise V27ContractError(
            "E4 holdout parity was already opened or completed; this contract forbids a second evaluation"
        )
    ledger = {
        "kind": "legalir_e4_holdout_access_ledger",
        "schema_version": 1,
        "state": "opened",
        "opened_at": _utc_now(),
        "holdout_access_count": 1,
        "lineage_contract_sha256": preflight_report["lineage_contract_sha256"],
        "prepared_predictions_sha256": prepare_report["predictions_sha256"],
        "policy": _policy_payload(),
        "public_labels_used": False,
    }
    write_json(ledger_path, ledger)
    return ledger_path


def _promotion_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    gain = float(metrics["gain"])
    rescues = int(metrics["successful_rescues"])
    destructive = int(metrics["destructive_changes"])
    return {
        "gain_strictly_positive": gain > EPSILON,
        "rescues_exceed_destructive_changes": rescues > destructive,
        "pass": bool(gain > EPSILON and rescues > destructive),
    }


def parity(args: argparse.Namespace, preflight_report: Mapping[str, Any], holdout_without_labels: Any) -> dict[str, Any]:
    if args.device != "cuda":
        raise V27ContractError("E4 parity is CUDA-bound in the GPU-only runtime profile")
    prepare_report, predictions = _load_prepared_predictions(args, preflight_report, holdout_without_labels)
    ledger_path = _open_holdout_ledger(args, preflight_report, prepare_report)
    # This is the sole E4 code path that opens the locked reference/qrels.
    holdout = load_split(
        "holdout", args.holdout_stage1, args.holdout_stage2,
        args.holdout_anchor, args.holdout_reference,
    )
    _validate_e4_predictions(holdout, predictions)
    transition = compact_metrics(transition_metrics_device(holdout, predictions, device=args.device))
    promotion = _promotion_gate(transition)
    report = {
        "kind": E4_KIND,
        "schema_version": E4_SCHEMA_VERSION,
        "stage": "locked_holdout_parity_complete",
        "lineage_contract_sha256": preflight_report["lineage_contract_sha256"],
        "e3b_authorization": preflight_report["lineage_contract"]["e3b_authorization"],
        "locked_policy": _policy_payload(),
        "prepared_predictions": prepare_report["predictions"],
        "prepared_predictions_sha256": prepare_report["predictions_sha256"],
        "holdout_reference": {
            "path": str(args.holdout_reference.resolve()),
            "sha256": sha256_file(args.holdout_reference),
            "queries": len(holdout.qids),
        },
        "metrics_vs_v16b": transition,
        "promotion": {
            **promotion,
            "promotion_to_e5_allowed": bool(promotion["pass"]),
            "reason": "strict positive Recall@5 gain and more rescues than destructive changes",
        },
        "holdout_candidate_data_opened": True,
        "holdout_labels_opened": True,
        "holdout_parity_opened": True,
        "holdout_access_count": 1,
        "public_labels_used": False,
        "public_predictions_materialized": False,
        "e5_executed": False,
    }
    report_path = _parity_report_path(args)
    write_json(report_path, report)
    ledger = _read_json_object(ledger_path, "E4 holdout access ledger")
    ledger.update({
        "state": "complete",
        "completed_at": _utc_now(),
        "parity_report": str(report_path.resolve()),
        "parity_report_sha256": sha256_file(report_path),
        "promotion_to_e5_allowed": bool(promotion["pass"]),
    })
    write_json(ledger_path, ledger)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    _configure_cpu_threads(args)
    preflight_report, holdout_split = preflight(args)
    if args.mode == "preflight":
        return preflight_report
    if args.mode == "prepare":
        return prepare(args, preflight_report, holdout_split)
    return parity(args, preflight_report, holdout_split)


def main() -> int:
    args = parse_args()
    output = {
        "preflight": args.output_dir / "e4_preflight.json",
        "prepare": _prepare_report_path(args),
        "parity": _parity_report_path(args),
    }[args.mode]
    if args.mode == "parity" and output.exists():
        raise FileExistsError("E4 locked-holdout parity report already exists; a second parity run is forbidden")
    report = run(args)
    write_json(output, report)
    print(json.dumps({
        "output": str(output.resolve()),
        "stage": report.get("stage"),
        "holdout_labels_opened": report.get("holdout_labels_opened", False),
        "promotion_to_e5_allowed": report.get("promotion", {}).get("promotion_to_e5_allowed", False),
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
