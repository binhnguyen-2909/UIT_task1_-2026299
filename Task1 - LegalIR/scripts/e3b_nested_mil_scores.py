#!/usr/bin/env python3
"""E3-B: leak-audited nested E2 refits plus full-c35 residual slot-5 gate.

E3-B never opens holdout or public data.  For every regime, outer fold and
policy split seed it fits two fresh E2 MIL students on A-fit, uses only A-stop
for epoch selection, scores C and V with those same frozen weights, chooses a
small E3-B policy using C labels, then applies it to V.  Score exports retain
every c35 MIL and passage logit needed to audit that dependency graph.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e2_mil_finetune_oof as e2  # noqa: E402
from e3b_residual_score_gate import (  # noqa: E402
    ABSTAIN_POLICY,
    E3BPolicy,
    apply_policy,
    compact_metrics,
    paired_group_confidence_interval,
    select_calibration_policy,
    validate_full_c35_scores,
)
from gpu_autoprobe import GIB, run_auto_probe  # noqa: E402
from v27_a_common import (  # noqa: E402
    CANDIDATE_LIMIT,
    EDITABLE_SLOT,
    V27ContractError,
    group_folds,
    json_sha256,
    load_anchor,
    read_json,
    sha256_file,
    transition_metrics,
    write_json,
)


E3B_KIND = "legalir_e3b_nested_mil_full_c35_oof"
E3B_SCHEMA_VERSION = 1
E3B_CHECKPOINT_KIND = "legalir_e3b_nested_mil_fold_checkpoint"
E3B_CHECKPOINT_SCHEMA_VERSION = 1
E3B_INNER_SEEDS = (20260910, 20260911)
E3B_NEURAL_SEEDS = (20260908, 20260909)
STANDARD_REGIME = "standard"
STRESS_REGIMES = ("law_source_family", "query_template_family", "near_duplicate_family")
FORBIDDEN_TERMS = ("holdout", "public", "submission", "locked")
EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "standard", "stress"), default="preflight")
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--e1-manifest", type=Path, required=True)
    parser.add_argument("--e1-manifest-meta", type=Path, required=True)
    parser.add_argument("--e1-frozen-scores", type=Path, required=True)
    parser.add_argument("--memory-features", type=Path, required=True)
    parser.add_argument("--reranker-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--standard-report", type=Path, help="required before MODE=stress")
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--inner-seeds", nargs="+", type=int, default=list(E3B_INNER_SEEDS))
    parser.add_argument("--neural-seeds", nargs="+", type=int, default=list(E3B_NEURAL_SEEDS))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument("--max-negatives", type=int, default=8)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.set_defaults(gradient_checkpointing=False)
    parser.add_argument("--auto-probe", dest="auto_probe", action="store_true")
    parser.add_argument("--no-auto-probe", dest="auto_probe", action="store_false")
    parser.set_defaults(auto_probe=True)
    parser.add_argument("--runtime-contract", type=Path, help="reuse a previously written E3-B auto-probe contract")
    parser.add_argument("--micro-batch-candidates", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--score-batch-candidates", nargs="+", type=int, default=[16, 32, 64, 128, 256])
    parser.add_argument("--vram-cap-gib", type=float, help="optional allocated-GPU cap; never reserves the GPU")
    parser.add_argument("--retain-best-weights", action="store_true", help="retain every E2 student weight artifact; requires substantial disk")
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _safe_path(path: Path, label: str) -> None:
    value = str(path).replace("\\", "/").casefold()
    if any(term in value for term in FORBIDDEN_TERMS):
        raise V27ContractError(f"E3-B is complement-only; {label} may not reference holdout/public: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"E3-B required input is missing: {path}")


def _regimes(args: argparse.Namespace) -> tuple[str, ...]:
    return (STANDARD_REGIME,) if args.mode == "standard" else STRESS_REGIMES


def _output_path(args: argparse.Namespace) -> Path:
    name = "e3b_preflight.json" if args.mode == "preflight" else f"e3b_{args.mode}_report.json"
    return args.report_path or args.output_dir / name


def _checkpoint_path(args: argparse.Namespace) -> Path:
    return args.checkpoint_path or args.output_dir / f"e3b_{args.mode}_progress.json"


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
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
                raise V27ContractError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def _load_memory_prior(path: Path, expected_qids: set[str]) -> tuple[dict[str, dict[str, int]], dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, dict) or payload.get("kind") != "legalir_v16_qrel_memory_features":
        raise V27ContractError("E3-B requires V16b memory feature artifact")
    raw_rows = payload.get("queries")
    if not isinstance(raw_rows, list):
        raise V27ContractError("V16b memory features must contain queries")
    ranks: dict[str, dict[str, int]] = {}
    for raw in raw_rows:
        if not isinstance(raw, dict) or "query_id" not in raw:
            raise V27ContractError("Invalid V16b memory feature query")
        qid = str(raw["query_id"])
        docs = raw.get("memory", {}).get("documents") if isinstance(raw.get("memory"), dict) else None
        by_doc: dict[str, int] = {}
        if isinstance(docs, list):
            for doc in docs:
                if not isinstance(doc, dict) or "doc_id" not in doc:
                    continue
                rank = doc.get("memory_rank")
                if isinstance(rank, int) and 1 <= rank <= CANDIDATE_LIMIT:
                    by_doc[str(doc["doc_id"])] = int(rank)
        ranks[qid] = by_doc
    missing = sorted(expected_qids - set(ranks))
    if missing:
        raise V27ContractError(f"V16b memory artifact misses E3-B anchor QIDs: {missing[:5]}")
    leakage = payload.get("leakage_policy")
    if not isinstance(leakage, dict):
        leakage = {}
    memory_prior_allowed = bool(
        leakage.get("qid_overlap") == []
        and leakage.get("labels_used_for_query_retrieval") is False
        and leakage.get("query_answers_used_only_for_oracle_metrics") is True
    )
    return ranks, {
        "memory_features_sha256": sha256_file(path),
        "memory_prior_allowed": memory_prior_allowed,
        "leakage_policy": leakage,
    }


def _lineage_contract(args: argparse.Namespace, memory_contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": E3B_KIND,
        "schema_version": E3B_SCHEMA_VERSION,
        "split": "complement",
        "stage1_sha256": sha256_file(args.stage1),
        "stage2_sha256": sha256_file(args.stage2),
        "anchor_sha256": sha256_file(args.anchor),
        "reference_sha256": sha256_file(args.reference),
        "e1_manifest_sha256": sha256_file(args.e1_manifest),
        "e1_manifest_meta_sha256": sha256_file(args.e1_manifest_meta),
        "e1_frozen_scores_sha256": sha256_file(args.e1_frozen_scores),
        "memory": dict(memory_contract),
        "reranker_checkpoint_fingerprint": e2.model_checkpoint_fingerprint(args.reranker_model),
        "scripts": {
            "e2": sha256_file(Path(e2.__file__)),
            "e3b": sha256_file(Path(__file__)),
            "gate": sha256_file(Path(__file__).with_name("e3b_residual_score_gate.py")),
            "autoprobe": sha256_file(Path(__file__).with_name("gpu_autoprobe.py")),
        },
        "e2": {
            "epochs": int(args.epochs),
            "gradient_accumulation": int(args.gradient_accumulation),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "max_negatives": int(args.max_negatives),
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "max_length": e2.E2_MAX_LENGTH,
            "temperature": e2.E2_TEMPERATURE,
            "loss_weights": e2.E2_LOSS_WEIGHTS,
        },
        "e3b": {
            "inner_seeds": [int(seed) for seed in args.inner_seeds],
            "neural_seeds": [int(seed) for seed in args.neural_seeds],
            "outer_folds": 5,
            "calibration_fraction": 0.20,
            "early_stop_fraction": 0.15,
            "candidate_limit": CANDIDATE_LIMIT,
            "frozen_top_k": EDITABLE_SLOT - 1,
            "memory_weights": [0.0, 0.10] if bool(memory_contract["memory_prior_allowed"]) else [0.0],
            "thresholds": [0.0, 0.25, 0.50, 1.00],
            "disagreement_weight": 0.25,
        },
    }


def _preflight(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    for path, label in (
        (args.stage1, "stage1"), (args.stage2, "stage2"), (args.anchor, "anchor"),
        (args.reference, "reference"), (args.e1_manifest, "E1 manifest"),
        (args.e1_manifest_meta, "E1 metadata"), (args.e1_frozen_scores, "E1 frozen scores"),
        (args.memory_features, "V16b memory features"),
    ):
        _safe_path(path, label)
    if not args.reranker_model.is_dir():
        raise FileNotFoundError(f"E3-B reranker checkpoint is missing: {args.reranker_model}")
    if len(set(args.inner_seeds)) != 2 or len(set(args.neural_seeds)) != 2:
        raise ValueError("E3-B requires exactly two distinct inner seeds and two distinct neural seeds")
    if any(value <= 0 for value in (*args.micro_batch_candidates, *args.score_batch_candidates)):
        raise ValueError("E3-B auto-probe batches must be positive")
    # Verifies E1 artifact provenance without reading complement qrels.
    e1_meta = e2._e1_manifest_contract(args)
    anchor = load_anchor(args.anchor)
    memory_ranks, memory_contract = _load_memory_prior(args.memory_features, set(anchor))
    lineage = _lineage_contract(args, memory_contract)
    manifest = {
        "kind": E3B_KIND,
        "schema_version": E3B_SCHEMA_VERSION,
        "stage": "label_blind_preflight",
        "lineage_contract": lineage,
        "lineage_contract_sha256": json_sha256(lineage),
        "e1_provenance": e1_meta,
        "queries": len(anchor),
        "planned_fit_count": 20 if args.mode == "standard" else 60 if args.mode == "stress" else 0,
        "gpu_required_for_fit": True,
        "holdout_opened": False,
        "public_labels_used": False,
        "labels_opened_after_manifest": False,
    }
    return manifest, memory_ranks


def _subset(split: Any, indices: Sequence[int]) -> Any:
    from v27_a_common import V27Split

    return V27Split(
        split.name,
        [split.qids[index] for index in indices],
        [split.questions[index] for index in indices],
        [split.candidates[index] for index in indices],
        [split.anchor[index] for index in indices],
        [split.neural_scores[index] for index in indices],
        [split.result_rows[index] for index in indices],
        [split.references[index] for index in indices] if split.references is not None else None,
        [split.source_candidates[index] for index in indices],
        [split.source_results[index] for index in indices],
    )


def _qids(split: Any, indices: Sequence[int]) -> list[str]:
    return [str(split.qids[int(index)]) for index in indices]


def _qid_hash(qids: Sequence[str]) -> str:
    return json_sha256({"qids": [str(qid) for qid in qids]})


def _a_fit_stop_indices(a_indices: np.ndarray, groups: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    local = np.arange(len(a_indices), dtype=np.int32)
    group_values = groups[a_indices]
    if len(a_indices) < 20 or len(np.unique(group_values)) < 3:
        raise V27ContractError("E3-B A partition cannot form a group-disjoint A-fit/A-stop split")
    fit_local, stop_local = next(
        GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=int(seed)).split(local, groups=group_values)
    )
    return a_indices[fit_local].astype(np.int32), a_indices[stop_local].astype(np.int32)


def _validate_partition(
    *,
    split: Any,
    groups: np.ndarray,
    outer_train: np.ndarray,
    validation: np.ndarray,
    a_fit: np.ndarray,
    a_stop: np.ndarray,
    calibration: np.ndarray,
) -> dict[str, Any]:
    partitions = {
        "a_fit": a_fit,
        "a_stop": a_stop,
        "calibration": calibration,
        "validation": validation,
    }
    all_indices = set(range(len(split.qids)))
    seen: set[int] = set()
    seen_groups: set[str] = set()
    for name, values in partitions.items():
        member = {int(value) for value in values.tolist()}
        if not member:
            raise V27ContractError(f"E3-B {name} partition is empty")
        if seen & member:
            raise V27ContractError(f"E3-B partitions overlap at {name}")
        seen |= member
        partition_groups = {str(groups[index]) for index in member}
        if seen_groups & partition_groups:
            raise V27ContractError(f"E3-B group leakage into {name}")
        seen_groups |= partition_groups
    if seen != all_indices:
        raise V27ContractError("E3-B partitions do not exactly cover complement")
    if set(map(int, outer_train.tolist())) != set(map(int, np.concatenate((a_fit, a_stop, calibration)).tolist())):
        raise V27ContractError("E3-B A-fit/A-stop/calibration do not reconstruct outer-train")
    return {
        name: {
            "queries": len(values),
            "qids": _qids(split, values),
            "qids_sha256": _qid_hash(_qids(split, values)),
            "groups_sha256": json_sha256({"groups": [str(groups[index]) for index in values]}),
        }
        for name, values in partitions.items()
    }


def make_partitions(
    *,
    split: Any,
    groups: np.ndarray,
    outer_train: np.ndarray,
    validation: np.ndarray,
    inner_seed: int,
    outer_fold: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Create A-fit/A-stop/C/V with disjoint query and group membership."""

    local = np.arange(len(outer_train), dtype=np.int32)
    a_local, c_local = next(
        GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=int(inner_seed) + int(outer_fold)).split(
            local, groups=groups[outer_train]
        )
    )
    a_indices = outer_train[a_local].astype(np.int32)
    calibration = outer_train[c_local].astype(np.int32)
    a_fit, a_stop = _a_fit_stop_indices(
        a_indices, groups, seed=int(inner_seed) + 10_000 + int(outer_fold),
    )
    metadata = _validate_partition(
        split=split,
        groups=groups,
        outer_train=outer_train,
        validation=validation,
        a_fit=a_fit,
        a_stop=a_stop,
        calibration=calibration,
    )
    return {
        "a_fit": a_fit,
        "a_stop": a_stop,
        "calibration": calibration,
        "validation": validation.astype(np.int32),
    }, metadata


def _weight_path(args: argparse.Namespace, *, regime: str, inner_seed: int, outer_fold: int, neural_seed: int) -> Path:
    return args.output_dir / "e3b_weights" / regime / f"inner_{inner_seed}_fold_{outer_fold + 1}_neural_{neural_seed}.pt"


def _retain_weights(model: Any, path: Path | None) -> dict[str, Any]:
    """Optionally retain exact best weights; hashing always occurs in memory."""

    state_sha = e2.model_state_sha256(model)
    if path is None:
        return {"state_sha256": state_sha, "retained_path": None, "retained_sha256": None}
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(model.state_dict(), temporary)
    os.replace(temporary, path)
    return {
        "state_sha256": state_sha,
        "retained_path": str(path.resolve()),
        "retained_sha256": sha256_file(path),
    }


def _score_path(
    args: argparse.Namespace,
    *,
    regime: str,
    partition: str,
    inner_seed: int,
    outer_fold: int,
    neural_seed: int,
) -> Path:
    name = f"{partition}_inner_{inner_seed}_fold_{outer_fold + 1}_neural_{neural_seed}.jsonl"
    return args.output_dir / "e3b_full_scores" / regime / name


def _export_scores(
    *,
    args: argparse.Namespace,
    data: e2.E2Data,
    indices: np.ndarray,
    model: Any,
    tokenizer: Any,
    regime: str,
    partition: str,
    inner_seed: int,
    outer_fold: int,
    neural_seed: int,
    fit_seed: int,
    partitions: Mapping[str, Any],
    lineage_sha256: str,
    weight_info: Mapping[str, Any],
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    rows = e2.score_full_c35(
        data.split, indices.tolist(), data.manifest, model, tokenizer,
        device=args.device, batch_size=int(args.score_batch_size),
    )
    common = {
        "regime": regime,
        "partition": partition,
        "outer_fold": int(outer_fold),
        "inner_seed": int(inner_seed),
        "neural_seed": int(neural_seed),
        "fit_seed": int(fit_seed),
    }
    materialized = [{**row, **common} for row in rows]
    path = _score_path(
        args, regime=regime, partition=partition, inner_seed=inner_seed,
        outer_fold=outer_fold, neural_seed=neural_seed,
    )
    _write_jsonl(path, materialized)
    qids = _qids(data.split, indices)
    c35_by_qid = {
        str(data.split.qids[index]): [str(value) for value in data.split.candidates[index][:CANDIDATE_LIMIT]]
        for index in indices
    }
    parsed = validate_full_c35_scores(materialized, c35_by_qid=c35_by_qid, expected_qids=qids)
    metadata = {
        "kind": "legalir_e3b_full_c35_scores",
        "schema_version": 1,
        **common,
        "score_path": str(path.resolve()),
        "score_sha256": sha256_file(path),
        "score_rows": len(materialized),
        "score_qids": qids,
        "score_qids_sha256": _qid_hash(qids),
        "partitions": dict(partitions),
        "lineage_contract_sha256": lineage_sha256,
        "weights": dict(weight_info),
        "holdout_opened": False,
        "public_labels_used": False,
    }
    meta_path = path.with_suffix(".meta.json")
    write_json(meta_path, metadata)
    metadata["metadata_path"] = str(meta_path.resolve())
    metadata["metadata_sha256"] = sha256_file(meta_path)
    return parsed, metadata


def _verify_score_artifact(artifact: Mapping[str, Any], *, lineage_sha256: str) -> None:
    path = Path(str(artifact.get("score_path", "")))
    meta_path = Path(str(artifact.get("metadata_path", "")))
    if not path.is_file() or not meta_path.is_file():
        raise V27ContractError("E3-B completed fold has a missing full-score artifact")
    if sha256_file(path) != artifact.get("score_sha256") or sha256_file(meta_path) != artifact.get("metadata_sha256"):
        raise V27ContractError("E3-B full-score artifact hash changed after checkpoint")
    metadata = read_json(meta_path)
    if not isinstance(metadata, dict) or metadata.get("lineage_contract_sha256") != lineage_sha256:
        raise V27ContractError("E3-B full-score artifact has incompatible lineage")


def _checkpoint_contract(
    *,
    manifest: Mapping[str, Any],
    args: argparse.Namespace,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "lineage_contract_sha256": manifest["lineage_contract_sha256"],
        "mode": args.mode,
        "regimes": list(_regimes(args)),
        "runtime": dict(runtime),
        "retain_best_weights": bool(args.retain_best_weights),
    }


def _load_or_create_checkpoint(
    *,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    path = _checkpoint_path(args)
    contract = _checkpoint_contract(manifest=manifest, args=args, runtime=runtime)
    contract_sha = json_sha256(contract)
    if args.resume and path.exists():
        state = read_json(path)
        if not isinstance(state, dict) or state.get("kind") != E3B_CHECKPOINT_KIND:
            raise V27ContractError(f"Invalid E3-B progress checkpoint: {path}")
        if state.get("contract_sha256") != contract_sha:
            raise V27ContractError("E3-B checkpoint runtime/data/code contract changed; use --no-resume")
        completed = state.get("completed_folds")
        if not isinstance(completed, dict):
            raise V27ContractError("E3-B checkpoint has no completed fold map")
        print(json.dumps({"event": "e3b_checkpoint_resume", "completed_folds": len(completed), "path": str(path.resolve())}, ensure_ascii=False), flush=True)
        return path, state
    state = {
        "kind": E3B_CHECKPOINT_KIND,
        "schema_version": E3B_CHECKPOINT_SCHEMA_VERSION,
        "contract": contract,
        "contract_sha256": contract_sha,
        "completed_folds": {},
        "complete": False,
    }
    write_json(path, state)
    print(json.dumps({"event": "e3b_checkpoint_start", "path": str(path.resolve())}, ensure_ascii=False), flush=True)
    return path, state


def _runtime_contract(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Get an immutable runtime choice before any E3-B fit starts."""

    if args.runtime_contract is not None:
        payload = read_json(args.runtime_contract)
        if not isinstance(payload, dict) or payload.get("kind") != "legalir_e3b_gpu_runtime_contract":
            raise V27ContractError(f"Invalid E3-B runtime contract: {args.runtime_contract}")
        runtime = payload.get("runtime")
        if not isinstance(runtime, dict) or payload.get("runtime_sha256") != json_sha256(runtime):
            raise V27ContractError("E3-B runtime contract has invalid runtime hash")
        return dict(runtime), payload
    if not args.auto_probe:
        raise V27ContractError("E3-B requires --auto-probe or a matching --runtime-contract before GPU fits")
    cap = None if args.vram_cap_gib is None else int(float(args.vram_cap_gib) * GIB)
    probe_path = args.output_dir / f"e3b_{args.mode}_runtime_autoprobe.json"
    payload = run_auto_probe(
        e2_args=args,
        output_path=probe_path,
        micro_batches=args.micro_batch_candidates,
        score_batches=args.score_batch_candidates,
        vram_cap_bytes=cap,
    )
    return dict(payload["runtime"]), payload


def _apply_runtime(args: argparse.Namespace, runtime: Mapping[str, Any]) -> None:
    required = {"micro_batch", "score_batch_size", "gradient_accumulation", "gradient_checkpointing", "max_length", "precision", "optimizer"}
    if not required.issubset(runtime):
        raise V27ContractError(f"E3-B runtime contract misses {sorted(required - set(runtime))}")
    if int(runtime["max_length"]) != e2.E2_MAX_LENGTH or runtime["precision"] != "fp32_master_weights_fp16_autocast" or runtime["optimizer"] != "AdamW":
        raise V27ContractError("E3-B runtime contract changes the registered E2 training path")
    if int(runtime["gradient_accumulation"]) != int(args.gradient_accumulation):
        raise V27ContractError("E3-B runtime contract changes gradient accumulation")
    if bool(runtime["gradient_checkpointing"]) != bool(args.gradient_checkpointing):
        raise V27ContractError("E3-B runtime contract changes gradient checkpointing")
    args.micro_batch = int(runtime["micro_batch"])
    args.score_batch_size = int(runtime["score_batch_size"])
    if args.micro_batch <= 0 or args.score_batch_size <= 0:
        raise V27ContractError("E3-B runtime contract has a non-positive batch")


def _validate_standard_gate(path: Path, *, lineage_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"E3-B stress requires completed standard report: {path}")
    report = read_json(path)
    if not isinstance(report, dict) or report.get("kind") != E3B_KIND:
        raise V27ContractError("E3-B standard report has unexpected kind")
    if report.get("mode") != "standard" or report.get("lineage_contract_sha256") != lineage_sha256:
        raise V27ContractError("E3-B standard report does not match this code/data/model lineage")
    if report.get("promotion_to_stress_allowed") is not True:
        raise V27ContractError("E3-B standard gate did not pass; stress allocation is not authorized")
    return report


def _completed_fold_key(regime: str, inner_seed: int, outer_fold: int) -> str:
    return f"{regime}|inner={inner_seed}|fold={outer_fold}"


def _restore_completed_fold(
    saved: Mapping[str, Any],
    *,
    validation_qids: Sequence[str],
    lineage_sha256: str,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, Any]]:
    if saved.get("validation_qids") != list(validation_qids):
        raise V27ContractError("E3-B saved fold has different outer-validation QIDs")
    predictions = saved.get("predictions")
    fold_report = saved.get("fold_report")
    if not isinstance(predictions, dict) or set(predictions) != set(validation_qids) or not isinstance(fold_report, dict):
        raise V27ContractError("E3-B saved fold is incomplete")
    artifacts = fold_report.get("score_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 4:
        raise V27ContractError("E3-B saved fold is missing its two-seed C/V score artifacts")
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise V27ContractError("E3-B score artifact record is invalid")
        _verify_score_artifact(artifact, lineage_sha256=lineage_sha256)
    return predictions, fold_report


def _fit_score_neural_seed(
    *,
    args: argparse.Namespace,
    data: e2.E2Data,
    partitions: Mapping[str, np.ndarray],
    partition_metadata: Mapping[str, Any],
    lineage_sha256: str,
    regime: str,
    inner_seed: int,
    outer_fold: int,
    neural_seed: int,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    fit_seed = int(neural_seed) + int(outer_fold)
    label = f"regime={regime};inner={inner_seed};neural={neural_seed};fold={outer_fold + 1}/5"
    print(json.dumps({
        "event": "e3b_neural_fit_start", "run": label,
        "a_fit_queries": len(partitions["a_fit"]), "a_stop_queries": len(partitions["a_stop"]),
    }, ensure_ascii=False), flush=True)
    model, tokenizer, training = e2.fit_e2_model(
        args,
        data.examples,
        data.split,
        data.manifest,
        _qids(data.split, partitions["a_fit"]),
        _qids(data.split, partitions["a_stop"]),
        fit_seed,
        label,
    )
    try:
        weights = _retain_weights(
            model,
            _weight_path(args, regime=regime, inner_seed=inner_seed, outer_fold=outer_fold, neural_seed=neural_seed)
            if args.retain_best_weights else None,
        )
        calibration_scores, calibration_artifact = _export_scores(
            args=args, data=data, indices=partitions["calibration"], model=model, tokenizer=tokenizer,
            regime=regime, partition="calibration", inner_seed=inner_seed, outer_fold=outer_fold,
            neural_seed=neural_seed, fit_seed=fit_seed, partitions=partition_metadata,
            lineage_sha256=lineage_sha256, weight_info=weights,
        )
        validation_scores, validation_artifact = _export_scores(
            args=args, data=data, indices=partitions["validation"], model=model, tokenizer=tokenizer,
            regime=regime, partition="validation", inner_seed=inner_seed, outer_fold=outer_fold,
            neural_seed=neural_seed, fit_seed=fit_seed, partitions=partition_metadata,
            lineage_sha256=lineage_sha256, weight_info=weights,
        )
        if calibration_artifact["weights"]["state_sha256"] != validation_artifact["weights"]["state_sha256"]:
            raise V27ContractError("E3-B C/V scores were not produced by the same frozen neural weights")
        neural_report = {
            "neural_seed": int(neural_seed),
            "fit_seed": fit_seed,
            "training": training,
            "weights": weights,
        }
        print(json.dumps({"event": "e3b_neural_fit_complete", "run": label, "weight_sha256": weights["state_sha256"]}, ensure_ascii=False), flush=True)
        return calibration_scores, validation_scores, [calibration_artifact, validation_artifact], neural_report
    finally:
        del model, tokenizer
        gc.collect()
        if args.device == "cuda":
            import torch
            torch.cuda.empty_cache()


def _run_inner_seed(
    *,
    args: argparse.Namespace,
    data: e2.E2Data,
    groups: np.ndarray,
    memory_ranks: Mapping[str, Mapping[str, int]],
    memory_prior_allowed: bool,
    lineage_sha256: str,
    regime: str,
    inner_seed: int,
    checkpoint_path: Path,
    checkpoint_state: dict[str, Any],
) -> dict[str, Any]:
    oof: dict[str, dict[str, list[str]]] = {}
    fold_reports: list[dict[str, Any]] = []
    for outer_fold, (outer_train, validation) in enumerate(group_folds(groups)):
        validation_qids = _qids(data.split, validation)
        key = _completed_fold_key(regime, inner_seed, outer_fold)
        saved = checkpoint_state["completed_folds"].get(key)
        if saved is not None:
            predictions, report = _restore_completed_fold(
                saved, validation_qids=validation_qids, lineage_sha256=lineage_sha256,
            )
            oof.update(predictions)
            fold_reports.append(report)
            print(json.dumps({"event": "e3b_fold_resumed", "run": key}, ensure_ascii=False), flush=True)
            continue
        partitions, partition_metadata = make_partitions(
            split=data.split,
            groups=groups,
            outer_train=outer_train,
            validation=validation,
            inner_seed=inner_seed,
            outer_fold=outer_fold,
        )
        print(json.dumps({
            "event": "e3b_fold_start", "run": key,
            "a_fit_queries": len(partitions["a_fit"]), "a_stop_queries": len(partitions["a_stop"]),
            "calibration_queries": len(partitions["calibration"]), "validation_queries": len(partitions["validation"]),
        }, ensure_ascii=False), flush=True)
        c_seed_scores: list[dict[str, dict[str, dict[str, Any]]]] = []
        v_seed_scores: list[dict[str, dict[str, dict[str, Any]]]] = []
        artifacts: list[dict[str, Any]] = []
        neural_reports: list[dict[str, Any]] = []
        for neural_seed in args.neural_seeds:
            c_scores, v_scores, pair_artifacts, neural_report = _fit_score_neural_seed(
                args=args, data=data, partitions=partitions, partition_metadata=partition_metadata,
                lineage_sha256=lineage_sha256, regime=regime, inner_seed=inner_seed,
                outer_fold=outer_fold, neural_seed=int(neural_seed),
            )
            c_seed_scores.append(c_scores)
            v_seed_scores.append(v_scores)
            artifacts.extend(pair_artifacts)
            neural_reports.append(neural_report)
        calibration_split = _subset(data.split, partitions["calibration"])
        policy, calibration_metrics, policy_trace = select_calibration_policy(
            calibration_split, c_seed_scores, memory_ranks,
            memory_prior_allowed=memory_prior_allowed,
        )
        validation_split = _subset(data.split, partitions["validation"])
        predictions, decisions = apply_policy(validation_split, list(range(len(validation_split.qids))), v_seed_scores, memory_ranks, policy)
        validation_metrics = compact_metrics(transition_metrics(validation_split, predictions))
        fold_report = {
            "outer_fold": int(outer_fold),
            "partitions": partition_metadata,
            "neural_models": neural_reports,
            "score_artifacts": artifacts,
            "selected_policy": {"kind": "v16b_anchor_abstain"} if policy is None else policy.as_dict(),
            "calibration": calibration_metrics,
            "policy_trace": policy_trace,
            "validation": validation_metrics,
            "decisions": decisions,
        }
        oof.update(predictions)
        fold_reports.append(fold_report)
        checkpoint_state["completed_folds"][key] = {
            "validation_qids": validation_qids,
            "predictions": predictions,
            "fold_report": fold_report,
        }
        write_json(checkpoint_path, checkpoint_state)
        print(json.dumps({"event": "e3b_fold_complete", "run": key, "selected_policy": fold_report["selected_policy"], "validation": validation_metrics}, ensure_ascii=False), flush=True)
    if set(oof) != set(map(str, data.split.qids)):
        raise V27ContractError(f"E3-B {regime}/inner={inner_seed}: OOF does not exactly cover complement")
    ordered = {str(qid): oof[str(qid)] for qid in data.split.qids}
    metrics = compact_metrics(transition_metrics(data.split, ordered))
    return {
        "inner_seed": int(inner_seed),
        "folds": fold_reports,
        "metrics": metrics,
        "paired_group_ci": paired_group_confidence_interval(
            data.split, ordered, groups, seed=int(inner_seed),
        ),
        "predictions": ordered,
    }


def _gate(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_seed = [
        bool(
            float(run["metrics"]["gain"]) > EPSILON
            and int(run["metrics"]["successful_rescues"]) > int(run["metrics"]["destructive_changes"])
        )
        for run in runs
    ]
    return {
        "per_inner_seed_pass": per_seed,
        "pass": bool(all(per_seed)),
        "minimum_gain": min(float(run["metrics"]["gain"]) for run in runs),
        "minimum_rescue_minus_destructive": min(
            int(run["metrics"]["successful_rescues"]) - int(run["metrics"]["destructive_changes"])
            for run in runs
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs <= 0 or args.gradient_accumulation <= 0 or args.max_negatives <= 0:
        raise ValueError("E3-B E2 hyperparameters must be positive")
    preflight, memory_ranks = _preflight(args)
    preflight_path = args.output_dir / f"e3b_{args.mode}_label_blind_manifest.json"
    write_json(preflight_path, preflight)
    if args.mode == "preflight":
        return preflight

    if args.device != "cuda":
        raise ValueError("E3-B refits require CUDA; CPU is supported only for preflight/tests")
    if args.mode == "stress":
        if args.standard_report is None:
            raise ValueError("MODE=stress requires --standard-report")
        _validate_standard_gate(args.standard_report, lineage_sha256=preflight["lineage_contract_sha256"])
    runtime, runtime_report = _runtime_contract(args)
    _apply_runtime(args, runtime)
    data = e2.load_e2_data(args)
    if set(map(str, data.split.qids)) != set(load_anchor(args.anchor)):
        raise V27ContractError("E3-B loaded complement QIDs differ from frozen anchor")
    for regime in _regimes(args):
        if regime not in data.groups_by_regime:
            raise V27ContractError(f"E3-B unknown grouping regime: {regime}")
    checkpoint_path, checkpoint_state = _load_or_create_checkpoint(
        args=args, manifest=preflight, runtime=runtime,
    )
    reports: dict[str, list[dict[str, Any]]] = {}
    for regime in _regimes(args):
        reports[regime] = [
            _run_inner_seed(
                args=args,
                data=data,
                groups=data.groups_by_regime[regime],
                memory_ranks=memory_ranks,
                memory_prior_allowed=bool(preflight["lineage_contract"]["memory"]["memory_prior_allowed"]),
                lineage_sha256=preflight["lineage_contract_sha256"],
                regime=regime,
                inner_seed=int(inner_seed),
                checkpoint_path=checkpoint_path,
                checkpoint_state=checkpoint_state,
            )
            for inner_seed in args.inner_seeds
        ]
    checkpoint_state["complete"] = True
    write_json(checkpoint_path, checkpoint_state)
    gates = {regime: _gate(runs) for regime, runs in reports.items()}
    standard_allowed = bool(gates.get(STANDARD_REGIME, {}).get("pass"))
    stress_allowed = bool(args.mode == "standard" and standard_allowed)
    e4_allowed = bool(args.mode == "stress" and all(gate["pass"] for gate in gates.values()))
    return {
        "kind": E3B_KIND,
        "schema_version": E3B_SCHEMA_VERSION,
        "status": "nested_oof_only_not_promoted_to_holdout",
        "mode": args.mode,
        "split": "complement",
        "lineage_contract": preflight["lineage_contract"],
        "lineage_contract_sha256": preflight["lineage_contract_sha256"],
        "label_blind_manifest": str(preflight_path.resolve()),
        "labels_opened_after_manifest": True,
        "holdout_opened": False,
        "public_labels_used": False,
        "runtime": runtime,
        "runtime_report": None if runtime_report is None else str((args.runtime_contract or args.output_dir / f"e3b_{args.mode}_runtime_autoprobe.json").resolve()),
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "completed_folds": len(checkpoint_state["completed_folds"]),
            "resumable": True,
        },
        "candidate_limit": CANDIDATE_LIMIT,
        "frozen_top_k": EDITABLE_SLOT - 1,
        "policy_count_including_abstain": 9 if bool(preflight["lineage_contract"]["memory"]["memory_prior_allowed"]) else 5,
        "regimes": reports,
        "gates": gates,
        "promotion_to_stress_allowed": stress_allowed,
        "promotion_to_e4_allowed": e4_allowed,
    }


def main() -> int:
    args = parse_args()
    output = _output_path(args)
    if output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite E3-B output: {output}")
    report = run(args)
    write_json(output, report)
    print(json.dumps({
        "output": str(output.resolve()),
        "stage": report.get("stage", report.get("status")),
        "holdout_opened": False,
        "promotion_to_stress_allowed": report.get("promotion_to_stress_allowed", False),
        "promotion_to_e4_allowed": report.get("promotion_to_e4_allowed", False),
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
