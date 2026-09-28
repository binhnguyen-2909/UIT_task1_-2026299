#!/usr/bin/env python3
"""E3-C: nested, incumbent-aware LoRA duo comparator on complement only.

E3-C is deliberately a parallel research branch, not a replacement for the
locked E3-B policy used by E4.  Each outer fold starts from a fixed, retained
E3-B student (neural seed 20260908), fits only a small LoRA comparator on
A-fit, early-stops on A-stop, chooses a fixed-margin policy on C, and reports
only V.  The script refuses to open holdout/public inputs and refuses to run
without the exact per-fold E3-B source checkpoints and full-c35 score exports.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e2_mil_finetune_oof as e2  # noqa: E402
import e3b_nested_mil_scores as e3b  # noqa: E402
from e3b_residual_score_gate import E3BPolicy, apply_policy as apply_e3b_policy, validate_full_c35_scores  # noqa: E402
from e3c_duo_gate import (  # noqa: E402
    ABSTAIN_POLICY,
    E3CPolicy,
    apply_policy,
    compact_metrics,
    select_calibration_policy,
    soft_delta_target,
    validate_duo_scores,
)
from gpu_transition_metrics import transition_metrics_device  # noqa: E402
from v27_a_common import (  # noqa: E402
    CANDIDATE_LIMIT,
    EDITABLE_SLOT,
    V27ContractError,
    group_folds,
    json_sha256,
    load_anchor,
    read_json,
    row_blocks,
    sha256_file,
    transition_metrics,
    verify_torch_checkpoint,
    write_json,
)
from v27_a_common import atomic_torch_save  # noqa: E402


E3C_KIND = "legalir_e3c_nested_duo_lora_oof"
E3C_SCHEMA_VERSION = 1
E3C_CHECKPOINT_KIND = "legalir_e3c_nested_duo_lora_progress"
E3C_SOURCE_NEURAL_SEED = 20260908
E3C_INNER_SEEDS = (20260910, 20260911)
STANDARD_REGIME = "standard"
STRESS_REGIMES = ("law_source_family", "query_template_family", "near_duplicate_family")
FORBIDDEN_TERMS = ("holdout", "public", "submission", "locked")
MAX_LENGTH = 1536
QUESTION_TOKEN_BUDGET = 384
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
EPSILON = 1.0e-12


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
    parser.add_argument("--reranker-model", type=Path, required=True)
    parser.add_argument("--e3b-output-dir", type=Path, required=True)
    parser.add_argument("--e3b-standard-report", type=Path, required=True)
    parser.add_argument("--e3b-stress-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--standard-report", type=Path, help="required before MODE=stress")
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--inner-seeds", nargs="+", type=int, default=list(E3C_INNER_SEEDS))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument("--micro-batch-candidates", nargs="+", type=int, default=[16])
    parser.add_argument("--score-batch-candidates", nargs="+", type=int, default=[1024, 512, 256, 128, 64, 32, 16, 8])
    parser.add_argument("--auto-probe", dest="auto_probe", action="store_true")
    parser.add_argument("--no-auto-probe", dest="auto_probe", action="store_false")
    parser.set_defaults(auto_probe=True)
    parser.add_argument("--runtime-contract", type=Path, help="reuse matching E3-C auto-probe contract")
    parser.add_argument("--vram-cap-gib", type=float)
    parser.add_argument("--cpu-threads", type=int, default=1, help="cap torch/BLAS CPU workers; tokenization remains host-side")
    parser.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.set_defaults(gradient_checkpointing=False)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _safe_file(path: Path, label: str) -> None:
    value = str(path).replace("\\", "/").casefold()
    if any(term in value for term in FORBIDDEN_TERMS):
        raise V27ContractError(f"E3-C is complement-only; {label} may not refer to holdout/public: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"E3-C required input is missing: {path}")


def _configure_cpu_threads(args: argparse.Namespace) -> None:
    """Prevent host BLAS/OpenMP pools from contending with CUDA work.

    Text tokenization and JSON I/O necessarily run on the host, but E3-C never
    needs a wide CPU numerical pool.  Keep this explicit in the run contract
    rather than relying on whatever thread count the server inherited.
    """

    threads = int(getattr(args, "cpu_threads", 1))
    if threads < 1:
        raise ValueError("E3-C --cpu-threads must be positive")
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_MAX_THREADS"] = str(threads)
    try:
        import torch
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        # Inter-op threads may already be configured by an embedding host; the
        # shell environment still keeps all subsequent worker pools bounded.
        pass


def _regimes(args: argparse.Namespace) -> tuple[str, ...]:
    return (STANDARD_REGIME,) if args.mode == "standard" else STRESS_REGIMES


def _output_path(args: argparse.Namespace) -> Path:
    return args.report_path or args.output_dir / ("e3c_preflight.json" if args.mode == "preflight" else f"e3c_{args.mode}_report.json")


def _checkpoint_path(args: argparse.Namespace) -> Path:
    return args.checkpoint_path or args.output_dir / f"e3c_{args.mode}_progress.json"


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            sink.write("\n")
    os.replace(temp, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise V27ContractError(f"{path}:{line_number}: expected JSON object")
                rows.append(row)
    return rows


def _qids(split: Any, indices: Sequence[int]) -> list[str]:
    return [str(split.qids[int(index)]) for index in indices]


def _subset(split: Any, indices: Sequence[int]) -> Any:
    return e3b._subset(split, indices)


def _source_weight_path(root: Path, regime: str, inner_seed: int, outer_fold: int) -> Path:
    return root / "e3b_weights" / regime / f"inner_{inner_seed}_fold_{outer_fold + 1}_neural_{E3C_SOURCE_NEURAL_SEED}.pt"


def _artifact_path(root: Path, regime: str, partition: str, inner_seed: int, outer_fold: int, neural_seed: int) -> Path:
    return root / "e3b_full_scores" / regime / f"{partition}_inner_{inner_seed}_fold_{outer_fold + 1}_neural_{neural_seed}.jsonl"


def _validate_e3b_report(report: Mapping[str, Any], *, mode: str) -> None:
    if report.get("kind") != e3b.E3B_KIND or report.get("mode") != mode:
        raise V27ContractError(f"E3-C requires an E3-B {mode} report")
    if report.get("split") != "complement" or report.get("holdout_opened") is not False:
        raise V27ContractError("E3-C refuses E3-B provenance that is not complement-only")
    if mode == "standard" and report.get("promotion_to_stress_allowed") is not True:
        raise V27ContractError("E3-B standard report did not authorize stress")
    if mode == "stress" and report.get("promotion_to_e4_allowed") is not True:
        raise V27ContractError("E3-B stress report did not pass its promotion gate")


def _e3b_folds(report: Mapping[str, Any], regime: str, inner_seed: int) -> list[Mapping[str, Any]]:
    raw = report.get("regimes", {}).get(regime)
    if not isinstance(raw, list):
        raise V27ContractError(f"E3-B report misses regime={regime}")
    for run in raw:
        if isinstance(run, Mapping) and int(run.get("inner_seed", -1)) == int(inner_seed):
            folds = run.get("folds")
            if isinstance(folds, list) and len(folds) == 5:
                return folds
    raise V27ContractError(f"E3-B report misses inner seed={inner_seed} in {regime}")


def _weight_record(fold: Mapping[str, Any]) -> Mapping[str, Any]:
    models = fold.get("neural_models")
    if not isinstance(models, list):
        raise V27ContractError("E3-B fold misses neural model provenance")
    for model in models:
        if isinstance(model, Mapping) and int(model.get("neural_seed", -1)) == E3C_SOURCE_NEURAL_SEED:
            weights = model.get("weights")
            if isinstance(weights, Mapping):
                return weights
    raise V27ContractError("E3-B fold has no fixed E3-C source neural seed")


def _validate_e3b_sources(
    *, root: Path, reports: Mapping[str, Mapping[str, Any]], inner_seeds: Sequence[int], regimes: Sequence[str]
) -> dict[str, dict[str, Any]]:
    source_map: dict[str, dict[str, Any]] = {}
    for regime in regimes:
        report = reports["standard"] if regime == STANDARD_REGIME else reports["stress"]
        for inner_seed in inner_seeds:
            folds = _e3b_folds(report, regime, int(inner_seed))
            for outer_fold, fold in enumerate(folds):
                if int(fold.get("outer_fold", -1)) != outer_fold:
                    raise V27ContractError("E3-B outer-fold ordering/provenance drift")
                weights = _weight_record(fold)
                expected = _source_weight_path(root, regime, int(inner_seed), outer_fold)
                if not expected.is_file():
                    raise FileNotFoundError(
                        "E3-C needs retained E3-B source weights but cannot find "
                        f"{expected}. E3-B must have been run with RETAIN_BEST_WEIGHTS=true; "
                        "do not substitute a base checkpoint."
                    )
                stored = weights.get("retained_sha256")
                if not isinstance(stored, str) or sha256_file(expected) != stored:
                    raise V27ContractError(f"E3-B retained weight hash mismatch: {expected}")
                state = weights.get("state_sha256")
                if not isinstance(state, str) or len(state) != 64:
                    raise V27ContractError("E3-B source state hash is absent; source lineage is not auditable")
                key = f"{regime}|inner={inner_seed}|fold={outer_fold}"
                source_map[key] = {
                    "path": str(expected.resolve()), "file_sha256": stored,
                    "state_sha256": state, "neural_seed": E3C_SOURCE_NEURAL_SEED,
                }
    return source_map


def _lineage(args: argparse.Namespace, sources: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": E3C_KIND,
        "schema_version": E3C_SCHEMA_VERSION,
        "split": "complement",
        "stage1_sha256": sha256_file(args.stage1),
        "stage2_sha256": sha256_file(args.stage2),
        "anchor_sha256": sha256_file(args.anchor),
        "reference_sha256": sha256_file(args.reference),
        "e1_manifest_sha256": sha256_file(args.e1_manifest),
        "e1_manifest_meta_sha256": sha256_file(args.e1_manifest_meta),
        "e1_frozen_scores_sha256": sha256_file(args.e1_frozen_scores),
        "reranker_checkpoint_fingerprint": e2.model_checkpoint_fingerprint(args.reranker_model),
        "e3b_standard_report_sha256": sha256_file(args.e3b_standard_report),
        "e3b_stress_report_sha256": sha256_file(args.e3b_stress_report),
        "e3b_source_neural_seed": E3C_SOURCE_NEURAL_SEED,
        "source_weights": dict(sources),
        "comparator": {
            "architecture": "query_candidate_incumbent_duo_lora",
            "max_length": MAX_LENGTH,
            "question_token_budget": QUESTION_TOKEN_BUDGET,
            "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT, "target_modules": ["query", "value"], "modules_to_save": ["classifier"]},
            "epochs": int(args.epochs), "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay), "gradient_accumulation": int(args.gradient_accumulation),
            "gradient_checkpointing": bool(args.gradient_checkpointing), "precision": "bf16",
            "target": "soft=(1+(I[c in G]-I[i in G])/|G|)/2", "prediction": "2*sigmoid(logit)-1",
            "thresholds": [0.00, 0.01, 0.02, 0.05, 0.10], "default": "v16b_anchor_abstain",
            "cpu_threads": int(args.cpu_threads),
        },
        "model_budget": {"e3b_two_students_parameters_billion": 2.61, "duo_comparator_parameters_billion": 0.76, "total_parameters_billion": 3.37, "under_4b": True},
        "scripts": {
            "e2": sha256_file(Path(e2.__file__)), "e3b": sha256_file(Path(e3b.__file__)),
            "e3c": sha256_file(Path(__file__)), "gate": sha256_file(Path(__file__).with_name("e3c_duo_gate.py")),
        },
    }


def _preflight(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    for path, label in ((args.stage1, "stage1"), (args.stage2, "stage2"), (args.anchor, "anchor"),
                        (args.reference, "reference"), (args.e1_manifest, "E1 manifest"),
                        (args.e1_manifest_meta, "E1 metadata"), (args.e1_frozen_scores, "E1 scores"),
                        (args.e3b_standard_report, "E3-B standard report"), (args.e3b_stress_report, "E3-B stress report")):
        _safe_file(path, label)
    if not args.reranker_model.is_dir() or not args.e3b_output_dir.is_dir():
        raise FileNotFoundError("E3-C reranker model or E3-B output directory is missing")
    if tuple(args.inner_seeds) != E3C_INNER_SEEDS:
        raise V27ContractError("E3-C requires exactly preregistered inner seeds 20260910 and 20260911")
    if args.epochs < 1 or args.epochs > 2 or args.gradient_accumulation <= 0:
        raise ValueError("E3-C requires 1..2 epochs and a positive gradient accumulation")
    if any(value <= 0 for value in (*args.micro_batch_candidates, *args.score_batch_candidates)):
        raise ValueError("E3-C auto-probe batches must be positive")
    e2_meta = e2._e1_manifest_contract(args)  # Label-blind E1 provenance only.
    standard, stress = read_json(args.e3b_standard_report), read_json(args.e3b_stress_report)
    if not isinstance(standard, Mapping) or not isinstance(stress, Mapping):
        raise V27ContractError("E3-B reports must be JSON objects")
    _validate_e3b_report(standard, mode="standard")
    _validate_e3b_report(stress, mode="stress")
    requested = (STANDARD_REGIME,) if args.mode in ("preflight", "standard") else STRESS_REGIMES
    sources = _validate_e3b_sources(root=args.e3b_output_dir, reports={"standard": standard, "stress": stress}, inner_seeds=args.inner_seeds, regimes=requested)
    anchor = load_anchor(args.anchor)
    lineage = _lineage(args, sources)
    return {
        "kind": E3C_KIND, "schema_version": E3C_SCHEMA_VERSION, "stage": "label_blind_preflight",
        "lineage_contract": lineage, "lineage_contract_sha256": json_sha256(lineage),
        "e1_provenance": e2_meta, "queries": len(anchor),
        "planned_comparator_fits": 10 if args.mode == "standard" else 30 if args.mode == "stress" else 0,
        "e3b_policy_locked_for_e4": {"threshold": 0.5, "memory_weight": 0.0},
        "e4_is_not_modified_or_authorized_by_e3c": True,
        "gpu_required_for_fit": True, "holdout_opened": False, "public_labels_used": False,
        "labels_opened_after_manifest": False,
    }, sources


class DuoEncodingCache:
    """Bounded CPU token cache; model input is always max 1536 tokens."""

    def __init__(self, tokenizer: Any, *, max_entries: int = 100_000):
        self.tokenizer, self.max_entries = tokenizer, int(max_entries)
        self.entries: OrderedDict[tuple[str, str], dict[str, list[int]]] = OrderedDict()

    def encode(self, pairs: Sequence[tuple[str, str]], device: str) -> Mapping[str, Any]:
        values: list[dict[str, list[int]]] = []
        for query, context in pairs:
            key = (query, context)
            item = self.entries.get(key)
            if item is None:
                raw = self.tokenizer(query, context, truncation=True, max_length=MAX_LENGTH, add_special_tokens=True)
                item = {name: list(value) for name, value in raw.items() if isinstance(value, list)}
                self.entries[key] = item
                if len(self.entries) > self.max_entries:
                    self.entries.popitem(last=False)
            else:
                self.entries.move_to_end(key)
            values.append(item)
        batch = self.tokenizer.pad(values, padding=True, return_tensors="pt")
        return {name: value.to(device) for name, value in batch.items()}


def _shorten(tokenizer: Any, text: str, budget: int) -> str:
    ids = tokenizer(str(text), add_special_tokens=False).get("input_ids", [])[: max(0, int(budget))]
    return tokenizer.decode(ids, skip_special_tokens=True)


def _duo_context(tokenizer: Any, candidate_row: Mapping[str, Any], incumbent_row: Mapping[str, Any]) -> str:
    question_reserve = QUESTION_TOKEN_BUDGET
    overhead = 32
    each = max(64, (MAX_LENGTH - question_reserve - overhead) // 2)
    candidate = _shorten(tokenizer, "\n".join(row_blocks(candidate_row)), each)
    incumbent = _shorten(tokenizer, "\n".join(row_blocks(incumbent_row)), each)
    return f"[CANDIDATE_EVIDENCE] {candidate} [INCUMBENT_SLOT5_EVIDENCE] {incumbent}"


def _eligible_pairs(data: e2.E2Data, indices: Sequence[int], *, with_targets: bool) -> dict[str, list[dict[str, Any]]]:
    """All non-top4/non-incumbent c35 candidates; no nomination pre-filter."""

    pairs: dict[str, list[dict[str, Any]]] = {}
    if with_targets and data.split.references is None:
        raise V27ContractError("E3-C supervised pair construction requires complement qrels")
    for index in indices:
        qid = str(data.split.qids[int(index)])
        anchor = [str(value) for value in data.split.anchor[int(index)]]
        c35 = [str(value) for value in data.split.candidates[int(index)][:CANDIDATE_LIMIT]]
        incumbent = anchor[EDITABLE_SLOT - 1]
        incumbent_row = data.manifest[(qid, incumbent)]
        eligible_incumbent = not bool(incumbent_row.get("empty_body")) and bool(row_blocks(incumbent_row))
        raw: list[dict[str, Any]] = []
        for candidate in c35:
            if candidate in set(anchor[:EDITABLE_SLOT]):
                continue
            candidate_row = data.manifest[(qid, candidate)]
            eligible = bool(eligible_incumbent and not bool(candidate_row.get("empty_body")) and row_blocks(candidate_row))
            row: dict[str, Any] = {
                "qid": qid, "query": str(data.split.questions[int(index)]), "candidate_doc_id": candidate,
                "incumbent_doc_id": incumbent, "eligible": eligible,
                "candidate_row": candidate_row, "incumbent_row": incumbent_row,
            }
            if with_targets and eligible:
                delta, target = soft_delta_target(gold=data.split.references[int(index)], incumbent=incumbent, candidate=candidate)
                row.update({"delta": delta, "target": target})
            raw.append(row)
        scoreable = [row for row in raw if row["eligible"]]
        for row in scoreable:
            row["query_weight"] = 1.0 / float(len(scoreable))
        pairs[qid] = raw
    return pairs


def _load_lora_model(args: argparse.Namespace, source: Mapping[str, Any]) -> tuple[Any, Any]:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    e2._set_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.reranker_model, local_files_only=True, use_fast=True)
    # Load/check the retained E3-B FP32 state before casting.  Hashing a BF16
    # cast would falsely claim that a correct E3-B source had drifted.
    model = AutoModelForSequenceClassification.from_pretrained(args.reranker_model, local_files_only=True)
    state = torch.load(source["path"], map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise V27ContractError("E3-C E3-B source checkpoint is not a state dict")
    model.load_state_dict(state, strict=True)
    model.to(args.device)
    actual_state = e2.model_state_sha256(model)
    if actual_state != source["state_sha256"]:
        raise V27ContractError("E3-C loaded E3-B source state does not match its recorded state SHA256")
    model.to(dtype=torch.bfloat16)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        from v27_a_common import enable_gradient_checkpointing
        enable_gradient_checkpointing(model)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=["query", "value"], modules_to_save=["classifier"], bias="none",
    ))
    model.train()
    return model, tokenizer


def _raw_logit(output: Any) -> Any:
    value = output.logits if hasattr(output, "logits") else output
    return value[..., 0] if value.shape[-1] == 1 else value[..., 1] - value[..., 0]


def _forward_logits(model: Any, cache: DuoEncodingCache, rows: Sequence[Mapping[str, Any]], *, device: str, micro_batch: int) -> Any:
    import torch

    chunks: list[Any] = []
    for start in range(0, len(rows), max(1, micro_batch)):
        part = rows[start:start + max(1, micro_batch)]
        pairs = [(str(row["query"]), str(row["context"])) for row in part]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            chunks.append(_raw_logit(model(**cache.encode(pairs, device))).float())
    return torch.cat(chunks, dim=0)


def _materialize_contexts(tokenizer: Any, pairs: Mapping[str, Sequence[dict[str, Any]]]) -> None:
    for rows in pairs.values():
        for row in rows:
            if row["eligible"]:
                row["query"] = _shorten(tokenizer, str(row["query"]), QUESTION_TOKEN_BUDGET)
                row["context"] = _duo_context(tokenizer, row["candidate_row"], row["incumbent_row"])


def _weighted_loss(model: Any, cache: DuoEncodingCache, rows: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> Any:
    import torch.nn.functional as F
    scoreable = [row for row in rows if row["eligible"]]
    if not scoreable:
        return None
    logits = _forward_logits(model, cache, scoreable, device=args.device, micro_batch=args.micro_batch)
    targets = logits.new_tensor([float(row["target"]) for row in scoreable])
    weights = logits.new_tensor([float(row["query_weight"]) for row in scoreable])
    return (F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * weights).sum() / weights.sum().clamp_min(1.0e-12)


def _eval_loss(model: Any, cache: DuoEncodingCache, pairs: Mapping[str, Sequence[dict[str, Any]]], args: argparse.Namespace) -> float:
    import torch
    values: list[float] = []
    model.eval()
    with torch.inference_mode():
        for rows in pairs.values():
            value = _weighted_loss(model, cache, rows, args)
            if value is not None:
                values.append(float(value.detach().cpu()))
    model.train()
    return float(np.mean(values)) if values else math.inf


def _adapter_state(model: Any) -> dict[str, Any]:
    from peft import get_peft_model_state_dict
    return {name: value.detach().cpu().clone() for name, value in get_peft_model_state_dict(model).items()}


def _load_adapter_state(model: Any, state: Mapping[str, Any]) -> None:
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(model, dict(state), adapter_name="default")


def _rng_state() -> dict[str, Any]:
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def _restore_rng(state: Mapping[str, Any]) -> None:
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def _training_checkpoint_path(args: argparse.Namespace, regime: str, inner_seed: int, outer_fold: int) -> Path:
    return args.output_dir / "e3c_training_checkpoints" / regime / f"inner_{inner_seed}_fold_{outer_fold + 1}.pt"


def _train_comparator(
    *, args: argparse.Namespace, pairs_fit: Mapping[str, Sequence[dict[str, Any]]], pairs_stop: Mapping[str, Sequence[dict[str, Any]]],
    source: Mapping[str, Any], regime: str, inner_seed: int, outer_fold: int,
) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    seed = int(inner_seed) + 100_000 + int(outer_fold)
    e2._set_seed(seed)
    model, tokenizer = _load_lora_model(args, source)
    cache = DuoEncodingCache(tokenizer)
    _materialize_contexts(tokenizer, pairs_fit)
    _materialize_contexts(tokenizer, pairs_stop)
    optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    checkpoint_path = _training_checkpoint_path(args, regime, inner_seed, outer_fold)
    start_epoch, best_loss, best_state, history = 0, math.inf, None, []
    if args.resume and checkpoint_path.is_file():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved.get("kind") == "legalir_e3c_duo_training_checkpoint" and saved.get("source_file_sha256") == source["file_sha256"]:
            _load_adapter_state(model, saved["adapter_state"])
            optimizer.load_state_dict(saved["optimizer_state"])
            _restore_rng(saved["rng_state"])
            start_epoch = int(saved["next_epoch"])
            best_loss = float(saved["best_stop_loss"])
            best_state = saved.get("best_adapter_state")
            history = list(saved.get("history", []))
            print(json.dumps({"event": "e3c_training_checkpoint_resume", "path": str(checkpoint_path.resolve()), "next_epoch": start_epoch}, ensure_ascii=False), flush=True)
    qids = [qid for qid, rows in pairs_fit.items() if any(row["eligible"] for row in rows)]
    if not qids:
        raise V27ContractError("E3-C A-fit contains no scoreable duo comparisons")
    for epoch in range(start_epoch, args.epochs):
        order = list(qids)
        random.Random(seed + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        accumulated, losses = 0, []
        started = time.monotonic()
        print(json.dumps({"event": "e3c_epoch_start", "run": f"regime={regime};inner={inner_seed};fold={outer_fold + 1}/5", "epoch": epoch + 1, "epochs": args.epochs, "trainable_queries": len(order), "micro_batch": args.micro_batch, "gradient_accumulation": args.gradient_accumulation, "precision": "bf16"}, ensure_ascii=False), flush=True)
        for position, qid in enumerate(order, 1):
            model.train()
            loss = _weighted_loss(model, cache, pairs_fit[qid], args)
            if loss is None:
                continue
            (loss / args.gradient_accumulation).backward()
            accumulated += 1
            losses.append(float(loss.detach().cpu()))
            if accumulated >= args.gradient_accumulation or position == len(order):
                torch.nn.utils.clip_grad_norm_((param for param in model.parameters() if param.requires_grad), 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); accumulated = 0
            if position % 100 == 0 or position == len(order):
                elapsed = time.monotonic() - started
                eta = elapsed / position * (len(order) - position)
                print(json.dumps({"event": "e3c_train_progress", "run": f"regime={regime};inner={inner_seed};fold={outer_fold + 1}/5", "epoch": epoch + 1, "queries_done": position, "queries_total": len(order), "elapsed_seconds": round(elapsed, 1), "eta_epoch_seconds": round(eta, 1)}, ensure_ascii=False), flush=True)
        stop_loss = _eval_loss(model, cache, pairs_stop, args)
        if stop_loss + EPSILON < best_loss:
            best_loss, best_state = stop_loss, _adapter_state(model)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)) if losses else math.inf, "a_stop_bce": stop_loss, "elapsed_seconds": time.monotonic() - started})
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save({"kind": "legalir_e3c_duo_training_checkpoint", "source_file_sha256": source["file_sha256"], "next_epoch": epoch + 1, "position": 0, "optimizer_state": optimizer.state_dict(), "adapter_state": _adapter_state(model), "best_adapter_state": best_state, "best_stop_loss": best_loss, "rng_state": _rng_state(), "history": history}, checkpoint_path)
        verify_torch_checkpoint(
            checkpoint_path,
            required_keys=("kind", "next_epoch", "optimizer_state", "adapter_state", "rng_state", "history"),
        )
        print(json.dumps({"event": "e3c_epoch_complete", "run": f"regime={regime};inner={inner_seed};fold={outer_fold + 1}/5", "epoch": epoch + 1, "a_stop_bce": stop_loss}, ensure_ascii=False), flush=True)
    if best_state is None:
        raise V27ContractError("E3-C did not retain an A-stop adapter state")
    _load_adapter_state(model, best_state)
    return model, tokenizer, {"source": dict(source), "history": history, "best_a_stop_bce": best_loss, "training_checkpoint": str(checkpoint_path.resolve()), "resumable": True}


def _score_pairs(model: Any, tokenizer: Any, pairs: Mapping[str, Sequence[dict[str, Any]]], args: argparse.Namespace) -> list[dict[str, Any]]:
    import torch
    cache = DuoEncodingCache(tokenizer)
    _materialize_contexts(tokenizer, pairs)
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for qid, query_rows in pairs.items():
            eligible = [row for row in query_rows if row["eligible"]]
            values: dict[str, float] = {}
            if eligible:
                logits = _forward_logits(model, cache, eligible, device=args.device, micro_batch=args.score_batch_size)
                for row, logit in zip(eligible, logits.detach().cpu().tolist(), strict=True):
                    values[str(row["candidate_doc_id"])] = float(2.0 / (1.0 + math.exp(-float(logit))) - 1.0)
            for row in query_rows:
                rows.append({
                    "qid": qid, "candidate_doc_id": str(row["candidate_doc_id"]), "incumbent_doc_id": str(row["incumbent_doc_id"]),
                    "eligible": bool(row["eligible"]), "predicted_delta": values.get(str(row["candidate_doc_id"])) if row["eligible"] else None,
                })
    return rows


def _duo_score_path(args: argparse.Namespace, regime: str, partition: str, inner_seed: int, outer_fold: int) -> Path:
    return args.output_dir / "e3c_duo_scores" / regime / f"{partition}_inner_{inner_seed}_fold_{outer_fold + 1}.jsonl"


def _export_scores(
    *, args: argparse.Namespace, model: Any, tokenizer: Any, pairs: Mapping[str, Sequence[dict[str, Any]]], data: e2.E2Data,
    indices: Sequence[int], regime: str, partition: str, inner_seed: int, outer_fold: int, lineage_sha: str,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    rows = _score_pairs(model, tokenizer, pairs, args)
    path = _duo_score_path(args, regime, partition, inner_seed, outer_fold)
    _write_jsonl(path, rows)
    parsed = validate_duo_scores(rows, split=data.split, expected_indices=indices)
    qids = _qids(data.split, indices)
    meta = {
        "kind": "legalir_e3c_duo_scores", "schema_version": 1, "regime": regime, "partition": partition,
        "inner_seed": int(inner_seed), "outer_fold": int(outer_fold), "score_path": str(path.resolve()),
        "score_sha256": sha256_file(path), "score_rows": len(rows), "score_qids": qids,
        "score_qids_sha256": json_sha256({"qids": qids}), "lineage_contract_sha256": lineage_sha,
        "holdout_opened": False, "public_labels_used": False,
    }
    meta_path = path.with_suffix(".meta.json")
    write_json(meta_path, meta)
    meta["metadata_path"], meta["metadata_sha256"] = str(meta_path.resolve()), sha256_file(meta_path)
    return parsed, meta


def _load_e3b_fixed_scores(
    *, args: argparse.Namespace, fold: Mapping[str, Any], split: Any, indices: Sequence[int], regime: str,
    inner_seed: int, outer_fold: int, partition: str,
) -> list[dict[str, dict[str, dict[str, Any]]]]:
    artifacts = fold.get("score_artifacts")
    if not isinstance(artifacts, list):
        raise V27ContractError("E3-B fold misses score artifacts needed for E3-C paired comparison")
    qids = _qids(split, indices)
    c35 = {str(split.qids[int(index)]): [str(value) for value in split.candidates[int(index)][:CANDIDATE_LIMIT]] for index in indices}
    result: list[dict[str, dict[str, dict[str, Any]]]] = []
    for neural_seed in (20260908, 20260909):
        wanted = next((row for row in artifacts if isinstance(row, Mapping) and row.get("partition") == partition and int(row.get("neural_seed", -1)) == neural_seed), None)
        expected = _artifact_path(args.e3b_output_dir, regime, partition, inner_seed, outer_fold, neural_seed)
        if wanted is None or not expected.is_file() or sha256_file(expected) != wanted.get("score_sha256"):
            raise V27ContractError(f"E3-C needs verified E3-B {partition} full-c35 score: {expected}")
        result.append(validate_full_c35_scores(_read_jsonl(expected), c35_by_qid=c35, expected_qids=qids))
    return result


def _pairwise_metrics(
    split: Any,
    before: Mapping[str, Mapping[str, Sequence[str]]],
    after: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    device: str,
) -> dict[str, Any]:
    metrics = transition_metrics_device(split, after, device=device, baseline_predictions=before)
    return {
        "before_recall": metrics["anchor_recall"], "after_recall": metrics["recall"],
        "gain": metrics["gain"], "changed_queries": metrics["changed_queries"],
        "improved_queries": metrics["improved_queries"], "harmed_queries": metrics["harmed_queries"],
        "net_query_equivalents": metrics["net_query_equivalents"], "queries": metrics["queries"],
    }


def _fold_key(regime: str, inner_seed: int, outer_fold: int) -> str:
    return f"{regime}|inner={inner_seed}|fold={outer_fold}"


def _checkpoint_contract(preflight: Mapping[str, Any], args: argparse.Namespace, runtime: Mapping[str, Any]) -> dict[str, Any]:
    return {"lineage_contract_sha256": preflight["lineage_contract_sha256"], "mode": args.mode, "regimes": list(_regimes(args)), "runtime": dict(runtime)}


def _load_or_create_checkpoint(args: argparse.Namespace, preflight: Mapping[str, Any], runtime: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _checkpoint_path(args); contract = _checkpoint_contract(preflight, args, runtime); digest = json_sha256(contract)
    if args.resume and path.is_file():
        state = read_json(path)
        if not isinstance(state, Mapping) or state.get("kind") != E3C_CHECKPOINT_KIND or state.get("contract_sha256") != digest:
            raise V27ContractError("E3-C progress checkpoint has incompatible data/runtime contract; use --no-resume")
        completed = state.get("completed_folds")
        if not isinstance(completed, dict):
            raise V27ContractError("E3-C progress checkpoint has no completed fold map")
        return path, dict(state)
    return path, {"kind": E3C_CHECKPOINT_KIND, "schema_version": 1, "contract": contract, "contract_sha256": digest, "completed_folds": {}, "complete": False}


def _runtime_from_contract(args: argparse.Namespace, data: e2.E2Data, sources: Mapping[str, Any]) -> tuple[dict[str, Any], str | None]:
    if args.runtime_contract:
        payload = read_json(args.runtime_contract)
        runtime = payload.get("runtime") if isinstance(payload, Mapping) else None
        if not isinstance(runtime, Mapping) or payload.get("runtime_sha256") != json_sha256(runtime):
            raise V27ContractError("Invalid E3-C runtime contract")
        return dict(runtime), str(args.runtime_contract.resolve())
    if not args.auto_probe:
        raise V27ContractError("E3-C requires --auto-probe or a matching --runtime-contract before GPU fits")
    # Actual BF16 LoRA backward/optimizer probe on a longest A-fit duo; this
    # deliberately does not reserve VRAM or change effective accumulation.
    import torch
    source = next(iter(sources.values()))
    first_index = list(range(min(len(data.split.qids), 80)))
    all_pairs = _eligible_pairs(data, first_index, with_targets=True)
    candidates = [row for rows in all_pairs.values() for row in rows if row["eligible"]]
    if not candidates:
        raise V27ContractError("E3-C auto-probe cannot find an eligible duo pair")
    candidates.sort(key=lambda row: len("\n".join(row_blocks(row["candidate_row"]))) + len("\n".join(row_blocks(row["incumbent_row"]))), reverse=True)
    successes: list[tuple[int, int, int]] = []
    for micro in sorted(set(int(value) for value in args.micro_batch_candidates)):
        model = tokenizer = optimizer = None
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            model, tokenizer = _load_lora_model(args, source)
            cache = DuoEncodingCache(tokenizer, max_entries=64)
            probe_rows = [copy.deepcopy(candidates[index % len(candidates)]) for index in range(micro)]
            _materialize_contexts(tokenizer, {"probe": probe_rows})
            for row in probe_rows: row["query_weight"] = 1.0 / float(len(probe_rows))
            optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
            loss = _weighted_loss(model, cache, probe_rows, argparse.Namespace(**{**vars(args), "micro_batch": micro}))
            loss.backward(); optimizer.step()
            free, total = torch.cuda.mem_get_info()
            cap = None if args.vram_cap_gib is None else int(args.vram_cap_gib * 1024**3)
            if cap is None or torch.cuda.max_memory_allocated() <= cap:
                max_score = 0
                for score_batch in sorted(set(int(value) for value in args.score_batch_candidates), reverse=True):
                    try:
                        score_rows = [copy.deepcopy(candidates[index % len(candidates)]) for index in range(score_batch)]
                        _materialize_contexts(tokenizer, {"probe-score": score_rows})
                        with torch.inference_mode():
                            _forward_logits(model, cache, score_rows, device=args.device, micro_batch=score_batch)
                        max_score = score_batch
                        break
                    except torch.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        continue
                if max_score:
                    successes.append((micro, max_score, int(torch.cuda.max_memory_allocated())))
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
        finally:
            del model, tokenizer, optimizer
            gc.collect(); torch.cuda.empty_cache()
    if not successes:
        raise RuntimeError("E3-C exact BF16 LoRA training probe OOM at micro-batch 1; use a larger/free GPU")
    micro, score, peak = max(successes, key=lambda item: (item[0], item[1]))
    runtime = {"micro_batch": micro, "score_batch_size": int(score), "gradient_accumulation": int(args.gradient_accumulation), "gradient_checkpointing": bool(args.gradient_checkpointing), "max_length": MAX_LENGTH, "precision": "bf16", "optimizer": "AdamW", "probe_peak_allocated_bytes": peak}
    path = args.output_dir / f"e3c_{args.mode}_runtime_autoprobe.json"
    payload = {"kind": "legalir_e3c_duo_runtime_contract", "runtime": runtime, "runtime_sha256": json_sha256(runtime), "holdout_opened": False, "public_labels_used": False}
    write_json(path, payload)
    return runtime, str(path.resolve())


def _apply_runtime(args: argparse.Namespace, runtime: Mapping[str, Any]) -> None:
    required = {"micro_batch", "score_batch_size", "gradient_accumulation", "gradient_checkpointing", "max_length", "precision", "optimizer"}
    if not required.issubset(runtime) or runtime["precision"] != "bf16" or int(runtime["max_length"]) != MAX_LENGTH or runtime["optimizer"] != "AdamW":
        raise V27ContractError("E3-C runtime contract is incompatible")
    if int(runtime["gradient_accumulation"]) != int(args.gradient_accumulation) or bool(runtime["gradient_checkpointing"]) != bool(args.gradient_checkpointing):
        raise V27ContractError("E3-C runtime contract changes locked optimization semantics")
    args.micro_batch, args.score_batch_size = int(runtime["micro_batch"]), int(runtime["score_batch_size"])


def _run_inner(
    *, args: argparse.Namespace, data: e2.E2Data, groups: np.ndarray, regime: str, inner_seed: int,
    e3b_report: Mapping[str, Any], sources: Mapping[str, Mapping[str, Any]], lineage_sha: str, checkpoint_path: Path, state: dict[str, Any],
) -> dict[str, Any]:
    oof: dict[str, dict[str, list[str]]] = {}; folds_report: list[dict[str, Any]] = []
    e3b_folds = _e3b_folds(e3b_report, regime, inner_seed)
    for outer_fold, (outer_train, validation) in enumerate(group_folds(groups)):
        validation_qids = _qids(data.split, validation)
        key = _fold_key(regime, inner_seed, outer_fold)
        saved = state["completed_folds"].get(key)
        if isinstance(saved, Mapping):
            predictions, report = saved.get("predictions"), saved.get("fold_report")
            if not isinstance(predictions, Mapping) or set(predictions) != set(validation_qids) or not isinstance(report, Mapping):
                raise V27ContractError("E3-C saved fold does not match validation partition")
            oof.update(predictions); folds_report.append(dict(report)); continue
        partitions, partition_metadata = e3b.make_partitions(split=data.split, groups=groups, outer_train=outer_train, validation=validation, inner_seed=inner_seed, outer_fold=outer_fold)
        source = sources[key]
        print(json.dumps({"event": "e3c_fold_start", "run": key, "a_fit_queries": len(partitions["a_fit"]), "a_stop_queries": len(partitions["a_stop"]), "calibration_queries": len(partitions["calibration"]), "validation_queries": len(partitions["validation"])}, ensure_ascii=False), flush=True)
        pairs_fit = _eligible_pairs(data, partitions["a_fit"].tolist(), with_targets=True)
        pairs_stop = _eligible_pairs(data, partitions["a_stop"].tolist(), with_targets=True)
        pairs_c = _eligible_pairs(data, partitions["calibration"].tolist(), with_targets=False)
        pairs_v = _eligible_pairs(data, partitions["validation"].tolist(), with_targets=False)
        model, tokenizer, training = _train_comparator(args=args, pairs_fit=pairs_fit, pairs_stop=pairs_stop, source=source, regime=regime, inner_seed=inner_seed, outer_fold=outer_fold)
        try:
            c_scores, c_artifact = _export_scores(args=args, model=model, tokenizer=tokenizer, pairs=pairs_c, data=data, indices=partitions["calibration"].tolist(), regime=regime, partition="calibration", inner_seed=inner_seed, outer_fold=outer_fold, lineage_sha=lineage_sha)
            v_scores, v_artifact = _export_scores(args=args, model=model, tokenizer=tokenizer, pairs=pairs_v, data=data, indices=partitions["validation"].tolist(), regime=regime, partition="validation", inner_seed=inner_seed, outer_fold=outer_fold, lineage_sha=lineage_sha)
        finally:
            del model, tokenizer; gc.collect()
            if args.device == "cuda":
                import torch; torch.cuda.empty_cache()
        c_split, v_split = _subset(data.split, partitions["calibration"]), _subset(data.split, partitions["validation"])
        policy, calibration, trace = select_calibration_policy(
            c_split, c_scores,
            metrics_fn=lambda split, prediction: transition_metrics_device(split, prediction, device=args.device),
        )
        predictions, decisions = apply_policy(v_split, list(range(len(v_split.qids))), v_scores, policy)
        e3b_scores = _load_e3b_fixed_scores(args=args, fold=e3b_folds[outer_fold], split=data.split, indices=partitions["validation"].tolist(), regime=regime, inner_seed=inner_seed, outer_fold=outer_fold, partition="validation")
        e3b_prediction, _ = apply_e3b_policy(v_split, list(range(len(v_split.qids))), e3b_scores, {}, E3BPolicy(memory_weight=0.0, threshold=0.5))
        report = {"outer_fold": outer_fold, "partitions": partition_metadata, "source_e3b": source, "training": training, "score_artifacts": [c_artifact, v_artifact], "selected_policy": {"kind": "v16b_anchor_abstain"} if policy is None else policy.as_dict(), "calibration": calibration, "policy_trace": trace, "validation_vs_v16b": compact_metrics(transition_metrics_device(v_split, predictions, device=args.device)), "validation_vs_e3b_fixed": _pairwise_metrics(v_split, e3b_prediction, predictions, device=args.device), "decisions": decisions}
        oof.update(predictions); folds_report.append(report)
        state["completed_folds"][key] = {"validation_qids": validation_qids, "predictions": predictions, "fold_report": report}
        write_json(checkpoint_path, state)
        print(json.dumps({"event": "e3c_fold_complete", "run": key, "selected_policy": report["selected_policy"], "validation_vs_v16b": report["validation_vs_v16b"], "validation_vs_e3b_fixed": report["validation_vs_e3b_fixed"]}, ensure_ascii=False), flush=True)
    if set(oof) != set(map(str, data.split.qids)):
        raise V27ContractError("E3-C OOF does not exactly cover complement")
    ordered = {str(qid): oof[str(qid)] for qid in data.split.qids}
    # Recompute E3-B OOF comparison from fold predictions to avoid combining
    # labels across any C/V partition during selection.
    e3b_oof: dict[str, dict[str, list[str]]] = {}
    for fold_report in folds_report:
        fold = int(fold_report["outer_fold"]); validation = group_folds(groups)[fold][1]
        scores = _load_e3b_fixed_scores(args=args, fold=e3b_folds[fold], split=data.split, indices=validation.tolist(), regime=regime, inner_seed=inner_seed, outer_fold=fold, partition="validation")
        pred, _ = apply_e3b_policy(_subset(data.split, validation), list(range(len(validation))), scores, {}, E3BPolicy(memory_weight=0.0, threshold=0.5))
        e3b_oof.update(pred)
    return {"inner_seed": int(inner_seed), "folds": folds_report, "metrics_vs_v16b": compact_metrics(transition_metrics_device(data.split, ordered, device=args.device)), "metrics_vs_e3b_fixed": _pairwise_metrics(data.split, e3b_oof, ordered, device=args.device), "predictions": ordered}


def _gate(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    passes = [float(run["metrics_vs_v16b"]["gain"]) > EPSILON and int(run["metrics_vs_v16b"]["successful_rescues"]) > int(run["metrics_vs_v16b"]["destructive_changes"]) for run in runs]
    return {"per_inner_seed_pass": passes, "pass": bool(all(passes)), "minimum_gain_vs_v16b": min(float(run["metrics_vs_v16b"]["gain"]) for run in runs), "minimum_rescue_minus_destructive": min(int(run["metrics_vs_v16b"]["successful_rescues"]) - int(run["metrics_vs_v16b"]["destructive_changes"]) for run in runs), "minimum_gain_vs_e3b_fixed": min(float(run["metrics_vs_e3b_fixed"]["gain"]) for run in runs)}


def _validate_standard_gate(path: Path, lineage_sha: str) -> None:
    report = read_json(path)
    if not isinstance(report, Mapping) or report.get("kind") != E3C_KIND or report.get("mode") != "standard" or report.get("lineage_contract_sha256") != lineage_sha or report.get("promotion_to_stress_allowed") is not True:
        raise V27ContractError("E3-C stress requires a matching passed standard report")


def run(args: argparse.Namespace) -> dict[str, Any]:
    _configure_cpu_threads(args)
    preflight, sources = _preflight(args)
    manifest_path = args.output_dir / f"e3c_{args.mode}_label_blind_manifest.json"
    write_json(manifest_path, preflight)
    if args.mode == "preflight":
        return preflight
    if args.device != "cuda":
        raise ValueError("E3-C LoRA comparator fits require CUDA; CPU supports preflight/tests only")
    if args.mode == "stress":
        if args.standard_report is None: raise ValueError("MODE=stress requires --standard-report")
        _validate_standard_gate(args.standard_report, preflight["lineage_contract_sha256"])
    data = e2.load_e2_data(args)  # This is deliberately after label-blind manifest write.
    runtime, runtime_report = _runtime_from_contract(args, data, sources)
    _apply_runtime(args, runtime)
    checkpoint_path, state = _load_or_create_checkpoint(args, preflight, runtime)
    standard, stress = read_json(args.e3b_standard_report), read_json(args.e3b_stress_report)
    reports: dict[str, list[dict[str, Any]]] = {}
    for regime in _regimes(args):
        report = standard if regime == STANDARD_REGIME else stress
        reports[regime] = [_run_inner(args=args, data=data, groups=data.groups_by_regime[regime], regime=regime, inner_seed=seed, e3b_report=report, sources=sources, lineage_sha=preflight["lineage_contract_sha256"], checkpoint_path=checkpoint_path, state=state) for seed in args.inner_seeds]
    state["complete"] = True; write_json(checkpoint_path, state)
    gates = {regime: _gate(rows) for regime, rows in reports.items()}
    return {"kind": E3C_KIND, "schema_version": E3C_SCHEMA_VERSION, "status": "nested_oof_research_only_not_promoted_to_holdout", "mode": args.mode, "split": "complement", "lineage_contract": preflight["lineage_contract"], "lineage_contract_sha256": preflight["lineage_contract_sha256"], "label_blind_manifest": str(manifest_path.resolve()), "labels_opened_after_manifest": True, "holdout_opened": False, "public_labels_used": False, "e4_is_not_modified_or_authorized_by_e3c": True, "runtime": runtime, "runtime_report": runtime_report, "checkpoint": {"path": str(checkpoint_path.resolve()), "completed_folds": len(state["completed_folds"]), "resumable": True}, "candidate_limit": CANDIDATE_LIMIT, "frozen_top_k": EDITABLE_SLOT - 1, "regimes": reports, "gates": gates, "promotion_to_stress_allowed": bool(args.mode == "standard" and gates[STANDARD_REGIME]["pass"]), "promotion_to_future_research_allowed": bool(args.mode == "stress" and all(row["pass"] for row in gates.values())), "promotion_to_e4_allowed": False}


def main() -> int:
    args = parse_args(); output = _output_path(args)
    if output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite E3-C report: {output}")
    report = run(args); write_json(output, report)
    print(json.dumps({"output": str(output.resolve()), "stage": report.get("stage", report.get("status")), "holdout_opened": False, "promotion_to_stress_allowed": report.get("promotion_to_stress_allowed", False), "promotion_to_e4_allowed": False}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
