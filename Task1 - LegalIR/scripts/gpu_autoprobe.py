"""Shared-GPU auto-probe for the locked E2/E3-B training and scoring paths.

Every candidate batch is measured in a fresh subprocess.  The probe is not a
training shortcut: it executes two real AdamW steps (the first materialises
optimizer state) using the same AMP, GradScaler, gradient accumulation and MIL
loss as E2.  A failed probe never changes precision, optimizer, evidence count
or token limit.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from v27_a_common import V27ContractError, json_sha256, row_blocks, write_json


KIB = 1024
GIB = KIB * KIB * KIB
DEFAULT_MICRO_BATCHES = (1, 2, 4, 8, 16)
DEFAULT_SCORE_BATCHES = (16, 32, 64, 128, 256)


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(item) for item in value]
    return value


def _namespace_from_payload(payload: Mapping[str, Any]) -> argparse.Namespace:
    values = dict(payload)
    for key in (
        "stage1", "stage2", "anchor", "reference", "e1_manifest",
        "e1_manifest_meta", "e1_frozen_scores", "reranker_model", "output_dir",
    ):
        if key in values:
            values[key] = Path(values[key])
    return argparse.Namespace(**values)


def _gpu_identity() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("E3-B auto-probe requires CUDA")
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    uuid = None
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        values = [line.strip() for line in output.splitlines() if line.strip()]
        uuid = values[index] if index < len(values) else None
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "index": int(index),
        "name": str(properties.name),
        "uuid": uuid,
        "total_memory_bytes": int(properties.total_memory),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
    }


def _heavy_qids(tokenizer: Any, examples: Mapping[str, Sequence[Mapping[str, Any]]], *, count: int) -> list[str]:
    """Rank actual training rows by evidence-passage count and encoded tokens."""

    scored: list[tuple[int, int, str]] = []
    for qid, rows in examples.items():
        pairs = [(str(row["query"]), passage) for row in rows for passage in row_blocks(row)]
        if not pairs:
            continue
        encoded = tokenizer(
            [query for query, _ in pairs],
            [passage for _, passage in pairs],
            truncation=True,
            max_length=768,
            padding=False,
        )
        token_count = sum(len(ids) for ids in encoded["input_ids"])
        scored.append((len(pairs), int(token_count), str(qid)))
    if not scored:
        raise V27ContractError("E3-B auto-probe found no trainable E2 query")
    scored.sort(key=lambda row: (-row[0], -row[1], row[2]))
    return [qid for _, _, qid in scored[: max(1, int(count))]]


def _trial_worker(spec: Mapping[str, Any], *, phase: str, batch_size: int) -> dict[str, Any]:
    """Run one isolated candidate. This function is invoked only in a child."""

    import torch
    import e2_mil_finetune_oof as e2
    from v27_a_common import PairEncodingCache, load_relevance_model

    args = _namespace_from_payload(spec["e2_args"])
    if args.device != "cuda":
        raise ValueError("E3-B auto-probe requires the CUDA E2 runtime")
    data = e2.load_e2_data(args)
    identity = _gpu_identity()
    reserve_min = int(spec.get("reserve_min_bytes", 2 * GIB))
    reserve_fraction = float(spec.get("reserve_fraction", 0.15))
    cap = spec.get("vram_cap_bytes")
    cap_bytes = None if cap is None else int(cap)
    free_before, total_before = torch.cuda.mem_get_info()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model = tokenizer = optimizer = scaler = None
    try:
        model, tokenizer = load_relevance_model(
            args.reranker_model,
            args.device,
            train=True,
            fp16=True,
            gradient_checkpointing=bool(args.gradient_checkpointing),
        )
        tokenizer._e2_pair_cache = PairEncodingCache(tokenizer, 768, max_entries=20_000)
        heavy_qids = _heavy_qids(tokenizer, data.examples, count=int(spec.get("heavy_query_count", 3)))
        torch.cuda.synchronize()
        if phase == "train":
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            scaler = torch.cuda.amp.GradScaler(enabled=True)
            # Two full optimizer updates: the first creates AdamW state and the
            # second is the steady-state memory/time observation.
            for update in range(2):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                for offset in range(int(args.gradient_accumulation)):
                    qid = heavy_qids[(update * int(args.gradient_accumulation) + offset) % len(heavy_qids)]
                    total, _ = e2.e2_loss(
                        e2._differentiable_scores(
                            model, tokenizer, data.examples[qid], device=args.device, micro_batch=int(batch_size),
                        ),
                        data.examples[qid],
                    )
                    scaler.scale(total / int(args.gradient_accumulation)).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
        elif phase == "score":
            index_by_qid = {str(qid): index for index, qid in enumerate(data.split.qids)}
            indices = [index_by_qid[qid] for qid in heavy_qids]
            e2.score_full_c35(
                data.split, indices, data.manifest, model, tokenizer,
                device=args.device, batch_size=int(batch_size),
            )
        else:
            raise ValueError(f"Unknown E3-B auto-probe phase: {phase}")
        torch.cuda.synchronize()
        free_after, total_after = torch.cuda.mem_get_info()
        reserve_bytes = max(reserve_min, int(reserve_fraction * int(free_before)))
        capacity = int(free_before) if cap_bytes is None else min(int(free_before), cap_bytes)
        peak_reserved = int(torch.cuda.max_memory_reserved())
        fits = bool(peak_reserved + reserve_bytes <= capacity and int(free_after) >= reserve_bytes)
        return {
            "ok": True,
            "phase": phase,
            "batch_size": int(batch_size),
            "heavy_qids": heavy_qids,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "free_bytes_before": int(free_before),
            "free_bytes_after": int(free_after),
            "device_total_bytes": int(total_after or total_before),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": peak_reserved,
            "reserve_bytes": reserve_bytes,
            "vram_cap_bytes": cap_bytes,
            "fits_budget": fits,
            "gpu": identity,
        }
    except torch.OutOfMemoryError:
        return {
            "ok": False,
            "phase": phase,
            "batch_size": int(batch_size),
            "reason": "cuda_oom",
            "gpu": identity,
        }
    finally:
        del model, tokenizer, optimizer, scaler
        gc.collect()
        torch.cuda.empty_cache()


def _read_worker_result(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    if completed.returncode != 0:
        return {"ok": False, "reason": "worker_failed", "stderr": completed.stderr[-4_000:]}
    for line in reversed(completed.stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "ok" in payload:
            return payload
    return {"ok": False, "reason": "worker_emitted_no_json", "stdout": completed.stdout[-4_000:]}


def _choose_fastest_fit(trials: Sequence[Mapping[str, Any]], *, phase: str) -> dict[str, Any]:
    viable = [trial for trial in trials if trial.get("ok") and trial.get("fits_budget")]
    if not viable:
        raise RuntimeError(f"E3-B auto-probe found no {phase} batch that fits the measured shared-GPU budget")
    return min(viable, key=lambda row: (float(row["elapsed_seconds"]), -int(row["batch_size"])))


def run_auto_probe(
    *,
    e2_args: argparse.Namespace,
    output_path: Path,
    micro_batches: Sequence[int] = DEFAULT_MICRO_BATCHES,
    score_batches: Sequence[int] = DEFAULT_SCORE_BATCHES,
    heavy_query_count: int = 3,
    reserve_fraction: float = 0.15,
    reserve_min_bytes: int = 2 * GIB,
    vram_cap_bytes: int | None = None,
) -> dict[str, Any]:
    """Measure candidate batches in clean workers and write a frozen contract."""

    if getattr(e2_args, "device", None) != "cuda":
        raise ValueError("E3-B auto-probe requires --device cuda")
    arg_payload = _as_jsonable(vars(e2_args))
    spec = {
        "kind": "legalir_e3b_gpu_autoprobe_spec",
        "e2_args": arg_payload,
        "heavy_query_count": int(heavy_query_count),
        "reserve_fraction": float(reserve_fraction),
        "reserve_min_bytes": int(reserve_min_bytes),
        "vram_cap_bytes": vram_cap_bytes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="e3b_autoprobe_", dir=str(output_path.parent)) as directory:
        spec_path = Path(directory) / "spec.json"
        write_json(spec_path, spec)
        def invoke(phase: str, batch: int) -> dict[str, Any]:
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker-spec", str(spec_path), "--phase", phase, "--batch-size", str(batch)],
                text=True,
                capture_output=True,
                check=False,
            )
            result = _read_worker_result(completed)
            result.setdefault("phase", phase)
            result.setdefault("batch_size", int(batch))
            return result
        train_trials = [invoke("train", int(batch)) for batch in micro_batches]
        score_trials = [invoke("score", int(batch)) for batch in score_batches]
    selected_train = _choose_fastest_fit(train_trials, phase="training")
    selected_score = _choose_fastest_fit(score_trials, phase="scoring")
    runtime = {
        "micro_batch": int(selected_train["batch_size"]),
        "score_batch_size": int(selected_score["batch_size"]),
        "gradient_accumulation": int(e2_args.gradient_accumulation),
        "gradient_checkpointing": bool(e2_args.gradient_checkpointing),
        "max_length": 768,
        "precision": "fp32_master_weights_fp16_autocast",
        "optimizer": "AdamW",
    }
    report = {
        "kind": "legalir_e3b_gpu_runtime_contract",
        "schema_version": 1,
        "runtime": runtime,
        "runtime_sha256": json_sha256(runtime),
        "shared_gpu_policy": {
            "reserve_fraction": float(reserve_fraction),
            "reserve_min_bytes": int(reserve_min_bytes),
            "vram_cap_bytes": vram_cap_bytes,
            "subprocess_per_trial": True,
            "training_steps_per_trial": 2,
        },
        "training_trials": train_trials,
        "scoring_trials": score_trials,
        "selected_training": selected_train,
        "selected_scoring": selected_score,
        "holdout_opened": False,
        "public_labels_used": False,
    }
    write_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-spec", type=Path, required=True)
    parser.add_argument("--phase", choices=("train", "score"), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("E3-B auto-probe batch size must be positive")
    payload = json.loads(args.worker_spec.read_text(encoding="utf-8"))
    print(json.dumps(_trial_worker(payload, phase=args.phase, batch_size=args.batch_size), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
