"""Controlled hard-negative LoRA fine-tune for the locked V13 reranker.

The training split is the clean V13 complement only.  It deliberately does not
read the locked holdout and does not perform a parameter sweep.  A small LoRA
adapter is trained on the existing local CrossEncoder, merged back into the
same model, then saved as a normal CrossEncoder directory so the unchanged
Stage-2 runner can evaluate it.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .batch_stage2_rerank import hydrate, load_passage_store, open_readonly
    from .v11_tiny_ltr import load_audit_split
    from .v11_tiny_ltr_complement import clean_complement
except ImportError:
    from batch_stage2_rerank import hydrate, load_passage_store, open_readonly
    from v11_tiny_ltr import load_audit_split
    from v11_tiny_ltr_complement import clean_complement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complement-stage1", type=Path, required=True)
    parser.add_argument("--complement-stage2", type=Path, required=True)
    parser.add_argument("--complement-reference", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-limit", type=int, default=35)
    parser.add_argument("--passages-per-doc", type=int, default=3)
    parser.add_argument("--max-examples", type=int, default=4000)
    parser.add_argument("--positive-per-query", type=int, default=1)
    parser.add_argument("--negative-per-query", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--wait-for-free-mb",
        type=int,
        default=2600,
        help="On CUDA, wait instead of competing with another process until this much VRAM is free.",
    )
    parser.add_argument("--wait-seconds", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def best_passage(query: str, doc_id: str, store: dict[str, list[dict[str, Any]]], count: int) -> str:
    hydrated = hydrate(query, [doc_id], count, store)
    if not hydrated:
        return ""
    passages = hydrated[0].get("passages") or [hydrated[0].get("text", "")]
    return str(passages[0] if passages else "")


def build_training_pairs(
    split: Any,
    passage_store: dict[str, list[dict[str, Any]]],
    candidate_limit: int,
    passages_per_doc: int,
    max_examples: int,
    positive_per_query: int,
    negative_per_query: int,
) -> tuple[list[tuple[str, str, float]], dict[str, int]]:
    if split.references is None:
        raise ValueError("training split has no references")
    pairs: list[tuple[str, str, float]] = []
    queries_used = 0
    positives = 0
    negatives = 0
    for query_index, query in enumerate(split.questions):
        candidates = list(dict.fromkeys(split.candidates[query_index]))[:candidate_limit]
        references = set(split.references[query_index])
        positive_ids = [doc_id for doc_id in candidates if doc_id in references]
        if not positive_ids:
            continue
        # The current reranker is the hard-negative miner.  Ties are resolved
        # by Stage-1 order and doc_id, so the training set is reproducible.
        neural = split.neural_scores[query_index]
        candidate_order = {doc_id: rank for rank, doc_id in enumerate(candidates)}
        negative_ids = [doc_id for doc_id in candidates if doc_id not in references]
        negative_ids.sort(
            key=lambda doc_id: (
                -float(neural.get(doc_id, float("-inf"))),
                candidate_order[doc_id],
                str(doc_id),
            )
        )
        selected_positive_ids = positive_ids[: max(1, positive_per_query)]
        selected_negative_ids = negative_ids[: max(1, negative_per_query)]
        query_pairs: list[tuple[str, str, float]] = []
        for doc_id in selected_positive_ids:
            text = best_passage(query, doc_id, passage_store, passages_per_doc)
            if text.strip():
                query_pairs.append((query, text, 1.0))
        for doc_id in selected_negative_ids:
            text = best_passage(query, doc_id, passage_store, passages_per_doc)
            if text.strip():
                query_pairs.append((query, text, 0.0))
        if not query_pairs:
            continue
        queries_used += 1
        positives += sum(label > 0.5 for _, _, label in query_pairs)
        negatives += sum(label <= 0.5 for _, _, label in query_pairs)
        pairs.extend(query_pairs)
        if len(pairs) >= max_examples:
            break
    pairs = pairs[:max_examples]
    stats = {
        "queries_used_before_cap": queries_used,
        "examples": len(pairs),
        "positive_examples_before_cap": positives,
        "negative_examples_before_cap": negatives,
        "positive_examples": sum(label > 0.5 for _, _, label in pairs),
        "negative_examples": sum(label <= 0.5 for _, _, label in pairs),
    }
    return pairs, stats


def wait_for_cuda_memory(torch: Any, minimum_free_mb: int, wait_seconds: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    started = time.perf_counter()
    observations: list[dict[str, int]] = []
    while True:
        free, total = torch.cuda.mem_get_info()
        free_mb = int(free // (1024 * 1024))
        total_mb = int(total // (1024 * 1024))
        observations.append({"free_mb": free_mb, "total_mb": total_mb})
        if free_mb >= minimum_free_mb:
            return {
                "waited_seconds": time.perf_counter() - started,
                "minimum_free_mb": minimum_free_mb,
                "observations": observations[-10:],
            }
        print(
            f"waiting for CUDA memory: free={free_mb} MiB, need={minimum_free_mb} MiB",
            flush=True,
        )
        time.sleep(max(5, wait_seconds))


def attach_lora(cross_encoder: Any, args: argparse.Namespace, torch: Any) -> tuple[Any, dict[str, Any]]:
    from peft import LoraConfig, TaskType, get_peft_model

    transformer = cross_encoder[0]
    base_model = transformer.model
    module_names = [name for name, _ in base_model.named_modules()]
    target_modules = [
        suffix for suffix in ("query", "key", "value")
        if any(name == suffix or name.endswith("." + suffix) for name in module_names)
    ]
    if not target_modules:
        raise RuntimeError("Could not locate query/key/value attention projections for LoRA")
    modules_to_save = ["classifier"] if any(
        name == "classifier" or name.endswith(".classifier") for name in module_names
    ) else None
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
        modules_to_save=modules_to_save,
    )
    peft_model = get_peft_model(base_model, config)
    transformer.model = peft_model
    if hasattr(peft_model, "gradient_checkpointing_enable"):
        peft_model.gradient_checkpointing_enable()
    if hasattr(peft_model, "enable_input_require_grads"):
        peft_model.enable_input_require_grads()
    trainable = sum(parameter.numel() for parameter in peft_model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in peft_model.parameters())
    return peft_model, {
        "target_modules": target_modules,
        "modules_to_save": modules_to_save or [],
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
        "trainable_fraction": float(trainable / max(1, total)),
        "gradient_checkpointing": bool(getattr(peft_model, "is_gradient_checkpointing", False)),
    }


def main() -> int:
    args = parse_args()
    if args.candidate_limit <= 0 or args.max_examples <= 0:
        raise ValueError("candidate-limit and max-examples must be positive")
    if args.positive_per_query <= 0 or args.negative_per_query <= 0:
        raise ValueError("positive-per-query and negative-per-query must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    started = time.perf_counter()
    raw_split = load_audit_split(
        "complement",
        args.complement_stage1,
        args.complement_stage2,
        args.complement_reference,
    )
    split, excluded_exact_ids = clean_complement(raw_split, args.complement_stage1)
    selected_ids = {
        doc_id
        for candidates in split.candidates
        for doc_id in candidates[: args.candidate_limit]
    }
    connection = open_readonly(args.index_dir / "legalir.sqlite3")
    passage_store = load_passage_store(connection, sorted(selected_ids))
    connection.close()
    pairs, training_stats = build_training_pairs(
        split,
        passage_store,
        args.candidate_limit,
        args.passages_per_doc,
        args.max_examples,
        args.positive_per_query,
        args.negative_per_query,
    )
    if not pairs:
        raise RuntimeError("No non-empty training pairs were built")
    report: dict[str, Any] = {
        "kind": "legalir_v13_controlled_reranker_lora_finetune",
        "leakage_policy": {
            "training_split": "clean complement only",
            "holdout_read": False,
            "holdout_used_for_selection": False,
            "public_labels_used": False,
            "v57_predictions_used": False,
        },
        "input": {
            "complement_stage1": str(args.complement_stage1.resolve()),
            "complement_stage2": str(args.complement_stage2.resolve()),
            "complement_reference": str(args.complement_reference.resolve()),
            "index_dir": str(args.index_dir.resolve()),
            "base_model": str(args.model.resolve()),
            "excluded_exact_memory_qids": excluded_exact_ids,
            "complement_queries_after_cleaning": len(split.qids),
            "candidate_limit": args.candidate_limit,
            "passages_per_doc": args.passages_per_doc,
        },
        "training": {
            "seed": args.seed,
            "max_examples": args.max_examples,
            "positive_per_query": args.positive_per_query,
            "negative_per_query": args.negative_per_query,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "learning_rate": args.learning_rate,
            "pairs": training_stats,
        },
        "output_dir": str(args.output_dir.resolve()),
        "started": not args.dry_run,
    }
    if args.dry_run:
        report["elapsed_seconds"] = time.perf_counter() - started
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "finetune_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0

    import torch
    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader

    memory = wait_for_cuda_memory(torch, args.wait_for_free_mb, args.wait_seconds) if args.device == "cuda" else None
    model_kwargs = {"torch_dtype": torch.float16} if args.device == "cuda" else {}
    cross_encoder = CrossEncoder(
        str(args.model.resolve()),
        device=args.device,
        local_files_only=True,
        max_length=args.max_length,
        model_kwargs=model_kwargs,
    )
    peft_model, lora_stats = attach_lora(cross_encoder, args, torch)
    examples = [InputExample(texts=[query, passage], label=float(label)) for query, passage, label in pairs]
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps({"training": training_stats, "lora": lora_stats, "cuda_memory": memory}, ensure_ascii=False),
        flush=True,
    )
    warmup_steps = max(1, int(len(loader) * args.epochs * 0.05))
    cross_encoder.old_fit(
        loader,
        epochs=args.epochs,
        optimizer_params={"lr": args.learning_rate},
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        max_grad_norm=1.0,
        use_amp=args.device == "cuda",
        show_progress_bar=True,
    )

    # Merge the adapter so the existing batch Stage-2 runner can load a plain
    # local CrossEncoder without any special adapter code.
    if hasattr(peft_model, "merge_and_unload"):
        merged = peft_model.merge_and_unload()
        cross_encoder[0].model = merged
    cross_encoder.save_pretrained(str(args.output_dir), create_model_card=False)
    reload_check = CrossEncoder(
        str(args.output_dir.resolve()),
        device="cpu",
        local_files_only=True,
        max_length=args.max_length,
    )
    check_scores = reload_check.predict(
        [[pairs[0][0], pairs[0][1]], [pairs[-1][0], pairs[-1][1]]],
        batch_size=1,
        show_progress_bar=False,
    )
    report.update({
        "cuda_memory": memory,
        "lora": lora_stats,
        "merged": bool(hasattr(peft_model, "merge_and_unload")),
        "reload_check_scores": [float(value) for value in check_scores],
        "output_model": str(args.output_dir.resolve()),
        "elapsed_seconds": time.perf_counter() - started,
    })
    (args.output_dir / "finetune_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
