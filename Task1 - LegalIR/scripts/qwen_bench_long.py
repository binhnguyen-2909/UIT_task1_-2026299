#!/usr/bin/env python3
"""30-minute long-read benchmark: split pack/tokenize/forward timing.

Legacy forward stays the cache source of truth until fixed-batch
equivalence (atol 0.002 on delta) passes. Never mixes caches.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ATOL = 2e-3


def bench_one_batch(*, dry_run: bool = False) -> dict:
    if dry_run:
        return {
            "pack_s": 0.0, "tokenize_s": 0.0, "forward_s": 0.0,
            "eval_mode": True, "no_grad": True, "dtype": "bfloat16",
            "use_cache": False, "logits_to_keep": 1, "dry_run": True,
        }
    raise NotImplementedError("GPU benchmark path not implemented in dry_run")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--qwen-model", type=Path, default=Path("models/Qwen3-Reranker-0.6B"))
    parser.add_argument("--output", type=Path, default=Path("Task1 - LegalIR/output/qwen_c200_research/qwen_bench_long.json"))
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_pack0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_model, local_files_only=True, padding_side="left", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    t_pack1 = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(
        args.qwen_model, local_files_only=True, torch_dtype=torch.bfloat16,
    ).to("cuda").eval()
    assert not model.training
    yes = tokenizer("yes", add_special_tokens=False).input_ids[0]
    no = tokenizer("no", add_special_tokens=False).input_ids[0]

    texts = [f"Query benchmark {i} Document " + ("điều khoản pháp luật " * 800) for i in range(min(args.pairs, 8))]
    t_tok0 = time.perf_counter()
    batch = tokenizer(texts, padding=True, return_tensors="pt")
    batch = {k: v.to("cuda") for k, v in batch.items()}
    t_tok1 = time.perf_counter()

    results: dict = {"eval_mode": not model.training, "no_grad": True, "dtype": "bfloat16"}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for bs in (1, 2, 4, 8):
            if bs > len(texts):
                break
            try:
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                t_f0 = time.perf_counter()
                sub = {k: v[:bs] for k, v in batch.items()}
                legacy = model(**sub).logits[:, -1, :].float()
                try:
                    opt = model(**sub, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
                except TypeError:
                    opt = model(**sub, use_cache=False).logits[:, -1, :].float()
                t_f1 = time.perf_counter()
                dd = (legacy[:, yes] - legacy[:, no] - (opt[:, yes] - opt[:, no])).abs().max().item()
                results[f"bs{bs}"] = {
                    "forward_s": t_f1 - t_f0, "peak_bytes": int(torch.cuda.max_memory_allocated()),
                    "max_delta_diff": float(dd), "equivalent": bool(dd <= ATOL),
                }
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                results[f"bs{bs}"] = {"ok": False}
                break
    results.update({
        "pack_s": t_pack1 - t_pack0, "tokenize_s": t_tok1 - t_tok0,
        "use_cache": False, "logits_to_keep": 1,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
