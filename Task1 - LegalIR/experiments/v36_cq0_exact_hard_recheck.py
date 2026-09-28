"""Recheck V36-CQ0 on the V34 HARD benchmark at V34 max_length=1024."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from v36_cq0_screen import evaluate_hard, load_jsonl, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hard-benchmark", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()
    started = time.perf_counter()
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(str(args.model.resolve()), local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    true_id = int(tokenizer.convert_tokens_to_ids("yes"))
    false_id = int(tokenizer.convert_tokens_to_ids("no"))
    hard = load_jsonl(args.hard_benchmark)
    base = AutoModelForCausalLM.from_pretrained(str(args.model.resolve()), local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
    baseline = {
        "all": evaluate_hard(base, tokenizer, hard, device, true_id, false_id, args.max_length, args.batch_size, False),
        "fold0": evaluate_hard(base, tokenizer, hard, device, true_id, false_id, args.max_length, args.batch_size, True),
    }
    finetuned_model = PeftModel.from_pretrained(base, str(args.checkpoint.resolve()), is_trainable=False).to(device).eval()
    finetuned = {
        "all": evaluate_hard(finetuned_model, tokenizer, hard, device, true_id, false_id, args.max_length, args.batch_size, False),
        "fold0": evaluate_hard(finetuned_model, tokenizer, hard, device, true_id, false_id, args.max_length, args.batch_size, True),
    }
    report = {
        "kind": "legalir_v36_cq0_exact_hard_recheck",
        "schema_version": 1,
        "stage": "V36-CQ0_recheck_max_length_1024_no_training",
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "training_performed": False,
        "model": str(args.model.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "v34_reference_gold_win_rate": 0.6793388429752066,
        "baseline_zero_shot": baseline,
        "finetuned": finetuned,
        "delta": {"all_hard_win_gain": finetuned["all"]["hard_win_rate"] - baseline["all"]["hard_win_rate"], "fold0_hard_win_gain": finetuned["fold0"]["hard_win_rate"] - baseline["fold0"]["hard_win_rate"]},
        "gate": {"hard_all_ge_0_72": finetuned["all"]["hard_win_rate"] >= 0.72, "hard_fold0_ge_0_72": finetuned["fold0"]["hard_win_rate"] >= 0.72, "pass_exact_hard": finetuned["all"]["hard_win_rate"] >= 0.72},
        "inputs": {"hard_benchmark": {"path": str(args.hard_benchmark.resolve()), "sha256": sha256_file(args.hard_benchmark)}},
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "v36_cq0_exact_hard_recheck.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
