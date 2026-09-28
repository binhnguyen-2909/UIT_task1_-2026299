"""Benchmark CrossEncoder inference microbatches before expensive training."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from v27_model import encode_pairs, load_model
from v27_common import load_metadata, sha256_file, write_json


def load_records(path: Path, questions: dict[str, dict], limit: int) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                qid = str(row["qid"])
                records.append({"qid": qid, "doc_id": str(row["doc_id"]), "query": str(questions[qid]["query"]), "pack": str(row["pack"])})
                if limit and len(records) >= limit:
                    break
    if len(records) < 32:
        raise ValueError("benchmark needs at least 32 E3B records")
    return records


def run_one(model, tokenizer, records: list[dict], device: torch.device, batch_size: int, batches: int, warmup: int) -> dict:
    model.eval()
    index = 0
    def forward(batch: list[dict]) -> int:
        encoded = encode_pairs(tokenizer, [r["query"] for r in batch], [r["pack"] for r in batch], device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            model(**encoded, return_dict=True)
        return int(encoded["attention_mask"].sum().item())
    for _ in range(warmup):
        batch = [records[(index + offset) % len(records)] for offset in range(batch_size)]
        forward(batch)
        index = (index + batch_size) % len(records)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    total_tokens = 0
    for _ in range(batches):
        batch = [records[(index + offset) % len(records)] for offset in range(batch_size)]
        total_tokens += forward(batch)
        index = (index + batch_size) % len(records)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
    peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    examples = batches * batch_size
    return {
        "batch_size": batch_size,
        "batches": batches,
        "examples": examples,
        "elapsed_seconds": elapsed,
        "examples_per_second": examples / max(elapsed, 1e-9),
        "tokens_per_second": total_tokens / max(elapsed, 1e-9),
        "peak_vram_reserved_bytes": int(peak_reserved),
        "peak_vram_allocated_bytes": int(peak_allocated),
        "oom": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e3b-manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--record-limit", type=int, default=1024)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    args = parser.parse_args()
    device = torch.device(args.device)
    questions = load_metadata(args.metadata)
    records = load_records(args.e3b_manifest, questions, args.record_limit)
    tokenizer, model = load_model(args.model, device)
    results = []
    for batch_size in args.batch_sizes:
        try:
            result = run_one(model, tokenizer, records, device, int(batch_size), args.batches, warmup=10)
        except torch.cuda.OutOfMemoryError:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            result = {"batch_size": int(batch_size), "oom": True}
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    viable = [row for row in results if not row.get("oom")]
    if not viable:
        raise RuntimeError("all benchmark microbatches OOM")
    winner = max(viable, key=lambda row: row["examples_per_second"])
    report = {
        "kind": "legalir_v27_inference_benchmark",
        "schema_version": 1,
        "holdout_opened": False,
        "public_labels_used": False,
        "device": str(device),
        "model": str(args.model.resolve()),
        "e3b_manifest": str(args.e3b_manifest.resolve()),
        "e3b_manifest_sha256": sha256_file(args.e3b_manifest),
        "record_count": len(records),
        "results": results,
        "winner": winner,
        "selection": "maximum examples_per_second among non-OOM batches",
        "training_target_effective_batch_queries": 32,
        "training_target_fp16": True,
    }
    write_json(args.output, report)
    print(json.dumps({"winner": winner, "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
