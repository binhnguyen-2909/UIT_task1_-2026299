"""Measure real CrossEncoder pair lengths before a costly max-length ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np

from batch_stage2_rerank import hydrate, load_passage_store, load_stage1, open_readonly


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-audit", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-limit", type=int, default=35)
    parser.add_argument("--passages-per-doc", type=int, default=3)
    parser.add_argument("--max-queries", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def stable_query_indices(queries: list[dict], limit: int) -> list[int]:
    indices = list(range(len(queries)))
    indices.sort(
        key=lambda index: hashlib.sha256(
            str(queries[index]["query_id"]).encode("utf-8")
        ).digest()
    )
    if limit > 0:
        indices = indices[:limit]
    return sorted(indices)


def main() -> int:
    args = parse_args()
    if min(args.candidate_limit, args.passages_per_doc, args.batch_size) <= 0:
        raise ValueError("candidate/passages/batch limits must be positive")
    started = time.perf_counter()
    queries, candidate_rows = load_stage1(args.stage1_audit)
    indices = stable_query_indices(queries, args.max_queries)
    selected_queries = [queries[index] for index in indices]
    selected_candidates = [candidate_rows[index] for index in indices]

    doc_ids = [
        doc_id
        for row in selected_candidates
        for doc_id in list(dict.fromkeys(row))[: args.candidate_limit]
    ]
    connection = open_readonly(args.index_dir / "legalir.sqlite3")
    passage_store = load_passage_store(connection, doc_ids)
    connection.close()

    pair_queries: list[str] = []
    pair_passages: list[str] = []
    for query_item, row in zip(selected_queries, selected_candidates, strict=True):
        query = str(query_item["query"])
        docs = list(dict.fromkeys(str(value) for value in row))[: args.candidate_limit]
        for item in hydrate(query, docs, args.passages_per_doc, passage_store):
            for passage in item["passages"]:
                pair_queries.append(query)
                pair_passages.append(str(passage))

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer.resolve()), local_files_only=True, use_fast=True
    )
    lengths: list[int] = []
    for start in range(0, len(pair_queries), args.batch_size):
        encoded = tokenizer(
            pair_queries[start : start + args.batch_size],
            pair_passages[start : start + args.batch_size],
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_length=True,
        )
        batch_lengths = encoded.get("length")
        if batch_lengths is None:
            batch_lengths = [len(value) for value in encoded["input_ids"]]
        lengths.extend(int(value) for value in batch_lengths)

    values = np.asarray(lengths, dtype=np.int32)
    thresholds = (256, 384, 512, 768, 1024)
    report = {
        "kind": "legalir_reranker_context_length_diagnostic",
        "stage1_audit": str(args.stage1_audit.resolve()),
        "index": str(args.index_dir.resolve()),
        "tokenizer": str(args.tokenizer.resolve()),
        "queries": len(selected_queries),
        "candidate_limit": args.candidate_limit,
        "passages_per_doc": args.passages_per_doc,
        "pairs": len(lengths),
        "length": {
            "min": int(values.min()) if len(values) else 0,
            "mean": statistics.fmean(lengths) if lengths else 0.0,
            "p50": float(np.percentile(values, 50)) if len(values) else 0.0,
            "p90": float(np.percentile(values, 90)) if len(values) else 0.0,
            "p95": float(np.percentile(values, 95)) if len(values) else 0.0,
            "p99": float(np.percentile(values, 99)) if len(values) else 0.0,
            "max": int(values.max()) if len(values) else 0,
        },
        "thresholds": {
            str(threshold): {
                "pairs_over": int(np.count_nonzero(values > threshold)),
                "fraction_over": float(np.mean(values > threshold)) if len(values) else 0.0,
            }
            for threshold in thresholds
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
