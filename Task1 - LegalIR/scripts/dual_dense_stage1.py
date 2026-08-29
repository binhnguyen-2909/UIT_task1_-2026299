"""Add complementary dense views to a frozen Stage-1 tail.

The dense rankings are generated without labels.  Optional references are read
only after every ranking is frozen, either to select an RRF configuration on a
development split or to report a fixed configuration on holdout.  A protected
prefix (five documents by default) keeps Stage-1 head behaviour unchanged.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from corpus_prf_stage1 import changed_top5, evaluate, load_inputs, parse_number_list
from evaluate_dense_v10_3 import top_dense_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-audit", type=Path, required=True)
    parser.add_argument("--dense-index", action="append", type=Path, required=True)
    parser.add_argument(
        "--weight-grid",
        action="append",
        required=True,
        help="Comma-separated RRF weights; one grid for each --dense-index",
    )
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--selected-audit", type=Path, required=True)
    parser.add_argument("--rrf-k", default="20,40,60")
    parser.add_argument("--lock-top", type=int, default=5)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--score-batch-size", type=int, default=32)
    parser.add_argument("--chunk-limit", type=int, default=200)
    return parser.parse_args()


def load_dense_ranking(
    index_dir: Path,
    queries: Sequence[dict[str, Any]],
    device: str,
    query_batch_size: int,
    score_batch_size: int,
    chunk_limit: int,
) -> tuple[list[list[str]], dict[str, Any]]:
    from sentence_transformers import SentenceTransformer

    resolved = index_dir.resolve()
    config = json.loads((resolved / "legalir.config.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (resolved / "dense_embeddings.manifest.json").read_text(encoding="utf-8")
    )
    runtime = manifest.get("runtime", {})
    for key, expected in (
        ("max_seq_length", config.get("dense_max_seq_length")),
        ("query_prefix", config.get("dense_query_prefix", "")),
        ("passage_prefix", config.get("dense_passage_prefix", "")),
    ):
        if runtime.get(key) != expected:
            raise ValueError(f"{resolved}: dense cache/config mismatch for {key}")

    chunk_ids = [
        str(value)
        for value in json.loads(
            (resolved / "dense_embeddings.ids.json").read_text(encoding="utf-8")
        )
    ]
    passage_embeddings = np.load(resolved / "dense_embeddings.npy", mmap_mode="r")
    if len(chunk_ids) != len(passage_embeddings):
        raise ValueError(f"{resolved}: dense IDs and embeddings differ")

    model = SentenceTransformer(
        str(config["dense_model_path"]), device=device, local_files_only=True
    )
    model.max_seq_length = int(config["dense_max_seq_length"])
    prefix = str(config.get("dense_query_prefix", ""))
    texts = [str(item["query"]) for item in queries]
    encoded = model.encode(
        [f"{prefix}{text}" if prefix else text for text in texts],
        batch_size=query_batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    rankings = top_dense_documents(
        encoded,
        passage_embeddings,
        chunk_ids,
        chunk_limit,
        score_batch_size,
    )
    metadata = {
        "index": str(resolved),
        "max_seq_length": int(config["dense_max_seq_length"]),
        "query_prefix": prefix,
        "passage_prefix": str(config.get("dense_passage_prefix", "")),
        "chunks": len(chunk_ids),
        "chunk_limit": int(chunk_limit),
        "mean_unique_documents": statistics.fmean(len(row) for row in rankings),
    }
    del model, encoded, passage_embeddings
    return rankings, metadata


def fuse_rankings(
    base: Sequence[str],
    channels: Sequence[Sequence[str]],
    weights: Sequence[float],
    rrf_k: int,
    lock_top: int,
) -> list[str]:
    if len(channels) != len(weights):
        raise ValueError("dense channels and weights differ in length")
    locked = list(dict.fromkeys(str(value) for value in base))[:lock_top]
    locked_ids = set(locked)
    scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, int] = {}
    serial = 0
    for weight, ranking in [(1.0, base), *zip(weights, channels, strict=True)]:
        if weight <= 0:
            continue
        seen: set[str] = set()
        rank = 0
        for value in ranking:
            doc_id = str(value)
            if doc_id in seen or doc_id in locked_ids:
                continue
            seen.add(doc_id)
            rank += 1
            scores[doc_id] += float(weight) / (rrf_k + rank)
            if doc_id not in first_seen:
                first_seen[doc_id] = serial
                serial += 1
    tail = sorted(scores, key=lambda value: (-scores[value], first_seen[value], value))
    return [*locked, *tail]


def build_payload(
    source: dict[str, Any],
    queries: Sequence[dict[str, Any]],
    rankings: Sequence[Sequence[str]],
    source_path: Path,
    configuration: dict[str, Any],
    metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    output_queries: list[dict[str, Any]] = []
    for item, ranking in zip(queries, rankings, strict=True):
        updated = dict(item)
        updated["answers"] = list(ranking[:5])
        updated["results"] = [
            {
                "doc_id": doc_id,
                "text": "",
                "score": 1.0 / rank,
                "method": "dual_dense_safe_tail_rrf",
                "rank": rank,
                "chunk_id": doc_id,
                "rerank_score": None,
                "bm25_rank": None,
                "dense_rank": None,
            }
            for rank, doc_id in enumerate(ranking[:5], 1)
        ]
        output_queries.append(updated)
    payload = {
        key: value
        for key, value in source.items()
        if key not in {"queries", "stage1_candidates", "metrics", "candidate_oracle"}
    }
    payload.update(
        {
            "kind": "legalir_dual_dense_safe_tail_stage1",
            "source_stage1_audit": str(source_path.resolve()),
            "dual_dense_configuration": configuration,
            "queries": output_queries,
            "stage1_candidates": [list(row) for row in rankings],
        }
    )
    if metrics is not None:
        payload["metrics"] = {
            "precision": metrics["precision_at_5"],
            "recall": metrics["recall"]["at_5"],
        }
        payload["candidate_oracle"] = {
            f"recall_at_{cutoff}": metrics["recall"][f"at_{cutoff}"]
            for cutoff in (5, 20, 35, 50, 80, 100, 200)
        }
    return payload


def main() -> int:
    args = parse_args()
    if len(args.dense_index) != len(args.weight_grid):
        raise ValueError("Provide exactly one --weight-grid per --dense-index")
    if min(
        args.query_batch_size,
        args.score_batch_size,
        args.chunk_limit,
    ) <= 0 or args.lock_top < 0:
        raise ValueError("batch/chunk limits must be positive and lock-top non-negative")
    weight_grids = [parse_number_list(value, float) for value in args.weight_grid]
    if any(weight < 0 for grid in weight_grids for weight in grid):
        raise ValueError("dense weights must be non-negative")
    rrf_values = parse_number_list(args.rrf_k, int)
    if any(value < 0 for value in rrf_values):
        raise ValueError("rrf-k must be non-negative")

    started = time.perf_counter()
    source, queries, base_rankings, gold = load_inputs(args.stage1_audit, args.reference)
    channel_rankings: list[list[list[str]]] = []
    channel_metadata: list[dict[str, Any]] = []
    for position, index_dir in enumerate(args.dense_index, 1):
        print(f"Dense channel {position}/{len(args.dense_index)}: {index_dir}", flush=True)
        rankings, metadata = load_dense_ranking(
            index_dir,
            queries,
            args.device,
            args.query_batch_size,
            args.score_batch_size,
            args.chunk_limit,
        )
        channel_rankings.append(rankings)
        channel_metadata.append(metadata)

    from itertools import product

    configurations = [
        (tuple(float(value) for value in weights), int(rrf_k))
        for weights in product(*weight_grids)
        for rrf_k in rrf_values
    ]
    if not gold and len(configurations) != 1:
        raise ValueError("Apply without labels requires singleton weight and rrf-k grids")

    baseline_metrics = evaluate(base_rankings, queries, gold) if gold else None
    channel_metrics = [
        evaluate(rankings, queries, gold) if gold else None
        for rankings in channel_rankings
    ]
    sweep: list[dict[str, Any]] = []
    ranking_cache: dict[tuple[tuple[float, ...], int], list[list[str]]] = {}
    for weights, rrf_k in configurations:
        fused = [
            fuse_rankings(
                base,
                [channel[query_index] for channel in channel_rankings],
                weights,
                rrf_k,
                args.lock_top,
            )
            for query_index, base in enumerate(base_rankings)
        ]
        key = (weights, rrf_k)
        ranking_cache[key] = fused
        result = evaluate(fused, queries, gold) if gold else None
        sweep.append(
            {
                "weights": list(weights),
                "rrf_k": rrf_k,
                "lock_top": args.lock_top,
                "metrics": result,
                "versus_baseline": changed_top5(base_rankings, fused),
            }
        )

    if gold:
        best = max(
            sweep,
            key=lambda row: (
                row["metrics"]["recall"]["at_35"],
                row["metrics"]["recall"]["at_50"],
                row["metrics"]["recall"]["at_80"],
                -sum(row["weights"]),
                -row["rrf_k"],
            ),
        )
    else:
        best = sweep[0]
    selected_key = (tuple(float(value) for value in best["weights"]), int(best["rrf_k"]))
    selected = ranking_cache[selected_key]
    configuration = {
        "channels": channel_metadata,
        "weights": list(selected_key[0]),
        "rrf_k": selected_key[1],
        "lock_top": args.lock_top,
    }
    payload = build_payload(
        source,
        queries,
        selected,
        args.stage1_audit,
        configuration,
        best["metrics"],
    )
    args.selected_audit.parent.mkdir(parents=True, exist_ok=True)
    args.selected_audit.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = {
        "kind": "legalir_dual_dense_safe_tail_sweep",
        "source_stage1_audit": str(args.stage1_audit.resolve()),
        "reference": str(args.reference.resolve()) if args.reference else None,
        "questions": len(queries),
        "channels": channel_metadata,
        "baseline_metrics": baseline_metrics,
        "channel_metrics": channel_metrics,
        "selected": best,
        "sweep": sweep,
        "elapsed_seconds": time.perf_counter() - started,
        "selected_audit": str(args.selected_audit.resolve()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "questions": len(queries),
                "baseline": baseline_metrics,
                "selected": best,
                "elapsed_seconds": report["elapsed_seconds"],
                "report": str(args.report.resolve()),
                "selected_audit": str(args.selected_audit.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
