"""Evaluate E5 prefix/sequence-length variants without lexical confounding.

Each index must contain a compatible ``dense_embeddings`` cache and the same
ordered legal chunks.  The script batches all clean evaluation queries on the
GPU, reproduces the dense channel's top-200-chunk retrieval, deduplicates by
document exactly as RRF does, and reports document recall at the roadmap
cutoffs.  It is a screening diagnostic: the winning variant must still pass
the complete Stage-1 and Stage-2 holdout gates before promotion.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", action="append", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--score-batch-size", type=int, default=32)
    parser.add_argument("--chunk-limit", type=int, default=200)
    return parser.parse_args()


def load_queries(path: Path) -> list[tuple[str, str, set[str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a question object: {path}")
    rows: list[tuple[str, str, set[str]]] = []
    for question_id, item in payload.items():
        if not isinstance(item, dict) or not isinstance(item.get("question"), str):
            raise ValueError(f"{question_id}: missing question")
        answers = item.get("answer")
        if not isinstance(answers, list) or not answers:
            raise ValueError(f"{question_id}: missing non-empty answers")
        rows.append((str(question_id), item["question"], {str(value) for value in answers}))
    return rows


def doc_id(chunk_id: str) -> str:
    return chunk_id.split("::chunk::", 1)[0]


def top_dense_documents(
    query_embeddings: np.ndarray,
    passage_embeddings: np.ndarray,
    chunk_ids: list[str],
    chunk_limit: int,
    score_batch_size: int,
) -> list[list[str]]:
    if chunk_limit <= 0 or score_batch_size <= 0:
        raise ValueError("chunk and score batch sizes must be positive")
    limit = min(chunk_limit, len(chunk_ids))
    rankings: list[list[str]] = []
    for start in range(0, len(query_embeddings), score_batch_size):
        query_batch = query_embeddings[start : start + score_batch_size]
        scores = query_batch @ passage_embeddings.T
        if limit == len(chunk_ids):
            indices = np.argsort(-scores, axis=1, kind="stable")
        else:
            indices = np.argpartition(-scores, limit - 1, axis=1)[:, :limit]
            local_scores = np.take_along_axis(scores, indices, axis=1)
            order = np.argsort(-local_scores, axis=1, kind="stable")
            indices = np.take_along_axis(indices, order, axis=1)
        for row in indices:
            seen: set[str] = set()
            documents: list[str] = []
            for index in row:
                current = doc_id(chunk_ids[int(index)])
                if current in seen:
                    continue
                seen.add(current)
                documents.append(current)
            rankings.append(documents)
    return rankings


def metrics(rankings: list[list[str]], rows: list[tuple[str, str, set[str]]]) -> dict[str, Any]:
    cutoffs = (1, 5, 20, 35, 50, 80, 100, 200)
    recall: dict[int, list[float]] = {cutoff: [] for cutoff in cutoffs}
    reciprocal_ranks: list[float] = []
    for ranking, (_, _, gold) in zip(rankings, rows, strict=True):
        for cutoff in cutoffs:
            recall[cutoff].append(len(gold & set(ranking[:cutoff])) / len(gold))
        first = next((position for position, value in enumerate(ranking, 1) if value in gold), None)
        reciprocal_ranks.append(0.0 if first is None else 1.0 / first)
    return {
        "recall": {f"at_{cutoff}": statistics.fmean(recall[cutoff]) for cutoff in cutoffs},
        "mrr": statistics.fmean(reciprocal_ranks),
        "mean_unique_documents": statistics.fmean(len(ranking) for ranking in rankings),
    }


def main() -> int:
    args = parse_args()
    if args.query_batch_size <= 0:
        raise ValueError("query-batch-size must be positive")
    from sentence_transformers import SentenceTransformer

    rows = load_queries(args.queries)
    reports: list[dict[str, Any]] = []
    baseline_rankings: list[list[str]] | None = None
    for index_dir in args.index:
        started = time.perf_counter()
        index_dir = index_dir.resolve()
        config = json.loads((index_dir / "legalir.config.json").read_text(encoding="utf-8"))
        manifest = json.loads(
            (index_dir / "dense_embeddings.manifest.json").read_text(encoding="utf-8")
        )
        runtime = manifest.get("runtime", {})
        for key, expected in (
            ("max_seq_length", config.get("dense_max_seq_length")),
            ("query_prefix", config.get("dense_query_prefix", "")),
            ("passage_prefix", config.get("dense_passage_prefix", "")),
        ):
            if runtime.get(key) != expected:
                raise ValueError(f"{index_dir}: cache/config mismatch for {key}")
        chunk_ids = json.loads((index_dir / "dense_embeddings.ids.json").read_text(encoding="utf-8"))
        passage_embeddings = np.load(index_dir / "dense_embeddings.npy", mmap_mode="r")
        if len(chunk_ids) != len(passage_embeddings):
            raise ValueError(f"{index_dir}: dense IDs and rows differ")
        model = SentenceTransformer(
            str(config["dense_model_path"]), device=args.device, local_files_only=True
        )
        model.max_seq_length = int(config["dense_max_seq_length"])
        prefix = str(config.get("dense_query_prefix", ""))
        query_embeddings = model.encode(
            [f"{prefix}{question}" if prefix else question for _, question, _ in rows],
            batch_size=args.query_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        rankings = top_dense_documents(
            query_embeddings,
            passage_embeddings,
            [str(value) for value in chunk_ids],
            args.chunk_limit,
            args.score_batch_size,
        )
        changed_top5 = None
        mean_top5_overlap = None
        if baseline_rankings is not None:
            changed_top5 = sum(
                left[:5] != right[:5] for left, right in zip(baseline_rankings, rankings, strict=True)
            )
            mean_top5_overlap = statistics.fmean(
                len(set(left[:5]) & set(right[:5])) / 5.0
                for left, right in zip(baseline_rankings, rankings, strict=True)
            )
        reports.append({
            "index": str(index_dir),
            "configuration": {
                "max_seq_length": int(config["dense_max_seq_length"]),
                "query_prefix": prefix,
                "passage_prefix": str(config.get("dense_passage_prefix", "")),
                "chunks": len(chunk_ids),
                "chunk_limit": args.chunk_limit,
            },
            "metrics": metrics(rankings, rows),
            "versus_baseline": {
                "changed_top5_queries": changed_top5,
                "mean_top5_overlap": mean_top5_overlap,
            },
            "wall_seconds": time.perf_counter() - started,
        })
        if baseline_rankings is None:
            baseline_rankings = rankings
        del model, query_embeddings, passage_embeddings

    result = {
        "kind": "v10_3_dense_prefix_length_ablation",
        "queries": len(rows),
        "reference": str(args.queries.resolve()),
        "variants": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
