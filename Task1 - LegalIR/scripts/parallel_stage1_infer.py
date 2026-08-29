"""Parallel, dense-precomputed Stage-1 inference for the LegalIR index.

This runner is intentionally limited to Stage 1 and never loads a reranker.
It batch-encodes all queries once, then uses independent read-only SQLite
connections in worker threads.  The ranking code and candidate formula are
unchanged; this only removes repeated single-query GPU encoding and allows
SQLite FTS reads to overlap.  It is useful for fit/holdout candidate-oracle
benchmarks and for producing a compliant public Stage-1 candidate.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from .evaluate import official_metrics
    from .main import _write_bundle
    from .sqlite_retriever import SQLiteFTSRetriever
except ImportError:
    from evaluate import official_metrics
    from main import _write_bundle
    from sqlite_retriever import SQLiteFTSRetriever

COMMON_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(COMMON_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMMON_SCRIPTS))
from model_policy import validate_model_plan  # noqa: E402


def load_input(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Input must be an object: {path}")
    for qid, item in payload.items():
        if not isinstance(item, dict) or not isinstance(item.get("question"), str):
            raise ValueError(f"{qid}: missing string question")
    return {str(qid): dict(item) for qid, item in payload.items()}


def cache_ids(path: Optional[Path]) -> Optional[List[str]]:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as stream:
        header = json.loads(next(stream))
        if not isinstance(header, dict) or header.get("_meta", {}).get("schema") != 1:
            raise ValueError(f"Unsupported rank cache header: {path}")
        return [str(json.loads(line)["question_id"]) for line in stream if line.strip()]


def load_query_encoder(config: Dict[str, Any], device: str) -> Tuple[Any, str]:
    model_path = config.get("dense_model_path")
    if not model_path:
        raise ValueError("The index has no dense_model_path")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_path), device=device, local_files_only=True)
    model.max_seq_length = int(config.get("dense_max_seq_length", 384))
    return model, str(config.get("dense_query_prefix", ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-registry", type=Path, required=True)
    parser.add_argument("--ids-cache", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--limit", type=int, help="Optional deterministic prefix limit after --ids-cache")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--profile", choices=("full", "chunk_only", "balanced", "no_phrase", "fast"), default="full",
        help="chunk_only keeps focused+dense+chunk-phrase; balanced also keeps bounded OR/char; no_phrase removes chunk phrase; fast removes optional branches",
    )
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--top-k-initial", type=int,
                        help="Override the Stage-1 internal candidate pool")
    parser.add_argument("--dense-weight", type=float,
                        help="Override dense RRF weight without rebuilding the index")
    parser.add_argument("--dense-rrf-k", type=int,
                        help="Override dense RRF k without rebuilding the index")
    parser.add_argument("--question-dense-weight", type=float,
                        help="Override question-neighbor dense RRF weight")
    parser.add_argument("--question-dense-rrf-k", type=int,
                        help="Override question-neighbor dense RRF k")
    parser.add_argument("--question-dense-candidates", type=int,
                        help="Override question-neighbor dense candidate count")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers <= 0 or args.encode_batch_size <= 0:
        raise ValueError("workers and encode-batch-size must be positive")
    started = time.perf_counter()
    index_dir = args.index_dir.resolve()
    db_path = index_dir / "legalir.sqlite3"
    config = json.loads((index_dir / "legalir.config.json").read_text(encoding="utf-8"))
    model_check = validate_model_plan(
        args.model_registry,
        {"embedding": config.get("dense_model_path"), "reranker": None},
    )
    input_rows = load_input(args.input)
    selected_ids = cache_ids(args.ids_cache)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if selected_ids is not None:
        if args.limit is not None:
            selected_ids = selected_ids[: args.limit]
        missing = [qid for qid in selected_ids if qid not in input_rows]
        if missing:
            raise ValueError(f"IDs from cache are missing from input: {missing[:5]}")
        input_rows = {qid: input_rows[qid] for qid in selected_ids}
    elif args.limit is not None:
        input_rows = dict(list(input_rows.items())[: args.limit])
    if not input_rows:
        raise ValueError("No questions selected")

    questions = list(input_rows.items())
    encoder, query_prefix = load_query_encoder(config, args.device)
    encoded = encoder.encode(
        [f"{query_prefix}{item['question']}" if query_prefix else item["question"]
         for _, item in questions],
        batch_size=args.encode_batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    local = threading.local()
    retrievers: List[SQLiteFTSRetriever] = []
    retriever_lock = threading.Lock()

    def get_retriever() -> SQLiteFTSRetriever:
        retriever = getattr(local, "retriever", None)
        if retriever is None:
            # The query vectors are already encoded. CPU read-only workers do
            # not instantiate another transformer model.
            retriever = SQLiteFTSRetriever.load(
                db_path, device="cpu", disable_reranker=True
            )
            if args.top_k_initial is not None:
                if args.top_k_initial <= 0:
                    raise ValueError("top-k-initial must be positive")
                retriever.top_k_initial = int(args.top_k_initial)
            for name in (
                "dense_weight",
                "dense_rrf_k",
                "question_dense_weight",
                "question_dense_rrf_k",
                "question_dense_candidates",
            ):
                value = getattr(args, name)
                if value is not None:
                    if name.endswith("weight") and float(value) < 0:
                        raise ValueError(f"{name} must be non-negative")
                    if name.endswith("rrf_k") and int(value) < 0:
                        raise ValueError(f"{name} must be non-negative")
                    if name == "question_dense_candidates" and int(value) <= 0:
                        raise ValueError("question-dense-candidates must be positive")
                    setattr(retriever, name, type(getattr(retriever, name))(value))
            if args.profile in {"balanced", "no_phrase", "fast", "chunk_only"}:
                retriever.enable_chunk_phrase = False
            if args.profile == "chunk_only":
                retriever.enable_chunk_phrase = True
                retriever.or_query_terms = 0
                retriever.question_dense_weight = 0.0
                retriever._question_dense = None
                retriever._fuse_character_neighbors = lambda query, candidates: candidates
            if args.profile == "balanced":
                retriever.enable_chunk_phrase = True
                retriever.chunk_phrase_pair_limit = min(retriever.chunk_phrase_pair_limit, 4)
                retriever.chunk_phrase_result_limit = min(retriever.chunk_phrase_result_limit, 80)
                retriever.or_query_terms = min(retriever.or_query_terms, 8)
                retriever.or_chunk_limit = min(retriever.or_chunk_limit, 200)
            if args.profile == "fast":
                # Keep focused lexical + dense RRF + legal citation matching,
                # while isolating the three optional high-latency branches.
                retriever.or_query_terms = 0
                retriever.question_dense_weight = 0.0
                retriever._question_dense = None
                retriever._fuse_character_neighbors = lambda query, candidates: candidates
            local.retriever = retriever
            with retriever_lock:
                retrievers.append(retriever)
        return retriever

    def run_one(position: int) -> Tuple[int, Any, List[str]]:
        qid, item = questions[position]
        retriever = get_retriever()
        result = retriever.retrieve(
            item["question"], qid, args.top_k, query_embedding=encoded[position]
        )
        candidate_ids = [
            str(candidate["doc_id"])
            for candidate in retriever.last_stage1_candidates
        ]
        return position, result, candidate_ids

    result_by_position: Dict[int, Any] = {}
    candidates_by_position: Dict[int, List[str]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, position) for position in range(len(questions))]
        for completed, future in enumerate(as_completed(futures), 1):
            position, result, candidate_ids = future.result()
            result_by_position[position] = result
            candidates_by_position[position] = candidate_ids
            if completed % args.checkpoint_every == 0 or completed == len(futures):
                print(f"Parallel Stage-1 progress: {completed}/{len(futures)}", flush=True)
    for retriever in retrievers:
        retriever.close()

    ordered_results = [result_by_position[position] for position in range(len(questions))]
    submission = {
        result.query_id: {"answer": [item.doc_id for item in result.results]}
        for result in ordered_results
    }
    _write_bundle(submission, args.output)
    audit: Dict[str, Any] = {
        "kind": "legalir_parallel_stage1_inference",
        "index": str(db_path),
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "questions": len(questions),
        "workers": args.workers,
        "encode_batch_size": args.encode_batch_size,
        "dense_model": config.get("dense_model_path"),
        "model_compliance": model_check,
        "dense_max_seq_length": config.get("dense_max_seq_length"),
        "reranker_used": False,
        "profile": args.profile,
        "overrides": {
            key: getattr(args, key)
            for key in (
                "top_k_initial",
                "dense_weight",
                "dense_rrf_k",
                "question_dense_weight",
                "question_dense_rrf_k",
                "question_dense_candidates",
            )
            if getattr(args, key) is not None
        },
        "latency_seconds": {
            "mean": statistics.fmean(result.processing_time for result in ordered_results),
            "p95": sorted(result.processing_time for result in ordered_results)[
                max(0, int(0.95 * len(ordered_results)) - 1)
            ],
            "total": sum(result.processing_time for result in ordered_results),
            "wall": time.perf_counter() - started,
        },
        "queries": [result.to_dict() for result in ordered_results],
        "stage1_candidates": [
            candidates_by_position[position] for position in range(len(questions))
        ],
    }
    if args.reference:
        references = load_input(args.reference)
        references = {qid: {"answer": item["answer"]} for qid, item in references.items()
                      if qid in submission}
        if set(references) != set(submission):
            raise ValueError("Reference IDs do not match selected questions")
        audit["metrics"] = official_metrics(submission, references)
        oracle: Dict[str, float] = {}
        for cutoff in (5, 20, 50, 100, 200):
            values = []
            for position, (qid, _) in enumerate(questions):
                gold = {str(value) for value in references[qid]["answer"]}
                candidates = set(candidates_by_position[position][:cutoff])
                values.append(len(gold & candidates) / max(1, len(gold)))
            oracle[f"recall_at_{cutoff}"] = statistics.fmean(values)
        audit["candidate_oracle"] = oracle
    audit_path = args.output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "questions": len(questions),
        "metrics": audit.get("metrics"),
        "candidate_oracle": audit.get("candidate_oracle"),
        "wall_seconds": audit["latency_seconds"]["wall"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
