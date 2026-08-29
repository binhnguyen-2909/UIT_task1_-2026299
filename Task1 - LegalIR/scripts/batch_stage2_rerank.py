"""Batch Stage-2 reranking from a parallel Stage-1 candidate audit.

The normal online retriever reranks one query at a time.  This utility keeps
the same CrossEncoder and RRF formula but batches all query–passage pairs, so
fit/holdout ablations are practical on a small GPU.  It never expands the
candidate set and therefore cannot hide a Stage-1 recall failure.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

try:
    from .evaluate import official_metrics
    from .main import _write_bundle
    from .reranker import Reranker, reciprocal_rank_fuse
except ImportError:
    from evaluate import official_metrics
    from main import _write_bundle
    from reranker import Reranker, reciprocal_rank_fuse

COMMON_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(COMMON_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMMON_SCRIPTS))
from model_policy import validate_model_plan  # noqa: E402


TOKEN_RE = re.compile(r"[0-9a-zà-ỹđ]+", re.IGNORECASE)


def normalize_tokens(value: str) -> set[str]:
    return {token.casefold() for token in TOKEN_RE.findall(value or "") if len(token) >= 3}


def load_stage1(path: Path) -> Tuple[List[Dict[str, Any]], List[List[str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    queries = payload.get("queries")
    candidates = payload.get("stage1_candidates")
    if not isinstance(queries, list) or not isinstance(candidates, list):
        raise ValueError("Stage-1 audit must contain queries and stage1_candidates")
    if len(queries) != len(candidates):
        raise ValueError("Stage-1 query/candidate lengths differ")
    return queries, [[str(value) for value in row] for row in candidates]


def load_references(path: Path, qids: Sequence[str]) -> Dict[str, Dict[str, List[str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = {
        str(qid): {"answer": [str(value) for value in payload[str(qid)]["answer"]]}
        for qid in qids
    }
    if set(selected) != set(qids):
        raise ValueError("Reference does not cover all Stage-1 questions")
    return selected


def open_readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        path.resolve().as_uri() + "?mode=ro", uri=True, check_same_thread=False
    )
    connection.row_factory = sqlite3.Row
    for pragma in (
        "PRAGMA cache_size=-524288",
        "PRAGMA temp_store=MEMORY",
        "PRAGMA mmap_size=2147483648",
    ):
        try:
            connection.execute(pragma)
        except sqlite3.DatabaseError:
            pass
    return connection


def load_passage_store(
    connection: sqlite3.Connection,
    doc_ids: Sequence[str],
) -> Dict[str, List[Dict[str, Any]]]:
    unique_ids = list(dict.fromkeys(str(value) for value in doc_ids))
    store: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if not unique_ids:
        return store
    placeholders = ",".join("?" for _ in unique_ids)
    rows = connection.execute(
        f"""
        SELECT doc_id, chunk_id, title, passage
        FROM chunks
        WHERE doc_id IN ({placeholders})
        ORDER BY doc_id, chunk_id
        """,
        unique_ids,
    )
    for row in rows:
        text = "\n".join(value for value in (row["title"], row["passage"]) if value)
        if text:
            store[str(row["doc_id"])].append({
                "doc_id": str(row["doc_id"]),
                "chunk_id": str(row["chunk_id"]),
                "text": text,
                # Passage overlap is query-dependent, but passage tokenization
                # is not.  Cache it once instead of repeating the same regex
                # work for every query that retrieves this document.
                "tokens": frozenset(normalize_tokens(text)),
            })
    return store


def hydrate(
    query: str,
    doc_ids: Sequence[str],
    passages_per_doc: int,
    store: Mapping[str, Sequence[Mapping[str, Any]]],
) -> List[Dict[str, Any]]:
    unique_ids = list(dict.fromkeys(str(value) for value in doc_ids))
    if not unique_ids:
        return []
    query_tokens = normalize_tokens(query)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for doc_id in unique_ids:
        for row in store.get(doc_id, ()):
            text = str(row["text"])
            cached_tokens = row.get("tokens")
            text_tokens = (
                cached_tokens
                if isinstance(cached_tokens, (set, frozenset))
                else normalize_tokens(text)
            )
            overlap = len(query_tokens & text_tokens)
            grouped[doc_id].append({
                "doc_id": doc_id,
                "chunk_id": str(row["chunk_id"]),
                "text": text,
                "overlap": overlap,
            })
    output: List[Dict[str, Any]] = []
    for doc_id in unique_ids:
        passages = sorted(
            grouped.get(doc_id, []),
            key=lambda item: (-int(item["overlap"]), str(item["chunk_id"])),
        )[:passages_per_doc]
        if not passages:
            passages = [{"doc_id": doc_id, "chunk_id": doc_id, "text": "", "overlap": 0}]
        output.append({
            "doc_id": doc_id,
            "score": 1.0,
            "text": passages[0]["text"],
            "passages": [item["text"] for item in passages],
        })
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-audit", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-registry", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-limit", type=int, default=20)
    parser.add_argument("--passages-per-doc", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--neural-weight", type=float, default=1.0)
    parser.add_argument("--rrf-k", type=int, default=10)
    parser.add_argument(
        "--reuse-audit",
        type=Path,
        help="Reuse compatible per-document neural scores and evaluate only new candidates",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.candidate_limit <= 0 or args.passages_per_doc <= 0 or args.batch_size <= 0:
        raise ValueError("candidate-limit, passages-per-doc and batch-size must be positive")
    started = time.perf_counter()
    compliance = validate_model_plan(args.model_registry, {"embedding": None, "reranker": str(args.model.resolve())})
    queries, candidate_rows = load_stage1(args.stage1_audit)
    cached_scores: Dict[str, Dict[str, float]] = {}
    reuse_candidate_limit: int | None = None
    if args.reuse_audit is not None:
        reuse = json.loads(args.reuse_audit.read_text(encoding="utf-8"))
        reuse_candidate_limit = int(reuse.get("candidate_limit", 0))
        if reuse_candidate_limit <= 0:
            raise ValueError("Reusable Stage-2 audit has invalid candidate_limit")
        expected = {
            "passages_per_doc": args.passages_per_doc,
            "max_length": args.max_length,
            "fp16": args.fp16,
        }
        for key, value in expected.items():
            if reuse.get(key) != value:
                raise ValueError(
                    f"Reusable Stage-2 audit has incompatible {key}: "
                    f"{reuse.get(key)!r} != {value!r}"
                )
        if Path(str(reuse.get("model", ""))).resolve() != args.model.resolve():
            raise ValueError("Reusable Stage-2 audit uses a different model")
        for item in reuse.get("queries", []):
            qid = str(item.get("question_id", ""))
            values = item.get("neural_scores")
            if qid and isinstance(values, dict):
                cached_scores[qid] = {
                    str(doc_id): float(score) for doc_id, score in values.items()
                }
        expected_qids = {str(item["query_id"]) for item in queries}
        if set(cached_scores) != expected_qids:
            raise ValueError("Reusable Stage-2 audit does not cover the Stage-1 query IDs")
    connection = open_readonly(args.index_dir / "legalir.sqlite3")
    reranker = Reranker(
        str(args.model.resolve()),
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        passage_aggregation="top3" if args.passages_per_doc >= 3 else "max",
        use_fp16=args.fp16,
    )

    all_doc_ids: List[str] = []
    for query_item, row in zip(queries, candidate_rows):
        qid = str(query_item["query_id"])
        cached = cached_scores.get(qid, {})
        all_doc_ids.extend(
            doc_id
            for doc_id in list(dict.fromkeys(row))[: args.candidate_limit]
            if doc_id not in cached
        )
    passage_store = load_passage_store(connection, all_doc_ids)

    lexical_by_query: List[List[Dict[str, Any]]] = []
    pairs: List[Tuple[str, str]] = []
    owners: List[Tuple[int, int, int]] = []
    for query_index, query_item in enumerate(queries):
        query = str(query_item["query"])
        qid = str(query_item["query_id"])
        cached = cached_scores.get(qid, {})
        docs = list(dict.fromkeys(candidate_rows[query_index]))[: args.candidate_limit]
        lexical = hydrate(query, docs, args.passages_per_doc, passage_store)
        lexical_by_query.append(lexical)
        for doc_index, item in enumerate(lexical):
            if str(item["doc_id"]) in cached:
                continue
            for passage_index, passage in enumerate(item["passages"]):
                pairs.append((query, str(passage)))
                owners.append((query_index, doc_index, passage_index))
    connection.close()

    print(f"Stage-2 scoring {len(pairs)} query-passage pairs...", flush=True)
    flat_scores = [float(value) for value in reranker.model.predict(
        pairs, batch_size=args.batch_size, show_progress_bar=False
    )]
    scores_by_query: List[List[List[float]]] = [
        [[] for _ in lexical] for lexical in lexical_by_query
    ]
    for (query_index, doc_index, _), score in zip(owners, flat_scores):
        scores_by_query[query_index][doc_index].append(float(score))

    submission: Dict[str, Dict[str, List[str]]] = {}
    traces: List[Dict[str, Any]] = []
    reused_document_scores = 0
    new_document_scores = 0
    for query_index, query_item in enumerate(queries):
        qid = str(query_item["query_id"])
        cached = cached_scores.get(qid, {})
        lexical = lexical_by_query[query_index]
        neural: List[Dict[str, Any]] = []
        for doc_index, item in enumerate(lexical):
            doc_id = str(item["doc_id"])
            if doc_id in cached:
                values: List[float] = []
                aggregate = float(cached[doc_id])
                reused_document_scores += 1
            else:
                values = scores_by_query[query_index][doc_index] or [float("-inf")]
                aggregate = reranker._aggregate_passage_scores(values)
                new_document_scores += 1
            neural.append({
                "doc_id": doc_id,
                "score": aggregate,
                "rerank_score": aggregate,
                "passage_scores": values,
            })
        # Reranker.rerank() sorts by rerank_score before normal online fusion.
        # Preserve that contract here; otherwise RRF would silently reuse the
        # Stage-1 candidate order and the CrossEncoder could not affect rank.
        neural.sort(
            key=lambda item: (-float(item["rerank_score"]), str(item["doc_id"]))
        )
        fused = reciprocal_rank_fuse(
            lexical, neural, neural_weight=args.neural_weight,
            rrf_k=args.rrf_k, top_k=args.top_k,
        )
        submission[qid] = {"answer": [str(item["doc_id"]) for item in fused]}
        traces.append({
            "question_id": qid,
            "candidate_count": len(lexical),
            "pairs": sum(len(values) for values in scores_by_query[query_index]),
            "reused_document_scores": sum(
                str(item["doc_id"]) in cached for item in lexical
            ),
            "answers": submission[qid]["answer"],
            "neural_scores": {
                str(item["doc_id"]): float(item["rerank_score"]) for item in neural
            },
        })
    _write_bundle(submission, args.output)
    audit: Dict[str, Any] = {
        "kind": "legalir_batch_stage2_rerank",
        "stage1_audit": str(args.stage1_audit.resolve()),
        "index": str((args.index_dir / "legalir.sqlite3").resolve()),
        "model": str(args.model.resolve()),
        "model_compliance": compliance,
        "candidate_limit": args.candidate_limit,
        "passages_per_doc": args.passages_per_doc,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "fp16": args.fp16,
        "pairs": len(pairs),
        "reuse_audit": str(args.reuse_audit.resolve()) if args.reuse_audit else None,
        "reuse_candidate_limit": reuse_candidate_limit,
        "reused_document_scores": reused_document_scores,
        "new_document_scores": new_document_scores,
        "questions": len(queries),
        "output": str(args.output.resolve()),
        "queries": traces,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if args.reference:
        references = load_references(args.reference, list(submission))
        audit["metrics"] = official_metrics(submission, references)
    audit_path = args.output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "questions": len(queries),
        "pairs": len(pairs),
        "reused_document_scores": reused_document_scores,
        "new_document_scores": new_document_scores,
        "metrics": audit.get("metrics"),
        "elapsed_seconds": audit["elapsed_seconds"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
