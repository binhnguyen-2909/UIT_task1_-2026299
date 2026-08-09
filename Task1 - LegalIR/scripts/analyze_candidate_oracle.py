"""Measure leak-free candidate-source recall before training a LegalIR reranker.

The official metric only accepts five documents, but a neural reranker can only
select gold documents that its retrieval stage exposed.  This tool reports
Recall@K for each independent source, the production fusion, and the oracle
union ceiling.  It uses labelled data only for evaluation, never for retrieval.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "scripts", Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from model_policy import validate_model_plan  # noqa: E402
from sqlite_retriever import SQLiteFTSRetriever  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k-initial", type=int, default=100)
    parser.add_argument("--cutoffs", default="5,10,20,35,100")
    parser.add_argument("--embedding-model")
    parser.add_argument("--model-registry", type=Path)
    parser.add_argument("--dense-device", default="auto")
    parser.add_argument("--dense-batch-size", type=int, default=32)
    parser.add_argument("--dense-max-seq-length", type=int, default=384)
    parser.add_argument("--query-prefix", default="query: ")
    parser.add_argument("--passage-prefix", default="passage: ")
    parser.add_argument("--question-dense-candidates", type=int, default=20)
    return parser.parse_args()


def ids(ranking: Iterable[Dict[str, Any]]) -> List[str]:
    return list(dict.fromkeys(str(item["doc_id"]) for item in ranking))


def recall(gold: Sequence[str], predicted: Iterable[str]) -> float:
    relevant = set(map(str, gold))
    return len(relevant & set(map(str, predicted))) / len(relevant) if relevant else 0.0


def max_score_fuse(
    base: Sequence[Dict[str, Any]],
    extra: Sequence[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    """Reuse independently computed max-score branches without querying twice."""

    by_id = {str(item["doc_id"]): dict(item) for item in base}
    for item in extra:
        doc_id = str(item["doc_id"])
        current = by_id.get(doc_id)
        if current is None or float(item["score"]) > float(current["score"]):
            by_id[doc_id] = dict(item)
    return sorted(
        by_id.values(),
        key=lambda item: (
            -float(item.get("score", 0.0)),
            -float(item.get("raw_fts_score", 0.0)),
            str(item["doc_id"]),
        ),
    )[:limit]


def rank_rrf_fuse(
    rankings: Sequence[tuple[float, Sequence[Dict[str, Any]], int]],
    limit: int,
) -> List[Dict[str, Any]]:
    """Small local RRF used only to reconstruct the production audit ranking."""

    by_id: Dict[str, Dict[str, Any]] = {}
    scores: Dict[str, float] = defaultdict(float)
    for weight, ranking, rrf_k in rankings:
        for rank, item in enumerate(ranking, 1):
            doc_id = str(item["doc_id"])
            scores[doc_id] += weight / (rrf_k + rank)
            by_id.setdefault(doc_id, dict(item))
    for doc_id, score in scores.items():
        by_id[doc_id]["score"] = score
    return sorted(
        by_id.values(),
        key=lambda item: (-float(item["score"]), str(item["doc_id"])),
    )[:limit]


def source_rankings(
    retriever: SQLiteFTSRetriever,
    query: str,
) -> Dict[str, List[Dict[str, Any]]]:
    _, tokens = retriever._match_expression(query)
    focused = retriever._chunk_candidates(tokens) if tokens else []
    alias = retriever._fuse_alias_candidates(tokens, [])
    question_lexical = retriever._fuse_question_neighbors(query, [])
    question_character = retriever._fuse_character_neighbors(query, [])
    or_bm25 = retriever._or_bm25_candidates(tokens)
    chunk_phrase = retriever._chunk_phrase_candidates(query)
    document_phrase = retriever._document_phrase_candidates(query)

    # Reuse source rankings instead of repeating the expensive FTS and
    # character-neighbor queries solely to reconstruct production fusion.
    fused = max_score_fuse(focused, alias, retriever.top_k_initial)
    fused = max_score_fuse(fused, question_lexical, retriever.top_k_initial)
    fused = rank_rrf_fuse(
        ((3.0, fused, 5), (1.0, question_character, 5)),
        retriever.top_k_initial,
    )
    fused = retriever._fuse_or_bm25(fused, or_bm25)
    fused = retriever._fuse_chunk_phrase(fused, chunk_phrase)
    fused = retriever._fuse_document_phrase(fused, document_phrase)

    rankings: Dict[str, List[Dict[str, Any]]] = {
        "focused_chunk": focused,
        "training_alias": alias,
        "question_lexical": question_lexical,
        "question_character": question_character,
        "or_bm25": or_bm25,
        "chunk_phrase": chunk_phrase,
        # Preserve the exact list entering the final dense/question RRF.  This
        # enables leak-free offline tuning of only the final fusion weights
        # without rerunning the expensive FTS branches for every grid point.
        "lexical_fusion": [dict(item) for item in fused],
    }
    if document_phrase:
        rankings["document_phrase"] = document_phrase

    if retriever._dense is not None:
        if retriever._dense._embeddings is None:
            retriever._dense.load(retriever.db_path.parent / "dense_embeddings")
        query_embedding = retriever._dense.encode_query(query)
        dense = retriever._dense.retrieve_embedding(
            query_embedding, top_k=retriever.top_k_initial
        )
        rankings["dense_passage"] = dense

        question_dense: List[Dict[str, Any]] = []
        if retriever._question_dense is not None:
            if retriever._question_dense._embeddings is None:
                question_path = retriever.db_path.parent / "question_dense_embeddings"
                retriever._question_dense.load(
                    question_path,
                    expected_runtime=retriever._dense.runtime_signature(),
                )
            question_dense = retriever._question_dense.retrieve_embedding(
                query_embedding,
                top_k_questions=retriever.question_dense_candidates,
            )
            rankings["question_dense"] = question_dense

        fused = retriever._dense.rrf_fuse_many((
            (1.0, fused, retriever.dense_rrf_k),
            (retriever.dense_weight, dense, retriever.dense_rrf_k),
            (
                retriever.question_dense_weight,
                question_dense,
                retriever.question_dense_rrf_k,
            ),
        ))

    retriever._attach_source_passages(
        fused,
        (
            chunk_phrase,
            focused,
            or_bm25,
            question_lexical,
            question_character,
            alias,
            rankings.get("dense_passage", []),
            rankings.get("question_dense", []),
        ),
        retriever.rerank_passages_per_doc,
    )

    rankings["production_fusion"] = fused
    return rankings


def main() -> int:
    args = parse_args()
    cutoffs = sorted({int(value) for value in args.cutoffs.split(",")})
    if not cutoffs or min(cutoffs) <= 0 or args.top_k_initial <= 0:
        raise ValueError("Cutoffs and top-k-initial must be positive")
    data = json.loads(args.data.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Evaluation data must be keyed by question ID")
    items = list(data.items())[: args.limit]
    embedding_model = None
    if args.embedding_model:
        embedding_model = str(Path(args.embedding_model).resolve())
        validate_model_plan(args.model_registry, {"embedding": embedding_model})

    retriever = SQLiteFTSRetriever(
        args.index,
        use_reranker=False,
        top_k_initial=args.top_k_initial,
        enable_dense=bool(embedding_model),
        dense_model_path=embedding_model,
        dense_device=args.dense_device,
        dense_batch_size=args.dense_batch_size,
        dense_max_seq_length=args.dense_max_seq_length,
        dense_query_prefix=args.query_prefix,
        dense_passage_prefix=args.passage_prefix,
        dense_weight=0.05,
        question_dense_weight=0.05,
        question_dense_candidates=args.question_dense_candidates,
    )
    sums: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    oracle_sums: Dict[int, float] = defaultdict(float)
    misses: List[Dict[str, Any]] = []
    try:
        for position, (question_id, item) in enumerate(items, 1):
            gold = [str(value) for value in item.get("answer", [])]
            rankings = source_rankings(retriever, str(item["question"]))
            source_ids = {name: ids(ranking) for name, ranking in rankings.items()}
            for name, ranking_ids in source_ids.items():
                for cutoff in cutoffs:
                    sums[name][cutoff] += recall(gold, ranking_ids[:cutoff])
            for cutoff in cutoffs:
                union = {
                    doc_id
                    for name, ranking_ids in source_ids.items()
                    if name != "production_fusion"
                    for doc_id in ranking_ids[:cutoff]
                }
                oracle_sums[cutoff] += recall(gold, union)
            production_ids = source_ids["production_fusion"]
            missing = [doc_id for doc_id in gold if doc_id not in production_ids]
            if missing:
                misses.append({
                    "question_id": str(question_id),
                    "missing_gold": missing,
                    "question": str(item["question"]),
                })
            if position % 25 == 0:
                print(f"Candidate audit: {position}/{len(items)}", flush=True)
    finally:
        retriever.close()

    denominator = len(items) or 1
    report = {
        "data": str(args.data.resolve()),
        "index": str(args.index.resolve()),
        "questions": len(items),
        "top_k_initial": args.top_k_initial,
        "cutoffs": cutoffs,
        "embedding_model": embedding_model,
        "source_macro_recall": {
            name: {
                str(cutoff): values[cutoff] / denominator
                for cutoff in cutoffs
            }
            for name, values in sorted(sums.items())
        },
        "oracle_union_macro_recall": {
            str(cutoff): oracle_sums[cutoff] / denominator for cutoff in cutoffs
        },
        "production_candidate_miss_questions": len(misses),
        "misses": misses,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        summary = dict(report)
        summary["misses"] = f"written to {args.output.resolve()}"
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
