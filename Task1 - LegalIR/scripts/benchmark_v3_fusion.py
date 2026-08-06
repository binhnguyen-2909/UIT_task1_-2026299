"""Fuse recall-v2 passage ranking with supervised OR-alias evidence on holdout."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

try:
    from .evaluate import official_metrics
    from .sqlite_retriever import SQLiteFTSRetriever
except ImportError:
    from evaluate import official_metrics
    from sqlite_retriever import SQLiteFTSRetriever


def quote_or(tokens: Sequence[str]) -> str:
    return " OR ".join(
        f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens
    )


def rrf(rankings: Sequence[Tuple[float, Sequence[str]]], k: int) -> List[str]:
    scores: Dict[str, float] = defaultdict(float)
    for weight, ranking in rankings:
        for rank, doc_id in enumerate(ranking, 1):
            scores[str(doc_id)] += weight / (k + rank)
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))


def supervised_ranking(
    retriever: SQLiteFTSRetriever,
    query: str,
    limit: int,
) -> Tuple[List[str], List[str], List[str]]:
    tokens = list(dict.fromkeys(retriever.tokenizer.tokenize(query)))
    expression = quote_or(tokens)
    if not expression:
        return [], [], []
    question_rows = retriever.connection.execute(
        """
        SELECT answers FROM question_examples WHERE question_examples MATCH ?
        ORDER BY bm25(question_examples, 0.0, 0.0, 1.0), question_id LIMIT ?
        """,
        (expression, limit),
    )
    question_scores: Dict[str, float] = defaultdict(float)
    for rank, row in enumerate(question_rows, 1):
        for doc_id in map(str, json.loads(row["answers"])):
            question_scores[doc_id] += 1.0 / (20.0 + rank)
    question = sorted(
        question_scores,
        key=lambda doc_id: (-question_scores[doc_id], doc_id),
    )
    alias = [
        str(row["doc_id"])
        for row in retriever.connection.execute(
            """
            SELECT doc_id FROM aliases WHERE aliases MATCH ?
            ORDER BY bm25(aliases, 0.0, 1.0), doc_id LIMIT ?
            """,
            (expression, limit),
        )
    ]
    return rrf([(1.0, question), (1.0, alias)], 20), question, alias


def v2_rankings(
    retriever: SQLiteFTSRetriever, query: str
) -> Tuple[List[str], List[str], List[str]]:
    _, terms = retriever._match_expression(query)
    candidates = retriever._chunk_candidates(terms) if terms else []
    candidates = retriever._fuse_alias_candidates(terms, candidates)
    candidates = retriever._fuse_question_neighbors(query, candidates)
    candidates = retriever._fuse_character_neighbors(query, candidates)
    focused = [str(item["doc_id"]) for item in candidates]
    or_candidates = retriever._or_bm25_candidates(terms)
    or_ranking = [str(item["doc_id"]) for item in or_candidates]
    fused = retriever._fuse_or_bm25(candidates, or_candidates)
    return [str(item["doc_id"]) for item in fused], focused, or_ranking


def score(predictions: Dict[str, List[str]], references: Dict[str, Any]) -> Dict[str, float]:
    return official_metrics(
        {qid: {"answer": docs[:5]} for qid, docs in predictions.items()},
        references,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--supervised-limit", type=int, default=160)
    args = parser.parse_args()

    raw = json.loads(args.train.read_text(encoding="utf-8"))
    items = [(str(qid), item) for qid, item in raw.items()]
    training = dict(record for index, record in enumerate(items) if index % 5 != args.fold)
    holdout = [record for index, record in enumerate(items) if index % 5 == args.fold][:args.limit]
    retriever = SQLiteFTSRetriever.load(args.index)
    retriever.replace_training_signals(training)

    variants: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    references: Dict[str, Any] = {}
    cached: Dict[str, Any] = {}
    weights = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
    try:
        for position, (qid, item) in enumerate(holdout, 1):
            references[qid] = {"answer": [str(doc) for doc in item["answer"]]}
            base, focused, or_ranking = v2_rankings(retriever, item["question"])
            supervised, question, alias = supervised_ranking(
                retriever, item["question"], args.supervised_limit
            )
            variants["v2"][qid] = base
            variants["focused_supervised_char"][qid] = focused
            variants["or_bm25"][qid] = or_ranking
            variants["supervised_sum_alias"][qid] = supervised
            for component_k in (0, 5, 10, 20, 40, 60):
                for or_weight in (0.5, 1.0, 1.5, 2.0, 7.0 / 3.0, 3.0, 4.0):
                    name = f"focused_or_w{or_weight:g}_k{component_k}"
                    variants[name][qid] = rrf(
                        [(1.0, focused), (or_weight, or_ranking)], component_k
                    )
            for k in (10, 20, 40, 60):
                for weight in weights:
                    name = f"v2_plus_supervised_w{weight:g}_k{k}"
                    variants[name][qid] = rrf(
                        [(1.0, base), (weight, supervised)], k
                    )
            cached[qid] = {
                "truth": references[qid]["answer"],
                "v2": base,
                "focused": focused,
                "or_bm25": or_ranking,
                "supervised": supervised[:80],
                "question": question[:80],
                "alias": alias[:80],
            }
            if position % 10 == 0 or position == len(holdout):
                print(f"progress {position}/{len(holdout)}", flush=True)
    finally:
        retriever.close()

    metrics = {name: score(value, references) for name, value in variants.items()}
    report = {
        "fold": args.fold,
        "questions": len(holdout),
        "training_questions": len(training),
        "metrics": dict(sorted(
            metrics.items(), key=lambda item: (-item[1]["recall"], -item[1]["precision"], item[0])
        )),
        "best": max(metrics, key=lambda name: (metrics[name]["recall"], metrics[name]["precision"])),
        "rankings": cached,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": report["best"], "top": list(report["metrics"].items())[:10]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
