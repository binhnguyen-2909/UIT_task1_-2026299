"""Leak-free supervised retrieval experiments for LegalIR recall v3.

This script intentionally uses only the labelled questions and SQLite FTS5.
It is fast enough to evaluate all five deterministic folds without touching
the 1.27 GB passage index.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

try:
    from .bm25_retriever import VietnameseTokenizer
    from .evaluate import official_metrics
except ImportError:
    from bm25_retriever import VietnameseTokenizer
    from evaluate import official_metrics


def expression(tokens: Sequence[str]) -> str:
    return " OR ".join(
        f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens
    )


def build_index(records: Sequence[Tuple[str, Dict[str, Any]]]) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript("""
        CREATE VIRTUAL TABLE questions USING fts5(
            question_id UNINDEXED,
            answers UNINDEXED,
            question,
            tokenize='unicode61 remove_diacritics 0'
        );
        CREATE VIRTUAL TABLE aliases USING fts5(
            doc_id UNINDEXED,
            alias_text,
            tokenize='unicode61 remove_diacritics 0'
        );
    """)
    aliases: Dict[str, List[str]] = defaultdict(list)
    connection.executemany(
        "INSERT INTO questions(question_id, answers, question) VALUES (?, ?, ?)",
        (
            (qid, json.dumps(item["answer"]), item["question"])
            for qid, item in records
        ),
    )
    for _, item in records:
        for doc_id in map(str, item["answer"]):
            aliases[doc_id].append(item["question"])
    connection.executemany(
        "INSERT INTO aliases(doc_id, alias_text) VALUES (?, ?)",
        ((doc_id, "\n".join(questions)) for doc_id, questions in aliases.items()),
    )
    connection.execute("INSERT INTO questions(questions) VALUES ('optimize')")
    connection.execute("INSERT INTO aliases(aliases) VALUES ('optimize')")
    return connection


def question_rankings(
    connection: sqlite3.Connection,
    query_expression: str,
    limit: int,
) -> Dict[str, List[str]]:
    rows = list(connection.execute(
        """
        SELECT question_id, answers, -bm25(questions, 0.0, 0.0, 1.0) AS score
        FROM questions WHERE questions MATCH ?
        ORDER BY bm25(questions, 0.0, 0.0, 1.0), question_id
        LIMIT ?
        """,
        (query_expression, limit),
    ))
    maximum: Dict[str, float] = {}
    sum_rrf: Dict[str, float] = defaultdict(float)
    score_sum: Dict[str, float] = defaultdict(float)
    for rank, row in enumerate(rows, 1):
        neighbor_score = max(0.0, float(row["score"]))
        for doc_id in map(str, json.loads(row["answers"])):
            maximum[doc_id] = max(maximum.get(doc_id, 0.0), 1.0 / (20.0 + rank))
            sum_rrf[doc_id] += 1.0 / (20.0 + rank)
            score_sum[doc_id] += neighbor_score
    return {
        "question_max": sorted(maximum, key=lambda doc: (-maximum[doc], doc)),
        "question_sum_rrf": sorted(sum_rrf, key=lambda doc: (-sum_rrf[doc], doc)),
        "question_score_sum": sorted(score_sum, key=lambda doc: (-score_sum[doc], doc)),
    }


def alias_ranking(
    connection: sqlite3.Connection,
    query_expression: str,
    limit: int,
) -> List[str]:
    rows = connection.execute(
        """
        SELECT doc_id FROM aliases WHERE aliases MATCH ?
        ORDER BY bm25(aliases, 0.0, 1.0), doc_id LIMIT ?
        """,
        (query_expression, limit),
    )
    return [str(row["doc_id"]) for row in rows]


def rrf(rankings: Iterable[Tuple[float, Sequence[str]]], k: int = 20) -> List[str]:
    scores: Dict[str, float] = defaultdict(float)
    for weight, ranking in rankings:
        for rank, doc_id in enumerate(ranking, 1):
            scores[doc_id] += weight / (k + rank)
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))


def metrics(predictions: Dict[str, List[str]], references: Dict[str, Any]) -> Dict[str, float]:
    return official_metrics(
        {qid: {"answer": docs[:5]} for qid, docs in predictions.items()},
        references,
    )


def benchmark_fold(
    items: Sequence[Tuple[str, Dict[str, Any]]],
    fold: int,
    neighbor_limit: int,
) -> Dict[str, Dict[str, float]]:
    training = [record for index, record in enumerate(items) if index % 5 != fold]
    holdout = [record for index, record in enumerate(items) if index % 5 == fold]
    connection = build_index(training)
    tokenizer = VietnameseTokenizer(bigrams=False)
    predictions: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    references: Dict[str, Any] = {}
    try:
        for qid, item in holdout:
            references[qid] = {"answer": [str(doc) for doc in item["answer"]]}
            tokens = list(dict.fromkeys(tokenizer.tokenize(item["question"])))
            query_expression = expression(tokens)
            if not query_expression:
                rankings = {
                    "question_max": [],
                    "question_sum_rrf": [],
                    "question_score_sum": [],
                }
                alias = []
            else:
                rankings = question_rankings(connection, query_expression, neighbor_limit)
                alias = alias_ranking(connection, query_expression, neighbor_limit)
            for name, ranking in rankings.items():
                predictions[name][qid] = ranking
            predictions["alias_or"][qid] = alias
            predictions["sum_alias_1_1"][qid] = rrf([
                (1.0, rankings["question_sum_rrf"]), (1.0, alias),
            ])
            predictions["max_alias_1_1"][qid] = rrf([
                (1.0, rankings["question_max"]), (1.0, alias),
            ])
            predictions["score_alias_1_1"][qid] = rrf([
                (1.0, rankings["question_score_sum"]), (1.0, alias),
            ])
    finally:
        connection.close()
    return {name: metrics(values, references) for name, values in predictions.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--neighbor-limit", type=int, default=160)
    parser.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    data = json.loads(args.train.read_text(encoding="utf-8"))
    items = [(str(qid), item) for qid, item in data.items()]
    fold_reports = {
        str(fold): benchmark_fold(items, fold, args.neighbor_limit)
        for fold in args.folds
    }
    names = sorted(next(iter(fold_reports.values())))
    average = {
        name: {
            metric: sum(report[name][metric] for report in fold_reports.values())
            / len(fold_reports)
            for metric in ("precision", "recall")
        }
        for name in names
    }
    report = {
        "split": "deterministic 5-fold by original_position % 5",
        "neighbor_limit": args.neighbor_limit,
        "folds": fold_reports,
        "average": average,
        "best_by_recall": max(average, key=lambda name: average[name]["recall"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
