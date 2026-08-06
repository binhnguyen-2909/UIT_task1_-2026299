"""Leak-free document-level FTS5 benchmark for LegalIR recall-v3 research.

This script deliberately uses a standalone database and the cached fold-0 v2
rankings.  It never mutates the production chunk index or retriever code.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

try:
    from .bm25_retriever import VietnameseTokenizer
    from .evaluate import official_metrics
    from .sqlite_retriever import display_title, iter_contexts
except ImportError:
    from bm25_retriever import VietnameseTokenizer
    from evaluate import official_metrics
    from sqlite_retriever import display_title, iter_contexts


def quote_token(token: str) -> str:
    return f'"{token.replace(chr(34), chr(34) * 2)}"'


def or_expression(tokens: Sequence[str]) -> str:
    return " OR ".join(quote_token(token) for token in tokens)


def rrf(rankings: Sequence[Tuple[float, Sequence[str]]], k: int) -> List[str]:
    scores: Dict[str, float] = defaultdict(float)
    for weight, ranking in rankings:
        for rank, doc_id in enumerate(ranking, 1):
            scores[str(doc_id)] += float(weight) / (k + rank)
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))


def score(
    predictions: Mapping[str, Sequence[str]], references: Mapping[str, Any]
) -> Dict[str, float]:
    return official_metrics(
        {qid: {"answer": list(docs[:5])} for qid, docs in predictions.items()},
        references,
    )


def candidate_recall(
    predictions: Mapping[str, Sequence[str]],
    references: Mapping[str, Any],
    cutoff: int = 40,
) -> float:
    values: List[float] = []
    for qid, target in references.items():
        truth = set(map(str, target["answer"]))
        found = set(map(str, predictions.get(qid, ())[:cutoff]))
        values.append(len(truth & found) / len(truth) if truth else 0.0)
    return sum(values) / len(values) if values else 0.0


def build_database(corpus: Path, database: Path) -> Dict[str, Any]:
    """Create exactly one FTS row per official document."""
    if database.exists():
        connection = sqlite3.connect(database)
        try:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'complete'"
            ).fetchone()
            count = connection.execute("SELECT count(*) FROM documents").fetchone()[0]
            if row and row[0] == "1" and count > 0:
                return {"reused": True, "documents": int(count)}
        except sqlite3.Error:
            pass
        finally:
            connection.close()

    database.parent.mkdir(parents=True, exist_ok=True)
    building = database.with_suffix(database.suffix + ".building")
    if building.exists():
        building.unlink()
    connection = sqlite3.connect(building)
    started = time.perf_counter()
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=MEMORY;
            PRAGMA cache_size=-262144;
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE VIRTUAL TABLE documents USING fts5(
                doc_id UNINDEXED,
                title,
                passage,
                tokenize='unicode61 remove_diacritics 0',
                prefix='2 3'
            );
            """
        )
        inserted = 0
        for record in iter_contexts(corpus):
            passage = record.get("passage")
            passage = passage if isinstance(passage, str) else ""
            title = display_title(record)
            # Preserve every official ID.  The BTC corpus contains 20 records
            # with an empty passage and no title; production uses the same
            # deterministic surrogate so these IDs remain valid candidates.
            if not passage.strip():
                passage = title or f"văn bản {record['id']}"
            connection.execute(
                "INSERT INTO documents(doc_id, title, passage) VALUES (?, ?, ?)",
                (str(record["id"]), title, passage),
            )
            inserted += 1
            if inserted % 250 == 0:
                connection.commit()
                print(f"build {inserted} documents", flush=True)
        connection.execute("INSERT INTO metadata VALUES ('complete', '1')")
        connection.execute(
            "INSERT INTO metadata VALUES ('documents', ?)", (str(inserted),)
        )
        connection.execute(
            "CREATE VIRTUAL TABLE documents_vocab USING fts5vocab(documents, 'row')"
        )
        connection.commit()
        connection.execute("PRAGMA optimize")
        connection.commit()
    finally:
        connection.close()
    building.replace(database)
    return {
        "reused": False,
        "documents": inserted,
        "build_seconds": time.perf_counter() - started,
    }


class DocumentFTS:
    def __init__(self, database: Path):
        self.connection = sqlite3.connect(database)
        self.connection.row_factory = sqlite3.Row
        self.tokenizer = VietnameseTokenizer(bigrams=False)

    def close(self) -> None:
        self.connection.close()

    def tokens(self, query: str) -> List[str]:
        return list(dict.fromkeys(self.tokenizer.tokenize(query)))

    def rare_tokens(self, tokens: Sequence[str], limit: int) -> List[str]:
        if not tokens:
            return []
        placeholders = ",".join("?" for _ in tokens)
        rows = self.connection.execute(
            f"SELECT term, doc FROM documents_vocab WHERE term IN ({placeholders})",
            tuple(tokens),
        )
        frequencies = {str(row["term"]): int(row["doc"]) for row in rows}
        # Missing terms cannot retrieve anything; keep original order as a stable
        # tie-break and prefer terms occurring in fewer documents.
        positions = {token: index for index, token in enumerate(tokens)}
        present = [token for token in tokens if token in frequencies]
        present.sort(key=lambda token: (frequencies[token], positions[token]))
        return present[:limit]

    def search(
        self,
        tokens: Sequence[str],
        limit: int = 80,
        title_weight: float = 3.0,
        passage_weight: float = 1.0,
    ) -> List[str]:
        expression = or_expression(tokens)
        if not expression:
            return []
        rows = self.connection.execute(
            """
            SELECT doc_id
            FROM documents
            WHERE documents MATCH ?
            ORDER BY bm25(documents, 0.0, ?, ?), doc_id
            LIMIT ?
            """,
            (expression, title_weight, passage_weight, limit),
        )
        return [str(row["doc_id"]) for row in rows]

    def phrase_search(self, tokens: Sequence[str], limit: int = 80) -> List[str]:
        if len(tokens) < 2:
            return []
        pairs = list(dict.fromkeys(zip(tokens, tokens[1:])))
        # Exact bigrams are a precision signal.  Cap the expression to avoid
        # turning long questions into an expensive disjunction.
        expression = " OR ".join(
            f'"{a.replace(chr(34), chr(34) * 2)} {b.replace(chr(34), chr(34) * 2)}"'
            for a, b in pairs[:16]
        )
        rows = self.connection.execute(
            """
            SELECT doc_id
            FROM documents
            WHERE documents MATCH ?
            ORDER BY bm25(documents, 0.0, 3.0, 1.0), doc_id
            LIMIT ?
            """,
            (expression, limit),
        )
        return [str(row["doc_id"]) for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--v2-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    build_info = build_database(args.corpus, args.database)
    raw = json.loads(args.train.read_text(encoding="utf-8"))
    items = [(str(qid), item) for qid, item in raw.items()]
    holdout = [record for index, record in enumerate(items) if index % 5 == args.fold][
        : args.limit
    ]
    cached_report = json.loads(args.v2_cache.read_text(encoding="utf-8"))
    cached = cached_report["rankings"]
    expected = [qid for qid, _ in holdout]
    if expected != list(cached)[: len(expected)]:
        raise ValueError("v2 cache does not match requested ordered holdout")

    references: Dict[str, Any] = {}
    variants: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    per_query: Dict[str, Any] = {}
    engine = DocumentFTS(args.database)
    started = time.perf_counter()
    try:
        for position, (qid, item) in enumerate(holdout, 1):
            references[qid] = {"answer": list(map(str, item["answer"]))}
            tokens = engine.tokens(str(item["question"]))
            rankings = {
                "doc_all_w1": engine.search(tokens, title_weight=1.0),
                "doc_all_w3": engine.search(tokens, title_weight=3.0),
                "doc_all_w8": engine.search(tokens, title_weight=8.0),
                "doc_first12_w3": engine.search(tokens[:12], title_weight=3.0),
                "doc_rare12_w1": engine.search(engine.rare_tokens(tokens, 12), title_weight=1.0),
                "doc_rare12_w3": engine.search(engine.rare_tokens(tokens, 12), title_weight=3.0),
                "doc_rare12_w8": engine.search(engine.rare_tokens(tokens, 12), title_weight=8.0),
                "doc_phrase": engine.phrase_search(tokens),
            }
            rankings["doc_rare12_phrase"] = rrf(
                [(1.0, rankings["doc_rare12_w3"]), (0.35, rankings["doc_phrase"])],
                20,
            )
            v2 = list(map(str, cached[qid]["v2"]))
            variants["v2"][qid] = v2
            for name, ranking in rankings.items():
                variants[name][qid] = ranking
                for weight in (0.15, 0.25, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0):
                    for k in (0, 10, 20, 40, 60):
                        fusion_name = f"v2_plus_{name}_w{weight:g}_k{k}"
                        variants[fusion_name][qid] = rrf(
                            [(1.0, v2), (weight, ranking)], k
                        )
            per_query[qid] = {
                "truth": references[qid]["answer"],
                "tokens": tokens,
                "rare12": engine.rare_tokens(tokens, 12),
                "rankings": rankings,
            }
            if position % 10 == 0 or position == len(holdout):
                print(f"evaluate {position}/{len(holdout)}", flush=True)
    finally:
        engine.close()

    metrics = {name: score(value, references) for name, value in variants.items()}
    candidate = {
        name: candidate_recall(value, references, 40)
        for name, value in variants.items()
    }
    union_coverage: Dict[str, float] = {}
    for name in (item for item in variants if item.startswith("doc_")):
        union_predictions = {
            qid: list(dict.fromkeys([
                *variants["v2"][qid][:40],
                *variants[name][qid][:40],
            ]))
            for qid in references
        }
        # Do not truncate here: this diagnoses whether the document-level
        # component contributes missing truth anywhere in its own top 40.
        values = []
        for qid, target in references.items():
            truth = set(map(str, target["answer"]))
            values.append(
                len(truth & set(union_predictions[qid])) / len(truth) if truth else 0.0
            )
        union_coverage[name] = sum(values) / len(values) if values else 0.0
    ordered = sorted(
        metrics,
        key=lambda name: (
            -metrics[name]["recall"],
            -metrics[name]["precision"],
            -candidate[name],
            name,
        ),
    )
    direct_names = [name for name in variants if name.startswith("doc_")]
    best_direct = max(
        direct_names,
        key=lambda name: (
            metrics[name]["recall"], metrics[name]["precision"], candidate[name]
        ),
    )
    report = {
        "fold": args.fold,
        "questions": len(holdout),
        "database": str(args.database),
        "build": build_info,
        "evaluation_seconds": time.perf_counter() - started,
        "baseline_v2": {
            "metrics": metrics["v2"],
            "candidate_recall_at_40": candidate["v2"],
        },
        "best": ordered[0],
        "best_metrics": metrics[ordered[0]],
        "best_candidate_recall_at_40": candidate[ordered[0]],
        "best_direct": best_direct,
        "best_direct_metrics": metrics[best_direct],
        "best_direct_candidate_recall_at_40": candidate[best_direct],
        "union_recall_v2_plus_direct_at_40_each": dict(sorted(
            union_coverage.items(), key=lambda item: (-item[1], item[0])
        )),
        "metrics": {name: metrics[name] for name in ordered},
        "candidate_recall_at_40": {
            name: candidate[name]
            for name in sorted(candidate, key=lambda item: (-candidate[item], item))
        },
        "rankings": per_query,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "best": report["best"],
                "best_metrics": report["best_metrics"],
                "best_direct": report["best_direct"],
                "best_direct_metrics": report["best_direct_metrics"],
                "best_direct_candidate_recall_at_40": report[
                    "best_direct_candidate_recall_at_40"
                ],
                "top": [(name, metrics[name], candidate[name]) for name in ordered[:10]],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
