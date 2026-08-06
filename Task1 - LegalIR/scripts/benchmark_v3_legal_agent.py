"""Leak-free benchmark of legal-domain lexical signals for LegalIR v3.

The production SQLite corpus is opened read-only.  Label-derived tables are
shadowed by TEMP FTS5 tables built only from the training side of a
deterministic fold, so exact-memory and nearest-question leakage are
impossible.  This file deliberately does not modify the production retriever.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

try:
    from .evaluate import official_metrics
    from .sqlite_retriever import SQLiteFTSRetriever
except ImportError:
    from evaluate import official_metrics
    from sqlite_retriever import SQLiteFTSRetriever


LEGAL_TYPE_RE = re.compile(
    r"\b(?:bộ\s+luật|luật|pháp\s+lệnh|nghị\s+định|thông\s+tư(?:\s+liên\s+tịch)?|"
    r"nghị\s+quyết|quyết\s+định|chỉ\s+thị|quy\s+chuẩn|tiêu\s+chuẩn|qcvn|tcvn)\b",
    re.IGNORECASE,
)
DOCUMENT_NUMBER_RE = re.compile(
    r"\b(?:\d{1,4}\s*/\s*(?:19|20)\d{2}\s*/\s*"
    r"(?=[0-9A-ZĐ-]*[A-ZĐ])[0-9A-ZĐ]+(?:-[0-9A-ZĐ]+)*|"
    r"(?:TCVN|QCVN)\s+\d+(?:-\d+)*(?::\d{4})?(?:/[A-ZĐ]+)?)\b",
    re.IGNORECASE,
)
STRUCTURE_RE = re.compile(
    r"\b(?:điều|(?<!tài\s)khoản|điểm|chương|mục|tiểu\s+mục|phụ\s+lục)\s+"
    r"(?:\d+(?:[./-]\d+)*[a-zđ]?|[ivxlcdm]+)\b",
    re.IGNORECASE,
)

# Question boilerplate is legal-domain language but weak document identity.
# It is excluded only from the auxiliary phrase/title branches, never from v2.
PHRASE_STOPWORDS = {
    "ai", "bao", "các", "cần", "cho", "có", "của", "được", "gì", "hay",
    "hiện", "hỏi", "không", "khi", "là", "làm", "một", "nào", "những",
    "như", "phải", "quy", "ra", "sao", "theo", "thế", "thì", "thực",
    "trong", "tại", "từ", "và", "về", "với",
}

SYNONYM_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("xử phạt", "mức phạt", "phạt tiền", "vi phạm hành chính"),
    ("thủ tục", "trình tự", "hồ sơ"),
    ("cấp phép", "giấy phép"),
    ("nghĩa vụ", "trách nhiệm"),
    ("thời hạn", "bao lâu"),
    ("điều kiện", "yêu cầu"),
    ("thu hồi", "hủy bỏ"),
    ("bồi thường", "thiệt hại"),
    ("chấm dứt", "hủy bỏ"),
)


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", str(text)).casefold().split())


def quote_fts(value: str) -> str:
    return f'"{value.replace(chr(34), chr(34) * 2)}"'


def unique(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def doc_ids(rows: Iterable[sqlite3.Row], limit: int = 40) -> List[str]:
    answer: List[str] = []
    seen: set[str] = set()
    for row in rows:
        doc_id = str(row["doc_id"])
        if doc_id in seen:
            continue
        seen.add(doc_id)
        answer.append(doc_id)
        if len(answer) >= limit:
            break
    return answer


def rrf(rankings: Iterable[Tuple[float, Sequence[str]]], k: int = 20) -> List[str]:
    scores: Dict[str, float] = defaultdict(float)
    for weight, ranking in rankings:
        for rank, doc_id in enumerate(ranking, 1):
            scores[str(doc_id)] += float(weight) / (k + rank)
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))


def score(
    predictions: Mapping[str, Sequence[str]], references: Mapping[str, Any]
) -> Dict[str, float]:
    return official_metrics(
        {qid: {"answer": list(ranking[:5])} for qid, ranking in predictions.items()},
        dict(references),
    )


def build_leak_free_temp_tables(
    connection: sqlite3.Connection,
    training: Sequence[Tuple[str, Dict[str, Any]]],
) -> None:
    """Shadow label tables so production methods see only the training split."""
    connection.executescript("""
        CREATE VIRTUAL TABLE temp.question_examples USING fts5(
            question_id UNINDEXED,
            answers UNINDEXED,
            question,
            tokenize='unicode61 remove_diacritics 0'
        );
        CREATE VIRTUAL TABLE temp.aliases USING fts5(
            doc_id UNINDEXED,
            alias_text,
            tokenize='unicode61 remove_diacritics 0'
        );
        CREATE TABLE temp.question_memory(question TEXT PRIMARY KEY, answers TEXT NOT NULL);
        CREATE VIRTUAL TABLE temp.legal_titles USING fts5(
            doc_id UNINDEXED,
            title,
            tokenize='unicode61 remove_diacritics 0'
        );
    """)
    aliases: Dict[str, List[str]] = defaultdict(list)
    connection.executemany(
        "INSERT INTO temp.question_examples(question_id, answers, question) VALUES (?, ?, ?)",
        (
            (qid, json.dumps(item["answer"]), item["question"])
            for qid, item in training
        ),
    )
    for _, item in training:
        for doc_id in map(str, item["answer"]):
            aliases[doc_id].append(item["question"])
    connection.executemany(
        "INSERT INTO temp.aliases(doc_id, alias_text) VALUES (?, ?)",
        ((doc_id, "\n".join(unique(questions))) for doc_id, questions in aliases.items()),
    )
    connection.execute(
        "INSERT INTO temp.legal_titles(doc_id, title) SELECT doc_id, title FROM main.document_meta"
    )
    connection.execute(
        "INSERT INTO temp.question_examples(question_examples) VALUES ('optimize')"
    )
    connection.execute("INSERT INTO temp.aliases(aliases) VALUES ('optimize')")
    connection.execute("INSERT INTO temp.legal_titles(legal_titles) VALUES ('optimize')")


def term_frequencies(
    connection: sqlite3.Connection, tokens: Sequence[str]
) -> Dict[str, int]:
    if not tokens:
        return {}
    placeholders = ",".join("?" for _ in tokens)
    rows = connection.execute(
        f"SELECT term, doc_freq FROM main.term_stats WHERE term IN ({placeholders})",
        tuple(tokens),
    )
    found = {str(row["term"]): int(row["doc_freq"]) for row in rows}
    return {token: found.get(token, 10**9) for token in tokens}


def original_tokens(retriever: SQLiteFTSRetriever, query: str) -> List[str]:
    return unique(retriever.tokenizer.tokenize(query))


def extract_legal_mentions(query: str) -> Dict[str, List[str]]:
    return {
        "types": unique(match.group(0) for match in LEGAL_TYPE_RE.finditer(query)),
        "numbers": unique(match.group(0) for match in DOCUMENT_NUMBER_RE.finditer(query)),
        "structures": unique(match.group(0) for match in STRUCTURE_RE.finditer(query)),
    }


def phrase_expression(
    retriever: SQLiteFTSRetriever,
    query: str,
    frequencies: Mapping[str, int],
) -> str:
    """Build bounded exact 2/3-token phrases containing a rare content token."""
    tokens = original_tokens(retriever, query)
    candidates: List[Tuple[int, int, str]] = []
    for width in (3, 2):
        for position in range(len(tokens) - width + 1):
            phrase = tokens[position:position + width]
            content = [token for token in phrase if token not in PHRASE_STOPWORDS]
            if len(content) < 2:
                continue
            rarity = min(frequencies.get(token, 10**9) for token in content)
            candidates.append((rarity, position, " ".join(phrase)))
    candidates.sort(key=lambda item: (item[0], -len(item[2]), item[1]))
    selected = unique(item[2] for item in candidates)[:10]
    return " OR ".join(quote_fts(phrase) for phrase in selected)


def legal_reference_expression(query: str) -> str:
    mentions = extract_legal_mentions(query)
    values = mentions["numbers"] + mentions["structures"]
    return " OR ".join(quote_fts(normalized(value)) for value in unique(values))


def title_expression(
    retriever: SQLiteFTSRetriever,
    query: str,
    frequencies: Mapping[str, int],
) -> str:
    mentions = extract_legal_mentions(query)
    tokens = original_tokens(retriever, query)
    content = [token for token in tokens if token not in PHRASE_STOPWORDS]
    positions = {token: index for index, token in enumerate(tokens)}
    rare = sorted(
        content,
        key=lambda token: (frequencies.get(token, 10**9), positions[token]),
    )[:8]
    # Exact document numbers/types receive an independent phrase clause while
    # rare content terms keep this useful for questions without citations.
    clauses = [quote_fts(normalized(value)) for value in mentions["numbers"]]
    clauses.extend(quote_fts(token) for token in rare)
    return " OR ".join(unique(clauses))


def expanded_tokens(retriever: SQLiteFTSRetriever, query: str) -> List[str]:
    lowered = normalized(query)
    additions: List[str] = []
    for group in SYNONYM_GROUPS:
        if any(trigger in lowered for trigger in group):
            additions.extend(group)
    return unique(retriever.tokenizer.tokenize(" ".join([query, *additions])))


def query_chunk_rank(
    connection: sqlite3.Connection,
    expression: str,
    title_weight: float = 3.0,
    row_limit: int = 400,
    time_budget_seconds: float = 8.0,
) -> List[str]:
    if not expression:
        return []
    deadline = time.perf_counter() + time_budget_seconds
    connection.set_progress_handler(
        lambda: int(time.perf_counter() >= deadline), 10_000
    )
    try:
        rows = connection.execute(
            """
            SELECT doc_id FROM main.chunks WHERE chunks MATCH ?
            ORDER BY bm25(chunks, 0.0, 0.0, ?, 1.0), doc_id LIMIT ?
            """,
            (expression, title_weight, row_limit),
        )
        return doc_ids(rows)
    except sqlite3.OperationalError:
        return []
    finally:
        connection.set_progress_handler(None, 0)


def query_title_rank(connection: sqlite3.Connection, expression: str) -> List[str]:
    if not expression:
        return []
    try:
        rows = connection.execute(
            """
            SELECT doc_id FROM temp.legal_titles WHERE legal_titles MATCH ?
            ORDER BY bm25(legal_titles, 0.0, 1.0), doc_id LIMIT 80
            """,
            (expression,),
        )
        return doc_ids(rows)
    except sqlite3.OperationalError:
        return []


def recall_at(
    predictions: Mapping[str, Sequence[str]],
    references: Mapping[str, Any],
    cutoff: int,
) -> float:
    values: List[float] = []
    for qid, item in references.items():
        relevant = {str(value) for value in item["answer"]}
        predicted = set(predictions.get(qid, ())[:cutoff])
        values.append(len(relevant & predicted) / len(relevant) if relevant else 0.0)
    return sum(values) / len(values) if values else 0.0


def benchmark_titles(
    train_path: Path,
    db_path: Path,
    fold: int,
    limit: int,
) -> Dict[str, Any]:
    """Cheap full-fold sweep using only the 8,532 title/link rows."""
    data = json.loads(train_path.read_text(encoding="utf-8"))
    items = [(str(qid), item) for qid, item in data.items()]
    holdout = [record for index, record in enumerate(items) if index % 5 == fold]
    if limit:
        holdout = holdout[:limit]

    retriever = SQLiteFTSRetriever.load(db_path)
    connection = retriever.connection
    connection.execute("""
        CREATE VIRTUAL TABLE temp.legal_titles USING fts5(
            doc_id UNINDEXED, title,
            tokenize='unicode61 remove_diacritics 0'
        )
    """)
    connection.execute(
        "INSERT INTO temp.legal_titles(doc_id, title) SELECT doc_id, title FROM main.document_meta"
    )
    connection.execute("INSERT INTO temp.legal_titles(legal_titles) VALUES ('optimize')")

    rankings: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    references: Dict[str, Any] = {}
    reference_qids: List[str] = []
    started = time.perf_counter()
    try:
        for ordinal, (qid, item) in enumerate(holdout, 1):
            query = str(item["question"])
            references[qid] = {"answer": [str(value) for value in item["answer"]]}
            mentions = extract_legal_mentions(query)
            if mentions["numbers"] or mentions["structures"]:
                reference_qids.append(qid)
            tokens = original_tokens(retriever, query)
            frequencies = term_frequencies(connection, tokens)
            positions = {token: index for index, token in enumerate(tokens)}
            rare = sorted(
                (token for token in tokens if token not in PHRASE_STOPWORDS
                 and frequencies.get(token, 10**9) < 10**9),
                key=lambda token: (frequencies[token], positions[token]),
            )[:8]
            title_or = query_title_rank(
                connection, " OR ".join(quote_fts(token) for token in rare)
            )
            title_and2 = query_title_rank(
                connection, " AND ".join(quote_fts(token) for token in rare[:2])
            )
            title_phrase = query_title_rank(
                connection, phrase_expression(retriever, query, frequencies)
            )
            title_reference = query_title_rank(
                connection, legal_reference_expression(query)
            )
            rankings["title_or"][qid] = title_or
            rankings["title_and2"][qid] = title_and2
            rankings["title_phrase"][qid] = title_phrase
            rankings["title_reference"][qid] = title_reference
            rankings["title_rrf_or_phrase"][qid] = rrf(
                ((3.0, title_or), (2.0, title_phrase)), k=20
            )
            rankings["title_rrf_all"][qid] = rrf(
                ((3.0, title_or), (2.0, title_phrase), (1.0, title_and2),
                 (4.0, title_reference)), k=20
            )
            if ordinal % 250 == 0:
                print(f"title benchmark {ordinal}/{len(holdout)}", flush=True)
    finally:
        retriever.close()

    metrics = {name: score(values, references) for name, values in rankings.items()}
    candidate_recall = {
        name: recall_at(values, references, 40) for name, values in rankings.items()
    }
    reference_metrics: Dict[str, Dict[str, float]] = {}
    if reference_qids:
        selected_refs = {qid: references[qid] for qid in reference_qids}
        reference_metrics = {
            name: score({qid: values[qid] for qid in reference_qids}, selected_refs)
            for name, values in rankings.items()
        }
    return {
        "protocol": {
            "split": f"original_position % 5 == {fold}",
            "questions": len(holdout),
            "uses_label_derived_training_signals": False,
            "corpus_field": "document_meta.title (name + URL slug)",
            "passage_chunks_read": False,
        },
        "metrics_at_5": metrics,
        "candidate_recall_at_40": candidate_recall,
        "explicit_reference_questions": len(reference_qids),
        "explicit_reference_metrics_at_5": reference_metrics,
        "runtime_seconds": time.perf_counter() - started,
    }
def v2_rankings(
    retriever: SQLiteFTSRetriever, query: str
) -> Tuple[List[str], List[str], List[str]]:
    """Return reconstructed v2, its focused branch and its OR branch."""
    _, search_terms = retriever._match_expression(query)
    focused = retriever._chunk_candidates(search_terms) if search_terms else []
    focused = retriever._fuse_alias_candidates(search_terms, focused)
    focused = retriever._fuse_question_neighbors(query, focused)
    focused = retriever._fuse_character_neighbors(query, focused)
    broad = retriever._or_bm25_candidates(search_terms)
    fused = retriever._fuse_or_bm25(focused, broad)
    return (
        [str(item["doc_id"]) for item in fused],
        [str(item["doc_id"]) for item in focused],
        [str(item["doc_id"]) for item in broad],
    )


def benchmark(
    train_path: Path,
    db_path: Path,
    fold: int,
    limit: int,
    cache_path: Path | None = None,
    resume: bool = False,
) -> Dict[str, Any]:
    data = json.loads(train_path.read_text(encoding="utf-8"))
    items = [(str(qid), item) for qid, item in data.items()]
    training = [record for index, record in enumerate(items) if index % 5 != fold]
    holdout_all = [record for index, record in enumerate(items) if index % 5 == fold]
    holdout = holdout_all[:limit] if limit else holdout_all

    retriever = SQLiteFTSRetriever.load(db_path)
    connection = retriever.connection
    build_leak_free_temp_tables(connection, training)
    retriever._question_ngram_cache = None

    signature = {
        "version": 1,
        "fold": fold,
        "holdout_questions": len(holdout),
        "train_size": train_path.stat().st_size,
        "db_size": db_path.stat().st_size,
    }
    rankings: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    references: Dict[str, Any] = {}
    coverage: Counter[str] = Counter()
    if resume and cache_path and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("signature") != signature:
            raise ValueError("Legal benchmark cache signature does not match this run")
        rankings.update({
            name: {qid: list(values) for qid, values in rows.items()}
            for name, rows in cached.get("rankings", {}).items()
        })
        references.update(cached.get("references", {}))
        coverage.update(cached.get("coverage", {}))
    initial_completed = sum(
        qid in references for qid in rankings.get("v2_reconstructed", {})
    )

    def save_cache() -> None:
        if cache_path is None:
            return
        payload = {
            "signature": signature,
            "references": references,
            "coverage": dict(coverage),
            "rankings": dict(rankings),
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(cache_path)

    started = time.perf_counter()
    try:
        for ordinal, (qid, item) in enumerate(holdout, 1):
            if qid in rankings.get("v2_reconstructed", {}) and qid in references:
                continue
            query = str(item["question"])
            query_started = time.perf_counter()
            mentions = extract_legal_mentions(query)

            raw_tokens = original_tokens(retriever, query)
            frequencies = term_frequencies(connection, raw_tokens)
            v2, focused, broad = v2_rankings(retriever, query)
            phrase = query_chunk_rank(
                connection, phrase_expression(retriever, query, frequencies),
                title_weight=6.0, row_limit=500,
            )
            title = query_title_rank(
                connection, title_expression(retriever, query, frequencies)
            )
            legal_ref = query_chunk_rank(
                connection, legal_reference_expression(query),
                title_weight=12.0, row_limit=800,
            )
            expanded = expanded_tokens(retriever, query)
            expanded_freq = term_frequencies(connection, expanded)
            expanded = sorted(
                (token for token in expanded if expanded_freq[token] < 10**9),
                key=lambda token: (expanded_freq[token], expanded.index(token)),
            )[:16]
            expanded_or = query_chunk_rank(
                connection,
                " OR ".join(quote_fts(token) for token in expanded),
                title_weight=3.0,
                row_limit=500,
            )

            for name, ranking in (
                ("v2_reconstructed", v2),
                ("v2_focused", focused),
                ("v2_or", broad),
                ("legal_phrase", phrase),
                ("legal_title", title),
                ("legal_reference", legal_ref),
                ("legal_synonym_or", expanded_or),
            ):
                rankings[name][qid] = ranking

            # Low-dimensional fixed grid.  Final selection is made on the first
            # half and checked independently on the second half below.
            for signal_name, signal in (
                ("phrase", phrase), ("title", title), ("reference", legal_ref),
                ("synonym", expanded_or),
            ):
                for weight in (1.0, 2.0, 3.0, 5.0):
                    name = f"v2_plus_{signal_name}_w{int(weight)}"
                    rankings[name][qid] = rrf(((10.0, v2), (weight, signal)), k=20)
            for title_weight, phrase_weight in ((1.0, 2.0), (2.0, 2.0), (2.0, 3.0)):
                name = f"v2_plus_title{int(title_weight)}_phrase{int(phrase_weight)}"
                rankings[name][qid] = rrf(
                    ((10.0, v2), (title_weight, title), (phrase_weight, phrase)), k=20
                )

            references[qid] = {"answer": [str(value) for value in item["answer"]]}
            for name, values in mentions.items():
                coverage[name] += bool(values)
            completed = len(references)
            query_seconds = time.perf_counter() - query_started
            if completed % 5 == 0 or query_seconds >= 30.0:
                print(
                    f"legal query completed={completed}, last={query_seconds:.1f}s",
                    flush=True,
                )
            if completed % 10 == 0 or completed == len(holdout):
                save_cache()
                elapsed = time.perf_counter() - started
                print(
                    f"legal benchmark {completed}/{len(holdout)} "
                    f"({elapsed:.1f}s this process)",
                    flush=True,
                )
    finally:
        retriever.close()
        save_cache()

    qids = [qid for qid, _ in holdout]
    midpoint = max(1, len(qids) // 2)
    partitions = {
        "tuning_first_half": qids[:midpoint],
        "validation_second_half": qids[midpoint:],
        "all": qids,
    }
    metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for partition, selected_qids in partitions.items():
        selected_refs = {qid: references[qid] for qid in selected_qids}
        metrics[partition] = {
            name: score({qid: values[qid] for qid in selected_qids}, selected_refs)
            for name, values in rankings.items()
        }

    tune = metrics["tuning_first_half"]
    candidates = [name for name in tune if name.startswith("v2_plus_")]
    selected = max(
        candidates,
        key=lambda name: (tune[name]["recall"], tune[name]["precision"], name),
    )
    return {
        "protocol": {
            "split": f"original_position % 5 == {fold}",
            "training_excludes_entire_holdout_fold": True,
            "evaluated_first_n_holdout": len(holdout),
            "configuration_selected_on": "first half only",
            "independent_check": "second half",
            "corpus_index_opened_read_only": True,
            "network_or_model": False,
        },
        "coverage": dict(coverage),
        "metrics": metrics,
        "selected_on_tuning": selected,
        "selected_tuning_metrics": tune[selected],
        "selected_validation_metrics": metrics["validation_second_half"].get(selected, {}),
        "selected_all_metrics": metrics["all"][selected],
        "runtime_seconds": time.perf_counter() - started,
        "resumed_from_questions": initial_completed,
    }


def dataset_stats(train_path: Path) -> Dict[str, Any]:
    data = json.loads(train_path.read_text(encoding="utf-8"))
    counts: Counter[str] = Counter()
    answers: Counter[int] = Counter()
    examples: Dict[str, List[str]] = defaultdict(list)
    for item in data.values():
        query = str(item["question"])
        mentions = extract_legal_mentions(query)
        for name, values in mentions.items():
            if values:
                counts[name] += 1
                if len(examples[name]) < 3:
                    examples[name].append(query)
        answers[len(item["answer"])] += 1
    return {
        "questions": len(data),
        "legal_mention_coverage": dict(counts),
        "answer_count_distribution": {str(key): value for key, value in sorted(answers.items())},
        "examples": dict(examples),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--db", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--light-only", action="store_true")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    report: Dict[str, Any] = {"dataset": dataset_stats(args.train)}
    if args.light_only:
        if args.db is None:
            parser.error("--db is required with --light-only")
        report["lightweight_title"] = benchmark_titles(
            args.train, args.db, args.fold, args.limit
        )
    elif not args.stats_only:
        if args.db is None:
            parser.error("--db is required unless --stats-only is used")
        cache = args.cache or args.output.with_suffix(".cache.json")
        report["benchmark"] = benchmark(
            args.train, args.db, args.fold, args.limit, cache, args.resume
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Windows runners may expose a cp1252 stdout even though artifacts are
    # UTF-8.  ASCII escaping keeps CLI progress/report output portable.
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
