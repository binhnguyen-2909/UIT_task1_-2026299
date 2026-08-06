"""Benchmark deeper OR-BM25 passage pools without modifying production.

The benchmark reuses the focused rankings cached by ``benchmark_v3_fusion``
and opens the 1.27 GB SQLite index in read-only mode.  A first ``failures``
pass only queries questions for which recall-v2 missed at least one relevant
document in its top five.  The resulting checkpoint can then be resumed in
``full`` mode, so completed expensive FTS queries are not repeated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

try:
    from .bm25_retriever import VietnameseTokenizer
    from .evaluate import official_metrics
except ImportError:
    from bm25_retriever import VietnameseTokenizer
    from evaluate import official_metrics


VERSION = 1
BASELINE_RECALL = 0.7916666666666667


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def quote_or(tokens: Sequence[str]) -> str:
    return " OR ".join(
        f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens
    )


def select_terms(
    connection: sqlite3.Connection,
    tokenizer: VietnameseTokenizer,
    query: str,
) -> List[str]:
    """Reproduce SQLiteFTSRetriever._match_expression term selection."""
    tokens = list(dict.fromkeys(tokenizer.tokenize(query)))
    if not tokens:
        return []
    placeholders = ",".join("?" for _ in tokens)
    frequencies = {
        str(row["term"]): int(row["doc_freq"])
        for row in connection.execute(
            f"SELECT term, doc_freq FROM term_stats WHERE term IN ({placeholders})",
            tokens,
        )
    }
    positions = {token: position for position, token in enumerate(tokens)}
    return sorted(
        frequencies,
        key=lambda token: (frequencies[token], positions[token]),
    )


def parse_term_counts(values: Sequence[str]) -> List[int | None]:
    parsed: List[int | None] = []
    for value in values:
        normalized = str(value).strip().casefold()
        count = None if normalized == "all" else int(normalized)
        if count is not None and count <= 0:
            raise ValueError("term counts must be positive integers or 'all'")
        if count not in parsed:
            parsed.append(count)
    return parsed


def term_label(count: int | None) -> str:
    return "all" if count is None else str(count)


def configuration_name(term_count: int | None, rows: int, docs: int) -> str:
    return f"terms{term_label(term_count)}_rows{rows}_docs{docs}"


def query_rows(
    connection: sqlite3.Connection,
    tokens: Sequence[str],
    row_limit: int,
) -> List[str]:
    expression = quote_or(tokens)
    if not expression:
        return []
    rows = connection.execute(
        """
        SELECT doc_id
        FROM chunks WHERE chunks MATCH ?
        ORDER BY bm25(chunks, 0.0, 0.0, 3.0, 1.0), doc_id
        LIMIT ?
        """,
        (expression, row_limit),
    )
    return [str(row["doc_id"]) for row in rows]


def collapse_rows(rows: Sequence[str], row_limit: int, doc_limit: int) -> List[str]:
    unique: List[str] = []
    seen = set()
    for doc_id in rows[:row_limit]:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        unique.append(doc_id)
        if len(unique) >= doc_limit:
            break
    return unique


def rrf(
    rankings: Iterable[Tuple[float, Sequence[str]]],
    k: int,
) -> List[str]:
    scores: Dict[str, float] = defaultdict(float)
    for weight, ranking in rankings:
        for rank, doc_id in enumerate(ranking, 1):
            scores[str(doc_id)] += weight / (k + rank)
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))


def metrics(
    predictions: Mapping[str, Sequence[str]],
    references: Mapping[str, Mapping[str, List[str]]],
) -> Dict[str, float]:
    return official_metrics(
        {qid: {"answer": list(ranking[:5])} for qid, ranking in predictions.items()},
        dict(references),
    )


def candidate_recall(
    rankings: Mapping[str, Sequence[str]],
    references: Mapping[str, Mapping[str, List[str]]],
    focused: Mapping[str, Sequence[str]] | None = None,
) -> float:
    values: List[float] = []
    for qid, reference in references.items():
        candidates = set(map(str, rankings[qid]))
        if focused is not None:
            candidates.update(map(str, focused[qid]))
        truth = set(map(str, reference["answer"]))
        values.append(len(candidates & truth) / len(truth) if truth else 0.0)
    return sum(values) / len(values) if values else 0.0


def checkpoint_signature(
    train: Path,
    components: Path,
    index: Path,
    row_limits: Sequence[int],
    doc_limits: Sequence[int],
    term_counts: Sequence[int | None],
) -> str:
    # Hashing the 1.27 GB database would add minutes of pure I/O.  Its exact
    # resolved path, size and nanosecond mtime are sufficient to reject a
    # changed local index while the two small semantic inputs use SHA-256.
    stat = index.stat()
    payload = {
        "version": VERSION,
        "train_sha256": file_sha256(train),
        "components_sha256": file_sha256(components),
        "index": str(index.resolve()),
        "index_size": stat.st_size,
        "index_mtime_ns": stat.st_mtime_ns,
        "row_limits": list(row_limits),
        "doc_limits": list(doc_limits),
        "term_counts": [term_label(value) for value in term_counts],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def load_checkpoint(path: Path, signature: str, resume: bool) -> Dict[str, Any]:
    if resume:
        if not path.exists():
            raise FileNotFoundError(f"Cannot resume: checkpoint not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("signature") != signature:
            raise ValueError("Checkpoint signature does not match benchmark inputs")
        return payload
    return {
        "version": VERSION,
        "signature": signature,
        "completed": {},
        "elapsed_seconds": 0.0,
    }


def failure_ids(rankings: Mapping[str, Mapping[str, Any]]) -> Tuple[List[str], int]:
    qids: List[str] = []
    missing_count = 0
    for qid, cached in rankings.items():
        missing = set(map(str, cached["truth"])) - set(map(str, cached["v2"][:5]))
        if missing:
            qids.append(qid)
            missing_count += len(missing)
    return qids, missing_count


def build_report(
    mode: str,
    cached: Mapping[str, Mapping[str, Any]],
    completed: Mapping[str, Mapping[str, Any]],
    term_counts: Sequence[int | None],
    row_limits: Sequence[int],
    doc_limits: Sequence[int],
    weights: Sequence[float],
    rrf_ks: Sequence[int],
    elapsed_seconds: float,
) -> Dict[str, Any]:
    failed_qids, missing_top5 = failure_ids(cached)
    missing_candidate = 0
    per_failure: Dict[str, Any] = {}
    recovery = {
        configuration_name(term_count, rows, docs): 0
        for term_count in term_counts
        for rows in row_limits
        for docs in doc_limits
    }
    for qid in failed_qids:
        entry = cached[qid]
        missed = sorted(
            set(map(str, entry["truth"])) - set(map(str, entry["v2"][:5]))
        )
        baseline_pool = set(map(str, entry["focused"])) | set(
            map(str, entry["or_bm25"])
        )
        absent = [doc_id for doc_id in missed if doc_id not in baseline_pool]
        missing_candidate += len(absent)
        recovered_by: Dict[str, List[str]] = {}
        if qid in completed:
            for name, ranking in completed[qid]["rankings"].items():
                recovered = [doc_id for doc_id in absent if doc_id in ranking]
                recovered_by[name] = recovered
                recovery[name] += len(recovered)
        per_failure[qid] = {
            "truth": list(map(str, entry["truth"])),
            "v2_top5": list(map(str, entry["v2"][:5])),
            "missed_top5": missed,
            "absent_from_focused_or_top40": absent,
            "recovered_by": recovered_by,
        }

    report: Dict[str, Any] = {
        "version": VERSION,
        "mode": mode,
        "baseline": {
            "name": "recall_v2",
            "holdout_recall": BASELINE_RECALL,
            "top5_missing_relevant": missing_top5,
            "affected_questions": len(failed_qids),
            "missing_from_focused_or_candidate_top40": missing_candidate,
        },
        "settings": {
            "term_counts": [term_label(value) for value in term_counts],
            "row_limits": list(row_limits),
            "collapsed_doc_limits": list(doc_limits),
            "focused_weight": 1.0,
            "or_weights": list(weights),
            "rrf_k": list(rrf_ks),
        },
        "queried_questions": len(completed),
        "elapsed_seconds": elapsed_seconds,
        "failure_candidate_recovery_counts": recovery,
        "failure_diagnostic": per_failure,
    }

    if mode != "full" or len(completed) != len(cached):
        report["note"] = (
            "Failure diagnostic only; do not use it to choose production weights. "
            "Run --mode full --resume before comparing top-5 recall."
        )
        return report

    references = {
        qid: {"answer": list(map(str, entry["truth"]))}
        for qid, entry in cached.items()
    }
    focused = {
        qid: list(map(str, entry["focused"])) for qid, entry in cached.items()
    }
    variants: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    for qid, entry in completed.items():
        for depth_name, or_ranking in entry["rankings"].items():
            variants[f"{depth_name}_or_only"][qid] = list(or_ranking)
            for weight in weights:
                for k in rrf_ks:
                    name = f"{depth_name}_orw{weight:g}_k{k}"
                    variants[name][qid] = rrf(
                        [(1.0, focused[qid]), (weight, or_ranking)], k
                    )

    scored = {name: metrics(value, references) for name, value in variants.items()}
    ordered_metrics = dict(
        sorted(
            scored.items(),
            key=lambda item: (
                -item[1]["recall"],
                -item[1]["precision"],
                item[0],
            ),
        )
    )
    candidate_metrics: Dict[str, Any] = {}
    for term_count in term_counts:
        for rows in row_limits:
            for docs in doc_limits:
                depth_name = configuration_name(term_count, rows, docs)
                ranking = {
                    qid: list(completed[qid]["rankings"][depth_name])
                    for qid in cached
                }
                candidate_metrics[depth_name] = {
                    "or_candidate_recall": candidate_recall(ranking, references),
                    "focused_union_candidate_recall": candidate_recall(
                        ranking, references, focused
                    ),
                }
    best_name = next(iter(ordered_metrics))
    report.update(
        {
            "metrics": ordered_metrics,
            "candidate_metrics": candidate_metrics,
            "best": best_name,
            "best_metrics": ordered_metrics[best_name],
            "recall_gain_vs_v2": ordered_metrics[best_name]["recall"]
            - BASELINE_RECALL,
        }
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=("failures", "full"), default="failures")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--term-counts",
        nargs="+",
        default=["12", "16", "20", "24", "all"],
        help="rare-term counts; use 'all' for every in-corpus query term",
    )
    parser.add_argument("--row-limits", type=int, nargs="+", default=[400])
    parser.add_argument("--doc-limits", type=int, nargs="+", default=[40])
    parser.add_argument(
        "--or-weights",
        type=float,
        nargs="+",
        default=[1.5, 2.0, 7.0 / 3.0, 2.5, 3.0],
    )
    parser.add_argument("--rrf-k", type=int, nargs="+", default=[10, 20, 30, 40])
    parser.add_argument("--checkpoint-every", type=int, default=1)
    args = parser.parse_args()

    row_limits = sorted(set(args.row_limits))
    doc_limits = sorted(set(args.doc_limits))
    term_counts = parse_term_counts(args.term_counts)
    if min(row_limits) <= 0 or min(doc_limits) <= 0:
        raise ValueError("row and document limits must be positive")
    component_report = json.loads(args.components.read_text(encoding="utf-8"))
    cached: Dict[str, Dict[str, Any]] = component_report["rankings"]
    train = json.loads(args.train.read_text(encoding="utf-8"))
    missing_questions = [qid for qid in cached if qid not in train]
    if missing_questions:
        raise ValueError(f"Questions missing from train.json: {missing_questions[:3]}")

    signature = checkpoint_signature(
        args.train,
        args.components,
        args.index,
        row_limits,
        doc_limits,
        term_counts,
    )
    checkpoint = load_checkpoint(args.checkpoint, signature, args.resume)
    completed: Dict[str, Dict[str, Any]] = checkpoint["completed"]
    failed_qids, _ = failure_ids(cached)
    selected = failed_qids if args.mode == "failures" else list(cached)
    pending = [qid for qid in selected if qid not in completed]
    # Production SQLiteFTSRetriever uses the tokenizer default (unigrams plus
    # synthetic ``a__b`` bigrams); matching it is essential because rare-term
    # selection is performed after tokenization.
    tokenizer = VietnameseTokenizer()
    database_uri = args.index.resolve().as_uri() + "?mode=ro"
    started = time.perf_counter()
    connection = sqlite3.connect(database_uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        for position, qid in enumerate(pending, 1):
            question = str(train[qid]["question"])
            all_terms = select_terms(connection, tokenizer, question)
            query_started = time.perf_counter()
            rankings: Dict[str, List[str]] = {}
            query_seconds: Dict[str, float] = {}
            row_cache: Dict[Tuple[str, ...], List[str]] = {}
            for term_count in term_counts:
                terms = all_terms if term_count is None else all_terms[:term_count]
                term_key = tuple(terms)
                term_name = term_label(term_count)
                can_reuse_cached_v2 = (
                    term_count == 12
                    and max(row_limits) == 400
                    and max(doc_limits) <= 40
                )
                term_started = time.perf_counter()
                if can_reuse_cached_v2:
                    rows = list(map(str, cached[qid]["or_bm25"]))
                elif term_key in row_cache:
                    rows = row_cache[term_key]
                else:
                    rows = query_rows(connection, terms, max(row_limits))
                    row_cache[term_key] = rows
                query_seconds[term_name] = time.perf_counter() - term_started
                for row_limit in row_limits:
                    for doc_limit in doc_limits:
                        name = configuration_name(term_count, row_limit, doc_limit)
                        # Cached v2 is already the unique top-40 ranking.  It
                        # is safe only for the default rows400/docs<=40 case.
                        if can_reuse_cached_v2:
                            rankings[name] = rows[:doc_limit]
                        else:
                            rankings[name] = collapse_rows(
                                rows, row_limit, doc_limit
                            )
            total_query_seconds = time.perf_counter() - query_started
            completed[qid] = {
                "all_terms": all_terms,
                "term_query_seconds": query_seconds,
                "query_seconds": total_query_seconds,
                "rankings": rankings,
            }
            checkpoint["elapsed_seconds"] = float(
                checkpoint.get("elapsed_seconds", 0.0)
            ) + total_query_seconds
            if position % args.checkpoint_every == 0 or position == len(pending):
                atomic_json(args.checkpoint, checkpoint)
            print(
                f"progress {len(selected) - len(pending) + position}/{len(selected)} "
                f"qid={qid} query={completed[qid]['query_seconds']:.2f}s",
                flush=True,
            )
    finally:
        connection.close()

    elapsed = float(checkpoint.get("elapsed_seconds", 0.0))
    # Include tiny Python/report overhead in the current process separately;
    # expensive SQL time remains resumable and exact in the checkpoint.
    report = build_report(
        args.mode,
        cached,
        completed,
        term_counts,
        row_limits,
        doc_limits,
        args.or_weights,
        args.rrf_k,
        elapsed,
    )
    report["current_process_seconds"] = time.perf_counter() - started
    atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "queried_questions": len(completed),
                "best": report.get("best"),
                "best_metrics": report.get("best_metrics"),
                "recall_gain_vs_v2": report.get("recall_gain_vs_v2"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
