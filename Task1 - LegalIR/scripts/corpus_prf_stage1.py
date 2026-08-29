"""Corpus-only pseudo-relevance feedback and legal citation completion.

The script consumes a completed Stage-1 audit, treats only its highest-ranked
corpus documents as pseudo-relevant, and retrieves one complementary ranking
from discriminative terms shared by those documents.  It never reads labels
while generating candidates.  Labels are optional and are used only after all
rankings are frozen to select an RRF configuration on a development split.

The selected output preserves the ``queries``/``stage1_candidates`` contract
used by ``batch_stage2_rerank.py`` so a promoted configuration can be tested
without recomputing the expensive base Stage-1 branches.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

TOKEN_RE = re.compile(r"[0-9]+(?:[./-][0-9a-zđ]+)*|[^\W_]+", re.IGNORECASE | re.UNICODE)
DOC_NUMBER_RE = re.compile(
    r"\b\d{1,4}\s*/\s*\d{4}\s*/\s*[0-9a-zđ-]{2,24}\b",
    re.IGNORECASE | re.UNICODE,
)
STOPWORDS = {
    "và", "của", "là", "các", "một", "những", "đó", "này", "tại",
    "từ", "với", "cho", "về", "do", "bởi", "thì", "mà", "cũng", "đang",
    "theo", "được", "phải", "có", "không", "trong", "trên", "khi", "nếu",
    "quy", "định", "điều", "khoản", "điểm", "luật", "nghị", "thông", "tư",
    "văn", "bản", "như", "thế", "nào", "gì", "bao", "gồm", "thuộc",
}


def parse_number_list(value: str, cast: type) -> list[Any]:
    output = [cast(part.strip()) for part in value.split(",") if part.strip()]
    if not output:
        raise argparse.ArgumentTypeError("list must not be empty")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-audit", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        help="Optional labels for development/holdout evaluation; omit for fixed public apply",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--selected-audit", type=Path, required=True)
    parser.add_argument(
        "--trace-from",
        type=Path,
        help="Reuse frozen corpus-only terms/document numbers from an earlier report",
    )
    parser.add_argument("--seed-docs", type=int, default=4)
    parser.add_argument("--seed-passages", type=int, default=1)
    parser.add_argument("--terms", type=int, default=4)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--max-document-frequency-ratio", type=float, default=0.08)
    parser.add_argument("--anchor-query-term", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--result-limit", type=int, default=200)
    parser.add_argument("--chunk-scan-limit", type=int, default=500)
    parser.add_argument("--prf-weights", default="0,0.1,0.2,0.35,0.5,0.75,1.0")
    parser.add_argument("--citation-weights", default="0,0.2,0.5,1.0")
    parser.add_argument("--rrf-k", default="10,20,40")
    parser.add_argument(
        "--lock-top",
        default="0,5,10",
        help="Comma-separated base prefix lengths protected from PRF reordering",
    )
    parser.add_argument("--top5-tolerance", type=float, default=0.001)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def open_readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        path.resolve().as_uri() + "?mode=ro", uri=True, check_same_thread=False
    )
    connection.row_factory = sqlite3.Row
    for pragma in (
        "PRAGMA cache_size=-262144",
        "PRAGMA temp_store=MEMORY",
        "PRAGMA mmap_size=2147483648",
    ):
        try:
            connection.execute(pragma)
        except sqlite3.DatabaseError:
            pass
    return connection


def tokenize(text: str) -> list[str]:
    return [value.casefold() for value in TOKEN_RE.findall(text or "")]


def quote_fts(term: str) -> str:
    return f'"{term.replace(chr(34), chr(34) * 2)}"'


def load_inputs(
    audit_path: Path, reference_path: Path | None
) -> tuple[dict[str, Any], list[dict[str, Any]], list[list[str]], dict[str, set[str]]]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    queries = audit.get("queries")
    candidates = audit.get("stage1_candidates")
    if not isinstance(queries, list) or not isinstance(candidates, list):
        raise ValueError("Stage-1 audit lacks queries/stage1_candidates")
    if len(queries) != len(candidates):
        raise ValueError("Stage-1 queries and candidates differ in length")
    gold: dict[str, set[str]] = {}
    if reference_path is not None:
        references = json.loads(reference_path.read_text(encoding="utf-8"))
        for item in queries:
            qid = str(item["query_id"])
            if qid not in references or not isinstance(references[qid].get("answer"), list):
                raise ValueError(f"Reference does not cover {qid}")
            gold[qid] = {str(value) for value in references[qid]["answer"]}
    return audit, queries, [[str(value) for value in row] for row in candidates], gold


def corpus_statistics(connection: sqlite3.Connection) -> tuple[dict[str, int], int]:
    frequencies = {
        str(row["term"]).casefold(): int(row["doc_freq"])
        for row in connection.execute("SELECT term, doc_freq FROM term_stats")
    }
    chunks = int(connection.execute("SELECT count(*) FROM chunks").fetchone()[0])
    return frequencies, chunks


def rare_query_terms(query: str, frequencies: dict[str, int]) -> list[str]:
    terms = list(dict.fromkeys(tokenize(query)))
    return sorted(
        (term for term in terms if frequencies.get(term, 0) > 0),
        key=lambda term: (frequencies[term], terms.index(term)),
    )


def seed_passages(
    connection: sqlite3.Connection,
    query: str,
    doc_ids: Sequence[str],
    frequencies: dict[str, int],
    passages_per_doc: int,
    scan_limit: int,
) -> list[list[str]]:
    unique_ids = list(dict.fromkeys(str(value) for value in doc_ids))
    if not unique_ids:
        return []
    placeholders = ",".join("?" for _ in unique_ids)
    rows = connection.execute(
        f"""
        SELECT doc_id, title, passage
        FROM chunks
        WHERE doc_id IN ({placeholders})
        ORDER BY doc_id, rowid
        LIMIT ?
        """,
        [*unique_ids, int(scan_limit)],
    )
    query_terms = set(tokenize(query))
    by_doc: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for row in rows:
        text = "\n".join(value for value in (row["title"], row["passage"]) if value)
        text_terms = set(tokenize(text))
        score = sum(
            math.log1p(1.0 + 10000.0 / max(1, frequencies.get(term, 10000)))
            for term in query_terms & text_terms
        )
        by_doc[str(row["doc_id"])].append((score, text))
    return [
        [text for _, text in sorted(by_doc.get(doc_id, ()), key=lambda item: -item[0])[:passages_per_doc]]
        for doc_id in unique_ids
    ]


def expansion_terms(
    query: str,
    seed_texts: Sequence[Sequence[str]],
    frequencies: dict[str, int],
    total_chunks: int,
    limit: int,
    min_support: int,
    max_frequency_ratio: float,
) -> list[str]:
    query_terms = set(tokenize(query))
    support: Counter[str] = Counter()
    for passages in seed_texts:
        document_terms: set[str] = set()
        for passage in passages:
            document_terms.update(tokenize(passage))
        support.update(document_terms)
    scored: list[tuple[float, str]] = []
    max_frequency = max(2, int(total_chunks * max_frequency_ratio))
    for term, count in support.items():
        frequency = frequencies.get(term, 0)
        if (
            count < min_support
            or term in query_terms
            or term in STOPWORDS
            or len(term) < 3
            or frequency < 2
            or frequency > max_frequency
        ):
            continue
        idf = math.log((total_chunks + 1.0) / (frequency + 1.0))
        scored.append((count * idf, term))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [term for _, term in scored[:limit]]


def shared_document_numbers(seed_texts: Sequence[Sequence[str]], min_support: int = 2) -> list[str]:
    support: Counter[str] = Counter()
    for passages in seed_texts:
        values = {
            re.sub(r"\s+", "", match.group(0)).casefold()
            for passage in passages
            for match in DOC_NUMBER_RE.finditer(passage)
        }
        support.update(values)
    return [value for value, count in support.most_common() if count >= min_support][:4]


def search_chunks(
    connection: sqlite3.Connection,
    expression: str,
    method: str,
    chunk_limit: int,
    document_limit: int,
) -> list[str]:
    if not expression:
        return []
    try:
        rows = connection.execute(
            """
            SELECT doc_id
            FROM chunks
            WHERE chunks MATCH ?
            ORDER BY bm25(chunks, 0.0, 0.0, 3.0, 1.0), doc_id, chunk_id
            LIMIT ?
            """,
            (expression, int(chunk_limit)),
        )
    except sqlite3.OperationalError:
        return []
    del method  # retained in the API for trace readability/call-site symmetry
    output: list[str] = []
    seen: set[str] = set()
    for row in rows:
        value = str(row["doc_id"])
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
        if len(output) >= document_limit:
            break
    return output


def prf_ranking(
    connection: sqlite3.Connection,
    query: str,
    terms: Sequence[str],
    frequencies: dict[str, int],
    anchor_query_term: bool,
    chunk_limit: int,
    document_limit: int,
) -> list[str]:
    if not terms:
        return []
    expansion = " OR ".join(quote_fts(term) for term in terms)
    expression = f"({expansion})"
    if anchor_query_term:
        anchors = rare_query_terms(query, frequencies)
        if anchors:
            expression = f"{quote_fts(anchors[0])} AND ({expansion})"
    return search_chunks(connection, expression, "corpus_prf", chunk_limit, document_limit)


def citation_ranking(
    connection: sqlite3.Connection,
    document_numbers: Sequence[str],
    chunk_limit: int,
    document_limit: int,
) -> list[str]:
    if not document_numbers:
        return []
    expression = " OR ".join(quote_fts(value) for value in document_numbers)
    return search_chunks(connection, expression, "citation_completion", chunk_limit, document_limit)


def fuse_rankings(
    base: Sequence[str],
    prf: Sequence[str],
    citations: Sequence[str],
    prf_weight: float,
    citation_weight: float,
    rrf_k: int,
    lock_top: int = 0,
) -> list[str]:
    locked = list(dict.fromkeys(str(value) for value in base))[:lock_top]
    locked_ids = set(locked)
    scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, int] = {}
    serial = 0
    for weight, ranking in ((1.0, base), (prf_weight, prf), (citation_weight, citations)):
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
            scores[doc_id] += weight / (rrf_k + rank)
            if doc_id not in first_seen:
                first_seen[doc_id] = serial
                serial += 1
    tail = sorted(scores, key=lambda value: (-scores[value], first_seen[value], value))
    return [*locked, *tail]


def evaluate(
    rankings: Sequence[Sequence[str]],
    queries: Sequence[dict[str, Any]],
    gold: dict[str, set[str]],
) -> dict[str, Any]:
    cutoffs = (1, 5, 20, 35, 50, 80, 100, 200)
    values: dict[int, list[float]] = {cutoff: [] for cutoff in cutoffs}
    precision5: list[float] = []
    for ranking, item in zip(rankings, queries, strict=True):
        relevant = gold[str(item["query_id"])]
        for cutoff in cutoffs:
            values[cutoff].append(len(relevant & set(ranking[:cutoff])) / max(1, len(relevant)))
        precision5.append(len(relevant & set(ranking[:5])) / 5.0)
    return {
        "precision_at_5": statistics.fmean(precision5),
        "recall": {f"at_{cutoff}": statistics.fmean(values[cutoff]) for cutoff in cutoffs},
    }


def changed_top5(left: Sequence[Sequence[str]], right: Sequence[Sequence[str]]) -> dict[str, Any]:
    return {
        "queries": sum(a[:5] != b[:5] for a, b in zip(left, right, strict=True)),
        "mean_overlap": statistics.fmean(
            len(set(a[:5]) & set(b[:5])) / 5.0 for a, b in zip(left, right, strict=True)
        ),
    }


def selected_payload(
    source: dict[str, Any],
    queries: Sequence[dict[str, Any]],
    rankings: Sequence[Sequence[str]],
    metrics: dict[str, Any] | None,
    configuration: dict[str, Any],
    source_path: Path,
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
                "method": "corpus_prf_rrf",
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
        key: value for key, value in source.items()
        if key not in {"queries", "stage1_candidates", "metrics", "candidate_oracle"}
    }
    payload.update({
        "kind": "legalir_corpus_prf_stage1",
        "source_stage1_audit": str(source_path.resolve()),
        "prf_configuration": configuration,
        "queries": output_queries,
        "stage1_candidates": [list(row) for row in rankings],
    })
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
    if (
        args.seed_docs <= 0
        or args.seed_passages <= 0
        or args.terms <= 0
        or args.min_support <= 0
        or not 0 < args.max_document_frequency_ratio < 1
        or args.result_limit <= 0
        or args.chunk_scan_limit <= 0
        or args.progress_every <= 0
    ):
        raise ValueError("invalid positive PRF limits/frequency ratio")
    started = time.perf_counter()
    source, queries, base_rankings, gold = load_inputs(args.stage1_audit, args.reference)
    connection = open_readonly(args.index_dir / "legalir.sqlite3")
    frequencies, total_chunks = corpus_statistics(connection)
    frozen_traces: dict[str, dict[str, Any]] = {}
    if args.trace_from is not None:
        prior_report = json.loads(args.trace_from.read_text(encoding="utf-8"))
        frozen_traces = {
            str(row["question_id"]): row for row in prior_report.get("traces", [])
        }
        if len(frozen_traces) != len(queries):
            raise ValueError("--trace-from does not cover the current Stage-1 query set")

    prf_rankings: list[list[str]] = []
    citation_rankings: list[list[str]] = []
    traces: list[dict[str, Any]] = []
    for position, (item, base) in enumerate(zip(queries, base_rankings, strict=True), 1):
        query = str(item["query"])
        seeds = base[: args.seed_docs]
        frozen = frozen_traces.get(str(item["query_id"]))
        if frozen is None:
            passages = seed_passages(
                connection,
                query,
                seeds,
                frequencies,
                args.seed_passages,
                args.chunk_scan_limit,
            )
            terms = expansion_terms(
                query,
                passages,
                frequencies,
                total_chunks,
                args.terms,
                args.min_support,
                args.max_document_frequency_ratio,
            )
            numbers = shared_document_numbers(passages, min_support=args.min_support)
        else:
            terms = [str(value) for value in frozen.get("terms", [])]
            numbers = [str(value) for value in frozen.get("document_numbers", [])]
        prf = prf_ranking(
            connection,
            query,
            terms,
            frequencies,
            args.anchor_query_term,
            args.chunk_scan_limit,
            args.result_limit,
        )
        citations = citation_ranking(
            connection, numbers, args.chunk_scan_limit, args.result_limit
        )
        prf_rankings.append(prf)
        citation_rankings.append(citations)
        traces.append({
            "question_id": str(item["query_id"]),
            "seed_documents": seeds,
            "terms": terms,
            "document_numbers": numbers,
            "prf_candidates": len(prf),
            "citation_candidates": len(citations),
        })
        if position % args.progress_every == 0 or position == len(queries):
            print(f"Corpus PRF progress: {position}/{len(queries)}", flush=True)
    connection.close()

    prf_weights = parse_number_list(args.prf_weights, float)
    citation_weights = parse_number_list(args.citation_weights, float)
    rrf_values = parse_number_list(args.rrf_k, int)
    lock_values = parse_number_list(args.lock_top, int)
    if any(value < 0 for value in (*prf_weights, *citation_weights, *rrf_values, *lock_values)):
        raise ValueError("RRF weights/k must be non-negative")

    baseline_metrics = evaluate(base_rankings, queries, gold) if gold else None
    sweep: list[dict[str, Any]] = []
    ranking_cache: dict[tuple[float, float, int, int], list[list[str]]] = {}
    for prf_weight in prf_weights:
        for citation_weight in citation_weights:
            for rrf_k in rrf_values:
                for lock_top in lock_values:
                    configuration = (
                        float(prf_weight), float(citation_weight), int(rrf_k), int(lock_top)
                    )
                    rankings = [
                        fuse_rankings(base, prf, citation, *configuration)
                        for base, prf, citation in zip(
                            base_rankings, prf_rankings, citation_rankings, strict=True
                        )
                    ]
                    ranking_cache[configuration] = rankings
                    result = evaluate(rankings, queries, gold) if gold else None
                    sweep.append({
                        "prf_weight": configuration[0],
                        "citation_weight": configuration[1],
                        "rrf_k": configuration[2],
                        "lock_top": configuration[3],
                        "metrics": result,
                        "versus_baseline": changed_top5(base_rankings, rankings),
                    })

    if gold:
        assert baseline_metrics is not None
        floor = baseline_metrics["recall"]["at_5"] - args.top5_tolerance
        eligible = [row for row in sweep if row["metrics"]["recall"]["at_5"] >= floor]
        if not eligible:
            eligible = sweep
        best = max(
            eligible,
            key=lambda row: (
                row["metrics"]["recall"]["at_35"],
                row["metrics"]["recall"]["at_5"],
                row["metrics"]["recall"]["at_80"],
                -row["prf_weight"] - row["citation_weight"],
                -row["rrf_k"],
                row["lock_top"],
            ),
        )
    else:
        if len(sweep) != 1:
            raise ValueError(
                "Public apply without --reference requires singleton PRF/citation/k/lock grids"
            )
        best = sweep[0]
    key = (
        float(best["prf_weight"]),
        float(best["citation_weight"]),
        int(best["rrf_k"]),
        int(best["lock_top"]),
    )
    selected = ranking_cache[key]
    structural = {
        "seed_docs": args.seed_docs,
        "seed_passages": args.seed_passages,
        "terms": args.terms,
        "min_support": args.min_support,
        "max_document_frequency_ratio": args.max_document_frequency_ratio,
        "anchor_query_term": args.anchor_query_term,
        "result_limit": args.result_limit,
    }
    selected_config = {
        **structural,
        **{name: best[name] for name in ("prf_weight", "citation_weight", "rrf_k", "lock_top")},
    }
    payload = selected_payload(
        source,
        queries,
        selected,
        best["metrics"],
        selected_config,
        args.stage1_audit,
    )
    args.selected_audit.parent.mkdir(parents=True, exist_ok=True)
    args.selected_audit.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = {
        "kind": "legalir_corpus_prf_sweep",
        "source_stage1_audit": str(args.stage1_audit.resolve()),
        "index": str((args.index_dir / "legalir.sqlite3").resolve()),
        "reference": str(args.reference.resolve()) if args.reference else None,
        "questions": len(queries),
        "structural_configuration": structural,
        "baseline_metrics": baseline_metrics,
        "selected": best,
        "selected_configuration": selected_config,
        "trace_summary": {
            "queries_with_prf": sum(bool(row["prf_candidates"]) for row in traces),
            "queries_with_citation_completion": sum(bool(row["citation_candidates"]) for row in traces),
            "mean_prf_candidates": statistics.fmean(row["prf_candidates"] for row in traces),
            "mean_citation_candidates": statistics.fmean(row["citation_candidates"] for row in traces),
        },
        "sweep": sweep,
        "traces": traces,
        "elapsed_seconds": time.perf_counter() - started,
        "selected_audit": str(args.selected_audit.resolve()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "questions": len(queries),
        "baseline": baseline_metrics,
        "selected_configuration": selected_config,
        "selected_metrics": best["metrics"],
        "trace_summary": report["trace_summary"],
        "elapsed_seconds": report["elapsed_seconds"],
        "report": str(args.report.resolve()),
        "selected_audit": str(args.selected_audit.resolve()),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
