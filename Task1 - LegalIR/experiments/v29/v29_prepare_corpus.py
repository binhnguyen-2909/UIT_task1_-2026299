"""Build the V29 hierarchical, token-bounded corpus manifest.

The corpus is unlabeled context only.  The manifest preserves legal structure
when it is present and uses bounded sentence/word fallback for malformed or
very long units.  No query, qrel, holdout label, or public label is read.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_SCRIPTS = SCRIPT_DIR.parents[1] / "scripts"
V27_EXPERIMENTS = SCRIPT_DIR.parent / "v27"
if str(REPO_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(REPO_SCRIPTS))
if str(V27_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(V27_EXPERIMENTS))

from legal_structure import (  # noqa: E402
    _ARTICLE_RE,
    _BULLET_RE,
    _CLAUSE_RE,
    _HIERARCHY_RE,
    _POINT_RE,
    _join_fragments,
    _logical_segments,
    clean_legal_lines,
)
from v27_common import sha256_file, write_json, write_jsonl  # noqa: E402


CONTEXT_RE = re.compile(r"(?:^|/)context_(?P<doc_id>[^/]+)\.json$", re.IGNORECASE)
SENTENCE_RE = re.compile(r"(?<=[.!?;:])\s+|(?<=。)\s*|\n+")
WORD_RE = re.compile(r"\S+")
PREFIX_RE = re.compile(r"^\s*(?P<value>\d{1,3})[.)]\s+")
POINT_PREFIX_RE = re.compile(r"^\s*(?P<value>[a-zđ])[.)]\s+", re.IGNORECASE)


def _compact(value: str) -> str:
    return " ".join(str(value or "").split())


def _token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])


def _split_long_words(tokenizer: Any, text: str, limit: int, overlap: int) -> list[str]:
    """Split a pathological sentence while measuring the real tokenizer length."""

    words = WORD_RE.findall(text)
    if not words:
        return []
    result: list[str] = []
    start = 0
    while start < len(words):
        # Binary search the largest fitting word window.  The previous
        # implementation tested every growing prefix, which is quadratic on
        # malformed documents containing one enormous paragraph.
        low, high = start + 1, len(words) + 1
        best = start
        while low < high:
            middle = (low + high) // 2
            candidate = " ".join(words[start:middle])
            if _token_count(tokenizer, candidate) <= limit:
                best = middle
                low = middle + 1
            else:
                high = middle
        if best == start:
            # A single unusually long token cannot be made shorter without
            # truncation.  Keep it as a visible diagnostic and let the caller
            # count it; this does not occur in the supplied corpus in practice.
            best = start + 1
        result.append(" ".join(words[start:best]))
        if best >= len(words):
            break
        next_start = max(start + 1, best - max(0, overlap))
        start = next_start
    return result


def _sentence_parts(text: str) -> list[str]:
    parts = [_compact(part) for part in SENTENCE_RE.split(text) if _compact(part)]
    return parts or [_compact(text)]


def _unit_labels(segment: str) -> tuple[str, str]:
    clause = ""
    point = ""
    match = PREFIX_RE.match(segment)
    if match:
        clause = match.group("value")
    match = POINT_PREFIX_RE.match(segment)
    if match:
        point = match.group("value")
    return clause, point


def _article_sections(lines: list[str]) -> Iterable[dict[str, str]]:
    """Yield preamble/article sections with the currently visible hierarchy."""

    positions = [index for index, line in enumerate(lines) if _ARTICLE_RE.match(line)]
    if not positions:
        yield {"unit_type": "fallback", "article_number": "", "article_heading": "", "hierarchy": "", "text": _join_fragments(lines)}
        return
    if positions[0] > 0:
        yield {"unit_type": "preamble", "article_number": "", "article_heading": "", "hierarchy": "", "text": _join_fragments(lines[: positions[0]])}
    current_hierarchy = ""
    hierarchy_at: dict[int, str] = {}
    for index, line in enumerate(lines):
        if _HIERARCHY_RE.match(line):
            current_hierarchy = line
        elif _ARTICLE_RE.match(line):
            hierarchy_at[index] = current_hierarchy
    for ordinal, start in enumerate(positions):
        end = positions[ordinal + 1] if ordinal + 1 < len(positions) else len(lines)
        article_lines = list(lines[start:end])
        # A new chapter/section between two articles belongs to the next one.
        if ordinal + 1 < len(positions):
            cut = next((i for i, line in enumerate(article_lines[1:], 1) if _HIERARCHY_RE.match(line)), None)
            if cut is not None:
                article_lines = article_lines[:cut]
        heading = article_lines[0]
        match = _ARTICLE_RE.match(heading)
        number = match.group("number") if match else ""
        body = _logical_segments(article_lines[1:])
        if not body:
            body = [heading]
        for segment in body:
            yield {
                "unit_type": "article",
                "article_number": number,
                "article_heading": heading,
                "hierarchy": hierarchy_at.get(start, ""),
                "text": segment,
            }


def _make_prefix(title: str, section: dict[str, str], clause: str, point: str) -> str:
    fields = ["[LAW]", title]
    if section["hierarchy"]:
        fields.extend(["[SECTION]", section["hierarchy"]])
    if section["article_heading"]:
        fields.extend(["[ARTICLE]", section["article_heading"]])
    if clause:
        fields.extend(["[CLAUSE]", clause])
    if point:
        fields.extend(["[POINT]", point])
    fields.append("[CONTENT]")
    return _compact(" ".join(fields))


def _build_chunks(doc_id: str, title: str, passage: str, tokenizer: Any, max_tokens: int, overlap_tokens: int) -> list[dict[str, Any]]:
    lines = clean_legal_lines(passage)
    rows: list[dict[str, Any]] = []
    if not lines:
        return rows
    descriptors: list[dict[str, Any]] = []
    for section_index, section in enumerate(_article_sections(lines)):
        text = section["text"]
        segments = _logical_segments(clean_legal_lines(text)) if section["unit_type"] == "fallback" else [text]
        if not segments:
            segments = [text]
        for segment_index, segment in enumerate(segments):
            clause, point = _unit_labels(segment)
            prefix = _make_prefix(title, section, clause, point)
            for sentence_index, sentence in enumerate(_sentence_parts(segment)):
                full = _compact(f"{prefix} {sentence}")
                descriptors.append({
                    "full": full,
                    "unit_type": "sentence",
                    "article_number": section["article_number"],
                    "article_heading": section["article_heading"],
                    "hierarchy": section["hierarchy"],
                    "clause_number": clause,
                    "point_label": point,
                    "sentence_index": sentence_index,
                })
    # Tokenize all sentence/semantic units in a Rust batch and let the
    # tokenizer produce exact 256-token overflow windows.  This is both more
    # faithful to the BGE input contract and substantially faster than making
    # repeated Python-side token-count calls for long units.
    texts = [row["full"] for row in descriptors]
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=True,
        max_length=max_tokens,
        stride=overlap_tokens,
        return_overflowing_tokens=True,
        padding=False,
    )
    mapping = encoded.get("overflow_to_sample_mapping", list(range(len(texts))))
    piece_counters: collections.Counter[int] = collections.Counter()
    for ids, source_index in zip(encoded["input_ids"], mapping):
        descriptor = descriptors[int(source_index)]
        piece = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        piece_index = piece_counters[int(source_index)]
        piece_counters[int(source_index)] += 1
        rows.append({
            "doc_id": str(doc_id),
            "chunk_id": f"{doc_id}:{len(rows):06d}",
            "ordinal": len(rows),
            "text": _compact(piece),
            "unit_type": descriptor["unit_type"] if piece_index == 0 and piece_counters[int(source_index)] == 1 else "bounded_clause",
            "article_number": descriptor["article_number"],
            "article_heading": descriptor["article_heading"],
            "hierarchy": descriptor["hierarchy"],
            "clause_number": descriptor["clause_number"],
            "point_label": descriptor["point_label"],
            "sentence_index": descriptor["sentence_index"],
            "piece_index": piece_index,
            "token_count": len(ids),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts-zip", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-passage-tokens", type=int, default=256)
    parser.add_argument("--overlap-tokens", type=int, default=32)
    args = parser.parse_args()
    if args.overlap_tokens >= args.max_passage_tokens:
        raise ValueError("overlap must be less than max passage tokens")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "hierarchical_manifest.jsonl"
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer.resolve()), local_files_only=True)
    started = time.perf_counter()
    doc_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    unit_counts: collections.Counter[str] = collections.Counter()
    hierarchy_docs = 0
    empty_docs = 0
    with zipfile.ZipFile(args.contexts_zip) as archive:
        members = []
        for name in archive.namelist():
            match = CONTEXT_RE.search(name)
            if match:
                members.append((str(match.group("doc_id")), name))
        members.sort(key=lambda item: (int(item[0]) if item[0].isdigit() else 10**18, item[0]))
        for doc_index, (doc_id, name) in enumerate(members):
            raw = json.loads(archive.read(name))
            title = _compact(raw.get("name", ""))
            passage = str(raw.get("passage", ""))
            chunks = _build_chunks(doc_id, title, passage, tokenizer, args.max_passage_tokens, args.overlap_tokens)
            if any(row["article_number"] for row in chunks):
                hierarchy_docs += 1
            if not chunks:
                empty_docs += 1
            for row in chunks:
                row["doc_index"] = doc_index
                row["title"] = title
                chunk_rows.append(row)
                unit_counts[row["unit_type"]] += 1
            doc_rows.append({"doc_id": doc_id, "title": title, "chunk_count": len(chunks), "empty_body": not bool(_compact(passage))})
    manifest_sha = write_jsonl(manifest_path, chunk_rows)
    token_counts = [int(row["token_count"]) for row in chunk_rows]
    stats = {
        "kind": "legalir_v29_hierarchical_chunk_stats",
        "schema_version": 1,
        "split": "corpus_unlabeled_for_clean_complement_retrieval",
        "holdout_opened": False,
        "public_labels_used": False,
        "contexts_zip": str(args.contexts_zip.resolve()),
        "contexts_zip_sha256": sha256_file(args.contexts_zip),
        "tokenizer": str(args.tokenizer.resolve()),
        "tokenizer_config_sha256": sha256_file(args.tokenizer / "tokenizer_config.json"),
        "docs": len(doc_rows),
        "chunks": len(chunk_rows),
        "empty_docs": empty_docs,
        "docs_with_detected_articles": hierarchy_docs,
        "unit_type_counts": dict(sorted(unit_counts.items())),
        "max_passage_tokens": args.max_passage_tokens,
        "overlap_tokens_for_long_units": args.overlap_tokens,
        "token_count": {
            "min": min(token_counts) if token_counts else 0,
            "max": max(token_counts) if token_counts else 0,
            "mean": sum(token_counts) / max(len(token_counts), 1),
            "p50": sorted(token_counts)[len(token_counts) // 2] if token_counts else 0,
            "p95": sorted(token_counts)[int(0.95 * (len(token_counts) - 1))] if token_counts else 0,
        },
        "manifest_sha256": manifest_sha,
        "elapsed_seconds": time.perf_counter() - started,
        "chunking_contract": {
            "hierarchy": "LAW -> SECTION/CHAPTER when visibly present -> ARTICLE -> CLAUSE -> POINT -> SENTENCE",
            "missing_levels": "not invented",
            "long_unit_policy": "sentence/clause split by tokenizer-bounded overflow windows with overlap_tokens token stride",
            "fixed_sliding_window_first": False,
        },
    }
    write_json(out / "chunk_stats.json", stats)
    print(json.dumps({"manifest": str(manifest_path.resolve()), "stats": stats}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
