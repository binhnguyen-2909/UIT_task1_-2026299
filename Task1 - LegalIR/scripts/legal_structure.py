"""Structure-aware cleaning and chunking for Vietnamese legal documents.

The official corpus contains HTML-to-text line wrapping, blank lines between
visual fragments, and documents ranging from a few characters to several
million characters.  Fixed-width character chunks frequently split an
``Điều`` from its ``Khoản``/``Điểm``.  This module keeps those legal units
together and only falls back to bounded character chunks for unstructured or
exceptionally long text.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple


_SPACE_RE = re.compile(r"[ \t\f\v\u00a0]+")
_SEPARATOR_RE = re.compile(r"^[\-_*–—=.•·\s]{3,}$")
_ARTICLE_RE = re.compile(
    r"^Điều\s+(?P<number>\d+[a-zđ]?)\s*(?:[.:]|\b)", re.IGNORECASE
)
_HIERARCHY_RE = re.compile(
    r"^(?:Phần|Chương|Mục|Tiểu mục)\s+(?:[IVXLCDM]+|\d+[a-zđ]?)\s*[.:]?",
    re.IGNORECASE,
)
_CLAUSE_RE = re.compile(r"^\d{1,3}[.)]\s+")
_POINT_RE = re.compile(r"^[a-zđ][.)]\s+", re.IGNORECASE)
_BULLET_RE = re.compile(r"^(?:[-–—+•]|\(\d{1,3}\)|\([a-zđ]\))\s+")


@dataclass(frozen=True)
class LegalChunk:
    """One clean index unit with auditable legal-structure metadata."""

    text: str
    unit_type: str
    article_number: str = ""
    hierarchy: str = ""


def clean_legal_lines(text: str) -> List[str]:
    """Normalize Unicode/spacing while preserving meaningful line starts."""

    normalized = unicodedata.normalize("NFC", str(text or ""))
    normalized = (
        normalized.replace("\u200b", "")
        .replace("\ufeff", "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )
    lines: List[str] = []
    for raw in normalized.split("\n"):
        line = _SPACE_RE.sub(" ", raw).strip()
        if not line or _SEPARATOR_RE.fullmatch(line):
            continue
        lines.append(line)
    return lines


def _join_fragments(lines: Sequence[str]) -> str:
    """Undo visual hard wrapping without changing the lexical content."""

    return " ".join(line.strip() for line in lines if line.strip()).strip()


def _fixed_chunks(text: str, chunk_size: int, overlap: int) -> Iterator[str]:
    text = str(text).strip()
    if not text:
        return
    start = 0
    minimum_boundary = max(1, int(chunk_size * 0.6))
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = max(
                text.rfind(". ", start + minimum_boundary, end),
                text.rfind("; ", start + minimum_boundary, end),
                text.rfind(" ", start + minimum_boundary, end),
            )
            if boundary > start:
                end = boundary + (1 if text[boundary] in ".;" else 0)
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)


def _logical_segments(lines: Sequence[str]) -> List[str]:
    """Group visual lines into clause/point-sized semantic segments."""

    segments: List[List[str]] = []
    for line in lines:
        starts_unit = bool(
            _CLAUSE_RE.match(line) or _POINT_RE.match(line) or _BULLET_RE.match(line)
        )
        if starts_unit or not segments:
            segments.append([line])
        else:
            segments[-1].append(line)
    return [_join_fragments(segment) for segment in segments if segment]


def _pack_article(
    article_lines: Sequence[str],
    hierarchy: str,
    chunk_size: int,
    overlap: int,
) -> Iterator[LegalChunk]:
    match = _ARTICLE_RE.match(article_lines[0])
    article_number = match.group("number") if match else ""
    heading = article_lines[0]
    prefix = _join_fragments([hierarchy, heading]) if hierarchy else heading
    body_segments = _logical_segments(article_lines[1:])
    if not body_segments:
        yield LegalChunk(prefix, "article", article_number, hierarchy)
        return

    packed: List[str] = []

    def emit(parts: Sequence[str]) -> Optional[LegalChunk]:
        body = _join_fragments(parts)
        text = _join_fragments([prefix, body]) if body else prefix
        if not text:
            return None
        return LegalChunk(text, "article", article_number, hierarchy)

    for segment in body_segments:
        proposed = _join_fragments([prefix, *packed, segment])
        if len(proposed) <= chunk_size:
            packed.append(segment)
            continue
        if packed:
            chunk = emit(packed)
            if chunk is not None:
                yield chunk
            packed = []
        if len(_join_fragments([prefix, segment])) <= chunk_size:
            packed = [segment]
            continue
        allowance = max(256, chunk_size - len(prefix) - 1)
        local_overlap = min(overlap, max(0, allowance // 5))
        for piece in _fixed_chunks(segment, allowance, local_overlap):
            chunk = emit([piece])
            if chunk is not None:
                yield chunk
    if packed:
        chunk = emit(packed)
        if chunk is not None:
            yield chunk


def iter_legal_chunks(text: str, chunk_size: int, overlap: int) -> Iterator[LegalChunk]:
    """Yield clean legal units, preserving Điều/Khoản/Điểm relationships.

    Documents without detectable ``Điều`` headings retain full coverage via
    the fixed-width fallback.  Preambles are indexed separately because the
    promulgating decision, scope, document number, and effective date often
    answer competition questions directly.
    """

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")
    lines = clean_legal_lines(text)
    if not lines:
        return
    article_positions = [index for index, line in enumerate(lines) if _ARTICLE_RE.match(line)]
    if not article_positions:
        cleaned = _join_fragments(lines)
        for piece in _fixed_chunks(cleaned, chunk_size, overlap):
            yield LegalChunk(piece, "fallback")
        return

    first_article = article_positions[0]
    preamble = _join_fragments(lines[:first_article])
    for piece in _fixed_chunks(preamble, chunk_size, overlap):
        yield LegalChunk(piece, "preamble")

    hierarchy_by_article: dict[int, str] = {}
    current_hierarchy = ""
    for index, line in enumerate(lines):
        if _HIERARCHY_RE.match(line):
            current_hierarchy = line
        elif _ARTICLE_RE.match(line):
            hierarchy_by_article[index] = current_hierarchy

    for article_index, start in enumerate(article_positions):
        end = (
            article_positions[article_index + 1]
            if article_index + 1 < len(article_positions)
            else len(lines)
        )
        article_lines = list(lines[start:end])
        if article_index + 1 < len(article_positions):
            next_hierarchy = next(
                (
                    offset
                    for offset, line in enumerate(article_lines[1:], start=1)
                    if _HIERARCHY_RE.match(line)
                ),
                None,
            )
            if next_hierarchy is not None:
                article_lines = article_lines[:next_hierarchy]
        yield from _pack_article(
            article_lines,
            hierarchy_by_article.get(start, ""),
            chunk_size,
            overlap,
        )


def iter_index_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
    strategy: str = "legal",
) -> Iterator[LegalChunk]:
    """Dispatch fixed or structure-aware chunking through one stable API."""

    if strategy == "legal":
        yield from iter_legal_chunks(text, chunk_size, overlap)
        return
    if strategy == "fixed":
        normalized = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
        for piece in _fixed_chunks(normalized, chunk_size, overlap):
            yield LegalChunk(piece, "fixed")
        return
    raise ValueError(f"Unknown chunk strategy: {strategy}")
