"""Algorithm-first, one-slot rescue around the locked v11h LegalIR anchor.

This module deliberately keeps v11h as the default answer.  It only proposes
an out-of-head candidate from the fixed c35 pool when typed legal constraints,
an exhaustive evidence-set view, and at least two independent signals agree.
The implementation is deterministic and has no trainable parameters.

The command line runner is intended for offline fit/holdout/public evaluation:
fit labels may select a conservative threshold, while the locked holdout is
only opened for reporting.  It can also be used without a fit split to inspect
the replacement/error oracle and to apply the default policy.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sqlite3
import statistics
import time
import zipfile
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from .evaluate import official_metrics
    from .legal_structure import iter_legal_chunks
    from .v11_exact_heading_rescue import normalize
except ImportError:  # pragma: no cover - exercised by direct script execution
    from evaluate import official_metrics
    from legal_structure import iter_legal_chunks
    from v11_exact_heading_rescue import normalize


NUMBER_RE = re.compile(r"\b\d+(?:[.,/]\d+)*(?:\s*(?:%|nam|thang|ngay|gio|phut|trieu|ty|dong))?\b")
ARTICLE_RE = re.compile(r"\b(?:dieu|article)\s+(\d+[a-z]?)\b", re.IGNORECASE)
CLAUSE_RE = re.compile(r"(?:^|\s)(\d{1,3})[.)]\s+")
POINT_RE = re.compile(r"(?:^|\s)([a-zđ])[.)]\s+", re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-z0-9]+")

# Function words are useful for parsing spans but weak as evidence.  They are
# excluded from coverage statistics while numbers and article references stay.
CONTENT_STOPWORDS = frozenset(
    (
        "a ai an anh bao bao_gom bao_nhieu bang ben bi bo co cua cung da dang de den "
        "du duoc hay ho hon khi khong la lai ma moi mot nay neu nhu nhung o phai qua "
        "rang sao se su tai theo thi trong tu ve voi va vao vi y"
    ).split()
)

MODALITY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("prohibition", r"\b(?:khong duoc|cam|nghiem cam|khong co quyen)\b"),
    ("obligation", r"\b(?:phai|bat buoc|co nghia vu|co trach nhiem)\b"),
    ("permission_question", r"\b(?:co duoc|duoc phep|co the)\b.*\b(?:khong|hay khong)\b"),
    ("permission", r"\b(?:duoc phep|co the|co quyen|duoc)\b"),
)
DOC_MODALITY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("prohibition", r"\b(?:khong duoc|cam|nghiem cam|khong co quyen)\b"),
    ("obligation", r"\b(?:phai|bat buoc|co nghia vu|co trach nhiem)\b"),
    ("permission", r"\b(?:duoc phep|co the|co quyen|duoc)\b"),
)

MODEL_BUDGET = {
    "v11h_embedding_parameters": 330_000_000,
    "v11h_reranker_parameters": 760_000_000,
    "phase_a_added_parameters": 0,
    "hard_ceiling_parameters": 4_000_000_000,
}


def validate_model_budget() -> dict[str, Any]:
    """Fail closed if a future change accidentally exceeds the 4B ceiling."""

    total = (
        MODEL_BUDGET["v11h_embedding_parameters"]
        + MODEL_BUDGET["v11h_reranker_parameters"]
        + MODEL_BUDGET["phase_a_added_parameters"]
    )
    if total >= MODEL_BUDGET["hard_ceiling_parameters"]:
        raise ValueError(f"model plan exceeds hard ceiling: {total} >= 4B")
    return {**MODEL_BUDGET, "total_parameters": total, "total_billions": total / 1e9, "under_4b": True}

ACTION_PHRASES = (
    "don phuong cham dut", "cham dut", "sa thai", "xu phat", "phat tien",
    "cap giay phep", "dang ky", "nop", "khai", "thu hoi", "boi thuong",
    "khieu nai", "to cao", "chuyen nhuong", "tach thua", "su dung",
    "tuyen dung", "ky hop dong", "gia han", "huy", "tam dinh chi",
)
ACTOR_PATTERNS = (
    r"(?:nguoi su dung lao dong)(?:\s+[a-z0-9]+){0,3}",
    r"(?:lao dong nu|nguoi lao dong)(?:\s+[a-z0-9]+){0,3}",
    r"(?:co quan nha nuoc|co quan co tham quyen|to chuc|ca nhan|doanh nghiep)(?:\s+[a-z0-9]+){0,3}",
)
CONDITION_CUES = (
    "trong thoi gian", "trong truong hop", "truong hop", "khi", "neu",
    "doi voi", "vi", "ke tu", "truoc khi", "sau khi",
)
EXCEPTION_CUES = ("tru", "ngoai tru", "ngoai tru truong hop", "trừ", "ngoai le")
TEMPORAL_CUES = (
    "trong thoi gian", "ke tu", "tu", "den", "truoc", "sau", "thoi han",
    "dinh ky", "hang nam", "hang thang",
)


def _content_tokens(value: str) -> tuple[str, ...]:
    return tuple(token for token in TOKEN_RE.findall(normalize(value)) if token not in CONTENT_STOPWORDS)


def _normal_tokens(value: str) -> tuple[str, ...]:
    return tuple(TOKEN_RE.findall(normalize(value)))


def _unique_in_order(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _span_after_cue(text: str, cues: Sequence[str], limit: int = 10) -> str:
    for cue in sorted(cues, key=len, reverse=True):
        match = re.search(rf"\b{re.escape(cue)}\b", text)
        if not match:
            continue
        tail = text[match.end():]
        # Stop at the next strong discourse boundary; retain the cue in the
        # element so a document must support the condition itself.
        tail = re.split(r"\b(?:thi|hay khong|khong|trừ|ngoai tru)\b|[?;]", tail, maxsplit=1)[0]
        tokens = _normal_tokens(tail)[:limit]
        if tokens:
            return " ".join((cue, *tokens))
    return ""


@dataclass(frozen=True)
class LegalConstraints:
    """Typed, deterministic legal elements extracted from one query."""

    elements: Mapping[str, str]
    critical: tuple[str, ...]
    modality: str
    numbers: tuple[str, ...]
    articles: tuple[str, ...]
    clauses: tuple[str, ...]
    points: tuple[str, ...]

    def non_empty(self) -> tuple[tuple[str, str], ...]:
        return tuple((key, value) for key, value in self.elements.items() if value)


def parse_constraints(question: str) -> LegalConstraints:
    """Parse actor/action/object/condition/exception/modality constraints.

    The parser intentionally favors high precision over full linguistic
    coverage.  Unknown question forms still receive a ``topic`` element, so
    the rescue gate degrades safely instead of silently accepting every tail.
    """

    text = " ".join(_normal_tokens(question))
    elements: dict[str, str] = {}

    actor_matches = [re.search(pattern, text) for pattern in ACTOR_PATTERNS]
    actor_matches = [match for match in actor_matches if match]
    if actor_matches:
        elements["actor"] = max((match.group(0) for match in actor_matches), key=len)

    action = next((phrase for phrase in ACTION_PHRASES if re.search(rf"\b{re.escape(phrase)}\b", text)), "")
    if action:
        elements["action"] = action
        action_match = re.search(rf"\b{re.escape(action)}\b", text)
        assert action_match is not None
        tail = text[action_match.end():]
        tail = re.split(r"\b(?:khi|neu|trong thoi gian|trong truong hop|doi voi|hay khong|khong)\b|[?;]", tail, maxsplit=1)[0]
        object_tokens = _content_tokens(tail)[:8]
        if object_tokens:
            elements["object"] = " ".join(object_tokens)

    condition = _span_after_cue(text, CONDITION_CUES, 10)
    if condition:
        elements["condition"] = condition
    exception = _span_after_cue(text, EXCEPTION_CUES, 10)
    if exception:
        elements["exception"] = exception
    temporal = _span_after_cue(text, TEMPORAL_CUES, 8)
    if temporal:
        elements["temporal"] = temporal

    modality = "unknown"
    for label, pattern in MODALITY_PATTERNS:
        if re.search(pattern, text):
            modality = label
            break
    if modality != "unknown":
        elements["modality"] = modality.replace("_", " ")

    numbers = _unique_in_order(match.group(0).replace(" ", "") for match in NUMBER_RE.finditer(text))
    if numbers:
        elements["number"] = " ".join(numbers)

    articles = _unique_in_order(match.group(1) for match in ARTICLE_RE.finditer(text))
    clauses = _unique_in_order(match.group(1) for match in CLAUSE_RE.finditer(text))
    points = _unique_in_order(match.group(1).lower() for match in POINT_RE.finditer(text))
    if articles:
        elements["article"] = " ".join(articles)
    if clauses:
        elements["clause"] = " ".join(clauses)
    if points:
        elements["point"] = " ".join(points)

    topic_tokens = _content_tokens(text)
    if topic_tokens:
        elements["topic"] = " ".join(topic_tokens[:18])
    if not elements:
        elements["topic"] = text

    # Condition, exception, modality, number and explicit references are the
    # hard negatives' usual differentiators.  Topic is always a backstop.
    critical = tuple(
        key for key in ("action", "object", "condition", "exception", "modality", "temporal", "number", "article", "clause", "point")
        if elements.get(key)
    )
    if not critical:
        critical = ("topic",)
    return LegalConstraints(elements, critical, modality, numbers, articles, clauses, points)


@dataclass(frozen=True)
class Passage:
    doc_id: str
    chunk_id: str
    title: str
    text: str
    unit_type: str = ""
    article: str = ""
    clause: str = ""
    point: str = ""
    tokens: frozenset[str] = field(init=False, repr=False)
    content_tokens: frozenset[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        normal = _normal_tokens(self.text)
        object.__setattr__(self, "tokens", frozenset(normal))
        object.__setattr__(self, "content_tokens", frozenset(token for token in normal if token not in CONTENT_STOPWORDS))


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    passages: tuple[Passage, ...]
    tokens: frozenset[str] = field(init=False, repr=False)
    content_tokens: frozenset[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        normal = _normal_tokens(self.text)
        object.__setattr__(self, "tokens", frozenset(normal))
        object.__setattr__(self, "content_tokens", frozenset(token for token in normal if token not in CONTENT_STOPWORDS))


def _structure_from_text(text: str) -> tuple[str, str, str]:
    article = ARTICLE_RE.search(text)
    clause = CLAUSE_RE.search(text)
    point = POINT_RE.search(text)
    return (
        article.group(1) if article else "",
        clause.group(1) if clause else "",
        point.group(1).lower() if point else "",
    )


def _idf(token: str, document_frequency: Mapping[str, int], corpus_size: int) -> float:
    return math.log((corpus_size + 1.0) / (float(document_frequency.get(token, 0)) + 1.0)) + 1.0


def _weighted_recall(query_tokens: Iterable[str], candidate_tokens: set[str], document_frequency: Mapping[str, int], corpus_size: int) -> float:
    query = set(query_tokens)
    if not query:
        return 0.0
    total = sum(_idf(token, document_frequency, corpus_size) for token in query)
    hit = sum(_idf(token, document_frequency, corpus_size) for token in query & candidate_tokens)
    return hit / total if total else 0.0


def _contains_phrase(phrase: str, text: str) -> bool:
    tokens = _content_tokens(phrase)
    if len(tokens) < 2:
        return bool(tokens and tokens[0] in text.split())
    return " ".join(tokens) in " ".join(_normal_tokens(text))


def _modality_matches(constraints: LegalConstraints, text: str) -> tuple[float, float]:
    doc_labels = {label for label, pattern in DOC_MODALITY_PATTERNS if re.search(pattern, text)}
    if not doc_labels:
        return 0.0, 0.0
    if constraints.modality == "permission_question":
        # A yes/no permission question may be answered by either permission or
        # prohibition; polarity is resolved by the statute's wording.
        return 1.0, 0.0
    if constraints.modality == "prohibition":
        return float("prohibition" in doc_labels), float("permission" in doc_labels and "prohibition" not in doc_labels)
    if constraints.modality == "obligation":
        return float("obligation" in doc_labels), float("permission" in doc_labels and "obligation" not in doc_labels)
    if constraints.modality == "permission":
        return float("permission" in doc_labels), float("prohibition" in doc_labels and "permission" not in doc_labels)
    return 0.5, 0.0


@dataclass(frozen=True)
class EvidenceSet:
    passages: tuple[Passage, ...]
    score: float
    coverage: float
    critical_coverage: float
    hierarchy: float
    modality: float
    number: float
    redundancy: float
    contradiction: float


@dataclass(frozen=True)
class CandidateFeatures:
    doc_id: str
    candidate_rank: int
    neural_score: float
    neural_rank: int
    constraint_scores: Mapping[str, float]
    element_mean: float
    element_min: float
    element_std: float
    critical_min: float
    coverage_ratio: float
    rare_coverage: float
    modality: float
    contradiction: float
    number_exact: float
    structure: float
    lexical: float
    evidence: EvidenceSet

    @property
    def independent_signals(self) -> int:
        return sum((
            self.critical_min >= 0.52 and self.coverage_ratio >= 0.55,
            self.evidence.critical_coverage >= 0.52 and self.evidence.score >= 0.42,
            self.modality >= 0.5 and self.contradiction <= 0.1,
            self.number_exact >= 1.0 if self.number_exact else False,
            self.structure >= 0.5,
        ))

    @property
    def selector_score(self) -> float:
        return (
            0.28 * self.critical_min
            + 0.18 * self.coverage_ratio
            + 0.20 * self.evidence.critical_coverage
            + 0.14 * self.evidence.score
            + 0.08 * self.lexical
            + 0.06 * self.modality
            + 0.04 * self.structure
            + 0.02 * self.number_exact
            - 0.16 * self.contradiction
            - 0.05 * self.evidence.redundancy
        )


def _passage_element_scores(
    passage: Passage,
    constraints: LegalConstraints,
    document_frequency: Mapping[str, int],
    corpus_size: int,
) -> tuple[dict[str, float], float, float, float]:
    modality, contradiction = _modality_matches(constraints, passage.text)
    number = 0.0
    if constraints.numbers:
        passage_numbers = {match.group(0).replace(" ", "") for match in NUMBER_RE.finditer(normalize(passage.text))}
        number = float(set(constraints.numbers).issubset(passage_numbers))
    scores: dict[str, float] = {}
    for key, value in constraints.non_empty():
        if key == "modality":
            scores[key] = modality
            continue
        if key == "number":
            scores[key] = number
            continue
        if key == "article":
            scores[key] = float(passage.article in constraints.articles and bool(passage.article))
            continue
        if key == "clause":
            scores[key] = float(passage.clause in constraints.clauses and bool(passage.clause))
            continue
        if key == "point":
            scores[key] = float(passage.point in constraints.points and bool(passage.point))
            continue
        tokens = set(_content_tokens(value))
        score = _weighted_recall(tokens, set(passage.content_tokens), document_frequency, corpus_size)
        if len(tokens) >= 2 and _contains_phrase(value, passage.text):
            score = min(1.0, score + 0.18)
        scores[key] = score
    return scores, modality, contradiction, number


def _evidence_set_score(
    selected: Sequence[Passage],
    constraints: LegalConstraints,
    passage_rows: Mapping[str, tuple[dict[str, float], float, float, float]],
    query_tokens: set[str],
    document_frequency: Mapping[str, int],
    corpus_size: int,
) -> EvidenceSet:
    if not selected:
        return EvidenceSet((), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    keys = [key for key, _ in constraints.non_empty()]
    all_scores = [passage_rows[item.chunk_id][0] for item in selected]
    coverage_by_key = {
        key: max((row.get(key, 0.0) for row in all_scores), default=0.0)
        for key in keys
    }
    weights = {key: (2.0 if key in constraints.critical else 1.0) for key in keys}
    total_weight = sum(weights.values()) or 1.0
    coverage = sum(weights[key] * value for key, value in coverage_by_key.items()) / total_weight
    critical_values = [coverage_by_key.get(key, 0.0) for key in constraints.critical]
    critical_coverage = statistics.fmean(critical_values) if critical_values else coverage
    modality = max((passage_rows[item.chunk_id][1] for item in selected), default=0.0)
    contradiction = max((passage_rows[item.chunk_id][2] for item in selected), default=0.0)
    number = max((passage_rows[item.chunk_id][3] for item in selected), default=0.0)
    articles = {item.article for item in selected if item.article}
    clauses = {(item.article, item.clause) for item in selected if item.article and item.clause}
    hierarchy = min(1.0, (len(articles) + 0.5 * len(clauses)) / max(len(selected), 1))
    token_sets = [set(item.content_tokens) for item in selected]
    pair_overlaps = [len(left & right) / max(len(left | right), 1) for left, right in itertools.combinations(token_sets, 2)]
    redundancy = statistics.fmean(pair_overlaps) if pair_overlaps else 0.0
    union_tokens = set().union(*token_sets)
    lexical = _weighted_recall(query_tokens, union_tokens, document_frequency, corpus_size)
    passage_quality = statistics.fmean([
        0.55 * max(row[0].values(), default=0.0)
        + 0.25 * row[1]
        + 0.10 * row[3]
        + 0.10 * _weighted_recall(query_tokens, set(item.content_tokens), document_frequency, corpus_size)
        for item in selected
        for row in [passage_rows[item.chunk_id]]
    ])
    score = (
        0.35 * critical_coverage
        + 0.17 * coverage
        + 0.15 * passage_quality
        + 0.10 * lexical
        + 0.10 * hierarchy
        + 0.08 * modality
        + 0.05 * number
        - 0.16 * redundancy
        - 0.25 * contradiction
    )
    return EvidenceSet(tuple(selected), max(0.0, min(1.0, score)), coverage, critical_coverage, hierarchy, modality, number, redundancy, contradiction)


def exhaustive_evidence_selection(
    passages: Sequence[Passage],
    constraints: LegalConstraints,
    query: str,
    document_frequency: Mapping[str, int] | None = None,
    corpus_size: int | None = None,
    top_k: int = 6,
    limit: int = 3,
) -> EvidenceSet:
    """Select the best 1–3 passage set from at most six passages (41 sets)."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    document_frequency = document_frequency or Counter()
    corpus_size = max(int(corpus_size or 1), 1)
    query_tokens = set(_content_tokens(query))
    ranked: list[tuple[float, Passage, tuple[dict[str, float], float, float, float]]] = []
    for passage in passages:
        row = _passage_element_scores(passage, constraints, document_frequency, corpus_size)
        lexical = _weighted_recall(query_tokens, set(passage.content_tokens), document_frequency, corpus_size)
        ranked.append((0.65 * max(row[0].values(), default=0.0) + 0.25 * lexical + 0.10 * row[1] - 0.20 * row[2], passage, row))
    ranked.sort(key=lambda item: (-item[0], item[1].chunk_id))
    shortlist = ranked[:max(top_k, limit)]
    if not shortlist:
        return EvidenceSet((), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    rows = {passage.chunk_id: row for _, passage, row in shortlist}
    candidates: list[EvidenceSet] = []
    for size in range(1, min(limit, len(shortlist)) + 1):
        for combo in itertools.combinations((item[1] for item in shortlist), size):
            candidates.append(_evidence_set_score(combo, constraints, rows, query_tokens, document_frequency, corpus_size))
    return max(candidates, key=lambda item: (item.score, item.critical_coverage, item.coverage, -len(item.passages), tuple(item.chunk_id for item in item.passages)))


def build_candidate_features(
    query: str,
    candidate_rank: int,
    doc: Document,
    constraints: LegalConstraints,
    neural_score: float,
    neural_rank: int,
    neural_values: Sequence[float],
    document_frequency: Mapping[str, int],
    corpus_size: int,
    with_evidence: bool = True,
) -> CandidateFeatures:
    element_scores: dict[str, float] = {}
    modality, contradiction = _modality_matches(constraints, doc.text)
    number_exact = 0.0
    if constraints.numbers:
        doc_numbers = {match.group(0).replace(" ", "") for match in NUMBER_RE.finditer(normalize(doc.text))}
        number_exact = float(set(constraints.numbers).issubset(doc_numbers))
    for key, value in constraints.non_empty():
        if key == "modality":
            element_scores[key] = modality
            continue
        if key == "number":
            element_scores[key] = number_exact
            continue
        if key == "article":
            element_scores[key] = float(any(passage.article in constraints.articles and bool(passage.article) for passage in doc.passages))
            continue
        if key == "clause":
            element_scores[key] = float(any(passage.clause in constraints.clauses and bool(passage.clause) for passage in doc.passages))
            continue
        if key == "point":
            element_scores[key] = float(any(passage.point in constraints.points and bool(passage.point) for passage in doc.passages))
            continue
        tokens = set(_content_tokens(value))
        score = _weighted_recall(tokens, set(doc.content_tokens), document_frequency, corpus_size)
        if len(tokens) >= 2 and _contains_phrase(value, doc.text):
            score = min(1.0, score + 0.18)
        element_scores[key] = score
    values = list(element_scores.values()) or [0.0]
    critical_values = [element_scores.get(key, 0.0) for key in constraints.critical]
    query_tokens = set(_content_tokens(query))
    lexical = _weighted_recall(query_tokens, set(doc.content_tokens), document_frequency, corpus_size)
    structure = 0.0
    if constraints.articles:
        structure = float(any(passage.article in constraints.articles for passage in doc.passages))
    elif constraints.clauses or constraints.points:
        structure = float(any(
            (not constraints.clauses or passage.clause in constraints.clauses)
            and (not constraints.points or passage.point in constraints.points)
            for passage in doc.passages
        ))
    evidence = (
        exhaustive_evidence_selection(
            doc.passages, constraints, query, document_frequency, corpus_size,
        )
        if with_evidence
        else EvidenceSet((), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, contradiction)
    )
    if not neural_values:
        normalized_neural = 0.0
    else:
        ordered = sorted(float(value) for value in neural_values)
        low, high = ordered[0], ordered[-1]
        normalized_neural = (float(neural_score) - low) / (high - low) if high > low else 0.5
    return CandidateFeatures(
        doc.doc_id, candidate_rank, float(neural_score), neural_rank,
        element_scores, statistics.fmean(values), min(values), statistics.pstdev(values),
        min(critical_values) if critical_values else min(values),
        statistics.fmean([value >= 0.45 for value in values]),
        statistics.fmean([value >= 0.65 for value in values]),
        modality, contradiction, number_exact, structure, lexical, evidence,
    )


def prefilter_score(features: CandidateFeatures) -> float:
    """Cheap high-recall score used before the exact evidence-set pass."""

    return (
        0.52 * features.critical_min
        + 0.24 * features.coverage_ratio
        + 0.16 * features.lexical
        + 0.08 * features.modality
        - 0.12 * features.contradiction
    )


def cheap_document_score(
    constraints: LegalConstraints,
    doc: Document,
    query_tokens: set[str],
    element_tokens: Mapping[str, set[str]],
    document_frequency: Mapping[str, int],
    corpus_size: int,
) -> float:
    """Score one document with only set arithmetic before evidence search."""

    modality, contradiction = _modality_matches(constraints, doc.text)
    values: list[float] = []
    for key in constraints.critical:
        if key == "modality":
            values.append(modality)
        elif key == "number":
            values.append(
                float(set(constraints.numbers).issubset({
                    match.group(0).replace(" ", "")
                    for match in NUMBER_RE.finditer(normalize(doc.text))
                }))
            )
        elif key in {"article", "clause", "point"}:
            values.append(float(any(
                (key != "article" or passage.article in constraints.articles)
                and (key != "clause" or passage.clause in constraints.clauses)
                and (key != "point" or passage.point in constraints.points)
                for passage in doc.passages
            )))
        else:
            values.append(_weighted_recall(element_tokens.get(key, set()), set(doc.content_tokens), document_frequency, corpus_size))
    critical_min = min(values) if values else 0.0
    coverage = sum(value >= 0.45 for value in values) / max(len(values), 1)
    lexical = _weighted_recall(query_tokens, set(doc.content_tokens), document_frequency, corpus_size)
    return 0.52 * critical_min + 0.24 * coverage + 0.16 * lexical + 0.08 * modality - 0.12 * contradiction


@dataclass(frozen=True)
class RescueConfig:
    min_selector_score: float = 0.49
    min_constraint_min: float = 0.42
    min_critical_coverage: float = 0.52
    min_evidence_score: float = 0.40
    min_margin: float = 0.035
    max_contradiction: float = 0.25
    min_independent_signals: int = 2
    candidate_start_rank: int = 6


def apply_safe_rescue(
    anchor: Sequence[str],
    candidate_features: Mapping[str, CandidateFeatures],
    config: RescueConfig = RescueConfig(),
) -> tuple[list[str], dict[str, Any] | None]:
    """Replace only the anchor's fifth slot when every safety gate passes."""

    answers = list(dict.fromkeys(str(value) for value in anchor))
    if len(answers) != 5:
        raise ValueError("anchor must contain exactly five unique IDs")
    old_id = answers[-1]
    old = candidate_features.get(old_id)
    if old is None:
        return answers, None
    proposals = [
        item for doc_id, item in candidate_features.items()
        if doc_id not in answers and item.candidate_rank >= config.candidate_start_rank
    ]
    if not proposals:
        return answers, None
    best = max(proposals, key=lambda item: (item.selector_score, -item.candidate_rank, item.doc_id))
    margin = best.selector_score - old.selector_score
    accepted = (
        best.selector_score >= config.min_selector_score
        and best.element_min >= config.min_constraint_min
        and best.critical_min >= config.min_constraint_min
        and best.evidence.critical_coverage >= config.min_critical_coverage
        and best.evidence.score >= config.min_evidence_score
        and best.contradiction <= config.max_contradiction
        and best.independent_signals >= config.min_independent_signals
        and margin >= config.min_margin
    )
    trace = {
        "old_id": old_id,
        "new_id": best.doc_id,
        "old_rank": old.candidate_rank,
        "new_rank": best.candidate_rank,
        "old_score": old.selector_score,
        "new_score": best.selector_score,
        "margin": margin,
        "new_critical_min": best.critical_min,
        "new_evidence_score": best.evidence.score,
        "new_evidence_critical_coverage": best.evidence.critical_coverage,
        "new_contradiction": best.contradiction,
        "independent_signals": best.independent_signals,
        "accepted": accepted,
    }
    if accepted:
        return [*answers[:-1], best.doc_id], trace
    return answers, trace


def intervention_report(
    baseline: Mapping[str, Mapping[str, Sequence[str]]],
    rescued: Mapping[str, Mapping[str, Sequence[str]]],
    references: Mapping[str, Mapping[str, Sequence[str]]],
) -> dict[str, Any]:
    successful = destructive = neutral = 0
    for qid, old_item in baseline.items():
        old = list(old_item["answer"])
        new = list(rescued[qid]["answer"])
        if old == new:
            continue
        gold = set(map(str, references[qid]["answer"]))
        old_tail, new_tail = old[-1], new[-1]
        if new_tail in gold and old_tail not in gold:
            successful += 1
        elif new_tail not in gold and old_tail in gold:
            destructive += 1
        else:
            neutral += 1
    denominator = successful + destructive
    return {
        "changes": successful + destructive + neutral,
        "successful_rescues": successful,
        "destructive_rescues": destructive,
        "neutral_changes": neutral,
        "intervention_precision": successful / denominator if denominator else 1.0,
        "destructive_rate": destructive / max(successful + destructive + neutral, 1),
    }


@dataclass
class AuditSplit:
    name: str
    qids: list[str]
    questions: list[str]
    candidates: list[list[str]]
    neural_scores: list[dict[str, float]]
    v11_scores: list[dict[str, float]]
    anchor: dict[str, dict[str, list[str]]]
    references: dict[str, dict[str, list[str]]] | None
    feature_blocks: list[np.ndarray] = field(default_factory=list)
    feature_names: tuple[str, ...] = ()
    excluded_qids: list[str] = field(default_factory=list)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_audit_split(
    name: str,
    stage1_path: Path,
    stage2_path: Path,
    score_cache: Path,
    score_key: str,
    contexts: Mapping[str, Document],
    reference_path: Path | None = None,
    candidate_limit: int = 35,
    limit: int | None = None,
    feature_cache: Path | None = None,
) -> AuditSplit:
    stage1 = _read_json(stage1_path)
    stage2 = _read_json(stage2_path)
    queries = stage1.get("queries")
    candidates = stage1.get("stage1_candidates")
    score_rows = stage2.get("queries")
    if not isinstance(queries, list) or not isinstance(candidates, list) or not isinstance(score_rows, list):
        raise ValueError(f"{name}: invalid stage audit")
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        queries = queries[:limit]
        candidates = candidates[:limit]
    by_qid = {str(row["question_id"]): row for row in score_rows}
    with np.load(score_cache, allow_pickle=False) as cache:
        values = np.asarray(cache[score_key], dtype=np.float32)
        metadata_raw = cache["metadata"].item()
    metadata = json.loads(str(metadata_raw))
    cache_qids = metadata.get(f"{name}_qids")
    cache_rows = {str(qid): values[index] for index, qid in enumerate(cache_qids)} if cache_qids else None
    raw_references = _read_json(reference_path) if reference_path else None
    feature_rows: dict[str, np.ndarray] | None = None
    feature_names: tuple[str, ...] = ()
    phrase_column: int | None = None
    if feature_cache is not None:
        with np.load(feature_cache, allow_pickle=False) as cache:
            feature_metadata = json.loads(str(cache["metadata"].item()))
            feature_values = np.asarray(cache[f"{name}_matrix"], dtype=np.float32)
            feature_names = tuple(str(value) for value in cache["feature_names"])
        feature_limit = int(feature_metadata["candidate_limit"])
        feature_qids = [str(value) for value in feature_metadata[f"{name}_qids"]]
        if feature_values.shape[0] != len(feature_qids) * feature_limit:
            raise ValueError(f"{name}: malformed v11 feature cache")
        feature_rows = {
            qid: feature_values[index * feature_limit:(index + 1) * feature_limit]
            for index, qid in enumerate(feature_qids)
        }
        phrase_column = feature_names.index("longest_phrase_tokens")
    qids: list[str] = []
    questions: list[str] = []
    candidate_rows: list[list[str]] = []
    neural_rows: list[dict[str, float]] = []
    v11_rows: list[dict[str, float]] = []
    base_feature_blocks: list[np.ndarray] = []
    anchor: dict[str, dict[str, list[str]]] = {}
    references: dict[str, dict[str, list[str]]] | None = {} if raw_references is not None else None
    excluded_qids: list[str] = []
    for query, raw_candidates in zip(queries, candidates):
        qid = str(query["query_id"])
        if qid not in by_qid:
            raise ValueError(f"{name}: stage2 missing {qid}")
        row = by_qid[qid]
        candidate_values = list(dict.fromkeys(str(value) for value in raw_candidates))
        selected = candidate_values[:candidate_limit]
        score_map = {str(key): float(value) for key, value in row.get("neural_scores", {}).items()}
        cached = cache_rows.get(qid) if cache_rows is not None else None
        if cached is None:
            # The v11h OOF cache intentionally excludes the three exact-memory
            # complement questions.  They are not eligible for fit-only
            # tuning and must be dropped rather than silently imputed.
            if name == "complement":
                excluded_qids.append(qid)
                continue
            raise ValueError(f"{name}: score cache missing {qid}")
        if len(cached) < len(selected):
            raise ValueError(f"{name}/{qid}: score cache shorter than candidates")
        feature_block = feature_rows.get(qid) if feature_rows is not None else None
        if feature_rows is not None and feature_block is None:
            if name == "complement":
                excluded_qids.append(qid)
                continue
            raise ValueError(f"{name}: feature cache missing {qid}")
        if feature_block is not None and len(feature_block) < len(selected):
            raise ValueError(f"{name}/{qid}: feature cache shorter than candidates")
        cached_scores = {doc_id: float(cached[index]) for index, doc_id in enumerate(selected)}
        # The score cache is the locked v11h selector.  Its rows are aligned to
        # the safe-tail Stage-1 candidate list; stage2 scores are retained as a
        # separate independent signal and must not silently replace it.
        ordered = sorted(range(len(selected)), key=lambda index: (-float(cached[index]), index))
        answers = [selected[index] for index in ordered[:5]]
        if feature_block is not None and phrase_column is not None:
            phrase_by_id = {
                doc_id: int(feature_block[index, phrase_column])
                for index, doc_id in enumerate(selected)
            }
            old_phrase = phrase_by_id.get(answers[-1], 0)
            best: tuple[int, int, str] | None = None
            for rank, doc_id in enumerate(selected, 1):
                if doc_id in answers:
                    continue
                phrase = phrase_by_id.get(doc_id, 0)
                proposal = (phrase, -rank, doc_id)
                if phrase >= 10 and (best is None or proposal > best):
                    best = proposal
            if best is not None and best[0] > old_phrase:
                answers[-1] = best[2]
        qids.append(qid)
        questions.append(str(query.get("query", query.get("question", ""))))
        candidate_rows.append(selected)
        neural_rows.append({doc_id: float(score_map.get(doc_id, 0.0)) for doc_id in selected})
        v11_rows.append(cached_scores)
        if feature_block is not None:
            base_feature_blocks.append(feature_block[:len(selected)])
        anchor[qid] = {"answer": answers}
        if references is not None:
            references[qid] = {"answer": [str(value) for value in raw_references[qid]["answer"]]}
    return AuditSplit(
        name, qids, questions, candidate_rows, neural_rows, v11_rows, anchor, references,
        base_feature_blocks, feature_names, excluded_qids,
    )


def build_anchor_for_question(question: str, selected: Sequence[str], scores: Sequence[float], contexts: Mapping[str, Document], pool: int = 35, threshold: int = 10) -> list[str]:
    ranked = list(selected[:pool])
    order = sorted(range(len(ranked)), key=lambda index: (-float(scores[index]), index))
    answers = [ranked[index] for index in order[:5]]
    qtokens = _normal_tokens(question)
    def match(doc_id: str) -> int:
        text = contexts.get(doc_id, Document(doc_id, "", "", ())).text
        for size in range(min(12, len(qtokens)), threshold - 1, -1):
            for start in range(len(qtokens) - size + 1):
                if " ".join(qtokens[start:start + size]) in " ".join(_normal_tokens(text)):
                    return size
        return 0
    old = match(answers[-1])
    best: tuple[int, int, str] | None = None
    for rank, doc_id in enumerate(ranked, 1):
        if doc_id in answers:
            continue
        value = match(doc_id)
        proposal = (value, -rank, doc_id)
        if value >= threshold and (best is None or proposal > best):
            best = proposal
    if best is not None and best[0] > old:
        answers[-1] = best[2]
    return answers


def hydrate_anchor(split: AuditSplit, contexts: Mapping[str, Document], score_cache: Path, score_key: str, pool: int = 35) -> None:
    with np.load(score_cache, allow_pickle=False) as cache:
        values = np.asarray(cache[score_key], dtype=np.float32)
        metadata = json.loads(str(cache["metadata"].item()))
    cache_qids = [str(value) for value in metadata.get(f"{split.name}_qids", split.qids)]
    row_by_qid = {qid: values[index] for index, qid in enumerate(cache_qids)}
    for index, qid in enumerate(split.qids):
        row = row_by_qid[qid]
        split.anchor[qid] = {"answer": build_anchor_for_question(split.questions[index], split.candidates[index], row, contexts, pool)}


def load_context_documents(
    contexts_zip: Path,
    needed_ids: set[str],
    index_path: Path | None = None,
    max_passages_per_doc: int = 12,
    evidence_source: str = "legal_chunking",
) -> dict[str, Document]:
    if max_passages_per_doc <= 0:
        raise ValueError("max_passages_per_doc must be positive")
    if evidence_source not in {"legal_chunking", "sqlite"}:
        raise ValueError("evidence_source must be legal_chunking or sqlite")
    raw: dict[str, tuple[str, str, str]] = {}
    if evidence_source == "legal_chunking":
        with zipfile.ZipFile(contexts_zip) as archive:
            for info in archive.infolist():
                if not info.filename.endswith(".json"):
                    continue
                item = json.loads(archive.read(info))
                doc_id = str(item["id"])
                if doc_id in needed_ids:
                    raw_text = str(item.get("passage", ""))[:12000]
                    raw[doc_id] = (normalize(str(item.get("name", ""))), normalize(raw_text), raw_text)
    # Keep a bounded, deterministic reservoir of raw rows per document.  The
    # previous v12 scan materialized every chunk and its token sets, which is
    # wasteful for long documents and can push a c35+c35 run into multi-GB
    # memory.  The full document text remains available as a fallback signal;
    # this reservoir only supplies the top-six evidence search.
    passage_rows: dict[str, list[tuple[str, str, str, str, str, str]]] = defaultdict(list)
    passage_seen: Counter[str] = Counter()
    if evidence_source == "sqlite" and (index_path is None or not index_path.exists()):
        raise FileNotFoundError("SQLite evidence source requires --index")
    if evidence_source == "sqlite" and index_path is not None and index_path.exists():
        uri = index_path.resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        try:
            # ``doc_id`` is UNINDEXED in the FTS table.  Repeating a WHERE IN
            # query therefore rescans the virtual table once per batch.  One
            # sequential pass is both faster and deterministic for this
            # bounded reservoir.
            sql = "SELECT doc_id,chunk_id,title,passage FROM chunks"
            scanned_rows = 0
            for doc_id, chunk_id, title, passage in connection.execute(sql):
                scanned_rows += 1
                if scanned_rows % 100_000 == 0:
                    print(f"sqlite evidence scan: {scanned_rows} chunks", flush=True)
                key = str(doc_id)
                if key not in needed_ids:
                    continue
                text = str(passage or "")
                parsed_article, clause, point = _structure_from_text(text)
                record = (
                    str(chunk_id), str(title or ""), text,
                    "", str(parsed_article), f"{clause}|{point}",
                )
                passage_seen[key] += 1
                rows = passage_rows[key]
                if len(rows) < max_passages_per_doc:
                    rows.append(record)
                else:
                    slot = zlib.crc32(str(chunk_id).encode("utf-8")) % passage_seen[key]
                    if slot < max_passages_per_doc:
                        rows[slot] = record
        finally:
            connection.close()
        # The canonical FTS index has the legal passage text already.  Build a
        # bounded document view from the sampled rows so sqlite mode avoids a
        # second pass over the 97MB contexts archive.
        for doc_id in needed_ids:
            rows = passage_rows.get(doc_id, [])
            if rows:
                title = rows[0][1]
                joined = " ".join(row[2] for row in rows)[:12000]
                raw[doc_id] = (normalize(title), normalize(joined), joined)
    documents: dict[str, Document] = {}
    for doc_id, (title, text, raw_text) in raw.items():
        if evidence_source == "legal_chunking":
            legal_chunks = list(itertools.islice(
                iter_legal_chunks(raw_text, 1800, 180), max_passages_per_doc,
            ))
            passages = tuple(
                Passage(
                    doc_id, f"{doc_id}::legal::{index}", title, chunk.text,
                    chunk.unit_type, chunk.article_number,
                    _structure_from_text(chunk.text)[1], _structure_from_text(chunk.text)[2],
                )
                for index, chunk in enumerate(legal_chunks)
            )
        else:
            selected_rows = passage_rows.get(doc_id, [])
            passages = tuple(
                Passage(
                    doc_id, chunk_id, chunk_title, passage_text, unit_type, article,
                    *(chunk_structure.split("|", 1) if "|" in chunk_structure else (chunk_structure, "")),
                )
                for chunk_id, chunk_title, passage_text, unit_type, article, chunk_structure in selected_rows
            )
        if not passages:
            passages = (Passage(doc_id, f"{doc_id}::fallback", title, text, "fallback", *_structure_from_text(text)),)
        documents[doc_id] = Document(doc_id, title, text, passages)
    return documents


def document_frequency(documents: Mapping[str, Document]) -> Counter[str]:
    result: Counter[str] = Counter()
    for doc in documents.values():
        result.update(doc.content_tokens)
    return result


def score_split(
    split: AuditSplit,
    documents: Mapping[str, Document],
    candidate_limit: int = 35,
    config: RescueConfig = RescueConfig(),
    evidence_candidates: int = 8,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, dict[str, Any]], dict[str, LegalConstraints]]:
    df = document_frequency(documents)
    corpus_size = max(len(documents), 1)
    output: dict[str, dict[str, list[str]]] = {}
    traces: dict[str, dict[str, Any]] = {}
    parsed: dict[str, LegalConstraints] = {}
    for index, qid in enumerate(split.qids):
        question = split.questions[index]
        constraints = parse_constraints(question)
        parsed[qid] = constraints
        candidates = split.candidates[index][:candidate_limit]
        neural = split.neural_scores[index]
        candidate_position = {doc_id: position for position, doc_id in enumerate(candidates)}
        neural_order = sorted(candidates, key=lambda doc_id: (-float(neural.get(doc_id, -1e9)), candidate_position[doc_id]))
        query_tokens = set(_content_tokens(question))
        element_tokens = {
            key: set(_content_tokens(value))
            for key, value in constraints.non_empty()
        }
        cheap: list[tuple[float, int, str]] = []
        for rank, doc_id in enumerate(candidates, 1):
            if rank < 6 or doc_id not in documents:
                continue
            cheap.append((
                cheap_document_score(
                    constraints, documents[doc_id], query_tokens, element_tokens,
                    df, corpus_size,
                ),
                -rank,
                doc_id,
            ))
        rescue_ids = {
            doc_id for _, _, doc_id in sorted(cheap, reverse=True)[:max(evidence_candidates, 1)]
        }
        rescue_ids.add(split.anchor[qid]["answer"][-1])
        features: dict[str, CandidateFeatures] = {}
        for rank, doc_id in enumerate(candidates, 1):
            if doc_id not in rescue_ids:
                continue
            doc = documents.get(doc_id)
            if doc is None:
                continue
            features[doc_id] = build_candidate_features(
                question, rank, doc, constraints, float(neural.get(doc_id, 0.0)),
                neural_order.index(doc_id) + 1, [neural.get(value, 0.0) for value in candidates],
                df, corpus_size,
            )
        rescued, trace = apply_safe_rescue(split.anchor[qid]["answer"], features, config)
        output[qid] = {"answer": rescued}
        traces[qid] = {
            "question": question,
            "constraints": {"elements": dict(constraints.elements), "critical": list(constraints.critical), "modality": constraints.modality},
            "anchor": list(split.anchor[qid]["answer"]),
            "rescue": trace,
            "evidence_shortlist_size": len(rescue_ids),
            "candidate_oracle_rank": next((rank for rank, doc_id in enumerate(candidates, 1) if split.references and doc_id in split.references[qid]["answer"]), None),
        }
        if (index + 1) % 100 == 0 or index + 1 == len(split.qids):
            print(f"safe-rescue/{split.name}: {index + 1}/{len(split.qids)}", flush=True)
    return output, traces, parsed


def references_for(split: AuditSplit) -> dict[str, dict[str, list[str]]]:
    if split.references is None:
        raise ValueError("references unavailable")
    return split.references


def evaluate_split(split: AuditSplit, predictions: Mapping[str, Mapping[str, Sequence[str]]]) -> dict[str, Any]:
    refs = references_for(split)
    baseline = official_metrics(split.anchor, refs)
    rescued = official_metrics(dict(predictions), refs)
    return {
        "baseline": baseline,
        "rescued": rescued,
        "delta": {key: rescued[key] - baseline[key] for key in ("recall", "precision")},
        "intervention": intervention_report(split.anchor, predictions, refs),
    }


def replacement_oracle_report(split: AuditSplit, candidate_limit: int = 35) -> dict[str, Any]:
    """Measure the exact upside available to a one-slot c35 rescue.

    The production policy is restricted to replacing slot five with a Stage-1
    candidate whose original rank is 6--35.  This oracle obeys that same
    restriction; it therefore measures reachable upside rather than an
    unconstrained reranking ceiling.
    """

    refs = references_for(split)
    slot_oracle: dict[str, dict[str, list[str]]] = {}
    c35_oracle: dict[str, dict[str, list[str]]] = {}
    location = Counter()
    tail_eligible = 0
    for index, qid in enumerate(split.qids):
        anchor = list(split.anchor[qid]["answer"])
        candidates = split.candidates[index][:candidate_limit]
        gold = set(str(value) for value in refs[qid]["answer"])
        missing_gold = [doc_id for doc_id in candidates if doc_id in gold and doc_id not in anchor]
        missing_ranks = [rank for rank, doc_id in enumerate(candidates, 1) if doc_id in gold and doc_id not in anchor]
        if not missing_ranks:
            location["none_missing_inside_c35"] += 1
        elif min(missing_ranks) <= 5:
            location["rank_1_5_outside_anchor"] += 1
        elif min(missing_ranks) <= 10:
            location["rank_6_10"] += 1
        else:
            location["rank_11_35"] += 1

        rescued = list(anchor)
        eligible = [
            doc_id for rank, doc_id in enumerate(candidates, 1)
            if 6 <= rank <= candidate_limit and doc_id in gold and doc_id not in anchor
        ]
        if eligible and anchor[-1] not in gold:
            rescued[-1] = eligible[0]
            tail_eligible += 1
        slot_oracle[qid] = {"answer": rescued}

        ideal = [doc_id for doc_id in candidates if doc_id in gold][:5]
        ideal.extend(doc_id for doc_id in anchor if doc_id not in ideal)
        ideal.extend(doc_id for doc_id in candidates if doc_id not in ideal)
        c35_oracle[qid] = {"answer": ideal[:5]}

    baseline = official_metrics(split.anchor, refs)
    slot_metrics = official_metrics(slot_oracle, refs)
    c35_metrics = official_metrics(c35_oracle, refs)
    return {
        "questions": len(split.qids),
        "candidate_limit": candidate_limit,
        "baseline": baseline,
        "missing_gold_location": dict(location),
        "slot5_rank6_35_eligible_queries": tail_eligible,
        "slot5_rank6_35_oracle": slot_metrics,
        "slot5_recall_upside": float(slot_metrics["recall"] - baseline["recall"]),
        "c35_top5_oracle": c35_metrics,
        "c35_recall_upside": float(c35_metrics["recall"] - baseline["recall"]),
        "slot5_intervention": intervention_report(split.anchor, slot_oracle, refs),
    }


def config_grid() -> list[RescueConfig]:
    return [
        RescueConfig(min_selector_score=score, min_margin=margin, min_critical_coverage=critical)
        for score, margin, critical in itertools.product((0.49, 0.53, 0.57, 0.61), (0.025, 0.05, 0.075, 0.10), (0.45, 0.52, 0.60))
    ]


def tune_config(
    split: AuditSplit,
    documents: Mapping[str, Document],
    candidate_limit: int,
    evidence_candidates: int = 8,
) -> tuple[RescueConfig, dict[str, Any]]:
    # Build the expensive candidate feature cache once, then evaluate only the
    # cheap one-slot policy grid against it.
    df = document_frequency(documents)
    corpus_size = max(len(documents), 1)
    feature_rows: dict[str, dict[str, CandidateFeatures]] = {}
    for index, qid in enumerate(split.qids):
        constraints = parse_constraints(split.questions[index])
        candidates = split.candidates[index][:candidate_limit]
        neural = split.neural_scores[index]
        candidate_position = {doc_id: position for position, doc_id in enumerate(candidates)}
        order = sorted(candidates, key=lambda doc_id: (-float(neural.get(doc_id, -1e9)), candidate_position[doc_id]))
        query_tokens = set(_content_tokens(split.questions[index]))
        element_tokens = {
            key: set(_content_tokens(value))
            for key, value in constraints.non_empty()
        }
        cheap: list[tuple[float, int, str]] = []
        for rank, doc_id in enumerate(candidates, 1):
            if rank < 6 or doc_id not in documents:
                continue
            cheap.append((
                cheap_document_score(
                    constraints, documents[doc_id], query_tokens, element_tokens,
                    df, corpus_size,
                ),
                -rank,
                doc_id,
            ))
        rescue_ids = {
            doc_id for _, _, doc_id in sorted(cheap, reverse=True)[:max(evidence_candidates, 1)]
        }
        rescue_ids.add(split.anchor[qid]["answer"][-1])
        feature_rows[qid] = {
            doc_id: build_candidate_features(
                split.questions[index], rank, documents[doc_id], constraints,
                float(neural.get(doc_id, 0.0)), order.index(doc_id) + 1,
                [neural.get(value, 0.0) for value in candidates], df, corpus_size,
            )
            for rank, doc_id in enumerate(candidates, 1) if doc_id in documents and doc_id in rescue_ids
        }
    leaderboard: list[dict[str, Any]] = []
    for config in config_grid():
        candidate_predictions: dict[str, dict[str, list[str]]] = {}
        for qid in split.qids:
            candidate_predictions[qid] = {"answer": apply_safe_rescue(split.anchor[qid]["answer"], feature_rows[qid], config)[0]}
        evaluation = evaluate_split(split, candidate_predictions)
        intervention = evaluation["intervention"]
        leaderboard.append({
            "config": config.__dict__,
            "evaluation": evaluation,
            "eligible": intervention["intervention_precision"] >= 0.85 and intervention["destructive_rate"] <= 0.03,
        })
    eligible = [item for item in leaderboard if item["eligible"]]
    winner = max(eligible or leaderboard, key=lambda item: (
        item["evaluation"]["rescued"]["recall"],
        item["evaluation"]["intervention"]["intervention_precision"],
        -item["evaluation"]["intervention"]["changes"],
    ))
    return RescueConfig(**winner["config"]), {"selected": winner, "leaderboard": sorted(leaderboard, key=lambda item: item["evaluation"]["rescued"]["recall"], reverse=True)[:12]}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts-zip", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--score-cache", type=Path, required=True)
    parser.add_argument(
        "--feature-cache", type=Path,
        help="Frozen v11 query-normalized feature cache; used to reproduce exact-heading anchors exactly",
    )
    parser.add_argument("--complement-stage1", type=Path)
    parser.add_argument("--complement-stage2", type=Path)
    parser.add_argument("--complement-reference", type=Path)
    parser.add_argument("--holdout-stage1", type=Path, required=True)
    parser.add_argument("--holdout-stage2", type=Path, required=True)
    parser.add_argument("--holdout-reference", type=Path, required=True)
    parser.add_argument("--public-stage1", type=Path, required=True)
    parser.add_argument("--public-stage2", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-limit", type=int, default=35)
    parser.add_argument("--tune-on-complement", action="store_true")
    parser.add_argument("--apply-public", action="store_true")
    parser.add_argument("--max-passages-per-doc", type=int, default=8)
    parser.add_argument("--evidence-candidates", type=int, default=8)
    parser.add_argument("--evidence-source", choices=("legal_chunking", "sqlite"), default="legal_chunking")
    parser.add_argument("--oracle-only", action="store_true", help="Write replacement-oracle diagnostics and stop before evidence scoring")
    parser.add_argument("--limit", type=int, help="Optional question limit for profiling; omit for the full split")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.candidate_limit < 5 or args.evidence_candidates <= 0:
        raise ValueError("candidate-limit must be at least five")
    started = time.perf_counter()
    model_budget = validate_model_budget()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    def candidate_ids(stage1_path: Path) -> set[str]:
        payload = _read_json(stage1_path)
        rows = payload["stage1_candidates"][:args.limit] if args.limit is not None else payload["stage1_candidates"]
        return {
            str(doc_id)
            for row in rows
            for doc_id in row[:args.candidate_limit]
        }

    holdout = load_audit_split(
        "holdout", args.holdout_stage1, args.holdout_stage2, args.score_cache,
        "holdout_scores", {}, args.holdout_reference, args.candidate_limit, args.limit,
        args.feature_cache,
    )
    public = load_audit_split(
        "public", args.public_stage1, args.public_stage2, args.score_cache,
        "public_scores", {}, None, args.candidate_limit, args.limit,
        args.feature_cache,
    )
    complement = None
    complement_documents: dict[str, Document] | None = None
    if args.complement_stage1 and args.complement_stage2 and args.complement_reference:
        complement = load_audit_split(
            "complement", args.complement_stage1, args.complement_stage2,
            args.score_cache, "oof_scores", {}, args.complement_reference,
            args.candidate_limit, args.limit, args.feature_cache,
        )
    oracle_report = {
        "kind": "legalir_v14_replacement_oracle",
        "selection_split": "complement only",
        "holdout_is_report_only": True,
        "candidate_limit": args.candidate_limit,
        "model_budget": model_budget,
        "complement": replacement_oracle_report(complement, args.candidate_limit) if complement is not None else None,
        "holdout": replacement_oracle_report(holdout, args.candidate_limit),
    }
    if args.oracle_only:
        oracle_path = args.output_dir / "v14_replacement_oracle_report.json"
        write_json(oracle_path, oracle_report)
        print(json.dumps({"report": str(oracle_path.resolve()), **oracle_report}, ensure_ascii=False, indent=2), flush=True)
        return 0
    config = RescueConfig()
    tuning_report: dict[str, Any] | None = None
    if args.tune_on_complement and complement is not None:
        complement_documents = load_context_documents(
            args.contexts_zip, candidate_ids(args.complement_stage1), args.index,
            args.max_passages_per_doc,
            args.evidence_source,
        )
        if args.feature_cache is None:
            hydrate_anchor(complement, complement_documents, args.score_cache, "oof_scores")
        config, tuning_report = tune_config(
            complement, complement_documents, args.candidate_limit, args.evidence_candidates,
        )
        del complement_documents
        complement_documents = None
    holdout_documents = load_context_documents(
        args.contexts_zip, candidate_ids(args.holdout_stage1), args.index,
        args.max_passages_per_doc,
        args.evidence_source,
    )
    if args.feature_cache is None:
        hydrate_anchor(holdout, holdout_documents, args.score_cache, "holdout_scores")
    holdout_predictions, holdout_traces, _ = score_split(
        holdout, holdout_documents, args.candidate_limit, config, args.evidence_candidates,
    )
    public_predictions = None
    public_traces = None
    if args.apply_public:
        del holdout_documents
        public_documents = load_context_documents(
            args.contexts_zip, candidate_ids(args.public_stage1), args.index,
            args.max_passages_per_doc,
            args.evidence_source,
        )
        if args.feature_cache is None:
            hydrate_anchor(public, public_documents, args.score_cache, "public_scores")
        public_predictions, public_traces, _ = score_split(
            public, public_documents, args.candidate_limit, config, args.evidence_candidates,
        )
        write_json(args.output_dir / "v14_safe_rescue.json", public_predictions)
    report: dict[str, Any] = {
        "kind": "legalir_v14_algorithm_first_safe_rescue",
        "baseline": "v11h score cache + exact-heading anchor",
        "candidate_limit": args.candidate_limit,
        "model_budget": model_budget,
        "model_parameters_added": model_budget["phase_a_added_parameters"],
        "reused_lineage": {
            "anchor": "v11h query-normalized pointwise LTR score cache; exact-heading t10 policy rebuilt from the same c35 candidate order",
            "candidate_generation": "v11h safe-tail corpus-only PRF/citation completion; no c35 expansion or new retrieval branch",
            "stage2_signal": "existing v11h Vietnamese cross-encoder neural_scores from the frozen c35 audit",
            "document_text": "selected-contexts.zip, normalized and capped at 12000 characters per candidate document",
            "evidence": (
                "v11_research_256/legalir.sqlite3 chunks; article/clause/point parsed from passage in hot path"
                if args.evidence_source == "sqlite"
                else "existing legal_structure.iter_legal_chunks over selected-contexts.zip; no new index/model"
            ),
            "training": "none in Phase A; optional threshold grid is fit/complement-label-only and never reads locked holdout labels",
        },
        "policy": config.__dict__,
        "architecture": {
            "typed_constraints": True,
            "hierarchical_passages": True,
            "evidence_set_search": "exhaustive_1_to_3_of_top6",
            "evidence_candidate_prefilter": args.evidence_candidates,
            "modality_exception_numeric_verifier": True,
            "max_replacements": 1,
            "replacement_slot": 5,
        },
        "replacement_oracle": oracle_report,
        "holdout": evaluate_split(holdout, holdout_predictions),
        "holdout_trace": holdout_traces,
        "complement_tuning": tuning_report,
        "public": {
            "applied": public_predictions is not None,
            "submission": str((args.output_dir / "v14_safe_rescue.json").resolve()) if public_predictions is not None else None,
            "queries_changed": sum(public_predictions[qid]["answer"] != public.anchor[qid]["answer"] for qid in public.qids) if public_predictions is not None else 0,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(args.output_dir / "v14_safe_rescue_report.json", report)
    print(json.dumps({
        "report": str((args.output_dir / "v14_safe_rescue_report.json").resolve()),
        "holdout": report["holdout"],
        "tuned_config": config.__dict__,
        "public": report["public"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
