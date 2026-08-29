"""Research-track v12 selector for the LegalIR pipeline.

This module deliberately keeps Stage-1 and the approved v11h candidates fixed.
It adds selector-side evidence only:

* deterministic multi-aspect parsing;
* Aspect-MaxSim over the existing BQB chunk embedding cache;
* greedy complementary evidence-set features;
* TF-IDF word/character features;
* grouped LambdaMART with an @5 ranking objective.

No public labels, v57 predictions, or document priors are used.  The command
opens the clean holdout final split once and only writes a public candidate
when both the OOF and locked final gates beat their respective baselines.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import statistics
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

try:
    from .evaluate import official_metrics
    from .v11_exact_heading_rescue import normalize
    from .v11_tiny_ltr import (
        AuditSplit,
        Document,
        build_leak_groups,
        build_query_normalized_features,
        load_audit_split,
        load_documents,
        load_priors,
        read_json,
        row_indices,
        write_json,
    )
except ImportError:
    from evaluate import official_metrics
    from v11_exact_heading_rescue import normalize
    from v11_tiny_ltr import (
        AuditSplit,
        Document,
        build_leak_groups,
        build_query_normalized_features,
        load_audit_split,
        load_documents,
        load_priors,
        read_json,
        row_indices,
        write_json,
    )


TOKEN_RE = re.compile(r"[0-9]+(?:[./-][0-9]+)*|[^\W_]+", re.UNICODE)
NUMBER_RE = re.compile(r"\d+")
ASPECT_SPLIT_RE = re.compile(
    r"\s*(?:[,;?]|\b(?:va|hoac|nhung|dong thoi|cung nhu|trong khi)\b)\s*",
    re.IGNORECASE,
)
ASPECT_MARKER_RE = re.compile(
    r"\b(?:trong truong hop|truong hop|doi voi|lien quan den|lien quan|"
    r"khi|neu|chi khi|tru truong hop|ngoai tru|co duoc|khong duoc|"
    r"phai|cam|duoc phep)\b",
    re.IGNORECASE,
)
LEGAL_PHRASE_RES = (
    re.compile(r"nguoi su dung lao dong(?:\s+\w+){0,3}", re.IGNORECASE),
    re.compile(r"lao dong nu(?:\s+\w+){0,3}", re.IGNORECASE),
    re.compile(r"nghi thai san(?:\s+\w+){0,3}", re.IGNORECASE),
    re.compile(r"don phuong cham dut(?:\s+\w+){0,5}", re.IGNORECASE),
    re.compile(r"hop dong lao dong(?:\s+\w+){0,3}", re.IGNORECASE),
)


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(normalize(text))]


def _clean_aspect(text: str) -> str:
    return " ".join(_tokens(text))


def parse_aspects(question: str, maximum: int = 6) -> list[str]:
    """Create stable legal aspects without an LLM.

    The full question is retained as a high-recall aspect.  Clause boundaries,
    modality markers, numbers, and common legal noun phrases are then added in
    source order and deduplicated.  The function is intentionally pure so it
    can be unit-tested and reproduced exactly during inference.
    """

    full = _clean_aspect(question)
    if not full:
        return []
    candidates: list[tuple[int, int, str, int]] = [(0, 0, full, 3)]
    normalized = normalize(question)
    for match in ASPECT_SPLIT_RE.finditer(normalized):
        left = normalized[: match.start()]
        if left.strip():
            start = max(0, match.start() - len(left))
            candidates.append((start, 0, _clean_aspect(left), 1))
    for match in ASPECT_MARKER_RE.finditer(normalized):
        tail = normalized[match.start() :]
        tail_tokens = _tokens(tail)
        if tail_tokens:
            # Keep the marker and a bounded legal predicate; this avoids a
            # single long clause swallowing every other aspect.
            window = " ".join(tail_tokens[: min(10, len(tail_tokens))])
            candidates.append((match.start(), 1, window, 3))
    for phrase_re in LEGAL_PHRASE_RES:
        for match in phrase_re.finditer(normalized):
            candidates.append((match.start(), 2, _clean_aspect(match.group(0)), 2))
    for match in re.finditer(r"\b\d+(?:[./-]\d+)*\b", normalized):
        candidates.append((match.start(), 3, match.group(0), 2))

    deduped: list[tuple[int, int, str, int]] = []
    seen: set[str] = set()
    for position, kind, value, priority in sorted(candidates, key=lambda item: (item[0], item[1])):
        value = _clean_aspect(value)
        if not value or len(value.split()) < 2 and not NUMBER_RE.search(value):
            continue
        if value in seen:
            continue
        seen.add(value)
        deduped.append((position, kind, value, priority))

    if len(deduped) <= maximum:
        return [item[2] for item in deduped]
    # Keep the full question, then the most informative marker/phrase spans,
    # and finally restore source order for deterministic feature alignment.
    first = deduped[0]
    chosen = [first]
    remaining = sorted(
        deduped[1:],
        key=lambda item: (-(item[3] + min(len(item[2].split()), 12) / 100.0), item[0], item[1]),
    )
    for item in remaining:
        if len(chosen) >= maximum:
            break
        chosen.append(item)
    return [item[2] for item in sorted(chosen, key=lambda item: (item[0], item[1]))]


@dataclass(frozen=True)
class ChunkEvidence:
    chunk_id: str
    title: str
    passage: str
    tokens: frozenset[str] = field(init=False, repr=False)
    title_tokens: frozenset[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Passage tokenization is the hottest operation in the evidence gate.
        # Cache it once per chunk so holdout/public feature construction only
        # performs set arithmetic.
        object.__setattr__(self, "tokens", frozenset(_tokens(self.passage)))
        object.__setattr__(self, "title_tokens", frozenset(_tokens(self.title)))


def load_documents_capped(
    contexts_zip: Path,
    needed_ids: set[str],
    priors: Mapping[str, int],
    max_chars: int = 12_000,
) -> tuple[dict[str, Document], Counter[str]]:
    """Load bounded document text for selector features.

    The canonical SQLite chunks remain the evidence source.  This bounded
    loader is only for the legacy v11 document-level features and prevents a
    large legal corpus from expanding to several gigabytes of Python strings.
    """

    documents: dict[str, Document] = {}
    document_frequency: Counter[str] = Counter()
    with zipfile.ZipFile(contexts_zip) as archive:
        infos = [info for info in archive.infolist() if info.filename.endswith(".json")]
        for position, info in enumerate(infos, 1):
            item = json.loads(archive.read(info))
            doc_id = str(item["id"])
            if doc_id not in needed_ids:
                continue
            body = normalize(str(item.get("passage", "")))[:max_chars]
            title = normalize(str(item.get("name", "")))
            tokens = frozenset(body.split())
            title_tokens = frozenset(title.split())
            document_frequency.update(tokens)
            documents[doc_id] = Document(
                text=body,
                title=title,
                tokens=tokens,
                title_tokens=title_tokens,
                numbers=frozenset(NUMBER_RE.findall(body)),
                token_count=len(body.split()),
                prior=int(priors.get(doc_id, 0)),
            )
            if position % 1000 == 0:
                print(f"capped contexts: {position}/{len(infos)} ({len(documents)} needed)", flush=True)
    missing = needed_ids - set(documents)
    if missing:
        print(f"warning: {len(missing)} candidate IDs absent from capped contexts", flush=True)
    return documents, document_frequency


def load_chunk_evidence(index_path: Path, needed_ids: set[str]) -> dict[str, list[ChunkEvidence]]:
    """Load only candidate passages from the read-only SQLite FTS table."""

    database_path = index_path / "legalir.sqlite3" if index_path.is_dir() else index_path
    connection = sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True)
    output: dict[str, list[ChunkEvidence]] = defaultdict(list)
    try:
        for doc_id, chunk_id, title, passage in connection.execute(
            "SELECT doc_id, chunk_id, title, passage FROM chunks"
        ):
            doc_id = str(doc_id)
            if doc_id in needed_ids:
                output[doc_id].append(
                    ChunkEvidence(str(chunk_id), str(title or ""), str(passage or ""))
                )
    finally:
        connection.close()
    return dict(output)


def _idf(token: str, document_frequency: Mapping[str, int], corpus_size: int) -> float:
    return math.log((corpus_size + 1.0) / (float(document_frequency.get(token, 0)) + 1.0)) + 1.0


def _passage_stats(
    passage: ChunkEvidence,
    query_tokens: set[str],
    aspects: Sequence[set[str]],
    document_frequency: Mapping[str, int],
    corpus_size: int,
) -> tuple[float, set[str], float, float]:
    tokens = set(passage.tokens)
    title_tokens = set(passage.title_tokens)
    idf_total = sum(_idf(token, document_frequency, corpus_size) for token in query_tokens)
    query_gain = sum(_idf(token, document_frequency, corpus_size) for token in tokens & query_tokens)
    query_score = query_gain / max(idf_total, 1e-9)
    aspect_score = (
        sum(len(tokens & aspect) / max(len(aspect), 1) for aspect in aspects) / max(len(aspects), 1)
        if aspects else 0.0
    )
    title_score = len(title_tokens & query_tokens) / max(len(query_tokens), 1)
    base = 0.52 * query_score + 0.38 * aspect_score + 0.10 * title_score
    return base, tokens, query_score, aspect_score


def greedy_evidence_selection(
    passages: Sequence[ChunkEvidence],
    query: str,
    aspects: Sequence[str],
    document_frequency: Mapping[str, int],
    corpus_size: int,
    limit: int = 3,
) -> list[ChunkEvidence]:
    """Select complementary passages with diminishing overlap."""

    query_tokens = set(_tokens(query))
    aspect_tokens = [set(_tokens(aspect)) for aspect in aspects if _tokens(aspect)]
    scored = [
        (passage, *_passage_stats(passage, query_tokens, aspect_tokens, document_frequency, corpus_size))
        for passage in passages
    ]
    selected: list[ChunkEvidence] = []
    covered: set[str] = set()
    remaining = list(scored)
    while remaining and len(selected) < limit:
        best_index = 0
        best_gain = -1.0
        for index, (passage, base, tokens, query_score, aspect_score) in enumerate(remaining):
            new_tokens = tokens - covered
            novelty = sum(_idf(token, document_frequency, corpus_size) for token in new_tokens)
            total_idf = sum(_idf(token, document_frequency, corpus_size) for token in query_tokens)
            gain = base + 0.30 * novelty / max(total_idf, 1e-9)
            gain += 0.20 * aspect_score
            if gain > best_gain:
                best_index, best_gain = index, gain
        passage, _, tokens, _, _ = remaining.pop(best_index)
        selected.append(passage)
        covered.update(tokens)
    return selected


def _evidence_row(
    passages: Sequence[ChunkEvidence],
    query: str,
    aspects: Sequence[str],
    document_frequency: Mapping[str, int],
    corpus_size: int,
) -> list[float]:
    names_count = 14
    if not passages:
        return [0.0] * names_count
    query_tokens = set(_tokens(query))
    aspect_tokens = [set(_tokens(aspect)) for aspect in aspects if _tokens(aspect)]
    scored = [
        (passage, *_passage_stats(passage, query_tokens, aspect_tokens, document_frequency, corpus_size))
        for passage in passages
    ]
    naive = [item[0] for item in sorted(scored, key=lambda item: (-item[1], item[0].chunk_id))[:3]]
    selected = greedy_evidence_selection(
        passages, query, aspects, document_frequency, corpus_size, limit=3
    )
    selected_stats = [
        _passage_stats(item, query_tokens, aspect_tokens, document_frequency, corpus_size)
        for item in selected
    ]
    naive_stats = [
        _passage_stats(item, query_tokens, aspect_tokens, document_frequency, corpus_size)
        for item in naive
    ]
    selected_tokens = set().union(*(stats[1] for stats in selected_stats)) if selected_stats else set()
    naive_tokens = set().union(*(stats[1] for stats in naive_stats)) if naive_stats else set()
    total_idf = sum(_idf(token, document_frequency, corpus_size) for token in query_tokens)
    selected_query = sum(_idf(token, document_frequency, corpus_size) for token in selected_tokens & query_tokens)
    naive_query = sum(_idf(token, document_frequency, corpus_size) for token in naive_tokens & query_tokens)
    aspect_coverages = [
        max((len(stats[1] & aspect) / max(len(aspect), 1) for stats in selected_stats), default=0.0)
        for aspect in aspect_tokens
    ]
    selected_redundancy = 1.0 - len(selected_tokens) / max(sum(len(stats[1]) for stats in selected_stats), 1)
    title_hits = sum(
        len(set(_tokens(item.title)) & query_tokens) > 0 for item in selected
    ) / max(len(selected), 1)
    return [
        len(selected) / 3.0,
        selected_query / max(total_idf, 1e-9),
        naive_query / max(total_idf, 1e-9),
        (selected_query - naive_query) / max(total_idf, 1e-9),
        statistics.mean(stats[0] for stats in selected_stats),
        max(stats[0] for stats in selected_stats),
        min(stats[0] for stats in selected_stats),
        statistics.mean(aspect_coverages) if aspect_coverages else 0.0,
        min(aspect_coverages) if aspect_coverages else 0.0,
        max(aspect_coverages) if aspect_coverages else 0.0,
        selected_redundancy,
        title_hits,
        len({item.chunk_id.split("::")[0] for item in selected}) / 3.0,
        len(selected) / max(len(passages), 1),
    ]


@dataclass
class DenseBank:
    embeddings: np.ndarray
    doc_rows: dict[str, np.ndarray]


def load_dense_bank(index_path: Path) -> DenseBank:
    if not index_path.is_dir():
        index_path = index_path.parent
    ids = json.loads((index_path / "dense_embeddings.ids.json").read_text(encoding="utf-8"))
    embeddings = np.load(index_path / "dense_embeddings.npy", mmap_mode="r")
    if len(ids) != embeddings.shape[0]:
        raise ValueError("dense embedding IDs and rows differ")
    doc_rows: dict[str, list[int]] = defaultdict(list)
    for row, chunk_id in enumerate(ids):
        doc_rows[str(chunk_id).split("::", 1)[0]].append(row)
    return DenseBank(embeddings=embeddings, doc_rows={key: np.asarray(value) for key, value in doc_rows.items()})


def _encode_aspects(
    questions: Sequence[str],
    aspect_lists: Sequence[Sequence[str]],
    model_path: Path,
    device: str,
    batch_size: int,
) -> list[np.ndarray]:
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("Aspect-MaxSim requires torch and sentence-transformers") from exc
    actual_device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(str(model_path), device=actual_device)
    model.max_seq_length = 256
    flat = ["query: " + aspect for aspects in aspect_lists for aspect in aspects]
    encoded = model.encode(
        flat,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    output: list[np.ndarray] = []
    offset = 0
    for aspects in aspect_lists:
        count = len(aspects)
        output.append(np.asarray(encoded[offset : offset + count], dtype=np.float32))
        offset += count
    print(f"Aspect-MaxSim encoder device: {actual_device}; aspect vectors: {len(flat)}", flush=True)
    return output


ASPECT_FEATURE_NAMES = [
    "v12_aspect_count", "v12_aspect_max_mean", "v12_aspect_max_min",
    "v12_aspect_max_max", "v12_aspect_max_std", "v12_aspect_cover70",
    "v12_aspect_cover80", "v12_aspect_weakest_margin", "v12_dense_chunk_count",
    *(f"v12_aspect_slot_{index}" for index in range(1, 7)),
]


def aspect_maxsim_features(
    split: AuditSplit,
    index_path: Path,
    model_path: Path,
    candidate_limit: int,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, list[str], list[list[str]]]:
    bank = load_dense_bank(index_path)
    aspects = [parse_aspects(question) for question in split.questions]
    vectors = _encode_aspects(split.questions, aspects, model_path, device, batch_size)
    rows: list[list[float]] = []
    for question_index, candidates in enumerate(split.candidates):
        selected = list(candidates[:candidate_limit])
        if len(selected) < candidate_limit:
            selected.extend([""] * (candidate_limit - len(selected)))
        query_vectors = vectors[question_index]
        for doc_id in selected:
            chunk_rows = bank.doc_rows.get(doc_id)
            if chunk_rows is None or not len(query_vectors):
                values = np.zeros(6, dtype=np.float32)
                chunk_count = 0
            else:
                chunk_matrix = np.asarray(bank.embeddings[chunk_rows], dtype=np.float32)
                values = np.max(query_vectors @ chunk_matrix.T, axis=1)
                chunk_count = len(chunk_rows)
            mean = float(np.mean(values)) if len(values) else 0.0
            minimum = float(np.min(values)) if len(values) else 0.0
            maximum = float(np.max(values)) if len(values) else 0.0
            standard_deviation = float(np.std(values)) if len(values) else 0.0
            row = [
                len(query_vectors) / 6.0,
                mean,
                minimum,
                maximum,
                standard_deviation,
                float(np.mean(values >= 0.70)) if len(values) else 0.0,
                float(np.mean(values >= 0.80)) if len(values) else 0.0,
                minimum - mean,
                math.log1p(chunk_count),
            ]
            row.extend([float(values[index]) if index < len(values) else 0.0 for index in range(6)])
            rows.append(row)
    return np.asarray(rows, dtype=np.float32), ASPECT_FEATURE_NAMES, aspects


@dataclass
class TfidfBank:
    ids: list[str]
    word_vectorizer: TfidfVectorizer
    char_vectorizer: TfidfVectorizer
    word_matrix: Any
    char_matrix: Any


def fit_tfidf(documents: Mapping[str, Any], max_char_features: int = 60_000) -> TfidfBank:
    ids = sorted(documents)
    # Character n-grams are an auxiliary ranker signal.  Capping the scanned
    # prefix and vocabulary keeps peak RAM bounded while retaining headings,
    # article numbers, and the opening legal clauses that drive retrieval.
    texts = [
        documents[doc_id].title + "\n" + documents[doc_id].text[:12_000]
        for doc_id in ids
    ]
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), sublinear_tf=True, min_df=1, dtype=np.float32,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
        min_df=3, max_features=max_char_features, dtype=np.float32,
    )
    print(f"TF-IDF fitting {len(texts)} documents (char cap={max_char_features})", flush=True)
    word_matrix = word_vectorizer.fit_transform(texts)
    char_matrix = char_vectorizer.fit_transform(texts)
    print(
        f"TF-IDF corpus={len(ids)} word_vocab={len(word_vectorizer.vocabulary_)} "
        f"char_vocab={len(char_vectorizer.vocabulary_)}",
        flush=True,
    )
    return TfidfBank(ids, word_vectorizer, char_vectorizer, word_matrix, char_matrix)


TFIDF_FEATURE_NAMES = [
    "v12_tfidf_word", "v12_tfidf_char", "v12_tfidf_mean", "v12_tfidf_max",
    "v12_tfidf_word_char_gap", "v12_tfidf_query_number_overlap",
]


def tfidf_features(
    split: AuditSplit,
    bank: TfidfBank,
    documents: Mapping[str, Any],
    candidate_limit: int,
) -> np.ndarray:
    id_to_row = {doc_id: index for index, doc_id in enumerate(bank.ids)}
    normalized_questions = [normalize(question) for question in split.questions]
    word_queries = bank.word_vectorizer.transform(normalized_questions)
    char_queries = bank.char_vectorizer.transform(normalized_questions)
    # One sparse-dense multiplication per split is substantially faster than
    # repeating the same corpus traversal for every query.
    word_scores_all = (word_queries @ bank.word_matrix.T).toarray()
    char_scores_all = (char_queries @ bank.char_matrix.T).toarray()
    document_numbers = {
        doc_id: set(NUMBER_RE.findall(documents[doc_id].text))
        for doc_id in documents
    }
    rows: list[list[float]] = []
    for question_index, (question, candidates) in enumerate(zip(split.questions, split.candidates)):
        normalized = normalized_questions[question_index]
        word_scores = word_scores_all[question_index]
        char_scores = char_scores_all[question_index]
        query_numbers = set(NUMBER_RE.findall(normalized))
        selected = list(candidates[:candidate_limit])
        if len(selected) < candidate_limit:
            selected.extend([""] * (candidate_limit - len(selected)))
        for doc_id in selected:
            index = id_to_row.get(doc_id)
            word = float(word_scores[index]) if index is not None else 0.0
            char = float(char_scores[index]) if index is not None else 0.0
            doc_numbers = document_numbers.get(doc_id, set())
            # Number overlap remains a separate, interpretable signal; the
            # existing v11 feature carries the document-side count.
            rows.append([
                word, char, (word + char) / 2.0, max(word, char), word - char,
                float(bool(query_numbers and doc_numbers and query_numbers & doc_numbers)),
            ])
    return np.asarray(rows, dtype=np.float32)


EVIDENCE_FEATURE_NAMES = [
    "v12_evidence_count", "v12_evidence_query_coverage", "v12_evidence_naive_coverage",
    "v12_evidence_coverage_gain", "v12_evidence_mean_score", "v12_evidence_max_score",
    "v12_evidence_min_score", "v12_evidence_aspect_mean", "v12_evidence_aspect_min",
    "v12_evidence_aspect_max", "v12_evidence_redundancy", "v12_evidence_title_hits",
    "v12_evidence_chunk_diversity", "v12_evidence_fraction",
]


def evidence_features(
    split: AuditSplit,
    chunk_evidence: Mapping[str, Sequence[ChunkEvidence]],
    aspects: Sequence[Sequence[str]],
    document_frequency: Mapping[str, int],
    corpus_size: int,
    candidate_limit: int,
) -> np.ndarray:
    rows: list[list[float]] = []
    for question_index, candidates in enumerate(split.candidates):
        selected = list(candidates[:candidate_limit])
        if len(selected) < candidate_limit:
            selected.extend([""] * (candidate_limit - len(selected)))
        for doc_id in selected:
            rows.append(
                _evidence_row(
                    chunk_evidence.get(doc_id, ()),
                    split.questions[question_index],
                    aspects[question_index],
                    document_frequency,
                    corpus_size,
                )
            )
        if (question_index + 1) % 250 == 0:
            print(f"evidence/{split.name}: {question_index + 1}/{len(split.qids)}", flush=True)
    return np.asarray(rows, dtype=np.float32)


def build_research_features(
    split: AuditSplit,
    documents: Mapping[str, Any],
    document_frequency: Mapping[str, int],
    index_path: Path,
    model_path: Path,
    candidate_limit: int,
    device: str,
    batch_size: int,
    tfidf_bank: TfidfBank,
    chunk_evidence: Mapping[str, Sequence[ChunkEvidence]],
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    base, target, base_names = build_query_normalized_features(
        split, documents, document_frequency, candidate_limit
    )
    aspect_matrix, aspect_names, aspects = aspect_maxsim_features(
        split, index_path, model_path, candidate_limit, device, batch_size
    )
    evidence_matrix = evidence_features(
        split, chunk_evidence, aspects, document_frequency, len(documents), candidate_limit
    )
    tfidf_matrix = tfidf_features(split, tfidf_bank, documents, candidate_limit)
    matrix = np.concatenate([base, aspect_matrix, evidence_matrix, tfidf_matrix], axis=1)
    names = list(base_names) + list(aspect_names) + EVIDENCE_FEATURE_NAMES + TFIDF_FEATURE_NAMES
    return matrix, target, names


def _references(split: AuditSplit, indices: Sequence[int]) -> dict[str, dict[str, list[str]]]:
    if split.references is None:
        raise ValueError("references unavailable")
    return {split.qids[index]: {"answer": sorted(split.references[index])} for index in indices}


def make_predictions(
    split: AuditSplit,
    indices: Sequence[int],
    scores: np.ndarray,
    candidate_limit: int,
) -> dict[str, dict[str, list[str]]]:
    output: dict[str, dict[str, list[str]]] = {}
    for local, question_index in enumerate(indices):
        candidates = list(dict.fromkeys(split.candidates[question_index][:candidate_limit]))
        order = np.argsort(-scores[local], kind="stable")
        answers = [candidates[index] for index in order if index < len(candidates) and candidates[index]][:5]
        output[split.qids[question_index]] = {"answer": answers}
    return output


def evaluate_predictions(
    split: AuditSplit,
    predictions: Mapping[str, Mapping[str, Sequence[str]]],
    indices: Sequence[int],
) -> dict[str, Any]:
    metrics = dict(official_metrics(dict(predictions), _references(split, indices)))
    return metrics


def make_ranker(seed: int) -> Any:
    from lightgbm import LGBMRanker

    return LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[5],
        truncation_level=8,
        num_leaves=15,
        max_depth=5,
        learning_rate=0.03,
        n_estimators=400,
        min_child_samples=30,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        reg_lambda=2.0,
        random_state=seed,
        verbosity=-1,
    )


def fit_ranker(model: Any, matrix: np.ndarray, target: np.ndarray, questions: Sequence[int], candidate_limit: int) -> Any:
    rows = row_indices(questions, candidate_limit)
    model.fit(matrix[rows], target[rows], group=[candidate_limit] * len(questions))
    return model


def rank_scores(model: Any, matrix: np.ndarray, questions: Sequence[int], candidate_limit: int) -> np.ndarray:
    rows = row_indices(questions, candidate_limit)
    return np.asarray(model.predict(matrix[rows]), dtype=np.float32).reshape(-1, candidate_limit)


def grouped_oof(
    split: AuditSplit,
    matrix: np.ndarray,
    target: np.ndarray,
    selection_indices: np.ndarray,
    groups: np.ndarray,
    candidate_limit: int,
    seed: int,
) -> tuple[np.ndarray, dict[int, int]]:
    output = np.full((len(selection_indices), candidate_limit), np.nan, dtype=np.float32)
    fold_map: dict[int, int] = {}
    splitter = GroupKFold(n_splits=5)
    for fold, (train_local, valid_local) in enumerate(
        splitter.split(selection_indices, groups=groups[selection_indices])
    ):
        train_questions = selection_indices[train_local]
        valid_questions = selection_indices[valid_local]
        model = fit_ranker(make_ranker(seed + fold), matrix, target, train_questions, candidate_limit)
        output[valid_local] = rank_scores(model, matrix, valid_questions, candidate_limit)
        for question in valid_questions:
            fold_map[int(question)] = fold
        print(f"LambdaMART OOF fold {fold + 1}/5: train={len(train_questions)} valid={len(valid_questions)}", flush=True)
    return output, fold_map


def evaluate_oof(
    split: AuditSplit,
    scores: np.ndarray,
    indices: Sequence[int],
    candidate_limit: int,
    fold_map: Mapping[int, int],
) -> dict[str, Any]:
    predictions = make_predictions(split, indices, scores, candidate_limit)
    result = evaluate_predictions(split, predictions, indices)
    fold_recalls = []
    for fold in sorted(set(fold_map[index] for index in indices)):
        fold_indices = [index for index in indices if fold_map[index] == fold]
        fold_predictions = {split.qids[index]: predictions[split.qids[index]] for index in fold_indices}
        fold_recalls.append(float(evaluate_predictions(split, fold_predictions, fold_indices)["recall"]))
    result["fold_recalls"] = fold_recalls
    result["fold_recall_std"] = statistics.pstdev(fold_recalls) if len(fold_recalls) > 1 else 0.0
    result["robust_recall"] = float(result["recall"]) - 0.25 * result["fold_recall_std"]
    return result


def baseline_metrics(split: AuditSplit, indices: Sequence[int]) -> dict[str, Any]:
    predictions = {split.qids[index]: {"answer": split.baseline[index]} for index in indices}
    return evaluate_predictions(split, predictions, indices)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts-zip", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--holdout-stage1", type=Path, required=True)
    parser.add_argument("--holdout-stage2", type=Path, required=True)
    parser.add_argument("--holdout-reference", type=Path, required=True)
    parser.add_argument("--public-stage1", type=Path, required=True)
    parser.add_argument("--public-stage2", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-limit", type=int, default=35)
    parser.add_argument("--final-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.candidate_limit < 5:
        raise ValueError("candidate-limit must be at least 5")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    holdout = load_audit_split("holdout", args.holdout_stage1, args.holdout_stage2, args.holdout_reference)
    public = load_audit_split("public", args.public_stage1, args.public_stage2, None)
    needed_ids = {
        doc_id
        for split in (holdout, public)
        for candidates in split.candidates
        for doc_id in candidates[: args.candidate_limit]
        if doc_id
    }
    needed_ids.update(doc_id for split in (holdout, public) for answers in split.baseline for doc_id in answers)
    index_dir = args.index if args.index.is_dir() else args.index.parent
    index_db = index_dir / "legalir.sqlite3"
    priors = load_priors(index_db)
    documents, document_frequency = load_documents_capped(args.contexts_zip, needed_ids, priors)
    chunks = load_chunk_evidence(index_db, needed_ids)
    tfidf_bank = fit_tfidf(documents)
    holdout_matrix, holdout_target, feature_names = build_research_features(
        holdout, documents, document_frequency, index_dir, args.model,
        args.candidate_limit, args.device, args.batch_size, tfidf_bank, chunks,
    )
    public_matrix, _, public_names = build_research_features(
        public, documents, document_frequency, index_dir, args.model,
        args.candidate_limit, args.device, args.batch_size, tfidf_bank, chunks,
    )
    if feature_names != public_names or holdout_target is None:
        raise RuntimeError("v12 feature schema mismatch or missing holdout labels")
    groups = build_leak_groups(holdout)
    all_indices = np.arange(len(holdout.qids), dtype=np.int32)
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.final_fraction, random_state=args.seed)
    selection_local, final_local = next(splitter.split(all_indices, groups=groups))
    selection_indices = all_indices[selection_local]
    final_indices = all_indices[final_local]
    selection_docs = set().union(*(holdout.references[index] for index in selection_indices))
    final_docs = set().union(*(holdout.references[index] for index in final_indices))
    if selection_docs & final_docs:
        raise RuntimeError("locked final split shares reference documents")
    print(
        f"locked split: selection={len(selection_indices)} final={len(final_indices)} "
        f"groups={len(set(groups))} reference_overlap=0 features={len(feature_names)}",
        flush=True,
    )
    oof_scores, fold_map = grouped_oof(
        holdout, holdout_matrix, holdout_target, selection_indices, groups,
        args.candidate_limit, args.seed,
    )
    oof = evaluate_oof(holdout, oof_scores, selection_indices, args.candidate_limit, fold_map)
    selection_baseline = baseline_metrics(holdout, selection_indices)
    locked_model = fit_ranker(
        make_ranker(args.seed + 101), holdout_matrix, holdout_target,
        selection_indices, args.candidate_limit,
    )
    final_scores = rank_scores(locked_model, holdout_matrix, final_indices, args.candidate_limit)
    final_predictions = make_predictions(holdout, final_indices, final_scores, args.candidate_limit)
    final = evaluate_predictions(holdout, final_predictions, final_indices)
    final_baseline = baseline_metrics(holdout, final_indices)
    promoted = (
        float(oof["recall"]) >= float(selection_baseline["recall"])
        and float(final["recall"]) >= float(final_baseline["recall"])
    )
    public_output: Path | None = None
    public_changes = 0
    if promoted:
        final_model = fit_ranker(
            make_ranker(args.seed + 202), holdout_matrix, holdout_target,
            all_indices, args.candidate_limit,
        )
        public_scores = rank_scores(final_model, public_matrix, np.arange(len(public.qids)), args.candidate_limit)
        public_predictions = make_predictions(public, np.arange(len(public.qids)), public_scores, args.candidate_limit)
        public_output = args.output_dir / "v12_lambdamart_candidate.json"
        write_json(public_output, public_predictions)
        public_changes = sum(
            public_predictions[qid]["answer"] != public.baseline[index]
            for index, qid in enumerate(public.qids)
        )
    report = {
        "kind": "legalir_v12_research_aspect_evidence_tfidf_lambdamart",
        "leakage_policy": {
            "stage1_fixed": True,
            "public_labels_used": False,
            "v57_predictions_used": False,
            "document_prior_used": False,
            "fit_audit_used_for_supervision": False,
            "grouping": "v11 shared-reference and question-similarity groups",
        },
        "candidate_limit": args.candidate_limit,
        "features": feature_names,
        "feature_blocks": {
            "v11_query_normalized": len(feature_names) - len(ASPECT_FEATURE_NAMES) - len(EVIDENCE_FEATURE_NAMES) - len(TFIDF_FEATURE_NAMES),
            "aspect_maxsim": list(ASPECT_FEATURE_NAMES),
            "evidence_set": list(EVIDENCE_FEATURE_NAMES),
            "tfidf": list(TFIDF_FEATURE_NAMES),
        },
        "locked_config": {
            "objective": "lambdarank",
            "eval_at": [5], "truncation_level": 8, "num_leaves": 15,
            "max_depth": 5, "learning_rate": 0.03, "n_estimators": 400,
            "min_child_samples": 30, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 1, "reg_lambda": 2.0,
        },
        "split": {
            "questions": len(holdout.qids), "selection_questions": len(selection_indices),
            "final_questions": len(final_indices), "groups": len(set(groups)),
            "reference_overlap": 0, "seed": args.seed,
        },
        "gates": {
            "selection_baseline": selection_baseline,
            "selection_oof_lambdamart": oof,
            "final_baseline": final_baseline,
            "final_lambdamart": final,
            "promoted": promoted,
            "aspect_target_0925": float(oof["recall"]) >= 0.925,
            "evidence_target_0930": float(oof["recall"]) >= 0.930,
            "lambdamart_target_0935": float(oof["recall"]) >= 0.935,
        },
        "promotion": {
            "public_candidate": str(public_output.resolve()) if public_output else None,
            "public_queries_changed": public_changes,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = args.output_dir / "v12_research_report.json"
    write_json(report_path, report)
    print(json.dumps({
        "report": str(report_path.resolve()), "selection_oof": oof,
        "final": final, "final_baseline": final_baseline,
        "promoted": promoted, "public_candidate": str(public_output) if public_output else None,
        "elapsed_seconds": report["elapsed_seconds"],
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
