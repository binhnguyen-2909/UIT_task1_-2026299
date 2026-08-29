"""V14 replacement-target pipeline around the frozen v11h LegalIR anchor.

This runner implements the algorithm-first roadmap after the replacement
oracle has shown that one-slot rescue has material upside.  It never globally
reranks v11h.  For every query it considers only Stage-1 ranks 6--35 and may
replace only answer slot five.

The selector target is metric-aligned:

* 1: replacing the v11h tail with this candidate increases Recall@5;
* 0: the replacement is neutral or destructive.

All model and threshold selection uses the complement split.  Standard,
law-family and query-template grouped OOF gates must pass before the locked
holdout is scored.  The deployed model is a five-fold complement ensemble;
the locked holdout and public labels are never used for training or tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import sqlite3
import statistics
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import joblib
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .evaluate import official_metrics
    from .legal_structure import iter_legal_chunks
    from .v11_exact_heading_rescue import normalize
    from .v14_safe_rescue import (
        AuditSplit,
        CandidateFeatures,
        Document,
        Passage,
        build_candidate_features,
        intervention_report,
        load_audit_split,
        parse_constraints,
        replacement_oracle_report,
        validate_model_budget,
    )
except ImportError:  # pragma: no cover - direct script execution
    from evaluate import official_metrics
    from legal_structure import iter_legal_chunks
    from v11_exact_heading_rescue import normalize
    from v14_safe_rescue import (
        AuditSplit,
        CandidateFeatures,
        Document,
        Passage,
        build_candidate_features,
        intervention_report,
        load_audit_split,
        parse_constraints,
        replacement_oracle_report,
        validate_model_budget,
    )


SEED = 20260828
TOKEN_RE = re.compile(r"[a-z0-9]+")
NUMBER_RE = re.compile(r"\d+")

# These are frozen v11 features, not newly inferred model outputs.  They cover
# retrieval rank, current CE, exact lexical/numeric matching and consensus.
BASE_FEATURES = (
    "rank", "inv_rank", "log_rank", "rank_fraction", "top10", "top20", "top35",
    "baseline_present", "baseline_inv_rank",
    "neural_present", "neural_signed_log", "neural_inv_rank", "neural_rank_fraction",
    "stage1_score_known", "stage1_score", "stage1_dense", "stage1_chunk_phrase", "stage1_citation",
    "query_log_tokens", "document_log_tokens", "title_log_tokens",
    "query_recall_document", "document_precision", "jaccard_document", "idf_recall_document",
    "query_recall_title", "title_precision", "jaccard_title", "idf_recall_title",
    "longest_phrase_tokens", "longest_phrase_fraction", "query_pair_token_coverage",
    "number_recall", "number_precision", "number_exact", "query_has_number",
    "exact_query_in_document", "source_inv_rank", "source_rank_fraction",
    "prf_new_candidate", "prf_rank_gain", "neural_query_zscore", "neural_query_minmax",
    "neural_margin_from_best_log", "neural_margin_from_fifth_log",
    "neural_top1", "neural_top3", "neural_top5",
    "rank_consensus_rr", "top5_vote_fraction", "longest_phrase_ge10",
    "longest_title_phrase_tokens", "longest_title_phrase_fraction", "exact_query_in_title",
    "query_document_log_length_ratio", "body_title_recall_gap",
)

PREFILTER_SIGNALS = (
    "neural_query_zscore",
    "idf_recall_document",
    "query_recall_document",
    "query_recall_title",
    "longest_phrase_tokens",
    "number_exact",
    "rank_consensus_rr",
    "top5_vote_fraction",
)

NEW_FEATURES = (
    "selector_score", "element_mean", "element_min", "element_std", "critical_min",
    "coverage_ratio", "rare_coverage", "modality", "contradiction", "number_exact",
    "structure", "lexical", "independent_signals", "evidence_score",
    "evidence_coverage", "evidence_critical_coverage", "evidence_hierarchy",
    "evidence_modality", "evidence_number", "evidence_redundancy",
    "evidence_contradiction", "evidence_passages",
)


@dataclass(frozen=True)
class RawDocument:
    doc_id: str
    title: str
    text: str


@dataclass
class PairDataset:
    split_name: str
    X: np.ndarray
    target: np.ndarray
    outcome: np.ndarray
    question_index: np.ndarray
    candidate_id: np.ndarray
    safe_gate: np.ndarray
    feature_names: tuple[str, ...]
    prefilter: dict[str, Any]


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _rank(values: Sequence[float], positions: Sequence[int]) -> dict[int, int]:
    order = sorted(range(len(values)), key=lambda index: (-float(values[index]), positions[index]))
    return {index: rank for rank, index in enumerate(order, 1)}


def _feature_columns(split: AuditSplit) -> tuple[list[int], tuple[str, ...]]:
    columns = {name: index for index, name in enumerate(split.feature_names)}
    missing = [name for name in BASE_FEATURES if name not in columns]
    if missing:
        raise ValueError(f"v11 feature cache is missing: {missing}")
    return [columns[name] for name in BASE_FEATURES], BASE_FEATURES


def _candidate_index(split: AuditSplit, question: int) -> dict[str, int]:
    return {doc_id: index for index, doc_id in enumerate(split.candidates[question])}


def prefilter_candidates(
    split: AuditSplit,
    question: int,
    candidate_limit: int,
    maximum: int,
    top_per_signal: int = 3,
) -> tuple[list[str], dict[str, Any]]:
    """High-recall, label-free union of independent frozen signals."""

    candidates = split.candidates[question][:candidate_limit]
    anchor = set(split.anchor[split.qids[question]]["answer"])
    eligible = [index for index in range(5, len(candidates)) if candidates[index] not in anchor]
    if not eligible:
        return [], {"signal_votes": {}, "aggregate": {}}
    columns = {name: index for index, name in enumerate(split.feature_names)}
    block = split.feature_blocks[question]
    votes: Counter[str] = Counter()
    aggregate: defaultdict[str, float] = defaultdict(float)
    signal_top: dict[str, list[str]] = {}

    # Stage-1 rank and the two learned v11 signals are represented explicitly.
    signal_values: dict[str, list[float]] = {
        "v11_selector": [float(split.v11_scores[question].get(candidates[index], -1e9)) for index in eligible],
        "cross_encoder": [float(split.neural_scores[question].get(candidates[index], -1e9)) for index in eligible],
        "stage1_rank": [-float(index) for index in eligible],
    }
    for name in PREFILTER_SIGNALS:
        signal_values[name] = [float(block[index, columns[name]]) for index in eligible]

    for name, values in signal_values.items():
        finite = np.asarray(values, dtype=np.float64)
        finite[~np.isfinite(finite)] = -1e9
        low, high = float(np.min(finite)), float(np.max(finite))
        scaled = (finite - low) / (high - low) if high > low else np.zeros_like(finite)
        ordered = sorted(range(len(eligible)), key=lambda local: (-float(finite[local]), eligible[local]))
        chosen = [candidates[eligible[local]] for local in ordered[:top_per_signal]]
        signal_top[name] = chosen
        for local, index in enumerate(eligible):
            doc_id = candidates[index]
            aggregate[doc_id] += float(scaled[local])
        for doc_id in chosen:
            votes[doc_id] += 1

    ordered_docs = sorted(
        (candidates[index] for index in eligible),
        key=lambda doc_id: (-votes[doc_id], -aggregate[doc_id], candidates.index(doc_id), doc_id),
    )
    selected = ordered_docs[:maximum]
    return selected, {
        "signal_top": signal_top,
        "signal_votes": {doc_id: int(votes[doc_id]) for doc_id in selected},
        "aggregate": {doc_id: float(aggregate[doc_id]) for doc_id in selected},
    }


def prefilter_oracle_report(
    split: AuditSplit,
    candidate_limit: int,
    maximum: int,
) -> tuple[dict[str, Any], list[list[str]], dict[str, Any]]:
    selected_rows: list[list[str]] = []
    details: dict[str, Any] = {}
    eligible = captured = 0
    captured_ranks: list[int] = []
    for question, qid in enumerate(split.qids):
        selected, trace = prefilter_candidates(split, question, candidate_limit, maximum)
        selected_rows.append(selected)
        details[qid] = trace
        if split.references is None:
            continue
        gold = set(split.references[qid]["answer"])
        anchor = split.anchor[qid]["answer"]
        candidate_row = split.candidates[question][:candidate_limit]
        rescuable = [
            doc_id for rank, doc_id in enumerate(candidate_row, 1)
            if 6 <= rank <= candidate_limit and doc_id in gold and doc_id not in anchor
        ]
        if rescuable and anchor[-1] not in gold:
            eligible += 1
            hits = [doc_id for doc_id in rescuable if doc_id in selected]
            if hits:
                captured += 1
                captured_ranks.append(min(selected.index(doc_id) + 1 for doc_id in hits))
    report = {
        "maximum_candidates": maximum,
        "rescuable_queries": eligible,
        "captured_queries": captured,
        "capture_rate": captured / eligible if eligible else 1.0,
        "median_prefilter_rank": statistics.median(captured_ranks) if captured_ranks else None,
    }
    return report, selected_rows, details


def load_raw_documents(contexts_zip: Path, needed_ids: set[str]) -> dict[str, RawDocument]:
    documents: dict[str, RawDocument] = {}
    with zipfile.ZipFile(contexts_zip) as archive:
        for info in archive.infolist():
            if not info.filename.endswith(".json"):
                continue
            item = json.loads(archive.read(info))
            doc_id = str(item["id"])
            if doc_id in needed_ids:
                documents[doc_id] = RawDocument(doc_id, str(item.get("name", "")), str(item.get("passage", "")))
    missing = needed_ids - set(documents)
    if missing:
        raise ValueError(f"contexts archive is missing {len(missing)} required documents")
    return documents


def load_term_stats(index_path: Path) -> tuple[Counter[str], int]:
    uri = index_path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        frequency = Counter({str(term): int(value) for term, value in connection.execute("SELECT term,doc_freq FROM term_stats")})
        corpus_size = int(connection.execute("SELECT count(*) FROM chunks").fetchone()[0])
    finally:
        connection.close()
    return frequency, corpus_size


def load_document_titles(index_path: Path) -> dict[str, str]:
    uri = index_path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        return {str(doc_id): str(title) for doc_id, title in connection.execute("SELECT doc_id,title FROM document_meta")}
    finally:
        connection.close()


CLAUSE_RE = re.compile(r"(?:^|\s)(\d{1,3})[.)]\s+")
POINT_RE = re.compile(r"(?:^|\s)([a-zđ])[.)]\s+", re.IGNORECASE)


def _candidate_feature_values(features: CandidateFeatures) -> np.ndarray:
    evidence = features.evidence
    return np.asarray(
        [
            features.selector_score,
            features.element_mean,
            features.element_min,
            features.element_std,
            features.critical_min,
            features.coverage_ratio,
            features.rare_coverage,
            features.modality,
            features.contradiction,
            features.number_exact,
            features.structure,
            features.lexical,
            float(features.independent_signals),
            evidence.score,
            evidence.coverage,
            evidence.critical_coverage,
            evidence.hierarchy,
            evidence.modality,
            evidence.number,
            evidence.redundancy,
            evidence.contradiction,
            float(len(evidence.passages)),
        ],
        dtype=np.float32,
    )


QUESTION_FEATURES = (
    "question_log_tokens", "constraint_count", "critical_count", "has_actor",
    "has_condition", "has_exception", "has_number", "has_structure_reference",
    "modality_prohibition", "modality_obligation", "modality_permission_question",
    "modality_permission",
)


def _question_features(question: str) -> np.ndarray:
    constraints = parse_constraints(question)
    keys = set(constraints.elements)
    return np.asarray(
        [
            math.log1p(len(TOKEN_RE.findall(normalize(question)))),
            float(len(constraints.elements)),
            float(len(constraints.critical)),
            float("actor" in keys),
            float("condition" in keys),
            float("exception" in keys),
            float(bool(constraints.numbers)),
            float(bool(constraints.articles or constraints.clauses or constraints.points)),
            float(constraints.modality == "prohibition"),
            float(constraints.modality == "obligation"),
            float(constraints.modality == "permission_question"),
            float(constraints.modality == "permission"),
        ],
        dtype=np.float32,
    )


def _build_legal_document(raw: RawDocument) -> Document:
    passages: list[Passage] = []
    for index, chunk in enumerate(iter_legal_chunks(raw.text, 1800, 180)):
        clause = CLAUSE_RE.search(chunk.text)
        point = POINT_RE.search(chunk.text)
        passages.append(
            Passage(
                raw.doc_id,
                f"{raw.doc_id}::v14::{index}",
                raw.title,
                chunk.text,
                chunk.unit_type,
                chunk.article_number,
                clause.group(1) if clause else "",
                point.group(1).lower() if point else "",
            )
        )
    if not passages:
        passages.append(Passage(raw.doc_id, f"{raw.doc_id}::fallback", raw.title, raw.text, "fallback"))
    return Document(raw.doc_id, raw.title, raw.text, tuple(passages))


def _model_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for prefix in ("candidate", "tail", "delta"):
        names.extend(f"{prefix}__v11__{name}" for name in BASE_FEATURES)
    for prefix in ("candidate", "tail", "delta"):
        names.extend(f"{prefix}__v14__{name}" for name in NEW_FEATURES)
    names.extend(
        (
            "candidate__v11_selector_score", "tail__v11_selector_score", "delta__v11_selector_score",
            "candidate__cross_encoder_score", "tail__cross_encoder_score", "delta__cross_encoder_score",
            "candidate__stage1_rank", "tail__stage1_rank", "rank_distance",
        )
    )
    names.extend(f"query__{name}" for name in QUESTION_FEATURES)
    return tuple(names)


def _dataset_signature(
    split: AuditSplit,
    contexts_zip: Path,
    index_path: Path,
    candidate_limit: int,
    prefilter_maximum: int,
) -> str:
    script = Path(__file__).resolve()
    payload = {
        "version": 2,
        "split": split.name,
        "qids": split.qids,
        "candidate_limit": candidate_limit,
        "prefilter_maximum": prefilter_maximum,
        "base_features": BASE_FEATURES,
        "new_features": NEW_FEATURES,
        "sources": [
            {"path": str(path.resolve()), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in (contexts_zip, index_path, script)
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _load_pair_cache(path: Path, signature: str) -> PairDataset | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata"].item()))
        if metadata.get("signature") != signature:
            return None
        return PairDataset(
            split_name=str(metadata["split_name"]),
            X=np.asarray(cache["X"], dtype=np.float32),
            target=np.asarray(cache["target"], dtype=np.int8),
            outcome=np.asarray(cache["outcome"]).astype(str),
            question_index=np.asarray(cache["question_index"], dtype=np.int32),
            candidate_id=np.asarray(cache["candidate_id"]).astype(str),
            safe_gate=np.asarray(cache["safe_gate"], dtype=np.bool_),
            feature_names=tuple(metadata["feature_names"]),
            prefilter=dict(metadata["prefilter"]),
        )


def _write_pair_cache(path: Path, signature: str, dataset: PairDataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "signature": signature,
        "split_name": dataset.split_name,
        "feature_names": list(dataset.feature_names),
        "prefilter": dataset.prefilter,
    }
    np.savez_compressed(
        path,
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        X=dataset.X,
        target=dataset.target,
        outcome=dataset.outcome,
        question_index=dataset.question_index,
        candidate_id=dataset.candidate_id,
        safe_gate=dataset.safe_gate,
    )


def build_pair_dataset(
    split: AuditSplit,
    contexts_zip: Path,
    index_path: Path,
    cache_path: Path,
    candidate_limit: int,
    prefilter_maximum: int,
) -> PairDataset:
    """Build candidate-vs-tail rows with exact hierarchical evidence."""

    signature = _dataset_signature(split, contexts_zip, index_path, candidate_limit, prefilter_maximum)
    cached = _load_pair_cache(cache_path, signature)
    if cached is not None:
        print(f"v14 pair cache hit: {cache_path}", flush=True)
        return cached

    prefilter, selected_rows, _ = prefilter_oracle_report(split, candidate_limit, prefilter_maximum)
    needed_ids = {
        doc_id
        for question, qid in enumerate(split.qids)
        for doc_id in [split.anchor[qid]["answer"][-1], *selected_rows[question]]
    }
    print(
        f"v14/{split.name}: loading {len(needed_ids)} raw documents; "
        f"prefilter capture={prefilter['capture_rate']:.4f}",
        flush=True,
    )
    raw_documents = load_raw_documents(contexts_zip, needed_ids)
    document_frequency, corpus_size = load_term_stats(index_path)
    base_columns, _ = _feature_columns(split)
    feature_names = _model_feature_names()

    # Build each legal document only once, then score every query that uses it.
    # Candidate lists heavily overlap; document-major extraction removes most
    # repeated chunking/tokenization while retaining exact query-specific
    # evidence-set optimization.
    query_states: list[dict[str, Any]] = []
    uses: defaultdict[str, list[int]] = defaultdict(list)
    for question, qid in enumerate(split.qids):
        candidates = split.candidates[question][:candidate_limit]
        positions = {doc_id: index for index, doc_id in enumerate(candidates)}
        neural_values = [float(split.neural_scores[question].get(doc_id, 0.0)) for doc_id in candidates]
        state = {
            "qid": qid,
            "candidates": candidates,
            "positions": positions,
            "tail_id": split.anchor[qid]["answer"][-1],
            "proposal_ids": selected_rows[question],
            "neural_values": neural_values,
            "neural_order": _rank(neural_values, list(range(len(candidates)))),
            "constraints": parse_constraints(split.questions[question]),
        }
        query_states.append(state)
        for doc_id in dict.fromkeys([state["tail_id"], *state["proposal_ids"]]):
            uses[doc_id].append(question)

    vectors_by_pair: dict[tuple[int, str], np.ndarray] = {}
    gates_by_pair: dict[tuple[int, str], bool] = {}
    for document_number, doc_id in enumerate(sorted(uses), 1):
        document = _build_legal_document(raw_documents[doc_id])
        for question in uses[doc_id]:
            state = query_states[question]
            candidate_index = state["positions"][doc_id]
            candidate = build_candidate_features(
                split.questions[question],
                candidate_index + 1,
                document,
                state["constraints"],
                state["neural_values"][candidate_index],
                state["neural_order"][candidate_index],
                state["neural_values"],
                document_frequency,
                corpus_size,
            )
            vectors_by_pair[(question, doc_id)] = _candidate_feature_values(candidate)
            gates_by_pair[(question, doc_id)] = bool(
                candidate.evidence.critical_coverage >= 0.25
                and candidate.evidence.score >= 0.25
                and candidate.contradiction <= 0.25
                and candidate.evidence.contradiction <= 0.25
            )
        if document_number % 250 == 0 or document_number == len(uses):
            print(f"v14 evidence/{split.name}: documents {document_number}/{len(uses)}", flush=True)

    rows: list[np.ndarray] = []
    target: list[int] = []
    outcomes: list[str] = []
    row_questions: list[int] = []
    row_candidates: list[str] = []
    gates: list[bool] = []
    positive_gate_total = positive_total = 0
    for question, state in enumerate(query_states):
        qid = state["qid"]
        positions = state["positions"]
        tail_id = state["tail_id"]
        proposal_ids = state["proposal_ids"]
        tail_new = vectors_by_pair[(question, tail_id)]
        tail_index = positions[tail_id]
        base_block = split.feature_blocks[question]
        tail_base = np.asarray(base_block[tail_index, base_columns], dtype=np.float32)
        tail_v11 = float(split.v11_scores[question].get(tail_id, 0.0))
        tail_neural = float(split.neural_scores[question].get(tail_id, 0.0))
        question_row = _question_features(split.questions[question])
        gold = set(split.references[qid]["answer"]) if split.references is not None else set()

        for doc_id in proposal_ids:
            candidate_index = positions[doc_id]
            candidate_new = vectors_by_pair[(question, doc_id)]
            candidate_base = np.asarray(base_block[candidate_index, base_columns], dtype=np.float32)
            candidate_v11 = float(split.v11_scores[question].get(doc_id, 0.0))
            candidate_neural = float(split.neural_scores[question].get(doc_id, 0.0))
            row = np.concatenate(
                (
                    candidate_base,
                    tail_base,
                    candidate_base - tail_base,
                    candidate_new,
                    tail_new,
                    candidate_new - tail_new,
                    np.asarray(
                        [
                            candidate_v11, tail_v11, candidate_v11 - tail_v11,
                            candidate_neural, tail_neural, candidate_neural - tail_neural,
                            float(candidate_index + 1), float(tail_index + 1), float(candidate_index - tail_index),
                        ],
                        dtype=np.float32,
                    ),
                    question_row,
                )
            )
            if len(row) != len(feature_names):
                raise RuntimeError("V14 replacement feature schema mismatch")
            success = bool(split.references is not None and doc_id in gold and tail_id not in gold)
            destructive = bool(split.references is not None and doc_id not in gold and tail_id in gold)
            outcome = "success" if success else "destructive" if destructive else "neutral"
            safe = gates_by_pair[(question, doc_id)]
            positive_total += int(success)
            positive_gate_total += int(success and safe)
            rows.append(row)
            target.append(int(success))
            outcomes.append(outcome)
            row_questions.append(question)
            row_candidates.append(doc_id)
            gates.append(safe)

    prefilter["positive_rows"] = positive_total
    prefilter["positive_rows_passing_safe_gate"] = positive_gate_total
    prefilter["positive_safe_gate_rate"] = positive_gate_total / positive_total if positive_total else 1.0
    dataset = PairDataset(
        split.name,
        np.asarray(rows, dtype=np.float32),
        np.asarray(target, dtype=np.int8),
        np.asarray(outcomes),
        np.asarray(row_questions, dtype=np.int32),
        np.asarray(row_candidates),
        np.asarray(gates, dtype=np.bool_),
        feature_names,
        prefilter,
    )
    _write_pair_cache(cache_path, signature, dataset)
    print(f"v14 pair cache written: {cache_path}", flush=True)
    return dataset


def _remap_roots(union: UnionFind, size: int) -> np.ndarray:
    roots = [union.find(index) for index in range(size)]
    remap = {root: group for group, root in enumerate(sorted(set(roots)))}
    return np.asarray([remap[root] for root in roots], dtype=np.int32)


def standard_leak_groups(split: AuditSplit, similarity_threshold: float = 0.75) -> np.ndarray:
    if split.references is None:
        raise ValueError("standard groups require references")
    union = UnionFind(len(split.qids))
    by_document: defaultdict[str, list[int]] = defaultdict(list)
    for question, qid in enumerate(split.qids):
        for doc_id in split.references[qid]["answer"]:
            by_document[doc_id].append(question)
    for indices in by_document.values():
        for index in indices[1:]:
            union.union(indices[0], index)
    vectors = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True,
    ).fit_transform(split.questions)
    similarities = (vectors @ vectors.T).tocsr()
    similarities.setdiag(0)
    for row in range(similarities.shape[0]):
        start, stop = similarities.indptr[row], similarities.indptr[row + 1]
        for column, value in zip(similarities.indices[start:stop], similarities.data[start:stop]):
            if column > row and float(value) >= similarity_threshold:
                union.union(row, int(column))
    return _remap_roots(union, len(split.qids))


def _law_family(title: str) -> str:
    first = title.split("|", 1)[0]
    tokens = TOKEN_RE.findall(normalize(first))
    if tokens and len(tokens[-1]) >= 6 and tokens[-1].isdigit():
        tokens.pop()
    # Keep the instrument identifier and subject prefix.  Removing article and
    # clause suffixes makes different provisions of one law stay in one fold.
    stop = next((index for index, token in enumerate(tokens) if token in {"dieu", "khoan", "diem"}), len(tokens))
    tokens = tokens[:stop]
    return " ".join(tokens[:18]) or "unknown-law"


def law_family_groups(split: AuditSplit, titles: Mapping[str, str]) -> np.ndarray:
    if split.references is None:
        raise ValueError("law groups require references")
    union = UnionFind(len(split.qids))
    by_family: defaultdict[str, list[int]] = defaultdict(list)
    for question, qid in enumerate(split.qids):
        families = {_law_family(titles.get(doc_id, doc_id)) for doc_id in split.references[qid]["answer"]}
        for family in families:
            by_family[family].append(question)
    for indices in by_family.values():
        for index in indices[1:]:
            union.union(indices[0], index)
    return _remap_roots(union, len(split.qids))


QUESTION_TYPES: tuple[tuple[str, str], ...] = (
    ("permission", r"\bco duoc\b|\bduoc .* hay khong\b"),
    ("definition", r"\bla gi\b|\bduoc hieu la\b"),
    ("duration", r"\bbao lau\b|\bthoi han\b|\bthoi hieu\b"),
    ("quantity", r"\bbao nhieu\b|\bmuc\b"),
    ("dossier", r"\bho so\b|\bbao gom nhung gi\b|\bgom nhung\b"),
    ("condition", r"\bdieu kien\b|\btruong hop nao\b|\bkhi nao\b"),
    ("authority", r"\bco quan nao\b|\bai co tham quyen\b|\bdo ai\b"),
    ("procedure", r"\bthu tuc\b|\bnhu the nao\b|\bthuc hien ra sao\b"),
    ("penalty", r"\bmuc phat\b|\bxu phat\b|\bphat bao nhieu\b"),
)


def query_template_signature(question: str) -> str:
    text = normalize(question)
    question_type = next((name for name, pattern in QUESTION_TYPES if re.search(pattern, text)), "other")
    constraints = parse_constraints(question)
    structural = "+".join(
        key for key in ("actor", "action", "object", "condition", "exception", "number", "article", "clause", "point")
        if key in constraints.elements
    ) or "topic"
    return f"{question_type}|{constraints.modality}|{structural}"


def template_groups(split: AuditSplit) -> np.ndarray:
    signatures = [query_template_signature(question) for question in split.questions]
    remap = {value: index for index, value in enumerate(sorted(set(signatures)))}
    return np.asarray([remap[value] for value in signatures], dtype=np.int32)


MODEL_CONFIGS: tuple[dict[str, Any], ...] = (
    {"kind": "logistic", "C": 0.05},
    {"kind": "logistic", "C": 0.20},
    {"kind": "logistic", "C": 1.00},
    {"kind": "lightgbm", "num_leaves": 7, "max_depth": 3, "n_estimators": 160},
    {"kind": "lightgbm", "num_leaves": 15, "max_depth": 4, "n_estimators": 220},
)


def make_model(config: Mapping[str, Any], seed: int) -> Any:
    if config["kind"] == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(config["C"]), solver="liblinear", max_iter=2000,
                random_state=seed,
            ),
        )
    if config["kind"] == "lightgbm":
        return LGBMClassifier(
            objective="binary",
            n_estimators=int(config["n_estimators"]),
            num_leaves=int(config["num_leaves"]),
            max_depth=int(config["max_depth"]),
            learning_rate=0.03,
            min_child_samples=35,
            subsample=0.85,
            colsample_bytree=0.75,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=seed,
            n_jobs=-1,
            deterministic=True,
            verbosity=-1,
        )
    raise ValueError(f"unknown V14 replacement model: {config}")


def _sample_weights(dataset: PairDataset, row_mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(row_mask)
    questions = dataset.question_index[rows]
    counts = Counter(int(value) for value in questions)
    target = dataset.target[rows]
    positives = max(int(target.sum()), 1)
    negatives = max(len(target) - positives, 1)
    positive_weight = min(40.0, negatives / positives)
    weights = np.asarray([1.0 / counts[int(question)] for question in questions], dtype=np.float64)
    weights *= np.where(target == 1, positive_weight, 1.0)
    selected_outcomes = dataset.outcome[rows]
    weights *= np.where(selected_outcomes == "destructive", 4.0, 1.0)
    weights *= np.where(selected_outcomes == "neutral", 0.75, 1.0)
    return weights / max(float(np.mean(weights)), 1e-9)


def _fit_model(model: Any, dataset: PairDataset, train_questions: np.ndarray) -> Any:
    mask = np.isin(dataset.question_index, train_questions)
    if int(dataset.target[mask].sum()) == 0:
        raise RuntimeError("replacement fold has no successful rescue rows")
    model.fit(dataset.X[mask], dataset.target[mask], **({"sample_weight": _sample_weights(dataset, mask)} if not hasattr(model, "named_steps") else {"logisticregression__sample_weight": _sample_weights(dataset, mask)}))
    return model


def _predict_model(model: Any, matrix: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)


def grouped_oof(
    dataset: PairDataset,
    groups: np.ndarray,
    config: Mapping[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    question_count = int(dataset.question_index.max()) + 1
    question_indices = np.arange(question_count, dtype=np.int32)
    question_target = np.zeros(question_count, dtype=np.int8)
    for question in question_indices:
        question_target[question] = int(np.any(dataset.target[dataset.question_index == question]))
    unique_groups = len(set(map(int, groups)))
    positive_questions = int(question_target.sum())
    folds = min(5, unique_groups, positive_questions)
    if folds < 3:
        raise RuntimeError(f"insufficient grouped positives for OOF: groups={unique_groups}, positives={positive_questions}")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    predictions = np.full(len(dataset.X), np.nan, dtype=np.float64)
    fold_by_question = np.full(question_count, -1, dtype=np.int8)
    models: list[Any] = []
    for fold, (train_local, valid_local) in enumerate(
        splitter.split(question_indices, question_target, groups=groups)
    ):
        train_questions = question_indices[train_local]
        valid_questions = question_indices[valid_local]
        model = _fit_model(make_model(config, seed + fold), dataset, train_questions)
        valid_mask = np.isin(dataset.question_index, valid_questions)
        predictions[valid_mask] = _predict_model(model, dataset.X[valid_mask])
        fold_by_question[valid_questions] = fold
        models.append(model)
        print(
            f"replacement OOF/{config['kind']} fold {fold + 1}/{folds}: "
            f"train_q={len(train_questions)} valid_q={len(valid_questions)}",
            flush=True,
        )
    if np.isnan(predictions).any() or np.any(fold_by_question < 0):
        raise RuntimeError("replacement OOF did not cover every row/question")
    return predictions, fold_by_question, models


def ensemble_predict(models: Sequence[Any], matrix: np.ndarray) -> np.ndarray:
    return np.mean(np.vstack([_predict_model(model, matrix) for model in models]), axis=0)


def predictions_from_probabilities(
    split: AuditSplit,
    dataset: PairDataset,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, Any]]:
    output: dict[str, dict[str, list[str]]] = {}
    trace: dict[str, Any] = {}
    for question, qid in enumerate(split.qids):
        answers = list(split.anchor[qid]["answer"])
        rows = np.flatnonzero((dataset.question_index == question) & dataset.safe_gate)
        if len(rows):
            best = max(rows, key=lambda row: (float(probabilities[row]), -int(row), str(dataset.candidate_id[row])))
            probability = float(probabilities[best])
            candidate = str(dataset.candidate_id[best])
            accepted = probability >= threshold and candidate not in answers
            if accepted:
                answers[-1] = candidate
            trace[qid] = {
                "candidate": candidate,
                "probability": probability,
                "threshold": threshold,
                "accepted": accepted,
                "safe_gate": True,
            }
        else:
            trace[qid] = {"candidate": None, "probability": None, "threshold": threshold, "accepted": False, "safe_gate": False}
        output[qid] = {"answer": answers}
    return output, trace


def evaluate_policy(
    split: AuditSplit,
    dataset: PairDataset,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[dict[str, Any], dict[str, dict[str, list[str]]], dict[str, Any]]:
    if split.references is None:
        raise ValueError("policy evaluation requires references")
    predictions, trace = predictions_from_probabilities(split, dataset, probabilities, threshold)
    baseline = official_metrics(split.anchor, split.references)
    rescued = official_metrics(predictions, split.references)
    result = {
        "baseline": baseline,
        "rescued": rescued,
        "delta": {key: float(rescued[key] - baseline[key]) for key in ("recall", "precision")},
        "intervention": intervention_report(split.anchor, predictions, split.references),
    }
    return result, predictions, trace


def select_threshold(
    split: AuditSplit,
    dataset: PairDataset,
    probabilities: np.ndarray,
    minimum_delta: float = 0.003,
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    top_values: list[float] = []
    for question in range(len(split.qids)):
        rows = np.flatnonzero((dataset.question_index == question) & dataset.safe_gate)
        if len(rows):
            top_values.append(float(np.max(probabilities[rows])))
    if not top_values:
        evaluation, _, _ = evaluate_policy(split, dataset, probabilities, 1.01)
        return 1.01, evaluation, []
    quantiles = np.quantile(np.asarray(top_values), np.linspace(0.0, 1.0, min(241, len(top_values))))
    thresholds = sorted(set(float(value) for value in quantiles) | {0.5, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 1.01})
    leaderboard: list[dict[str, Any]] = []
    for threshold in thresholds:
        evaluation, _, _ = evaluate_policy(split, dataset, probabilities, threshold)
        intervention = evaluation["intervention"]
        eligible = bool(
            evaluation["delta"]["recall"] >= minimum_delta
            and intervention["intervention_precision"] >= 0.85
            and intervention["destructive_rate"] <= 0.03
            and intervention["successful_rescues"] > 0
        )
        leaderboard.append({"threshold": threshold, "evaluation": evaluation, "eligible": eligible})
    eligible_rows = [row for row in leaderboard if row["eligible"]]
    winner = max(
        eligible_rows or leaderboard,
        key=lambda row: (
            row["eligible"],
            row["evaluation"]["rescued"]["recall"],
            row["evaluation"]["intervention"]["intervention_precision"],
            -row["evaluation"]["intervention"]["changes"],
            row["threshold"],
        ),
    )
    top = sorted(
        leaderboard,
        key=lambda row: (
            row["eligible"], row["evaluation"]["rescued"]["recall"],
            row["evaluation"]["intervention"]["intervention_precision"], -row["evaluation"]["intervention"]["changes"],
        ),
        reverse=True,
    )[:20]
    return float(winner["threshold"]), dict(winner["evaluation"]), top


def fold_diagnostics(
    split: AuditSplit,
    predictions: Mapping[str, Mapping[str, Sequence[str]]],
    fold_by_question: np.ndarray,
) -> dict[str, Any]:
    if split.references is None:
        raise ValueError("fold diagnostics require references")
    rows: list[dict[str, Any]] = []
    for fold in sorted(set(map(int, fold_by_question))):
        questions = np.flatnonzero(fold_by_question == fold)
        qids = [split.qids[index] for index in questions]
        subset_predictions = {qid: predictions[qid] for qid in qids}
        subset_anchor = {qid: split.anchor[qid] for qid in qids}
        subset_refs = {qid: split.references[qid] for qid in qids}
        baseline = official_metrics(subset_anchor, subset_refs)
        rescued = official_metrics(subset_predictions, subset_refs)
        rows.append({
            "fold": fold,
            "questions": len(qids),
            "baseline_recall": baseline["recall"],
            "rescued_recall": rescued["recall"],
            "delta_recall": float(rescued["recall"] - baseline["recall"]),
        })
    return {
        "folds": rows,
        "worst_fold_delta": min((row["delta_recall"] for row in rows), default=0.0),
        "delta_std": statistics.pstdev([row["delta_recall"] for row in rows]) if len(rows) > 1 else 0.0,
    }


def model_parameter_count(model: Any) -> int:
    if hasattr(model, "named_steps"):
        estimator = model.named_steps["logisticregression"]
        return int(estimator.coef_.size + estimator.intercept_.size)
    booster = model.booster_
    dump = booster.dump_model()
    leaves = 0
    stack = [tree["tree_structure"] for tree in dump["tree_info"]]
    while stack:
        node = stack.pop()
        if "leaf_value" in node:
            leaves += 1
        else:
            stack.extend((node["left_child"], node["right_child"]))
    return leaves


def _validation_passes(evaluation: Mapping[str, Any], minimum_delta: float) -> bool:
    intervention = evaluation["intervention"]
    return bool(
        evaluation["delta"]["recall"] >= minimum_delta
        and intervention["intervention_precision"] >= 0.85
        and intervention["destructive_rate"] <= 0.03
        and intervention["successful_rescues"] > 0
    )


def _load_split(
    name: str,
    stage1: Path,
    stage2: Path,
    reference: Path | None,
    score_cache: Path,
    feature_cache: Path,
    candidate_limit: int,
    limit: int | None = None,
) -> AuditSplit:
    score_key = {"complement": "oof_scores", "holdout": "holdout_scores", "public": "public_scores"}[name]
    return load_audit_split(
        name,
        stage1,
        stage2,
        score_cache,
        score_key,
        {},
        reference,
        candidate_limit,
        limit,
        feature_cache,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts-zip", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--score-cache", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--complement-stage1", type=Path, required=True)
    parser.add_argument("--complement-stage2", type=Path, required=True)
    parser.add_argument("--complement-reference", type=Path, required=True)
    parser.add_argument("--holdout-stage1", type=Path, required=True)
    parser.add_argument("--holdout-stage2", type=Path, required=True)
    parser.add_argument("--holdout-reference", type=Path, required=True)
    parser.add_argument("--public-stage1", type=Path, required=True)
    parser.add_argument("--public-stage2", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-limit", type=int, default=35)
    parser.add_argument("--prefilter-maximum", type=int, default=18)
    parser.add_argument("--minimum-promote-recall", type=float, default=0.925)
    parser.add_argument("--apply-public", action="store_true")
    parser.add_argument("--limit", type=int, help="Profile-only prefix limit; omit for every gate/report used for promotion")
    parser.add_argument("--dataset-only", action="store_true", help="Build/check the complement pair cache, then stop before model validation")
    parser.add_argument("--prefilter-only", action="store_true", help="Measure label-free shortlist oracle capture and stop before reading document text")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.candidate_limit != 35:
        raise ValueError("V14 frozen policy requires candidate-limit=35")
    if args.prefilter_maximum < 6 or args.prefilter_maximum > 30:
        raise ValueError("prefilter-maximum must be between 6 and 30")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    neural_budget = validate_model_budget()

    complement = _load_split(
        "complement", args.complement_stage1, args.complement_stage2,
        args.complement_reference, args.score_cache, args.feature_cache,
        args.candidate_limit, args.limit,
    )
    holdout = _load_split(
        "holdout", args.holdout_stage1, args.holdout_stage2,
        args.holdout_reference, args.score_cache, args.feature_cache,
        args.candidate_limit, args.limit,
    )
    public = _load_split(
        "public", args.public_stage1, args.public_stage2,
        None, args.score_cache, args.feature_cache, args.candidate_limit, args.limit,
    )
    complement_oracle = replacement_oracle_report(complement, args.candidate_limit)
    holdout_oracle = replacement_oracle_report(holdout, args.candidate_limit)
    expected_holdout = 0.9188333333333334
    observed_holdout = float(holdout_oracle["baseline"]["recall"])
    if args.limit is None and abs(observed_holdout - expected_holdout) > 1e-12:
        raise RuntimeError(
            f"v11h anchor mismatch: expected {expected_holdout:.15f}, got {observed_holdout:.15f}"
        )
    if args.prefilter_only:
        complement_prefilter, _, _ = prefilter_oracle_report(
            complement, args.candidate_limit, args.prefilter_maximum,
        )
        holdout_prefilter, _, _ = prefilter_oracle_report(
            holdout, args.candidate_limit, args.prefilter_maximum,
        )
        prefilter_path = args.output_dir / "v14_prefilter_report.json"
        prefilter_report = {
            "kind": "legalir_v14_label_free_prefilter_oracle",
            "maximum_candidates": args.prefilter_maximum,
            "complement_selection": complement_prefilter,
            "holdout_report_only": holdout_prefilter,
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(prefilter_path, prefilter_report)
        print(json.dumps({"report": str(prefilter_path.resolve()), **prefilter_report}, ensure_ascii=False, indent=2), flush=True)
        return 0

    complement_dataset = build_pair_dataset(
        complement,
        args.contexts_zip,
        args.index,
        args.output_dir / "cache" / "complement_pairs.npz",
        args.candidate_limit,
        args.prefilter_maximum,
    )
    if args.dataset_only:
        profile_path = args.output_dir / "v14_dataset_profile.json"
        profile = {
            "kind": "legalir_v14_pair_dataset_profile",
            "questions": len(complement.qids),
            "rows": len(complement_dataset.X),
            "features": len(complement_dataset.feature_names),
            "target_positive_rows": int(complement_dataset.target.sum()),
            "prefilter": complement_dataset.prefilter,
            "replacement_oracle": complement_oracle,
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(profile_path, profile)
        print(json.dumps({"report": str(profile_path.resolve()), **profile}, ensure_ascii=False, indent=2), flush=True)
        return 0
    titles = load_document_titles(args.index)
    validation_groups = {
        "standard": standard_leak_groups(complement),
        "law_family_ood": law_family_groups(complement, titles),
        "query_template_ood": template_groups(complement),
    }
    group_report = {
        name: {
            "groups": len(set(map(int, groups))),
            "largest_group": max(Counter(map(int, groups)).values()),
        }
        for name, groups in validation_groups.items()
    }

    candidates_report: list[dict[str, Any]] = []
    runtime: dict[str, dict[str, Any]] = {}
    for config_index, config in enumerate(MODEL_CONFIGS):
        config_key = f"{config['kind']}:{config_index}"
        standard_probabilities, standard_folds, standard_models = grouped_oof(
            complement_dataset, validation_groups["standard"], config, SEED + 100 * config_index,
        )
        threshold, standard_evaluation, threshold_leaderboard = select_threshold(
            complement, complement_dataset, standard_probabilities, 0.003,
        )
        standard_predictions, _ = predictions_from_probabilities(
            complement, complement_dataset, standard_probabilities, threshold,
        )
        standard_fold_report = fold_diagnostics(complement, standard_predictions, standard_folds)

        law_probabilities, law_folds, _ = grouped_oof(
            complement_dataset, validation_groups["law_family_ood"], config,
            SEED + 100 * config_index + 31,
        )
        law_evaluation, law_predictions, _ = evaluate_policy(
            complement, complement_dataset, law_probabilities, threshold,
        )
        law_fold_report = fold_diagnostics(complement, law_predictions, law_folds)

        template_probabilities, template_folds, _ = grouped_oof(
            complement_dataset, validation_groups["query_template_ood"], config,
            SEED + 100 * config_index + 67,
        )
        template_evaluation, template_predictions, _ = evaluate_policy(
            complement, complement_dataset, template_probabilities, threshold,
        )
        template_fold_report = fold_diagnostics(complement, template_predictions, template_folds)

        gates = {
            "standard_delta_at_least_0_003": _validation_passes(standard_evaluation, 0.003),
            "law_family_delta_at_least_0_002": _validation_passes(law_evaluation, 0.002),
            "query_template_delta_at_least_0_002": _validation_passes(template_evaluation, 0.002),
            "worst_fold_not_below_minus_0_002": min(
                standard_fold_report["worst_fold_delta"],
                law_fold_report["worst_fold_delta"],
                template_fold_report["worst_fold_delta"],
            ) >= -0.002,
        }
        row = {
            "config": dict(config),
            "threshold": threshold,
            "standard": {"evaluation": standard_evaluation, "folds": standard_fold_report},
            "law_family_ood": {"evaluation": law_evaluation, "folds": law_fold_report},
            "query_template_ood": {"evaluation": template_evaluation, "folds": template_fold_report},
            "gates": gates,
            "eligible": all(gates.values()),
            "threshold_leaderboard": threshold_leaderboard,
        }
        candidates_report.append(row)
        runtime[config_key] = {
            "models": standard_models,
            "probabilities": standard_probabilities,
            "folds": standard_folds,
        }
        print(
            f"V14 config {config_index + 1}/{len(MODEL_CONFIGS)} {config}: "
            f"standard_delta={standard_evaluation['delta']['recall']:+.6f} "
            f"law_delta={law_evaluation['delta']['recall']:+.6f} "
            f"template_delta={template_evaluation['delta']['recall']:+.6f} "
            f"eligible={row['eligible']}",
            flush=True,
        )

    winner = max(
        candidates_report,
        key=lambda row: (
            row["eligible"],
            min(
                row["standard"]["evaluation"]["delta"]["recall"],
                row["law_family_ood"]["evaluation"]["delta"]["recall"],
                row["query_template_ood"]["evaluation"]["delta"]["recall"],
            ),
            row["standard"]["evaluation"]["delta"]["recall"],
            row["standard"]["evaluation"]["intervention"]["intervention_precision"],
        ),
    )
    winner_index = candidates_report.index(winner)
    winner_key = f"{winner['config']['kind']}:{winner_index}"
    winner_models = runtime[winner_key]["models"]
    selector_parameters = sum(model_parameter_count(model) for model in winner_models)

    report: dict[str, Any] = {
        "kind": "legalir_v14_metric_aligned_one_slot_replacement",
        "status": "complement_gate_passed" if winner["eligible"] else "complement_gate_failed",
        "model_budget": {
            **neural_budget,
            "replacement_selector_scalar_parameters": selector_parameters,
            "deployed_total_including_selector": neural_budget["total_parameters"] + selector_parameters,
            "deployed_total_billions": (neural_budget["total_parameters"] + selector_parameters) / 1e9,
            "strictly_under_4b": neural_budget["total_parameters"] + selector_parameters < 4_000_000_000,
        },
        "replacement_oracle": {"complement": complement_oracle, "holdout_report_only": holdout_oracle},
        "prefilter": complement_dataset.prefilter,
        "validation_groups": group_report,
        "winner": winner,
        "model_leaderboard": sorted(
            candidates_report,
            key=lambda row: (
                row["eligible"],
                min(
                    row["standard"]["evaluation"]["delta"]["recall"],
                    row["law_family_ood"]["evaluation"]["delta"]["recall"],
                    row["query_template_ood"]["evaluation"]["delta"]["recall"],
                ),
            ),
            reverse=True,
        ),
        "reused_logic": {
            "v11h_anchor": "Frozen v11h grouped-OOF pointwise selector scores plus longest_phrase_tokens exact-heading t10 from v2_query_norm_c80.npz.",
            "candidate_generation": "Frozen v11h safe-tail Stage-1 c35; no c50, dual-dense or external retrieval expansion.",
            "cross_encoder": "Frozen v11h Vietnamese cross-encoder neural_scores; no reranker fine-tuning in Phase A.",
            "v11_features": "Frozen 89-column v11 query-normalized feature cache; the explicit reused columns are recorded in BASE_FEATURES.",
            "legal_structure": "Existing legal_structure.iter_legal_chunks at 1800 chars/180 overlap; all chunks are considered before exhaustive top-six evidence-set search.",
            "v14_new_logic": "Deterministic typed legal constraints, exhaustive 1-3 passage evidence, modality/numeric/exception checks, metric-aligned candidate-vs-tail classifier and a one-slot safety gate.",
            "selection_labels": "Complement labels only. Holdout is report-only and public has no labels.",
        },
        "architecture": {
            "candidate_limit": args.candidate_limit,
            "prefilter_maximum": args.prefilter_maximum,
            "max_replacements": 1,
            "replacement_slot": 5,
            "replacement_candidate_ranks": [6, 35],
            "evidence_search": "all legal chunks -> top6 -> exhaustive combinations of size 1,2,3",
            "validation": ["standard grouped OOF", "law-family OOD", "query-template OOD", "worst-fold"],
        },
        "holdout": {"opened": False},
        "public": {"applied": False},
    }

    report_path = args.output_dir / "v14_replacement_report.json"
    if not winner["eligible"]:
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(report_path, report)
        print(json.dumps({
            "report": str(report_path.resolve()),
            "status": report["status"],
            "winner": winner,
            "elapsed_seconds": report["elapsed_seconds"],
        }, ensure_ascii=False, indent=2), flush=True)
        return 2

    model_path = args.output_dir / "v14_replacement_fold_ensemble.joblib"
    joblib.dump(
        {
            "models": winner_models,
            "feature_names": complement_dataset.feature_names,
            "threshold": winner["threshold"],
            "config": winner["config"],
            "candidate_limit": args.candidate_limit,
            "prefilter_maximum": args.prefilter_maximum,
        },
        model_path,
    )

    holdout_dataset = build_pair_dataset(
        holdout,
        args.contexts_zip,
        args.index,
        args.output_dir / "cache" / "holdout_pairs.npz",
        args.candidate_limit,
        args.prefilter_maximum,
    )
    if holdout_dataset.feature_names != complement_dataset.feature_names:
        raise RuntimeError("complement/holdout replacement feature schemas differ")
    holdout_probabilities = ensemble_predict(winner_models, holdout_dataset.X)
    holdout_evaluation, holdout_predictions, holdout_trace = evaluate_policy(
        holdout, holdout_dataset, holdout_probabilities, float(winner["threshold"]),
    )
    holdout_intervention = holdout_evaluation["intervention"]
    promoted = bool(
        holdout_evaluation["rescued"]["recall"] >= args.minimum_promote_recall
        and holdout_evaluation["delta"]["recall"] > 0.0
        and holdout_intervention["intervention_precision"] >= 0.85
        and holdout_intervention["destructive_rate"] <= 0.03
    )
    report["status"] = "holdout_promoted" if promoted else "holdout_rejected"
    report["holdout"] = {
        "opened": True,
        "frozen_threshold": winner["threshold"],
        "evaluation": holdout_evaluation,
        "prefilter": holdout_dataset.prefilter,
        "minimum_promote_recall": args.minimum_promote_recall,
        "promoted": promoted,
        "trace": holdout_trace,
    }

    if promoted and args.apply_public:
        public_dataset = build_pair_dataset(
            public,
            args.contexts_zip,
            args.index,
            args.output_dir / "cache" / "public_pairs.npz",
            args.candidate_limit,
            args.prefilter_maximum,
        )
        if public_dataset.feature_names != complement_dataset.feature_names:
            raise RuntimeError("complement/public replacement feature schemas differ")
        public_probabilities = ensemble_predict(winner_models, public_dataset.X)
        public_predictions, public_trace = predictions_from_probabilities(
            public, public_dataset, public_probabilities, float(winner["threshold"]),
        )
        submission_path = args.output_dir / "v14_safe_rescue.json"
        write_json(submission_path, public_predictions)
        report["public"] = {
            "applied": True,
            "submission": str(submission_path.resolve()),
            "queries_changed": sum(
                public_predictions[qid]["answer"] != public.anchor[qid]["answer"]
                for qid in public.qids
            ),
            "prefilter": public_dataset.prefilter,
            "trace": public_trace,
        }

    report["artifacts"] = {
        "model": str(model_path.resolve()),
        "report": str(report_path.resolve()),
    }
    report["elapsed_seconds"] = time.perf_counter() - started
    write_json(report_path, report)
    print(json.dumps({
        "report": str(report_path.resolve()),
        "status": report["status"],
        "holdout": holdout_evaluation,
        "public": report["public"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, ensure_ascii=False, indent=2), flush=True)
    return 0 if promoted else 3


if __name__ == "__main__":
    raise SystemExit(main())
