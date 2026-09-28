"""Leak-safe QREL-memory retrieval and oracle audit for the V16-RGM track.

This is the first executable slice of the V16-RGM proposal.  It uses only
labelled *training* questions to build a query -> historical-query -> document
memory channel.  The existing v11h candidate ranking remains the anchor; this
module never changes the production submission by itself.

The implementation deliberately separates three concerns:

* ``QRELMemory`` builds a reusable non-parametric memory index and produces
  degree-corrected document features;
* ``fuse_memory_candidates`` adds only the highest-confidence memory documents
  to an existing anchor candidate pool;
* the CLI measures ``Oracle(anchor + memory) - Oracle(anchor)`` before any
  selector or graph teacher is trained.

The memory index is fold-safe by construction when its training records are
disjoint from the evaluated query records.  Exact question/qid overlap is
rejected by default because it is a common source of spectacular but invalid
memory results.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from .data_quality import collapse_training_questions, normalize_question
except ImportError:
    from data_quality import collapse_training_questions, normalize_question


DEFAULT_MEMORY_K = 50
DEFAULT_MEMORY_TOP_K = 5
DEFAULT_TEMPERATURE = 0.07
DEFAULT_DEGREE_EXPONENT = 0.5
DEFAULT_RECIPROCAL_FLOOR = 0.25


def _as_unique_strings(values: Iterable[Any]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values))


def _normalise_rows(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError(f"{name} must be a non-empty 2-D array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError(f"{name} contains a zero vector")
    return (array / norms).astype(np.float32, copy=False)


def _normalise_vector(value: np.ndarray, name: str = "query_embedding") -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1 or not array.size:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        raise ValueError(f"{name} is a zero vector")
    return (array / norm).astype(np.float32, copy=False)


def _top_indices(scores: np.ndarray, limit: int) -> np.ndarray:
    """Return deterministic descending-score indices without a full sort."""

    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    count = min(max(int(limit), 0), len(scores))
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if count == len(scores):
        candidates = np.arange(len(scores), dtype=np.int64)
    else:
        candidates = np.argpartition(-scores, count - 1)[:count]
    # Lexsort makes ties stable by row index, which keeps reports and cached
    # feature rows reproducible across NumPy versions.
    order = np.lexsort((candidates, -scores[candidates]))
    return candidates[order].astype(np.int64, copy=False)


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    shifted = (values - float(np.max(values))) / float(temperature)
    exponentials = np.exp(np.clip(shifted, -80.0, 80.0))
    total = float(exponentials.sum())
    return exponentials / total if total > 0 else np.full_like(values, 1.0 / len(values))


@dataclass(frozen=True)
class MemoryNeighbor:
    """One historical query selected for a target query."""

    question_id: str
    similarity: float
    weight: float
    reciprocal_score: float
    view_agreement: float
    answers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "similarity": self.similarity,
            "weight": self.weight,
            "reciprocal_score": self.reciprocal_score,
            "view_agreement": self.view_agreement,
            "answers": list(self.answers),
        }


@dataclass
class MemoryResult:
    """Memory evidence and per-document features for one target query."""

    query_id: str
    neighbors: tuple[MemoryNeighbor, ...]
    documents: list[dict[str, Any]]
    query_entropy: float

    def ranked_documents(self) -> list[dict[str, Any]]:
        return list(self.documents)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query_entropy": self.query_entropy,
            "neighbors": [neighbor.to_dict() for neighbor in self.neighbors],
            "documents": [dict(item) for item in self.documents],
        }


class QRELMemory:
    """Non-parametric query -> historical qrel document memory.

    ``embeddings`` are the normalized training-question vectors.  Optional
    secondary views can be supplied as another mapping of normalized or raw
    training vectors; query-time vectors for those views are passed through
    ``query_views``.  The primary view controls similarity and reciprocal
    gating, while secondary views contribute candidate coverage and agreement.
    """

    def __init__(
        self,
        question_ids: Sequence[str],
        embeddings: np.ndarray,
        answers: Mapping[str, Iterable[Any]],
        *,
        memory_k: int = DEFAULT_MEMORY_K,
        temperature: float = DEFAULT_TEMPERATURE,
        degree_exponent: float = DEFAULT_DEGREE_EXPONENT,
        reciprocal_floor: float = DEFAULT_RECIPROCAL_FLOOR,
        secondary_embeddings: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        if memory_k <= 0:
            raise ValueError("memory_k must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if degree_exponent < 0:
            raise ValueError("degree_exponent must be non-negative")
        if not 0.0 <= reciprocal_floor <= 1.0:
            raise ValueError("reciprocal_floor must be between 0 and 1")
        ids = tuple(str(value) for value in question_ids)
        if len(ids) != len(set(ids)):
            raise ValueError("memory question IDs must be unique")
        if not ids:
            raise ValueError("memory question IDs must not be empty")
        primary = _normalise_rows(embeddings, "embeddings")
        if len(ids) != len(primary):
            raise ValueError("question_ids and embeddings row counts differ")
        if primary.shape[0] < 1:
            raise ValueError("memory index must contain at least one question")

        self.question_ids = ids
        self.embeddings = primary
        answer_lookup = {str(question_id): values for question_id, values in answers.items()}
        self.answers = {
            question_id: _as_unique_strings(answer_lookup.get(question_id, ()))
            for question_id in ids
        }
        if any(not self.answers[question_id] for question_id in ids):
            missing = [question_id for question_id in ids if not self.answers[question_id]]
            raise ValueError(f"memory questions have no qrels: {missing[:5]}")
        self.memory_k = min(int(memory_k), len(ids))
        self.temperature = float(temperature)
        self.degree_exponent = float(degree_exponent)
        self.reciprocal_floor = float(reciprocal_floor)
        self.primary_view = "dense"
        self._views: dict[str, np.ndarray] = {self.primary_view: primary}
        for name, values in (secondary_embeddings or {}).items():
            view_name = str(name)
            if not view_name or view_name == self.primary_view:
                raise ValueError("secondary view names must be non-empty and distinct from dense")
            view = _normalise_rows(values, f"secondary_embeddings[{view_name!r}]")
            if view.shape[0] != primary.shape[0]:
                raise ValueError(f"secondary view {view_name!r} row count differs from embeddings")
            self._views[view_name] = view
        self._document_degree = self._build_document_degree()
        self._reverse_thresholds = self._build_reverse_thresholds()

    def _build_document_degree(self) -> Counter[str]:
        degree: Counter[str] = Counter()
        for question_id in self.question_ids:
            for doc_id in self.answers[question_id]:
                degree[doc_id] += 1
        return degree

    def _build_reverse_thresholds(self) -> np.ndarray:
        """Compute each train query's kth non-self similarity threshold.

        For an unseen query q and a train neighbor qi, q is reciprocal when
        ``sim(q, qi)`` reaches qi's kth-nearest training-query similarity.  A
        block implementation avoids materializing the full N x N matrix.
        """

        count = len(self.embeddings)
        if count == 1:
            return np.full(1, -np.inf, dtype=np.float32)
        neighbors = min(self.memory_k, count - 1)
        thresholds = np.empty(count, dtype=np.float32)
        block_size = 1024
        for start in range(0, count, block_size):
            stop = min(start + block_size, count)
            scores = self.embeddings[start:stop] @ self.embeddings.T
            local_rows = np.arange(stop - start)
            global_rows = np.arange(start, stop)
            scores[local_rows, global_rows] = -np.inf
            # The kth largest score is the smallest score among the selected
            # reciprocal neighborhood.  ``partition`` is deterministic enough
            # here because the final predicate uses >= and ties are harmless.
            thresholds[start:stop] = -np.partition(
                -scores, neighbors - 1, axis=1
            )[:, neighbors - 1]
        return thresholds

    @property
    def document_degree(self) -> dict[str, int]:
        return dict(self._document_degree)

    def _view_neighbors(
        self,
        query_embedding: np.ndarray,
        query_views: Mapping[str, np.ndarray] | None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        vectors: dict[str, np.ndarray] = {
            self.primary_view: _normalise_vector(query_embedding)
        }
        if query_views:
            for name, value in query_views.items():
                if name not in self._views:
                    raise ValueError(f"query view {name!r} has no indexed training view")
                vector = _normalise_vector(np.asarray(value), f"query_views[{name!r}]")
                if vector.shape[0] != self._views[name].shape[1]:
                    raise ValueError(f"query view {name!r} dimension differs from indexed view")
                vectors[name] = vector
        score_by_view = {
            name: self._views[name] @ vectors[name]
            for name in vectors
        }
        top_by_view = {
            name: _top_indices(scores, self.memory_k)
            for name, scores in score_by_view.items()
        }
        return score_by_view, top_by_view

    def retrieve(
        self,
        query_id: str,
        query_embedding: np.ndarray,
        *,
        query_views: Mapping[str, np.ndarray] | None = None,
        anchor_candidates: Sequence[str] = (),
        anchor_scores: Mapping[str, float] | None = None,
    ) -> MemoryResult:
        """Retrieve historical qrels and aggregate them into doc features."""

        score_by_view, top_by_view = self._view_neighbors(query_embedding, query_views)
        primary_scores = score_by_view[self.primary_view]
        candidate_indices = sorted(
            {
                int(index)
                for indices in top_by_view.values()
                for index in indices
            },
            key=lambda index: (-float(primary_scores[index]), index),
        )[: self.memory_k]
        if not candidate_indices:
            return MemoryResult(str(query_id), (), [], 0.0)

        active_views = tuple(top_by_view)
        top_sets = {name: set(map(int, indices)) for name, indices in top_by_view.items()}
        similarities = np.asarray(
            [float(primary_scores[index]) for index in candidate_indices],
            dtype=np.float32,
        )
        soft_weights = _softmax(similarities, self.temperature)
        neighbors: list[MemoryNeighbor] = []
        for local_index, train_index in enumerate(candidate_indices):
            similarity = float(primary_scores[train_index])
            reciprocal = float(
                similarity + 1e-7 >= float(self._reverse_thresholds[train_index])
            )
            agreement = sum(
                train_index in top_sets[name] for name in active_views
            ) / max(len(active_views), 1)
            gated_weight = float(
                soft_weights[local_index]
                * (self.reciprocal_floor + (1.0 - self.reciprocal_floor) * reciprocal)
                * agreement
            )
            question_key = self.question_ids[train_index]
            neighbors.append(MemoryNeighbor(
                question_id=question_key,
                similarity=similarity,
                weight=gated_weight,
                reciprocal_score=reciprocal,
                view_agreement=float(agreement),
                answers=self.answers[question_key],
            ))

        total_weight = sum(neighbor.weight for neighbor in neighbors)
        if total_weight > 0:
            entropy_weights = np.asarray(
                [neighbor.weight / total_weight for neighbor in neighbors],
                dtype=np.float64,
            )
            entropy = float(
                -np.sum(entropy_weights * np.log(np.clip(entropy_weights, 1e-12, None)))
                / math.log(len(entropy_weights))
                if len(entropy_weights) > 1 else 0.0
            )
        else:
            entropy = 0.0

        support: dict[str, list[MemoryNeighbor]] = defaultdict(list)
        for neighbor in neighbors:
            for doc_id in neighbor.answers:
                support[doc_id].append(neighbor)

        anchor_set = {str(value) for value in anchor_candidates}
        anchor_floor = (
            min((float(anchor_scores[doc_id]) for doc_id in anchor_set if anchor_scores and doc_id in anchor_scores), default=0.0)
            if anchor_scores else 0.0
        )
        documents: list[dict[str, Any]] = []
        for doc_id, doc_neighbors in support.items():
            doc_similarities = sorted(
                (neighbor.similarity for neighbor in doc_neighbors), reverse=True
            )
            raw_score = sum(
                neighbor.weight for neighbor in doc_neighbors
            )
            degree = int(self._document_degree.get(doc_id, 0))
            corrected = raw_score / (max(float(degree), 1.0) ** self.degree_exponent)
            top_three = doc_similarities[:3]
            documents.append({
                "doc_id": doc_id,
                "memory_score": float(raw_score),
                "prototype_max": float(doc_similarities[0]),
                "prototype_top3_mean": float(sum(top_three) / len(top_three)),
                "neighbor_vote_count": len(doc_neighbors),
                "unique_neighbor_count": len({neighbor.question_id for neighbor in doc_neighbors}),
                "neighbor_similarity_max": float(doc_similarities[0]),
                "neighbor_similarity_gap_1_2": float(
                    doc_similarities[0] - doc_similarities[1]
                    if len(doc_similarities) > 1 else 0.0
                ),
                "neighbor_entropy": float(entropy),
                "reciprocal_neighbor_score": float(
                    sum(neighbor.weight * neighbor.reciprocal_score for neighbor in doc_neighbors)
                    / max(sum(neighbor.weight for neighbor in doc_neighbors), 1e-12)
                ),
                "dense_lexical_neighbor_agreement": float(
                    sum(neighbor.weight * neighbor.view_agreement for neighbor in doc_neighbors)
                    / max(sum(neighbor.weight for neighbor in doc_neighbors), 1e-12)
                ),
                "document_qrel_degree": degree,
                "degree_corrected_memory_score": float(corrected),
                "memory_rank": 0,
                "is_memory_outside_c35": bool(doc_id not in anchor_set),
                "memory_vs_anchor_margin": float(corrected - anchor_floor),
            })
        documents.sort(
            key=lambda item: (
                -float(item["degree_corrected_memory_score"]),
                -float(item["prototype_max"]),
                str(item["doc_id"]),
            )
        )
        for rank, item in enumerate(documents, 1):
            item["memory_rank"] = rank
        return MemoryResult(
            query_id=str(query_id),
            neighbors=tuple(neighbors),
            documents=documents,
            query_entropy=entropy,
        )


def fuse_memory_candidates(
    anchor_candidates: Sequence[str],
    memory_result: MemoryResult,
    *,
    memory_top_k: int = DEFAULT_MEMORY_TOP_K,
    candidate_limit: int | None = None,
) -> list[str]:
    """Preserve the anchor order and append only new top memory documents."""

    if memory_top_k <= 0:
        raise ValueError("memory_top_k must be positive")
    anchor = list(dict.fromkeys(str(value) for value in anchor_candidates))
    selected = anchor[:candidate_limit] if candidate_limit is not None else anchor
    seen = set(selected)
    for item in memory_result.ranked_documents()[:memory_top_k]:
        doc_id = str(item["doc_id"])
        if doc_id in seen:
            continue
        selected.append(doc_id)
        seen.add(doc_id)
        if candidate_limit is not None and len(selected) >= candidate_limit:
            break
    return selected


def recall(gold: Iterable[str], predicted: Iterable[str]) -> float:
    relevant = {str(value) for value in gold}
    return len(relevant & {str(value) for value in predicted}) / len(relevant) if relevant else 0.0


def evaluate_oracle(
    rows: Sequence[Mapping[str, Any]],
    anchor_by_qid: Mapping[str, Sequence[str]],
    memory_results: Mapping[str, MemoryResult],
    *,
    anchor_cutoff: int = 35,
    memory_top_k: int = DEFAULT_MEMORY_TOP_K,
) -> dict[str, Any]:
    """Measure the retrieval headroom that memory can add to the anchor."""

    if anchor_cutoff <= 0 or memory_top_k <= 0:
        raise ValueError("anchor_cutoff and memory_top_k must be positive")
    values: list[float] = []
    fused_values: list[float] = []
    outside_gold = 0
    outside_gold_queries = 0
    memory_candidate_queries = 0
    for row in rows:
        qid = str(row["question_id"])
        if qid not in anchor_by_qid or qid not in memory_results:
            raise ValueError(f"Missing anchor or memory result for {qid}")
        gold = _as_unique_strings(row.get("answer", ()))
        anchor = list(dict.fromkeys(str(value) for value in anchor_by_qid[qid]))[:anchor_cutoff]
        result = memory_results[qid]
        fused = fuse_memory_candidates(anchor, result, memory_top_k=memory_top_k)
        anchor_gold = set(gold) & set(anchor)
        fused_gold = set(gold) & set(fused)
        outside = fused_gold - anchor_gold
        outside_gold += len(outside)
        outside_gold_queries += bool(outside)
        memory_candidate_queries += any(
            bool(item["is_memory_outside_c35"]) for item in result.documents[:memory_top_k]
        )
        values.append(recall(gold, anchor))
        fused_values.append(recall(gold, fused))
    denominator = len(rows) or 1
    anchor_value = float(sum(values) / denominator)
    fused_value = float(sum(fused_values) / denominator)
    return {
        "queries": len(rows),
        "anchor_cutoff": anchor_cutoff,
        "memory_top_k": memory_top_k,
        "anchor_oracle_recall": anchor_value,
        "memory_union_oracle_recall": fused_value,
        "delta_oracle": fused_value - anchor_value,
        "gold_recovered_outside_anchor": outside_gold,
        "queries_with_gold_recovered_outside_anchor": outside_gold_queries,
        "queries_with_memory_candidate_outside_anchor": memory_candidate_queries,
    }


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_records(path: Path, *, clean_duplicates: bool = False) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object keyed by question ID: {path}")
    if clean_duplicates:
        payload, _ = collapse_training_questions(payload)
    rows: dict[str, dict[str, Any]] = {}
    for raw_qid, raw_item in payload.items():
        if not isinstance(raw_item, dict) or not isinstance(raw_item.get("question"), str):
            raise ValueError(f"{raw_qid}: missing string question")
        item = dict(raw_item)
        if "answer" in item and item["answer"] is not None and not isinstance(item["answer"], list):
            raise ValueError(f"{raw_qid}: answer must be a list when present")
        rows[str(raw_qid)] = item
    if not rows:
        raise ValueError(f"No question records found: {path}")
    return rows


def validate_split_isolation(
    memory_records: Mapping[str, Mapping[str, Any]],
    query_records: Mapping[str, Mapping[str, Any]],
    *,
    allow_exact_overlap: bool = False,
) -> dict[str, Any]:
    memory_ids = {str(value) for value in memory_records}
    query_ids = {str(value) for value in query_records}
    qid_overlap = sorted(memory_ids & query_ids)
    memory_questions = {
        normalize_question(item["question"]): qid
        for qid, item in memory_records.items()
    }
    query_questions = {
        normalize_question(item["question"]): qid
        for qid, item in query_records.items()
    }
    normalized_overlap = sorted(set(memory_questions) & set(query_questions))
    if qid_overlap:
        raise ValueError(f"memory/query question-ID overlap: {qid_overlap[:5]}")
    if normalized_overlap and not allow_exact_overlap:
        raise ValueError(
            "memory/query normalized-question overlap detected; "
            "use a fold-isolated split or explicitly pass --allow-exact-overlap"
        )
    return {
        "memory_questions": len(memory_records),
        "query_questions": len(query_records),
        "qid_overlap": qid_overlap,
        "normalized_question_overlap_count": len(normalized_overlap),
        "normalized_question_overlap": normalized_overlap[:20],
        "exact_overlap_allowed": bool(allow_exact_overlap),
    }


def drop_normalized_question_overlap(
    memory_records: Mapping[str, Mapping[str, Any]],
    query_records: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Remove memory rows whose normalized question occurs in the query fold."""

    query_questions = {
        normalize_question(item["question"])
        for item in query_records.values()
    }
    removed = [
        str(qid)
        for qid, item in memory_records.items()
        if normalize_question(item["question"]) in query_questions
    ]
    removed_set = set(removed)
    kept = {
        str(qid): dict(item)
        for qid, item in memory_records.items()
        if str(qid) not in removed_set
    }
    return kept, sorted(removed)


def _load_embedding_rows(
    embedding_path: Path,
    metadata_path: Path | None,
) -> tuple[np.ndarray, list[str] | None]:
    embeddings = np.load(str(embedding_path))
    qids: list[str] | None = None
    if metadata_path is not None:
        metadata = read_json(metadata_path)
        if not isinstance(metadata, list):
            raise ValueError(f"Embedding metadata must be a list: {metadata_path}")
        qids = []
        for row in metadata:
            if not isinstance(row, dict) or "question_id" not in row:
                raise ValueError(f"Embedding metadata rows need question_id: {metadata_path}")
            qids.append(str(row["question_id"]))
        if len(qids) != len(embeddings):
            raise ValueError("Embedding metadata and embedding row counts differ")
        if len(qids) != len(set(qids)):
            raise ValueError("Embedding metadata question IDs must be unique")
    return np.asarray(embeddings, dtype=np.float32), qids


def _default_embedding_metadata(path: Path) -> Path:
    if path.name.endswith(".npy"):
        return path.with_name(path.name[:-4] + ".meta.json")
    return path.with_suffix(path.suffix + ".meta.json")


def _reorder_embeddings(
    embeddings: np.ndarray,
    embedding_ids: Sequence[str] | None,
    ordered_ids: Sequence[str],
    *,
    name: str,
) -> np.ndarray:
    if embedding_ids is None:
        if len(embeddings) != len(ordered_ids):
            raise ValueError(f"{name} rows do not match the question record count")
        return embeddings
    positions = {str(qid): index for index, qid in enumerate(embedding_ids)}
    missing = [str(qid) for qid in ordered_ids if str(qid) not in positions]
    if missing:
        raise ValueError(f"{name} is missing question IDs: {missing[:5]}")
    return embeddings[[positions[str(qid)] for qid in ordered_ids]]


def _encode_queries(
    records: Mapping[str, Mapping[str, Any]],
    model_path: str,
    *,
    device: str,
    batch_size: int,
    max_seq_length: int,
    query_prefix: str,
) -> np.ndarray:
    if batch_size <= 0 or max_seq_length <= 0:
        raise ValueError("batch_size and max_seq_length must be positive")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:  # pragma: no cover - exercised only without neural deps
        raise RuntimeError(
            "sentence-transformers is required when --query-embeddings is omitted"
        ) from error
    model = SentenceTransformer(model_path, device=device, local_files_only=True)
    model.max_seq_length = int(max_seq_length)
    texts = [
        f"{query_prefix}{item['question']}" if query_prefix else str(item["question"])
        for item in records.values()
    ]
    return np.asarray(model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ), dtype=np.float32)


def _load_anchor(path: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    payload = read_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("stage1_candidates"), list):
        queries = payload.get("queries")
        candidates = payload["stage1_candidates"]
        if not isinstance(queries, list) or len(queries) != len(candidates):
            raise ValueError(f"Invalid Stage-1 audit: {path}")
        by_qid: dict[str, list[str]] = {}
        questions: dict[str, str] = {}
        for query, ranking in zip(queries, candidates, strict=True):
            if not isinstance(query, dict) or not isinstance(ranking, list):
                raise ValueError(f"Invalid Stage-1 row: {path}")
            qid = str(query["query_id"])
            by_qid[qid] = list(dict.fromkeys(str(value) for value in ranking))
            questions[qid] = str(query.get("query", query.get("question", "")))
        return by_qid, questions
    if isinstance(payload, dict):
        by_qid = {}
        questions = {}
        for raw_qid, raw_item in payload.items():
            if not isinstance(raw_item, dict):
                raise ValueError(f"Invalid anchor row {raw_qid}: {path}")
            ranking = raw_item.get("candidates", raw_item.get("answer"))
            if not isinstance(ranking, list):
                raise ValueError(f"{raw_qid}: anchor needs candidates or answer list")
            qid = str(raw_qid)
            by_qid[qid] = list(dict.fromkeys(str(value) for value in ranking))
            questions[qid] = str(raw_item.get("question", ""))
        return by_qid, questions
    raise ValueError(f"Unsupported anchor format: {path}")


def build_memory_from_files(
    memory_records: Mapping[str, Mapping[str, Any]],
    query_records: Mapping[str, Mapping[str, Any]],
    memory_embeddings: np.ndarray,
    memory_embedding_ids: Sequence[str] | None,
    query_embeddings: np.ndarray,
    query_embedding_ids: Sequence[str] | None,
    *,
    memory_k: int,
    temperature: float,
    degree_exponent: float,
    reciprocal_floor: float,
    anchor_by_qid: Mapping[str, Sequence[str]],
    anchor_scores_by_qid: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, MemoryResult]:
    ordered_memory_ids = list(memory_records)
    ordered_query_ids = list(query_records)
    memory_matrix = _reorder_embeddings(
        memory_embeddings, memory_embedding_ids, ordered_memory_ids, name="memory embeddings"
    )
    query_matrix = _reorder_embeddings(
        query_embeddings, query_embedding_ids, ordered_query_ids, name="query embeddings"
    )
    answers = {
        qid: _as_unique_strings(item.get("answer", ()))
        for qid, item in memory_records.items()
    }
    memory = QRELMemory(
        ordered_memory_ids,
        memory_matrix,
        answers,
        memory_k=memory_k,
        temperature=temperature,
        degree_exponent=degree_exponent,
        reciprocal_floor=reciprocal_floor,
    )
    results: dict[str, MemoryResult] = {}
    for index, qid in enumerate(ordered_query_ids):
        if qid not in anchor_by_qid:
            raise ValueError(f"Anchor candidates missing query {qid}")
        results[qid] = memory.retrieve(
            qid,
            query_matrix[index],
            anchor_candidates=anchor_by_qid[qid],
            anchor_scores=(anchor_scores_by_qid or {}).get(qid),
        )
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-data", type=Path, required=True)
    parser.add_argument("--query-data", type=Path, required=True)
    parser.add_argument("--anchor-audit", type=Path, required=True)
    parser.add_argument("--memory-embeddings", type=Path, required=True)
    parser.add_argument("--memory-embedding-meta", type=Path)
    parser.add_argument("--query-embeddings", type=Path)
    parser.add_argument("--query-embedding-meta", type=Path)
    parser.add_argument("--embedding-model")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--query-prefix", default="query: ")
    parser.add_argument("--memory-k", type=int, default=DEFAULT_MEMORY_K)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--degree-exponent", type=float, default=DEFAULT_DEGREE_EXPONENT)
    parser.add_argument("--reciprocal-floor", type=float, default=DEFAULT_RECIPROCAL_FLOOR)
    parser.add_argument("--anchor-cutoff", type=int, default=35)
    parser.add_argument("--memory-top-k", type=int, default=DEFAULT_MEMORY_TOP_K)
    parser.add_argument("--allow-exact-overlap", action="store_true")
    parser.add_argument(
        "--drop-exact-overlap",
        action="store_true",
        help="Safely remove exact normalized question duplicates from memory before retrieval",
    )
    parser.add_argument("--features-output", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.anchor_cutoff <= 0 or args.memory_top_k <= 0:
        raise ValueError("anchor-cutoff and memory-top-k must be positive")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    started = time.perf_counter()
    # Match the current v11h index contract: its question cache is built from
    # duplicate-collapsed training rows, not the raw 6,000-row JSON.
    memory_records = load_records(args.memory_data, clean_duplicates=True)
    query_records = load_records(args.query_data)
    isolation = validate_split_isolation(
        memory_records,
        query_records,
        allow_exact_overlap=args.allow_exact_overlap or args.drop_exact_overlap,
    )
    dropped_overlap_ids: list[str] = []
    if args.drop_exact_overlap:
        memory_records, dropped_overlap_ids = drop_normalized_question_overlap(
            memory_records, query_records
        )
        isolation["normalized_question_overlap_dropped"] = len(dropped_overlap_ids)
        isolation["dropped_memory_question_ids"] = dropped_overlap_ids[:20]
        isolation["memory_questions_after_overlap_filter"] = len(memory_records)
        isolation["overlap_policy"] = "drop_exact_normalized_overlap"
        isolation["exact_overlap_allowed"] = False
    else:
        isolation["normalized_question_overlap_dropped"] = 0
        isolation["dropped_memory_question_ids"] = []
        isolation["overlap_policy"] = "allow_explicit" if args.allow_exact_overlap else "reject"
    anchor_by_qid, anchor_questions = _load_anchor(args.anchor_audit)
    query_ids = list(query_records)
    if set(anchor_by_qid) != set(query_ids):
        missing = sorted(set(query_ids) - set(anchor_by_qid))
        extra = sorted(set(anchor_by_qid) - set(query_ids))
        raise ValueError(f"Anchor/query IDs differ; missing={missing[:5]}, extra={extra[:5]}")
    for qid, anchor_question in anchor_questions.items():
        if anchor_question and normalize_question(anchor_question) != normalize_question(query_records[qid]["question"]):
            raise ValueError(f"Anchor question text differs from query data for {qid}")

    memory_meta = args.memory_embedding_meta
    if memory_meta is None:
        candidate = _default_embedding_metadata(args.memory_embeddings)
        memory_meta = candidate if candidate.is_file() else None
    memory_embeddings, memory_embedding_ids = _load_embedding_rows(
        args.memory_embeddings, memory_meta
    )
    if args.query_embeddings is not None:
        query_meta = args.query_embedding_meta
        if query_meta is None:
            candidate = _default_embedding_metadata(args.query_embeddings)
            query_meta = candidate if candidate.is_file() else None
        query_embeddings, query_embedding_ids = _load_embedding_rows(
            args.query_embeddings, query_meta
        )
    else:
        if not args.embedding_model:
            raise ValueError("Provide --query-embeddings or --embedding-model")
        query_embeddings = _encode_queries(
            query_records,
            str(Path(args.embedding_model).resolve()),
            device=args.device,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            query_prefix=args.query_prefix,
        )
        query_embedding_ids = None

    results = build_memory_from_files(
        memory_records,
        query_records,
        memory_embeddings,
        memory_embedding_ids,
        query_embeddings,
        query_embedding_ids,
        memory_k=args.memory_k,
        temperature=args.temperature,
        degree_exponent=args.degree_exponent,
        reciprocal_floor=args.reciprocal_floor,
        anchor_by_qid=anchor_by_qid,
    )
    labelled_rows = [
        {"question_id": qid, **item}
        for qid, item in query_records.items()
        if isinstance(item.get("answer"), list)
    ]
    metrics = evaluate_oracle(
        labelled_rows,
        anchor_by_qid,
        results,
        anchor_cutoff=args.anchor_cutoff,
        memory_top_k=args.memory_top_k,
    ) if labelled_rows else None

    feature_payload = {
        "kind": "legalir_v16_qrel_memory_features",
        "memory": {
            "memory_k": args.memory_k,
            "temperature": args.temperature,
            "degree_exponent": args.degree_exponent,
            "reciprocal_floor": args.reciprocal_floor,
        },
        "anchor_cutoff": args.anchor_cutoff,
        "memory_top_k": args.memory_top_k,
        "leakage_policy": {
            **isolation,
            "labels_used_for_query_retrieval": False,
            "query_answers_used_only_for_oracle_metrics": bool(labelled_rows),
        },
        "queries": [
            {
                "query_id": qid,
                "anchor_candidates": list(anchor_by_qid[qid][:args.anchor_cutoff]),
                "fused_candidates": fuse_memory_candidates(
                    anchor_by_qid[qid][:args.anchor_cutoff],
                    results[qid],
                    memory_top_k=args.memory_top_k,
                ),
                "memory": results[qid].to_dict(),
            }
            for qid in query_ids
        ],
    }
    if args.features_output:
        args.features_output.parent.mkdir(parents=True, exist_ok=True)
        args.features_output.write_text(
            json.dumps(feature_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    report = {
        "kind": "legalir_v16a_qrel_memory_oracle",
        "memory_data": str(args.memory_data.resolve()),
        "query_data": str(args.query_data.resolve()),
        "anchor_audit": str(args.anchor_audit.resolve()),
        "memory_embeddings": str(args.memory_embeddings.resolve()),
        "query_embeddings": str(args.query_embeddings.resolve()) if args.query_embeddings else None,
        "embedding_model": str(Path(args.embedding_model).resolve()) if args.embedding_model else None,
        "memory": feature_payload["memory"],
        "anchor_cutoff": args.anchor_cutoff,
        "memory_top_k": args.memory_top_k,
        "questions": len(query_records),
        "labelled_questions": len(labelled_rows),
        "leakage_policy": feature_payload["leakage_policy"],
        "metrics": metrics,
        "features_output": str(args.features_output.resolve()) if args.features_output else None,
        "elapsed_seconds": time.perf_counter() - started,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(rendered, flush=True)
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
