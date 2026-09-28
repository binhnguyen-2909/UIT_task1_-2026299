"""Offline cross-encoder reranking with a transparent lexical fallback."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


TOKEN_RE = re.compile(r"[0-9]+(?:[./-][0-9a-z]+)*|[^\W_]+", re.UNICODE)
REFERENCE_RE = re.compile(r"\b(?:điều|khoản|điểm|chương)\s+[0-9]+[a-z]?\b", re.IGNORECASE)


def _tokens(text: str) -> List[str]:
    return TOKEN_RE.findall(unicodedata.normalize("NFC", str(text)).casefold())


def reciprocal_rank_fuse(
    lexical: List[Dict[str, Any]],
    neural: List[Dict[str, Any]],
    neural_weight: float = 0.1,
    rrf_k: int = 0,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Fuse lexical and neural ranks while preserving neural audit fields."""

    if neural_weight < 0 or rrf_k < 0:
        raise ValueError("RRF weight and k must be non-negative")
    lexical_rank = {
        str(item["doc_id"]): rank for rank, item in enumerate(lexical, 1)
    }
    neural_order = list(neural)
    if any("rerank_score" in item for item in neural_order):
        neural_order.sort(
            key=lambda item: (
                -float(item.get("rerank_score", float("-inf"))),
                str(item["doc_id"]),
            )
        )
    neural_rank = {
        str(item["doc_id"]): rank for rank, item in enumerate(neural_order, 1)
    }
    universe = list(dict.fromkeys([*lexical_rank, *neural_rank]))
    by_id = {str(item["doc_id"]): dict(item) for item in lexical}
    by_id.update({str(item["doc_id"]): dict(item) for item in neural})
    missing_rank = len(universe) + 1
    for doc_id in universe:
        by_id[doc_id]["rerank_fusion_score"] = (
            1.0 / (rrf_k + lexical_rank.get(doc_id, missing_rank))
            + neural_weight / (rrf_k + neural_rank.get(doc_id, missing_rank))
        )
    output = sorted(
        by_id.values(),
        key=lambda item: (
            -float(item["rerank_fusion_score"]),
            lexical_rank.get(str(item["doc_id"]), missing_rank),
            str(item["doc_id"]),
        ),
    )
    for rank, item in enumerate(output, 1):
        item["rank"] = rank
    return output[:top_k] if top_k is not None else output


class LegalRelevanceFeatures:
    """Feature helper used by the fallback and by audit output."""

    @staticmethod
    def compute(query: str, passage: str) -> Dict[str, float]:
        query_tokens = Counter(_tokens(query))
        passage_tokens = Counter(_tokens(passage))
        overlap = sum((query_tokens & passage_tokens).values())
        query_total = max(1, sum(query_tokens.values()))
        exact_refs = set(REFERENCE_RE.findall(query.casefold()))
        reference_match = float(any(reference in passage.casefold() for reference in exact_refs))
        phrase_match = float(query.casefold() in passage.casefold())
        return {
            "token_recall": overlap / query_total,
            "reference_match": reference_match,
            "phrase_match": phrase_match,
        }


class Reranker:
    """Rank candidates; neural models are accepted only as local directories."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 16,
        max_length: Optional[int] = None,
        passage_aggregation: str = "max",
        use_fp16: bool = False,
    ):
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.max_length = int(max_length) if max_length is not None else None
        if passage_aggregation not in {"max", "top3"}:
            raise ValueError("passage_aggregation must be 'max' or 'top3'")
        self.passage_aggregation = passage_aggregation
        self.use_fp16 = bool(use_fp16)
        self.device = None if device in (None, "auto") else device
        self.use_fallback = model_name is None
        self.model: Any = None
        self.last_trace: Optional[Dict[str, Any]] = None
        if not self.use_fallback:
            model_path = Path(str(model_name))
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Reranker model must be a downloaded local directory, got: {model_name}"
                )
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError("Install sentence-transformers to use the reranker") from exc
            model_kwargs: Dict[str, Any] = {
                "device": self.device,
                "local_files_only": True,
            }
            if self.max_length is not None:
                if self.max_length <= 0:
                    raise ValueError("max_length must be positive")
                model_kwargs["max_length"] = self.max_length
            self.model = CrossEncoder(str(model_path), **model_kwargs)
            if self.use_fp16 and str(device or "").casefold() == "cuda":
                # FP16 changes arithmetic precision, not the model's parameter
                # count. It is an explicit runtime optimization for 6 GB GPUs.
                self.model.model.half()

    def _aggregate_passage_scores(self, scores: List[float]) -> float:
        """Aggregate evidence without rewarding a document for length alone."""

        if not scores:
            return float("-inf")
        ordered = sorted((float(value) for value in scores), reverse=True)
        if self.passage_aggregation == "max" or len(ordered) == 1:
            return ordered[0]
        # The best legal unit remains dominant; corroborating units contribute
        # only a small bounded amount.
        return ordered[0] + 0.15 * ordered[1] + (
            0.05 * ordered[2] if len(ordered) > 2 else 0.0
        )

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        if not candidates:
            self.last_trace = {
                "input_ids": [],
                "neural_ids": [],
                "neural_scores": {},
            }
            return []
        output = [dict(candidate) for candidate in candidates]
        input_ids = [str(candidate["doc_id"]) for candidate in output]
        if self.use_fallback:
            scores = []
            for candidate in output:
                passages = candidate.get("passages") or [str(candidate["text"])]
                base_score = float(candidate.get("score", 0.0))
                passage_scores = []
                passage_features = []
                for passage in passages:
                    features = LegalRelevanceFeatures.compute(query, str(passage))
                    passage_features.append(features)
                    passage_scores.append(
                        0.65 * features["token_recall"]
                        + 0.20 * features["reference_match"]
                        + 0.05 * features["phrase_match"]
                        + 0.10 * math.tanh(base_score)
                    )
                best_index = max(range(len(passage_scores)), key=passage_scores.__getitem__)
                score = self._aggregate_passage_scores(passage_scores)
                features = passage_features[best_index]
                candidate["text"] = str(passages[best_index])
                candidate["rerank_passage_index"] = best_index
                candidate["rerank_features"] = features
                candidate["passage_scores"] = [float(value) for value in passage_scores]
                scores.append(score)
        else:
            pairs = []
            owners = []
            passages_by_candidate = []
            for candidate_index, candidate in enumerate(output):
                passages = [
                    str(value)
                    for value in (
                        candidate.get("passages") or [candidate.get("text", "")]
                    )
                ]
                passages_by_candidate.append(passages)
                for passage in passages:
                    pairs.append((query, passage))
                    owners.append(candidate_index)
            flat_scores = [float(value) for value in self.model.predict(
                pairs, batch_size=self.batch_size, show_progress_bar=False
            )]
            scores = [float("-inf")] * len(output)
            passage_scores_by_candidate: List[List[float]] = [[] for _ in output]
            best_indices = [0] * len(output)
            passage_positions = [0] * len(output)
            for owner, score in zip(owners, flat_scores):
                position = passage_positions[owner]
                passage_positions[owner] += 1
                passage_scores_by_candidate[owner].append(float(score))
                if score > scores[owner]:
                    scores[owner] = score
                    best_indices[owner] = position
            for candidate_index, candidate in enumerate(output):
                best_index = best_indices[candidate_index]
                candidate["text"] = passages_by_candidate[candidate_index][best_index]
                candidate["rerank_passage_index"] = best_index
                candidate["passage_scores"] = passage_scores_by_candidate[candidate_index]
                scores[candidate_index] = self._aggregate_passage_scores(
                    passage_scores_by_candidate[candidate_index]
                )

        for candidate, score in zip(output, scores):
            candidate["rerank_score"] = float(score)
        output.sort(
            key=lambda item: (
                -float(item["rerank_score"]),
                -float(item.get("score", 0.0)),
                str(item["doc_id"]),
                str(item.get("chunk_id", "")),
            )
        )
        for rank, candidate in enumerate(output, 1):
            candidate["rank"] = rank
        self.last_trace = {
            "input_ids": input_ids,
            "neural_ids": [str(candidate["doc_id"]) for candidate in output],
            "neural_scores": {
                str(candidate["doc_id"]): float(candidate["rerank_score"])
                for candidate in output
            },
            "passage_aggregation": self.passage_aggregation,
        }
        return output[:top_k] if top_k is not None else output

    def fit(self, *_: Any, **__: Any) -> "Reranker":
        """CrossEncoder fine-tuning needs an explicit training loop; never fake it."""
        raise NotImplementedError(
            "Use sentence-transformers CrossEncoderTrainer with labelled pairs; "
            "this inference component intentionally has no fake fit()."
        )

    def save_config(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps({"model_name": self.model_name, "batch_size": self.batch_size}),
            encoding="utf-8",
        )
