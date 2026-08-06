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
    ):
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.device = None if device in (None, "auto") else device
        self.use_fallback = model_name is None
        self.model: Any = None
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
            self.model = CrossEncoder(
                str(model_path), device=self.device, local_files_only=True
            )

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        output = [dict(candidate) for candidate in candidates]
        if self.use_fallback:
            scores = []
            for candidate in output:
                features = LegalRelevanceFeatures.compute(query, str(candidate["text"]))
                base_score = float(candidate.get("score", 0.0))
                score = (
                    0.65 * features["token_recall"]
                    + 0.20 * features["reference_match"]
                    + 0.05 * features["phrase_match"]
                    + 0.10 * math.tanh(base_score)
                )
                candidate["rerank_features"] = features
                scores.append(score)
        else:
            pairs = [(query, str(candidate["text"])) for candidate in output]
            scores = [float(value) for value in self.model.predict(
                pairs, batch_size=self.batch_size, show_progress_bar=False
            )]

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
