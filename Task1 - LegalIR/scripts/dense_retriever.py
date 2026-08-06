"""Offline dense retrieval with a deterministic TF-IDF fallback."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


TOKEN_RE = re.compile(r"[0-9]+(?:[./-][0-9a-z]+)*|[^\W_]+", re.UNICODE)


def _tokens(text: str) -> List[str]:
    return TOKEN_RE.findall(unicodedata.normalize("NFC", str(text)).casefold())


class TFIDFFallback:
    """Compact L2-normalized TF-IDF; no external model or network required."""

    def __init__(self):
        self.doc_vectors: Optional[np.ndarray] = None
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def fit(self, documents: List[str]) -> None:
        if not documents:
            raise ValueError("Cannot build TF-IDF on an empty corpus")
        document_frequency: Counter[str] = Counter()
        for document in documents:
            document_frequency.update(set(_tokens(document)))
        minimum_frequency = 1 if len(documents) < 20 else 2
        kept_terms = sorted(term for term, frequency in document_frequency.items() if frequency >= minimum_frequency)
        self.vocab = {term: index for index, term in enumerate(kept_terms)}
        self.idf = {
            term: math.log((1.0 + len(documents)) / (1.0 + document_frequency[term])) + 1.0
            for term in kept_terms
        }
        self.doc_vectors = self._encode(documents)

    def _encode(self, texts: List[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for row, text in enumerate(texts):
            counts = Counter(_tokens(text))
            for term, frequency in counts.items():
                column = self.vocab.get(term)
                if column is not None:
                    vectors[row, column] = (1.0 + math.log(frequency)) * self.idf[term]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors /= np.where(norms > 0, norms, 1.0)
        return vectors

    def encode(self, texts: List[str]) -> np.ndarray:
        return self._encode(texts)

    def save(self, path: Path) -> None:
        if self.doc_vectors is None:
            raise RuntimeError("TF-IDF index has not been built")
        np.savez_compressed(
            Path(path),
            doc_vectors=self.doc_vectors,
            vocab=np.array(self.vocab, dtype=object),
            idf=np.array(self.idf, dtype=object),
        )

    def load(self, path: Path) -> None:
        data = np.load(Path(path), allow_pickle=True)
        self.doc_vectors = data["doc_vectors"]
        self.vocab = data["vocab"].item()
        self.idf = data["idf"].item()


class DenseRetriever:
    """SentenceTransformer retrieval loaded strictly from a local path.

    Passing no ``model_path`` intentionally selects TF-IDF.  This prevents an
    accidental Hugging Face download during competition execution.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        max_seq_length: int = 512,
        force_fallback: bool = False,
    ):
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.max_seq_length = int(max_seq_length)
        self.device = None if device in (None, "auto") else device
        self.documents: List[Dict[str, Any]] = []
        self.doc_embeddings: Optional[np.ndarray] = None
        self.use_fallback = force_fallback or model_name is None

        if self.use_fallback:
            self.model: Any = TFIDFFallback()
        else:
            model_path = Path(str(model_name))
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Dense model must be a downloaded local directory, got: {model_name}"
                )
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("Install sentence-transformers to use a dense model") from exc
            self.model = SentenceTransformer(
                str(model_path), device=self.device, local_files_only=True
            )
            self.model.max_seq_length = self.max_seq_length

    def build_index(self, documents: List[Dict[str, Any]], save_path: Optional[Path] = None) -> None:
        self.documents = [dict(document) for document in documents]
        texts = [str(document["text"]) for document in self.documents]
        if self.use_fallback:
            self.model.fit(texts)
            self.doc_embeddings = self.model.doc_vectors
        else:
            self.doc_embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
        if save_path:
            self.save(save_path)

    def retrieve(self, query: str, top_k: int = 20, threshold: float = 0.0) -> List[Dict[str, Any]]:
        if self.doc_embeddings is None:
            raise RuntimeError("Dense index has not been built")
        if self.use_fallback:
            query_vector = self.model.encode([query])[0]
        else:
            query_vector = self.model.encode(
                [query], convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True
            )[0]
        similarities = self.doc_embeddings @ query_vector
        indices = np.argsort(-similarities, kind="stable")[: max(0, top_k)]
        results = []
        for index in indices:
            score = float(similarities[index])
            if score <= threshold:
                continue
            document = self.documents[int(index)]
            results.append({
                "doc_id": str(document["doc_id"]),
                "chunk_id": str(document.get("chunk_id", document["doc_id"])),
                "text": str(document["text"]),
                "score": score,
                "rank": len(results) + 1,
                "method": "tfidf" if self.use_fallback else "dense",
                "position": int(document.get("position", 0)),
            })
        return results

    def batch_retrieve(self, queries: List[str], top_k: int = 20, threshold: float = 0.0) -> List[List[Dict[str, Any]]]:
        return [self.retrieve(query, top_k, threshold) for query in queries]

    def save(self, path: Path) -> None:
        if self.doc_embeddings is None:
            raise RuntimeError("Dense index has not been built")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.use_fallback:
            self.model.save(path.with_suffix(".npz"))
        else:
            np.save(path.with_suffix(".npy"), self.doc_embeddings)
        metadata = {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "max_seq_length": self.max_seq_length,
            "documents": self.documents,
            "use_fallback": self.use_fallback,
        }
        path.with_suffix(".meta.json").write_text(
            json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path, device: Optional[str] = None) -> "DenseRetriever":
        path = Path(path)
        metadata = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        instance = cls(
            model_name=metadata.get("model_name"),
            device=device,
            batch_size=metadata["batch_size"],
            max_seq_length=metadata["max_seq_length"],
            force_fallback=metadata["use_fallback"],
        )
        instance.documents = metadata["documents"]
        if instance.use_fallback:
            instance.model.load(path.with_suffix(".npz"))
            instance.doc_embeddings = instance.model.doc_vectors
        else:
            instance.doc_embeddings = np.load(path.with_suffix(".npy"))
        return instance


def preprocess_query(query: str) -> str:
    return " ".join(unicodedata.normalize("NFC", str(query)).split())
