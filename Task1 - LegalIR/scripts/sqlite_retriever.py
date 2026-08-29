"""Scalable SQLite FTS5 retriever for the 350MB+ official LegalIR corpus."""

from __future__ import annotations

import json
import hashlib
import itertools
import os
import re
import sqlite3
import time
import unicodedata
import urllib.parse
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:
    from .bm25_retriever import VietnameseTokenizer, expand_query
    from .data_quality import (
        load_raw_training_records,
        load_training_records,
        normalize_corpus_text,
    )
    from .legal_structure import iter_index_chunks
    from .reranker import Reranker, reciprocal_rank_fuse
    from .retrieval_types import QueryResult, RetrievalResult
except ImportError:
    from bm25_retriever import VietnameseTokenizer, expand_query
    from data_quality import load_raw_training_records, load_training_records, normalize_corpus_text
    from legal_structure import iter_index_chunks
    from reranker import Reranker, reciprocal_rank_fuse
    from retrieval_types import QueryResult, RetrievalResult


# ─── Dense embedding cache ────────────────────────────────────────────────────────

class DenseCache:
    """GPU-accelerated embedding cache backed by SentenceTransformer.

    Encode-once, query-many.  On a 6 GB laptop GPU the full 294 K chunk
    index fits comfortably (≈ 294 K × 768 × 4 B ≈ 0.9 GB).
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        batch_size: int = 16,
        max_seq_length: int = 384,
        query_prefix: str = "",
        passage_prefix: str = "",
    ):
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self._embeddings: Optional[np.ndarray] = None
        self._model: Any = None  # SentenceTransformer
        self._chunk_meta: List[str] = []  # "doc_id|chunk_id"

    def runtime_signature(self) -> Dict[str, Any]:
        """Return model/encoding settings that make caches incompatible."""

        model_path = Path(self.model_path).resolve()
        marker: Dict[str, Any] = {"path": str(model_path)}
        config_path = model_path / "config.json"
        if config_path.is_file():
            stat = config_path.stat()
            marker.update({"config_size": stat.st_size, "config_mtime_ns": stat.st_mtime_ns})
        return {
            "schema": 2,
            "model": marker,
            "max_seq_length": int(self.max_seq_length),
            "query_prefix": self.query_prefix,
            "passage_prefix": self.passage_prefix,
        }

    def _build_signature(self, chunks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        digest = hashlib.sha256()
        for chunk in chunks:
            for value in (
                chunk.get("chunk_id", chunk.get("doc_id", "")),
                chunk.get("title", ""),
                chunk.get("passage", ""),
            ):
                digest.update(str(value).encode("utf-8", errors="surrogatepass"))
                digest.update(b"\0")
        return {
            "runtime": self.runtime_signature(),
            "rows": len(chunks),
            "ordered_content_sha256": digest.hexdigest(),
        }

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            import torch
            requested = str(self.device or "auto").lower()
            device = requested
            if requested == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            elif requested == "cuda" and not torch.cuda.is_available():
                device = "cpu"
            self._model = SentenceTransformer(
                self.model_path,
                device=device,
                local_files_only=True,
            )
            self._model.max_seq_length = self.max_seq_length
            self._model.eval()
        return self._model

    @staticmethod
    def _chunk_text(chunk: Dict[str, Any]) -> str:
        title = chunk.get("title", "")
        passage = chunk.get("passage", "")
        return (title + "\n" + passage).strip()

    def _passage_text(self, chunk: Dict[str, Any]) -> str:
        text = self._chunk_text(chunk)
        return f"{self.passage_prefix}{text}" if self.passage_prefix else text

    def build(
        self,
        chunks: List[Dict[str, Any]],
        save_path: Optional[Path] = None,
        checkpoint_every: int = 500,
    ) -> Dict[str, Any]:
        """Encode all chunk texts and persist to disk.

        Saves a checkpoint every ``checkpoint_every`` batches so that an interrupted
        run can resume without re-encoding everything.
        """
        total_chunks = len(chunks)
        total_batches = (total_chunks + self.batch_size - 1) // self.batch_size
        started = time.perf_counter()
        signature = self._build_signature(chunks)

        self._chunk_meta = [
            str(c.get("chunk_id", c["doc_id"]))
            for c in chunks
        ]

        checkpoint_path: Optional[Path] = None
        resume_batch = 0
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            final_embeddings = save_path.with_suffix(".npy")
            final_ids = save_path.with_suffix(".ids.json")
            final_manifest = save_path.with_suffix(".manifest.json")
            if final_embeddings.exists() and final_ids.exists() and final_manifest.exists():
                saved_signature = json.loads(final_manifest.read_text(encoding="utf-8"))
                if saved_signature == signature:
                    self.load(save_path, expected_signature=signature)
                    return {
                        "chunks": total_chunks,
                        "embedding_dim": int(self._embeddings.shape[1]),
                        "seconds": 0.0,
                        "skipped_existing": True,
                    }
            checkpoint_path = save_path.with_suffix(".ckpt.npy")
            checkpoint_meta_path = save_path.with_suffix(".ckpt.meta.json")
            # Check for existing checkpoint
            if checkpoint_path.exists():
                if checkpoint_meta_path.exists():
                    meta = json.loads(checkpoint_meta_path.read_text(encoding="utf-8"))
                    if meta.get("signature") == signature:
                        resume_batch = meta.get("resume_batch", 0)
                        print(f"DenseCache: resuming from batch {resume_batch}/{total_batches}", flush=True)
                        self._embeddings = np.load(str(checkpoint_path))
                    else:
                        resume_batch = 0
                    if resume_batch >= total_batches and self._embeddings is not None:
                        # Fully encoded, just rename
                        np.save(str(save_path.with_suffix(".npy")), self._embeddings)
                        save_path.with_suffix(".ids.json").write_text(
                            json.dumps(self._chunk_meta, ensure_ascii=False), encoding="utf-8"
                        )
                        save_path.with_suffix(".manifest.json").write_text(
                            json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8"
                        )
                        checkpoint_meta_path.unlink(missing_ok=True)
                        checkpoint_path.unlink(missing_ok=True)
                        elapsed = time.perf_counter() - started
                        return {"chunks": total_chunks, "embedding_dim": self._embeddings.shape[1], "seconds": elapsed, "resumed": True}
                else:
                    checkpoint_path = None

        model = self._load_model()
        # Allocate output array
        embed_dim = model.get_embedding_dimension() or 768
        if resume_batch > 0 and self._embeddings is not None and self._embeddings.shape[0] >= total_chunks:
            embeddings = self._embeddings
        else:
            embeddings = np.zeros((total_chunks, embed_dim), dtype=np.float32)
            if resume_batch > 0 and self._embeddings is not None:
                embeddings[: self._embeddings.shape[0]] = self._embeddings
                del self._embeddings

        # Encode remaining batches
        batches_since_ckpt = 0
        for batch_start in range(0, total_chunks, self.batch_size):
            batch_idx = batch_start // self.batch_size
            if batch_idx < resume_batch:
                continue
            batch_end = min(batch_start + self.batch_size, total_chunks)
            batch_texts = [
                self._passage_text(chunk)
                for chunk in chunks[batch_start:batch_end]
            ]
            batch_emb = model.encode(
                batch_texts,
                batch_size=len(batch_texts),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
            embeddings[batch_start:batch_end] = batch_emb

            # Progress log every 50 batches
            if (batch_idx + 1) % 50 == 0:
                elapsed = time.perf_counter() - started
                rate = (batch_end) / elapsed if elapsed > 0 else 0
                eta = (total_chunks - batch_end) / rate if rate > 0 else 0
                print(f"DenseCache progress: {batch_end}/{total_chunks} chunks ({batch_end*100//total_chunks}%, batch {batch_idx+1}/{total_batches}, {elapsed:.0f}s elapsed, ETA {eta:.0f}s)", flush=True)

            batches_since_ckpt += 1
            done_chunks = min(batch_end, total_chunks)
            if batches_since_ckpt >= checkpoint_every:
                # Save checkpoint
                ckpt_path = save_path.with_suffix(".ckpt.npy") if save_path else None
                if ckpt_path:
                    np.save(str(ckpt_path), embeddings)
                    save_path.with_suffix(".ckpt.meta.json").write_text(
                        json.dumps(
                            {
                                "resume_batch": batch_idx + 1,
                                "total_chunks": total_chunks,
                                "signature": signature,
                            },
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                print(f"DenseCache checkpoint {done_chunks}/{total_chunks} chunks (batch {batch_idx+1}/{total_batches}) saved", flush=True)
                batches_since_ckpt = 0

        self._embeddings = embeddings
        elapsed = time.perf_counter() - started

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            # Save final
            np.save(str(save_path.with_suffix(".npy")), embeddings)
            save_path.with_suffix(".ids.json").write_text(
                json.dumps(self._chunk_meta, ensure_ascii=False), encoding="utf-8"
            )
            save_path.with_suffix(".manifest.json").write_text(
                json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            # Remove checkpoint files
            for path in (
                save_path.with_suffix(".ckpt.npy"),
                save_path.with_suffix(".ckpt.meta.json"),
            ):
                path.unlink(missing_ok=True)

        return {"chunks": total_chunks, "embedding_dim": embeddings.shape[1], "seconds": elapsed}

    def load(
        self,
        path: Path,
        expected_signature: Optional[Dict[str, Any]] = None,
    ) -> None:
        path = Path(path)
        manifest_path = path.with_suffix(".manifest.json")
        if not manifest_path.is_file():
            raise ValueError(f"Dense cache has no compatibility manifest: {manifest_path}")
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved.get("runtime") != self.runtime_signature():
            raise ValueError("Dense cache was built with a different model or encoding configuration")
        if expected_signature is not None and saved != expected_signature:
            raise ValueError("Dense cache does not match the ordered corpus chunks")
        self._embeddings = np.load(str(path.with_suffix(".npy")))
        self._chunk_meta = json.loads(
            path.with_suffix(".ids.json").read_text(encoding="utf-8")
        )
        if len(self._chunk_meta) != len(self._embeddings):
            raise ValueError("Dense cache IDs and embedding row counts differ")

    def encode_query(self, query: str) -> np.ndarray:
        """Encode one query once so document and question caches can share it."""
        model = self._load_model()
        return model.encode(
            [f"{self.query_prefix}{query}" if self.query_prefix else query],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0].astype(np.float32)

    def retrieve_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = 40,
    ) -> List[Dict[str, Any]]:
        """Retrieve chunks using a precomputed normalized query vector."""
        if self._embeddings is None:
            raise RuntimeError("DenseCache: call build() or load() first")
        scores = self._embeddings @ query_embedding
        limit = min(int(top_k), len(scores))
        if limit <= 0:
            return []
        if limit == len(scores):
            top_indices = np.argsort(-scores, kind="stable")
        else:
            top_indices = np.argpartition(-scores, limit - 1)[:limit]
            top_indices = top_indices[np.argsort(-scores[top_indices], kind="stable")]
        results = []
        for idx in top_indices:
            chunk_id = self._chunk_meta[idx]
            if "::chunk::" in chunk_id:
                doc_id = chunk_id.split("::chunk::", 1)[0]
            else:
                doc_id = chunk_id
            results.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "text": "",
                "score": float(scores[idx]),
                "raw_fts_score": float(scores[idx]),
                "method": "dense_embedding",
            })
        return results

    def retrieve(
        self,
        query: str,
        top_k: int = 40,
    ) -> List[Dict[str, Any]]:
        return self.retrieve_embedding(self.encode_query(query), top_k=top_k)

    def rrf_fuse(
        self,
        fts_candidates: List[Dict[str, Any]],
        dense_candidates: List[Dict[str, Any]],
        fts_weight: float = 1.0,
        dense_weight: float = 1.0,
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        by_id: Dict[str, Dict[str, Any]] = {}
        fused: Dict[str, float] = defaultdict(float)
        for weight, ranking in ((fts_weight, fts_candidates), (dense_weight, dense_candidates)):
            seen: set[str] = set()
            rank = 0
            for item in ranking:
                doc_id = item["doc_id"]
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                rank += 1
                fused[doc_id] += weight / (rrf_k + rank)
                if doc_id not in by_id:
                    by_id[doc_id] = dict(item)
        for doc_id, score in fused.items():
            by_id[doc_id]["score"] = score
        return sorted(
            by_id.values(),
            key=lambda item: (-float(item["score"]), item["doc_id"]),
        )

    @staticmethod
    def rrf_fuse_many(
        rankings: Sequence[Tuple[float, Sequence[Dict[str, Any]], int]],
    ) -> List[Dict[str, Any]]:
        """Fuse each independent ranking once, with an explicit weight/k.

        Sequential pairwise RRF unintentionally re-ranks and re-counts the
        lexical branch every time another neural branch is added.  A single
        multi-way pass keeps ablations and tuned weights mathematically stable.
        """

        by_id: Dict[str, Dict[str, Any]] = {}
        fused: Dict[str, float] = defaultdict(float)
        methods: Dict[str, set[str]] = defaultdict(set)
        for weight, ranking, rrf_k in rankings:
            if weight <= 0:
                continue
            seen: set[str] = set()
            rank = 0
            for item in ranking:
                doc_id = str(item["doc_id"])
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                rank += 1
                fused[doc_id] += float(weight) / (int(rrf_k) + rank)
                methods[doc_id].add(str(item.get("method", "unknown")))
                if doc_id not in by_id:
                    by_id[doc_id] = dict(item)
        for doc_id, score in fused.items():
            by_id[doc_id]["score"] = score
            by_id[doc_id]["method"] = "+".join(sorted(methods[doc_id]))
        return sorted(
            by_id.values(),
            key=lambda item: (-float(item["score"]), item["doc_id"]),
        )


class QuestionDenseCache:
    """Supervised query-neighbor cache backed by labelled train questions.

    This is deliberately separate from passage embeddings: a nearest labelled
    question contributes its known document IDs, while lexical/ passage search
    remains the primary ranking signal.  The cache is built only from the
    supplied training split and never from public answers.
    """

    def __init__(self, query_prefix: str = ""):
        self.query_prefix = query_prefix
        self._embeddings: Optional[np.ndarray] = None
        self._meta: List[Dict[str, Any]] = []

    @staticmethod
    def _build_signature(
        rows: Sequence[Dict[str, Any]], embedder: DenseCache
    ) -> Dict[str, Any]:
        digest = hashlib.sha256()
        for row in rows:
            for value in (
                row.get("question_id", ""),
                row.get("question", ""),
                json.dumps(row.get("answers", []), ensure_ascii=False, separators=(",", ":")),
            ):
                digest.update(str(value).encode("utf-8", errors="surrogatepass"))
                digest.update(b"\0")
        return {
            "schema": 2,
            "runtime": embedder.runtime_signature(),
            "rows": len(rows),
            "ordered_content_sha256": digest.hexdigest(),
        }

    def build(
        self,
        rows: List[Dict[str, Any]],
        embedder: DenseCache,
        save_path: Optional[Path] = None,
        checkpoint_every: int = 250,
    ) -> Dict[str, Any]:
        model = embedder._load_model()
        total = len(rows)
        batch_size = embedder.batch_size
        total_batches = (total + batch_size - 1) // batch_size
        started = time.perf_counter()
        signature = self._build_signature(rows, embedder)
        self._meta = [{
            "question_id": str(row["question_id"]),
            "answers": [str(value) for value in row["answers"]],
        } for row in rows]
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            final_embeddings = save_path.with_suffix(".npy")
            final_meta = save_path.with_suffix(".meta.json")
            final_manifest = save_path.with_suffix(".manifest.json")
            if final_embeddings.exists() and final_meta.exists() and final_manifest.exists():
                saved_signature = json.loads(final_manifest.read_text(encoding="utf-8"))
                if saved_signature == signature:
                    self.load(save_path, expected_signature=signature)
                    return {
                        "questions": total,
                        "embedding_dim": int(self._embeddings.shape[1]),
                        "seconds": 0.0,
                        "skipped_existing": True,
                    }

        embed_dim = model.get_embedding_dimension() or 768
        resume_batch = 0
        embeddings = np.zeros((total, embed_dim), dtype=np.float32)
        if save_path:
            checkpoint = save_path.with_suffix(".ckpt.npy")
            checkpoint_meta = save_path.with_suffix(".ckpt.meta.json")
            if checkpoint.exists() and checkpoint_meta.exists():
                saved = json.loads(checkpoint_meta.read_text(encoding="utf-8"))
                if saved.get("signature") == signature:
                    candidate = np.load(str(checkpoint))
                    if candidate.shape == embeddings.shape:
                        embeddings = candidate
                        resume_batch = int(saved.get("resume_batch", 0))
                        print(
                            f"QuestionDenseCache: resuming from batch {resume_batch}/{total_batches}",
                            flush=True,
                        )
        for batch_start in range(0, total, batch_size):
            batch_index = batch_start // batch_size
            if batch_index < resume_batch:
                continue
            batch_end = min(batch_start + batch_size, total)
            texts = [
                f"{self.query_prefix}{row['question']}" if self.query_prefix else str(row["question"])
                for row in rows[batch_start:batch_end]
            ]
            embeddings[batch_start:batch_end] = model.encode(
                texts,
                batch_size=len(texts),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
            if (batch_index + 1) % 50 == 0:
                print(
                    f"QuestionDenseCache progress: {batch_end}/{total} questions",
                    flush=True,
                )
            if save_path and checkpoint_every > 0 and (batch_index + 1) % checkpoint_every == 0:
                np.save(str(save_path.with_suffix(".ckpt.npy")), embeddings)
                save_path.with_suffix(".ckpt.meta.json").write_text(
                    json.dumps(
                        {
                            "resume_batch": batch_index + 1,
                            "total": total,
                            "signature": signature,
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

        self._embeddings = embeddings
        if save_path:
            np.save(str(save_path.with_suffix(".npy")), embeddings)
            save_path.with_suffix(".meta.json").write_text(
                json.dumps(self._meta, ensure_ascii=False), encoding="utf-8"
            )
            save_path.with_suffix(".manifest.json").write_text(
                json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            for path in (
                save_path.with_suffix(".ckpt.npy"),
                save_path.with_suffix(".ckpt.meta.json"),
            ):
                path.unlink(missing_ok=True)
        return {
            "questions": total,
            "embedding_dim": int(embeddings.shape[1]),
            "seconds": time.perf_counter() - started,
        }

    def load(
        self,
        path: Path,
        expected_signature: Optional[Dict[str, Any]] = None,
        expected_runtime: Optional[Dict[str, Any]] = None,
    ) -> None:
        path = Path(path)
        manifest_path = path.with_suffix(".manifest.json")
        if not manifest_path.is_file():
            raise ValueError(f"Question cache has no compatibility manifest: {manifest_path}")
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if expected_signature is not None and saved != expected_signature:
            raise ValueError("Question cache does not match the ordered training questions")
        if expected_runtime is not None and saved.get("runtime") != expected_runtime:
            raise ValueError("Question cache was built with a different embedding configuration")
        self._embeddings = np.load(str(path.with_suffix(".npy")))
        self._meta = json.loads(
            path.with_suffix(".meta.json").read_text(encoding="utf-8")
        )
        if len(self._meta) != len(self._embeddings):
            raise ValueError("Question cache metadata and embedding row counts differ")

    def retrieve_embedding(
        self,
        query_embedding: np.ndarray,
        top_k_questions: int = 20,
    ) -> List[Dict[str, Any]]:
        if self._embeddings is None:
            raise RuntimeError("QuestionDenseCache: call build() or load() first")
        limit = min(int(top_k_questions), len(self._embeddings))
        if limit <= 0:
            return []
        scores = self._embeddings @ query_embedding
        indices = np.argpartition(-scores, limit - 1)[:limit]
        indices = indices[np.argsort(-scores[indices], kind="stable")]
        seen: set[str] = set()
        results: List[Dict[str, Any]] = []
        for index in indices:
            row = self._meta[int(index)]
            for doc_id in row["answers"]:
                doc_id = str(doc_id)
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                results.append({
                    "doc_id": doc_id,
                    "chunk_id": str(row["question_id"]),
                    "text": "",
                    "score": float(scores[int(index)]),
                    "raw_fts_score": float(scores[int(index)]),
                    "method": "dense_question_neighbor",
                })
        return results


def normalize_question(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", str(text)).casefold().split())


def link_text(link: Any) -> str:
    """Turn a legal URL slug into searchable words without making a request."""
    if not isinstance(link, str) or not link:
        return ""
    parsed = urllib.parse.urlparse(link)
    decoded = urllib.parse.unquote(parsed.path)
    decoded = re.sub(r"[-_/]+", " ", decoded)
    decoded = re.sub(r"\b(?:html?|aspx?)\b", " ", decoded, flags=re.IGNORECASE)
    return " ".join(decoded.split())


def display_title(record: Dict[str, Any]) -> str:
    name = record.get("name")
    parts = [name.strip()] if isinstance(name, str) and name.strip() else []
    url_words = link_text(record.get("link"))
    if url_words:
        parts.append(url_words)
    return " | ".join(parts)


def iter_contexts(source: Path) -> Iterator[Dict[str, Any]]:
    """Yield valid official records without retaining the corpus in memory."""
    source = Path(source)
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source, "r") as archive:
            for item in sorted(archive.infolist(), key=lambda info: info.filename):
                if item.is_dir() or item.filename.startswith("__MACOSX/"):
                    continue
                if Path(item.filename).suffix.lower() != ".json":
                    continue
                record = json.loads(archive.read(item).decode("utf-8"))
                if isinstance(record, dict) and "id" in record:
                    yield record
        return
    if source.is_dir():
        for path in sorted(source.rglob("context_*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(record, dict) and "id" in record:
                yield record
        return
    raise ValueError(f"SQLite backend expects selected-contexts.zip or extracted directory: {source}")


def iter_text_chunks(text: str, chunk_size: int, overlap: int) -> Iterator[str]:
    """Yield bounded chunks without materialising a document-sized list."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")
    text = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return
    start = 0
    minimum_boundary = max(1, int(chunk_size * 0.6))
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = max(
                text.rfind("\n", start + minimum_boundary, end),
                text.rfind(" ", start + minimum_boundary, end),
            )
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)


def training_signals_from_data(
    data: Dict[str, Any],
) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, int]]:
    """Return doc aliases, exact question memory, and document label priors."""
    aliases: Dict[str, List[str]] = defaultdict(list)
    memory: Dict[str, List[str]] = defaultdict(list)
    priors: Dict[str, int] = defaultdict(int)
    for item in data.values():
        question = item.get("question")
        answers = item.get("answer")
        if not isinstance(question, str) or not isinstance(answers, list):
            continue
        normalized = normalize_question(question)
        for doc_id in map(str, answers):
            aliases[doc_id].append(question)
            priors[doc_id] += 1
            if doc_id not in memory[normalized]:
                memory[normalized].append(doc_id)
    alias_text = {
        doc_id: "\n".join(dict.fromkeys(questions))
        for doc_id, questions in aliases.items()
    }
    return alias_text, dict(memory), dict(priors)


def load_training_signals(path: Optional[Path]) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, int]]:
    if path is None:
        return {}, {}, {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Training data must be a JSON object keyed by question ID")
    return training_signals_from_data(data)


class SQLiteFTSRetriever:
    """Document-level BM25F index backed by SQLite FTS5.

    FTS5 streams to disk and therefore remains usable for the official corpus,
    unlike a dense ``chunks × vocabulary`` TF-IDF matrix.

    V60 Enhancements:
    - Legal-aware BM25F weights (title=5.0)
    - Enhanced chunk phrase weight (0.35)
    - Citation-aware retrieval
    - Tuned dense/reranker weights
    """

    BACKEND = "sqlite_chunk_fts5_v6_legal_enhanced"
    DOCUMENT_BASE_WEIGHT = 1.0
    DOCUMENT_PHRASE_WEIGHT = 0.35  # V60: increased from 0.25
    DOCUMENT_RRF_K = 0
    DOCUMENT_PHRASE_PAIR_LIMIT = 16
    DOCUMENT_PHRASE_RESULT_LIMIT = 80

    # Chunk-level phrase fusion: weight/k tuned leak-free on holdout 100.
    # V60: Slightly higher chunk phrase weight for better exact phrase matching
    CHUNK_PHRASE_BASE_WEIGHT = 1.0
    CHUNK_PHRASE_WEIGHT = 0.35  # V60: increased from 0.25
    CHUNK_PHRASE_RRF_K = 0
    CHUNK_PHRASE_PAIR_LIMIT = 16
    CHUNK_PHRASE_RESULT_LIMIT = 80

    # V60: Citation matching weight (strong signal for legal documents)
    CITATION_MATCH_WEIGHT = 2.0
    CITATION_RRF_K = 10

    def __init__(
        self,
        db_path: Path,
        use_reranker: bool = True,
        reranker_model: Optional[str] = None,
        device: Optional[str] = None,
        top_k_initial: int = 100,
        use_query_expansion: bool = False,
        # V60: Enhanced title weight for legal structure priority
        title_weight: float = 5.0,
        alias_weight: float = 2.0,
        passage_weight: float = 1.0,
        question_weight: float = 4.0,
        max_query_terms: int = 12,
        or_query_terms: int = 12,
        or_chunk_limit: int = 400,
        base_fusion_weight: float = 3.0,
        or_fusion_weight: float = 7.0,
        retrieval_rrf_k: int = 20,
        document_fts_db: Optional[Path] = None,
        chunk_size: int = 1400,
        chunk_overlap: int = 200,
        chunk_strategy: str = "fixed",
        enable_chunk_phrase: bool = True,
        chunk_phrase_weight: Optional[float] = None,
        chunk_phrase_pair_limit: Optional[int] = None,
        chunk_phrase_result_limit: Optional[int] = None,
        enable_dense: bool = False,
        dense_model_path: Optional[str] = None,
        dense_device: Optional[str] = None,
        dense_checkpoint_every: int = 500,
        dense_batch_size: int = 16,
        dense_max_seq_length: int = 384,
        dense_query_prefix: str = "",
        dense_passage_prefix: str = "",
        # V60: Dense weight kept small but tunable
        dense_weight: float = 0.08,
        dense_rrf_k: int = 60,
        dense_max_chunks_per_doc: int = 4,
        question_dense_weight: float = 0.02,
        question_dense_rrf_k: int = 60,
        question_dense_candidates: int = 20,
        # V60: Increased candidates for better reranking
        rerank_candidates: int = 50,
        # V60: More passages per doc for better coverage
        rerank_passages_per_doc: int = 3,
        reranker_batch_size: int = 16,
        reranker_max_length: Optional[int] = None,
        reranker_aggregation: str = "max",
        reranker_fp16: bool = False,
        adaptive_rerank: bool = False,
        normalize_corpus: bool = False,
        clean_training_data: bool = False,
        # V60: Slightly higher reranker weight
        reranker_rrf_weight: float = 0.85,
        reranker_rrf_k: int = 5,
    ):
        self.db_path = Path(db_path)
        self.use_reranker = use_reranker
        self.reranker_model = reranker_model
        self.device = device
        self.top_k_initial = int(top_k_initial)
        self.use_query_expansion = use_query_expansion
        self.title_weight = float(title_weight)
        self.alias_weight = float(alias_weight)
        self.passage_weight = float(passage_weight)
        self.question_weight = float(question_weight)
        self.enable_chunk_phrase = bool(enable_chunk_phrase)
        self.chunk_phrase_weight = (
            float(chunk_phrase_weight) if chunk_phrase_weight is not None
            else self.CHUNK_PHRASE_WEIGHT
        )
        self.chunk_phrase_pair_limit = (
            int(chunk_phrase_pair_limit) if chunk_phrase_pair_limit is not None
            else self.CHUNK_PHRASE_PAIR_LIMIT
        )
        self.chunk_phrase_result_limit = (
            int(chunk_phrase_result_limit) if chunk_phrase_result_limit is not None
            else self.CHUNK_PHRASE_RESULT_LIMIT
        )
        if max_query_terms <= 0:
            raise ValueError("max_query_terms must be positive")
        self.max_query_terms = int(max_query_terms)
        if or_query_terms <= 0 or or_chunk_limit <= 0 or retrieval_rrf_k < 0:
            raise ValueError("OR retrieval limits must be positive and RRF k non-negative")
        if base_fusion_weight < 0 or or_fusion_weight < 0:
            raise ValueError("Fusion weights must be non-negative")
        self.or_query_terms = int(or_query_terms)
        self.or_chunk_limit = int(or_chunk_limit)
        self.base_fusion_weight = float(base_fusion_weight)
        self.or_fusion_weight = float(or_fusion_weight)
        self.retrieval_rrf_k = int(retrieval_rrf_k)
        self.document_fts_db = (
            Path(document_fts_db).resolve() if document_fts_db is not None else None
        )
        if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("Require chunk_size > chunk_overlap >= 0")
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        if chunk_strategy not in {"legal", "fixed"}:
            raise ValueError("chunk_strategy must be 'legal' or 'fixed'")
        self.chunk_strategy = str(chunk_strategy)
        self.tokenizer = VietnameseTokenizer(bigrams=False)
        self.enable_dense = bool(enable_dense)
        self.dense_device = dense_device or device or "cuda"
        self.dense_model_path = dense_model_path
        self._dense_checkpoint_every = int(dense_checkpoint_every)
        self.dense_batch_size = int(dense_batch_size)
        self.dense_max_seq_length = int(dense_max_seq_length)
        self.dense_query_prefix = dense_query_prefix
        self.dense_passage_prefix = dense_passage_prefix
        self.dense_weight = float(dense_weight)
        self.dense_rrf_k = int(dense_rrf_k)
        self.dense_max_chunks_per_doc = int(dense_max_chunks_per_doc)
        self.question_dense_weight = float(question_dense_weight)
        self.question_dense_rrf_k = int(question_dense_rrf_k)
        self.question_dense_candidates = int(question_dense_candidates)
        self.rerank_candidates = int(rerank_candidates)
        self.rerank_passages_per_doc = int(rerank_passages_per_doc)
        self.reranker_batch_size = int(reranker_batch_size)
        self.reranker_max_length = (
            int(reranker_max_length) if reranker_max_length is not None else None
        )
        self.reranker_aggregation = str(reranker_aggregation)
        self.reranker_fp16 = bool(reranker_fp16)
        self.adaptive_rerank = bool(adaptive_rerank)
        self.normalize_corpus = bool(normalize_corpus)
        self.clean_training_data = bool(clean_training_data)
        self.reranker_rrf_weight = float(reranker_rrf_weight)
        self.reranker_rrf_k = int(reranker_rrf_k)
        self.last_stage1_candidates: List[Dict[str, Any]] = []
        if self.dense_weight < 0 or self.dense_rrf_k < 0 or self.dense_max_chunks_per_doc <= 0:
            raise ValueError("Dense weight must be non-negative and dense RRF k non-negative")
        if (
            self.question_dense_weight < 0
            or self.question_dense_rrf_k < 0
            or self.question_dense_candidates <= 0
        ):
            raise ValueError("Question dense weight/RRF must be non-negative and candidates positive")
        if (
            self.rerank_candidates <= 0
            or self.rerank_passages_per_doc <= 0
            or self.reranker_batch_size <= 0
            or int(dense_batch_size) <= 0
        ):
            raise ValueError("Dense batch size and rerank candidate count must be positive")
        if self.reranker_rrf_weight < 0 or self.reranker_rrf_k < 0:
            raise ValueError("Reranker RRF weight and k must be non-negative")
        self._dense: Optional[DenseCache] = None
        self._question_dense: Optional[QuestionDenseCache] = None
        if enable_dense and dense_model_path:
            self._dense = DenseCache(
                dense_model_path,
                device=self.dense_device,
                batch_size=self.dense_batch_size,
                max_seq_length=self.dense_max_seq_length,
                query_prefix=self.dense_query_prefix,
                passage_prefix=self.dense_passage_prefix,
            )
            self._question_dense = QuestionDenseCache(query_prefix=self.dense_query_prefix)
        if use_reranker:
            self.reranker = Reranker(
                reranker_model,
                device=device,
                batch_size=self.reranker_batch_size,
                max_length=self.reranker_max_length,
                passage_aggregation=self.reranker_aggregation,
                use_fp16=self.reranker_fp16,
            )
        else:
            self.reranker = None
        self._connection: Optional[sqlite3.Connection] = None
        self._document_connection: Optional[sqlite3.Connection] = None
        self._document_count: Optional[int] = None
        self._question_ngram_cache: Optional[Dict[str, Any]] = None
        # V60: Legal citation patterns
        self._citation_pattern = re.compile(
            r"(?<!\w)(?:Điều|Khoản|Điểm|Mục)\s+\d+[a-zđ]?(?:\s*[.,:;]\s*\d+[a-zđ]?)*",
            re.IGNORECASE
        )
        self._doc_number_pattern = re.compile(
            r"(?<!\w)\d{1,6}\s*[/\-]\s*\d{4}\s*[/\-]?"
            r"(?:NĐ|NQ|Luật|Decree|Law|QĐ|Circular)?\s*"
            r"[A-ZĐÂĂÊÔƠƯ]{0,3}(?:\s*[/\-]?\s*[A-ZĐÂĂÊÔƠƯ]{0,3}){0,2}(?!\w)",
            re.IGNORECASE
        )

    def _connect(self, readonly: bool = False) -> sqlite3.Connection:
        if readonly:
            uri = self.db_path.resolve().as_uri() + "?mode=ro"
            # Parallel Stage-1 benchmarking owns one connection per worker
            # and closes them from the coordinator after futures complete.
            connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
        else:
            connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        # The official corpus database is close to 1 GB and every query fans
        # out over several FTS5 virtual tables. Keep the read-only inference
        # connection warm without changing any ranking signal. SQLite accepts
        # these pragmas on older builds too; a read-only build may reject one,
        # so fall back to the default safely.
        if readonly:
            for pragma in (
                "PRAGMA cache_size=-524288",
                "PRAGMA temp_store=MEMORY",
                "PRAGMA mmap_size=2147483648",
            ):
                try:
                    connection.execute(pragma)
                except sqlite3.DatabaseError:
                    pass
        return connection

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            if not self.db_path.exists():
                raise FileNotFoundError(f"SQLite index not found: {self.db_path}")
            self._connection = self._connect(readonly=True)
        return self._connection

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._document_connection is not None:
            self._document_connection.close()
            self._document_connection = None

    @property
    def document_connection(self) -> sqlite3.Connection:
        """Open the optional one-row-per-document FTS database lazily/read-only."""
        if self.document_fts_db is None:
            raise RuntimeError("Document FTS is not enabled")
        if self._document_connection is None:
            if not self.document_fts_db.is_file():
                raise FileNotFoundError(
                    f"Document FTS database not found: {self.document_fts_db}"
                )
            uri = self.document_fts_db.as_uri() + "?mode=ro"
            connection = sqlite3.connect(uri, uri=True)
            connection.row_factory = sqlite3.Row
            try:
                complete = connection.execute(
                    "SELECT value FROM metadata WHERE key = 'complete'"
                ).fetchone()
                count = int(
                    connection.execute("SELECT count(*) FROM documents").fetchone()[0]
                )
                if complete is None or str(complete[0]) != "1" or count <= 0:
                    raise ValueError(
                        "Document FTS database is incomplete or contains no documents"
                    )
            except (sqlite3.Error, ValueError):
                connection.close()
                raise
            self._document_connection = connection
            self._document_count = count
        return self._document_connection

    def document_fts_info(self) -> Dict[str, Any]:
        """Return auditable optional-database and validated fusion settings."""
        if self.document_fts_db is None:
            return {"enabled": False}
        _ = self.document_connection
        return {
            "enabled": True,
            "path": str(self.document_fts_db),
            "documents": self._document_count,
            "base_weight": self.DOCUMENT_BASE_WEIGHT,
            "phrase_weight": self.DOCUMENT_PHRASE_WEIGHT,
            "rrf_k": self.DOCUMENT_RRF_K,
            "pair_limit": self.DOCUMENT_PHRASE_PAIR_LIMIT,
            "result_limit": self.DOCUMENT_PHRASE_RESULT_LIMIT,
        }

    def build(self, corpus: Path, train_data: Optional[Path] = None) -> Dict[str, Any]:
        self.close()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        building = self.db_path.with_suffix(self.db_path.suffix + ".building")
        if building.exists():
            building.unlink()
        if self.clean_training_data:
            training_records, training_quality = load_training_records(train_data)
        else:
            training_records = load_raw_training_records(train_data)
            training_quality = {
                "enabled": False,
                "input_records": len(training_records),
                "clean_records": len(training_records),
                "duplicate_question_rows_merged": 0,
                "conflicting_duplicate_groups_unioned": 0,
                "skipped_invalid_rows": 0,
            }
        aliases, memory, priors = training_signals_from_data(training_records)
        connection = sqlite3.connect(building)
        try:
            connection.executescript("""
                PRAGMA journal_mode=OFF;
                PRAGMA synchronous=OFF;
                PRAGMA temp_store=MEMORY;
                PRAGMA cache_size=-262144;
                CREATE VIRTUAL TABLE chunks USING fts5(
                    doc_id UNINDEXED,
                    chunk_id UNINDEXED,
                    title,
                    passage,
                    tokenize='unicode61 remove_diacritics 0'
                );
                CREATE TABLE chunk_structure(
                    chunk_id TEXT PRIMARY KEY,
                    unit_type TEXT NOT NULL,
                    article_number TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE aliases USING fts5(
                    doc_id UNINDEXED,
                    alias_text,
                    tokenize='unicode61 remove_diacritics 0'
                );
                CREATE VIRTUAL TABLE question_examples USING fts5(
                    question_id UNINDEXED,
                    answers UNINDEXED,
                    question,
                    tokenize='unicode61 remove_diacritics 0'
                );
                CREATE TABLE document_meta(doc_id TEXT PRIMARY KEY, title TEXT NOT NULL);
                CREATE TABLE document_prior(doc_id TEXT PRIMARY KEY, label_count INTEGER NOT NULL);
                CREATE TABLE question_memory(question TEXT PRIMARY KEY, answers TEXT NOT NULL);
                CREATE TABLE term_stats(term TEXT PRIMARY KEY, doc_freq INTEGER NOT NULL);
                CREATE TABLE manifest(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)
            inserted = 0
            inserted_chunks = 0
            empty_passages = 0
            started = time.perf_counter()
            for record in iter_contexts(Path(corpus)):
                doc_id = str(record["id"])
                passage = record.get("passage")
                passage = passage if isinstance(passage, str) else ""
                title = display_title(record)
                if self.normalize_corpus:
                    passage = normalize_corpus_text(passage)
                    title = normalize_corpus_text(title)
                if not passage.strip():
                    empty_passages += 1
                    passage = title or f"văn bản {doc_id}"
                connection.execute(
                    "INSERT INTO document_meta(doc_id, title) VALUES (?, ?)",
                    (doc_id, title),
                )
                alias_text = aliases.get(doc_id)
                if alias_text:
                    connection.execute(
                        "INSERT INTO aliases(doc_id, alias_text) VALUES (?, ?)",
                        (doc_id, alias_text),
                    )
                for position, chunk in enumerate(
                    iter_index_chunks(
                        passage,
                        self.chunk_size,
                        self.chunk_overlap,
                        self.chunk_strategy,
                    )
                ):
                    connection.execute(
                        "INSERT INTO chunks(doc_id, chunk_id, title, passage) VALUES (?, ?, ?, ?)",
                        (
                            doc_id,
                            f"{doc_id}::chunk::{position}",
                            title,
                            chunk.text,
                        ),
                    )
                    connection.execute(
                        """INSERT INTO chunk_structure(
                            chunk_id, unit_type, article_number
                        ) VALUES (?, ?, ?)""",
                        (
                            f"{doc_id}::chunk::{position}",
                            chunk.unit_type,
                            chunk.article_number,
                        ),
                    )
                    inserted_chunks += 1
                inserted += 1
                if inserted % 250 == 0:
                    connection.commit()
            connection.executemany(
                "INSERT INTO document_prior(doc_id, label_count) VALUES (?, ?)",
                sorted(priors.items()),
            )
            connection.executemany(
                "INSERT INTO question_memory(question, answers) VALUES (?, ?)",
                ((question, json.dumps(answers)) for question, answers in sorted(memory.items())),
            )
            connection.executemany(
                "INSERT INTO question_examples(question_id, answers, question) VALUES (?, ?, ?)",
                (
                    (str(qid), json.dumps(item.get("answer", [])), item["question"])
                    for qid, item in training_records.items()
                    if isinstance(item, dict) and isinstance(item.get("question"), str)
                ),
            )
            manifest = {
                "backend": self.BACKEND,
                "documents": inserted,
                "chunks": inserted_chunks,
                "empty_passages_with_surrogate": empty_passages,
                "training_alias_documents": len(aliases),
                "question_memory_entries": len(memory),
                "training_question_examples": len(training_records),
                "training_quality": training_quality,
                "build_seconds": time.perf_counter() - started,
                "title_weight": self.title_weight,
                "alias_weight": self.alias_weight,
                "passage_weight": self.passage_weight,
                "question_weight": self.question_weight,
                "or_query_terms": self.or_query_terms,
                "or_chunk_limit": self.or_chunk_limit,
                "base_fusion_weight": self.base_fusion_weight,
                "or_fusion_weight": self.or_fusion_weight,
                "retrieval_rrf_k": self.retrieval_rrf_k,
                "enable_chunk_phrase": self.enable_chunk_phrase,
                "chunk_phrase_weight": self.chunk_phrase_weight,
                "chunk_phrase_pair_limit": self.chunk_phrase_pair_limit,
                "chunk_phrase_result_limit": self.chunk_phrase_result_limit,
                "enable_dense": self.enable_dense,
                "dense_model_path": self.dense_model_path,
                "dense_device": self.dense_device,
                "dense_max_chunks_per_doc": self.dense_max_chunks_per_doc,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "chunk_strategy": self.chunk_strategy,
                "corpus": str(Path(corpus).resolve()),
                "train_data": str(Path(train_data).resolve()) if train_data else None,
            }
            connection.executemany(
                "INSERT INTO manifest(key, value) VALUES (?, ?)",
                ((key, json.dumps(value, ensure_ascii=False)) for key, value in manifest.items()),
            )
            connection.commit()
            connection.execute("INSERT INTO chunks(chunks) VALUES ('optimize')")
            connection.execute("INSERT INTO aliases(aliases) VALUES ('optimize')")
            connection.execute("INSERT INTO question_examples(question_examples) VALUES ('optimize')")
            connection.execute(
                "CREATE VIRTUAL TABLE chunk_vocab USING fts5vocab(chunks, 'row')"
            )
            connection.execute(
                "INSERT INTO term_stats(term, doc_freq) SELECT term, doc FROM chunk_vocab"
            )
            connection.execute("DROP TABLE chunk_vocab")
            connection.commit()
        finally:
            connection.close()
        building.replace(self.db_path)

        # Persist the index contract before the expensive optional dense pass.
        # If embedding is interrupted, --reuse-index can resume from the
        # checkpoint without rebuilding the 1 GB SQLite FTS database.
        self._write_config()

        # ── Dense embedding index ────────────────────────────────────────────
        if self._dense is not None:
            # Collect all chunks we just inserted (we already iterated them above,
            # so re-stream from corpus to avoid holding them all in RAM).
            all_chunks: List[Dict[str, Any]] = []
            for record in iter_contexts(Path(corpus)):
                doc_id = str(record["id"])
                passage = record.get("passage")
                passage = passage if isinstance(passage, str) else ""
                title = display_title(record)
                if self.normalize_corpus:
                    passage = normalize_corpus_text(passage)
                    title = normalize_corpus_text(title)
                if not passage.strip():
                    passage = title or f"văn bản {doc_id}"
                for chunk_id, chunk in enumerate(
                    iter_index_chunks(
                        passage,
                        self.chunk_size,
                        self.chunk_overlap,
                        self.chunk_strategy,
                    ),
                    start=1,
                ):
                    all_chunks.append({
                        "doc_id": doc_id,
                        "chunk_id": f"{doc_id}::chunk::{chunk_id - 1}",
                        "title": title,
                        "passage": chunk.text,
                        "unit_type": chunk.unit_type,
                        "article_number": chunk.article_number,
                    })
            all_chunks = self._select_dense_chunks(all_chunks)
            dense_save = self.db_path.parent / "dense_embeddings"
            checkpoint_every = getattr(self, "_dense_checkpoint_every", 500)
            print(f"Building dense index for {len(all_chunks)} chunks on {self.dense_device} …", flush=True)
            build_result = self._dense.build(all_chunks, save_path=dense_save, checkpoint_every=checkpoint_every)
            print(f"Dense index done in {build_result['seconds']:.1f}s", flush=True)
            manifest["dense_build"] = build_result
            manifest["dense_model"] = self.dense_model_path
            question_dense_build = self._build_question_dense_from_index()
            if question_dense_build is not None:
                manifest["question_dense_build"] = question_dense_build

        self._write_config()
        return manifest

    def build_dense_from_index(self) -> Dict[str, Any]:
        """Build only the dense cache from an existing SQLite index.

        This avoids rebuilding the 1+ GB FTS index when iterating on an
        embedding model or fusion weight. The lexical index remains the source
        of truth for chunk IDs and training signals.
        """
        if self._dense is None:
            raise ValueError("Dense model is required for --reuse-index")
        rows = self.connection.execute(
            "SELECT doc_id, chunk_id, title, passage FROM chunks ORDER BY rowid"
        )
        chunks = self._select_dense_chunks(dict(row) for row in rows)
        dense_save = self.db_path.parent / "dense_embeddings"
        dense_build = self._dense.build(
            chunks,
            save_path=dense_save,
            checkpoint_every=self._dense_checkpoint_every,
        )
        question_dense_build = self._build_question_dense_from_index()
        self._write_config()
        documents = self.connection.execute(
            "SELECT count(*) FROM document_meta"
        ).fetchone()[0]
        return {
            "backend": self.BACKEND,
            "documents": int(documents),
            "chunks": len(chunks),
            "build_seconds": 0.0,
            "dense_build": dense_build,
            "question_dense_build": question_dense_build,
        }

    def _build_question_dense_from_index(self) -> Optional[Dict[str, Any]]:
        if self._dense is None or self._question_dense is None:
            return None
        rows: List[Dict[str, Any]] = []
        for row in self.connection.execute(
            "SELECT question_id, answers, question FROM question_examples ORDER BY rowid"
        ):
            rows.append({
                "question_id": str(row["question_id"]),
                "answers": [str(value) for value in json.loads(row["answers"])],
                "question": str(row["question"]),
            })
        if not rows:
            return None
        return self._question_dense.build(
            rows,
            self._dense,
            save_path=self.db_path.parent / "question_dense_embeddings",
            checkpoint_every=self._dense_checkpoint_every,
        )

    def _select_dense_chunks(self, chunks: Iterator[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep evenly spaced chunks so dense search covers long documents."""
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for chunk in chunks:
            grouped[str(chunk["doc_id"])].append(chunk)
        selected: List[Dict[str, Any]] = []
        limit = self.dense_max_chunks_per_doc
        for doc_chunks in grouped.values():
            if len(doc_chunks) <= limit:
                selected.extend(doc_chunks)
                continue
            positions = np.linspace(0, len(doc_chunks) - 1, num=limit, dtype=int)
            selected.extend(doc_chunks[int(position)] for position in positions)
        return selected

    def _write_config(self) -> None:
        config = {
            "backend": self.BACKEND,
            "db_path": self.db_path.name,
            "use_reranker": self.use_reranker,
            "reranker_model": self.reranker_model,
            "top_k_initial": self.top_k_initial,
            "use_query_expansion": self.use_query_expansion,
            "title_weight": self.title_weight,
            "alias_weight": self.alias_weight,
            "passage_weight": self.passage_weight,
            "question_weight": self.question_weight,
            "enable_chunk_phrase": self.enable_chunk_phrase,
            "chunk_phrase_weight": self.chunk_phrase_weight,
            "chunk_phrase_pair_limit": self.chunk_phrase_pair_limit,
            "chunk_phrase_result_limit": self.chunk_phrase_result_limit,
            "max_query_terms": self.max_query_terms,
            "or_query_terms": self.or_query_terms,
            "or_chunk_limit": self.or_chunk_limit,
            "base_fusion_weight": self.base_fusion_weight,
            "or_fusion_weight": self.or_fusion_weight,
            "retrieval_rrf_k": self.retrieval_rrf_k,
            "enable_dense": self.enable_dense,
            "dense_model_path": self.dense_model_path,
            "dense_device": self.dense_device,
            "dense_batch_size": self.dense_batch_size,
            "dense_max_seq_length": self.dense_max_seq_length,
            "dense_query_prefix": self.dense_query_prefix,
            "dense_passage_prefix": self.dense_passage_prefix,
            "dense_weight": self.dense_weight,
            "dense_rrf_k": self.dense_rrf_k,
            "dense_max_chunks_per_doc": self.dense_max_chunks_per_doc,
            "question_dense_weight": self.question_dense_weight,
            "question_dense_rrf_k": self.question_dense_rrf_k,
            "question_dense_candidates": self.question_dense_candidates,
            "rerank_candidates": self.rerank_candidates,
            "rerank_passages_per_doc": self.rerank_passages_per_doc,
            "reranker_batch_size": self.reranker_batch_size,
            "reranker_max_length": self.reranker_max_length,
            "reranker_aggregation": self.reranker_aggregation,
            "reranker_fp16": self.reranker_fp16,
            "adaptive_rerank": self.adaptive_rerank,
            "normalize_corpus": self.normalize_corpus,
            "clean_training_data": self.clean_training_data,
            "reranker_rrf_weight": self.reranker_rrf_weight,
            "reranker_rrf_k": self.reranker_rrf_k,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "chunk_strategy": self.chunk_strategy,
        }
        self.db_path.with_suffix(".config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(
        cls,
        db_path: Path,
        device: Optional[str] = None,
        document_fts_db: Optional[Path] = None,
        reranker_model: Optional[str] = None,
        disable_reranker: bool = False,
        disable_dense: bool = False,
        rerank_candidates: Optional[int] = None,
        rerank_passages_per_doc: Optional[int] = None,
        reranker_batch_size: Optional[int] = None,
        reranker_max_length: Optional[int] = None,
        reranker_fp16: Optional[bool] = None,
        reranker_rrf_weight: Optional[float] = None,
        reranker_rrf_k: Optional[int] = None,
        reranker_aggregation: Optional[str] = None,
        adaptive_rerank: Optional[bool] = None,
        normalize_corpus: Optional[bool] = None,
        clean_training_data: Optional[bool] = None,
    ) -> "SQLiteFTSRetriever":
        db_path = Path(db_path)
        config = json.loads(db_path.with_suffix(".config.json").read_text(encoding="utf-8"))
        return cls(
            db_path=db_path,
            use_reranker=(
                False
                if disable_reranker
                else bool(config["use_reranker"] or reranker_model)
            ),
            reranker_model=(
                None
                if disable_reranker
                else (reranker_model or config.get("reranker_model"))
            ),
            device=device,
            top_k_initial=config["top_k_initial"],
            use_query_expansion=config["use_query_expansion"],
            # V60: Use config values with V60 defaults for missing keys
            title_weight=config["title_weight"] if "title_weight" in config else 5.0,
            alias_weight=config["alias_weight"],
            passage_weight=config["passage_weight"],
            question_weight=config.get("question_weight", 4.0),
            max_query_terms=config.get("max_query_terms", 12),
            or_query_terms=config.get("or_query_terms", 12),
            or_chunk_limit=config.get("or_chunk_limit", 400),
            base_fusion_weight=config.get("base_fusion_weight", 3.0),
            or_fusion_weight=config.get("or_fusion_weight", 7.0),
            retrieval_rrf_k=config.get("retrieval_rrf_k", 20),
            document_fts_db=document_fts_db,
            chunk_size=config.get("chunk_size", 1400),
            chunk_overlap=config.get("chunk_overlap", 200),
            chunk_strategy=config.get("chunk_strategy", "fixed"),
            enable_chunk_phrase=config.get("enable_chunk_phrase", True),
            chunk_phrase_weight=config.get("chunk_phrase_weight"),
            chunk_phrase_pair_limit=config.get("chunk_phrase_pair_limit"),
            chunk_phrase_result_limit=config.get("chunk_phrase_result_limit"),
            enable_dense=(config.get("enable_dense", False) and not disable_dense),
            dense_model_path=config.get("dense_model_path"),
            dense_device=device or config.get("dense_device"),
            dense_batch_size=config.get("dense_batch_size", 16),
            dense_max_seq_length=config.get("dense_max_seq_length", 384),
            dense_query_prefix=config.get("dense_query_prefix", ""),
            dense_passage_prefix=config.get("dense_passage_prefix", ""),
            # V60: Default 0.08 for dense weight
            dense_weight=config.get("dense_weight", 0.08),
            dense_rrf_k=config.get("dense_rrf_k", 60),
            dense_max_chunks_per_doc=config.get("dense_max_chunks_per_doc", 4),
            question_dense_weight=config.get("question_dense_weight", 0.0),
            question_dense_rrf_k=config.get("question_dense_rrf_k", 60),
            question_dense_candidates=config.get("question_dense_candidates", 20),
            # V60: Increased defaults for reranking
            rerank_candidates=(
                rerank_candidates if rerank_candidates is not None
                else config.get("rerank_candidates", 50)
            ),
            rerank_passages_per_doc=(
                rerank_passages_per_doc if rerank_passages_per_doc is not None
                else config.get("rerank_passages_per_doc", 3)
            ),
            reranker_batch_size=(
                reranker_batch_size if reranker_batch_size is not None
                else config.get("reranker_batch_size", 16)
            ),
            reranker_max_length=(
                reranker_max_length
                if reranker_max_length is not None
                else config.get("reranker_max_length")
            ),
            reranker_aggregation=(
                reranker_aggregation
                if reranker_aggregation is not None
                else config.get("reranker_aggregation", "max")
            ),
            reranker_fp16=(
                bool(reranker_fp16)
                if reranker_fp16 is not None
                else bool(config.get("reranker_fp16", False))
            ),
            adaptive_rerank=(
                bool(adaptive_rerank)
                if adaptive_rerank is not None
                else bool(config.get("adaptive_rerank", False))
            ),
            normalize_corpus=(
                bool(normalize_corpus)
                if normalize_corpus is not None
                else bool(config.get("normalize_corpus", False))
            ),
            clean_training_data=(
                bool(clean_training_data)
                if clean_training_data is not None
                else bool(config.get("clean_training_data", False))
            ),
            # V60: Default 0.85 for reranker weight
            reranker_rrf_weight=(
                reranker_rrf_weight if reranker_rrf_weight is not None
                else config.get("reranker_rrf_weight", 0.85)
            ),
            reranker_rrf_k=(
                reranker_rrf_k if reranker_rrf_k is not None
                else config.get("reranker_rrf_k", 5)
            ),
        )

    def manifest(self) -> Dict[str, Any]:
        return {
            row["key"]: json.loads(row["value"])
            for row in self.connection.execute("SELECT key, value FROM manifest")
        }

    def replace_training_signals(self, data: Dict[str, Any]) -> Dict[str, int]:
        """Replace the small supervised layer without rebuilding passage chunks."""
        self.close()
        self._question_ngram_cache = None
        aliases, memory, priors = training_signals_from_data(data)
        connection = self._connect(readonly=False)
        try:
            connection.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS question_examples USING fts5(
                    question_id UNINDEXED,
                    answers UNINDEXED,
                    question,
                    tokenize='unicode61 remove_diacritics 0'
                )
            """)
            connection.execute("DELETE FROM aliases")
            connection.execute("DELETE FROM question_examples")
            connection.execute("DELETE FROM document_prior")
            connection.execute("DELETE FROM question_memory")
            connection.executemany(
                "INSERT INTO aliases(doc_id, alias_text) VALUES (?, ?)", sorted(aliases.items())
            )
            connection.executemany(
                "INSERT INTO document_prior(doc_id, label_count) VALUES (?, ?)", sorted(priors.items())
            )
            connection.executemany(
                "INSERT INTO question_memory(question, answers) VALUES (?, ?)",
                ((question, json.dumps(answers)) for question, answers in sorted(memory.items())),
            )
            connection.executemany(
                "INSERT INTO question_examples(question_id, answers, question) VALUES (?, ?, ?)",
                (
                    (str(qid), json.dumps(item.get("answer", [])), item["question"])
                    for qid, item in data.items()
                    if isinstance(item, dict) and isinstance(item.get("question"), str)
                ),
            )
            connection.execute("INSERT INTO aliases(aliases) VALUES ('optimize')")
            connection.execute("INSERT INTO question_examples(question_examples) VALUES ('optimize')")
            for key, value in (
                ("training_alias_documents", len(aliases)),
                ("question_memory_entries", len(memory)),
                ("training_question_examples", len(data)),
            ):
                connection.execute(
                    "INSERT OR REPLACE INTO manifest(key, value) VALUES (?, ?)",
                    (key, json.dumps(value)),
                )
            connection.commit()
        finally:
            connection.close()
        return {
            "training_alias_documents": len(aliases),
            "question_memory_entries": len(memory),
            "prior_documents": len(priors),
            "training_question_examples": len(data),
        }

    def _exact_memory(self, query: str) -> List[str]:
        row = self.connection.execute(
            "SELECT answers FROM question_memory WHERE question = ?", (normalize_question(query),)
        ).fetchone()
        return [str(value) for value in json.loads(row["answers"])] if row else []

    def _match_expression(self, query: str) -> Tuple[str, List[str]]:
        tokens = list(dict.fromkeys(self.tokenizer.tokenize(query)))
        if tokens:
            placeholders = ",".join("?" for _ in tokens)
            try:
                frequencies = {
                    str(row["term"]): int(row["doc_freq"])
                    for row in self.connection.execute(
                        f"SELECT term, doc_freq FROM term_stats WHERE term IN ({placeholders})",
                        tokens,
                    )
                }
            except sqlite3.OperationalError:
                frequencies = {token: 0 for token in tokens}
            # Terms absent from the corpus cannot retrieve anything. Among the
            # rest, rare terms carry more legal-document identity and avoid
            # scanning huge posting lists for generic wording.
            positions = {token: position for position, token in enumerate(tokens)}
            tokens = sorted(
                frequencies,
                key=lambda token: (frequencies[token], positions[token]),
            )[: self.max_query_terms]
        # Escape a token as an FTS phrase. Tokenizer strips quotes/operators,
        # but double escaping keeps this safe for future tokenizer changes.
        expression = " OR ".join(
            f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens
        )
        return expression, tokens

    @staticmethod
    def _focused_expressions(tokens: List[str]) -> List[str]:
        quoted = [f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens]
        if len(quoted) == 1:
            return quoted
        if len(quoted) >= 3:
            # Requiring three of the four rarest terms keeps posting-list
            # intersections small while tolerating one paraphrased word.
            return [
                " AND ".join(parts)
                for parts in itertools.combinations(quoted[:4], 3)
            ]
        return [" AND ".join(quoted)]

    def _chunk_candidates(self, tokens: List[str]) -> List[Dict[str, Any]]:
        expressions = self._focused_expressions(tokens)
        accumulated: Dict[str, Dict[str, Any]] = {}
        per_expression = max(20, self.top_k_initial)
        for expression in expressions:
            rows = list(self.connection.execute(
                """
                SELECT doc_id,
                       chunk_id,
                       title,
                       snippet(chunks, 3, '', '', ' … ', 96) AS evidence,
                       -bm25(chunks, 0.0, 0.0, ?, ?) AS score
                FROM chunks
                WHERE chunks MATCH ?
                ORDER BY bm25(chunks, 0.0, 0.0, ?, ?), doc_id, chunk_id
                LIMIT ?
                """,
                (
                    self.title_weight, self.passage_weight, expression,
                    self.title_weight, self.passage_weight, per_expression,
                ),
            ))
            for rank, row in enumerate(rows, 1):
                doc_id = str(row["doc_id"])
                rrf = 1.0 / (60.0 + rank)
                current = accumulated.get(doc_id)
                if current is None:
                    accumulated[doc_id] = {
                        "doc_id": doc_id,
                        "chunk_id": str(row["chunk_id"]),
                        "text": "\n".join(
                            value for value in (row["title"], row["evidence"]) if value
                        ),
                        "score": rrf,
                        "raw_fts_score": float(row["score"]),
                        "method": "sqlite_chunk_fts5",
                        "evidence_kind": "chunk",
                    }
                else:
                    # Best focused-query rank is more reliable than rewarding
                    # generic documents that happen to match every combination.
                    current["score"] = max(float(current["score"]), rrf)
                    if float(row["score"]) > current["raw_fts_score"]:
                        current["raw_fts_score"] = float(row["score"])
                        current["chunk_id"] = str(row["chunk_id"])
                        current["text"] = "\n".join(
                            value for value in (row["title"], row["evidence"]) if value
                        )
        # Very short/OOV-heavy questions can yield too few triple matches.
        # Broaden only then, anchoring each pair on the rarest indexed term.
        if len(accumulated) < 10 and len(tokens) >= 2:
            anchor = f'"{tokens[0].replace(chr(34), chr(34) * 2)}"'
            for token in tokens[1:4]:
                other = f'"{token.replace(chr(34), chr(34) * 2)}"'
                rows = list(self.connection.execute(
                    """
                    SELECT doc_id, chunk_id, title,
                           snippet(chunks, 3, '', '', ' … ', 96) AS evidence,
                           -bm25(chunks, 0.0, 0.0, ?, ?) AS score
                    FROM chunks WHERE chunks MATCH ?
                    ORDER BY bm25(chunks, 0.0, 0.0, ?, ?), doc_id, chunk_id LIMIT ?
                    """,
                    (
                        self.title_weight, self.passage_weight,
                        f"{anchor} AND {other}",
                        self.title_weight, self.passage_weight, per_expression,
                    ),
                ))
                for rank, row in enumerate(rows, 1):
                    doc_id = str(row["doc_id"])
                    if doc_id not in accumulated:
                        accumulated[doc_id] = {
                            "doc_id": doc_id,
                            "chunk_id": str(row["chunk_id"]),
                            "text": "\n".join(
                                value for value in (row["title"], row["evidence"]) if value
                            ),
                            "score": 0.5 / (60.0 + rank),
                            "raw_fts_score": float(row["score"]),
                            "method": "sqlite_chunk_fts5_broad_fallback",
                            "evidence_kind": "chunk",
                        }
        ranked = sorted(
            accumulated.values(),
            key=lambda item: (-float(item["score"]), -float(item["raw_fts_score"]), item["doc_id"]),
        )
        return ranked[: self.top_k_initial]

    def _fuse_alias_candidates(
        self, tokens: List[str], candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not tokens or not self.manifest().get("training_alias_documents", 0):
            return candidates
        by_id = {item["doc_id"]: item for item in candidates}
        for expression in self._focused_expressions(tokens):
            rows = list(self.connection.execute(
                """
                SELECT doc_id,
                       snippet(aliases, 1, '', '', ' … ', 64) AS evidence,
                       -bm25(aliases, 0.0, 1.0) AS score
                FROM aliases WHERE aliases MATCH ?
                ORDER BY bm25(aliases, 0.0, 1.0), doc_id LIMIT ?
                """,
                (expression, self.top_k_initial),
            ))
            for rank, row in enumerate(rows, 1):
                doc_id = str(row["doc_id"])
                alias_rrf = self.alias_weight / (60.0 + rank)
                current = by_id.get(doc_id)
                if current is None:
                    by_id[doc_id] = {
                        "doc_id": doc_id,
                        "chunk_id": doc_id,
                        "text": str(row["evidence"] or ""),
                        "score": alias_rrf,
                        "raw_fts_score": float(row["score"]),
                        "method": "sqlite_training_alias",
                        "evidence_kind": "query",
                    }
                elif alias_rrf > float(current["score"]):
                    current["score"] = alias_rrf
                    current["method"] = "sqlite_chunk_alias_fusion"
        return sorted(
            by_id.values(),
            key=lambda item: (-float(item["score"]), -float(item["raw_fts_score"]), item["doc_id"]),
        )[: self.top_k_initial]

    def _fuse_question_neighbors(
        self, query: str, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Fuse labels from lexically nearest labelled training questions."""
        if not self.manifest().get("training_question_examples", 0):
            return candidates
        tokens = list(dict.fromkeys(self.tokenizer.tokenize(query)))
        if not tokens:
            return candidates
        expression = " OR ".join(
            f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens
        )
        try:
            rows = list(self.connection.execute(
                """
                SELECT question_id, answers, question,
                       -bm25(question_examples, 0.0, 0.0, 1.0) AS score
                FROM question_examples WHERE question_examples MATCH ?
                ORDER BY bm25(question_examples, 0.0, 0.0, 1.0), question_id
                LIMIT ?
                """,
                (expression, self.top_k_initial),
            ))
        except sqlite3.OperationalError:
            return candidates

        by_id = {item["doc_id"]: item for item in candidates}
        for rank, row in enumerate(rows, 1):
            neighbor_rrf = self.question_weight / (20.0 + rank)
            for doc_id in map(str, json.loads(row["answers"])):
                current = by_id.get(doc_id)
                if current is None:
                    by_id[doc_id] = {
                        "doc_id": doc_id,
                        "chunk_id": str(row["question_id"]),
                        "text": str(row["question"]),
                        "score": neighbor_rrf,
                        "raw_fts_score": float(row["score"]),
                        "method": "sqlite_train_question_knn",
                        "evidence_kind": "query",
                    }
                elif neighbor_rrf > float(current["score"]):
                    current["score"] = neighbor_rrf
                    current["raw_fts_score"] = max(
                        float(current["raw_fts_score"]), float(row["score"])
                    )
                    current["method"] = "sqlite_question_knn_fusion"
        return sorted(
            by_id.values(),
            key=lambda item: (-float(item["score"]), -float(item["raw_fts_score"]), item["doc_id"]),
        )[: self.top_k_initial]

    @staticmethod
    def _accent_fold(text: str) -> str:
        folded = "".join(
            character
            for character in unicodedata.normalize("NFD", str(text).casefold())
            if unicodedata.category(character) != "Mn"
        ).replace("đ", "d")
        return " ".join(re.findall(r"[a-z0-9]+", folded))

    @classmethod
    def _character_features(cls, text: str) -> set[str]:
        normalized = f" {cls._accent_fold(text)} "
        return {
            f"{width}:{normalized[index:index + width]}"
            for width in (3, 4)
            for index in range(max(0, len(normalized) - width + 1))
        }

    def _load_question_ngram_index(self) -> Dict[str, Any]:
        if self._question_ngram_cache is not None:
            return self._question_ngram_cache
        rows: List[Dict[str, Any]] = []
        postings: Dict[str, List[int]] = defaultdict(list)
        try:
            examples = self.connection.execute(
                "SELECT question_id, answers, question FROM question_examples"
            )
            for row in examples:
                features = self._character_features(str(row["question"]))
                index = len(rows)
                rows.append({
                    "question_id": str(row["question_id"]),
                    "answers": [str(value) for value in json.loads(row["answers"])],
                    "question": str(row["question"]),
                    "features": features,
                })
                for feature in features:
                    postings[feature].append(index)
        except sqlite3.OperationalError:
            pass
        self._question_ngram_cache = {"rows": rows, "postings": postings}
        return self._question_ngram_cache

    def _fuse_character_neighbors(
        self, query: str, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Viettel-style accent-folded 3/4-gram Jaccard, validated on LegalIR."""
        index = self._load_question_ngram_index()
        rows: List[Dict[str, Any]] = index["rows"]
        if not rows:
            return candidates
        query_features = self._character_features(query)
        intersections: Counter[int] = Counter()
        for feature in query_features:
            intersections.update(index["postings"].get(feature, ()))
        neighbors: List[Tuple[float, int]] = []
        for row_index, intersection in intersections.items():
            denominator = len(query_features) + len(rows[row_index]["features"]) - intersection
            neighbors.append((intersection / denominator if denominator else 0.0, row_index))
        neighbors.sort(key=lambda item: (-item[0], item[1]))

        ngram_docs: Dict[str, Dict[str, Any]] = {}
        for similarity, row_index in neighbors[:80]:
            row = rows[row_index]
            for doc_id in row["answers"]:
                current = ngram_docs.get(doc_id)
                if current is None or similarity > current["similarity"]:
                    ngram_docs[doc_id] = {
                        "similarity": similarity,
                        "question_id": row["question_id"],
                        "question": row["question"],
                    }
        ngram_ranked = sorted(
            ngram_docs,
            key=lambda doc_id: (-ngram_docs[doc_id]["similarity"], doc_id),
        )[: self.top_k_initial]

        by_id = {item["doc_id"]: dict(item) for item in candidates}
        combined: Dict[str, float] = {
            item["doc_id"]: 3.0 / (5.0 + rank)
            for rank, item in enumerate(candidates, 1)
        }
        for rank, doc_id in enumerate(ngram_ranked, 1):
            combined[doc_id] = combined.get(doc_id, 0.0) + 1.0 / (5.0 + rank)
            if doc_id not in by_id:
                source = ngram_docs[doc_id]
                by_id[doc_id] = {
                    "doc_id": doc_id,
                    "chunk_id": source["question_id"],
                    "text": source["question"],
                    "score": 0.0,
                    "raw_fts_score": source["similarity"],
                    "method": "train_question_char_ngram",
                    "evidence_kind": "query",
                }
            else:
                by_id[doc_id]["method"] = "sqlite_char_ngram_fusion"
        for doc_id, score in combined.items():
            by_id[doc_id]["score"] = score
        return sorted(
            by_id.values(),
            key=lambda item: (-float(item["score"]), item["doc_id"]),
        )[: self.top_k_initial]

    # =========================================================================
    # V60: Legal Citation and Document Number Matching
    # =========================================================================

    def _extract_legal_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract legal citations and document numbers from query.

        Returns dict with 'citations' and 'doc_numbers' lists.
        """
        # Extract legal references: Điều, Khoản, Điểm, Mục
        citations = self._citation_pattern.findall(query)

        # Extract document numbers: Nghị định 45/2021, Luật 37/QH15, etc.
        doc_numbers = self._doc_number_pattern.findall(query)

        return {
            "citations": [c.casefold().strip() for c in citations],
            "doc_numbers": [n.strip() for n in doc_numbers],
        }

    def _citation_candidates(self, query: str) -> List[Dict[str, Any]]:
        """Find documents containing legal citations from query.

        V60: Boost documents that contain citations referenced in the query.
        This helps when the question mentions "theo Điều 10 Luật X" but
        the actual answer is a different document that cites that reference.
        """
        entities = self._extract_legal_entities(query)
        citations = entities["citations"]
        doc_numbers = entities["doc_numbers"]

        if not citations and not doc_numbers:
            return []

        # Build FTS expression for citations and doc numbers
        terms = []
        for citation in citations:
            # Normalize citation for FTS matching
            normalized = re.sub(r"\s+", " ", citation).strip()
            terms.append(normalized)

        for doc_num in doc_numbers:
            # Clean up document number
            cleaned = re.sub(r"\s+", "", doc_num).strip()
            if cleaned:
                terms.append(cleaned)

        if not terms:
            return []

        # Query FTS for matching chunks
        all_candidates: Dict[str, Dict[str, Any]] = {}
        for term in terms:
            # Try exact phrase match
            escaped = f'"{term.replace(chr(34), chr(34) * 2)}"'
            try:
                rows = list(self.connection.execute(
                    """
                    SELECT doc_id, chunk_id, title,
                           snippet(chunks, 3, '', '', ' … ', 64) AS evidence,
                           -bm25(chunks, 0.0, 0.0, 3.0, 1.0) AS score
                    FROM chunks WHERE chunks MATCH ?
                    ORDER BY bm25(chunks, 0.0, 0.0, 3.0, 1.0), doc_id
                    LIMIT 20
                    """,
                    (escaped,),
                ))
                for row in rows:
                    doc_id = str(row["doc_id"])
                    if doc_id not in all_candidates:
                        all_candidates[doc_id] = {
                            "doc_id": doc_id,
                            "chunk_id": str(row["chunk_id"]),
                            "text": "\n".join(v for v in (row["title"], row["evidence"]) if v),
                            "score": self.CITATION_MATCH_WEIGHT / (self.CITATION_RRF_K + 1),
                            "raw_fts_score": float(row["score"]),
                            "method": "sqlite_citation_match",
                            "evidence_kind": "citation",
                            "matched_terms": [term],
                        }
                    else:
                        all_candidates[doc_id]["score"] = max(
                            all_candidates[doc_id]["score"],
                            self.CITATION_MATCH_WEIGHT / (self.CITATION_RRF_K + 1)
                        )
                        all_candidates[doc_id]["matched_terms"].append(term)
            except sqlite3.OperationalError:
                continue

        return sorted(
            all_candidates.values(),
            key=lambda x: (-float(x["score"]), x["doc_id"])
        )[: self.top_k_initial]

    def _fuse_citation_candidates(
        self,
        base_candidates: List[Dict[str, Any]],
        citation_candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Fuse citation matching into the main ranking.

        V60: Citation matches are strong legal signals and get dedicated
        RRF fusion with the base candidates.
        """
        if not citation_candidates:
            return base_candidates

        by_id = {item["doc_id"]: dict(item) for item in base_candidates}
        fused: Dict[str, float] = {}

        # Base candidates
        for rank, item in enumerate(base_candidates, 1):
            doc_id = item["doc_id"]
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (20.0 + rank)

        # Citation candidates
        for rank, item in enumerate(citation_candidates, 1):
            doc_id = item["doc_id"]
            fused[doc_id] = fused.get(doc_id, 0.0) + (
                self.CITATION_MATCH_WEIGHT / (self.CITATION_RRF_K + rank)
            )
            if doc_id not in by_id:
                by_id[doc_id] = dict(item)
            else:
                by_id[doc_id]["method"] = "sqlite_citation_fusion"

        for doc_id, score in fused.items():
            by_id[doc_id]["score"] = score

        return sorted(
            by_id.values(),
            key=lambda item: (-float(item["score"]), item["doc_id"]),
        )[: self.top_k_initial]

    def _or_bm25_candidates(self, tokens: List[str]) -> List[Dict[str, Any]]:
        """Retrieve a high-recall list with a bounded OR query over passage chunks.

        This deliberately remains separate from the focused-query branch.  The
        two rankings make different errors, which is what made their RRF fusion
        substantially stronger on the labelled holdout.
        """
        selected_tokens = tokens[: self.or_query_terms]
        if not selected_tokens:
            return []
        expression = " OR ".join(
            f'"{token.replace(chr(34), chr(34) * 2)}"' for token in selected_tokens
        )
        try:
            rows = self.connection.execute(
                """
                SELECT doc_id, chunk_id, title,
                       snippet(chunks, 3, '', '', ' … ', 96) AS evidence,
                       -bm25(chunks, 0.0, 0.0, 3.0, 1.0) AS score
                FROM chunks WHERE chunks MATCH ?
                ORDER BY bm25(chunks, 0.0, 0.0, 3.0, 1.0), doc_id
                LIMIT ?
                """,
                (expression, self.or_chunk_limit),
            )
            unique: Dict[str, Dict[str, Any]] = {}
            for row in rows:
                doc_id = str(row["doc_id"])
                if doc_id in unique:
                    continue
                unique[doc_id] = {
                    "doc_id": doc_id,
                    "chunk_id": str(row["chunk_id"]),
                    "text": "\n".join(
                        value for value in (row["title"], row["evidence"]) if value
                    ),
                    "score": float(row["score"]),
                    "raw_fts_score": float(row["score"]),
                    "method": "sqlite_or_bm25",
                    "evidence_kind": "chunk",
                }
                if len(unique) >= self.top_k_initial:
                    break
            return list(unique.values())
        except sqlite3.OperationalError:
            return []

    def _fuse_or_bm25(
        self,
        base_candidates: List[Dict[str, Any]],
        or_candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Fuse the validated supervised/focused and OR-BM25 rankings by RRF."""
        by_id = {item["doc_id"]: dict(item) for item in base_candidates}
        fused: Dict[str, float] = {}
        for weight, ranking in (
            (self.base_fusion_weight, base_candidates),
            (self.or_fusion_weight, or_candidates),
        ):
            for rank, item in enumerate(ranking, 1):
                doc_id = item["doc_id"]
                fused[doc_id] = fused.get(doc_id, 0.0) + weight / (
                    self.retrieval_rrf_k + rank
                )
                if doc_id not in by_id:
                    by_id[doc_id] = dict(item)
                elif ranking is or_candidates:
                    by_id[doc_id]["method"] = "sqlite_supervised_or_rrf"
                    by_id[doc_id]["raw_fts_score"] = max(
                        float(by_id[doc_id].get("raw_fts_score", 0.0)),
                        float(item.get("raw_fts_score", 0.0)),
                    )
        for doc_id, score in fused.items():
            by_id[doc_id]["score"] = score
        return sorted(
            by_id.values(),
            key=lambda item: (-float(item["score"]), item["doc_id"]),
        )[: self.top_k_initial]

    @staticmethod
    def _quoted_phrase(first: str, second: str) -> str:
        first = first.replace(chr(34), chr(34) * 2)
        second = second.replace(chr(34), chr(34) * 2)
        return f'"{first} {second}"'

    def _document_phrase_candidates(self, query: str) -> List[Dict[str, Any]]:
        """Rank documents containing exact adjacent query-token pairs.

        The expression, BM25 weights and limits intentionally match the
        leak-free v3 holdout benchmark.  This branch is disabled unless an
        explicit ``--document-fts-db`` is supplied.
        """
        if self.document_fts_db is None:
            return []
        tokens = list(dict.fromkeys(self.tokenizer.tokenize(query)))
        if len(tokens) < 2:
            return []
        pairs: Sequence[Tuple[str, str]] = list(
            dict.fromkeys(zip(tokens, tokens[1:]))
        )[: self.DOCUMENT_PHRASE_PAIR_LIMIT]
        expression = " OR ".join(
            self._quoted_phrase(first, second) for first, second in pairs
        )
        rows = self.document_connection.execute(
            """
            SELECT doc_id, -bm25(documents, 0.0, 3.0, 1.0) AS score
            FROM documents
            WHERE documents MATCH ?
            ORDER BY bm25(documents, 0.0, 3.0, 1.0), doc_id
            LIMIT ?
            """,
            (expression, self.DOCUMENT_PHRASE_RESULT_LIMIT),
        )
        return [
            {
                "doc_id": str(row["doc_id"]),
                "chunk_id": str(row["doc_id"]),
                "text": "",
                "score": float(row["score"]),
                "raw_fts_score": float(row["score"]),
                "method": "document_phrase_fts",
            }
            for row in rows
        ]

    def _fuse_document_phrase(
        self,
        base_candidates: List[Dict[str, Any]],
        phrase_candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Fuse the complete v2 ranking with document phrase RRF (1:0.25, k=0)."""
        if not phrase_candidates:
            return base_candidates
        by_id = {item["doc_id"]: dict(item) for item in base_candidates}
        fused: Dict[str, float] = defaultdict(float)
        for weight, ranking in (
            (self.DOCUMENT_BASE_WEIGHT, base_candidates),
            (self.DOCUMENT_PHRASE_WEIGHT, phrase_candidates),
        ):
            for rank, item in enumerate(ranking, 1):
                doc_id = item["doc_id"]
                fused[doc_id] += weight / (self.DOCUMENT_RRF_K + rank)
                if doc_id not in by_id:
                    by_id[doc_id] = dict(item)
        for doc_id, score in fused.items():
            by_id[doc_id]["score"] = score
            by_id[doc_id]["method"] = "sqlite_v2_document_phrase_rrf"
        return sorted(
            by_id.values(),
            key=lambda item: (-float(item["score"]), item["doc_id"]),
        )[: self.top_k_initial]

    def _chunk_phrase_candidates(self, query: str) -> List[Dict[str, Any]]:
        """Rank documents by bigram phrases found in any chunk of the corpus.

        Complements the document-level phrase retriever by exposing the
        specific chunk where the phrase lives, which is useful when the
        document is long (>=chunk_size). The expression, BM25 weights and
        limit intentionally mirror the validated v3 holdout benchmark so
        the only variable is the FTS5 table (chunks vs. documents).
        """
        if not self.enable_chunk_phrase:
            return []
        tokens = list(dict.fromkeys(self.tokenizer.tokenize(query)))
        if len(tokens) < 2:
            return []
        pairs: Sequence[Tuple[str, str]] = list(
            dict.fromkeys(zip(tokens, tokens[1:]))
        )[: self.chunk_phrase_pair_limit]
        expression = " OR ".join(
            self._quoted_phrase(first, second) for first, second in pairs
        )
        try:
            rows = self.connection.execute(
                """
                SELECT doc_id, chunk_id, title,
                       snippet(chunks, 3, '', '', ' … ', 96) AS evidence,
                       -bm25(chunks, 0.0, 3.0, 1.0) AS score
                FROM chunks
                WHERE chunks MATCH ?
                ORDER BY bm25(chunks, 0.0, 3.0, 1.0), doc_id, chunk_id
                LIMIT ?
                """,
               (expression, self.chunk_phrase_result_limit),
            )
        except sqlite3.OperationalError:
            return []
        # Collapse to one row per doc_id while keeping the highest-scoring chunk.
        unique: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            doc_id = str(row["doc_id"])
            current = unique.get(doc_id)
            raw_score = float(row["score"])
            if current is None or raw_score > float(current["raw_fts_score"]):
                unique[doc_id] = {
                    "doc_id": doc_id,
                    "chunk_id": str(row["chunk_id"]),
                    "text": "\n".join(
                        value for value in (row["title"], row["evidence"]) if value
                    ),
                    "score": raw_score,
                    "raw_fts_score": raw_score,
                    "method": "sqlite_chunk_phrase_fts",
                    "evidence_kind": "chunk",
                }
        return sorted(
            unique.values(),
            key=lambda item: (-float(item["raw_fts_score"]), item["doc_id"]),
        )[: self.top_k_initial]

    def _fuse_chunk_phrase(
        self,
        base_candidates: List[Dict[str, Any]],
        phrase_candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Fuse the complete v2 ranking with chunk-level phrase RRF (1:0.25, k=0).

        Symmetric with :meth:`_fuse_document_phrase` so the two phrase
        layers stack on the same base ranking. Phrase weight mirrors the
        conservative 0.25 chosen on the v3 holdout for document phrases;
        if both phrase layers are enabled, their additive contribution
        remains bounded.
        """
        if not phrase_candidates:
            return base_candidates
        by_id = {item["doc_id"]: dict(item) for item in base_candidates}
        fused: Dict[str, float] = defaultdict(float)
        for weight, ranking in (
            (self.CHUNK_PHRASE_BASE_WEIGHT, base_candidates),
            (self.CHUNK_PHRASE_WEIGHT, phrase_candidates),
        ):
            for rank, item in enumerate(ranking, 1):
                doc_id = item["doc_id"]
                fused[doc_id] += weight / (self.CHUNK_PHRASE_RRF_K + rank)
                if doc_id not in by_id:
                    by_id[doc_id] = dict(item)
        for doc_id, score in fused.items():
            by_id[doc_id]["score"] = score
            if phrase_candidates and doc_id in {
                item["doc_id"] for item in phrase_candidates
            }:
                by_id[doc_id]["method"] = "sqlite_v2_chunk_phrase_rrf"
        return sorted(
            by_id.values(),
            key=lambda item: (-float(item["score"]), item["doc_id"]),
        )[: self.top_k_initial]

    def _hydrate_candidate_text(self, candidates: List[Dict[str, Any]]) -> None:
        """Attach real chunk text before neural reranking.

        Several retrieval branches intentionally carry only IDs or snippets.
        Passing those empty snippets to a cross-encoder silently turns the
        reranker into a query-only scorer, which is worse than the lexical
        fallback. Resolve the exact chunk first and use the first chunk of the
        document only as a deterministic fallback.
        """
        if not candidates:
            return
        # Retrieval branches already carry a query-focused snippet whenever
        # possible.  Keep it: reloading by an overloaded question/chunk ID can
        # replace useful query-neighbour evidence with an unrelated corpus row.
        missing = [item for item in candidates if not str(item.get("text", "")).strip()]
        if not missing:
            return
        chunk_ids = list(dict.fromkeys(
            str(item.get("chunk_id", ""))
            for item in missing
            if item.get("chunk_id") and item.get("evidence_kind") != "query"
        ))
        by_chunk: Dict[str, str] = {}
        if chunk_ids:
            placeholders = ",".join("?" for _ in chunk_ids)
            rows = self.connection.execute(
                f"SELECT chunk_id, title, passage FROM chunks WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            )
            for row in rows:
                by_chunk[str(row["chunk_id"])] = "\n".join(
                    value for value in (row["title"], row["passage"]) if value
                )

        missing_doc_ids = list(dict.fromkeys(
            str(item["doc_id"])
            for item in missing
            if str(item.get("chunk_id", "")) not in by_chunk
        ))
        by_doc: Dict[str, str] = {}
        if missing_doc_ids:
            placeholders = ",".join("?" for _ in missing_doc_ids)
            query = f"""
                SELECT doc_id, title, passage
                FROM chunks
                WHERE doc_id IN ({placeholders})
                ORDER BY doc_id, chunk_id
            """
            rows = self.connection.execute(
                query,
                missing_doc_ids,
            )
            for row in rows:
                doc_id = str(row["doc_id"])
                if doc_id not in by_doc:
                    by_doc[doc_id] = "\n".join(
                        value for value in (row["title"], row["passage"]) if value
                    )

        for item in missing:
            chunk_id = str(item.get("chunk_id", ""))
            item["text"] = by_chunk.get(chunk_id) or by_doc.get(str(item["doc_id"]), item.get("text", ""))

    @staticmethod
    def _attach_source_passages(
        candidates: List[Dict[str, Any]],
        source_rankings: Sequence[Sequence[Dict[str, Any]]],
        passages_per_doc: int = 2,
    ) -> None:
        """Reuse evidence already produced by retrieval instead of rescanning FTS."""

        if passages_per_doc <= 0:
            raise ValueError("passages_per_doc must be positive")
        evidence: Dict[str, List[str]] = defaultdict(list)
        # Sources are supplied strongest-first. Keep unique passages and let
        # max-passage reranking choose among the complementary retrieval views.
        for ranking in source_rankings:
            for item in ranking:
                doc_id = str(item["doc_id"])
                text = str(item.get("text", "")).strip()
                if text and text not in evidence[doc_id]:
                    evidence[doc_id].append(text)
        for item in candidates:
            doc_id = str(item["doc_id"])
            own = str(item.get("text", "")).strip()
            passages = list(evidence.get(doc_id, ()))
            if own and own not in passages:
                passages.append(own)
            item["passages"] = passages[:passages_per_doc] or [own]

    def _hydrate_candidate_passages(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        passages_per_doc: Optional[int] = None,
    ) -> None:
        """Attach query-focused passages for max-passage neural reranking."""

        if not candidates:
            return
        limit = int(passages_per_doc or self.rerank_passages_per_doc)
        if limit <= 0:
            raise ValueError("passages_per_doc must be positive")
        self._hydrate_candidate_text(candidates)
        _, tokens = self._match_expression(query)
        selected: Dict[str, List[str]] = defaultdict(list)
        # Keep evidence already produced by the retrieval branches, then add
        # complementary full chunks from the same official document.
        for item in candidates:
            doc_id = str(item["doc_id"])
            for passage in item.get("passages", ()):
                value = str(passage).strip()
                if value and value not in selected[doc_id] and len(selected[doc_id]) < limit:
                    selected[doc_id].append(value)
        if tokens:
            expression = " OR ".join(
                f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens
            )
            doc_ids = list(dict.fromkeys(str(item["doc_id"]) for item in candidates))
            for start in range(0, len(doc_ids), 20):
                batch = doc_ids[start : start + 20]
                placeholders = ",".join("?" for _ in batch)
                rows = self.connection.execute(
                    f"""
                    SELECT doc_id, title, passage,
                           -bm25(chunks, 0.0, 0.0, 3.0, 1.0) AS score
                    FROM chunks
                    WHERE chunks MATCH ? AND doc_id IN ({placeholders})
                    ORDER BY bm25(chunks, 0.0, 0.0, 3.0, 1.0), doc_id, chunk_id
                    LIMIT ?
                    """,
                    (expression, *batch, len(batch) * 50),
                )
                for row in rows:
                    doc_id = str(row["doc_id"])
                    if len(selected[doc_id]) >= limit:
                        continue
                    text = "\n".join(
                        value for value in (row["title"], row["passage"]) if value
                    )
                    if text and text not in selected[doc_id]:
                        selected[doc_id].append(text)
        for item in candidates:
            doc_id = str(item["doc_id"])
            passages = selected.get(doc_id, [])
            fallback = str(item.get("text", ""))
            if fallback and fallback not in passages and len(passages) < limit:
                passages.append(fallback)
            item["passages"] = passages or [fallback]

    def retrieve(
        self,
        query: str,
        query_id: str = "",
        top_k: int = 5,
        query_embedding: Optional[np.ndarray] = None,
    ) -> QueryResult:
        if not 1 <= top_k <= 5:
            raise ValueError("BTC Task 1 permits 1 to 5 predictions")
        started = time.perf_counter()
        memory_answers = self._exact_memory(query)

        processed = expand_query(query) if self.use_query_expansion else query
        _, search_terms = self._match_expression(processed)
        focused_candidates = self._chunk_candidates(search_terms) if search_terms else []
        candidates = focused_candidates
        candidates = self._fuse_alias_candidates(search_terms, candidates)
        candidates = self._fuse_question_neighbors(query, candidates)
        candidates = self._fuse_character_neighbors(query, candidates)
        supervised_candidates = [dict(item) for item in candidates]
        or_candidates = self._or_bm25_candidates(search_terms)
        candidates = self._fuse_or_bm25(candidates, or_candidates)
        chunk_phrase_candidates = self._chunk_phrase_candidates(query)
        candidates = self._fuse_chunk_phrase(candidates, chunk_phrase_candidates)
        document_phrase_candidates = self._document_phrase_candidates(query)
        candidates = self._fuse_document_phrase(
            candidates, document_phrase_candidates
        )

        # ── V60: Citation matching branch ───────────────────────────────────
        # Boost documents containing legal citations referenced in the query
        citation_candidates = self._citation_candidates(query)
        candidates = self._fuse_citation_candidates(candidates, citation_candidates)

        # ── Dense branch ─────────────────────────────────────────────────────
        # DenseCache uses a checkpointed NumPy index, so FAISS is optional and
        # not required for a clean install on the competition machine.
        dense_candidates: Optional[List[Dict[str, Any]]] = None
        question_dense_candidates: Optional[List[Dict[str, Any]]] = None
        if self._dense is not None:
            if self._dense._embeddings is None:
                dense_path = self.db_path.parent / "dense_embeddings"
                self._dense.load(dense_path)
            if self._dense._embeddings is not None:
                if query_embedding is None:
                    query_embedding = self._dense.encode_query(query)
                dense_candidates = self._dense.retrieve_embedding(
                    query_embedding, top_k=self.top_k_initial
                )
                if self._question_dense is not None and self._question_dense._embeddings is None:
                    question_path = self.db_path.parent / "question_dense_embeddings"
                    if (
                        question_path.with_suffix(".npy").is_file()
                        and question_path.with_suffix(".meta.json").is_file()
                    ):
                        self._question_dense.load(
                            question_path,
                            expected_runtime=self._dense.runtime_signature(),
                        )
                if self._question_dense is not None and self._question_dense._embeddings is not None:
                    question_dense_candidates = self._question_dense.retrieve_embedding(
                        query_embedding,
                        top_k_questions=self.question_dense_candidates,
                    )
                candidates = self._dense.rrf_fuse_many((
                    (1.0, candidates, self.dense_rrf_k),
                    (self.dense_weight, dense_candidates or [], self.dense_rrf_k),
                    (
                        self.question_dense_weight,
                        question_dense_candidates or [],
                        self.question_dense_rrf_k,
                    ),
                ))

        if memory_answers:
            memory_ids = set(memory_answers)
            exact = [{
                "doc_id": doc_id,
                "chunk_id": doc_id,
                "text": "exact normalized training-question match",
                "score": 1_000_000.0 - rank,
                "raw_fts_score": 1_000_000.0 - rank,
                "method": "train_exact_memory",
            } for rank, doc_id in enumerate(memory_answers)]
            candidates = exact + [
                item for item in candidates if item["doc_id"] not in memory_ids
            ]
        # Preserve the pre-reranker candidate pool for leak-free oracle audits.
        # This is diagnostic state only and does not affect final ranking.
        self.last_stage1_candidates = [dict(item) for item in candidates]
        if self.reranker is not None:
            locked = candidates[:len(memory_answers)] if memory_answers else []
            locked_ids = {str(item["doc_id"]) for item in locked}
            shortlist = [
                item for item in candidates if str(item["doc_id"]) not in locked_ids
            ][: max(top_k, self.rerank_candidates)]
            self._attach_source_passages(
                shortlist,
                (
                    chunk_phrase_candidates,
                    focused_candidates,
                    or_candidates,
                    supervised_candidates,
                    dense_candidates or [],
                    question_dense_candidates or [],
                ),
                self.rerank_passages_per_doc,
            )
            if self.adaptive_rerank:
                # Stage A supplies one best hit per document. Stage B adds
                # neighboring/query-focused legal units before CrossEncoder
                # scoring, so a correct document is not lost due to a
                # two-passage quota.
                self._hydrate_candidate_passages(
                    query,
                    shortlist,
                    self.rerank_passages_per_doc,
                )
            reranked = self.reranker.rerank(
                query,
                shortlist,
                top_k=len(shortlist),
            )
            reranked = reciprocal_rank_fuse(
                shortlist,
                reranked,
                neural_weight=self.reranker_rrf_weight,
                rrf_k=self.reranker_rrf_k,
            )
            candidates = locked + reranked

        used_fallback = False
        if not candidates:
            used_fallback = True
            fallback_rows = list(self.connection.execute(
                "SELECT doc_id, label_count FROM document_prior ORDER BY label_count DESC, doc_id LIMIT ?",
                (top_k,),
            ))
            if not fallback_rows:
                fallback_rows = list(self.connection.execute(
                    "SELECT doc_id, 0 AS label_count FROM document_meta ORDER BY doc_id LIMIT ?",
                    (top_k,),
                ))
            candidates = [{
                "doc_id": str(row["doc_id"]), "chunk_id": str(row["doc_id"]),
                "text": "", "score": 0.0, "rank": rank,
                "method": "zero-score-fallback", "rerank_score": None,
            } for rank, row in enumerate(fallback_rows, 1)]

        selected = candidates[:top_k]
        self._hydrate_candidate_text(selected)
        selected_ids = {item["doc_id"] for item in selected}
        if len(selected) < top_k:
            fill_rows = list(self.connection.execute(
                """
                SELECT m.doc_id, COALESCE(p.label_count, 0) AS label_count
                FROM document_meta m LEFT JOIN document_prior p ON p.doc_id = m.doc_id
                ORDER BY label_count DESC, m.doc_id
                """
            ))
            for row in fill_rows:
                doc_id = str(row["doc_id"])
                if doc_id in selected_ids:
                    continue
                selected.append({
                    "doc_id": doc_id, "chunk_id": doc_id, "text": "",
                    "score": 0.0, "method": "recall_fill", "rerank_score": None,
                })
                selected_ids.add(doc_id)
                if len(selected) == top_k:
                    break
        results = [RetrievalResult(
            doc_id=item["doc_id"],
            text=item.get("text", ""),
            score=float(item.get("score", 0.0)),
            method=item.get("method", "sqlite_fts5"),
            rank=rank,
            chunk_id=item.get("chunk_id", item["doc_id"]),
            rerank_score=(
                float(item["rerank_score"]) if item.get("rerank_score") is not None else None
            ),
        ) for rank, item in enumerate(selected, 1)]
        return QueryResult(
            query_id=str(query_id), query=query, results=results,
            processing_time=time.perf_counter() - started,
            metadata={
                "backend": self.BACKEND,
                "candidates": len(candidates),
                "exact_memory": bool(memory_answers),
                "zero_score_fallback": used_fallback,
                "search_terms": search_terms,
                "or_candidates": len(or_candidates),
                "document_phrase": {
                    "enabled": self.document_fts_db is not None,
                    "candidates": len(document_phrase_candidates),
                    "base_weight": self.DOCUMENT_BASE_WEIGHT,
                    "phrase_weight": self.DOCUMENT_PHRASE_WEIGHT,
                    "rrf_k": self.DOCUMENT_RRF_K,
                },
                "chunk_phrase": {
                    "enabled": self.enable_chunk_phrase,
                    "candidates": len(chunk_phrase_candidates),
                    "base_weight": self.CHUNK_PHRASE_BASE_WEIGHT,
                    "phrase_weight": self.chunk_phrase_weight,
                    "rrf_k": self.CHUNK_PHRASE_RRF_K,
                    "pair_limit": self.chunk_phrase_pair_limit,
                    "result_limit": self.chunk_phrase_result_limit,
                },
                # V60: Citation matching
                "citation": {
                    "enabled": True,
                    "candidates": len(citation_candidates),
                    "weight": self.CITATION_MATCH_WEIGHT,
                    "rrf_k": self.CITATION_RRF_K,
                },
                "fusion": {
                    "base_weight": self.base_fusion_weight,
                    "or_weight": self.or_fusion_weight,
                    "rrf_k": self.retrieval_rrf_k,
                },
                "dense": {
                    "enabled": self.enable_dense,
                    "candidates": len(dense_candidates) if dense_candidates is not None else 0,
                    "weight": self.dense_weight,
                },
                "question_dense": {
                    "enabled": question_dense_candidates is not None,
                    "candidates": len(question_dense_candidates) if question_dense_candidates is not None else 0,
                    "weight": self.question_dense_weight,
                },
                "reranker": {
                    "enabled": self.reranker is not None,
                    "kind": "cross_encoder" if self.reranker_model else "none",
                    "candidates": self.rerank_candidates,
                    "passages_per_doc": self.rerank_passages_per_doc,
                    "rrf_weight": self.reranker_rrf_weight,
                    "rrf_k": self.reranker_rrf_k,
                    "trace": getattr(self.reranker, "last_trace", None),
                },
            },
        )
