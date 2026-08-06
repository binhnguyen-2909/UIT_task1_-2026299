"""Hybrid LegalIR pipeline: chunk retrieval, RRF, reranking and doc collapse."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # Package execution: python -m scripts.main
    from .bm25_retriever import BM25Retriever, expand_query
    from .dense_retriever import DenseRetriever, preprocess_query
    from .reranker import Reranker
    from .retrieval_types import QueryResult, RetrievalResult
except ImportError:  # Direct execution: python scripts/main.py
    from bm25_retriever import BM25Retriever, expand_query
    from dense_retriever import DenseRetriever, preprocess_query
    from reranker import Reranker
    from retrieval_types import QueryResult, RetrievalResult


class HybridRetriever:
    """Retrieve chunks in two views, fuse ranks, rerank, return unique BTC IDs."""

    def __init__(
        self,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
        use_reranker: bool = True,
        use_query_expansion: bool = False,
        top_k_initial: int = 60,
        top_k_final: int = 5,
        rrf_k: int = 60,
        embedding_model: Optional[str] = None,
        reranker_model: Optional[str] = None,
        device: Optional[str] = None,
    ):
        if bm25_weight < 0 or dense_weight < 0 or bm25_weight + dense_weight <= 0:
            raise ValueError("At least one retrieval weight must be positive")
        if not 1 <= top_k_final <= 5:
            raise ValueError("BTC output requires top_k_final between 1 and 5")
        self.bm25_weight = float(bm25_weight)
        self.dense_weight = float(dense_weight)
        self.use_reranker = bool(use_reranker)
        self.use_query_expansion = bool(use_query_expansion)
        self.top_k_initial = int(top_k_initial)
        self.top_k_final = int(top_k_final)
        self.rrf_k = int(rrf_k)
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.device = device
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever(model_name=embedding_model, device=device)
        self.reranker = Reranker(model_name=reranker_model, device=device) if use_reranker else None
        self.indexed = False

    def index_documents(self, documents: List[Dict[str, Any]], save_path: Optional[Path] = None) -> None:
        if not documents:
            raise ValueError("No document chunks supplied")
        required = {"doc_id", "text"}
        if any(not required.issubset(document) for document in documents):
            raise ValueError("Every chunk must contain doc_id and text")
        self.bm25.build_index(documents)
        self.dense.build_index(documents)
        self.indexed = True
        if save_path:
            self.save(save_path)

    @staticmethod
    def _best_chunk_per_doc(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prevent a long document's many chunks from occupying several ranks."""
        best: Dict[str, Dict[str, Any]] = {}
        for result in results:
            doc_id = str(result["doc_id"])
            current = best.get(doc_id)
            if current is None or (float(result["score"]), -int(result["rank"])) > (
                float(current["score"]), -int(current["rank"])
            ):
                best[doc_id] = dict(result)
        return sorted(best.values(), key=lambda item: (int(item["rank"]), str(item["doc_id"])))

    def _rrf(self, bm25_results: List[Dict[str, Any]], dense_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sparse = self._best_chunk_per_doc(bm25_results)
        dense = self._best_chunk_per_doc(dense_results)
        fused: Dict[str, Dict[str, Any]] = {}
        for method, weight, results in (
            ("bm25", self.bm25_weight, sparse),
            ("dense", self.dense_weight, dense),
        ):
            for rank, result in enumerate(results, 1):
                doc_id = str(result["doc_id"])
                record = fused.setdefault(doc_id, {
                    **result,
                    "doc_id": doc_id,
                    "score": 0.0,
                    "method": "rrf",
                    "bm25_rank": None,
                    "dense_rank": None,
                    "evidence_rank": 10**9,
                })
                record["score"] += weight / (self.rrf_k + rank)
                record[f"{method}_rank"] = rank
                # Use the best-ranked source chunk as reranker evidence.
                if rank < record["evidence_rank"]:
                    record["text"] = result["text"]
                    record["chunk_id"] = result.get("chunk_id", doc_id)
                    record["evidence_rank"] = rank
        output = list(fused.values())
        output.sort(key=lambda item: (-float(item["score"]), str(item["doc_id"])))
        for rank, item in enumerate(output, 1):
            item["rank"] = rank
        return output

    def retrieve(self, query: str, query_id: str = "", top_k: Optional[int] = None) -> QueryResult:
        if not self.indexed:
            raise RuntimeError("Index is not built or loaded")
        final_k = self.top_k_final if top_k is None else int(top_k)
        if not 1 <= final_k <= 5:
            raise ValueError("BTC Task 1 permits 1 to 5 predictions per question")
        started = time.perf_counter()
        clean_query = preprocess_query(query)
        sparse_query = expand_query(clean_query) if self.use_query_expansion else clean_query
        bm25_results = self.bm25.retrieve(sparse_query, self.top_k_initial)
        dense_results = self.dense.retrieve(clean_query, self.top_k_initial)
        candidates = self._rrf(bm25_results, dense_results)
        used_zero_score_fallback = False
        if not candidates:
            # Preserve the submission contract for an out-of-vocabulary query.
            # The audit flag makes this visible instead of silently pretending
            # retrieval succeeded.
            used_zero_score_fallback = True
            seen = set()
            for document in self.bm25.documents:
                doc_id = str(document["doc_id"])
                if doc_id in seen:
                    continue
                candidates.append({
                    **document,
                    "doc_id": doc_id,
                    "chunk_id": str(document.get("chunk_id", doc_id)),
                    "score": 0.0,
                    "method": "zero-score-fallback",
                    "rank": len(candidates) + 1,
                    "bm25_rank": None,
                    "dense_rank": None,
                })
                seen.add(doc_id)
                if len(candidates) >= final_k:
                    break
        candidate_limit = min(len(candidates), max(final_k * 4, 20))
        if self.reranker is not None:
            candidates = self.reranker.rerank(clean_query, candidates[:candidate_limit])

        selected = candidates[:final_k]
        results = [RetrievalResult(
            doc_id=str(item["doc_id"]),
            text=str(item["text"]),
            score=float(item["score"]),
            method=str(item.get("method", "rrf")),
            rank=rank,
            chunk_id=str(item.get("chunk_id", item["doc_id"])),
            rerank_score=float(item["rerank_score"]) if item.get("rerank_score") is not None else None,
            bm25_rank=item.get("bm25_rank"),
            dense_rank=item.get("dense_rank"),
        ) for rank, item in enumerate(selected, 1)]
        return QueryResult(
            query_id=str(query_id),
            query=query,
            results=results,
            processing_time=time.perf_counter() - started,
            metadata={
                "bm25_candidates": len(bm25_results),
                "dense_candidates": len(dense_results),
                "fused_documents": len(candidates),
                "dense_backend": "tfidf" if self.dense.use_fallback else "sentence-transformer",
                "reranker_backend": "lexical" if self.reranker and self.reranker.use_fallback else (
                    "cross-encoder" if self.reranker else "disabled"
                ),
                "zero_score_fallback": used_zero_score_fallback,
            },
        )

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.bm25.save(directory / "bm25")
        self.dense.save(directory / "dense")
        config = {
            "bm25_weight": self.bm25_weight,
            "dense_weight": self.dense_weight,
            "use_reranker": self.use_reranker,
            "use_query_expansion": self.use_query_expansion,
            "top_k_initial": self.top_k_initial,
            "top_k_final": self.top_k_final,
            "rrf_k": self.rrf_k,
            "embedding_model": self.embedding_model,
            "reranker_model": self.reranker_model,
        }
        (directory / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, device: Optional[str] = None) -> "HybridRetriever":
        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        config["device"] = device
        instance = cls(**config)
        instance.bm25 = BM25Retriever.load(directory / "bm25")
        instance.dense = DenseRetriever.load(directory / "dense", device=device)
        instance.indexed = True
        return instance
