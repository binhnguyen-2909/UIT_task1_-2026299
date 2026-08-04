"""
Hybrid Retrieval Pipeline
========================
Combines BM25 sparse retrieval and dense retrieval for legal information retrieval.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass, field
from datetime import datetime

from .bm25_retriever import BM25Retriever, expand_query
from .dense_retriever import DenseRetriever, preprocess_query
from .reranker import Reranker, LegalRelevanceFeatures

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Single retrieval result."""
    doc_id: str
    text: str
    score: float
    method: str
    rank: int = 0
    rerank_score: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            'doc_id': self.doc_id,
            'text': self.text,
            'score': self.score,
            'method': self.method,
            'rank': self.rank,
            'rerank_score': self.rerank_score
        }


@dataclass
class QueryResult:
    """Result for a single query."""
    query_id: str
    query: str
    results: List[RetrievalResult] = field(default_factory=list)
    processing_time: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'query_id': self.query_id,
            'query': self.query,
            'answers': [r.doc_id for r in self.results],
            'results': [r.to_dict() for r in self.results],
            'processing_time': self.processing_time,
            'metadata': self.metadata
        }


class HybridRetriever:
    """
    Hybrid retrieval system combining BM25 and dense retrieval.

    Supports:
    - BM25 sparse retrieval
    - Dense embedding retrieval
    - Score fusion (RRF, weighted average)
    - Cross-encoder reranking
    """

    def __init__(
        self,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        use_reranker: bool = True,
        use_query_expansion: bool = True,
        top_k_initial: int = 100,
        top_k_final: int = 5
    ):
        """
        Initialize hybrid retriever.

        Args:
            bm25_weight: Weight for BM25 scores
            dense_weight: Weight for dense retrieval scores
            use_reranker: Whether to use cross-encoder reranking
            use_query_expansion: Whether to expand queries with legal synonyms
            top_k_initial: Number of candidates to retrieve initially
            top_k_final: Number of final results to return
        """
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.use_reranker = use_reranker
        self.use_query_expansion = use_query_expansion
        self.top_k_initial = top_k_initial
        self.top_k_final = top_k_final

        # Initialize components
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever()
        self.reranker = Reranker() if use_reranker else None
        self.legal_features = LegalRelevanceFeatures()

        # Document store
        self.documents = []
        self.doc_id_to_idx = {}

        # Index metadata
        self.indexed = False
        self.index_time = None

    def index_documents(self, documents: List[Dict], save_path: Optional[Path] = None):
        """
        Index documents for retrieval.

        Args:
            documents: List of document dicts with 'doc_id' and 'text' keys
            save_path: Optional path to save the index
        """
        logger.info(f"Indexing {len(documents)} documents...")

        self.documents = documents
        self.doc_id_to_idx = {doc['doc_id']: i for i, doc in enumerate(documents)}

        # Build BM25 index
        logger.info("Building BM25 index...")
        self.bm25.build_index(documents)

        # Build dense index
        logger.info("Building dense index...")
        self.dense.build_index(documents)

        # Fit reranker
        if self.reranker:
            logger.info("Fitting reranker...")
            self.reranker.fit([doc['text'] for doc in documents])

        self.indexed = True
        self.index_time = datetime.now()

        # Save if path provided
        if save_path:
            self.save(save_path)

        logger.info("Indexing complete!")

    def retrieve(
        self,
        query: str,
        query_id: Optional[str] = None,
        top_k: Optional[int] = None,
        return_scores: bool = False
    ) -> QueryResult:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Query text
            query_id: Optional query ID
            top_k: Number of results to return (default: self.top_k_final)
            return_scores: Whether to return detailed scores

        Returns:
            QueryResult with retrieved documents
        """
        if not self.indexed:
            raise RuntimeError("Documents not indexed. Call index_documents() first.")

        import time
        start_time = time.time()

        if top_k is None:
            top_k = self.top_k_final

        # Preprocess query
        processed_query = preprocess_query(query)

        # Expand query if enabled
        if self.use_query_expansion:
            expanded_query = expand_query(processed_query)
        else:
            expanded_query = processed_query

        # BM25 retrieval
        bm25_results = self.bm25.retrieve(processed_query, top_k=self.top_k_initial)

        # Dense retrieval
        dense_results = self.dense.retrieve(expanded_query, top_k=self.top_k_initial)

        # Fuse results using Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(
            bm25_results,
            dense_results,
            k=60
        )

        # Apply reranking if enabled
        if self.reranker and len(fused_results) > 0:
            reranked = self.reranker.rerank(
                processed_query,
                [r.to_dict() for r in fused_results],
                top_k=top_k * 2  # Get more candidates for final selection
            )

            # Update results
            for i, r in enumerate(reranked):
                if i < len(fused_results):
                    fused_results[i].rerank_score = r.get('rerank_score', r.get('score'))
                    fused_results[i].rank = i + 1

            # Re-sort by rerank score
            fused_results = sorted(
                fused_results[:top_k * 2],
                key=lambda x: x.rerank_score if x.rerank_score else x.score,
                reverse=True
            )[:top_k]
        else:
            # Just use fused scores
            for i, r in enumerate(fused_results[:top_k]):
                r.rank = i + 1

        # Final selection
        final_results = fused_results[:top_k]

        processing_time = time.time() - start_time

        # Create result object
        result = QueryResult(
            query_id=query_id or str(hash(query)),
            query=query,
            results=final_results,
            processing_time=processing_time,
            metadata={
                'bm25_weight': self.bm25_weight,
                'dense_weight': self.dense_weight,
                'use_reranker': self.use_reranker,
                'query_expansion': expanded_query != processed_query,
                'num_candidates': len(fused_results)
            }
        )

        return result

    def batch_retrieve(
        self,
        queries: List[Tuple[str, str]],  # List of (query_id, query)
        top_k: Optional[int] = None,
        show_progress: bool = True
    ) -> List[QueryResult]:
        """
        Retrieve documents for multiple queries.

        Args:
            queries: List of (query_id, query) tuples
            top_k: Number of results per query
            show_progress: Whether to show progress bar

        Returns:
            List of QueryResult objects
        """
        results = []

        if show_progress:
            try:
                from tqdm import tqdm
                queries = tqdm(queries, desc="Retrieving")
            except ImportError:
                pass

        for query_id, query in queries:
            results.append(self.retrieve(query, query_id, top_k))

        return results

    def _reciprocal_rank_fusion(
        self,
        results1: List[RetrievalResult],
        results2: List[RetrievalResult],
        k: int = 60
    ) -> List[RetrievalResult]:
        """
        Fuse two result sets using Reciprocal Rank Fusion.

        RRF formula: score(d) = sum(1 / (k + rank(d)))

        Args:
            results1: First result set
            results2: Second result set
            k: RRF parameter (higher = more weight to lower ranks)

        Returns:
            Fused result set
        """
        # Build score maps
        scores = {}

        # Add BM25 scores
        for i, result in enumerate(results1):
            rrf_score = 1 / (k + i + 1)
            doc_key = result.doc_id

            if doc_key not in scores:
                scores[doc_key] = {
                    'doc_id': doc_key,
                    'text': result.text,
                    'bm25_score': result.score,
                    'dense_score': 0.0,
                    'bm25_rank': i + 1,
                    'dense_rank': None,
                    'rrf_score': 0.0
                }

            scores[doc_key]['rrf_score'] += self.bm25_weight * rrf_score

        # Add dense scores
        for i, result in enumerate(results2):
            rrf_score = 1 / (k + i + 1)
            doc_key = result.doc_id

            if doc_key not in scores:
                scores[doc_key] = {
                    'doc_id': doc_key,
                    'text': result.text,
                    'bm25_score': 0.0,
                    'dense_score': result.score,
                    'bm25_rank': None,
                    'dense_rank': i + 1,
                    'rrf_score': 0.0
                }
            else:
                scores[doc_key]['dense_rank'] = i + 1

            scores[doc_key]['rrf_score'] += self.dense_weight * rrf_score

        # Sort by RRF score
        sorted_docs = sorted(scores.values(), key=lambda x: x['rrf_score'], reverse=True)

        # Convert to RetrievalResult objects
        fused = []
        for i, doc in enumerate(sorted_docs):
            result = RetrievalResult(
                doc_id=doc['doc_id'],
                text=doc['text'],
                score=doc['rrf_score'],
                method='hybrid',
                rank=i + 1
            )
            result.bm25_score = doc.get('bm25_score', 0)
            result.dense_score = doc.get('dense_score', 0)
            fused.append(result)

        return fused

    def weighted_fusion(
        self,
        results1: List[RetrievalResult],
        results2: List[RetrievalResult],
        alpha: float = 0.5
    ) -> List[RetrievalResult]:
        """
        Fuse results using weighted score combination.

        Args:
            results1: BM25 results
            results2: Dense results
            alpha: Weight for first result set

        Returns:
            Fused results
        """
        # Normalize scores
        max1 = max((r.score for r in results1), default=1)
        max2 = max((r.score for r in results2), default=1)

        scores = {}

        for r in results1:
            doc_key = r.doc_id
            if doc_key not in scores:
                scores[doc_key] = {
                    'doc_id': doc_key,
                    'text': r.text,
                    'norm_score1': r.score / max1,
                    'norm_score2': 0.0,
                    'count1': 1,
                    'count2': 0
                }
            else:
                scores[doc_key]['norm_score1'] += r.score / max1
                scores[doc_key]['count1'] += 1

        for r in results2:
            doc_key = r.doc_id
            if doc_key not in scores:
                scores[doc_key] = {
                    'doc_id': doc_key,
                    'text': r.text,
                    'norm_score1': 0.0,
                    'norm_score2': r.score / max2,
                    'count1': 0,
                    'count2': 1
                }
            else:
                scores[doc_key]['norm_score2'] += r.score / max2
                scores[doc_key]['count2'] += 1

        # Compute combined scores
        for doc_key, scores_doc in scores.items():
            scores_doc['combined'] = (
                alpha * scores_doc['norm_score1'] / max(scores_doc['count1'], 1) +
                (1 - alpha) * scores_doc['norm_score2'] / max(scores_doc['count2'], 1)
            )

        # Sort and convert
        sorted_docs = sorted(scores.values(), key=lambda x: x['combined'], reverse=True)

        fused = []
        for i, doc in enumerate(sorted_docs):
            result = RetrievalResult(
                doc_id=doc['doc_id'],
                text=doc['text'],
                score=doc['combined'],
                method='hybrid_weighted',
                rank=i + 1
            )
            fused.append(result)

        return fused

    def save(self, path: Path):
        """Save the retriever state."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save BM25
        self.bm25.save(path / "bm25")

        # Save dense
        self.dense.save(path / "dense")

        # Save metadata
        metadata = {
            'bm25_weight': self.bm25_weight,
            'dense_weight': self.dense_weight,
            'use_reranker': self.use_reranker,
            'use_query_expansion': self.use_query_expansion,
            'top_k_initial': self.top_k_initial,
            'top_k_final': self.top_k_final,
            'indexed': self.indexed,
            'index_time': self.index_time.isoformat() if self.index_time else None,
            'doc_count': len(self.documents)
        }

        with open(path / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        # Save reranker if used
        if self.reranker:
            self.reranker.save(path / "reranker")

        logger.info(f"HybridRetriever saved to {path}")

    @classmethod
    def load(cls, path: Path) -> 'HybridRetriever':
        """Load the retriever state."""
        path = Path(path)

        # Load metadata
        with open(path / "metadata.json", 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Create instance
        retriever = cls(
            bm25_weight=metadata['bm25_weight'],
            dense_weight=metadata['dense_weight'],
            use_reranker=metadata['use_reranker'],
            use_query_expansion=metadata['use_query_expansion'],
            top_k_initial=metadata['top_k_initial'],
            top_k_final=metadata['top_k_final']
        )

        # Load components
        retriever.bm25 = BM25Retriever.load(path / "bm25")
        retriever.dense = DenseRetriever.load(path / "dense")

        if retriever.reranker:
            retriever.reranker = Reranker.load(path / "reranker")

        retriever.documents = retriever.dense.documents
        retriever.doc_id_to_idx = {doc['doc_id']: i for i, doc in enumerate(retriever.documents)}
        retriever.indexed = metadata['indexed']
        retriever.index_time = datetime.fromisoformat(metadata['index_time']) if metadata['index_time'] else None

        logger.info(f"HybridRetriever loaded from {path}")
        return retriever


# Main execution for testing
if __name__ == "__main__":
    # Test hybrid retrieval
    docs = [
        {"doc_id": "1", "text": "Luật quản lý thuế quy định về thuế thu nhập doanh nghiệp phải nộp hàng năm."},
        {"doc_id": "2", "text": "Nghị định số 123/2020/NĐ-CP về xử phạt vi phạm hành chính trong lĩnh vực thuế. Mức phạt tiền từ 5 đến 50 triệu đồng."},
        {"doc_id": "3", "text": "Thông tư 80/2021/TT-BTC hướng dẫn thủ tục khai thuế, nộp thuế điện tử cho doanh nghiệp."},
        {"doc_id": "4", "text": "Quy định về trách nhiệm của người nộp thuế trong việc kê khai thuế đầy đủ và đúng hạn."},
        {"doc_id": "5", "text": "Chính sách ưu đãi thuế TNDN cho doanh nghiệp nhỏ và vừa theo Nghị định 132/2020/NĐ-CP."},
    ]

    # Initialize and index
    retriever = HybridRetriever(
        bm25_weight=0.4,
        dense_weight=0.6,
        use_reranker=False
    )

    retriever.index_documents(docs)

    # Test queries
    test_queries = [
        ("q1", "thuế thu nhập doanh nghiệp"),
        ("q2", "xử phạt vi phạm thuế"),
        ("q3", "thủ tục khai thuế điện tử"),
    ]

    print("\n" + "="*60)
    print("LEGAL INFORMATION RETRIEVAL - TEST RESULTS")
    print("="*60)

    for query_id, query in test_queries:
        result = retriever.retrieve(query, query_id)

        print(f"\nQuery [{query_id}]: {query}")
        print(f"Processing time: {result.processing_time*1000:.1f}ms")
        print(f"Top {len(result.results)} results:")

        for i, r in enumerate(result.results):
            print(f"  {i+1}. Doc {r.doc_id} (score={r.score:.4f}, method={r.method})")
            print(f"     {r.text[:80]}...")
