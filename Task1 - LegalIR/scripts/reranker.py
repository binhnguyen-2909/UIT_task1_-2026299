"""
Cross-Encoder Reranker Module
=============================
Reranks retrieved documents using a cross-encoder model.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Check for transformers availability
try:
    from sentence_transformers import CrossEncoder
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed. Reranker will use fallback scoring.")


class SimpleReranker:
    """
    Simple fallback reranker using word overlap and BM25 scoring.
    Used when transformers is not available.
    """

    def __init__(self):
        self.documents = []

    def fit(self, documents: List[str]):
        """Fit reranker on documents."""
        self.documents = documents

    def score(self, query: str, documents: List[str]) -> np.ndarray:
        """
        Score documents against query.

        Args:
            query: Query text
            documents: List of document texts

        Returns:
            Array of scores
        """
        query_words = set(query.lower().split())
        scores = []

        for doc in documents:
            doc_words = set(doc.lower().split())

            # Word overlap score
            overlap = len(query_words & doc_words)
            union = len(query_words | doc_words)
            jaccard = overlap / union if union > 0 else 0

            # Length penalty
            length_ratio = min(len(doc), len(query)) / max(len(doc), len(query))

            # Combined score
            score = 0.7 * jaccard + 0.3 * length_ratio
            scores.append(score)

        return np.array(scores)


class Reranker:
    """
    Cross-encoder reranker for refining retrieval results.

    Cross-encoders compute relevance scores by processing query-document pairs
    jointly, providing more accurate relevance estimates than bi-encoders.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        device: Optional[str] = None,
        max_length: int = 512
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.use_fallback = not TRANSFORMERS_AVAILABLE
        self.documents = []

        # Set device
        if device is None:
            if TRANSFORMERS_AVAILABLE:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Initialize model
        if TRANSFORMERS_AVAILABLE:
            logger.info(f"Loading reranker model: {model_name}")
            self.model = CrossEncoder(model_name, max_length=max_length, device=self.device)
        else:
            logger.info("Using simple fallback reranker")
            self.model = SimpleReranker()

    def fit(self, documents: List[str]):
        """Fit reranker on documents."""
        self.documents = documents
        self.model.fit(documents)

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Rerank candidate documents for a query.

        Args:
            query: Query text
            candidates: List of candidate documents from initial retrieval
            top_k: Number of documents to return after reranking

        Returns:
            List of reranked documents with new scores
        """
        if not candidates:
            return []

        doc_texts = [c['text'] for c in candidates]

        # Compute cross-encoder scores
        if self.use_fallback:
            scores = self.model.score(query, doc_texts)
        else:
            pairs = [[query, doc] for doc in doc_texts]
            scores = self.model.predict(pairs)

        if not isinstance(scores, np.ndarray):
            scores = np.array(scores)

        # Update candidates with new scores and rerank
        for i, candidate in enumerate(candidates):
            candidate['rerank_score'] = float(scores[i])

        # Sort by rerank score
        reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)

        # Assign new ranks
        for i, doc in enumerate(reranked):
            doc['rerank'] = i + 1

        return reranked[:top_k]

    def batch_rerank(
        self,
        queries: List[str],
        candidates_list: List[List[Dict]],
        top_k: int = 10
    ) -> List[List[Dict]]:
        """
        Rerank multiple queries.

        Args:
            queries: List of queries
            candidates_list: List of candidate lists per query
            top_k: Number of documents per query

        Returns:
            List of reranked results per query
        """
        results = []

        for query, candidates in zip(queries, candidates_list):
            results.append(self.rerank(query, candidates, top_k))

        return results

    def save(self, path: Path):
        """Save reranker state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'device': self.device,
            'use_fallback': self.use_fallback
        }

        with open(path.with_suffix('.meta.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"Reranker metadata saved to {path}")

    @classmethod
    def load(cls, path: Path) -> 'Reranker':
        """Load reranker state."""
        path = Path(path)

        with open(path.with_suffix('.meta.json'), 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        reranker = cls(
            model_name=metadata['model_name'],
            max_length=metadata['max_length'],
            device=metadata['device']
        )

        logger.info(f"Reranker loaded from {path}")
        return reranker


# Vietnamese legal query-document relevance features
class LegalRelevanceFeatures:
    """
    Extract additional legal-specific features for reranking.
    """

    def __init__(self):
        # Legal keywords by category
        self.legal_keywords = {
            'điều_luật': ['điều', 'khoản', 'mục', 'chương', 'luật', 'nghị định', 'thông tư', 'quyết định'],
            'thẩm_quyền': ['thẩm quyền', 'quyền hạn', 'trách nhiệm', 'nghĩa vụ', 'chức năng'],
            'thủ_tục': ['thủ tục', 'trình tự', 'hồ sơ', 'đăng ký', 'cấp', 'giấy phép'],
            'xử_phạt': ['xử phạt', 'phạt tiền', 'vi phạm', 'hành chính', 'kiểm tra', 'giám sát'],
            'quyền_lợi': ['quyền', 'lợi ích', 'ưu đãi', 'hỗ trợ', 'bồi thường'],
        }

    def extract(self, query: str, document: str) -> Dict[str, float]:
        """
        Extract legal relevance features.

        Args:
            query: Query text
            document: Document text

        Returns:
            Dictionary of feature scores
        """
        query_lower = query.lower()
        doc_lower = document.lower()

        features = {}

        # Keyword overlap per category
        for category, keywords in self.legal_keywords.items():
            query_kw = sum(1 for kw in keywords if kw in query_lower)
            doc_kw = sum(1 for kw in keywords if kw in doc_lower)
            features[f'{category}_query'] = query_kw / len(keywords)
            features[f'{category}_doc'] = doc_kw / len(keywords)

        # Legal reference detection
        import re
        query_refs = re.findall(r'(?:điều|chương|khoản)\s*\d+', query_lower)
        doc_refs = re.findall(r'(?:điều|chương|khoản)\s*\d+', doc_lower)

        features['query_has_legal_ref'] = 1.0 if query_refs else 0.0
        features['doc_has_legal_ref'] = 1.0 if doc_refs else 0.0
        features['legal_ref_match'] = 1.0 if set(query_refs) & set(doc_refs) else 0.0

        # Document structure features
        features['doc_has_article'] = 1.0 if 'điều' in doc_lower else 0.0
        features['doc_has_numbered_list'] = 1.0 if re.search(r'^\d+\.', doc_lower, re.MULTILINE) else 0.0

        return features

    def combine_scores(
        self,
        base_score: float,
        legal_features: Dict[str, float]
    ) -> float:
        """
        Combine base relevance score with legal features.

        Args:
            base_score: Base relevance score
            legal_features: Extracted legal features

        Returns:
            Combined score
        """
        # Weight legal-specific features
        legal_score = (
            0.3 * legal_features.get('thẩm_quyền_doc', 0) +
            0.2 * legal_features.get('thủ_tục_doc', 0) +
            0.2 * legal_features.get('xử_phạt_doc', 0) +
            0.15 * legal_features.get('điều_luật_doc', 0) +
            0.15 * legal_features.get('quyền_lợi_doc', 0) +
            0.5 * legal_features.get('legal_ref_match', 0)
        )

        # Combine with base score
        combined = 0.7 * base_score + 0.3 * legal_score * 10

        return min(combined, 1.0)


# Main execution for testing
if __name__ == "__main__":
    # Test reranker
    docs = [
        "Điều 1. Luật quản lý thuế quy định về thuế thu nhập doanh nghiệp.",
        "Nghị định số 123/2020/NĐ-CP về xử phạt vi phạm hành chính trong lĩnh vực thuế.",
        "Thông tư 80/2021/TT-BTC hướng dẫn thủ tục khai thuế và nộp thuế.",
        "Quy định về trách nhiệm của người nộp thuế trong việc kê khai.",
        "Chính sách ưu đãi thuế cho doanh nghiệp nhỏ và vừa theo Nghị định 132/2020."
    ]

    query = "quy định về thuế thu nhập doanh nghiệp"

    # Simulate initial retrieval candidates
    candidates = [
        {'doc_id': '0', 'text': docs[0], 'score': 0.8},
        {'doc_id': '1', 'text': docs[1], 'score': 0.6},
        {'doc_id': '2', 'text': docs[2], 'score': 0.5},
        {'doc_id': '3', 'text': docs[3], 'score': 0.4},
        {'doc_id': '4', 'text': docs[4], 'score': 0.3},
    ]

    # Test reranking
    reranker = Reranker()
    reranker.fit(docs)

    results = reranker.rerank(query, candidates, top_k=3)

    print(f"Query: {query}")
    print(f"Reranked results:")
    for r in results:
        print(f"  Rank {r['rerank']}: Doc {r['doc_id']} (base={r['score']:.3f}, rerank={r['rerank_score']:.3f})")
        print(f"    {r['text'][:80]}...")
