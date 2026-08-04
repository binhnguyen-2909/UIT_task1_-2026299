"""
BM25 Sparse Retrieval Module
============================
BM25 (Best Matching 25) algorithm for sparse text retrieval.
"""

import pickle
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import Counter
import math
import logging

logger = logging.getLogger(__name__)


class VietnameseTokenizer:
    """Simple Vietnamese tokenizer for BM25."""

    def __init__(self, stopwords: Optional[List[str]] = None):
        self.stopwords = set(stopwords) if stopwords else set()
        # Common Vietnamese words to filter
        self.vietnamese_stopwords = {
            'và', 'của', 'là', 'có', 'được', 'theo', 'nào', 'này', 'năm',
            'thế', 'ra', 'sao', 'như', 'đó', 'với', 'trong', 'cho', 'để',
            'tại', 'từ', 'hoặc', 'về', 'pháp', 'luật', 'quy', 'định',
            'hành', 'viên', 'người', 'doanh', 'nghiệp', 'cơ', 'quan',
            'nhà', 'nước', 'thực', 'hiện', 'hồ', 'sơ', 'thủ', 'tục',
            'một', 'các', 'khác', 'này', 'đó', 'vào', 'bộ', 'sở',
            'đến', 'khi', 'trên', 'dưới', 'sau', 'trước', 'giữa',
            'không', 'có', 'là', 'về', 'tại', 'để', 'với', 'cho',
        }
        self.stopwords.update(self.vietnamese_stopwords)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize Vietnamese text."""
        # Lowercase and normalize
        text = text.lower().strip()

        # Remove punctuation and split by whitespace
        import re
        tokens = re.findall(r'[\w]+', text, re.UNICODE)

        # Remove stopwords and short tokens
        tokens = [t for t in tokens if t not in self.stopwords and len(t) > 1]

        return tokens


class BM25:
    """
    BM25 retrieval algorithm implementation.

    BM25 formula:
    score(D, Q) = sum over q in Q of:
        IDF(q) * (tf(q, D) * (k1 + 1)) / (tf(q, D) + k1 * (1 - b + b * |D|/avgdl))
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.tokenizer = VietnameseTokenizer()
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.doc_count = 0
        self.doc_freqs = {}  # term -> document frequency
        self.idf = {}  # term -> IDF score
        self.doc_term_freqs = []  # List of term frequencies per document

    def fit(self, documents: List[str]) -> 'BM25':
        """
        Build BM25 index from documents.

        Args:
            documents: List of document texts
        """
        self.doc_count = len(documents)
        self.doc_lengths = []
        self.doc_term_freqs = []
        self.doc_freqs = Counter()

        for doc in documents:
            # Tokenize document
            tokens = self.tokenizer.tokenize(doc)
            self.doc_lengths.append(len(tokens))

            # Count term frequencies
            term_freqs = Counter(tokens)
            self.doc_term_freqs.append(term_freqs)

            # Update document frequencies
            for term in set(tokens):
                self.doc_freqs[term] += 1

        # Calculate average document length
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 1

        # Calculate IDF for all terms
        self._calculate_idf()

        logger.info(f"BM25 fitted on {self.doc_count} documents")
        logger.info(f"Average doc length: {self.avg_doc_length:.1f} tokens")
        logger.info(f"Vocabulary size: {len(self.doc_freqs)} unique terms")

        return self

    def _calculate_idf(self):
        """Calculate IDF scores for all terms."""
        for term, df in self.doc_freqs.items():
            # IDF formula with smoothing
            self.idf[term] = math.log(
                (self.doc_count - df + 0.5) / (df + 0.5) + 1
            )

    def get_scores(self, query: str) -> np.ndarray:
        """
        Get BM25 scores for all documents given a query.

        Args:
            query: Query text

        Returns:
            Array of scores for each document
        """
        query_tokens = self.tokenizer.tokenize(query)
        scores = np.zeros(self.doc_count)

        for q in query_tokens:
            if q not in self.idf:
                continue

            idf_q = self.idf[q]

            for i, doc_freqs in enumerate(self.doc_term_freqs):
                tf_q_d = doc_freqs.get(q, 0)
                if tf_q_d > 0:
                    # BM25 formula
                    numerator = tf_q_d * (self.k1 + 1)
                    denominator = tf_q_d + self.k1 * (1 - self.b + self.b * self.doc_lengths[i] / self.avg_doc_length)
                    scores[i] += idf_q * (numerator / denominator)

        return scores

    def get_top_k(self, query: str, k: int = 20) -> List[Tuple[int, float]]:
        """
        Get top-k documents for a query.

        Args:
            query: Query text
            k: Number of top documents to return

        Returns:
            List of (doc_index, score) tuples
        """
        scores = self.get_scores(query)
        top_indices = np.argsort(scores)[::-1][:k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    def save(self, path: Path):
        """Save BM25 index to disk."""
        data = {
            'k1': self.k1,
            'b': self.b,
            'doc_lengths': self.doc_lengths,
            'avg_doc_length': self.avg_doc_length,
            'doc_count': self.doc_count,
            'doc_freqs': dict(self.doc_freqs),
            'idf': self.idf,
            'doc_term_freqs': [dict(d) for d in self.doc_term_freqs],
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"BM25 index saved to {path}")

    @classmethod
    def load(cls, path: Path) -> 'BM25':
        """Load BM25 index from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        bm25 = cls(k1=data['k1'], b=data['b'])
        bm25.doc_lengths = data['doc_lengths']
        bm25.avg_doc_length = data['avg_doc_length']
        bm25.doc_count = data['doc_count']
        bm25.doc_freqs = Counter(data['doc_freqs'])
        bm25.idf = data['idf']
        bm25.doc_term_freqs = [Counter(d) for d in data['doc_term_freqs']]

        logger.info(f"BM25 index loaded from {path}")
        return bm25


class BM25Retriever:
    """
    High-level BM25 retriever with caching and batch processing.
    """

    def __init__(self, index_path: Optional[Path] = None, k1: float = 1.5, b: float = 0.75):
        self.index_path = Path(index_path) if index_path else None
        self.bm25 = BM25(k1=k1, b=b)
        self.documents = []
        self.doc_id_map = {}  # index -> doc_id

    def build_index(self, documents: List[Dict], save_path: Optional[Path] = None):
        """
        Build BM25 index from documents.

        Args:
            documents: List of document dictionaries with 'doc_id' and 'text' keys
            save_path: Optional path to save the index
        """
        # Extract texts and build mapping
        self.documents = documents
        self.doc_id_map = {i: doc['doc_id'] for i, doc in enumerate(documents)}

        texts = [doc['text'] for doc in documents]
        self.bm25.fit(texts)

        if save_path:
            self.save(save_path)

    def retrieve(self, query: str, top_k: int = 20) -> List[Dict]:
        """
        Retrieve top-k documents for a query.

        Args:
            query: Query text
            top_k: Number of documents to retrieve

        Returns:
            List of retrieved documents with scores
        """
        results = self.bm25.get_top_k(query, k=top_k)

        retrieved = []
        for idx, score in results:
            retrieved.append({
                'doc_id': self.doc_id_map[idx],
                'text': self.documents[idx]['text'],
                'score': score,
                'rank': len(retrieved) + 1,
                'method': 'bm25'
            })

        return retrieved

    def batch_retrieve(self, queries: List[str], top_k: int = 20) -> List[List[Dict]]:
        """
        Retrieve documents for multiple queries.

        Args:
            queries: List of query texts
            top_k: Number of documents per query

        Returns:
            List of retrieval results per query
        """
        return [self.retrieve(q, top_k) for q in queries]

    def save(self, path: Path):
        """Save retriever state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save BM25 index
        self.bm25.save(path.with_suffix('.pkl'))

        # Save metadata
        metadata = {
            'doc_id_map': self.doc_id_map,
            'documents': self.documents
        }
        with open(path.with_suffix('.meta.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"BM25Retriever saved to {path}")

    @classmethod
    def load(cls, path: Path) -> 'BM25Retriever':
        """Load retriever state."""
        path = Path(path)

        # Load metadata
        with open(path.with_suffix('.meta.json'), 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Load BM25
        bm25 = BM25.load(path.with_suffix('.pkl'))

        retriever = cls()
        retriever.bm25 = bm25
        retriever.documents = metadata['documents']
        retriever.doc_id_map = metadata['doc_id_map']

        logger.info(f"BM25Retriever loaded from {path}")
        return retriever


# Vietnamese legal domain expansion
LEGAL_QUERY_EXPANSION = {
    'phạt': ['xử phạt', 'vi phạm', 'hành chính', 'tiền phạt'],
    'thủ tục': ['trình tự', 'hồ sơ', 'đăng ký', 'cấp phép'],
    'quyền': ['quyền hạn', 'thẩm quyền', 'quyết định'],
    'nghĩa vụ': ['trách nhiệm', 'nghĩa vụ', 'bồi thường'],
    'công ty': ['doanh nghiệp', 'tổ chức', 'pháp nhân'],
}


def expand_query(query: str) -> str:
    """
    Expand query with legal domain synonyms.

    Args:
        query: Original query

    Returns:
        Expanded query
    """
    expanded_terms = [query]

    for term, synonyms in LEGAL_QUERY_EXPANSION.items():
        if term in query.lower():
            expanded_terms.extend(synonyms)

    return ' '.join(expanded_terms)


# Main execution for testing
if __name__ == "__main__":
    # Test BM25
    docs = [
        "Luật quản lý thuế quy định về thuế thu nhập doanh nghiệp.",
        "Nghị định về xử phạt vi phạm hành chính trong lĩnh vực thuế.",
        "Thông tư hướng dẫn thủ tục khai thuế và nộp thuế.",
        "Quy định về trách nhiệm của người nộp thuế.",
        "Chính sách ưu đãi thuế cho doanh nghiệp nhỏ và vừa."
    ]

    retriever = BM25Retriever()

    # Build index
    documents = [{'doc_id': str(i), 'text': doc} for i, doc in enumerate(docs)]
    retriever.build_index(documents)

    # Test retrieval
    query = "thuế doanh nghiệp"
    results = retriever.retrieve(query, top_k=3)

    print(f"Query: {query}")
    print(f"Top 3 results:")
    for r in results:
        print(f"  Doc {r['doc_id']} (score={r['score']:.3f}): {r['text'][:60]}...")
