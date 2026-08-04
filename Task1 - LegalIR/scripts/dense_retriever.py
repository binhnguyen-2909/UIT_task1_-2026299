"""
Dense Retrieval Module using Sentence Transformers
====================================================
Dense passage retrieval with multilingual embeddings for Vietnamese legal texts.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Check for transformers availability
try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Dense retrieval will use TF-IDF fallback.")


class TFIDFFallback:
    """
    Simple TF-IDF fallback when sentence-transformers is not available.
    """

    def __init__(self):
        self.doc_vectors = None
        self.documents = []
        self.vocab = {}
        self.idf = {}

    def fit(self, documents: List[str]):
        """Build TF-IDF index."""
        from collections import Counter
        import math

        self.documents = documents
        doc_count = len(documents)

        # Build vocabulary
        word_docs = Counter()
        for doc in documents:
            words = set(doc.lower().split())
            for word in words:
                word_docs[word] += 1

        # Filter vocabulary
        min_doc_freq = 2
        self.vocab = {word: idx for idx, (word, freq) in enumerate(word_docs.items()) if freq >= min_doc_freq}

        # Calculate IDF
        for word, freq in word_docs.items():
            if word in self.vocab:
                self.idf[word] = math.log(doc_count / freq)

        # Build document vectors
        self.doc_vectors = np.zeros((doc_count, len(self.vocab)))

        for i, doc in enumerate(documents):
            words = doc.lower().split()
            word_freq = Counter(words)

            for word, freq in word_freq.items():
                if word in self.vocab:
                    tf = 1 + math.log(freq) if freq > 0 else 0
                    self.doc_vectors[i, self.vocab[word]] = tf * self.idf.get(word, 0)

        # Normalize
        norms = np.linalg.norm(self.doc_vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self.doc_vectors /= norms

        logger.info(f"TF-IDF index built with {len(self.vocab)} terms")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to vectors."""
        vectors = np.zeros((len(texts), len(self.vocab)))

        for i, text in enumerate(texts):
            words = text.lower().split()
            word_freq = Counter(words)

            for word, freq in word_freq.items():
                if word in self.vocab:
                    tf = 1 + math.log(freq) if freq > 0 else 0
                    vectors[i, self.vocab[word]] = tf * self.idf.get(word, 0)

        # Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        vectors /= norms

        return vectors

    def save(self, path: Path):
        """Save TF-IDF index."""
        data = {
            'doc_vectors': self.doc_vectors,
            'vocab': self.vocab,
            'idf': self.idf,
            'documents': self.documents
        }
        np.save(path, data)

    def load(self, path: Path):
        """Load TF-IDF index."""
        data = np.load(path, allow_pickle=True).item()
        self.doc_vectors = data['doc_vectors']
        self.vocab = data['vocab']
        self.idf = data['idf']
        self.documents = data['documents']


class DenseRetriever:
    """
    Dense passage retriever using sentence transformers.

    Supports:
    - Multilingual MiniLM (default, ~118M params)
    - XLM-RoBERTa (~278M params)
    - TF-IDF fallback when transformers unavailable
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
        max_seq_length: int = 512
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.documents = []
        self.doc_embeddings = None
        self.doc_id_map = {}
        self.use_fallback = not TRANSFORMERS_AVAILABLE

        # Set device
        if device is None:
            if TRANSFORMERS_AVAILABLE:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Initialize model
        if TRANSFORMERS_AVAILABLE:
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name, device=self.device)
            self.model.max_seq_length = max_seq_length
        else:
            logger.info("Using TF-IDF fallback")
            self.model = TFIDFFallback()

    def build_index(self, documents: List[Dict], save_path: Optional[Path] = None):
        """
        Build dense index from documents.

        Args:
            documents: List of document dicts with 'doc_id' and 'text' keys
            save_path: Optional path to save the index
        """
        self.documents = documents
        self.doc_id_map = {i: doc['doc_id'] for i, doc in enumerate(documents)}

        texts = [doc['text'] for doc in documents]

        if self.use_fallback:
            self.model.fit(texts)
            self.doc_embeddings = self.model.doc_vectors
        else:
            logger.info(f"Encoding {len(texts)} documents...")
            self.doc_embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )

        logger.info(f"Dense index built: {self.doc_embeddings.shape}")

        if save_path:
            self.save(save_path)

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        threshold: float = 0.0
    ) -> List[Dict]:
        """
        Retrieve top-k documents for a query.

        Args:
            query: Query text
            top_k: Number of documents to retrieve
            threshold: Minimum similarity score

        Returns:
            List of retrieved documents with scores
        """
        # Encode query
        if self.use_fallback:
            query_vec = self.model.encode([query])[0]
        else:
            query_vec = self.model.encode(
                [query],
                convert_to_numpy=True,
                show_progress_bar=False
            )[0]

        # Compute similarities
        similarities = np.dot(self.doc_embeddings, query_vec)

        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            score = float(similarities[idx])
            if score < threshold:
                continue

            results.append({
                'doc_id': self.doc_id_map[idx],
                'text': self.documents[idx]['text'],
                'score': score,
                'rank': len(results) + 1,
                'method': 'dense'
            })

        return results

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 20,
        threshold: float = 0.0
    ) -> List[List[Dict]]:
        """
        Retrieve documents for multiple queries.

        Args:
            queries: List of query texts
            top_k: Number of documents per query
            threshold: Minimum similarity score

        Returns:
            List of retrieval results per query
        """
        results = []

        for query in queries:
            results.append(self.retrieve(query, top_k, threshold))

        return results

    def save(self, path: Path):
        """Save retriever state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.use_fallback:
            self.model.save(path.with_suffix('.npz'))
        else:
            # Save embeddings
            np.save(path.with_suffix('.npy'), self.doc_embeddings)

        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'device': self.device,
            'batch_size': self.batch_size,
            'max_seq_length': self.max_seq_length,
            'doc_id_map': self.doc_id_map,
            'documents': self.documents,
            'use_fallback': self.use_fallback
        }

        with open(path.with_suffix('.meta.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"DenseRetriever saved to {path}")

    @classmethod
    def load(cls, path: Path) -> 'DenseRetriever':
        """Load retriever state."""
        path = Path(path)

        # Load metadata
        with open(path.with_suffix('.meta.json'), 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Create instance
        retriever = cls(
            model_name=metadata['model_name'],
            device=metadata['device'],
            batch_size=metadata['batch_size'],
            max_seq_length=metadata['max_seq_length']
        )

        retriever.documents = metadata['documents']
        retriever.doc_id_map = metadata['doc_id_map']
        retriever.use_fallback = metadata.get('use_fallback', not TRANSFORMERS_AVAILABLE)

        # Load embeddings
        if retriever.use_fallback:
            retriever.model = TFIDFFallback()
            retriever.model.load(path.with_suffix('.npz'))
            retriever.doc_embeddings = retriever.model.doc_vectors
        else:
            retriever.model = SentenceTransformer(metadata['model_name'], device=retriever.device)
            retriever.doc_embeddings = np.load(path.with_suffix('.npy'))

        logger.info(f"DenseRetriever loaded from {path}")
        return retriever


# Query preprocessing for Vietnamese legal texts
def preprocess_query(query: str) -> str:
    """
    Preprocess Vietnamese legal query.

    Args:
        query: Raw query

    Returns:
        Preprocessed query
    """
    # Remove extra whitespace
    query = ' '.join(query.split())

    # Normalize common abbreviations
    replacements = {
        'kg': 'kilogram',
        'km': 'kilômét',
        'm2': 'mét vuông',
        'm3': 'mét khối',
        'tỷ': 'tỷ đồng',
        'tr': 'triệu',
    }

    for abbr, full in replacements.items():
        if abbr in query.lower():
            query = query.lower().replace(abbr, full)

    return query


# Main execution for testing
if __name__ == "__main__":
    # Test dense retrieval
    docs = [
        "Luật quản lý thuế quy định về thuế thu nhập doanh nghiệp.",
        "Nghị định về xử phạt vi phạm hành chính trong lĩnh vực thuế.",
        "Thông tư hướng dẫn thủ tục khai thuế và nộp thuế.",
        "Quy định về trách nhiệm của người nộp thuế.",
        "Chính sách ưu đãi thuế cho doanh nghiệp nhỏ và vừa."
    ]

    retriever = DenseRetriever()

    # Build index
    documents = [{'doc_id': str(i), 'text': doc} for i, doc in enumerate(docs)]
    retriever.build_index(documents)

    # Test retrieval
    query = "thuế doanh nghiệp phải nộp bao nhiêu"
    results = retriever.retrieve(query, top_k=3)

    print(f"Query: {query}")
    print(f"Top 3 results:")
    for r in results:
        print(f"  Doc {r['doc_id']} (score={r['score']:.3f}): {r['text'][:60]}...")
