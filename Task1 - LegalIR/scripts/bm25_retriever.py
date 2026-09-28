"""Fast deterministic BM25 retrieval for Vietnamese legal passages."""

from __future__ import annotations

import json
import math
import pickle
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# Keep negation, modality and legal-domain words: they frequently decide which
# provision is relevant.  Only remove very common function words.
DEFAULT_STOPWORDS = {
    "và", "của", "là", "các", "một", "những", "đó", "này", "tại",
    "từ", "với", "cho", "về", "do", "bởi", "thì", "mà", "cũng", "đang",
}
TOKEN_RE = re.compile(r"[0-9]+(?:[./-][0-9a-z]+)*|[^\W_]+", re.UNICODE)


class VietnameseTokenizer:
    """Small dependency-free tokenizer with optional word bigrams."""

    def __init__(self, stopwords: Optional[Iterable[str]] = None, bigrams: bool = True):
        self.stopwords = set(DEFAULT_STOPWORDS if stopwords is None else stopwords)
        self.bigrams = bigrams

    def tokenize(self, text: str) -> List[str]:
        normalized = unicodedata.normalize("NFC", str(text)).casefold()
        words = [token for token in TOKEN_RE.findall(normalized) if token not in self.stopwords]
        if not self.bigrams:
            return words
        # Bigrams partially recover Vietnamese multi-syllable terms without
        # requiring a word-segmentation package in the submission environment.
        return words + [f"{a}__{b}" for a, b in zip(words, words[1:])]


class BM25:
    """BM25 using postings, so query time does not scan the whole corpus."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = float(k1)
        self.b = float(b)
        self.tokenizer = VietnameseTokenizer()
        self.doc_lengths: List[int] = []
        self.avg_doc_length = 1.0
        self.doc_count = 0
        self.idf: Dict[str, float] = {}
        self.postings: Dict[str, List[Tuple[int, int]]] = {}

    def fit(self, documents: List[str]) -> "BM25":
        if not documents:
            raise ValueError("Cannot build BM25 on an empty corpus")
        raw_postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.doc_lengths = []
        for index, document in enumerate(documents):
            counts = Counter(self.tokenizer.tokenize(document))
            self.doc_lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                raw_postings[term].append((index, frequency))

        self.doc_count = len(documents)
        self.avg_doc_length = sum(self.doc_lengths) / self.doc_count or 1.0
        self.postings = dict(raw_postings)
        self.idf = {
            term: math.log(1.0 + (self.doc_count - len(items) + 0.5) / (len(items) + 0.5))
            for term, items in self.postings.items()
        }
        return self

    def get_scores(self, query: str) -> List[float]:
        scores = [0.0] * self.doc_count
        query_counts = Counter(self.tokenizer.tokenize(query))
        for term, query_frequency in query_counts.items():
            if term not in self.postings:
                continue
            idf = self.idf[term]
            for doc_index, term_frequency in self.postings[term]:
                length_norm = 1.0 - self.b + self.b * self.doc_lengths[doc_index] / self.avg_doc_length
                numerator = term_frequency * (self.k1 + 1.0)
                denominator = term_frequency + self.k1 * length_norm
                scores[doc_index] += query_frequency * idf * numerator / denominator
        return scores

    def get_top_k(self, query: str, k: int = 20) -> List[Tuple[int, float]]:
        if k <= 0:
            return []
        scores = self.get_scores(query)
        ranked = sorted(enumerate(scores), key=lambda pair: (-pair[1], pair[0]))
        return [(index, score) for index, score in ranked[:k] if score > 0.0]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self.__dict__, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "BM25":
        with Path(path).open("rb") as handle:
            state = pickle.load(handle)
        instance = cls(state["k1"], state["b"])
        instance.__dict__.update(state)
        return instance


class BM25Retriever:
    """BM25 plus stable chunk metadata and persistence."""

    def __init__(self, index_path: Optional[Path] = None, k1: float = 1.5, b: float = 0.75):
        self.index_path = Path(index_path) if index_path else None
        self.bm25 = BM25(k1, b)
        self.documents: List[Dict[str, Any]] = []

    def build_index(self, documents: List[Dict[str, Any]], save_path: Optional[Path] = None) -> None:
        self.documents = [dict(document) for document in documents]
        self.bm25.fit([str(document["text"]) for document in self.documents])
        if save_path:
            self.save(save_path)

    def retrieve(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        results = []
        for rank, (index, score) in enumerate(self.bm25.get_top_k(query, top_k), 1):
            document = self.documents[index]
            results.append({
                "doc_id": str(document["doc_id"]),
                "chunk_id": str(document.get("chunk_id", document["doc_id"])),
                "text": str(document["text"]),
                "score": float(score),
                "rank": rank,
                "method": "bm25",
                "position": int(document.get("position", 0)),
            })
        return results

    def batch_retrieve(self, queries: List[str], top_k: int = 20) -> List[List[Dict[str, Any]]]:
        return [self.retrieve(query, top_k) for query in queries]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.bm25.save(path.with_suffix(".pkl"))
        path.with_suffix(".meta.json").write_text(
            json.dumps({"documents": self.documents}, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "BM25Retriever":
        path = Path(path)
        instance = cls()
        instance.bm25 = BM25.load(path.with_suffix(".pkl"))
        instance.documents = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))["documents"]
        return instance


def expand_query(query: str) -> str:
    """Conservative rule expansion; only adds a handful of stable synonyms.

    V60: Enhanced with legal-specific expansions for better retrieval.
    """
    # General Vietnamese synonyms
    groups = {
        "xử phạt": ("vi phạm", "mức phạt"),
        "thủ tục": ("trình tự", "hồ sơ"),
        "nghĩa vụ": ("trách nhiệm",),
        "cấp phép": ("giấy phép",),
    }

    # V60: Legal-specific expansions
    legal_groups = {
        # Legal act types
        "luật": ("quyết định", "nghị định", "thông tư"),
        "quyết định": ("qđ", "qd"),
        "nghị định": ("nd",),
        "thông tư": ("tt",),
        # Legal structure
        "điều": ("khoản", "điểm"),
        "khoản": ("điều", "điểm"),
        # Legal actions
        "ban hành": ("có hiệu lực", "có hiệu lực thi hành"),
        "hết hiệu lực": ("bãi bỏ", "thay thế"),
        # Permissions
        "được phép": ("cho phép", "cấp phép"),
        "không được": ("cấm", "ngăn cấm", "cấm"),
        # Obligations
        "phải": ("bắt buộc", "bắt buộc phải"),
        "có trách nhiệm": ("chịu trách nhiệm", "chịu trách nhiệm về"),
        # Time-related
        "thời hạn": ("hạn", "hạn chót", "trong thời gian"),
        # Organization
        "cơ quan": ("đơn vị", "tổ chức"),
        "người": ("công dân", "cá nhân"),
    }

    additions: List[str] = []
    lowered = query.casefold()
    for trigger, synonyms in groups.items():
        if trigger in lowered:
            additions.extend(synonyms)

    # V60: Apply legal expansions
    for trigger, expansions in legal_groups.items():
        if trigger in lowered:
            additions.extend(expansions)

    return " ".join([query, *additions]).strip()
