"""Experimental warmup-prior retriever; the corpus-based pipeline is primary."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

try:
    from .bm25_retriever import VietnameseTokenizer
except ImportError:
    from bm25_retriever import VietnameseTokenizer


class LegalRetrievalSystem:
    """Weak prior from labelled questions; useful only as an ablation signal.

    It cannot discover a new BTC document ID and therefore must never replace
    corpus retrieval on the public test.
    """

    def __init__(self):
        self.tokenizer = VietnameseTokenizer(bigrams=True)
        self.postings: Dict[str, Counter[str]] = defaultdict(Counter)
        self.document_prior: Counter[str] = Counter()
        self.document_frequency: Counter[str] = Counter()
        self.question_count = 0

    def train(self, warmup_data: Dict[str, Any]) -> None:
        self.postings.clear()
        self.document_prior.clear()
        self.document_frequency.clear()
        self.question_count = len(warmup_data)
        for item in warmup_data.values():
            question = item.get("question", "")
            answers = [str(value) for value in item.get("answer", [])]
            terms = set(self.tokenizer.tokenize(question))
            for doc_id in answers:
                self.document_prior[doc_id] += 1
            for term in terms:
                self.document_frequency[term] += 1
                self.postings[term].update(answers)

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if not self.question_count:
            raise RuntimeError("Call train() before retrieve()")
        scores: Counter[str] = Counter()
        for term in set(self.tokenizer.tokenize(query)):
            if term not in self.postings:
                continue
            weight = math.log((1 + self.question_count) / (1 + self.document_frequency[term])) + 1.0
            for doc_id, frequency in self.postings[term].items():
                scores[doc_id] += weight * frequency
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
