"""Backend-neutral result objects shared by LegalIR retrievers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RetrievalResult:
    doc_id: str
    text: str
    score: float
    method: str
    rank: int
    chunk_id: str = ""
    rerank_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    dense_rank: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QueryResult:
    query_id: str
    query: str
    results: List[RetrievalResult] = field(default_factory=list)
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "answers": [result.doc_id for result in self.results],
            "results": [result.to_dict() for result in self.results],
            "processing_time": self.processing_time,
            "metadata": self.metadata,
        }
