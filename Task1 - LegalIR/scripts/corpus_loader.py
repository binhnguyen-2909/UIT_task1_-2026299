"""Load and chunk the official DSC 2026 selected legal contexts."""

from __future__ import annotations

import json
import logging
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


logger = logging.getLogger(__name__)


class VietnameseTextProcessor:
    """Conservative normalization utilities for Vietnamese legal text."""

    LEGAL_PATTERNS = {
        "article_ref": r"(?:Điều|Chương|Khoản|Mục)\s+\d+[a-z]?",
        "number": r"\d+(?:[.,]\d+)*(?:\s*(?:đồng|người|ngày|tháng|năm|vụ))?",
        "law_ref": (
            r"(?:Nghị định|Luật|Bộ luật|Thông tư|Nghị quyết|Quyết định)"
            r"\s+(?:số\s+)?[\w./-]+"
        ),
        "date": r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}",
    }

    def normalize(self, text: str) -> str:
        """Normalize whitespace without destroying paragraph boundaries."""
        if not isinstance(text, str):
            raise TypeError(f"Document text must be a string, got {type(text).__name__}")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[\u200b-\u200d\ufeff]", "", text)
        text = re.sub(r"[\t\f\v ]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def split_sentences(self, text: str) -> List[str]:
        """Split on sentence boundaries while retaining punctuation."""
        text = self.normalize(text)
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]

    def split_paragraphs(self, text: str) -> List[str]:
        return [part.strip() for part in self.normalize(text).split("\n\n") if part.strip()]

    def extract_legal_entities(self, text: str) -> Dict[str, List[str]]:
        return {
            entity_type: sorted(set(re.findall(pattern, text, re.IGNORECASE)))
            for entity_type, pattern in self.LEGAL_PATTERNS.items()
        }


class LegalCorpusLoader:
    """Load BTC corpus formats into ``doc_id -> passage``.

    Supported inputs:
    - ``selected-contexts.zip`` containing many ``context_*.json`` files;
    - a directory containing JSON/TXT contexts (searched recursively);
    - a single context JSON record with ``id`` and ``passage``;
    - a JSON mapping of IDs to strings or records.
    """

    TEXT_FIELDS = ("passage", "content", "text")

    def __init__(self, corpus_path: Optional[Path] = None):
        self.corpus_path = Path(corpus_path) if corpus_path else None
        self.processor = VietnameseTextProcessor()
        self.documents: Dict[str, str] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

    def load_corpus(self, corpus_path: Optional[Path] = None) -> Dict[str, str]:
        source = corpus_path if corpus_path is not None else self.corpus_path
        if source is None:
            raise ValueError("No corpus path provided")
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Corpus not found: {path}")

        self.documents = {}
        self.metadata = {}
        if path.is_dir():
            self._load_directory(path)
        elif path.suffix.lower() == ".zip":
            self._load_zip(path)
        elif path.suffix.lower() == ".json":
            self._merge_json(json.loads(path.read_text(encoding="utf-8")), path.stem)
        elif path.suffix.lower() == ".txt":
            self._add_document(path.stem, path.read_text(encoding="utf-8"), {})
        else:
            raise ValueError(f"Unsupported corpus input: {path}")

        if not self.documents:
            raise ValueError(
                f"No legal contexts found in {path}. Expected context JSON records "
                "with 'id' and 'passage' fields."
            )
        logger.info("Loaded %d legal contexts from %s", len(self.documents), path)
        return self.documents

    def _load_directory(self, directory: Path) -> None:
        json_files = sorted(directory.rglob("*.json"))
        txt_files = sorted(directory.rglob("*.txt"))
        for json_path in json_files:
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid context JSON: {json_path}") from exc
            self._merge_json(data, json_path.stem)
        for text_path in txt_files:
            self._add_document(text_path.stem, text_path.read_text(encoding="utf-8"), {})

    def _load_zip(self, archive_path: Path) -> None:
        with zipfile.ZipFile(archive_path, "r") as archive:
            for info in sorted(archive.infolist(), key=lambda item: item.filename):
                if info.is_dir() or info.filename.startswith("__MACOSX/"):
                    continue
                suffix = Path(info.filename).suffix.lower()
                if suffix == ".json":
                    try:
                        data = json.loads(archive.read(info).decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise ValueError(f"Invalid context JSON in ZIP: {info.filename}") from exc
                    self._merge_json(data, Path(info.filename).stem)
                elif suffix == ".txt":
                    self._add_document(
                        Path(info.filename).stem,
                        archive.read(info).decode("utf-8"),
                        {},
                    )

    def _merge_json(self, data: Any, fallback_id: str) -> None:
        if isinstance(data, list):
            for index, record in enumerate(data):
                self._merge_json(record, f"{fallback_id}_{index}")
            return
        if not isinstance(data, dict):
            return

        if self._record_text(data) is not None:
            doc_id = str(data.get("id", fallback_id))
            metadata = {key: data[key] for key in ("name", "link") if key in data}
            self._add_document(doc_id, self._record_text(data) or "", metadata)
            return

        # Mapping format: {doc_id: "text"} or {doc_id: {passage: ...}}.
        for key, value in data.items():
            if isinstance(value, str):
                self._add_document(str(key), value, {})
            elif isinstance(value, dict) and self._record_text(value) is not None:
                doc_id = str(value.get("id", key))
                metadata = {name: value[name] for name in ("name", "link") if name in value}
                self._add_document(doc_id, self._record_text(value) or "", metadata)

    def _record_text(self, record: Mapping[str, Any]) -> Optional[str]:
        for field in self.TEXT_FIELDS:
            value = record.get(field)
            if isinstance(value, str):
                return value
        return None

    def _add_document(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> None:
        doc_id = str(doc_id).strip()
        normalized = self.processor.normalize(text)
        if not doc_id or not normalized:
            return
        if doc_id in self.documents and self.documents[doc_id] != normalized:
            raise ValueError(f"Conflicting duplicate context ID: {doc_id}")
        self.documents[doc_id] = normalized
        self.metadata[doc_id] = metadata

    def chunk_documents(
        self,
        documents: Dict[str, str],
        chunk_size: int = 1200,
        overlap: int = 150,
    ) -> List[Dict[str, Any]]:
        """Create bounded overlapping chunks while preserving BTC doc IDs."""
        chunks: List[Dict[str, Any]] = []
        for doc_id, content in documents.items():
            text_chunks = self._create_chunks(content, chunk_size, overlap)
            for position, text in enumerate(text_chunks):
                chunks.append(
                    {
                        "chunk_id": f"{doc_id}::chunk::{position}",
                        "doc_id": str(doc_id),
                        "text": text,
                        "position": position,
                        "total_chunks": len(text_chunks),
                        "metadata": self.metadata.get(str(doc_id), {}),
                        "entities": self.processor.extract_legal_entities(text),
                    }
                )
        logger.info("Created %d chunks from %d contexts", len(chunks), len(documents))
        return chunks

    def _create_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
        text = self.processor.normalize(text)
        if not text:
            return []
        if len(text) <= chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        minimum_boundary = max(1, int(chunk_size * 0.6))
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                boundary = max(text.rfind("\n", start + minimum_boundary, end),
                               text.rfind(" ", start + minimum_boundary, end))
                if boundary > start:
                    end = boundary
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            next_start = max(0, end - overlap)
            if next_start <= start:
                next_start = end
            start = next_start
        return chunks

    def build_metadata(self, documents: Dict[str, str]) -> Dict[str, Any]:
        lengths = [len(text) for text in documents.values()]
        return {
            "total_documents": len(documents),
            "total_characters": sum(lengths),
            "avg_doc_length": sum(lengths) / len(lengths) if lengths else 0,
            "document_ids": list(documents.keys()),
        }


def load_warmup_data(warmup_path: Path) -> Dict[str, Any]:
    with Path(warmup_path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Warmup/public input must be a JSON object")
    return data


def extract_unique_documents(warmup_data: Dict[str, Any]) -> set[str]:
    doc_ids: set[str] = set()
    for item in warmup_data.values():
        answer = item.get("answer", []) if isinstance(item, dict) else []
        if isinstance(answer, list):
            doc_ids.update(str(doc_id) for doc_id in answer)
    return doc_ids


if __name__ == "__main__":
    sample = {"123": "Điều 1. Phạm vi điều chỉnh.\n\nNội dung văn bản pháp luật."}
    loader = LegalCorpusLoader()
    for chunk in loader.chunk_documents(sample, chunk_size=50, overlap=10):
        print(chunk["chunk_id"], chunk["text"])
