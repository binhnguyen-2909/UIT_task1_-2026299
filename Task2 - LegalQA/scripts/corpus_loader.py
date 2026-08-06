"""Independent loader for BTC selected legal contexts used by LegalQA."""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


class LegalQACorpusLoader:
    TEXT_FIELDS = ("passage", "content", "text")

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.documents: Dict[str, str] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def normalize(text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("Context passage must be a string")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[\u200b-\u200d\ufeff]", "", text)
        text = re.sub(r"[\t\f\v ]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    def load(self, path: Optional[Path] = None) -> Dict[str, str]:
        selected = path if path is not None else self.path
        if selected is None:
            raise ValueError("No corpus path provided")
        source = Path(selected)
        if not source.exists():
            raise FileNotFoundError(f"Corpus not found: {source}")
        self.documents, self.metadata = {}, {}
        if source.is_dir():
            for item in sorted(source.rglob("*.json")):
                self._merge(json.loads(item.read_text(encoding="utf-8")), item.stem)
            for item in sorted(source.rglob("*.txt")):
                self._add(item.stem, item.read_text(encoding="utf-8"), {})
        elif source.suffix.lower() == ".zip":
            with zipfile.ZipFile(source, "r") as archive:
                for item in sorted(archive.infolist(), key=lambda info: info.filename):
                    if item.is_dir() or item.filename.startswith("__MACOSX/"):
                        continue
                    suffix = Path(item.filename).suffix.lower()
                    if suffix == ".json":
                        self._merge(json.loads(archive.read(item).decode("utf-8")), Path(item.filename).stem)
                    elif suffix == ".txt":
                        self._add(Path(item.filename).stem, archive.read(item).decode("utf-8"), {})
        elif source.suffix.lower() == ".json":
            self._merge(json.loads(source.read_text(encoding="utf-8")), source.stem)
        elif source.suffix.lower() == ".txt":
            self._add(source.stem, source.read_text(encoding="utf-8"), {})
        else:
            raise ValueError(f"Unsupported corpus source: {source}")
        if not self.documents:
            raise ValueError("No valid contexts found; expected records with id and passage")
        return self.documents

    def _text(self, record: Mapping[str, Any]) -> Optional[str]:
        for field in self.TEXT_FIELDS:
            if isinstance(record.get(field), str):
                return record[field]
        return None

    def _merge(self, value: Any, fallback_id: str) -> None:
        if isinstance(value, list):
            for index, item in enumerate(value):
                self._merge(item, f"{fallback_id}_{index}")
            return
        if not isinstance(value, dict):
            return
        passage = self._text(value)
        if passage is not None:
            doc_id = str(value.get("id", fallback_id))
            metadata = {key: value[key] for key in ("name", "link") if key in value}
            self._add(doc_id, passage, metadata)
            return
        for key, item in value.items():
            if isinstance(item, str):
                self._add(str(key), item, {})
            elif isinstance(item, dict) and self._text(item) is not None:
                self._add(
                    str(item.get("id", key)),
                    self._text(item) or "",
                    {name: item[name] for name in ("name", "link") if name in item},
                )

    def _add(self, doc_id: str, passage: str, metadata: Dict[str, Any]) -> None:
        doc_id = str(doc_id).strip()
        passage = self.normalize(passage)
        if not doc_id or not passage:
            return
        if doc_id in self.documents and self.documents[doc_id] != passage:
            raise ValueError(f"Conflicting duplicate context ID: {doc_id}")
        self.documents[doc_id] = passage
        self.metadata[doc_id] = metadata

    def chunks(self, chunk_size: int = 1800, overlap: int = 250) -> List[Dict[str, Any]]:
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("Require chunk_size > overlap >= 0")
        output: List[Dict[str, Any]] = []
        for doc_id, passage in self.documents.items():
            pieces = self._split(passage, chunk_size, overlap)
            for position, text in enumerate(pieces):
                output.append({
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}::chunk::{position}",
                    "position": position,
                    "text": text,
                    "metadata": self.metadata.get(doc_id, {}),
                })
        return output

    @staticmethod
    def _split(text: str, size: int, overlap: int) -> List[str]:
        if len(text) <= size:
            return [text]
        pieces, start = [], 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                floor = start + int(size * 0.6)
                boundary = max(text.rfind("\n", floor, end), text.rfind(" ", floor, end))
                if boundary > start:
                    end = boundary
            piece = text[start:end].strip()
            if piece:
                pieces.append(piece)
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
        return pieces
