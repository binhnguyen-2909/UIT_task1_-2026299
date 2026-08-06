"""Inspect the official selected-context corpus before indexing either task."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


DOC_PATTERNS = {
    "bộ_luật": r"\bbộ\s+luật\b",
    "luật": r"\bluật\b",
    "nghị_định": r"\bnghị\s+định\b",
    "thông_tư": r"\bthông\s+tư\b",
    "quyết_định": r"\bquyết\s+định\b",
    "nghị_quyết": r"\bnghị\s+quyết\b",
    "pháp_lệnh": r"\bpháp\s+lệnh\b",
}
CITATION_PATTERNS = {
    "điều": r"\bđiều\s+\d+[a-z]?",
    "khoản": r"\bkhoản\s+\d+",
    "điểm": r"\bđiểm\s+[a-zđ]",
    "chương": r"\bchương\s+(?:\d+|[ivxlcdm]+)",
}
TOPICS = {
    "lao_động": ("lao động", "hợp đồng lao động"),
    "doanh_nghiệp": ("doanh nghiệp", "công ty"),
    "đất_đai": ("đất đai", "quyền sử dụng đất"),
    "thuế": ("thuế", "người nộp thuế"),
    "giao_thông": ("giao thông", "phương tiện", "lái xe"),
    "y_tế": ("y tế", "khám bệnh", "bảo hiểm y tế"),
    "giáo_dục": ("giáo dục", "đào tạo", "học sinh", "sinh viên"),
}


class CorpusAnalyzer:
    """Schema-aware, read-only corpus audit."""

    TEXT_FIELDS = ("passage", "content", "text")

    def __init__(self):
        self.corpus: Dict[str, str] = {}
        self.names: Dict[str, str] = {}
        self.summary: Dict[str, Any] = {}

    @staticmethod
    def _text(record: Mapping[str, Any]) -> Optional[str]:
        for field in CorpusAnalyzer.TEXT_FIELDS:
            if isinstance(record.get(field), str):
                return record[field]
        return None

    def _add(self, doc_id: Any, text: str, name: str = "") -> None:
        key = str(doc_id).strip()
        normalized = re.sub(r"[\t ]+", " ", text.replace("\r\n", "\n")).strip()
        if not key or not normalized:
            return
        if key in self.corpus and self.corpus[key] != normalized:
            raise ValueError(f"Conflicting duplicate context ID: {key}")
        self.corpus[key] = normalized
        self.names[key] = str(name)

    def _merge(self, value: Any, fallback_id: str) -> None:
        if isinstance(value, list):
            for index, item in enumerate(value):
                self._merge(item, f"{fallback_id}_{index}")
            return
        if not isinstance(value, dict):
            return
        text = self._text(value)
        if text is not None:
            self._add(value.get("id", fallback_id), text, value.get("name", ""))
            return
        for key, item in value.items():
            if isinstance(item, str):
                self._add(key, item)
            elif isinstance(item, dict) and self._text(item) is not None:
                self._add(item.get("id", key), self._text(item) or "", item.get("name", ""))

    def load_corpus(self, corpus_path: str | Path) -> bool:
        path = Path(corpus_path)
        if not path.exists():
            raise FileNotFoundError(f"Corpus not found: {path}")
        self.corpus, self.names = {}, {}
        if path.is_dir():
            for item in sorted(path.rglob("*.json")):
                self._merge(json.loads(item.read_text(encoding="utf-8")), item.stem)
            for item in sorted(path.rglob("*.txt")):
                self._add(item.stem, item.read_text(encoding="utf-8"))
        elif path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path, "r") as archive:
                for item in sorted(archive.infolist(), key=lambda info: info.filename):
                    if item.is_dir() or item.filename.startswith("__MACOSX/"):
                        continue
                    suffix = Path(item.filename).suffix.lower()
                    if suffix == ".json":
                        self._merge(json.loads(archive.read(item).decode("utf-8")), Path(item.filename).stem)
                    elif suffix == ".txt":
                        self._add(Path(item.filename).stem, archive.read(item).decode("utf-8"))
        elif path.suffix.lower() == ".json":
            self._merge(json.loads(path.read_text(encoding="utf-8")), path.stem)
        elif path.suffix.lower() == ".txt":
            self._add(path.stem, path.read_text(encoding="utf-8"))
        else:
            raise ValueError(f"Unsupported corpus source: {path}")
        if not self.corpus:
            raise ValueError("No context records found")
        return True

    def analyze(self) -> Dict[str, Any]:
        if not self.corpus:
            raise RuntimeError("Load a corpus before analysis")
        lengths = [len(text) for text in self.corpus.values()]
        type_counts: Counter[str] = Counter()
        citation_counts: Counter[str] = Counter()
        topic_counts: Counter[str] = Counter()
        exact_hashes: Dict[str, list[str]] = defaultdict(list)
        vocabulary: Counter[str] = Counter()
        for doc_id, text in self.corpus.items():
            lowered = text.casefold()
            document_type = next((name for name, pattern in DOC_PATTERNS.items() if re.search(pattern, lowered)), "unknown")
            type_counts[document_type] += 1
            for name, pattern in CITATION_PATTERNS.items():
                citation_counts[name] += len(re.findall(pattern, lowered))
            for topic, terms in TOPICS.items():
                if any(term in lowered for term in terms):
                    topic_counts[topic] += 1
            exact_hashes[hashlib.sha256(" ".join(lowered.split()).encode("utf-8")).hexdigest()].append(doc_id)
            vocabulary.update(token for token in re.findall(r"[^\W_]+", lowered) if len(token) > 1)
        duplicate_groups = [ids for ids in exact_hashes.values() if len(ids) > 1]
        self.summary = {
            "documents": len(self.corpus),
            "characters": sum(lengths),
            "length": {
                "min": min(lengths),
                "max": max(lengths),
                "mean": sum(lengths) / len(lengths),
                "median": sorted(lengths)[len(lengths) // 2],
            },
            "document_types": dict(type_counts.most_common()),
            "citations": dict(citation_counts.most_common()),
            "topics": dict(topic_counts.most_common()),
            "unique_tokens": len(vocabulary),
            "top_tokens": vocabulary.most_common(30),
            "very_short_documents": sum(length < 100 for length in lengths),
            "exact_duplicate_groups": duplicate_groups,
            "id_types": dict(Counter(type(doc_id).__name__ for doc_id in self.corpus)),
            "recommended_chunking": {"task1_chars": 1200, "task2_chars": 1800, "overlap_chars": 150},
        }
        print(json.dumps(self.summary, ensure_ascii=False, indent=2))
        return self.summary

    def export_summary(self, output_path: str | Path) -> None:
        if not self.summary:
            self.analyze()
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    analyzer = CorpusAnalyzer()
    analyzer.load_corpus(args.corpus)
    analyzer.analyze()
    if args.export:
        analyzer.export_summary(args.export)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
