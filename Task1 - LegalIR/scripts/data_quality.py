"""Deterministic, BTC-corpus-only preprocessing helpers for LegalIR.

The competition data is the source of truth. These helpers only normalize
representation and merge repeated labels; they never create legal text or
fetch anything outside the supplied files.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple


_HORIZONTAL_SPACE_RE = re.compile(r"[ \t\f\v\u00a0]+")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize_question(text: Any) -> str:
    """Return a stable key while preserving Vietnamese characters."""

    value = unicodedata.normalize("NFC", str(text or ""))
    value = value.replace("\u200b", "").replace("\ufeff", "")
    return " ".join(value.split()).casefold()


def normalize_corpus_text(text: Any) -> str:
    """Normalize layout noise without deleting legal punctuation or negation."""

    value = unicodedata.normalize("NFC", str(text or ""))
    value = value.replace("\u200b", "").replace("\ufeff", "")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = _CONTROL_RE.sub("", value)
    return "\n".join(
        _HORIZONTAL_SPACE_RE.sub(" ", line).strip()
        for line in value.split("\n")
    ).strip()


def collapse_training_questions(
    data: Mapping[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """Collapse duplicate questions and union their official labels.

    The first record remains the representative so provenance stays stable;
    the answer list is the ordered union of labels observed in all duplicate
    rows. This avoids turning one duplicate row into a false negative for
    another and avoids overweighting duplicates in question memory.
    """

    cleaned: Dict[str, Dict[str, Any]] = {}
    groups: Dict[str, str] = {}
    merged = 0
    conflicting = 0
    skipped = 0
    for raw_qid, raw_item in data.items():
        if not isinstance(raw_item, dict) or not isinstance(raw_item.get("question"), str):
            skipped += 1
            continue
        question = normalize_corpus_text(raw_item["question"])
        key = normalize_question(question)
        if not key:
            skipped += 1
            continue
        answers = raw_item.get("answer")
        if not isinstance(answers, list):
            skipped += 1
            continue
        qid = str(raw_qid)
        representative = groups.get(key)
        if representative is None:
            groups[key] = qid
            cleaned[qid] = {
                "question": question,
                "answer": list(dict.fromkeys(str(value) for value in answers)),
            }
            continue
        merged += 1
        target = cleaned[representative]
        before = set(target["answer"])
        target["answer"] = list(dict.fromkeys([
            *target["answer"],
            *(str(value) for value in answers),
        ]))
        if set(target["answer"]) != before:
            conflicting += 1

    return cleaned, {
        "input_records": len(data),
        "clean_records": len(cleaned),
        "duplicate_question_rows_merged": merged,
        "conflicting_duplicate_groups_unioned": conflicting,
        "skipped_invalid_rows": skipped,
    }


def load_training_records(
    path: Path | None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    if path is None:
        return {}, {
            "input_records": 0,
            "clean_records": 0,
            "duplicate_question_rows_merged": 0,
            "conflicting_duplicate_groups_unioned": 0,
            "skipped_invalid_rows": 0,
        }
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Training data must be a JSON object keyed by question ID")
    return collapse_training_questions(payload)


def load_raw_training_records(path: Path | None) -> Dict[str, Any]:
    """Load the supplied labels without changing them (v10 compatibility)."""

    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Training data must be a JSON object keyed by question ID")
    return payload
