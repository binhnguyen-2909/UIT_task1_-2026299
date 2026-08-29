"""Deterministic v10+ post-processing without adding a neural model.

The base v10 submission remains the anchor.  This trial only replaces slot 5
when a document title contains either an exact question phrase of at least ten
tokens or an exact legal document number.  The top four base answers are
protected, and the base index/model configuration is never modified.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import unicodedata
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


TOKEN_RE = re.compile(r"[0-9]+(?:[./-][0-9a-zđĐ]+)*|[^\W_]+", re.UNICODE)
DOC_NUMBER_RE = re.compile(
    r"(?<!\w)\d{1,6}\s*[/\-]\s*\d{4}\s*[/\-]\s*"
    r"[A-ZĐÂĂÊÔƠƯ]{1,10}(?:\s*[-/]\s*[A-ZĐÂĂÊÔƠƯ]{1,10}){0,3}(?!\w)",
)


def tokens(text: str) -> List[str]:
    normalized = unicodedata.normalize("NFD", str(text)).casefold()
    normalized = "".join(
        character for character in normalized
        if unicodedata.category(character) != "Mn"
    ).replace("đ", "d")
    return TOKEN_RE.findall(normalized)


def normalized_number(value: str) -> str:
    value = unicodedata.normalize("NFD", str(value)).casefold()
    value = "".join(
        character for character in value
        if unicodedata.category(character) != "Mn"
    ).replace("đ", "d")
    value = re.sub(r"\s+", "", value).replace("–", "-").replace("—", "-")
    return value.replace("/", "-")


def phrase_index(title_tokens: Mapping[str, Sequence[str]], lengths: Iterable[int]) -> Dict[Tuple[str, ...], List[str]]:
    wanted = set(int(value) for value in lengths)
    index: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
    for doc_id, values in title_tokens.items():
        for size in sorted(wanted):
            if len(values) < size:
                continue
            for start in range(len(values) - size + 1):
                key = tuple(values[start : start + size])
                if not index[key] or index[key][-1] != doc_id:
                    index[key].append(doc_id)
    return dict(index)


def longest_title_match(query_values: Sequence[str], title_values: Sequence[str]) -> int:
    if not query_values or not title_values:
        return 0
    title = tuple(title_values)
    for size in range(min(len(query_values), len(title)), 0, -1):
        for start in range(len(query_values) - size + 1):
            phrase = tuple(query_values[start : start + size])
            if any(title[pos : pos + size] == phrase for pos in range(len(title) - size + 1)):
                return size
    return 0


def load_submission(path: Path) -> Dict[str, Dict[str, List[str]]]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if "submission.json" not in names:
            raise ValueError(f"{path} does not contain submission.json")
        payload = json.loads(archive.read("submission.json").decode("utf-8"))
    return {
        str(qid): {"answer": [str(value) for value in item.get("answer", [])]}
        for qid, item in payload.items()
    }


def load_titles(index: Path) -> Dict[str, str]:
    uri = index.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        return {
            str(row["doc_id"]): str(row["title"] or "")
            for row in connection.execute("SELECT doc_id, title FROM document_meta")
        }
    finally:
        connection.close()


def find_number_candidates(
    question: str,
    title_numbers: Mapping[str, Sequence[str]],
) -> List[Tuple[str, int, str]]:
    query_numbers = [normalized_number(value) for value in DOC_NUMBER_RE.findall(question)]
    if not query_numbers:
        return []
    wanted = set(query_numbers)
    matches: List[Tuple[str, int, str]] = []
    for doc_id, values in title_numbers.items():
        overlap = [value for value in values if value in wanted]
        if overlap:
            matches.append((doc_id, 2, f"document_number:{overlap[0]}"))
    return matches


def rescue_one(
    question: str,
    base_answers: Sequence[str],
    title_tokens: Mapping[str, Sequence[str]],
    title_numbers: Mapping[str, Sequence[str]],
    exact_index: Mapping[Tuple[str, ...], Sequence[str]],
) -> Tuple[List[str], Dict[str, Any]]:
    answers = list(dict.fromkeys(str(value) for value in base_answers))[:5]
    while len(answers) < 5:
        break
    protected = set(answers[:4])
    candidates: List[Tuple[int, int, str, str]] = []

    number_matches = find_number_candidates(question, title_numbers)
    for doc_id, priority, reason in number_matches:
        if doc_id not in protected:
            candidates.append((priority, 100, doc_id, reason))

    query_values = tokens(question)
    for size in range(min(16, len(query_values)), 9, -1):
        for start in range(len(query_values) - size + 1):
            phrase = tuple(query_values[start : start + size])
            for doc_id in exact_index.get(phrase, ()):
                if doc_id not in protected:
                    candidates.append((1, size, doc_id, f"exact_heading:{size}"))

    if not candidates:
        return answers, {"changed": False, "reason": None}

    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    priority, match_len, doc_id, reason = candidates[0]
    old_slot = answers[4] if len(answers) >= 5 else None
    old_len = longest_title_match(query_values, title_tokens.get(old_slot, ())) if old_slot else 0

    # Exact legal number is strong enough on its own. Exact heading must beat
    # the existing fifth document, matching v57's conservative one-slot gate.
    allowed = priority >= 2 or match_len > old_len
    if not allowed or doc_id == old_slot:
        return answers, {
            "changed": False,
            "reason": reason,
            "candidate": doc_id,
            "match_length": match_len,
            "old_slot_match_length": old_len,
        }

    updated = answers[:4] + [doc_id]
    return updated, {
        "changed": True,
        "reason": reason,
        "candidate": doc_id,
        "old_slot": old_slot,
        "match_length": match_len,
        "old_slot_match_length": old_len,
    }


def write_submission(path: Path, submission: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "submission.json",
            json.dumps(submission, ensure_ascii=False, indent=2),
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    base = load_submission(args.base_zip)
    questions = json.loads(args.input.read_text(encoding="utf-8"))
    titles = load_titles(args.index)
    title_tokens = {doc_id: tokens(title) for doc_id, title in titles.items()}
    title_numbers = {
        doc_id: [normalized_number(value) for value in DOC_NUMBER_RE.findall(title)]
        for doc_id, title in titles.items()
    }
    exact_index = phrase_index(title_tokens, range(10, 17))

    submission: Dict[str, Dict[str, List[str]]] = {}
    changes: List[Dict[str, Any]] = []
    for qid, item in questions.items():
        qid = str(qid)
        answers = base.get(qid, {}).get("answer", [])
        updated, detail = rescue_one(
            str(item.get("question", "")),
            answers,
            title_tokens,
            title_numbers,
            exact_index,
        )
        submission[qid] = {"answer": updated[:5]}
        if detail.get("changed"):
            changes.append({"qid": qid, **detail})

    write_submission(args.output, submission)
    report = {
        "kind": "v10_deterministic_one_slot_rescue",
        "base_zip": str(args.base_zip.resolve()),
        "index": str(args.index.resolve()),
        "questions": len(submission),
        "changed": len(changes),
        "methods": {
            "exact_heading_min_tokens": 10,
            "exact_heading_max_tokens": 16,
            "document_number": "exact normalized title match",
            "protected_slots": 4,
            "new_model": False,
        },
        "changes": changes,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"questions": len(submission), "changed": len(changes), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
