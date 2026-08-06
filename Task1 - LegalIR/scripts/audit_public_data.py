"""Deep, streaming audit of every Task 1 public-test file supplied by BTC."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TASK_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = TASK_ROOT / "data" / "raw" / "public_test"


def percentile(sorted_values: List[int], ratio: float) -> int:
    if not sorted_values:
        return 0
    return sorted_values[round((len(sorted_values) - 1) * ratio)]


def load_question_file(path: Path, labelled: bool) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} root must be a JSON object")
    errors: List[str] = []
    questions: List[str] = []
    answer_counts: Counter[int] = Counter()
    answer_ids = set()
    for qid, item in data.items():
        if not isinstance(qid, str) or not qid:
            errors.append(f"invalid question id: {qid!r}")
            continue
        if not isinstance(item, dict):
            errors.append(f"{qid}: value is not an object")
            continue
        question = item.get("question")
        if not isinstance(question, str) or not question.strip():
            errors.append(f"{qid}: invalid question")
        else:
            questions.append(" ".join(question.casefold().split()))
        answer = item.get("answer")
        if labelled:
            if not isinstance(answer, list) or not 1 <= len(answer) <= 5:
                errors.append(f"{qid}: labelled answer must contain 1-5 IDs")
            else:
                answer_counts[len(answer)] += 1
                answer_ids.update(str(value) for value in answer)
                if len(answer) != len(set(map(str, answer))):
                    errors.append(f"{qid}: duplicate answer IDs")
        elif answer is not None:
            errors.append(f"{qid}: public answer must be null")
    duplicate_questions = sum(count - 1 for count in Counter(questions).values() if count > 1)
    return data, {
        "path": str(path),
        "questions": len(data),
        "fields": sorted(set().union(*(item.keys() for item in data.values() if isinstance(item, dict)))),
        "errors": errors[:100],
        "error_count": len(errors),
        "duplicate_normalized_questions": duplicate_questions,
        "answer_count_distribution": dict(sorted(answer_counts.items())),
        "unique_answer_ids": len(answer_ids),
    }


def audit_corpus(archive_path: Path, extracted_root: Path) -> Tuple[set[str], Dict[str, Any]]:
    errors: List[str] = []
    ids: set[str] = set()
    filename_ids: set[str] = set()
    passage_hashes: Dict[str, List[str]] = defaultdict(list)
    lengths: List[int] = []
    raw_hashes: Dict[str, str] = {}
    raw_sizes: Dict[str, int] = {}
    field_counts: Counter[Tuple[str, ...]] = Counter()
    estimated_chunks = 0
    empty_passage_ids: List[str] = []

    with zipfile.ZipFile(archive_path, "r") as archive:
        # Every member is read below.  ``ZipFile.read`` already verifies CRC,
        # so a separate ``testzip`` pass would decompress the 489 MB corpus
        # twice without adding coverage.
        corrupt = None
        files = [
            item for item in archive.infolist()
            if not item.is_dir() and not item.filename.startswith("__MACOSX/")
        ]
        for item in files:
            if Path(item.filename).suffix.lower() != ".json":
                errors.append(f"non-JSON archive member: {item.filename}")
                continue
            raw = archive.read(item)
            relative_name = Path(item.filename).name
            raw_hashes[relative_name] = hashlib.sha256(raw).hexdigest()
            raw_sizes[relative_name] = len(raw)
            try:
                record = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                errors.append(f"{item.filename}: invalid UTF-8/JSON: {exc}")
                continue
            if not isinstance(record, dict):
                errors.append(f"{item.filename}: root is not an object")
                continue
            field_counts[tuple(sorted(record))] += 1
            doc_id = str(record.get("id", "")).strip()
            passage = record.get("passage")
            match = re.fullmatch(r"context_(.+)\.json", relative_name)
            filename_id = match.group(1) if match else ""
            if not doc_id:
                errors.append(f"{item.filename}: missing id")
                continue
            if doc_id in ids:
                errors.append(f"duplicate context id: {doc_id}")
            ids.add(doc_id)
            if filename_id:
                filename_ids.add(filename_id)
            if filename_id != doc_id:
                errors.append(f"{item.filename}: filename id {filename_id!r} != record id {doc_id!r}")
            # BTC supplies two valid context shapes: some records omit ``name``.
            if "name" in record and not isinstance(record.get("name"), str):
                errors.append(f"{item.filename}: name is present but not a string")
            if not isinstance(record.get("link"), str):
                errors.append(f"{item.filename}: link is not a string")
            if not isinstance(passage, str):
                errors.append(f"{item.filename}: passage is not a string")
                continue
            if not passage.strip():
                # Empty official passages are retained and indexed from their
                # title/URL surrogate.  They are a data warning, not corruption.
                empty_passage_ids.append(doc_id)
                continue
            length = len(passage)
            lengths.append(length)
            estimated_chunks += 1 if length <= 1200 else max(1, (length - 150 + 1049) // 1050)
            passage_hashes[hashlib.sha256(" ".join(passage.casefold().split()).encode("utf-8")).hexdigest()].append(doc_id)

    extracted_files = sorted(extracted_root.rglob("context_*.json")) if extracted_root.exists() else []
    extracted_names = {path.name for path in extracted_files}
    archive_names = set(raw_hashes)
    mismatch_hashes: List[str] = []
    for path in extracted_files:
        expected_hash = raw_hashes.get(path.name)
        if expected_hash is None:
            continue
        raw = path.read_bytes()
        if len(raw) != raw_sizes[path.name] or hashlib.sha256(raw).hexdigest() != expected_hash:
            mismatch_hashes.append(path.name)

    sorted_lengths = sorted(lengths)
    duplicate_passages = [values for values in passage_hashes.values() if len(values) > 1]
    report = {
        "archive": str(archive_path),
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "zip_test_error_member": corrupt,
        "contexts": len(ids),
        "json_members": len(raw_hashes),
        "uncompressed_bytes": sum(raw_sizes.values()),
        "field_shapes": {"|".join(fields): count for fields, count in field_counts.items()},
        "passage_length": {
            "min": min(lengths) if lengths else 0,
            "p25": percentile(sorted_lengths, 0.25),
            "median": percentile(sorted_lengths, 0.50),
            "p75": percentile(sorted_lengths, 0.75),
            "p95": percentile(sorted_lengths, 0.95),
            "p99": percentile(sorted_lengths, 0.99),
            "max": max(lengths) if lengths else 0,
            "mean": statistics.fmean(lengths) if lengths else 0,
            "total": sum(lengths),
        },
        "estimated_chunks_1200_overlap150": estimated_chunks,
        "empty_passage_count": len(empty_passage_ids),
        "empty_passage_ids": empty_passage_ids,
        "duplicate_passage_groups": duplicate_passages[:100],
        "duplicate_passage_group_count": len(duplicate_passages),
        "schema_error_count": len(errors),
        "schema_errors": errors[:100],
        "extracted": {
            "root": str(extracted_root),
            "files": len(extracted_files),
            "missing_from_extracted": sorted(archive_names - extracted_names)[:100],
            "extra_in_extracted": sorted(extracted_names - archive_names)[:100],
            "raw_content_mismatch_count": len(mismatch_hashes),
            "raw_content_mismatches": mismatch_hashes[:100],
        },
    }
    return ids, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=TASK_ROOT / "reports" / "public_data_audit.json")
    args = parser.parse_args()
    root = args.root
    train, train_report = load_question_file(root / "train.json", labelled=True)
    public, public_report = load_question_file(root / "public-official.json", labelled=False)
    corpus_ids, corpus_report = audit_corpus(
        root / "selected-contexts.zip", root / "selected-contexts"
    )
    train_answer_ids = {
        str(doc_id) for item in train.values() for doc_id in item.get("answer", [])
    }
    train_questions = {" ".join(item["question"].casefold().split()) for item in train.values()}
    public_questions = {" ".join(item["question"].casefold().split()) for item in public.values()}
    report = {
        "train": train_report,
        "public": public_report,
        "corpus": corpus_report,
        "cross_checks": {
            "train_public_id_overlap": len(set(train) & set(public)),
            "train_public_exact_question_overlap": len(train_questions & public_questions),
            "train_answer_ids_missing_from_corpus": sorted(train_answer_ids - corpus_ids),
            "corpus_ids_unused_by_train": len(corpus_ids - train_answer_ids),
        },
        "ready": (
            not train_report["error_count"]
            and not public_report["error_count"]
            and not corpus_report["schema_error_count"]
            and corpus_report["zip_test_error_member"] is None
            and not corpus_report["extracted"]["missing_from_extracted"]
            and not corpus_report["extracted"]["extra_in_extracted"]
            and not corpus_report["extracted"]["raw_content_mismatch_count"]
            and not (train_answer_ids - corpus_ids)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "ready": report["ready"],
        "train_questions": len(train),
        "public_questions": len(public),
        "contexts": corpus_report["contexts"],
        "estimated_chunks": corpus_report["estimated_chunks_1200_overlap150"],
        "schema_errors": corpus_report["schema_error_count"],
        "duplicate_passage_groups": corpus_report["duplicate_passage_group_count"],
        "extracted_content_mismatches": corpus_report["extracted"]["raw_content_mismatch_count"],
        "missing_train_answer_ids": len(train_answer_ids - corpus_ids),
        "report": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
