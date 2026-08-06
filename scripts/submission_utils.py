"""Shared DSC 2026 input/submission contract helpers.

The official scorers assume that prediction IDs exactly match the reference
IDs.  These helpers validate that invariant before an archive is uploaded.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


class SubmissionError(ValueError):
    """Raised when an input or submission violates the public-test contract."""


def load_json_object(path: Path | str) -> Dict[str, Any]:
    """Load a UTF-8 JSON object and reject non-object roots."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise SubmissionError(f"{path} must contain a JSON object at the root")
    return data


def load_submission(path: Path | str) -> Dict[str, Any]:
    """Load submission JSON directly or from a strict submission.zip."""
    path = Path(path)
    if path.suffix.lower() != ".zip":
        return load_json_object(path)

    with zipfile.ZipFile(path, "r") as archive:
        files = [info for info in archive.infolist() if not info.is_dir()]
        names = [info.filename for info in files]
        if names != ["submission.json"]:
            raise SubmissionError(
                f"{path} must contain exactly one root file named "
                f"submission.json; found {names}"
            )
        try:
            data = json.loads(archive.read(files[0]).decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise SubmissionError("submission.json must be UTF-8") from exc
    if not isinstance(data, dict):
        raise SubmissionError("submission.json must contain a JSON object")
    return data


def expected_ids(input_path: Path | str) -> List[str]:
    """Return question IDs from warmup/public/private input JSON."""
    return list(load_json_object(input_path).keys())


def validate_submission(
    submission: Mapping[str, Any],
    task: int,
    expected_question_ids: Optional[Iterable[str]] = None,
) -> List[str]:
    """Return all contract violations without stopping at the first error."""
    if task not in (1, 2):
        raise ValueError("task must be 1 or 2")

    errors: List[str] = []
    actual_ids = set(submission.keys())
    if expected_question_ids is not None:
        expected = set(expected_question_ids)
        missing = sorted(expected - actual_ids)
        extra = sorted(actual_ids - expected)
        if missing:
            errors.append(
                f"Missing {len(missing)} question IDs; first: {missing[:5]}"
            )
        if extra:
            errors.append(f"Unexpected {len(extra)} question IDs; first: {extra[:5]}")

    for qid, item in submission.items():
        if not isinstance(qid, str) or not qid:
            errors.append(f"Invalid question ID: {qid!r}")
            continue
        if not isinstance(item, dict):
            errors.append(f"{qid}: value must be an object")
            continue
        if "answer" not in item:
            errors.append(f"{qid}: missing 'answer' field")
            continue

        answer = item["answer"]
        if task == 1:
            if not isinstance(answer, list):
                errors.append(f"{qid}: answer must be a list")
                continue
            if not 1 <= len(answer) <= 5:
                errors.append(f"{qid}: answer must contain 1 to 5 doc IDs")
            if any(not isinstance(doc_id, str) or not doc_id.strip() for doc_id in answer):
                errors.append(f"{qid}: every doc ID must be a non-empty string")
            if len(answer) != len(set(answer)):
                errors.append(f"{qid}: duplicate doc IDs are not allowed")
        else:
            if not isinstance(answer, str):
                errors.append(f"{qid}: answer must be a string")
            elif not answer.strip():
                errors.append(f"{qid}: answer must not be empty")

    return errors


def validate_submission_file(
    submission_path: Path | str,
    task: int,
    input_path: Optional[Path | str] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Load and validate one JSON/ZIP submission."""
    submission = load_submission(submission_path)
    ids = expected_ids(input_path) if input_path else None
    return submission, validate_submission(submission, task, ids)


def write_submission_bundle(
    submission: Mapping[str, Any], output_path: Path | str
) -> Tuple[Path, Path]:
    """Write sibling submission.json and deterministic submission.zip."""
    output_path = Path(output_path)
    zip_path = output_path if output_path.suffix.lower() == ".zip" else output_path.with_suffix(".zip")
    json_path = zip_path.with_name("submission.json")
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(submission, ensure_ascii=False, indent=2).encode("utf-8")
    json_path.write_bytes(payload)

    info = zipfile.ZipInfo("submission.json", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(info, payload)
    return json_path, zip_path

