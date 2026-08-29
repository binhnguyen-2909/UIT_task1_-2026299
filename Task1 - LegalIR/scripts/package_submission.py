"""Validate and package a LegalIR answer mapping as a scorer submission ZIP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-questions", type=int, default=None)
    parser.add_argument(
        "--expected-question-data",
        type=Path,
        default=None,
        help=(
            "Official public-question JSON. When supplied, require an exact "
            "question-ID match before creating the ZIP."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Submission must be a non-empty JSON object")
    normalized = {}
    for question_id, value in payload.items():
        if not isinstance(value, dict) or not isinstance(value.get("answer"), list):
            raise ValueError(f"Invalid answer object for question {question_id}")
        answers = [str(item) for item in value["answer"]]
        if not 1 <= len(answers) <= 5:
            raise ValueError(f"Question {question_id} has {len(answers)} answers; expected 1..5")
        if len(set(answers)) != len(answers):
            raise ValueError(f"Question {question_id} contains duplicate answer IDs")
        normalized[str(question_id)] = {"answer": answers}
    if args.expected_questions is not None and len(normalized) != args.expected_questions:
        raise ValueError(
            f"Expected {args.expected_questions} questions, got {len(normalized)}"
        )
    if args.expected_question_data is not None:
        expected_payload = json.loads(
            args.expected_question_data.read_text(encoding="utf-8")
        )
        if not isinstance(expected_payload, dict) or not expected_payload:
            raise ValueError("Expected question data must be a non-empty JSON object")
        expected_ids = {str(question_id) for question_id in expected_payload}
        actual_ids = set(normalized)
        if actual_ids != expected_ids:
            missing = sorted(expected_ids - actual_ids)
            unexpected = sorted(actual_ids - expected_ids)
            raise ValueError(
                "Submission question IDs do not match expected data: "
                f"missing={len(missing)} unexpected={len(unexpected)} "
                f"missing_sample={missing[:3]} unexpected_sample={unexpected[:3]}"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(normalized, ensure_ascii=False, indent=2)
    with ZipFile(args.output, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("submission.json", text)
    print(json.dumps({
        "questions": len(normalized),
        "question_id_match": args.expected_question_data is not None,
        "output": str(args.output.resolve()),
        "member": "submission.json",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
