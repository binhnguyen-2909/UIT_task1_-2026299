"""Validate DSC 2026 Task 1/Task 2 submissions before upload."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from submission_utils import SubmissionError, validate_submission_file


REPO_ROOT = Path(__file__).resolve().parent.parent


def verify(task: int, submission_path: Path, input_path: Path | None) -> bool:
    """Validate one submission and print a compact report."""
    label = "LegalIR" if task == 1 else "LegalQA"
    print(f"\nTask {task} - {label}")
    print("-" * 60)
    try:
        submission, errors = validate_submission_file(
            submission_path, task=task, input_path=input_path
        )
    except (OSError, ValueError, SubmissionError) as exc:
        print(f"[INVALID] {exc}")
        return False

    print(f"Questions: {len(submission)}")
    if task == 1:
        counts = [len(item["answer"]) for item in submission.values()]
        print(f"Predicted doc IDs: {sum(counts)} (avg {sum(counts)/len(counts):.2f})")
    else:
        lengths = [len(item["answer"]) for item in submission.values()]
        print(f"Answer characters: {sum(lengths):,} (avg {sum(lengths)/len(lengths):.0f})")

    if errors:
        print(f"[INVALID] {len(errors)} error(s)")
        for error in errors[:20]:
            print(f"  - {error}")
        return False
    print("[VALID] Format, IDs, and archive contract passed")
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=("1", "2", "both"), default="both",
        help="Validate one task independently or both tasks together",
    )
    parser.add_argument(
        "--task1-submission",
        type=Path,
        default=(
            REPO_ROOT
            / "Task1 - LegalIR"
            / "output"
            / "public_submission_v10_ce_rrf_top35.zip"
        ),
    )
    parser.add_argument(
        "--task2-submission",
        type=Path,
        default=REPO_ROOT / "Task2 - LegalQA" / "output" / "submission.zip",
    )
    parser.add_argument(
        "--task1-input",
        type=Path,
        default=None,
        help="optional warmup/public-official JSON used to enforce exact question IDs",
    )
    parser.add_argument(
        "--task2-input",
        type=Path,
        default=REPO_ROOT / "Task2 - LegalQA" / "warmup.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print("=" * 60)
    print("DSC 2026 SUBMISSION VERIFICATION")
    print("=" * 60)
    checks = []
    if args.task in ("1", "both"):
        checks.append(verify(1, args.task1_submission, args.task1_input))
    if args.task in ("2", "both"):
        checks.append(verify(2, args.task2_submission, args.task2_input))
    ready = all(checks)
    print("\n" + ("READY TO UPLOAD" if ready else "FIX ERRORS BEFORE UPLOAD"))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
