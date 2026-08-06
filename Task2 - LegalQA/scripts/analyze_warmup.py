"""Portable Task 2 warmup statistics."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


ROOT = Path(__file__).resolve().parent.parent


def analyze(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    question_lengths = [len(item["question"]) for item in data.values()]
    answer_lengths = [len(item["answer"]) for item in data.values()]
    references = Counter()
    import re
    for item in data.values():
        references.update(match.casefold() for match in re.findall(
            r"\b(?:Điều|Khoản|Điểm|Chương)\s+(?:\d+[a-z]?|[IVXLCDM]+)", item["answer"], re.IGNORECASE
        ))
    result = {
        "questions": len(data),
        "question_length": {
            "min": min(question_lengths), "max": max(question_lengths),
            "mean": statistics.fmean(question_lengths), "median": statistics.median(question_lengths),
        },
        "answer_length": {
            "min": min(answer_lengths), "max": max(answer_lengths),
            "mean": statistics.fmean(answer_lengths), "median": statistics.median(answer_lengths),
        },
        "top_legal_references": references.most_common(30),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=Path, default=ROOT / "warmup.json")
    args = parser.parse_args()
    analyze(args.warmup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
