"""Materialize the clean v11 holdout as train minus fit-training IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-questions", type=int, default=1000)
    args = parser.parse_args()
    train = json.loads(args.train.read_text(encoding="utf-8"))
    fit = json.loads(args.fit.read_text(encoding="utf-8"))
    holdout = {str(qid): item for qid, item in train.items() if str(qid) not in fit}
    if len(holdout) != args.expected_questions:
        raise ValueError(
            f"expected {args.expected_questions} holdout questions, found {len(holdout)}"
        )
    if set(holdout) & set(fit) or set(holdout) | set(fit) != set(map(str, train)):
        raise ValueError("fit/holdout partition is invalid")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(holdout, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "train": len(train), "fit": len(fit), "holdout": len(holdout),
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
