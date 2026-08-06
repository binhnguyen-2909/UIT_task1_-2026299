"""Create the labelled-warmup sanity submission (never an official inference)."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def create(warmup: Path, output: Path) -> None:
    data = json.loads(warmup.read_text(encoding="utf-8"))
    submission = {}
    for qid, item in data.items():
        if not isinstance(item, dict) or not isinstance(item.get("answer"), list):
            raise ValueError("This script only accepts labelled Task 1 warmup data")
        answers = list(dict.fromkeys(str(value) for value in item["answer"]))[:5]
        if not answers:
            raise ValueError(f"{qid}: empty labelled answer")
        submission[str(qid)] = {"answer": answers}
    output = output if output.suffix.lower() == ".zip" else output.with_suffix(".zip")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(submission, ensure_ascii=False, indent=2).encode("utf-8")
    output.with_name("submission.json").write_bytes(payload)
    info = zipfile.ZipInfo("submission.json", (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(info, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=Path, default=ROOT / "warmup.json")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "submission.zip")
    args = parser.parse_args()
    create(args.warmup, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
