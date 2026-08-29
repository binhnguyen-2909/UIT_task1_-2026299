"""Fail-closed validator for the BTC under-4B rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    limit = float(payload["limit_billion_parameters"])
    failed = False
    for name, profile in payload["profiles"].items():
        total = sum(float(value) for value in profile["components"].values())
        declared = float(profile["total_billion_parameters"])
        valid = total < limit and abs(total - declared) < 0.02
        print(f"{name}: total={total:.2f}B, limit<{limit:.2f}B, valid={valid}")
        if not valid:
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
