"""Summarize failed V27 promotion runs without opening holdout artifacts."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from v27_common import read_json, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-report", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--negative-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = read_json(args.oof_report)
    audit = read_json(args.negative_audit)
    rows = [json.loads(line) for line in args.decisions.read_text(encoding="utf-8").splitlines() if line.strip()]
    rescues = [row for row in rows if float(row["delta"]) > 1e-12]
    destructive = [row for row in rows if float(row["delta"]) < -1e-12]
    changed_neutral = [row for row in rows if bool(row.get("changed")) and abs(float(row["delta"])) <= 1e-12]
    by_fold = {}
    for fold in sorted({int(row["outer_fold"]) for row in rows}):
        fold_rows = [row for row in rows if int(row["outer_fold"]) == fold]
        by_fold[str(fold)] = {
            "queries": len(fold_rows),
            "rescues": sum(float(row["delta"]) > 1e-12 for row in fold_rows),
            "destructive": sum(float(row["delta"]) < -1e-12 for row in fold_rows),
            "changed_neutral": sum(bool(row.get("changed")) and abs(float(row["delta"])) <= 1e-12 for row in fold_rows),
            "gain": sum(float(row["delta"]) for row in fold_rows) / max(len(fold_rows), 1),
        }
    payload = {
        "kind": "legalir_v27_failed_gate_forensics",
        "schema_version": 1,
        "holdout_opened": False,
        "public_labels_used": False,
        "source_report": str(args.oof_report.resolve()),
        "gate": read_json(args.oof_report).get("contract_checks", {}),
        "overall": report["overall"],
        "by_outer_fold": by_fold,
        "thresholds": report.get("fold_thresholds", []),
        "decision_reasons": dict(collections.Counter(str(row.get("reason")) for row in rows)),
        "rescues": rescues,
        "destructive": destructive,
        "changed_neutral": changed_neutral,
        "training_audit": {"rows": audit["training_rows"], "negative_mix": audit["rows_by_negative_type"], "fallbacks": audit["fallback_rows_by_type"], "qids_without_positive_in_c35": len(audit["qids_without_positive_in_c35"])},
        "diagnosis": [
            "Intervention selector made only a small number of beneficial edits; most edits were neutral.",
            "One destructive edit makes rescue precision 0.75, below the 0.90 contract.",
            "The standard-grouped OOF fold map is balanced, while the four regime maps are diagnostic fold maps rather than jointly isolated training splits.",
        ],
        "decision": "reject_v27_and_do_not_open_holdout",
    }
    write_json(args.output, payload)
    print(json.dumps({"output": str(args.output.resolve()), "rescues": len(rescues), "destructive": len(destructive), "changed_neutral": len(changed_neutral)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
