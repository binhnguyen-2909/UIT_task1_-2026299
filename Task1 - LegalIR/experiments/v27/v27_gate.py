"""Apply the preregistered V27-A clean-complement promotion gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from v27_common import read_json, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = read_json(args.report)
    overall = report["overall"]
    checks = {
        "oof_gain_ge_0.010": overall["gain"] >= 0.010,
        "robust_gain_ge_0.008": overall["robust_gain"] >= 0.008,
        "bootstrap_lb_ge_0.003": overall["bootstrap_lower_95"] >= 0.003,
        "net_query_equivalent_ge_12": overall["net_query_equivalent"] >= 12,
        "rescue_precision_ge_0.90": overall["rescue_precision"] >= 0.90,
        "destructive_le_2": overall["destructive"] <= 2,
        "top1_4_byte_identical": bool(report["contract_checks"]["top1_4_byte_identical"]),
        "empty_body_output_zero": int(report["contract_checks"]["empty_body_output"]) == 0,
    }
    regime_checks = {}
    for regime, regime_report in report["regimes"].items():
        gains = [float(value) for value in regime_report.get("fold_gains", [])]
        regime_checks[regime] = {"fold_gains": gains, "no_negative_fold": bool(gains) and min(gains) >= 0.0}
    checks["four_transfer_regimes_no_negative_fold"] = all(item["no_negative_fold"] for item in regime_checks.values())
    payload = {
        "kind": "legalir_v27_promotion_gate",
        "schema_version": 1,
        "holdout_opened": False,
        "public_labels_used": False,
        "source_report": str(args.report.resolve()),
        "checks": checks,
        "regimes": regime_checks,
        "pass": all(checks.values()),
        "holdout_action": "eligible_for_one_holdout_run" if all(checks.values()) else "do_not_open_holdout; perform_cpu_forensics",
    }
    write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
