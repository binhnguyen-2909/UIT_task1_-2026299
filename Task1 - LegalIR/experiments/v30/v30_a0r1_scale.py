"""V30-A0R1: probe-only LoRA adapter strength sweep."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v30_a0_screen import LoRAEncoder, _probe_eval, _read_jsonl  # noqa: E402
from v27_common import sha256_file, write_json  # noqa: E402


def _set_scale(encoder: LoRAEncoder, scale: float) -> int:
    changed = 0
    for module in encoder.model.modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict) and "default" in scaling and hasattr(module, "set_scale"):
            module.set_scale("default", scale)
            changed += 1
    if changed == 0:
        raise RuntimeError("no active default LoRA modules found")
    return changed


def _compact(metrics: dict[str, Any], scale: float) -> dict[str, Any]:
    baseline_median = float(metrics["baseline_median_gold_rank"])
    return {
        "scale": scale,
        "gold_docs_at_100": int(metrics["gold_docs_at_100"]),
        "gold_docs_at_100_gain": int(metrics["gold_docs_at_100"]) - int(metrics["baseline_gold_docs_at_100"]),
        "median_gold_rank": float(metrics["median_gold_rank"]),
        "median_ratio": float(metrics["median_gold_rank"]) / max(baseline_median, 1e-9),
        "hard_negative_win": float(metrics["hard_negative_win_rate"]),
        "hard_negative_win_gain": None,
        "lost_top50": int(metrics["lost_top50"]),
        "gained_top50": int(metrics.get("gained_top50", 0)),
    }


def _gate(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "gold_docs_at_100_gain_ge_5": int(metrics["gold_docs_at_100_gain"]) >= 5,
        "median_ratio_le_0_70": float(metrics["median_ratio"]) <= 0.70,
        "hard_negative_win_gain_ge_0_08": float(metrics["hard_negative_win_gain"]) >= 0.08,
        "lost_top50_le_2": int(metrics["lost_top50"]) <= 2,
        "pass": bool(int(metrics["gold_docs_at_100_gain"]) >= 5 and float(metrics["median_ratio"]) <= 0.70 and float(metrics["hard_negative_win_gain"]) >= 0.08 and int(metrics["lost_top50"]) <= 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--encode-batch-size", type=int, default=512)
    parser.add_argument("--scales", default="0,0.25,0.5,0.75,1")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    probe_rows = _read_jsonl(args.probe_manifest)
    scales = [float(value) for value in args.scales.split(",")]
    if 0.0 not in scales:
        scales.insert(0, 0.0)
    started = time.perf_counter()

    baseline = _probe_eval(None, probe_rows, args.encode_batch_size, baseline_only=True)
    results: list[dict[str, Any]] = []
    raw_results: dict[str, dict[str, Any]] = {}
    for scale in scales:
        if scale == 0.0:
            metrics = dict(baseline)
        else:
            if "encoder" not in locals():
                encoder = LoRAEncoder(args.model, args.device, 96, 256, adapter_path=args.adapter)
            _set_scale(encoder, scale)
            metrics = _probe_eval(encoder, probe_rows, args.encode_batch_size)
        compact = _compact(metrics, scale)
        # The teacher baseline is the frozen dense ranking, not Stage2.
        compact["hard_negative_win_gain"] = float(metrics["hard_negative_win_rate"]) - float(baseline["hard_negative_win_rate"])
        compact["gate"] = _gate(compact)
        results.append(compact)
        raw_results[str(scale)] = metrics
        print(json.dumps({"scale": scale, "metrics": compact}, ensure_ascii=False), flush=True)

    passed = [row for row in results if row["gate"]["pass"]]
    selected = min(passed, key=lambda row: (row["median_gold_rank"], -row["gold_docs_at_100"])) if passed else None
    report = {
        "kind": "legalir_v30_a0r1_adapter_scale_sweep",
        "schema_version": 1,
        "model": "BAAI/bge-m3",
        "model_path": str(args.model.resolve()),
        "adapter_path": str(args.adapter.resolve()),
        "probe_manifest": str(args.probe_manifest.resolve()),
        "probe_manifest_sha256": sha256_file(args.probe_manifest),
        "scales": scales,
        "encode_batch_size": args.encode_batch_size,
        "results": results,
        "selected": selected,
        "raw_results": raw_results,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(args.output, report)
    print(json.dumps({"report": str(args.output.resolve()), "selected": selected, "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
