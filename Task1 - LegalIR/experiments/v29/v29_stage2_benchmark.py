"""Small label-free Stage2 throughput benchmark for the V29 handoff."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for path in (REPO_ROOT / "scripts", SCRIPT_DIR.parent / "v27"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from batch_stage2_rerank import hydrate, load_passage_store, open_readonly  # noqa: E402
from v27_common import load_metadata, sha256_file, write_json  # noqa: E402
from v29_a1d1_stage2_diag import DirectStage2, _benchmark  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    questions = load_metadata(args.metadata)
    qids = list(questions)[:4]
    doc_ids = sorted({doc_id for qid in qids for doc_id in questions[qid]["c35"][:8]})
    connection = open_readonly(args.index)
    store = load_passage_store(connection, doc_ids)
    connection.close()
    pairs = []
    for qid in qids:
        hydrated = hydrate(questions[qid]["query"], questions[qid]["c35"][:8], 3, store)
        pairs.extend((questions[qid]["query"], str(passage)) for item in hydrated for passage in item["passages"] if str(passage).strip())
    scorer = DirectStage2(args.model, args.device, args.max_length, args.fp16)
    rows = _benchmark(scorer, pairs, [4, 8, 12], args.device)
    report = {
        "kind": "legalir_v29_stage2_efficiency",
        "schema_version": 1,
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "inference_contract": {
            "model": str(args.model.resolve()),
            "model_config_sha256": sha256_file(args.model / "config.json"),
            "max_length": args.max_length,
            "dynamic_padding": True,
            "eval": True,
            "inference_mode": True,
            "fp16": bool(args.fp16),
        },
        "sample": {"qids": qids, "pairs_available": len(pairs), "pairs_measured_per_batch": min(len(pairs), 16)},
        "device": args.device,
        "benchmark": rows,
        "selected_batch_size": max(rows, key=lambda row: float(row["pairs_per_second"]))["batch_size"],
    }
    write_json(args.output, report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
