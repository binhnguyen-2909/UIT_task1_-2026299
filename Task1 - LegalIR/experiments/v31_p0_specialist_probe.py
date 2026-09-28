"""Encode the existing V30-A0 probe with the aggressive LoRA specialist."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, SCRIPT_DIR / "v27", SCRIPT_DIR / "v30"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from v27_common import sha256_file, write_json, write_jsonl  # noqa: E402
from v30_a0_screen import LoRAEncoder, _read_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--encode-batch-size", type=int, default=512)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(args.probe_manifest)
    unique_texts: list[str] = []
    text_index: dict[str, int] = {}
    per_query_indices: list[list[list[int]]] = []
    for row in rows:
        query_indices = []
        for candidate in row["candidates"]:
            candidate_indices = []
            for chunk in candidate["chunks"]:
                text = str(chunk["text"])
                if text not in text_index:
                    text_index[text] = len(unique_texts)
                    unique_texts.append(text)
                candidate_indices.append(text_index[text])
            query_indices.append(candidate_indices)
        per_query_indices.append(query_indices)
    encoder = LoRAEncoder(args.model, args.device, 96, 256, adapter_path=args.adapter)
    started = time.perf_counter()
    query_vectors = encoder.encode([str(row["query"]) for row in rows], 96, max(1, min(args.encode_batch_size, 64)))
    chunk_vectors = encoder.encode(unique_texts, 256, args.encode_batch_size)
    output_rows = []
    for qi, row in enumerate(rows):
        candidates = row["candidates"]
        scores = [float(np.max(query_vectors[qi] @ chunk_vectors[indices].T)) if indices else float("-inf") for indices in per_query_indices[qi]]
        order = sorted(range(len(candidates)), key=lambda index: (-scores[index], int(candidates[index]["dense_rank"]), str(candidates[index]["doc_id"])))
        output_rows.append({
            "qid": str(row["qid"]),
            "references": [str(docid) for docid in row["references"]],
            "candidates": [{"doc_id": str(candidates[index]["doc_id"]), "dense_rank": int(candidates[index]["dense_rank"]), "dense_score": float(candidates[index]["dense_score"]), "ft_dense_rank": rank, "ft_dense_score": scores[index]} for rank, index in enumerate(order, 1)],
        })
    write_jsonl(args.output, output_rows)
    report = {
        "kind": "legalir_v31_p0_a0_specialist_probe_scores",
        "model": "BAAI/bge-m3",
        "model_path": str(args.model.resolve()),
        "adapter_path": str(args.adapter.resolve()),
        "probe_manifest": str(args.probe_manifest.resolve()),
        "probe_manifest_sha256": sha256_file(args.probe_manifest),
        "queries": len(rows),
        "unique_probe_chunks": len(unique_texts),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "encode_batch_size": args.encode_batch_size,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(args.output.with_name("v31_p0_specialist_probe_report.json"), report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
