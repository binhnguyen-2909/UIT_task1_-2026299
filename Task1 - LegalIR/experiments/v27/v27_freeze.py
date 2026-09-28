"""Freeze the V27 clean-complement baseline and grouped four-fold split."""

from __future__ import annotations

import argparse
from pathlib import Path

from v27_common import (
    SEED,
    baseline_recall,
    build_grouped_folds,
    canonical_json,
    load_metadata,
    sha256_file,
    sha256_lines,
    write_json,
    write_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--group-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    questions = load_metadata(args.metadata)
    out = args.output_dir
    (out / "baseline").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "splits").mkdir(parents=True, exist_ok=True)
    qids = list(questions)

    qrel_rows = [
        {"qid": qid, "docids": sorted(questions[qid]["references"], key=lambda value: (int(value) if value.isdigit() else 10**18, value))}
        for qid in qids
    ]
    c35_rows = [{"qid": qid, "docids": questions[qid]["c35"]} for qid in qids]
    baseline_rows = [
        {"qid": qid, "query": questions[qid]["query"], "candidates_c35": questions[qid]["c35"], "answer": questions[qid]["anchor"]}
        for qid in qids
    ]
    qrel_hash = write_jsonl(out / "data" / "qrels.jsonl", qrel_rows)
    c35_hash = write_jsonl(out / "data" / "c35_manifest.jsonl", c35_rows)
    baseline_hash = write_jsonl(out / "baseline" / "baseline_predictions.jsonl", baseline_rows)
    split = build_grouped_folds(qids, args.group_manifest, n_splits=4)
    write_json(out / "splits" / "grouped_4fold.json", split)

    input_hashes = {
        "train_json": sha256_file(args.train_json),
        "metadata": sha256_file(args.metadata),
        "group_manifest": sha256_file(args.group_manifest),
        "qrels": qrel_hash,
        "c35_manifest": c35_hash,
        "baseline_predictions": baseline_hash,
    }
    report = {
        "kind": "legalir_v27_baseline_freeze",
        "schema_version": 1,
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "seed": SEED,
        "baseline_recall_clean_complement": baseline_recall(questions),
        "baseline_recall_locked_reference": 0.922833,
        "candidate_pool": "c35",
        "candidate_count_per_query": 35,
        "top1_4_locked": True,
        "empty_body_veto": True,
        "queries": len(questions),
        "input_hashes": input_hashes,
        "canonical_json_contract": canonical_json({"seed": SEED, "candidate_pool": "c35", "top1_4_locked": True, "empty_body_veto": True}),
    }
    write_json(out / "baseline" / "baseline_report.json", report)
    write_json(out / "baseline" / "input_hashes.json", input_hashes)
    print({"output": str(out.resolve()), "queries": len(questions), "clean_complement_baseline": report["baseline_recall_clean_complement"], "folds": len(split["folds"]), "hashes": input_hashes})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
