"""Freeze the completed V29-A1D Stage2 score cache for CPU-only analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR.parent / "v27",):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from v27_common import load_metadata, read_json, sha256_file, write_json, write_jsonl  # noqa: E402


def _numeric_key(value: str) -> tuple[int, str]:
    return (int(value), value) if str(value).isdigit() else (10**18, str(value))


def _fold(qid: str, question: Mapping[str, Any]) -> int | str:
    value = question.get("fold")
    return int(value) if value is not None else int(qid) % 5


def _stage1(path: Path, questions: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    payload = read_json(path)
    result = {str(row["query_id"]): [str(doc) for doc in docs] for row, docs in zip(payload["queries"], payload["stage1_candidates"], strict=True)}
    if set(questions) - set(result):
        raise ValueError("Stage1 audit misses current metadata qids")
    for qid, question in questions.items():
        if result[qid][:35] != list(question["c35"]):
            raise ValueError(f"{qid}: Stage1 top35 mismatch")
    return result


def _dense(path: Path, questions: Mapping[str, Mapping[str, Any]], k_star: int) -> dict[str, dict[str, Mapping[str, Any]]]:
    result: dict[str, dict[str, Mapping[str, Any]]] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["qid"])
            if qid not in questions:
                continue
            result[qid] = {str(item["doc_id"]): item for item in row["candidates"][:k_star]}
    if set(result) != set(questions):
        raise ValueError("dense pool coverage mismatch")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--stage1-audit", type=Path, required=True)
    parser.add_argument("--dense-pools", type=Path, required=True)
    parser.add_argument("--a0-audit", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    questions = load_metadata(args.metadata)
    stage1 = _stage1(args.stage1_audit, questions)
    audit = read_json(args.a0_audit)
    k_star = int(audit["k_selection"]["selected_k_star"])
    dense = _dense(args.dense_pools, questions, k_star)
    expected: set[tuple[str, str]] = {
        (qid, str(item["doc_id"]))
        for qid in questions
        for item in dense[qid].values()
        if str(item["doc_id"]) not in set(stage1[qid][:80])
    }
    rows = []
    seen: set[tuple[str, str]] = set()
    with args.scores.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            qid, doc_id = str(raw["qid"]), str(raw["doc_id"])
            key = (qid, doc_id)
            if key not in expected:
                raise ValueError(f"score line {line_number}: tuple outside frozen dense-new boundary {key}")
            if key in seen:
                raise ValueError(f"duplicate score tuple {key}")
            score = float(raw["stage2_score"])
            if not math.isfinite(score):
                raise ValueError(f"non-finite Stage2 score {key}")
            dense_item = dense[qid][doc_id]
            old_rank = next((rank for rank, candidate in enumerate(stage1[qid], 1) if candidate == doc_id), None)
            rows.append({
                "qid": qid,
                "docid": doc_id,
                "dense_rank": int(dense_item["rank"]),
                "dense_score": float(dense_item["score"]),
                "stage2_score": score,
                "old_rank": old_rank,
                "is_new_dense_candidate": True,
                "fold": _fold(qid, questions[qid]),
            })
            seen.add(key)
    if seen != expected:
        raise ValueError(f"frozen score coverage mismatch: got={len(seen)} expected={len(expected)}")
    rows.sort(key=lambda row: (_numeric_key(str(row["qid"])), int(row["dense_rank"]), _numeric_key(str(row["docid"]))))
    output_hash = write_jsonl(args.output, rows)
    report = {
        "kind": "legalir_v29_a1d_frozen_score_cache",
        "schema_version": 1,
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "selection_performed": False,
        "frozen": True,
        "row_count": len(rows),
        "query_count": len({row["qid"] for row in rows}),
        "k_star": k_star,
        "fields": ["qid", "docid", "dense_rank", "dense_score", "stage2_score", "old_rank", "is_new_dense_candidate", "fold"],
        "source_scores": str(args.scores.resolve()),
        "source_scores_sha256": sha256_file(args.scores),
        "metadata": str(args.metadata.resolve()),
        "metadata_sha256": sha256_file(args.metadata),
        "stage1_audit_sha256": sha256_file(args.stage1_audit),
        "dense_pools_sha256": sha256_file(args.dense_pools),
        "output": str(args.output.resolve()),
        "output_sha256": output_hash,
    }
    write_json(args.output.with_name(args.output.stem + "_report.json"), report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
