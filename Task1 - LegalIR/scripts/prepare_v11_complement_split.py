"""Build a complementary clean retrieval index without re-embedding corpus chunks.

The source v11 index was trained on 6,000 questions and evaluated on a clean
1,000-question holdout.  This utility copies only the SQLite corpus index,
replaces its small supervised layer with those 1,000 holdout questions, and
selects deterministic questions from the other 6,000.  Queries selected for
the complementary audit are therefore absent from exact memory, aliases,
question examples, and document priors used by their retrieval index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

try:
    from .sqlite_retriever import SQLiteFTSRetriever
    from .data_quality import normalize_question
except ImportError:
    from sqlite_retriever import SQLiteFTSRetriever
    from data_quality import normalize_question


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def link_or_copy(source: Path, target: Path) -> str:
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--holdout-audit", type=Path, required=True)
    parser.add_argument("--query-count", type=int, default=1200)
    parser.add_argument("--seed", default="v11-complement-20260825")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.query_count <= 0:
        raise ValueError("query-count must be positive")
    source_db = args.source_index / "legalir.sqlite3"
    source_config = args.source_index / "legalir.config.json"
    if not source_db.is_file() or not source_config.is_file():
        raise FileNotFoundError("source index is incomplete")
    if args.output_index.exists() and any(args.output_index.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty index: {args.output_index}")
    args.output_index.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    train = read_json(args.train_json)
    audit = read_json(args.holdout_audit)
    holdout_ids = [str(row["query_id"]) for row in audit["queries"]]
    if len(holdout_ids) != len(set(holdout_ids)):
        raise ValueError("holdout audit has duplicate IDs")
    complement_training = {qid: train[qid] for qid in holdout_ids}
    training_question_keys = {
        normalize_question(item["question"])
        for item in complement_training.values()
        if isinstance(item, dict) and isinstance(item.get("question"), str)
    }
    remaining = [
        str(qid) for qid, item in train.items()
        if str(qid) not in set(holdout_ids)
        and isinstance(item, dict)
        and isinstance(item.get("question"), str)
        and normalize_question(item["question"]) not in training_question_keys
    ]
    remaining.sort(
        key=lambda qid: hashlib.sha256(f"{args.seed}:{qid}".encode("utf-8")).digest()
    )
    selected_ids = remaining[:args.query_count]
    if len(selected_ids) != args.query_count:
        raise ValueError("not enough non-holdout questions")
    complement_queries = {qid: train[qid] for qid in selected_ids}
    training_path = args.artifact_dir / "complement_index_training_1000.json"
    queries_path = args.artifact_dir / f"complement_queries_{args.query_count}.json"
    write_json(training_path, complement_training)
    write_json(queries_path, complement_queries)
    shutil.copy2(source_db, args.output_index / source_db.name)
    shutil.copy2(source_config, args.output_index / source_config.name)
    cache_modes: dict[str, str] = {}
    for source in sorted(args.source_index.glob("dense_embeddings*")):
        if source.is_file():
            cache_modes[source.name] = link_or_copy(source, args.output_index / source.name)
    retriever = SQLiteFTSRetriever.load(
        args.output_index / "legalir.sqlite3", device="cpu", disable_reranker=True
    )
    signal_report = retriever.replace_training_signals(complement_training)
    retriever.close()
    report = {
        "kind": "legalir_v11_complement_clean_split",
        "source_index": str(args.source_index.resolve()),
        "output_index": str(args.output_index.resolve()),
        "source_training_questions": len(train) - len(holdout_ids),
        "complement_index_training_questions": len(complement_training),
        "complement_query_questions": len(complement_queries),
        "training_query_overlap": len(set(complement_training) & set(complement_queries)),
        "normalized_question_overlap": len(
            training_question_keys
            & {normalize_question(item["question"]) for item in complement_queries.values()}
        ),
        "signal_report": signal_report,
        "dense_cache_modes": cache_modes,
        "training_artifact": str(training_path.resolve()),
        "query_artifact": str(queries_path.resolve()),
        "selection": "SHA256(seed:question_id), label-blind",
        "seed": args.seed,
    }
    report_path = args.artifact_dir / "complement_split_report.json"
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
