"""V34-A0 zero-shot Qwen3-Reranker hard-boundary benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import CrossEncoder

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (REPO_ROOT / "scripts", SCRIPT_DIR / "v27"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from batch_stage2_rerank import hydrate, load_passage_store, open_readonly  # noqa: E402
from v27_common import load_metadata, sha256_file, write_json, write_jsonl  # noqa: E402


VIEWS = ("Q0", "Q1", "Q2")
INSTRUCTION = (
    "Given a Vietnamese legal question, determine whether the candidate legal provision directly provides the legal basis needed to answer the question. "
    "Distinguish carefully: legal subject; regulated act; condition; exception; negation; authority; numerical threshold; time limit; legal consequence. "
    "A document that is only topically similar but does not directly support the question should be considered irrelevant."
)


def _num(value: str) -> tuple[int, str]:
    return (int(value), value) if str(value).isdigit() else (10**18, str(value))


def _load_views(path: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    result = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                result[str(row["qid"])] = {view: row[view] for view in VIEWS}
    return result


def _build_benchmark(questions: dict[str, dict[str, Any]], views: dict[str, dict[str, list[dict[str, Any]]]], connection: Any) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    docs_needed: set[str] = set()
    plans = []
    for qid in sorted(questions, key=_num):
        question = questions[qid]
        refs = set(question["references"])
        anchor = set(str(x) for x in question["anchor"])
        rank_by_doc: dict[str, int] = {}
        for view in VIEWS:
            for item in views[qid][view][:500]:
                doc = str(item["doc_id"])
                rank_by_doc[doc] = min(rank_by_doc.get(doc, 10**9), int(item["rank"]))
        golds = sorted((doc for doc in refs - anchor if doc in rank_by_doc), key=lambda doc: (rank_by_doc[doc], _num(doc)))
        if not golds:
            continue
        gold = golds[0]
        false_docs = [doc for doc in rank_by_doc if doc not in refs]
        # The strongest false documents are the closest dense candidates.  The
        # incumbent is explicitly included when it is not gold.
        false_docs.sort(key=lambda doc: (0 if doc == str(question["anchor"][4]) else 1, rank_by_doc[doc], _num(doc)))
        false_docs = false_docs[:5]
        if len(false_docs) < 5:
            continue
        for hard_index, hard in enumerate(false_docs, 1):
            plans.append((qid, gold, hard, rank_by_doc[gold], rank_by_doc[hard], hard_index))
            docs_needed.update((gold, hard))
    store = load_passage_store(connection, sorted(docs_needed, key=_num))
    for qid, gold, hard, gold_rank, hard_rank, hard_index in plans:
        query = str(questions[qid]["query"])
        gold_text = hydrate(query, [gold], 1, store)[0]["text"]
        hard_text = hydrate(query, [hard], 1, store)[0]["text"]
        if not gold_text.strip() or not hard_text.strip():
            continue
        selected.append({"pair_id": len(selected), "qid": qid, "query": query, "gold_doc": gold, "hard_doc": hard, "gold_rank_best_view": gold_rank, "hard_rank_best_view": hard_rank, "hard_source_rank": hard_index, "gold_text": gold_text, "hard_text": hard_text})
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    questions = load_metadata(args.metadata)
    views = _load_views(args.views)
    connection = open_readonly(args.index)
    try:
        benchmark = _build_benchmark(questions, views, connection)
    finally:
        connection.close()
    write_jsonl(args.output_dir / "hard_boundary_benchmark.jsonl", benchmark)
    if not benchmark:
        raise RuntimeError("empty hard-boundary benchmark")
    device = args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    model = CrossEncoder(str(args.model.resolve()), device=device, prompts={"legal": INSTRUCTION}, default_prompt_name="legal", local_files_only=True, max_length=args.max_length, model_kwargs={"torch_dtype": torch.float16} if device.startswith("cuda") else {})
    pairs = [(row["query"], row["gold_text"]) for row in benchmark] + [(row["query"], row["hard_text"]) for row in benchmark]
    scores = np.asarray(model.predict(pairs, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True), dtype=np.float64)
    gold_scores = scores[: len(benchmark)]
    hard_scores = scores[len(benchmark) :]
    output_rows = []
    for row, gold_score, hard_score in zip(benchmark, gold_scores, hard_scores, strict=True):
        output_rows.append({**row, "gold_score": float(gold_score), "hard_score": float(hard_score), "gold_wins": bool(gold_score > hard_score), "tie": bool(gold_score == hard_score), "margin": float(gold_score - hard_score)})
    write_jsonl(args.output_dir / "hard_boundary_scores.jsonl", output_rows)
    wins = np.asarray([row["gold_wins"] for row in output_rows], dtype=bool)
    ties = np.asarray([row["tie"] for row in output_rows], dtype=bool)
    by_source = defaultdict(list)
    for row in output_rows:
        by_source[int(row["hard_source_rank"])].append(row)
    source_report = {str(source): {"pairs": len(rows), "wins": sum(row["gold_wins"] for row in rows), "win_rate": sum(row["gold_wins"] for row in rows) / len(rows)} for source, rows in sorted(by_source.items())}
    report = {"kind": "legalir_v34_a0_qwen3_reranker_hard_boundary", "schema_version": 1, "model": str(args.model.resolve()), "instruction": INSTRUCTION, "device": device, "max_length": args.max_length, "batch_size": args.batch_size, "pairs": len(output_rows), "queries": len({row["qid"] for row in output_rows}), "gold_win_rate": float(wins.mean()), "ties": int(ties.sum()), "consistent_pairwise_coverage": 1.0, "margin_median": float(np.median([row["margin"] for row in output_rows])), "margin_mean": float(np.mean([row["margin"] for row in output_rows])), "by_hard_source_rank": source_report, "gate": {"win_ge_0_68": float(wins.mean()) >= 0.68, "win_ge_0_75": float(wins.mean()) >= 0.75, "win_le_0_60": float(wins.mean()) <= 0.60, "pass_pointwise": float(wins.mean()) >= 0.68}, "inputs": {"metadata": str(args.metadata.resolve()), "metadata_sha256": sha256_file(args.metadata), "views": str(args.views.resolve()), "views_sha256": sha256_file(args.views), "index": str(args.index.resolve())}, "outputs": {"benchmark": str((args.output_dir / "hard_boundary_benchmark.jsonl").resolve()), "scores": str((args.output_dir / "hard_boundary_scores.jsonl").resolve())}, "elapsed_seconds": time.perf_counter() - started}
    write_json(args.output_dir / "hard_boundary_report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
