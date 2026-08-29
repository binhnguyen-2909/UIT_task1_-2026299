"""Repair Stage-2 RRF using already-computed CrossEncoder scores.

This is intentionally a post-processing step: it never loads a model and never
changes the Stage-1 candidate set.  It is useful when an expensive scoring run
completed successfully but its fusion/audit writer had a ranking bug.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

try:
    from .evaluate import official_metrics
    from .main import _write_bundle
    from .reranker import reciprocal_rank_fuse
except ImportError:
    from evaluate import official_metrics
    from main import _write_bundle
    from reranker import reciprocal_rank_fuse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-audit", type=Path, required=True)
    parser.add_argument("--scores-audit", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-limit", type=int, default=20)
    parser.add_argument("--top-k", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--neural-weight", type=float, default=1.0)
    parser.add_argument("--rrf-k", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.candidate_limit <= 0:
        raise ValueError("candidate-limit must be positive")
    started = time.perf_counter()
    stage1 = json.loads(args.stage1_audit.read_text(encoding="utf-8"))
    scores_audit = json.loads(args.scores_audit.read_text(encoding="utf-8"))
    queries = stage1.get("queries")
    candidate_rows = stage1.get("stage1_candidates")
    score_rows = scores_audit.get("queries")
    if not isinstance(queries, list) or not isinstance(candidate_rows, list):
        raise ValueError("Stage-1 audit must contain queries and stage1_candidates")
    if not isinstance(score_rows, list):
        raise ValueError("Scores audit must contain queries")
    score_by_qid = {str(row["question_id"]): row for row in score_rows}
    if len(score_by_qid) != len(score_rows):
        raise ValueError("Scores audit has duplicate question IDs")

    submission: Dict[str, Dict[str, List[str]]] = {}
    traces: List[Dict[str, Any]] = []
    for query_index, query_item in enumerate(queries):
        qid = str(query_item["query_id"])
        score_row = score_by_qid.get(qid)
        if score_row is None:
            raise ValueError(f"Scores audit does not cover question {qid}")
        docs = list(dict.fromkeys(str(value) for value in candidate_rows[query_index]))[
            : args.candidate_limit
        ]
        lexical = [
            {"doc_id": doc_id, "score": 1.0, "rank": rank}
            for rank, doc_id in enumerate(docs, 1)
        ]
        score_map = {
            str(doc_id): float(value)
            for doc_id, value in score_row.get("neural_scores", {}).items()
        }
        missing = [doc_id for doc_id in docs if doc_id not in score_map]
        if missing:
            raise ValueError(f"Scores audit missing {len(missing)} candidates for {qid}")
        neural = [
            {
                "doc_id": doc_id,
                "score": score_map[doc_id],
                "rerank_score": score_map[doc_id],
            }
            for doc_id in docs
        ]
        neural.sort(key=lambda item: (-float(item["rerank_score"]), str(item["doc_id"])))
        fused = reciprocal_rank_fuse(
            lexical,
            neural,
            neural_weight=args.neural_weight,
            rrf_k=args.rrf_k,
            top_k=args.top_k,
        )
        answers = [str(item["doc_id"]) for item in fused]
        submission[qid] = {"answer": answers}
        traces.append({
            "question_id": qid,
            "candidate_count": len(docs),
            "answers": answers,
            "neural_scores": score_map,
            "neural_order": [str(item["doc_id"]) for item in neural],
        })

    _write_bundle(submission, args.output)
    audit: Dict[str, Any] = {
        "kind": "legalir_stage2_fusion_repair",
        "stage1_audit": str(args.stage1_audit.resolve()),
        "scores_audit": str(args.scores_audit.resolve()),
        "candidate_limit": args.candidate_limit,
        "neural_weight": args.neural_weight,
        "rrf_k": args.rrf_k,
        "top_k": args.top_k,
        "questions": len(queries),
        "output": str(args.output.resolve()),
        "queries": traces,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if args.reference:
        references = json.loads(args.reference.read_text(encoding="utf-8"))
        selected = {
            qid: {"answer": [str(value) for value in references[qid]["answer"]]}
            for qid in submission
        }
        audit["metrics"] = official_metrics(submission, selected)
    args.output.with_suffix(".audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "questions": len(queries),
        "metrics": audit.get("metrics"),
        "elapsed_seconds": audit["elapsed_seconds"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
