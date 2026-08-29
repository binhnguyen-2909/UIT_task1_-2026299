"""Sweep Stage-2 RRF weights from a fixed, already-scored audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    from .evaluate import official_metrics
    from .reranker import reciprocal_rank_fuse
except ImportError:
    from evaluate import official_metrics
    from reranker import reciprocal_rank_fuse


def parse_grid(value: str, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-audit", type=Path, required=True)
    parser.add_argument("--scores-audit", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-limit", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--weights", default="0.25,0.5,0.75,1,1.25,1.5,2,3")
    parser.add_argument("--rrf-ks", default="0,5,10,20,40")
    args = parser.parse_args()

    stage1 = json.loads(args.stage1_audit.read_text(encoding="utf-8"))
    scores_audit = json.loads(args.scores_audit.read_text(encoding="utf-8"))
    references = json.loads(args.reference.read_text(encoding="utf-8"))
    queries = stage1["queries"]
    candidate_rows = stage1["stage1_candidates"]
    score_by_qid = {str(row["question_id"]): row for row in scores_audit["queries"]}
    weights = parse_grid(args.weights, float)
    rrf_ks = parse_grid(args.rrf_ks, int)
    results: List[Dict[str, Any]] = []

    for weight in weights:
        for rrf_k in rrf_ks:
            submission: Dict[str, Dict[str, List[str]]] = {}
            for query_index, query_item in enumerate(queries):
                qid = str(query_item["query_id"])
                docs = list(dict.fromkeys(str(value) for value in candidate_rows[query_index]))[
                    : args.candidate_limit
                ]
                lexical = [{"doc_id": doc_id, "score": 1.0} for doc_id in docs]
                score_map = {
                    str(doc_id): float(value)
                    for doc_id, value in score_by_qid[qid]["neural_scores"].items()
                }
                neural = [
                    {"doc_id": doc_id, "rerank_score": score_map[doc_id]}
                    for doc_id in docs
                ]
                neural.sort(
                    key=lambda item: (-float(item["rerank_score"]), str(item["doc_id"]))
                )
                fused = reciprocal_rank_fuse(
                    lexical, neural, neural_weight=weight, rrf_k=rrf_k, top_k=args.top_k
                )
                submission[qid] = {
                    "answer": [str(item["doc_id"]) for item in fused]
                }
            metrics = official_metrics(
                submission,
                {
                    qid: {"answer": [str(value) for value in references[qid]["answer"]]}
                    for qid in submission
                },
            )
            results.append({
                "neural_weight": weight,
                "rrf_k": rrf_k,
                **metrics,
            })

    results.sort(key=lambda row: (-row["recall"], -row["precision"], row["rrf_k"]))
    payload = {
        "kind": "legalir_stage2_fusion_sweep",
        "stage1_audit": str(args.stage1_audit.resolve()),
        "scores_audit": str(args.scores_audit.resolve()),
        "candidate_limit": args.candidate_limit,
        "results": results,
        "best": results[0] if results else None,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": payload["best"], "top10": results[:10]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
