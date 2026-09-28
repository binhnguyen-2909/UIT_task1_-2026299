"""V32-C view-local Stage2 probe after deterministic query-view gating."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (REPO_ROOT / "scripts", SCRIPT_DIR / "v27", SCRIPT_DIR / "v29"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from batch_stage2_rerank import hydrate, load_passage_store, open_readonly  # noqa: E402
from v27_common import load_metadata, load_sidecar, sha256_file, write_json, write_jsonl  # noqa: E402
from v29_a1d2_full_oof import _score_qid  # noqa: E402
from v29_a1d1_stage2_diag import DirectStage2  # noqa: E402


def _num(value: str) -> tuple[int, str]:
    return (int(value), value) if str(value).isdigit() else (10**18, str(value))


def _load_views(path: Path, qids: set[str]) -> dict[str, dict[str, Any]]:
    result = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["qid"])
            if qid in qids:
                result[qid] = row
    if set(result) != qids:
        raise ValueError(f"view candidate coverage mismatch: {len(result)} != {len(qids)}")
    return result


def _load_stage2_rows(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    result = {}
    if not path.exists():
        return result
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                result[(str(row["qid"]), str(row["view"]), str(row["doc_id"]))] = row
    return result


def _pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Threshold diagnostic; report every distinct margin, no model tuning."""
    margins = sorted({float(row["margin"]) for row in rows if row["stage2_score"] is not None}, reverse=True)
    points = []
    for tau in margins:
        selected = [row for row in rows if row["stage2_score"] is not None and float(row["margin"]) >= tau]
        if not selected:
            continue
        relevant = sum(bool(row["is_gold"]) for row in selected)
        destructive = len(selected) - relevant
        points.append({"tau": tau, "selected": len(selected), "rescue": relevant, "destructive": destructive, "precision": relevant / len(selected)})
    return points


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--probe-fold", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    all_questions = load_metadata(args.metadata)
    questions = {qid: row for qid, row in all_questions.items() if args.probe_fold < 0 or int(qid) % 5 == args.probe_fold}
    qids = set(questions)
    views = _load_views(args.views, qids)
    sidecar = load_sidecar(args.sidecar, all_questions)
    scorer = DirectStage2(args.model, args.device, args.max_length, bool(args.fp16))
    connection = open_readonly(args.index)
    score_rows: list[dict[str, Any]] = []
    view_winners: list[dict[str, Any]] = []
    try:
        for qid in sorted(questions, key=_num):
            question = questions[qid]
            old_top5 = {str(doc) for doc in question["anchor"]}
            old_slot5 = str(question["anchor"][4])
            incumbent = float(sidecar[(qid, old_slot5)])
            union_docs: list[str] = []
            docs_by_view: dict[str, list[str]] = {}
            for view in ("Q1", "Q2"):
                docs = []
                for item in views[qid][view][: args.top_k]:
                    doc_id = str(item["doc_id"])
                    if doc_id not in old_top5 and doc_id not in docs:
                        docs.append(doc_id)
                docs_by_view[view] = docs
                for doc_id in docs:
                    if doc_id not in union_docs:
                        union_docs.append(doc_id)
            store = load_passage_store(connection, union_docs)
            for view in ("Q1", "Q2"):
                docs = docs_by_view[view]
                hydrated = {str(item["doc_id"]): item for item in hydrate(str(views[qid][view]), docs, 3, store)}
                aggregate, passage_scores, pair_count = _score_qid(scorer, str(views[qid][view]), docs, store, args.batch_size)
                refs = set(question["references"])
                ranked = sorted(aggregate, key=lambda doc: (-float(aggregate[doc]), docs.index(doc), _num(doc)))
                winner = ranked[0] if ranked else None
                view_winners.append({"qid": qid, "view": view, "winner_doc": winner, "winner_stage2": aggregate.get(winner), "winner_is_gold": bool(winner in refs if winner else False), "old_slot5_stage2": incumbent, "margin": float(aggregate[winner] - incumbent) if winner else None, "candidate_count": len(docs), "query_passage_pairs": pair_count})
                for doc_id in docs:
                    score = aggregate.get(doc_id)
                    score_rows.append({"qid": qid, "view": view, "doc_id": doc_id, "view_rank": docs.index(doc_id) + 1, "stage2_score": score, "old_slot5_stage2": incumbent, "margin": float(score - incumbent) if score is not None else None, "is_gold": doc_id in refs, "passage_scores": passage_scores.get(doc_id, []), "hydrated": bool(hydrated.get(doc_id, {}).get("passages"))})
            if len(view_winners) % 20 == 0:
                print(json.dumps({"processed_views": len(view_winners), "target_views": len(questions) * 2, "elapsed_seconds": time.perf_counter() - started}), flush=True)
    finally:
        connection.close()
    write_jsonl(args.output_dir / "stage2_view_scores.jsonl", score_rows)
    write_jsonl(args.output_dir / "stage2_view_winners.jsonl", view_winners)

    refs_by_qid = {qid: set(row["references"]) for qid, row in questions.items()}
    rows_by_qid: dict[str, list[dict[str, Any]]] = {}
    for row in score_rows:
        rows_by_qid.setdefault(str(row["qid"]), []).append(row)
    winner_gold = sum(bool(row["winner_is_gold"]) for row in view_winners)
    query_any_winner_gold = sum(any(bool(row["winner_is_gold"]) for row in view_winners if row["qid"] == qid) for qid in questions)
    recovered_qids = 0
    any_gold_top20 = 0
    best_gold_rank: list[int] = []
    for qid in questions:
        rows = rows_by_qid.get(qid, [])
        gold_rows = [row for row in rows if row["is_gold"]]
        if gold_rows:
            recovered_qids += 1
            best_gold_rank.append(min(int(row["view_rank"]) for row in gold_rows))
        any_gold_top20 += int(bool(gold_rows))
    winner_candidates = []
    for qid in questions:
        by_view = [row for row in view_winners if row["qid"] == qid and row["winner_doc"] is not None]
        # One action candidate per view; use the strongest local winner as a
        # diagnostic only.  No promoted answer is written in this probe.
        if by_view:
            winner_candidates.append(max(by_view, key=lambda row: (float(row["margin"]), row["view"])))
    positive = [row for row in winner_candidates if row["margin"] is not None and float(row["margin"]) > 0]
    summary = {
        "kind": "legalir_v32_c_view_local_stage2_probe",
        "schema_version": 1,
        "queries": len(questions),
        "views": ["Q1", "Q2"],
        "top_k": args.top_k,
        "probe_fold": args.probe_fold,
        "stage2": {"view_winners": len(view_winners), "winner_gold": winner_gold, "winner_gold_rate_per_view": winner_gold / max(len(view_winners), 1), "queries_with_any_gold_view_winner": query_any_winner_gold, "query_any_gold_view_winner_rate": query_any_winner_gold / max(len(questions), 1), "positive_margin_query_candidates": len(positive), "positive_margin_gold": sum(bool(row["winner_is_gold"]) for row in positive), "positive_margin_precision": sum(bool(row["winner_is_gold"]) for row in positive) / max(len(positive), 1), "recovered_queries_top_k": recovered_qids, "best_gold_view_rank_median": float(np.median(best_gold_rank)) if best_gold_rank else None},
        "threshold_pareto": _pareto(winner_candidates),
        "threshold_pareto_all_view_winners": _pareto(view_winners),
        "gate": {"winner_gold_rate_ge_0_30": winner_gold / max(len(view_winners), 1) >= 0.30, "query_winner_gold_rate_ge_0_30": query_any_winner_gold / max(len(questions), 1) >= 0.30, "positive_margin_pocket_precision_ge_0_90_and_rescue": any(float(row["precision"]) >= 0.90 and int(row["rescue"]) > 0 and int(row["destructive"]) <= 2 for row in _pareto(winner_candidates))},
        "inputs": {"metadata": str(args.metadata.resolve()), "metadata_sha256": sha256_file(args.metadata), "views": str(args.views.resolve()), "sidecar": str(args.sidecar.resolve()), "index": str(args.index.resolve()), "model": str(args.model.resolve()), "device": args.device, "batch_size": args.batch_size, "max_length": args.max_length},
        "outputs": {"scores": str((args.output_dir / "stage2_view_scores.jsonl").resolve()), "winners": str((args.output_dir / "stage2_view_winners.jsonl").resolve())},
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(args.output_dir / "stage2_view_report.json", summary)
    print(json.dumps({"report": str((args.output_dir / "stage2_view_report.json").resolve()), "gate": summary["gate"], "stage2": summary["stage2"], "elapsed_seconds": summary["elapsed_seconds"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
