"""V29-A1D-2 full clean-complement OOF challenger pass.

The online policy locks the existing answer's top four and compares the best
new dense challenger directly with the frozen old slot5 Stage2 score.  Thus
the full pass needs only new-document Stage2 scores; it does not need the
missing historical c36--c80 scores to make a promotion/abstention decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for path in (REPO_ROOT / "scripts", SCRIPT_DIR.parent / "v27"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from batch_stage2_rerank import hydrate, load_passage_store, open_readonly  # noqa: E402
from v27_common import load_metadata, load_sidecar, read_json, sha256_file, write_json  # noqa: E402
from v29_a1d1_stage2_diag import DirectStage2  # noqa: E402


def _numeric_key(value: str) -> tuple[int, str]:
    return (int(value), value) if str(value).isdigit() else (10**18, str(value))


def _stage1(path: Path, questions: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    payload = read_json(path)
    result = {str(row["query_id"]): [str(doc) for doc in docs] for row, docs in zip(payload["queries"], payload["stage1_candidates"], strict=True)}
    if set(questions) - set(result):
        raise ValueError("Stage1 audit misses current metadata qids")
    for qid, question in questions.items():
        if result[qid][:35] != list(question["c35"]):
            raise ValueError(f"{qid}: Stage1 top35 does not match frozen c35")
    return result


def _dense(path: Path, questions: Mapping[str, Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["qid"])
            if qid not in questions:
                continue
            candidates = row["candidates"]
            if len(candidates) < 500:
                raise ValueError(f"{qid}: dense pool has fewer than 500 rows")
            normalized = []
            seen: set[str] = set()
            for expected_rank, item in enumerate(candidates, 1):
                doc_id = str(item["doc_id"])
                if doc_id in seen or int(item["rank"]) != expected_rank:
                    raise ValueError(f"{qid}: malformed/duplicate dense rank")
                seen.add(doc_id)
                normalized.append({"doc_id": doc_id, "rank": int(item["rank"]), "score": float(item["score"])})
            result[qid] = normalized
    if set(result) != set(questions):
        raise ValueError("dense pool coverage mismatch")
    return result


def _score_qid(scorer: DirectStage2, query: str, docs: list[str], store: Mapping[str, Any], batch_size: int) -> tuple[dict[str, float], dict[str, list[float]], int]:
    hydrated = {str(item["doc_id"]): item for item in hydrate(query, docs, 3, store)}
    pairs: list[tuple[str, str]] = []
    owners: list[tuple[str, int]] = []
    for doc_id in docs:
        passages = [str(value) for value in hydrated.get(doc_id, {}).get("passages", []) if str(value).strip()]
        for passage_index, passage in enumerate(passages):
            pairs.append((query, passage))
            owners.append((doc_id, passage_index))
    # Grouping by approximate pair length reduces dynamic-padding waste while
    # leaving the semantic score independent of batch order.
    order = sorted(range(len(pairs)), key=lambda index: (len(pairs[index][0]) + len(pairs[index][1]), owners[index]))
    ordered_pairs = [pairs[index] for index in order]
    ordered_scores = scorer.score(ordered_pairs, batch_size)
    scores: dict[str, list[float]] = {}
    for index, value in zip(order, ordered_scores, strict=True):
        doc_id, _ = owners[index]
        scores.setdefault(doc_id, []).append(float(value))
    aggregate = {doc_id: max(values) for doc_id, values in scores.items() if values}
    return aggregate, scores, len(pairs)


def _append_json_line(stream: Any, row: Mapping[str, Any]) -> None:
    stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _reconcile_score_cache(path: Path, done: set[str]) -> None:
    """Drop score rows for qids without a committed prediction checkpoint."""
    if not path.exists() or not done:
        return
    temporary = path.with_name(path.name + ".reconcile")
    try:
        with path.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as target:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                if str(row["qid"]) in done:
                    target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _bootstrap_lb(gains: np.ndarray, seed: int, repeats: int = 10000) -> float:
    rng = np.random.default_rng(seed)
    batch = 500
    estimates: list[np.ndarray] = []
    for start in range(0, repeats, batch):
        size = min(batch, repeats - start)
        indices = rng.integers(0, len(gains), size=(size, len(gains)))
        estimates.append(gains[indices].mean(axis=1))
    return float(np.quantile(np.concatenate(estimates), 0.025))


def _load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                done.add(str(json.loads(line)["qid"]))
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--stage1-audit", type=Path, required=True)
    parser.add_argument("--dense-pools", type=Path, required=True)
    parser.add_argument("--a0-audit", type=Path, required=True)
    parser.add_argument("--stage2-sidecar", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    questions = load_metadata(args.metadata)
    stage1 = _stage1(args.stage1_audit, questions)
    dense = _dense(args.dense_pools, questions)
    sidecar = load_sidecar(args.stage2_sidecar, questions)
    audit = read_json(args.a0_audit)
    k_star = int(audit["k_selection"]["selected_k_star"])
    if not 1 <= k_star <= 1000:
        raise ValueError("invalid selected K*")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    score_path = out / "a1d2_full_dense_new_stage2_scores.jsonl"
    prediction_path = out / "a1d2_full_oof_predictions.jsonl"
    if not args.resume and (score_path.exists() or prediction_path.exists()):
        raise FileExistsError("existing A1D-2 output; use --resume explicitly")
    done = _load_done(prediction_path) if args.resume else set()
    if args.resume:
        _reconcile_score_cache(score_path, done)
    mode = "a" if args.resume else "w"
    device = args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    scorer = DirectStage2(args.model, device, args.max_length, bool(args.fp16 and device.startswith("cuda")))
    connection = open_readonly(args.index)
    total_docs = 0
    total_pairs = 0
    empty_docs = 0
    processed = 0
    try:
        with score_path.open(mode, encoding="utf-8") as score_stream, prediction_path.open(mode, encoding="utf-8") as prediction_stream:
            for qid, question in questions.items():
                if qid in done:
                    continue
                existing = set(stage1[qid][:80])
                dense_new = [item for item in dense[qid][:k_star] if str(item["doc_id"]) not in existing]
                docs = [str(item["doc_id"]) for item in dense_new]
                store = load_passage_store(connection, docs)
                aggregate, passage_scores, pair_count = _score_qid(scorer, str(question["query"]), docs, store, args.batch_size)
                total_docs += len(docs)
                total_pairs += pair_count
                empty_docs += len(docs) - len(aggregate)
                dense_by_doc = {str(item["doc_id"]): item for item in dense_new}
                for doc_id in docs:
                    _append_json_line(score_stream, {
                        "qid": qid,
                        "doc_id": doc_id,
                        "dense_rank": int(dense_by_doc[doc_id]["rank"]),
                        "dense_score": float(dense_by_doc[doc_id]["score"]),
                        "stage2_score": aggregate.get(doc_id),
                        "passage_scores": passage_scores.get(doc_id, []),
                    })
                old_slot5 = str(question["anchor"][4])
                old_slot5_score = sidecar[(qid, old_slot5)]
                challengers = sorted(aggregate, key=lambda doc_id: (-aggregate[doc_id], docs.index(doc_id), _numeric_key(doc_id)))
                challenger = challengers[0] if challengers else None
                challenger_score = aggregate.get(challenger) if challenger is not None else None
                promoted = challenger is not None and float(challenger_score) > float(old_slot5_score)
                answer = list(question["anchor"][:4]) + ([challenger] if promoted else [old_slot5])
                references = set(question["references"])
                _append_json_line(prediction_stream, {
                    "qid": qid,
                    "answer": answer,
                    "old_answer": list(question["anchor"]),
                    "top1_4_byte_identical": answer[:4] == list(question["anchor"][:4]),
                    "old_slot5_doc": old_slot5,
                    "old_slot5_stage2": float(old_slot5_score),
                    "challenger_doc": challenger,
                    "challenger_stage2": challenger_score,
                    "challenger_dense_rank": dense_by_doc.get(challenger, {}).get("rank") if challenger else None,
                    "promoted": bool(promoted),
                    "promotion_is_relevant": bool(promoted and challenger in references),
                    "recall_old": len(set(question["anchor"]) & references) / max(len(references), 1),
                    "recall_new": len(set(answer) & references) / max(len(references), 1),
                    "new_recall_gain": len(set(answer) & references) / max(len(references), 1) - len(set(question["anchor"]) & references) / max(len(references), 1),
                    "dense_new_docs": len(docs),
                    "query_passage_pairs": pair_count,
                })
                processed += 1
                score_stream.flush()
                prediction_stream.flush()
                if processed % 10 == 0 or processed == len(questions) - len(done):
                    print(json.dumps({"processed_qids": processed, "total_qids": len(questions), "last_qid": qid, "dense_docs": len(docs), "pairs": pair_count, "elapsed_seconds": time.perf_counter() - started}), flush=True)
    finally:
        connection.close()
    rows = []
    with prediction_path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != len(questions):
        raise RuntimeError(f"A1D-2 incomplete: predictions={len(rows)} questions={len(questions)}")
    rows.sort(key=lambda row: _numeric_key(str(row["qid"])))
    old_recall = np.asarray([float(row["recall_old"]) for row in rows], dtype=np.float64)
    new_recall = np.asarray([float(row["recall_new"]) for row in rows], dtype=np.float64)
    gains = new_recall - old_recall
    promotions = [row for row in rows if row["promoted"]]
    relevant_promotions = [row for row in promotions if row["promotion_is_relevant"]]
    fold_gains = []
    for fold in range(5):
        fold_rows = [row for row in rows if int(row["qid"]) % 5 == fold]
        fold_gains.append(float(np.mean([float(row["new_recall_gain"]) for row in fold_rows])))
    report = {
        "kind": "legalir_v29_a1d2_full_oof",
        "schema_version": 1,
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "selection_performed": False,
        "inputs": {
            "metadata": str(args.metadata.resolve()),
            "metadata_sha256": sha256_file(args.metadata),
            "stage1_audit": str(args.stage1_audit.resolve()),
            "stage1_audit_sha256": sha256_file(args.stage1_audit),
            "dense_pools": str(args.dense_pools.resolve()),
            "dense_pools_sha256": sha256_file(args.dense_pools),
            "a0_audit": str(args.a0_audit.resolve()),
            "a0_audit_sha256": sha256_file(args.a0_audit),
            "stage2_sidecar": str(args.stage2_sidecar.resolve()),
            "stage2_sidecar_sha256": sha256_file(args.stage2_sidecar),
            "index": str(args.index.resolve()),
            "index_sha256": sha256_file(args.index),
            "model": str(args.model.resolve()),
            "model_config_sha256": sha256_file(args.model / "config.json"),
        },
        "contract": {
            "k_star": k_star,
            "queries": len(questions),
            "candidate_union": "existing c80 union dense top-K*",
            "scored_documents": "dense top-K* minus existing c80",
            "frozen_top1_4": True,
            "challenger_rule": "best new dense Stage2 score strictly above frozen old slot5; otherwise abstain",
            "frozen_c35_scores_reused": True,
            "frozen_c80_scores_available": False,
            "full_c80_scores_not_required_for_locked_prefix_slot5_policy": True,
            "reranker_model": str(args.model.resolve()),
            "max_length": args.max_length,
            "passages_per_doc": 3,
            "passage_aggregation": "max",
            "dynamic_padding": True,
            "eval": True,
            "inference_mode": True,
            "fp16": bool(args.fp16 and device.startswith("cuda")),
        },
        "counts": {
            "queries": len(rows),
            "dense_new_documents_scored": total_docs,
            "query_passage_pairs_scored": total_pairs,
            "empty_dense_new_documents": empty_docs,
            "promotions": len(promotions),
            "relevant_promotions": len(relevant_promotions),
            "destructive_changes": len(promotions) - len(relevant_promotions),
        },
        "metrics": {
            "baseline_recall": float(np.mean(old_recall)),
            "new_recall": float(np.mean(new_recall)),
            "gain": float(np.mean(gains)),
            "robust_gain_min_mod5": min(fold_gains),
            "fold_gains_mod5": fold_gains,
            "bootstrap_lb95_gain": _bootstrap_lb(gains, seed=42),
            "promotion_precision": len(relevant_promotions) / max(len(promotions), 1),
            "top1_4_byte_identical": all(bool(row["top1_4_byte_identical"]) for row in rows),
            "empty_output": any(not row["answer"] for row in rows),
        },
        "gate": {
            "gain_ge_0_010": float(np.mean(gains)) >= 0.010,
            "robust_gain_ge_0_008": min(fold_gains) >= 0.008,
            "bootstrap_lb95_ge_0_003": _bootstrap_lb(gains, seed=42) >= 0.003,
            "destructive_le_2": len(promotions) - len(relevant_promotions) <= 2,
            "precision_ge_0_90": len(relevant_promotions) / max(len(promotions), 1) >= 0.90,
            "top1_4_byte_identical": all(bool(row["top1_4_byte_identical"]) for row in rows),
            "empty_output_zero": not any(not row["answer"] for row in rows),
            "holdout_closed": True,
            "pass": bool(
                float(np.mean(gains)) >= 0.010
                and min(fold_gains) >= 0.008
                and _bootstrap_lb(gains, seed=42) >= 0.003
                and len(promotions) - len(relevant_promotions) <= 2
                and len(relevant_promotions) / max(len(promotions), 1) >= 0.90
                and all(bool(row["top1_4_byte_identical"]) for row in rows)
                and not any(not row["answer"] for row in rows)
            ),
        },
        "artifacts": {
            "dense_new_scores": str(score_path.resolve()),
            "dense_new_scores_sha256": sha256_file(score_path),
            "predictions": str(prediction_path.resolve()),
            "predictions_sha256": sha256_file(prediction_path),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(out / "a1d2_full_oof_report.json", report)
    print(json.dumps({"report": str((out / 'a1d2_full_oof_report.json').resolve()), "queries": len(rows), "dense_docs": total_docs, "pairs": total_pairs, "gain": report["metrics"]["gain"], "pass": report["gate"]["pass"], "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
