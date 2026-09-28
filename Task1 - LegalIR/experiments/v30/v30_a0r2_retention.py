"""V30-A0R2: mixed-negative LoRA adaptation with base-teacher retention."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
V27_DIR = SCRIPT_DIR.parent / "v27"
for path in (SCRIPT_DIR, V27_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from v27_common import read_json, sha256_file, write_json, write_jsonl  # noqa: E402
from v30_a0_screen import LoRAEncoder, _probe_eval, _read_jsonl, _sample_group  # noqa: E402


SEED = 42
TEMPERATURE = 0.02
LAMBDA_RETENTION = 0.5


def _mixed_group(row: Mapping[str, Any], rng: random.Random) -> tuple[str, list[str]]:
    positives = [chunk["text"] for doc in row["positive_docs"] for chunk in doc["chunks"]]
    negatives = [doc["chunk"]["text"] for doc in row["negative_docs"]]
    if not positives or not negatives:
        raise ValueError(f"{row['qid']}: missing positive or negative chunks")
    hard = negatives[:6]
    medium = negatives[6:11]
    easy = negatives[11:18]

    def pick(pool: list[str], count: int) -> list[str]:
        source = pool or negatives
        if len(source) >= count:
            return rng.sample(source, count)
        selected = list(source)
        while len(selected) < count:
            selected.append(rng.choice(source))
        return selected

    positive = rng.choice(positives)
    return positive, [*pick(hard, 3), *pick(medium, 2), *pick(easy, 2)]


def _retention_candidates(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    ranges = ((1, 10), (11, 50), (51, 200), (201, 500))
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for lo, hi in ranges:
        pool = [candidate for candidate in row["candidates"] if lo <= int(candidate["dense_rank"]) <= hi and np.isfinite(float(candidate["dense_score"]))]
        for candidate in pool[:2]:
            docid = str(candidate["doc_id"])
            if docid not in used:
                selected.append(candidate)
                used.add(docid)
    if len(selected) != 8:
        raise ValueError(f"{row['qid']}: retention rank buckets do not contain 8 candidates")
    return selected


def _make_retention_rows(probe_rows: list[dict[str, Any]], query_summaries: list[dict[str, Any]], output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_qid = {str(row["qid"]): row for row in probe_rows}
    lost = [str(item["qid"]) for item in query_summaries if item["baseline_top50"] and not item["new_top50"]]
    stable = [str(item["qid"]) for item in query_summaries if item["baseline_top50"] and item["new_top50"]]
    rng = random.Random(SEED)
    stable_sample = rng.sample(stable, min(len(stable), 2 * len(lost)))
    if len(stable_sample) < 2 * len(lost):
        remaining = [str(item["qid"]) for item in query_summaries if str(item["qid"]) not in set(lost) and str(item["qid"]) not in set(stable_sample)]
        stable_sample.extend(remaining[: 2 * len(lost) - len(stable_sample)])
    selected_qids = lost + stable_sample
    retention_rows = []
    for qid in selected_qids:
        source = by_qid[qid]
        candidates = _retention_candidates(source)
        retention_rows.append({
            "qid": qid,
            "query": source["query"],
            "kind": "lost_top50" if qid in set(lost) else "stable_top50",
            "candidates": [{"doc_id": str(candidate["doc_id"]), "baseline_rank": int(candidate["dense_rank"]), "teacher_score": float(candidate["dense_score"]), "text": str(candidate["chunks"][0]["text"])} for candidate in candidates],
        })
    write_jsonl(output_dir / "v30_a0r2_retention_manifest.jsonl", retention_rows)
    prep = {
        "kind": "legalir_v30_a0r2_retention_prepare",
        "lost_top50_queries": len(lost),
        "stable_top50_queries": len(stable_sample),
        "retention_queries": len(retention_rows),
        "ratio": "1 lost : 2 stable",
        "rank_buckets": [[1, 10], [11, 50], [51, 200], [201, 500]],
        "teacher": "frozen original BGE-M3 dense_score",
        "lambda_retention": LAMBDA_RETENTION,
        "manifest_sha256": sha256_file(output_dir / "v30_a0r2_retention_manifest.jsonl"),
    }
    write_json(output_dir / "v30_a0r2_prepare_report.json", prep)
    return retention_rows, prep


def _retention_loss(encoder: LoRAEncoder, row: Mapping[str, Any]) -> torch.Tensor:
    passages = [str(candidate["text"]) for candidate in row["candidates"]]
    teacher_scores = torch.tensor([float(candidate["teacher_score"]) for candidate in row["candidates"]], dtype=torch.float32, device=encoder.device)
    qvec = encoder._forward([str(row["query"])], encoder.query_max_len, True)
    pvec = encoder._forward(passages, encoder.passage_max_len, True)
    student_scores = (qvec @ pvec.T).squeeze(0)
    teacher_probs = F.softmax(teacher_scores / TEMPERATURE, dim=-1)
    return F.kl_div(F.log_softmax(student_scores / TEMPERATURE, dim=-1), teacher_probs, reduction="batchmean")


def _gate(metrics: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    median_ratio = float(metrics["median_gold_rank"]) / max(float(baseline["median_gold_rank"]), 1e-9)
    win_gain = float(metrics["hard_negative_win_rate"]) - float(baseline["hard_negative_win_rate"])
    gold_gain = int(metrics["gold_docs_at_100"]) - int(baseline["gold_docs_at_100"])
    lost = int(metrics["lost_top50"])
    return {
        "gold_docs_at_100_gain": gold_gain,
        "hard_negative_win_gain": win_gain,
        "median_ratio": median_ratio,
        "lost_top50": lost,
        "pass": bool(gold_gain >= 5 and win_gain >= 0.08 and median_ratio <= 0.70 and lost <= 2),
    }


def _selection_key(metrics: Mapping[str, Any], baseline: Mapping[str, Any]) -> tuple[Any, ...]:
    gate = _gate(metrics, baseline)
    if int(gate["lost_top50"]) <= 2:
        return (1, -float(metrics["median_gold_rank"]), int(metrics["gold_docs_at_100"]))
    return (0, -int(gate["lost_top50"]), -float(metrics["median_gold_rank"]), int(metrics["gold_docs_at_100"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--query-report", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--encode-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--lambda-retention", type=float, default=LAMBDA_RETENTION)
    parser.add_argument("--steps", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = _read_jsonl(args.train_manifest)
    probe_rows = _read_jsonl(args.probe_manifest)
    query_report = read_json(args.query_report)
    query_summaries = query_report["raw_results"]["1.0"]["query_summaries"]
    retention_rows, retention_prep = _make_retention_rows(probe_rows, query_summaries, args.output_dir)
    retention_prep["lambda_retention"] = args.lambda_retention
    write_json(args.output_dir / "v30_a0r2_prepare_report.json", retention_prep)
    valid_train = [row for row in train_rows if row["positive_docs"] and row["negative_docs"]]
    baseline = _probe_eval(None, probe_rows, args.encode_batch_size, baseline_only=True)
    encoder = LoRAEncoder(args.model, args.device, 96, 256)
    optimizer = torch.optim.AdamW([parameter for parameter in encoder.model.parameters() if parameter.requires_grad], lr=args.learning_rate)
    rng = random.Random(SEED)
    total_steps = args.steps if args.steps > 0 else len(valid_train)
    accumulation = 8
    events = {max(1, math.ceil(total_steps * fraction)) for fraction in (0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00)}
    evaluations: list[dict[str, Any]] = []
    best_key: tuple[Any, ...] | None = None
    best_path: Path | None = None
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    adaptation_steps = 0
    retention_steps = 0
    skipped_rows = len(train_rows) - len(valid_train)
    for step in range(1, total_steps + 1):
        if step % 2 == 1:
            row = valid_train[(step // 2) % len(valid_train)]
            positive, negatives = _mixed_group(row, rng)
            loss = encoder.group_loss(str(row["query"]), [positive, *negatives], TEMPERATURE)
            adaptation_steps += 1
        else:
            row = retention_rows[(step // 2 - 1) % len(retention_rows)]
            loss = args.lambda_retention * _retention_loss(encoder, row)
            retention_steps += 1
        (loss / accumulation).backward()
        if step % accumulation == 0 or step == total_steps or step in events:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step in events:
            metrics = _probe_eval(encoder, probe_rows, args.encode_batch_size)
            checkpoint = {"step": step, "elapsed_seconds": time.perf_counter() - started, **metrics}
            evaluations.append(checkpoint)
            event_path = args.output_dir / "checkpoints" / f"step{step:04d}"
            encoder.save(event_path)
            key = _selection_key(metrics, baseline)
            if best_key is None or key > best_key:
                best_key = key
                best_path = args.output_dir / "checkpoints" / "best"
                encoder.save(best_path)
            print(json.dumps({"step": step, "gate": _gate(metrics, baseline), "selection_key": key}, ensure_ascii=False), flush=True)
    final = evaluations[-1] if evaluations else _probe_eval(encoder, probe_rows, args.encode_batch_size)
    report = {
        "kind": "legalir_v30_a0r2_retention_preserving_screen",
        "schema_version": 1,
        "model": "BAAI/bge-m3",
        "model_path": str(args.model.resolve()),
        "probe_manifest": str(args.probe_manifest.resolve()),
        "probe_manifest_sha256": sha256_file(args.probe_manifest),
        "train_manifest": str(args.train_manifest.resolve()),
        "query_report": str(args.query_report.resolve()),
        "config": {"lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.05, "target_modules": ["query", "value"], "query_max_len": 96, "passage_max_len": 256, "temperature": TEMPERATURE, "learning_rate": args.learning_rate, "lambda_retention": args.lambda_retention, "mixed_negative_layout": "3 hard + 2 medium + 2 easy", "alternating_ratio": "50% adaptation / 50% retention", "gradient_accumulation": accumulation, "steps": total_steps},
        "retention_prepare": retention_prep,
        "baseline_probe": baseline,
        "evaluations": evaluations,
        "final_probe": final,
        "final_gate": _gate(final, baseline),
        "best_gate": None if not evaluations else _gate(max(evaluations, key=lambda item: _selection_key(item, baseline)), baseline),
        "best_checkpoint": None if best_path is None else str(best_path.resolve()),
        "adaptation_steps": adaptation_steps,
        "retention_steps": retention_steps,
        "skipped_train_rows": skipped_rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(args.output_dir / "v30_a0r2_report.json", report)
    print(json.dumps({"report": str((args.output_dir / "v30_a0r2_report.json").resolve()), "final_gate": report["final_gate"], "best_checkpoint": report["best_checkpoint"], "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
