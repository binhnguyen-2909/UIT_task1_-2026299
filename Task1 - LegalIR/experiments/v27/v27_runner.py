"""Run the V27-A0/A1/A2 stages with one frozen, label-safe E3B manifest."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from v27_common import (
    SEED,
    label_for_doc,
    load_metadata,
    read_json,
    sha256_file,
    write_json,
    write_jsonl,
)
from v27_evaluate import evaluate_policy, fit_platt, select_thresholds
from v27_model import load_groups, load_model, save_checkpoint, score_questions, train_model


def load_packs(path: Path, questions: Mapping[str, Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    packs: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                key = (str(row["qid"]), str(row["doc_id"]))
                if key in packs:
                    raise ValueError(f"duplicate E3B row {key}")
                packs[key] = row
    expected = {(qid, doc) for qid, question in questions.items() for doc in question["c35"]}
    if set(packs) != expected:
        raise ValueError(f"E3B coverage mismatch: got={len(packs)} expected={len(expected)}")
    return packs


def load_canonical(path: Path) -> dict[str, str]:
    result = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                result[str(row["doc_id"])] = str(row["canonical_doc_id"])
    return result


def score_rows(scores: Mapping[tuple[str, str], float], questions: Mapping[str, Mapping[str, Any]], qids: Sequence[str], fold: int | None = None) -> list[dict[str, Any]]:
    rows = []
    for qid in qids:
        for doc in questions[qid]["c35"]:
            if (qid, doc) not in scores:
                continue
            row: dict[str, Any] = {"qid": qid, "doc_id": doc, "raw_logit": float(scores[(qid, doc)])}
            if fold is not None:
                row["fold"] = int(fold)
            rows.append(row)
    return rows


def pair_metrics(scores: Mapping[tuple[str, str], float], questions: Mapping[str, Mapping[str, Any]], qids: Sequence[str], canonical: Mapping[str, str]) -> dict[str, Any]:
    pair_results = []
    labels, values = [], []
    for qid in qids:
        row = questions[qid]
        scored_docs = [doc for doc in row["c35"] if (qid, doc) in scores]
        gold = [doc for doc in scored_docs if label_for_doc(row, doc, canonical)]
        negatives = [doc for doc in scored_docs if not label_for_doc(row, doc, canonical)]
        labels.extend(label_for_doc(row, doc, canonical) for doc in scored_docs)
        values.extend(float(scores[(qid, doc)]) for doc in scored_docs)
        for positive in gold:
            for negative in negatives:
                pair_results.append(float(scores[(qid, positive)] > scores[(qid, negative)]))
    return {
        "queries": len(qids),
        "gold_vs_non_gold_pairwise_accuracy": float(np.mean(pair_results)) if pair_results else 0.0,
        "relevance_auc": float(roc_auc_score(labels, values)) if len(set(labels)) > 1 else 0.0,
        "pairs": len(pair_results),
    }


def intervention_metrics(scores: Mapping[tuple[str, str], float], questions: Mapping[str, Mapping[str, Any]], qids: Sequence[str], canonical: Mapping[str, str]) -> dict[str, Any]:
    calibration = fit_platt(scores, questions, qids, canonical)
    probabilities = {(qid, doc): 1.0 / (1.0 + np.exp(-np.clip(calibration["coefficient"] * score + calibration["intercept"], -60, 60))) for (qid, doc), score in scores.items()}
    fold_map = {qid: 0 for qid in qids}
    threshold = select_thresholds(questions, qids, probabilities, canonical, fold_map)
    summary, _ = evaluate_policy(questions, qids, probabilities, canonical, tau_abs=threshold["tau_abs"], tau_delta=threshold["tau_delta"], fold_map=fold_map)
    return {"platt": calibration, "threshold": threshold, "summary": summary}


def deterministic_subset(qids: Sequence[str], fraction: float, seed: int = SEED) -> list[str]:
    ordered = sorted(map(str, qids), key=lambda qid: __import__("hashlib").sha256(f"{qid}|{seed}|screen".encode()).hexdigest())
    return ordered[:max(1, math.ceil(len(ordered) * fraction))]


def load_runtime(args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]], dict[str, str], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    questions = load_metadata(args.metadata)
    packs = load_packs(args.e3b_manifest, questions)
    canonical = load_canonical(args.canonical_map)
    groups = load_groups(args.pairs)
    split = read_json(args.split)
    return questions, packs, canonical, groups, split


def train_one(args: argparse.Namespace, questions, groups, qids: Sequence[str], output_dir: Path, learning_rate: float, epochs: float) -> tuple[Any, Any, dict[str, Any]]:
    device = torch.device(args.device)
    tokenizer, model = load_model(args.model, device)
    report = train_model(
        model, tokenizer, groups, qids,
        device=device,
        learning_rate=learning_rate,
        epochs=epochs,
        microbatch=args.microbatch,
        effective_batch_queries=32,
        fp16=device.type == "cuda",
        seed=SEED,
    )
    save_checkpoint(model, tokenizer, output_dir / "checkpoint")
    write_json(output_dir / "training_report.json", report)
    return tokenizer, model, report


def run_a0(args: argparse.Namespace) -> int:
    questions, packs, canonical, _, _ = load_runtime(args)
    qids = sorted(questions)[:args.sample_queries] if args.sample_queries else sorted(questions)
    device = torch.device(args.device)
    tokenizer, model = load_model(args.model, device)
    scores = score_questions(model, tokenizer, questions, packs, qids, device=device, batch_size=args.score_batch_size, fp16=device.type == "cuda")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "a0_scores.jsonl", score_rows(scores, questions, qids))
    report = {"kind": "legalir_v27_a0_zero_shot", "holdout_opened": False, "public_labels_used": False, "model": str(args.model.resolve()), "queries": len(qids), "score_batch_size": args.score_batch_size, "metrics": pair_metrics(scores, questions, qids, canonical), "inputs": {"metadata": sha256_file(args.metadata), "e3b_manifest": sha256_file(args.e3b_manifest)}}
    write_json(out / "a0_report.json", report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def run_screen(args: argparse.Namespace) -> int:
    questions, packs, canonical, groups, split = load_runtime(args)
    fold = next(row for row in split["folds"] if int(row["fold"]) == args.fold)
    train_qids = deterministic_subset(fold["fit_qids"], args.screen_fraction)
    calibration_qids = list(map(str, fold["calibration_qids"]))
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    for learning_rate in (1e-5, 2e-5):
        tag = f"lr_{learning_rate:.0e}".replace("-", "m")
        stage_dir = out / tag
        tokenizer, model, training = train_one(args, questions, groups, train_qids, stage_dir, learning_rate, args.screen_epochs)
        scores = score_questions(model, tokenizer, questions, packs, calibration_qids, device=torch.device(args.device), batch_size=args.score_batch_size, fp16=torch.device(args.device).type == "cuda")
        write_jsonl(stage_dir / "calibration_logits.jsonl", score_rows(scores, questions, calibration_qids, fold=args.fold))
        metrics = {"pair": pair_metrics(scores, questions, calibration_qids, canonical), "intervention": intervention_metrics(scores, questions, calibration_qids, canonical)}
        report = {"learning_rate": learning_rate, "train_qids": len(train_qids), "calibration_qids": len(calibration_qids), "training": training, "metrics": metrics}
        write_json(stage_dir / "screen_report.json", report)
        reports.append(report)
        print(json.dumps({"learning_rate": learning_rate, "pair": metrics["pair"], "intervention": metrics["intervention"]["summary"]}, ensure_ascii=False), flush=True)
        del model, tokenizer
        if torch.device(args.device).type == "cuda":
            torch.cuda.empty_cache()
    winner = max(reports, key=lambda report: (report["metrics"]["pair"]["gold_vs_non_gold_pairwise_accuracy"], report["metrics"]["pair"]["relevance_auc"], report["metrics"]["intervention"]["summary"]["gain"], -report["metrics"]["intervention"]["summary"]["destructive"]))
    summary = {"kind": "legalir_v27_a1_screen", "holdout_opened": False, "public_labels_used": False, "fold": args.fold, "screen_fraction": args.screen_fraction, "epochs": args.screen_epochs, "candidates": reports, "winner_learning_rate": winner["learning_rate"], "selection": "pairwise accuracy, AUC, intervention gain, destructive edits"}
    write_json(out / "a1_screen_report.json", summary)
    print(json.dumps({"winner_learning_rate": winner["learning_rate"], "output": str((out / 'a1_screen_report.json').resolve())}, ensure_ascii=False))
    return 0


def run_a2(args: argparse.Namespace) -> int:
    questions, packs, _, groups, split = load_runtime(args)
    fold = next(row for row in split["folds"] if int(row["fold"]) == args.fold)
    out = args.output_dir / f"fold_{args.fold}"
    tokenizer, model, training = train_one(args, questions, groups, fold["fit_qids"], out, args.learning_rate, args.epochs)
    device = torch.device(args.device)
    for role, qids in (("calibration", fold["calibration_qids"]), ("outer", fold["test_qids"])):
        scores = score_questions(model, tokenizer, questions, packs, qids, device=device, batch_size=args.score_batch_size, fp16=device.type == "cuda")
        write_jsonl(out / f"fold_{args.fold}_{role}_logits.jsonl", score_rows(scores, questions, qids, fold=args.fold))
    write_json(out / "a2_fold_report.json", {"fold": args.fold, "fit_qids": len(fold["fit_qids"]), "calibration_qids": len(fold["calibration_qids"]), "outer_qids": len(fold["test_qids"]), "learning_rate": args.learning_rate, "epochs": args.epochs, "training": training})
    print(json.dumps({"fold": args.fold, "output": str(out.resolve()), "training": training}, ensure_ascii=False))
    return 0


def run_oof(args: argparse.Namespace) -> int:
    questions, packs, _, groups, split = load_runtime(args)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    fold_reports = []
    for fold_row in split["folds"]:
        fold = int(fold_row["fold"])
        fold_dir = out / f"fold_{fold}"
        checkpoint = fold_dir / "checkpoint"
        reusable_checkpoint = checkpoint
        if fold == args.reuse_fold and args.reuse_checkpoint_dir:
            reusable_checkpoint = args.reuse_checkpoint_dir
        if fold == args.reuse_fold and reusable_checkpoint.exists():
            tokenizer, model = load_model(reusable_checkpoint, device)
            training = {"reused_checkpoint": str(reusable_checkpoint.resolve())}
        else:
            tokenizer, model, training = train_one(args, questions, groups, fold_row["fit_qids"], fold_dir, args.learning_rate, args.epochs)
        for role, qids in (("calibration", fold_row["calibration_qids"]), ("outer", fold_row["test_qids"])):
            scores = score_questions(model, tokenizer, questions, packs, qids, device=device, batch_size=args.score_batch_size, fp16=device.type == "cuda")
            write_jsonl(fold_dir / f"fold_{fold}_{role}_logits.jsonl", score_rows(scores, questions, qids, fold=fold))
        report = {"fold": fold, "fit_qids": len(fold_row["fit_qids"]), "calibration_qids": len(fold_row["calibration_qids"]), "outer_qids": len(fold_row["test_qids"]), "training": training}
        write_json(fold_dir / "oof_fold_report.json", report)
        fold_reports.append(report)
        print(json.dumps({"fold": fold, "fit_qids": report["fit_qids"], "outer_qids": report["outer_qids"]}, ensure_ascii=False), flush=True)
        del model, tokenizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_json(out / "oof_training_report.json", {"kind": "legalir_v27_a2_oof_training", "holdout_opened": False, "public_labels_used": False, "learning_rate": args.learning_rate, "epochs": args.epochs, "folds": fold_reports})
    print(json.dumps({"output": str(out.resolve()), "folds": len(fold_reports)}, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("a0", "screen", "a2", "oof"))
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--e3b-manifest", type=Path, required=True)
    parser.add_argument("--canonical-map", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--score-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=float, default=1.25)
    parser.add_argument("--screen-epochs", type=float, default=0.4)
    parser.add_argument("--screen-fraction", type=float, default=0.30)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--sample-queries", type=int, default=200)
    parser.add_argument("--reuse-fold", type=int, default=-1)
    parser.add_argument("--reuse-checkpoint-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.command == "a0":
        return run_a0(args)
    if args.command == "screen":
        return run_screen(args)
    if args.command == "a2":
        return run_a2(args)
    return run_oof(args)


if __name__ == "__main__":
    raise SystemExit(main())
