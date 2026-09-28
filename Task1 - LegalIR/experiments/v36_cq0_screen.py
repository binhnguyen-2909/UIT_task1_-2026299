"""V36-CQ0: one-fold clean-supervision Qwen3-Reranker pilot.

The model is used with the official yes/no reranker interface.  Training uses
only P/N/C relations from V36-R; U and CF_UNCERTAIN never enter the loss.
This is a screen, not a full OOF run and not a submission pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


INSTRUCTION = (
    "Given a Vietnamese legal question, determine whether the candidate legal provision directly provides the legal basis needed to answer the question. "
    "Distinguish carefully: legal subject, regulated act, condition, exception, negation, authority, numerical threshold, time limit, and legal consequence. "
    "A document that is only topically similar but does not directly support the question is irrelevant."
)
SYSTEM_PROMPT = 'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
SEED = 3600


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8", newline="\n") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: malformed JSONL") from exc
    return rows


def load_questions(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != "legalir_v18b_evidence_pack_manifest" or payload.get("split") != "clean_complement_only":
        raise ValueError("metadata is not clean-complement")
    if payload.get("holdout_opened") is not False or payload.get("public_labels_used") is not False:
        raise ValueError("metadata labels are not closed")
    return {str(row["qid"]): {"qid": str(row["qid"]), "query": str(row["query"]), "fold": int(row.get("fold", -1))} for row in payload["questions"]}


def numeric_key(value: str) -> tuple[int, str]:
    return (int(value), value) if str(value).isdigit() else (10**18, str(value))


def build_groups(relations: list[dict[str, Any]], max_c_per_query: int = 2, max_n_per_query: int = 2) -> list[dict[str, Any]]:
    by_qid: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"P": [], "N": [], "C": []})
    for row in relations:
        bucket = str(row.get("bucket"))
        if bucket not in {"P", "N", "C"}:
            continue
        by_qid[str(row["qid"])][bucket].append(row)
    groups = []
    for qid in sorted(by_qid, key=numeric_key):
        if int(qid) % 5 == 0:
            continue
        parts = by_qid[qid]
        if not parts["P"] or (not parts["C"] and not parts["N"]):
            continue
        positive = sorted(parts["P"], key=lambda row: numeric_key(str(row["doc_id"])))[0]
        counterfactuals = sorted(parts["C"], key=lambda row: str(row.get("counterfactual_id", "")))[:max_c_per_query]
        negatives = sorted(parts["N"], key=lambda row: str(row["doc_id"]))[:max_n_per_query]
        groups.append({"qid": qid, "positive": positive, "counterfactuals": counterfactuals, "negatives": negatives})
    return groups


def make_encoder(tokenizer: Any, query: str, document: str, max_length: int) -> list[int]:
    content = f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {document}"
    prefix = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    budget = max_length - len(prefix_ids) - len(suffix_ids)
    if budget < 16:
        raise ValueError("max_length leaves no room for pair content")
    content_ids = tokenizer.encode(content, add_special_tokens=False, truncation=True, max_length=budget)
    return [*prefix_ids, *content_ids, *suffix_ids]


def pad_batch(tokenizer: Any, sequences: list[list[int]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    pad_id = int(tokenizer.pad_token_id)
    width = max(len(sequence) for sequence in sequences)
    input_ids = [[pad_id] * (width - len(sequence)) + sequence for sequence in sequences]
    attention = [[0] * (width - len(sequence)) + [1] * len(sequence) for sequence in sequences]
    return torch.tensor(input_ids, dtype=torch.long, device=device), torch.tensor(attention, dtype=torch.long, device=device)


def score_pairs(model: Any, tokenizer: Any, pairs: list[tuple[str, str]], device: torch.device, true_id: int, false_id: int, max_length: int, batch_size: int, train: bool) -> torch.Tensor:
    values: list[torch.Tensor] = []
    context = torch.enable_grad() if train else torch.inference_mode()
    with context:
        for start in range(0, len(pairs), batch_size):
            sequences = [make_encoder(tokenizer, query, document, max_length) for query, document in pairs[start:start + batch_size]]
            input_ids, attention = pad_batch(tokenizer, sequences, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False, return_dict=True).logits[:, -1, :]
                values.append((logits[:, true_id] - logits[:, false_id]).float())
    return torch.cat(values, dim=0) if values else torch.zeros(0, device=device)


def evaluate_hard(model: Any, tokenizer: Any, rows: list[dict[str, Any]], device: torch.device, true_id: int, false_id: int, max_length: int, batch_size: int, fold_only: bool) -> dict[str, Any]:
    selected = [row for row in rows if not fold_only or int(str(row["qid"])) % 5 == 0]
    pairs = [(str(row["query"]), str(row["gold_text"])) for row in selected] + [(str(row["query"]), str(row["hard_text"])) for row in selected]
    scores = score_pairs(model, tokenizer, pairs, device, true_id, false_id, max_length, batch_size, False).cpu().numpy()
    n = len(selected)
    wins = scores[:n] > scores[n:]
    margins = scores[:n] - scores[n:]
    return {"pairs": n, "queries": len({str(row["qid"]) for row in selected}), "gold_wins": int(wins.sum()), "hard_win_rate": float(wins.mean()) if n else 0.0, "margin_mean": float(margins.mean()) if n else 0.0, "margin_median": float(np.median(margins)) if n else 0.0}


def evaluate_counterfactual(model: Any, tokenizer: Any, rows: list[dict[str, Any]], questions: Mapping[str, Mapping[str, Any]], device: torch.device, true_id: int, false_id: int, max_length: int, batch_size: int, fold_only: bool) -> dict[str, Any]:
    selected = [row for row in rows if not fold_only or int(str(row["qid"])) % 5 == 0]
    pairs = []
    for row in selected:
        original = str(row.get("original_query") or questions[str(row["qid"])]["query"])
        pairs.extend(((original, str(row["premise"])), (str(row["query"]), str(row["premise"]))))
    scores = score_pairs(model, tokenizer, pairs, device, true_id, false_id, max_length, batch_size, False).cpu().numpy()
    if not selected:
        return {"pairs": 0, "accuracy": 0.0}
    original, mutated = scores[0::2], scores[1::2]
    wins = original > mutated
    return {"pairs": len(selected), "teacher_style_flip_accuracy": float(wins.mean()), "original_mean": float(original.mean()), "counterfactual_mean": float(mutated.mean()), "margin_mean": float((original - mutated).mean())}


def evaluate_normal(model: Any, tokenizer: Any, relations: list[dict[str, Any]], device: torch.device, true_id: int, false_id: int, max_length: int, batch_size: int, fold_only: bool) -> dict[str, Any]:
    positives = {}
    negatives = []
    for row in relations:
        if row.get("bucket") == "P":
            positives.setdefault(str(row["qid"]), row)
        elif row.get("bucket") == "N":
            negatives.append(row)
    selected = [row for row in negatives if not fold_only or int(str(row["qid"])) % 5 == 0 if str(row["qid"]) in positives]
    pairs = []
    for row in selected:
        pairs.extend(((str(positives[str(row["qid"])]["query"]), str(positives[str(row["qid"])]["premise"])), (str(row["query"]), str(row["premise"]))))
    scores = score_pairs(model, tokenizer, pairs, device, true_id, false_id, max_length, batch_size, False).cpu().numpy()
    if not selected:
        return {"pairs": 0, "accuracy": 0.0}
    wins = scores[0::2] > scores[1::2]
    return {"pairs": len(selected), "positive_beats_negative": int(wins.sum()), "accuracy": float(wins.mean())}


def train_one_epoch(model: Any, tokenizer: Any, groups: list[dict[str, Any]], device: torch.device, true_id: int, false_id: int, max_length: int, microbatch: int, accumulation: int, learning_rate: float, output_log: Path) -> dict[str, Any]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    random.Random(SEED).shuffle(groups)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    steps = 0
    optimizer_steps = 0
    started = time.perf_counter()
    with output_log.open("w", encoding="utf-8") as log:
        for index, group in enumerate(groups, 1):
            positive = group["positive"]
            items: list[tuple[str, str, str]] = [(str(positive["query"]), str(positive["premise"]), "P")]
            items.extend((str(row["query"]), str(row["premise"]), "C") for row in group["counterfactuals"])
            items.extend((str(row["query"]), str(row["premise"]), "N") for row in group["negatives"])
            scores = score_pairs(model, tokenizer, [(query, premise) for query, premise, _ in items], device, true_id, false_id, max_length, microbatch, True)
            p = scores[0]
            c = scores[1:1 + len(group["counterfactuals"])]
            n = scores[1 + len(group["counterfactuals"]):]
            rank_loss = -F.log_softmax(scores, dim=0)[0]
            margin_terms = []
            if len(c):
                margin_terms.append(F.relu(0.2 - (p - c)).mean())
            if len(c) and len(n):
                margin_terms.append(F.relu(0.2 - (c[:, None] - n[None, :])).mean())
            cf_loss = torch.stack(margin_terms).mean() if margin_terms else torch.zeros((), device=device)
            loss = (rank_loss + 0.5 * cf_loss) / accumulation
            scaler.scale(loss).backward()
            losses.append(float(loss.detach().cpu()) * accumulation)
            steps += 1
            if steps % accumulation == 0 or index == len(groups):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            if index % 25 == 0 or index == len(groups):
                event = {"group": index, "groups_total": len(groups), "optimizer_steps": optimizer_steps, "mean_loss_recent": float(np.mean(losses[-25:])), "elapsed_seconds": time.perf_counter() - started}
                log.write(json.dumps(event) + "\n")
                log.flush()
                print(json.dumps(event, ensure_ascii=False), flush=True)
    return {"groups": len(groups), "optimizer_steps": optimizer_steps, "mean_loss": float(np.mean(losses)) if losses else 0.0, "elapsed_seconds": time.perf_counter() - started}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--relations", type=Path, required=True)
    parser.add_argument("--counterfactuals", type=Path, required=True)
    parser.add_argument("--hard-benchmark", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--microbatch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    args = parser.parse_args()
    started = time.perf_counter()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    questions = load_questions(args.metadata)
    relations = load_jsonl(args.relations)
    counterfactuals = load_jsonl(args.counterfactuals)
    hard = load_jsonl(args.hard_benchmark)
    clean_cf = [row for row in relations if row.get("bucket") == "C"]
    groups = build_groups(relations)
    if not clean_cf or not groups:
        raise RuntimeError(f"insufficient clean training data: C={len(clean_cf)} groups={len(groups)}")
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(str(args.model.resolve()), local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    true_id = int(tokenizer.convert_tokens_to_ids("yes"))
    false_id = int(tokenizer.convert_tokens_to_ids("no"))
    if true_id < 0 or false_id < 0:
        raise ValueError("Qwen yes/no token IDs are unavailable")
    base = AutoModelForCausalLM.from_pretrained(str(args.model.resolve()), local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True)
    base.config.use_cache = False
    lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "v_proj"], bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(base, lora_config).to(device)
    model.eval()
    trainable_params = sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad)
    total_params = sum(int(parameter.numel()) for parameter in model.parameters())
    baseline = {
        "hard_all": evaluate_hard(model, tokenizer, hard, device, true_id, false_id, args.max_length, args.microbatch_size, False),
        "hard_fold0": evaluate_hard(model, tokenizer, hard, device, true_id, false_id, args.max_length, args.microbatch_size, True),
        "counterfactual_all": evaluate_counterfactual(model, tokenizer, clean_cf, questions, device, true_id, false_id, args.max_length, args.microbatch_size, False),
        "counterfactual_fold0": evaluate_counterfactual(model, tokenizer, clean_cf, questions, device, true_id, false_id, args.max_length, args.microbatch_size, True),
        "normal_all": evaluate_normal(model, tokenizer, relations, device, true_id, false_id, args.max_length, args.microbatch_size, False),
        "normal_fold0": evaluate_normal(model, tokenizer, relations, device, true_id, false_id, args.max_length, args.microbatch_size, True),
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training = train_one_epoch(model, tokenizer, groups, device, true_id, false_id, args.max_length, args.microbatch_size, args.gradient_accumulation, args.learning_rate, args.output_dir / "v36_cq0_training.jsonl")
    model.eval()
    finetuned = {
        "hard_all": evaluate_hard(model, tokenizer, hard, device, true_id, false_id, args.max_length, args.microbatch_size, False),
        "hard_fold0": evaluate_hard(model, tokenizer, hard, device, true_id, false_id, args.max_length, args.microbatch_size, True),
        "counterfactual_all": evaluate_counterfactual(model, tokenizer, clean_cf, questions, device, true_id, false_id, args.max_length, args.microbatch_size, False),
        "counterfactual_fold0": evaluate_counterfactual(model, tokenizer, clean_cf, questions, device, true_id, false_id, args.max_length, args.microbatch_size, True),
        "normal_all": evaluate_normal(model, tokenizer, relations, device, true_id, false_id, args.max_length, args.microbatch_size, False),
        "normal_fold0": evaluate_normal(model, tokenizer, relations, device, true_id, false_id, args.max_length, args.microbatch_size, True),
    }
    checkpoint = args.output_dir / "cq0_checkpoint"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    peak_vram = float(torch.cuda.max_memory_allocated(device) / 1024**3) if device.type == "cuda" else 0.0
    hard_gain = finetuned["hard_fold0"]["hard_win_rate"] - baseline["hard_fold0"]["hard_win_rate"]
    normal_degradation = baseline["normal_fold0"]["accuracy"] - finetuned["normal_fold0"]["accuracy"]
    report = {
        "kind": "legalir_v36_cq0_clean_supervision_qwen_screen",
        "schema_version": 1,
        "stage": "V36-CQ0_one_fold_one_epoch",
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "model": str(args.model.resolve()),
        "device": str(device),
        "config": {"lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.05, "target_modules": ["q_proj", "v_proj"], "max_length": args.max_length, "microbatch_size": args.microbatch_size, "gradient_accumulation": args.gradient_accumulation, "learning_rate": args.learning_rate, "epochs": 1, "margin": 0.2, "loss": "L_rank + 0.5 L_cf", "train_fold": "qid % 5 != 0", "validation_fold": "qid % 5 == 0", "trainable_params": trainable_params, "total_params": total_params},
        "data": {"relations_total": len(relations), "P": sum(row.get("bucket") == "P" for row in relations), "N": sum(row.get("bucket") == "N" for row in relations), "C": len(clean_cf), "U_not_used": True, "CF_uncertain_not_used": True, "train_groups": len(groups), "hard_benchmark_rows": len(hard)},
        "baseline_zero_shot": baseline,
        "finetuned": finetuned,
        "delta": {"hard_fold0_win_gain": hard_gain, "counterfactual_fold0_accuracy_gain": finetuned["counterfactual_fold0"]["teacher_style_flip_accuracy"] - baseline["counterfactual_fold0"]["teacher_style_flip_accuracy"], "normal_fold0_degradation": normal_degradation},
        "gate": {"hard_fold0_ge_0_72": finetuned["hard_fold0"]["hard_win_rate"] >= 0.72, "counterfactual_fold0_ge_0_80": finetuned["counterfactual_fold0"]["teacher_style_flip_accuracy"] >= 0.80, "normal_degradation_le_0_02": normal_degradation <= 0.02, "pass_cq0": finetuned["hard_fold0"]["hard_win_rate"] >= 0.72 and finetuned["counterfactual_fold0"]["teacher_style_flip_accuracy"] >= 0.80 and normal_degradation <= 0.02},
        "training": training,
        "peak_vram_gb": peak_vram,
        "inputs": {"metadata": {"path": str(args.metadata.resolve()), "sha256": sha256_file(args.metadata)}, "relations": {"path": str(args.relations.resolve()), "sha256": sha256_file(args.relations)}, "counterfactuals": {"path": str(args.counterfactuals.resolve()), "sha256": sha256_file(args.counterfactuals)}, "hard_benchmark": {"path": str(args.hard_benchmark.resolve()), "sha256": sha256_file(args.hard_benchmark)}},
        "outputs": {"checkpoint": str(checkpoint.resolve()), "training_log": str((args.output_dir / "v36_cq0_training.jsonl").resolve())},
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(args.output_dir / "v36_cq0_screen_report.json", report)
    print(json.dumps({"report": str((args.output_dir / "v36_cq0_screen_report.json").resolve()), "gate": report["gate"], "baseline_hard_fold0": baseline["hard_fold0"], "finetuned_hard_fold0": finetuned["hard_fold0"], "finetuned_cf_fold0": finetuned["counterfactual_fold0"], "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
