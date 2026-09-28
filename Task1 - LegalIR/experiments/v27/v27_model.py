"""Small full-fine-tuning runner for the V27 mMARCO CrossEncoder."""

from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from v27_common import MAX_LENGTH, SEED, write_json


def load_model(model_dir: Path, device: torch.device) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True)
    if int(getattr(model.config, "num_labels", 1)) != 1:
        raise ValueError(f"V27 expects a scalar CrossEncoder, got num_labels={model.config.num_labels}")
    model.to(device)
    return tokenizer, model


def load_groups(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["qid"])
            positive = str(row["positive_doc"])
            item = grouped[qid].setdefault(positive, {"qid": qid, "query": str(row["query"]), "positive_doc": positive, "positive_pack": str(row["positive_pack"]), "negative_docs": [], "negative_packs": [], "negative_types": []})
            if not str(row["positive_pack"]).strip() or not str(row["negative_pack"]).strip():
                raise ValueError(f"{qid}/{positive}: training pair contains an empty pack")
            item["negative_docs"].append(str(row["negative_doc"]))
            item["negative_packs"].append(str(row["negative_pack"]))
            item["negative_types"].append(str(row["negative_type"]))
    result = {qid: list(items.values()) for qid, items in grouped.items()}
    for qid, items in result.items():
        for item in items:
            if len(item["negative_docs"]) != 4 or len(set(item["negative_docs"])) != 4:
                raise ValueError(f"{qid}/{item['positive_doc']}: expected four unique negatives")
    return result


def encode_pairs(tokenizer: Any, queries: Sequence[str], docs: Sequence[str], device: torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer(list(queries), list(docs), padding=True, truncation=True, max_length=MAX_LENGTH, pad_to_multiple_of=8, return_tensors="pt")
    return {key: value.to(device, non_blocking=True) for key, value in encoded.items()}


def scalar_logits(model: Any, tokenizer: Any, queries: Sequence[str], docs: Sequence[str], device: torch.device, fp16: bool) -> torch.Tensor:
    batch = encode_pairs(tokenizer, queries, docs, device)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16 and device.type == "cuda"):
        output = model(**batch, return_dict=True).logits
    return output.reshape(-1).float()


def train_model(
    model: Any,
    tokenizer: Any,
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    fit_qids: Sequence[str],
    *,
    device: torch.device,
    learning_rate: float,
    epochs: float,
    microbatch: int,
    effective_batch_queries: int = 32,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.06,
    max_grad_norm: float = 1.0,
    fp16: bool = True,
    seed: int = SEED,
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    try:
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay, fused=True)
        optimizer_name = "adamw_torch_fused"
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
        optimizer_name = "adamw_torch"
    qids = list(map(str, fit_qids))
    queries_per_epoch = math.ceil(len(qids) / max(effective_batch_queries, 1))
    total_steps = max(1, math.ceil(queries_per_epoch * epochs))
    warmup_steps = max(1, math.ceil(total_steps * warmup_ratio))
    scaler = torch.cuda.amp.GradScaler(enabled=fp16 and device.type == "cuda")
    model.train()
    start = time.perf_counter()
    optimizer_steps = 0
    loss_trace: list[float] = []
    examples_seen = 0
    while optimizer_steps < total_steps:
        epoch = optimizer_steps // max(queries_per_epoch, 1)
        shuffled = list(qids)
        random.Random(seed + epoch).shuffle(shuffled)
        for start_q in range(0, len(shuffled), effective_batch_queries):
            if optimizer_steps >= total_steps:
                break
            batch_qids = shuffled[start_q:start_q + effective_batch_queries]
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            for qid in batch_qids:
                qgroups = list(groups.get(qid, ()))
                if not qgroups:
                    continue
                q_loss_value = 0.0
                for group in qgroups:
                    query = str(group.get("query", ""))
                    docs = [str(group["positive_pack"])] + [str(value) for value in group["negative_packs"]]
                    queries = [query] * len(docs)
                    logits_chunks: list[torch.Tensor] = []
                    for index in range(0, len(docs), microbatch):
                        logits_chunks.append(scalar_logits(model, tokenizer, queries[index:index + microbatch], docs[index:index + microbatch], device, fp16))
                    logits = torch.cat(logits_chunks)
                    labels = torch.zeros_like(logits)
                    labels[0] = 1.0
                    point = F.binary_cross_entropy_with_logits(logits, labels)
                    pair = F.softplus(-(logits[0] - logits[1:])).mean()
                    q_loss = 0.60 * point + 0.40 * pair
                    scaled = q_loss / max(len(qgroups) * len(batch_qids), 1)
                    scaler.scale(scaled).backward()
                    q_loss_value += float(q_loss.detach().cpu()) / len(qgroups)
                    examples_seen += len(docs)
                batch_loss += q_loss_value / max(len(batch_qids), 1)
            if optimizer_steps < warmup_steps:
                lr_scale = float(optimizer_steps + 1) / float(warmup_steps)
            else:
                lr_scale = 1.0
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * lr_scale
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm).detach().cpu())
            scaler.step(optimizer)
            scaler.update()
            optimizer_steps += 1
            loss_trace.append(batch_loss)
            if optimizer_steps % 10 == 0 or optimizer_steps == total_steps:
                print(json.dumps({"step": optimizer_steps, "steps_total": total_steps, "loss": batch_loss, "grad_norm": grad_norm, "fit_queries": len(qids)}, ensure_ascii=False), flush=True)
    elapsed = time.perf_counter() - start
    return {"fit_queries": len(qids), "optimizer_steps": optimizer_steps, "planned_steps": total_steps, "epochs": epochs, "learning_rate": learning_rate, "microbatch": microbatch, "effective_batch_queries": effective_batch_queries, "optimizer": optimizer_name, "fp16": fp16, "examples_seen": examples_seen, "examples_per_second": examples_seen / max(elapsed, 1e-9), "elapsed_seconds": elapsed, "loss_trace": loss_trace}


def score_records(model: Any, tokenizer: Any, records: Sequence[Mapping[str, Any]], *, device: torch.device, batch_size: int = 4, fp16: bool = True) -> list[float]:
    model.eval()
    outputs: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            queries = [str(row["query"]) for row in batch]
            docs = [str(row["pack"]) for row in batch]
            outputs.extend(float(value) for value in scalar_logits(model, tokenizer, queries, docs, device, fp16).cpu().tolist())
    return outputs


def score_questions(model: Any, tokenizer: Any, questions: Mapping[str, Mapping[str, Any]], packs: Mapping[tuple[str, str], Mapping[str, Any]], qids: Sequence[str], *, device: torch.device, batch_size: int = 4, fp16: bool = True) -> dict[tuple[str, str], float]:
    records = [{"qid": qid, "doc_id": doc, "query": questions[qid]["query"], "pack": packs[(qid, doc)]["pack"]} for qid in qids for doc in questions[qid]["c35"] if not bool(packs[(qid, doc)].get("empty")) and str(packs[(qid, doc)].get("pack", "")).strip()]
    scores = score_records(model, tokenizer, records, device=device, batch_size=batch_size, fp16=fp16)
    return {(str(row["qid"]), str(row["doc_id"])): float(score) for row, score in zip(records, scores, strict=True)}


def save_checkpoint(model: Any, tokenizer: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))


def save_training_report(path: Path, report: Mapping[str, Any]) -> None:
    write_json(path, dict(report))
