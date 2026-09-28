"""V30-A0: fast BGE-M3 LoRA retrieval screen on a fixed probe set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
V29_DIR = SCRIPT_DIR.parent / "v29"
V27_DIR = SCRIPT_DIR.parent / "v27"
for path in (V29_DIR, V27_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from v27_common import load_metadata, load_sidecar, read_json, sha256_file, write_json, write_jsonl  # noqa: E402
from v29_a0_dense import _sample_manifest  # noqa: E402


TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)
SEED = 42
RRF_UNUSED = None


def _numeric_key(value: str) -> tuple[int, str]:
    return (int(value), value) if str(value).isdigit() else (10**18, str(value))


def _tokens(value: str) -> set[str]:
    return set(TOKEN_RE.findall(str(value).casefold()))


def _load_dense(path: Path, qids: set[str]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            raw = json.loads(line)
            qid = str(raw["qid"])
            if qid not in qids:
                continue
            candidates = [{"doc_id": str(item["doc_id"]), "rank": int(item["rank"]), "score": float(item["score"])} for item in raw["candidates"][:1000]]
            if len(candidates) < 500 or len({item["doc_id"] for item in candidates}) != len(candidates):
                raise ValueError(f"{qid}: invalid dense pool")
            result[qid] = candidates
    if set(result) != qids:
        raise ValueError("dense pool coverage mismatch")
    return result


def _load_stage1(path: Path, questions: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    raw = read_json(path)
    result = {str(row["query_id"]): [str(doc) for doc in docs] for row, docs in zip(raw["queries"], raw["stage1_candidates"], strict=True)}
    if set(questions) - set(result):
        raise ValueError("stage1 audit misses metadata qids")
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _dense_best_chunk(indices: list[int], qvec: np.ndarray, chunk_vectors: np.ndarray) -> int:
    scores = np.asarray(chunk_vectors[indices], dtype=np.float32) @ np.asarray(qvec, dtype=np.float32)
    return int(indices[int(np.argmax(scores))])


def _lexical_best_chunk(indices: list[int], query_tokens: set[str], texts: list[str]) -> int:
    def key(index: int) -> tuple[float, int]:
        terms = _tokens(texts[index])
        return (len(query_tokens & terms) / math.sqrt(max(len(terms), 1)), -index)
    return max(indices, key=key)


def _chunk_record(index: int, docid: str, texts: list[str], score: float | None = None) -> dict[str, Any]:
    row = {"chunk_id": f"{docid}:{index}", "text": texts[index]}
    if score is not None:
        row["zero_shot_score"] = float(score)
    return row


def _select_chunks(docid: str, n: int, query_vec: np.ndarray, chunk_vectors: np.ndarray, texts: list[str], doc_indices: Mapping[str, list[int]]) -> list[dict[str, Any]]:
    indices = doc_indices.get(docid, [])
    if not indices:
        return []
    scores = np.asarray(chunk_vectors[indices], dtype=np.float32) @ np.asarray(query_vec, dtype=np.float32)
    order = np.argsort(-scores, kind="stable")[:n]
    return [_chunk_record(indices[int(pos)], docid, texts, float(scores[int(pos)])) for pos in order]


def _negative_doc_pool(qid: str, question: Mapping[str, Any], dense: Mapping[str, list[dict[str, Any]]], sidecar: Mapping[tuple[str, str], float], stage1: Mapping[str, list[str]]) -> list[str]:
    refs = set(question["references"])
    selected: list[str] = []

    def add(docid: str) -> None:
        docid = str(docid)
        if docid and docid not in refs and docid not in selected:
            selected.append(docid)

    dense_rows = dense[qid]
    for lo, hi, count in ((1, 20, 6), (21, 50, 5), (51, 500, 7)):
        for item in dense_rows[lo - 1 : hi]:
            if len(selected) >= sum((6, 5, 7)):
                break
            if str(item["doc_id"]) not in refs:
                add(str(item["doc_id"]))
                if len(selected) >= (6 if lo == 1 else 11 if lo == 21 else 18):
                    break
    false_old = sorted((doc for doc in stage1[qid][:35] if doc not in refs), key=lambda doc: (-float(sidecar[(qid, doc)]), _numeric_key(doc)))
    for docid in false_old:
        add(docid)
        if len(selected) >= 20:
            break
    for item in dense_rows[:500]:
        add(str(item["doc_id"]))
        if len(selected) >= 20:
            break
    return selected[:20]


def _prepare_manifests(args: argparse.Namespace, questions: Mapping[str, Mapping[str, Any]], dense: Mapping[str, list[dict[str, Any]]], stage1: Mapping[str, list[str]], sidecar: Mapping[tuple[str, str], float]) -> dict[str, Any]:
    print(json.dumps({"status": "sampling_hierarchical_manifest"}), flush=True)
    texts, chunk_doc_ids, doc_counts = _sample_manifest(args.hierarchical_manifest, 8)
    chunk_vectors = np.load(args.dense_chunk_embeddings, mmap_mode="r")
    if chunk_vectors.shape[0] != len(texts):
        raise ValueError(f"cached chunk embedding count mismatch {chunk_vectors.shape[0]} != {len(texts)}")
    doc_indices: dict[str, list[int]] = {}
    for index, docid in enumerate(chunk_doc_ids):
        doc_indices.setdefault(str(docid), []).append(index)
    model = SentenceTransformer(str(args.model.resolve()), device=args.device, local_files_only=True)
    model.max_seq_length = 128
    ordered_qids = sorted(questions, key=_numeric_key)
    query_vectors = np.asarray(model.encode([questions[qid]["query"] for qid in ordered_qids], batch_size=args.query_batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)
    query_vec_by_qid = {qid: query_vectors[index] for index, qid in enumerate(ordered_qids)}
    del model
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    train_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    for qid in ordered_qids:
        question = questions[qid]
        qvec = query_vec_by_qid[qid]
        qtokens = _tokens(question["query"])
        refs = set(question["references"])
        positive_docs = []
        for docid in sorted(refs, key=_numeric_key):
            indices = doc_indices.get(docid, [])
            if not indices:
                continue
            p1 = _dense_best_chunk(indices, qvec, chunk_vectors)
            p2 = _lexical_best_chunk(indices, qtokens, texts)
            positive_docs.append({"doc_id": docid, "chunks": [_chunk_record(p1, docid, texts), *([] if p2 == p1 else [_chunk_record(p2, docid, texts)])]})
        negative_docs = []
        for docid in _negative_doc_pool(qid, question, dense, sidecar, stage1):
            indices = doc_indices.get(docid, [])
            if indices:
                best = _dense_best_chunk(indices, qvec, chunk_vectors)
                negative_docs.append({"doc_id": docid, "chunk": _chunk_record(best, docid, texts)})
        if _fold(qid) != args.val_fold:
            train_rows.append({"qid": qid, "fold": _fold(qid), "query": question["query"], "positive_docs": positive_docs, "negative_docs": negative_docs})

        if _fold(qid) == args.val_fold:
            pool = list(dense[qid][:500])
            seen = {str(item["doc_id"]) for item in pool}
            for docid in sorted(refs, key=_numeric_key):
                if docid not in seen:
                    pool.append({"doc_id": docid, "rank": 1001, "score": float("-inf")})
            candidates = []
            for item in pool:
                docid = str(item["doc_id"])
                keep = 4 if docid in refs else 2
                chunks = _select_chunks(docid, keep, qvec, chunk_vectors, texts, doc_indices)
                if chunks:
                    candidates.append({"doc_id": docid, "dense_rank": int(item["rank"]), "dense_score": float(item["score"]), "chunks": chunks})
            tracked = sorted((docid for docid in refs if docid not in set(stage1[qid][:120])), key=_numeric_key)
            probe_rows.append({"qid": qid, "fold": _fold(qid), "query": question["query"], "references": sorted(refs, key=_numeric_key), "tracked_v29_gold": tracked, "candidates": candidates, "hard_negative_doc_ids": [row["doc_id"] for row in negative_docs]})
    out = args.output_dir
    write_jsonl(out / "v30_a0_train_manifest.jsonl", train_rows)
    write_jsonl(out / "v30_a0_probe_manifest.jsonl", probe_rows)
    prep = {
        "kind": "legalir_v30_a0_probe_prepare",
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "fold": "qid % 5",
        "validation_fold": args.val_fold,
        "model": "BAAI/bge-m3",
        "model_path": str(args.model.resolve()),
        "sampled_chunks": len(texts),
        "documents": len(doc_counts),
        "train_queries": len(train_rows),
        "validation_queries": len(probe_rows),
        "train_group_size": 8,
        "hard_negative_pool_target": 20,
        "probe_candidate_docs": int(np.mean([len(row["candidates"]) for row in probe_rows])) if probe_rows else 0,
        "tracked_v29_gold": sum(len(row["tracked_v29_gold"]) for row in probe_rows),
        "manifest_sha256": {"train": sha256_file(out / "v30_a0_train_manifest.jsonl"), "probe": sha256_file(out / "v30_a0_probe_manifest.jsonl")},
    }
    write_json(out / "v30_a0_prepare_report.json", prep)
    print(json.dumps(prep, ensure_ascii=False), flush=True)
    return {"train_rows": train_rows, "probe_rows": probe_rows, "prep": prep}


def _fold(qid: str) -> int:
    return int(qid) % 5


class LoRAEncoder:
    def __init__(self, model_path: Path, device: str, query_max_len: int = 96, passage_max_len: int = 256, adapter_path: Path | None = None):
        self.device = torch.device(device)
        self.query_max_len = query_max_len
        self.passage_max_len = passage_max_len
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path.resolve()), local_files_only=True)
        base = AutoModel.from_pretrained(str(model_path.resolve()), local_files_only=True, torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32)
        if adapter_path is None:
            config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["query", "value"], bias="none", task_type=TaskType.FEATURE_EXTRACTION)
            self.model = get_peft_model(base, config)
        else:
            self.model = PeftModel.from_pretrained(base, str(adapter_path.resolve()), is_trainable=True)
        self.model.to(self.device)
        self.model.train()
        trainable = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in self.model.parameters())
        if trainable <= 0:
            raise RuntimeError("LoRA attached zero trainable parameters")
        self.stats = {"trainable_params": trainable, "total_params": total, "trainable_percent": 100.0 * trainable / total}

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def _forward(self, texts: list[str], max_length: int, train: bool) -> torch.Tensor:
        encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        encoded = {key: value.to(self.device, non_blocking=True) for key, value in encoded.items()}
        context = torch.enable_grad() if train else torch.inference_mode()
        with context:
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    hidden = self.model(**encoded, return_dict=True).last_hidden_state[:, 0]
            else:
                hidden = self.model(**encoded, return_dict=True).last_hidden_state[:, 0]
            return F.normalize(hidden.float(), p=2, dim=-1)

    def group_loss(self, query: str, passages: list[str], temperature: float = 0.02) -> torch.Tensor:
        self.model.train()
        qvec = self._forward([query], self.query_max_len, True)
        pvec = self._forward(passages, self.passage_max_len, True)
        logits = (qvec @ pvec.T) / temperature
        return F.cross_entropy(logits, torch.zeros(1, dtype=torch.long, device=self.device))

    def encode(self, texts: list[str], max_length: int, batch_size: int) -> np.ndarray:
        self.model.eval()
        values = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                values.append(self._forward(texts[start : start + batch_size], max_length, False).cpu().numpy())
        return np.concatenate(values, axis=0) if values else np.zeros((0, 1024), dtype=np.float32)


def _sample_group(row: Mapping[str, Any], rng: random.Random) -> tuple[str, list[str]]:
    positives = [chunk["text"] for doc in row["positive_docs"] for chunk in doc["chunks"]]
    negatives = [doc["chunk"]["text"] for doc in row["negative_docs"]]
    if not positives or not negatives:
        raise ValueError(f"{row['qid']}: insufficient positive/negative chunks")
    positive = rng.choice(positives)
    sampled = rng.sample(negatives, 7) if len(negatives) >= 7 else [rng.choice(negatives) for _ in range(7)]
    return positive, sampled


def _probe_eval(encoder: LoRAEncoder | None, probe_rows: list[dict[str, Any]], encode_batch_size: int, baseline_only: bool = False) -> dict[str, Any]:
    unique_texts: list[str] = []
    text_index: dict[str, int] = {}
    query_texts = [row["query"] for row in probe_rows]
    doc_chunk_indices: list[list[list[int]]] = []
    for row in probe_rows:
        q_indices = []
        for candidate in row["candidates"]:
            indices = []
            for chunk in candidate["chunks"]:
                text = str(chunk["text"])
                if text not in text_index:
                    text_index[text] = len(unique_texts)
                    unique_texts.append(text)
                indices.append(text_index[text])
            q_indices.append(indices)
        doc_chunk_indices.append(q_indices)
    if not baseline_only and encoder is None:
        raise ValueError("encoder is required for non-baseline probe evaluation")
    qvec = None if baseline_only else encoder.encode(query_texts, 96, max(1, min(encode_batch_size, 64)))
    cvec = None if baseline_only else encoder.encode(unique_texts, 256, encode_batch_size)
    gold_ranks = []
    baseline_gold_ranks = []
    recalls = {20: 0, 50: 0, 100: 0, 500: 0}
    baseline_recalls = {20: 0, 50: 0, 100: 0, 500: 0}
    hard_pairs = []
    tracked = []
    lost_top50 = 0
    gained_top50 = 0
    new_gold_at_100 = 0
    baseline_gold_at_100 = 0
    total_gold_docs = 0
    query_summaries = []
    transitions = {
        "base_1_10": {"new_1_10": 0, "new_11_50": 0, "new_gt50": 0},
        "base_11_50": {"new_1_10": 0, "new_11_50": 0, "new_gt50": 0},
        "base_51_100": {"new_le50": 0, "new_gt50": 0},
        "base_101_500": {"new_le100": 0, "new_gt100": 0},
        "base_gt500": {"new_le100": 0, "new_gt100": 0},
    }
    for qi, row in enumerate(probe_rows):
        candidates = row["candidates"]
        baseline_order = sorted(range(len(candidates)), key=lambda i: (-float(candidates[i]["dense_score"]), int(candidates[i]["dense_rank"]), _numeric_key(str(candidates[i]["doc_id"]))))
        fine_scores = [float(candidates[i]["dense_score"]) for i in range(len(candidates))] if baseline_only else [float(np.max(qvec[qi] @ cvec[indices].T)) if indices else float("-inf") for indices in doc_chunk_indices[qi]]
        fine_order = baseline_order if baseline_only else sorted(range(len(candidates)), key=lambda i: (-fine_scores[i], int(candidates[i]["dense_rank"]), _numeric_key(str(candidates[i]["doc_id"]))))
        fine_rank = {str(candidates[i]["doc_id"]): rank for rank, i in enumerate(fine_order, 1)}
        baseline_rank = {str(candidates[i]["doc_id"]): rank for rank, i in enumerate(baseline_order, 1)}
        refs = set(row["references"])
        ref_ranks = [fine_rank[doc] for doc in refs if doc in fine_rank]
        base_ref_ranks = [baseline_rank[doc] for doc in refs if doc in baseline_rank]
        for doc in refs:
            old_rank = baseline_rank.get(doc, 1001)
            new_rank = fine_rank.get(doc, 1001)
            if old_rank <= 10:
                bucket = transitions["base_1_10"]
                bucket["new_1_10" if new_rank <= 10 else "new_11_50" if new_rank <= 50 else "new_gt50"] += 1
            elif old_rank <= 50:
                bucket = transitions["base_11_50"]
                bucket["new_1_10" if new_rank <= 10 else "new_11_50" if new_rank <= 50 else "new_gt50"] += 1
            elif old_rank <= 100:
                transitions["base_51_100"]["new_le50" if new_rank <= 50 else "new_gt50"] += 1
            elif old_rank <= 500:
                transitions["base_101_500"]["new_le100" if new_rank <= 100 else "new_gt100"] += 1
            else:
                transitions["base_gt500"]["new_le100" if new_rank <= 100 else "new_gt100"] += 1
        best_rank = min(ref_ranks, default=1001)
        base_best = min(base_ref_ranks, default=1001)
        gold_ranks.append(best_rank)
        baseline_gold_ranks.append(base_best)
        total_gold_docs += len(refs)
        for cutoff in recalls:
            recalls[cutoff] += int(any(rank <= cutoff for rank in ref_ranks))
            baseline_recalls[cutoff] += int(any(rank <= cutoff for rank in base_ref_ranks))
        gold_indices = [i for i, candidate in enumerate(candidates) if str(candidate["doc_id"]) in refs]
        neg_indices = [i for i, candidate in enumerate(candidates) if str(candidate["doc_id"]) not in refs]
        for gi in gold_indices:
            for ni in neg_indices[:20]:
                hard_pairs.append(float(fine_scores[gi] > fine_scores[ni]))
        if row["tracked_v29_gold"]:
            for docid in row["tracked_v29_gold"]:
                tracked.append({"qid": row["qid"], "doc_id": docid, "old_dense_rank": baseline_rank.get(docid, 1001), "new_probe_rank": fine_rank.get(docid, 1001), "rank_delta": fine_rank.get(docid, 1001) - baseline_rank.get(docid, 1001)})
        base_top50 = any(baseline_rank.get(doc, 1001) <= 50 for doc in refs)
        new_top50 = any(fine_rank.get(doc, 1001) <= 50 for doc in refs)
        lost_top50 += int(base_top50 and not new_top50)
        gained_top50 += int(not base_top50 and new_top50)
        query_summaries.append({"qid": str(row["qid"]), "baseline_best_rank": int(base_best), "new_best_rank": int(best_rank), "baseline_top50": bool(base_top50), "new_top50": bool(new_top50)})
        baseline_gold_at_100 += sum(1 for doc in refs if baseline_rank.get(doc, 1001) <= 100)
        new_gold_at_100 += sum(1 for doc in refs if fine_rank.get(doc, 1001) <= 100)
    fine_ranks = np.asarray(gold_ranks, dtype=np.float64)
    base_ranks = np.asarray(baseline_gold_ranks, dtype=np.float64)
    return {
        "queries": len(probe_rows),
        "baseline_recall_at": {str(k): baseline_recalls[k] / max(len(probe_rows), 1) for k in baseline_recalls},
        "recall_at": {str(k): recalls[k] / max(len(probe_rows), 1) for k in recalls},
        "baseline_gold_docs_at_100": baseline_gold_at_100,
        "gold_docs_at_100": new_gold_at_100,
        "gold_docs_at_100_gain": new_gold_at_100 - baseline_gold_at_100,
        "baseline_median_gold_rank": float(np.median(base_ranks)),
        "median_gold_rank": float(np.median(fine_ranks)),
        "baseline_mean_log_gold_rank": float(np.mean(np.log1p(base_ranks))),
        "mean_log_gold_rank": float(np.mean(np.log1p(fine_ranks))),
        "hard_negative_win_rate": float(np.mean(hard_pairs)) if hard_pairs else 0.0,
        "hard_negative_pairs": len(hard_pairs),
        "lost_top50": lost_top50,
        "gained_top50": gained_top50,
        "rank_transition_matrix": transitions,
        "query_summaries": query_summaries,
        "tracked_v29_gold": tracked,
    }


def _load_encoder(args: argparse.Namespace, adapter_path: Path | None = None) -> LoRAEncoder:
    return LoRAEncoder(args.model, args.device, 96, 256, adapter_path=adapter_path)


def _benchmark(args: argparse.Namespace, train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    encoder = _load_encoder(args)
    optimizer = torch.optim.AdamW([p for p in encoder.model.parameters() if p.requires_grad], lr=1e-4)
    rng = random.Random(SEED)
    started = time.perf_counter()
    peak_before = torch.cuda.max_memory_allocated(torch.device(args.device)) if torch.cuda.is_available() and args.device.startswith("cuda") else 0
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for step in range(50):
        row = train_rows[step % len(train_rows)]
        positive, negatives = _sample_group(row, rng)
        loss = encoder.group_loss(row["query"], [positive, *negatives])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(args.device))
        peak = torch.cuda.max_memory_allocated(torch.device(args.device)) / 1024**3
    else:
        peak = peak_before / 1024**3
    elapsed = time.perf_counter() - started
    del optimizer, encoder
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {"benchmark_steps": 50, "groups_per_second": 50 / max(elapsed, 1e-9), "passages_per_second": 50 * 8 / max(elapsed, 1e-9), "seconds_per_step": elapsed / 50, "peak_vram_gb": peak, "mean_loss": float(np.mean(losses))}


def _gate(metrics: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    median_ratio = float(metrics["median_gold_rank"]) / max(float(baseline["median_gold_rank"]), 1e-9)
    win_gain = float(metrics["hard_negative_win_rate"]) - float(baseline["hard_negative_win_rate"])
    top100_gain = int(metrics["gold_docs_at_100"]) - int(baseline["gold_docs_at_100"])
    return {"median_rank_ratio_le_0_70": median_ratio <= 0.70, "hard_negative_win_gain_ge_0_08": win_gain >= 0.08, "gold_docs_at_100_gain_ge_5": top100_gain >= 5, "lost_top50_le_2": int(metrics["lost_top50"]) <= 2, "median_rank_ratio": median_ratio, "hard_negative_win_gain": win_gain, "gold_docs_at_100_gain": top100_gain, "pass": bool(median_ratio <= 0.70 and win_gain >= 0.08 and top100_gain >= 5 and int(metrics["lost_top50"]) <= 2)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--dense-pools", type=Path, required=True)
    parser.add_argument("--stage1-audit", type=Path, required=True)
    parser.add_argument("--hierarchical-manifest", type=Path, required=True)
    parser.add_argument("--dense-chunk-embeddings", type=Path, required=True)
    parser.add_argument("--stage2-sidecar", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--reuse-manifests", action="store_true")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--resume-epoch", type=int, default=1)
    parser.add_argument("--resume-group", type=int, default=0)
    parser.add_argument("--resume-metrics", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    random.seed(SEED)
    train_manifest = args.output_dir / "v30_a0_train_manifest.jsonl"
    probe_manifest = args.output_dir / "v30_a0_probe_manifest.jsonl"
    prepare_report = args.output_dir / "v30_a0_prepare_report.json"
    if args.reuse_manifests and train_manifest.exists() and probe_manifest.exists() and prepare_report.exists():
        print(json.dumps({"status": "reusing_manifests", "train": str(train_manifest), "probe": str(probe_manifest)}), flush=True)
        prepared = {"train_rows": _read_jsonl(train_manifest), "probe_rows": _read_jsonl(probe_manifest), "prep": read_json(prepare_report)}
    else:
        questions = load_metadata(args.metadata)
        dense = _load_dense(args.dense_pools, set(questions))
        stage1 = _load_stage1(args.stage1_audit, questions)
        sidecar = load_sidecar(args.stage2_sidecar, questions)
        prepared = _prepare_manifests(args, questions, dense, stage1, sidecar)
    train_rows, probe_rows = prepared["train_rows"], prepared["probe_rows"]
    benchmark = {"skipped": True} if args.skip_benchmark else _benchmark(args, train_rows)
    baseline = _probe_eval(None, probe_rows, args.encode_batch_size, baseline_only=True)
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    encoder = _load_encoder(args, args.resume_from)
    optimizer = torch.optim.AdamW([p for p in encoder.model.parameters() if p.requires_grad], lr=1e-4)
    rng = random.Random(SEED)
    accumulation = 8
    epochs = 2
    total_groups = len(train_rows)
    evaluations: list[dict[str, Any]] = []
    best_score = None
    best_path = None
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    skipped_train_rows = 0
    resume_epoch = max(1, int(args.resume_epoch)) if args.resume_from else 1
    resume_group = max(0, int(args.resume_group)) if args.resume_from else 0
    global_group = resume_group if args.resume_from and resume_epoch == 1 else 0
    if args.resume_from:
        resume_metrics = read_json(args.resume_metrics) if args.resume_metrics else _probe_eval(encoder, probe_rows, args.encode_batch_size)
        evaluations.append({"epoch": resume_epoch, "group": resume_group, "global_group": global_group, "resumed_checkpoint": str(args.resume_from.resolve()), "elapsed_seconds": 0.0, **resume_metrics})
        best_score = (float(resume_metrics["gold_docs_at_100"]), -float(resume_metrics["median_gold_rank"]), float(resume_metrics["hard_negative_win_rate"]))
        best_path = args.resume_from
        print(json.dumps({"status": "resumed", "checkpoint": str(args.resume_from.resolve()), "epoch": resume_epoch, "group": resume_group, "metrics": resume_metrics}, ensure_ascii=False), flush=True)
    for epoch in range(resume_epoch, epochs + 1):
        order = list(range(total_groups))
        random.Random(SEED + epoch).shuffle(order)
        events = set()
        if epoch == 1:
            events.update(max(1, math.ceil(total_groups * fraction)) for fraction in (0.25, 0.50, 0.75, 1.00))
        else:
            events.update(max(1, math.ceil(total_groups * fraction)) for fraction in (0.50, 1.00))
        for local_group, index in enumerate(order, 1):
            row = train_rows[index]
            if args.resume_from and epoch == resume_epoch and local_group <= resume_group:
                if row["positive_docs"] and row["negative_docs"]:
                    _sample_group(row, rng)
                continue
            try:
                positive, negatives = _sample_group(row, rng)
            except ValueError:
                skipped_train_rows += 1
                continue
            loss = encoder.group_loss(row["query"], [positive, *negatives]) / accumulation
            loss.backward()
            if local_group % accumulation == 0 or local_group == total_groups:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            global_group += 1
            if local_group in events:
                eval_metrics = _probe_eval(encoder, probe_rows, args.encode_batch_size)
                checkpoint = {"epoch": epoch, "group": local_group, "global_group": global_group, "elapsed_seconds": time.perf_counter() - started, **eval_metrics}
                evaluations.append(checkpoint)
                score = (float(eval_metrics["gold_docs_at_100"]), -float(eval_metrics["median_gold_rank"]), float(eval_metrics["hard_negative_win_rate"]))
                if best_score is None or score > best_score:
                    best_score = score
                    best_path = args.output_dir / "checkpoints" / "best"
                    encoder.save(best_path)
        encoder.save(args.output_dir / "checkpoints" / f"epoch{epoch}_last")
    final = evaluations[-1] if evaluations else _probe_eval(encoder, probe_rows, args.encode_batch_size)
    report = {
        "kind": "legalir_v30_a0_lora_bge_m3_screen",
        "schema_version": 1,
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "model": "BAAI/bge-m3",
        "model_path": str(args.model.resolve()),
        "fold": "qid % 5",
        "validation_fold": args.val_fold,
        "config": {"lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.05, "target_modules": ["query", "value"], "query_max_len": 96, "passage_max_len": 256, "temperature": 0.02, "group_size": 8, "gradient_accumulation": accumulation, "learning_rate": 1e-4, "epochs": epochs, "gradient_checkpointing": False, "trainable_stats": encoder.stats},
        "benchmark": benchmark,
        "baseline_probe": baseline,
        "evaluations": evaluations,
        "final_probe": final,
        "gate": _gate(final, baseline),
        "best_checkpoint": None if best_path is None else str(best_path.resolve()),
        "resumed_from": None if args.resume_from is None else str(args.resume_from.resolve()),
        "skipped_train_rows": skipped_train_rows,
        "artifacts": {"train_manifest": str((args.output_dir / "v30_a0_train_manifest.jsonl").resolve()), "probe_manifest": str((args.output_dir / "v30_a0_probe_manifest.jsonl").resolve())},
        "elapsed_training_seconds": time.perf_counter() - started,
    }
    write_json(args.output_dir / "v30_a0_report.json", report)
    print(json.dumps({"report": str((args.output_dir / "v30_a0_report.json").resolve()), "benchmark": benchmark, "baseline": baseline, "final": final, "gate": report["gate"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
