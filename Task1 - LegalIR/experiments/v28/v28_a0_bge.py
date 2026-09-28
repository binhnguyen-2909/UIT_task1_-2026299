"""V28-A0 zero-shot boundary test for BAAI/bge-reranker-v2-m3."""

from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from v27_common import load_metadata, read_json, sha256_file, write_json, write_jsonl


ELEMENT_PATTERNS = {
    "SUBJECT": ("đối tượng", "chủ thể", "người nào", "ai được", "ai có", "cơ quan", "tổ chức", "cá nhân", "pháp nhân"),
    "CONDITION": ("điều kiện", "trường hợp", "khi nào", "nếu", "được cấp", "được hưởng", "có được"),
    "EXCEPTION": ("ngoại lệ", "trừ", "không áp dụng", "ngoài", "ngoại trừ"),
    "PENALTY": ("mức phạt", "phạt", "xử phạt", "tiền phạt", "tước", "buộc", "chế tài"),
    "TIME": ("bao lâu", "thời hạn", "khi nào", "trong thời hạn", "kể từ", "sau bao"),
    "AUTHORITY": ("thẩm quyền", "cơ quan nào", "ai có quyền", "thẩm phán", "sở ", "bộ ", "ủy ban", "tòa án"),
    "ARTICLE_COLLISION": ("điều", "khoản", "nghị định", "thông tư", "luật", "quy định"),
    "LEXICAL_COLLISION": ("gồm", "bao gồm", "những gì", "như thế nào", "quy trình", "thủ tục"),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def error_types(query: str) -> list[str]:
    q = " ".join(str(query).casefold().split())
    hits = [name for name, patterns in ELEMENT_PATTERNS.items() if any(pattern in q for pattern in patterns)]
    return hits or ["OTHER"]


def load_packs(path: Path) -> dict[tuple[str, str], str]:
    result = {}
    for row in read_jsonl(path):
        result[(str(row["qid"]), str(row["doc_id"]))] = str(row.get("pack", ""))
    return result


def batched(values: list[Any], size: int):
    for start in range(0, len(values), size):
        yield start, values[start : start + size]


def score_pairs(model, tokenizer, pair_rows: list[dict[str, Any]], packs: dict[tuple[str, str], str], *, batch_size: int, max_length: int, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    model_pairs = []
    for row in pair_rows:
        model_pairs.append((row["query"], packs[(str(row["qid"]), str(row["incumbent_doc"]))], "incumbent"))
        model_pairs.append((row["query"], packs[(str(row["qid"]), str(row["challenger_doc"]))], "challenger"))
    logits: list[float] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for _, batch in batched(model_pairs, batch_size):
            queries = [item[0] for item in batch]
            docs = [item[1] for item in batch]
            inputs = tokenizer(queries, docs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                raw = model(**inputs).logits.reshape(-1).float().detach().cpu().numpy().tolist()
            logits.extend(float(value) for value in raw)
    scored = []
    for index, row in enumerate(pair_rows):
        incumbent = logits[2 * index]
        challenger = logits[2 * index + 1]
        scored.append({
            **row,
            "incumbent_logit": incumbent,
            "challenger_logit": challenger,
            "margin": challenger - incumbent,
            "model_wins": challenger > incumbent,
            "error_types": error_types(row["query"]),
        })
    elapsed = time.perf_counter() - started
    return scored, {"model_pairs": len(model_pairs), "pair_rows": len(pair_rows), "elapsed_seconds": elapsed, "pairs_per_second": len(model_pairs) / max(elapsed, 1e-9)}


def aggregate(scored: list[dict[str, Any]], questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    gold_rescue = [row for row in scored if row["challenger_is_raw_gold"] and row.get("baseline_miss", True) and not row["incumbent_is_raw_gold"]]
    all_rows = [row for row in scored if not row["incumbent_is_raw_gold"]]
    pairwise = float(np.mean([row["model_wins"] for row in gold_rescue])) if gold_rescue else 0.0
    margins = [float(row["margin"]) for row in gold_rescue]
    labels = [int(row["challenger_is_raw_gold"]) for row in all_rows]
    values = [float(row["margin"]) for row in all_rows]
    auc = float(roc_auc_score(labels, values)) if len(set(labels)) > 1 else 0.0
    by_type = {}
    for kind in sorted({kind for row in gold_rescue for kind in row["error_types"]}):
        subset = [row for row in gold_rescue if kind in row["error_types"]]
        by_type[kind] = {
            "pairs": len(subset),
            "pairwise_win_rate": float(np.mean([row["model_wins"] for row in subset])) if subset else 0.0,
            "mean_margin": float(np.mean([row["margin"] for row in subset])) if subset else 0.0,
        }
    # Boundary MRR uses the incumbent plus all scored challengers per qid.  It
    # is intentionally a diagnostic ranking metric, not a submission policy.
    per_qid = []
    for qid in sorted({str(row["qid"]) for row in gold_rescue}):
        subset = [row for row in scored if str(row["qid"]) == qid and not row["incumbent_is_raw_gold"]]
        gold_docs = {str(row["challenger_doc"]) for row in subset if row["challenger_is_raw_gold"]}
        candidates = [(str(row["challenger_doc"]), float(row["challenger_logit"])) for row in subset]
        incumbent = subset[0]
        candidates.append((str(incumbent["incumbent_doc"]), float(incumbent["incumbent_logit"])))
        candidates.sort(key=lambda item: (-item[1], item[0]))
        rank = next((index + 1 for index, (doc, _) in enumerate(candidates) if doc in gold_docs), None)
        if rank is not None:
            per_qid.append(1.0 / rank)
    return {
        "boundary_pairs": len(scored),
        "gold_rescue_pairs": len(gold_rescue),
        "pairwise_win_rate": pairwise,
        "mean_positive_negative_margin": float(np.mean(margins)) if margins else 0.0,
        "boundary_auc": auc,
        "boundary_mrr": float(np.mean(per_qid)) if per_qid else 0.0,
        "gold_rescue_qids": len({str(row["qid"]) for row in gold_rescue}),
        "by_error_type": by_type,
    }


def oracle_swap(scored: list[dict[str, Any]], questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_qid: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in scored:
        by_qid[str(row["qid"])].append(row)
    details = []
    for qid, question in questions.items():
        rows = by_qid.get(str(qid), [])
        if not rows:
            continue
        incumbent = rows[0]
        best = max(rows, key=lambda row: (float(row["challenger_logit"]), -int(row["challenger_rank"]), str(row["challenger_doc"])))
        answer = list(question["anchor"][:4]) + [str(best["challenger_doc"])] if float(best["challenger_logit"]) > float(incumbent["incumbent_logit"]) else list(question["anchor"])
        refs = set(map(str, question["references"]))
        before = len(set(question["anchor"]) & refs) / max(len(refs), 1)
        after = len(set(answer) & refs) / max(len(refs), 1)
        details.append({"qid": qid, "incumbent": question["anchor"][4], "selected": str(best["challenger_doc"]), "selected_is_gold": bool(best["challenger_is_raw_gold"]), "swapped": answer != question["anchor"], "baseline_recall": before, "oracle_recall": after, "delta": after - before})
    deltas = [float(row["delta"]) for row in details]
    return {
        "queries_scored": len(details),
        "swaps": sum(bool(row["swapped"]) for row in details),
        "rescues": sum(delta > 1e-12 for delta in deltas),
        "destructive": sum(delta < -1e-12 for delta in deltas),
        "baseline_recall": float(np.mean([row["baseline_recall"] for row in details])) if details else 0.0,
        "oracle_recall": float(np.mean([row["oracle_recall"] for row in details])) if details else 0.0,
        "oracle_gain": float(np.mean(deltas)) if deltas else 0.0,
        "net_query_equivalent": sum(delta > 1e-12 for delta in deltas) - sum(delta < -1e-12 for delta in deltas),
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--boundary-pairs", type=Path, required=True)
    parser.add_argument("--e3b-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    questions = load_metadata(args.metadata)
    pair_rows = read_jsonl(args.boundary_pairs)
    packs = load_packs(args.e3b_manifest)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=True)
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    scored, efficiency = score_pairs(model, tokenizer, pair_rows, packs, batch_size=args.batch_size, max_length=args.max_length, device=device)
    if device.type == "cuda":
        efficiency["peak_memory_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2
        efficiency["peak_memory_reserved_mb"] = torch.cuda.max_memory_reserved(device) / 1024**2
    efficiency.update({"device": str(device), "model": str(args.model.resolve()), "max_length": args.max_length, "batch_size": args.batch_size, "model_config_sha256": sha256_file(args.model / "config.json")})
    summary = aggregate(scored, questions)
    oracle = oracle_swap(scored, questions)
    payload = {
        "kind": "legalir_v28_a0_bge_zero_shot_boundary_report",
        "schema_version": 1,
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "model": "BAAI/bge-reranker-v2-m3",
        "model_path": str(args.model.resolve()),
        "summary": summary,
        "oracle_swap": {key: value for key, value in oracle.items() if key != "details"},
        "efficiency": efficiency,
        "gate": {
            "pairwise_win_rate_ge_0.72": summary["pairwise_win_rate"] >= 0.72,
            "important_regimes_ge_0.60": all(row["pairwise_win_rate"] >= 0.60 for row in summary["by_error_type"].values()),
            "oracle_gain_ge_0.006": oracle["oracle_gain"] >= 0.006,
            "oracle_gain_ge_0.008": oracle["oracle_gain"] >= 0.008,
            "decision": "pass_to_v28_a1_lora" if summary["pairwise_win_rate"] >= 0.72 and all(row["pairwise_win_rate"] >= 0.60 for row in summary["by_error_type"].values()) and oracle["oracle_gain"] >= 0.006 else "stop_reranker_path_and_repair_e3b_or_candidate_lane",
        },
    }
    write_json(out / "v28_a0_report.json", payload)
    write_jsonl(out / "v28_a0_scores.jsonl", scored)
    write_jsonl(out / "v28_a0_oracle_details.jsonl", oracle["details"])
    write_json(out / "v28_gpu_efficiency.json", efficiency)
    print(json.dumps({"report": str((out / "v28_a0_report.json").resolve()), "summary": summary, "oracle_swap": {key: value for key, value in oracle.items() if key != "details"}, "gate": payload["gate"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
