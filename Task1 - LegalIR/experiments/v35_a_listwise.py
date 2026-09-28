"""V35-A small listwise Qwen screen with reversed-order bias audit."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (REPO_ROOT / "scripts", SCRIPT_DIR / "v27"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from batch_stage2_rerank import hydrate, load_passage_store, open_readonly  # noqa: E402
from v27_common import load_metadata, sha256_file, write_json, write_jsonl  # noqa: E402


def _load(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _num(value: str) -> tuple[int, str]:
    return (int(value), value) if str(value).isdigit() else (10**18, str(value))


def _choice(text: str, labels: set[str]) -> str | None:
    match = re.search(r"\b(C[0-4])\b", str(text).upper())
    return match.group(1) if match and match.group(1) in labels else None


def _prompt(query: str, context: str, candidates: list[dict[str, Any]]) -> str:
    body = "\n\n".join(f"[{item['label']}]\n{item['text'][:900]}" for item in candidates)
    return f"""Return only one candidate identifier: C0, C1, C2, C3, or C4. Do not explain.

You are selecting the Vietnamese legal provision that most directly adds an applicable legal basis to the current top-4 evidence.

QUESTION:
{query[:1200]}

CURRENT TOP-1..4 EVIDENCE:
{context[:1800]}

CANDIDATES:
{body}

Choose the candidate that matches the legal subject, act, conditions, exceptions, negation, authority, numeric/time limits, and legal consequence. FINAL CANDIDATE:"""


def _generate(model: Any, tokenizer: Any, prompts: list[str], device: str, batch_size: int, max_input_tokens: int) -> list[str]:
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    outputs = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        rendered = []
        for prompt in batch:
            messages = [{"role": "system", "content": "You are a precise legal retrieval ranker. Obey the output format exactly."}, {"role": "user", "content": prompt}]
            try:
                rendered.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))
            except TypeError:
                rendered.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        encoded = tokenizer(rendered, return_tensors="pt", padding=True, truncation=True, max_length=max_input_tokens).to(device)
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=8, do_sample=False, temperature=None, pad_token_id=tokenizer.eos_token_id)
        prompt_len = encoded["input_ids"].shape[1]
        outputs.extend(tokenizer.decode(row[prompt_len:], skip_special_tokens=True).strip() for row in generated)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--hard-benchmark", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    questions = load_metadata(args.metadata)
    views = {str(row["qid"]): row for row in _load(args.views)}
    hard = _load(args.hard_benchmark)
    qids = sorted({str(row["qid"]) for row in hard}, key=_num)
    docs_needed: set[str] = set()
    sets: dict[str, list[dict[str, Any]]] = {}
    for qid in qids:
        anchor = [str(x) for x in questions[qid]["anchor"]]
        docs: list[dict[str, Any]] = [{"label": "C0", "doc_id": anchor[4], "source": "incumbent"}]
        used = {anchor[4]}
        for label, view in (("C1", "Q0"), ("C2", "Q1"), ("C3", "Q2")):
            item = next((x for x in views[qid][view][:500] if str(x["doc_id"]) not in set(anchor[:4]) and str(x["doc_id"]) not in used), None)
            if item is not None:
                doc = str(item["doc_id"])
                docs.append({"label": label, "doc_id": doc, "source": view, "dense_rank": int(item["rank"])})
                used.add(doc)
        item = next((x for x in views[qid]["Q0"][1:500] if str(x["doc_id"]) not in set(anchor[:4]) and str(x["doc_id"]) not in used), None)
        if item is not None:
            doc = str(item["doc_id"])
            docs.append({"label": "C4", "doc_id": doc, "source": "dense_extra", "dense_rank": int(item["rank"])})
            used.add(doc)
        for doc in docs:
            docs_needed.add(doc["doc_id"])
        for item in views[qid]["Q0"][:4]:
            docs_needed.add(str(item["doc_id"]))
        sets[qid] = docs
    connection = open_readonly(args.index)
    try:
        store = load_passage_store(connection, sorted(docs_needed, key=_num))
    finally:
        connection.close()
    prompt_rows = []
    prompts = []
    for qid in qids:
        query = str(questions[qid]["query"])
        context = "\n\n".join(hydrate(query, [str(item["doc_id"])], 1, store)[0]["text"][:450] for item in views[qid]["Q0"][:4])
        original = [{**item, "text": hydrate(query, [str(item["doc_id"])], 1, store)[0]["text"]} for item in sets[qid]]
        reversed_set = list(reversed(original))
        for order_name, candidate_set in (("original", original), ("reversed", reversed_set)):
            prompts.append(_prompt(query, context, candidate_set))
            prompt_rows.append({"qid": qid, "order": order_name, "candidate_docs": candidate_set, "labels": {str(x["label"]): str(x["doc_id"]) for x in candidate_set}, "gold_docs": sorted(set(questions[qid]["references"]) & {str(x["doc_id"]) for x in candidate_set}, key=_num)})
    device = args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(str(args.model.resolve()), local_files_only=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(str(args.model.resolve()), local_files_only=True, torch_dtype=dtype).to(device).eval()
    outputs = _generate(model, tokenizer, prompts, device, args.batch_size, args.max_input_tokens)
    rows = []
    for item, output in zip(prompt_rows, outputs, strict=True):
        labels = set(item["labels"])
        choice = _choice(output, labels)
        chosen_doc = item["labels"].get(choice) if choice else None
        rows.append({**item, "output": output, "choice": choice, "chosen_doc": chosen_doc, "chosen_is_gold": bool(chosen_doc in item["gold_docs"] if chosen_doc else False), "gold_in_set": bool(item["gold_docs"])})
    write_jsonl(args.output_dir / "listwise_scores.jsonl", rows)
    by_qid: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_qid.setdefault(str(row["qid"]), {})[str(row["order"])] = row
    paired = [pair for pair in by_qid.values() if "original" in pair and "reversed" in pair]
    gold_available = [pair for pair in paired if pair["original"]["gold_in_set"]]
    same_choice = [pair for pair in gold_available if pair["original"]["choice"] is not None and pair["original"]["choice"] == pair["reversed"]["choice"]]
    consistent_gold = [pair for pair in same_choice if pair["original"]["chosen_is_gold"]]
    chosen_gold = sum(bool(pair["original"]["chosen_is_gold"]) for pair in gold_available if pair["original"]["choice"] is not None)
    report = {"kind": "legalir_v35_a_listwise_screen", "schema_version": 1, "model": str(args.model.resolve()), "model_size_fallback": "Qwen3-1.7B used because Qwen3-0.6B base was not local", "queries": len(qids), "prompt_runs": len(rows), "gold_available_queries": len(gold_available), "parseable_runs": sum(row["choice"] is not None for row in rows), "position_consistent_gold_available": len(same_choice), "position_consistent_coverage": len(same_choice) / max(len(gold_available), 1), "gold_selected_among_available": chosen_gold, "gold_selection_rate": chosen_gold / max(len(gold_available), 1), "gold_selected_among_consistent": len(consistent_gold), "gold_precision_consistent": len(consistent_gold) / max(len(same_choice), 1), "position_sensitive_available": len(gold_available) - len(same_choice), "gate": {"gold_selection_ge_0_60": chosen_gold / max(len(gold_available), 1) >= 0.60, "consistent_coverage_ge_0_60": len(same_choice) / max(len(gold_available), 1) >= 0.60, "gold_precision_consistent_ge_0_75": len(consistent_gold) / max(len(same_choice), 1) >= 0.75, "pass_listwise_screen": chosen_gold / max(len(gold_available), 1) >= 0.60 and len(same_choice) / max(len(gold_available), 1) >= 0.60 and len(consistent_gold) / max(len(same_choice), 1) >= 0.75}, "inputs": {"metadata": str(args.metadata.resolve()), "metadata_sha256": sha256_file(args.metadata), "views": str(args.views.resolve()), "views_sha256": sha256_file(args.views), "hard_benchmark": str(args.hard_benchmark.resolve()), "hard_benchmark_sha256": sha256_file(args.hard_benchmark)}, "output": str((args.output_dir / "listwise_scores.jsonl").resolve()), "elapsed_seconds": time.perf_counter() - started}
    write_json(args.output_dir / "listwise_report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
