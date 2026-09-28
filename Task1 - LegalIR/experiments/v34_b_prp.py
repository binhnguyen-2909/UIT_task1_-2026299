"""V34-B direct pairwise ranking prompt with position-bias control."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (REPO_ROOT / "scripts", SCRIPT_DIR / "v27"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from batch_stage2_rerank import hydrate, load_passage_store, open_readonly  # noqa: E402
from v27_common import load_metadata, sha256_file, write_json, write_jsonl  # noqa: E402


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _choice(text: str) -> str | None:
    # Generated output is intentionally constrained to A/B.  Prefer a clean
    # first token, then accept the first standalone choice for robustness.
    clean = str(text).strip().upper()
    if clean.startswith("A") and not clean.startswith("ANSWER"):
        return "A"
    if clean.startswith("B") and not clean.startswith("BECAUSE"):
        return "B"
    match = re.search(r"\b([AB])\b", clean)
    return match.group(1) if match else None


def _prompt(query: str, context: str, a: str, b: str) -> str:
    # Keep the answer instruction visible even after tokenizer truncation.
    context, a, b = str(context)[:2400], str(a)[:1200], str(b)[:1200]
    return f"""Return exactly one character: A or B. Do not explain.

You are comparing two candidate legal provisions for a Vietnamese legal retrieval system.

QUESTION:
{query}

CURRENT TOP-1..4 EVIDENCE:
{context}

CANDIDATE A:
{a}

CANDIDATE B:
{b}

Which candidate is more likely to add a directly applicable legal provision to the current top-4 results? Consider subject, act, condition, exception, negation, authority, numeric/time limits, and legal consequence.

FINAL ANSWER (A or B only):"""


def _generate(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompts: list[str], device: str, batch_size: int, max_input_tokens: int) -> list[str]:
    outputs: list[str] = []
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded_texts = []
        for prompt in batch:
            messages = [{"role": "user", "content": prompt}]
            try:
                encoded_texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))
            except TypeError:
                encoded_texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        encoded = tokenizer(encoded_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_input_tokens).to(device)
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=8, do_sample=False, temperature=None, pad_token_id=tokenizer.eos_token_id)
        prompt_len = encoded["input_ids"].shape[1]
        for row in generated[:, prompt_len:]:
            outputs.append(tokenizer.decode(row, skip_special_tokens=True).strip())
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    questions = load_metadata(args.metadata)
    views = {str(row["qid"]): row for row in _load_jsonl(args.views)}
    benchmark = _load_jsonl(args.benchmark)
    qids = {str(row["qid"]) for row in benchmark}
    docs = {str(row["gold_doc"]) for row in benchmark} | {str(row["hard_doc"]) for row in benchmark}
    for qid in qids:
        docs.update(str(item["doc_id"]) for item in views[qid]["Q0"][:4])
    connection = open_readonly(args.index)
    try:
        store = load_passage_store(connection, sorted(docs, key=lambda x: (int(x), x) if x.isdigit() else (10**18, x)))
    finally:
        connection.close()
    rows = []
    prompts_ab, prompts_ba = [], []
    for pair in benchmark:
        qid = str(pair["qid"])
        context_docs = [str(item["doc_id"]) for item in views[qid]["Q0"][:4]]
        context = "\n\n".join(hydrate(pair["query"], [doc], 1, store)[0]["text"][:1800] for doc in context_docs)
        prompts_ab.append(_prompt(pair["query"], context, pair["gold_text"], pair["hard_text"]))
        prompts_ba.append(_prompt(pair["query"], context, pair["hard_text"], pair["gold_text"]))
        rows.append({**pair, "context": context})
    device = args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(str(args.model.resolve()), local_files_only=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(str(args.model.resolve()), local_files_only=True, torch_dtype=dtype)
    model.to(device).eval()
    outputs_ab = _generate(model, tokenizer, prompts_ab, device, args.batch_size, args.max_input_tokens)
    outputs_ba = _generate(model, tokenizer, prompts_ba, device, args.batch_size, args.max_input_tokens)
    for row, out_ab, out_ba in zip(rows, outputs_ab, outputs_ba, strict=True):
        choice_ab, choice_ba = _choice(out_ab), _choice(out_ba)
        row.update({"output_ab": out_ab, "output_ba": out_ba, "choice_ab": choice_ab, "choice_ba": choice_ba, "parseable": choice_ab is not None and choice_ba is not None, "consistent": (choice_ab, choice_ba) in (("A", "B"), ("B", "A")), "gold_wins_consistent": (choice_ab, choice_ba) == ("A", "B")})
    write_jsonl(args.output_dir / "prp_scores.jsonl", rows)
    parseable = [row for row in rows if row["parseable"]]
    consistent = [row for row in rows if row["consistent"]]
    gold_wins = sum(row["gold_wins_consistent"] for row in consistent)
    report = {"kind": "legalir_v34_b_direct_pairwise_prompting", "schema_version": 1, "model": str(args.model.resolve()), "model_size_fallback": "Qwen3-1.7B used because Qwen3-0.6B base was not local", "pairs": len(rows), "queries": len({row["qid"] for row in rows}), "parseable": len(parseable), "parseable_rate": len(parseable) / max(len(rows), 1), "consistent": len(consistent), "consistent_coverage": len(consistent) / max(len(rows), 1), "gold_wins_among_consistent": gold_wins, "gold_pairwise_precision": gold_wins / max(len(consistent), 1), "position_sensitive": len(parseable) - len(consistent), "gate": {"coverage_ge_0_60": len(consistent) / max(len(rows), 1) >= 0.60, "gold_precision_ge_0_75": gold_wins / max(len(consistent), 1) >= 0.75, "pass_prp": len(consistent) / max(len(rows), 1) >= 0.60 and gold_wins / max(len(consistent), 1) >= 0.75}, "inputs": {"metadata": str(args.metadata.resolve()), "metadata_sha256": sha256_file(args.metadata), "views": str(args.views.resolve()), "views_sha256": sha256_file(args.views), "benchmark": str(args.benchmark.resolve()), "benchmark_sha256": sha256_file(args.benchmark)}, "output": str((args.output_dir / "prp_scores.jsonl").resolve()), "elapsed_seconds": time.perf_counter() - started}
    write_json(args.output_dir / "prp_report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
