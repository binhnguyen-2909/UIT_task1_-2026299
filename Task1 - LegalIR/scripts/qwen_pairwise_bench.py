#!/usr/bin/env python3
"""Pairwise verifier throughput bench (20 pairs), then go/no-go.

Loads a frozen generative model once (bf16, batch 1, no-thinking),
scores A/B/C/D single-token choices in both orders, times 20 pairs,
writes output/verifier_probe/bench_20pairs.json with pairs_per_sec
and go_full_7182 flag.

Research-only: default Qwen/Qwen3-1.7B is NOT in approved_models.json.
Compliance path uses approved AITeamVN/Vi-Qwen2-1.5B-RAG.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def _task_root() -> Path:
    cwd_base = Path.cwd() / "Task1 - LegalIR"
    if cwd_base.is_dir():
        return cwd_base.parent
    return Path(__file__).resolve().parent.parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument(
        "--qwen-model",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Research default (NOT approved). Use approved fallback for submission path.",
    )
    parser.add_argument(
        "--approved-model",
        type=str,
        default="AITeamVN/Vi-Qwen2-1.5B-RAG",
    )
    parser.add_argument(
        "--use-approved",
        action="store_true",
        help="Bench the approved 1.5B model instead of research 1.7B.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Task1 - LegalIR/output/verifier_probe/bench_20pairs.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = _task_root()
    out_path = (root / args.output) if not args.output.is_absolute() else args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_id = str(args.approved_model) if bool(args.use_approved) else str(args.qwen_model)
    result: dict = {
        "kind": "legalir_pairwise_verifier_bench",
        "model_id": model_id,
        "use_approved": bool(args.use_approved),
        "pairs_requested": int(args.pairs),
        "batch_size": 1,
        "dtype": "bfloat16",
        "no_thinking": True,
        "both_orders": True,
    }

    if bool(args.dry_run):
        result.update(
            {
                "pairs_scored": 0,
                "pairs_per_sec": 0.0,
                "go_full_7182": False,
                "reason": "dry_run_no_gpu",
            }
        )
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    import sys

    sys.path.insert(0, str(root / "Task1 - LegalIR/scripts"))
    from qwen_pairwise_verifier import build_pair_prompt, prompt_sha256
    from verifier_probe_manifest import build_probe_manifest

    manifest = build_probe_manifest(n_probe=60, seed=20260915)
    qids = list(manifest["qids"])[: max(int(args.pairs), 1)]

    # Synthetic long evidence at F scale (~3993 tokens/doc). Real F packs are
    # 4096-token inputs; bench uses repeated legal text to match forward cost
    # without needing raw contexts ZIP or GPU cache.
    doc_a = ("điều khoản pháp luật điều kiện ngoại lệ văn bản hợp đồng " * 700).strip()
    doc_b = ("nghị định thông tư điều luật khoản điểm hiệu lực thi hành " * 700).strip()
    prompts_ab = [build_pair_prompt(f"Câu hỏi {qid}?", doc_a, doc_b) for qid in qids]
    prompts_ba = [build_pair_prompt(f"Câu hỏi {qid}?", doc_b, doc_a) for qid in qids]
    result["prompt_sha256_ab0"] = prompt_sha256(prompts_ab[0])
    result["prompt_sha256_ba0"] = prompt_sha256(prompts_ba[0])

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - env without torch
        result.update(
            {
                "pairs_scored": 0,
                "pairs_per_sec": 0.0,
                "go_full_7182": False,
                "reason": f"missing_dependency: {exc}",
            }
        )
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, local_files_only=True, trust_remote_code=False,
        )
    except Exception as exc:
        result.update(
            {
                "pairs_scored": 0,
                "pairs_per_sec": 0.0,
                "go_full_7182": False,
                "reason": f"model_not_cached_locally: {model_id}: {exc}",
                "hint": "set HF_HUB_OFFLINE=0 to download once, or pass --use-approved with cached 1.5B",
            }
        )
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Single-token choice ids for A/B/C/D (first token of each label).
    choice_ids = {}
    for label in ("A", "B", "C", "D"):
        ids = tokenizer(label, add_special_tokens=False).get("input_ids", [])
        if not ids:
            raise RuntimeError(f"tokenizer has no id for choice {label}")
        choice_ids[label] = int(ids[0])
    result["choice_token_ids"] = dict(choice_ids)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16,
    ).to("cuda").eval()
    assert not model.training

    def score_one(prompt: str) -> list[float]:
        batch = tokenizer(prompt, return_tensors="pt")
        batch = {k: v.to("cuda") for k, v in batch.items()}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**batch).logits[0, -1, :].float().cpu()
        return [float(logits[choice_ids[c]]) for c in ("A", "B", "C", "D")]

    # Warmup 1 pair (2 forwards), then time N pairs (2N forwards).
    score_one(prompts_ab[0])
    score_one(prompts_ba[0])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = min(int(args.pairs), len(prompts_ab))
    for i in range(n):
        score_one(prompts_ab[i])
        score_one(prompts_ba[i])
    torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - t0, 1e-6)
    pps = n / elapsed
    result.update(
        {
            "pairs_scored": n,
            "forwards": n * 2,
            "elapsed_s": elapsed,
            "pairs_per_sec": pps,
            "go_full_7182": bool(pps >= 7182 / 21600),
            "required_throughput": 7182 / 21600,
            "model_revision": hashlib.sha256(model_id.encode()).hexdigest()[:16],
        }
    )
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
