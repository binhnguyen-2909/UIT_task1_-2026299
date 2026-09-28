#!/usr/bin/env python3
"""GPU-only, resumable Qwen3 0.6B cache for frozen complement c35/c200 pools."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwen_c200_common import (  # noqa: E402
    INSTRUCTION, LONG_MAX_TOKENS, POOL_SIZES, QWEN_BRANCH_KIND, SHORT_MAX_TOKENS,
    cache_lineage, cap_cpu_threads, checkpoint_fingerprint, load_candidate_universe,
    load_raw_documents, needed_document_ids, pack_document, pool_for, require_complement_file, require_contexts_zip,
    score_order, sha256_file, validate_cache_coverage, write_jsonl,
)
from v27_a_common import V27ContractError, json_sha256, write_json  # noqa: E402


PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "score"), default="preflight")
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--contexts-zip", type=Path, required=True)
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=("A_c35_short", "D_c200_long"), default=["A_c35_short", "D_c200_long"])
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batch-candidates", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--max-length-short", type=int, default=SHORT_MAX_TOKENS)
    parser.add_argument("--max-length-long", type=int, default=LONG_MAX_TOKENS)
    parser.add_argument("--attention", choices=("sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def _variant_length(args: argparse.Namespace, variant: str) -> int:
    return int(args.max_length_short if variant == "A_c35_short" else args.max_length_long)


def _shard_qids(qids: Sequence[str], index: int, count: int) -> list[str]:
    if count <= 0 or not 0 <= index < count:
        raise ValueError("shard-index must be within [0, shard-count)")
    return [qid for qid in qids if int(hashlib.sha256(qid.encode("utf-8")).hexdigest(), 16) % count == index]


def _paths(args: argparse.Namespace, variant: str) -> tuple[Path, Path, Path]:
    root = args.output_dir / "qwen_cache" / variant
    stem = f"shard_{args.shard_index:03d}_of_{args.shard_count:03d}"
    return root / "shards" / f"{stem}.jsonl", root / "metadata" / f"{stem}.json", root / "progress" / f"{stem}.json"


def _load_tokenizer_model(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.qwen_model, local_files_only=True, padding_side="left", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {"local_files_only": True, "torch_dtype": torch.bfloat16, "attn_implementation": args.attention}
    try:
        model = AutoModelForCausalLM.from_pretrained(args.qwen_model, **kwargs).to(args.device).eval()
    except (ImportError, ValueError) as exc:
        if args.attention != "flash_attention_2":
            raise
        # A server without flash-attn remains valid; the cache metadata records
        # the actual SDPA fallback so shards cannot be mixed silently.
        kwargs["attn_implementation"] = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(args.qwen_model, **kwargs).to(args.device).eval()
    ids_yes = tokenizer("yes", add_special_tokens=False).input_ids
    ids_no = tokenizer("no", add_special_tokens=False).input_ids
    if len(ids_yes) != 1 or len(ids_no) != 1:
        raise V27ContractError("Qwen reranker yes/no tokens are not single-token IDs")
    return model, tokenizer, {"yes_token_id": int(ids_yes[0]), "no_token_id": int(ids_no[0]), "attention": kwargs["attn_implementation"]}


def _prompt_components(tokenizer: Any, question: str, document: str, max_length: int) -> tuple[list[int], dict[str, Any]]:
    query = f"<Instruct>: {INSTRUCTION}\n<Query>: {question}\n<Document>: "
    prefix_ids = tokenizer.encode(PREFIX, add_special_tokens=False)
    query_ids = tokenizer.encode(query, add_special_tokens=False)
    suffix_ids = tokenizer.encode(SUFFIX, add_special_tokens=False)
    available = int(max_length) - len(prefix_ids) - len(query_ids) - len(suffix_ids)
    if available <= 32:
        raise V27ContractError("Qwen prompt/template consumes the entire configured context length")
    doc_ids = tokenizer(document, add_special_tokens=False).input_ids[:available]
    return prefix_ids + query_ids + doc_ids + suffix_ids, {
        "input_tokens": len(prefix_ids) + len(query_ids) + len(doc_ids) + len(suffix_ids),
        "document_tokens_after_template": len(doc_ids),
        "document_token_budget": available,
        "template_tokens": len(prefix_ids) + len(query_ids) + len(suffix_ids),
    }


def _score_batch(model: Any, tokenizer: Any, rows: Sequence[Mapping[str, Any]], *, yes: int, no: int, device: str) -> list[tuple[float, float]]:
    import torch
    batch = tokenizer.pad({"input_ids": [list(row["input_ids"]) for row in rows]}, padding=True, return_tensors="pt")
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(**batch).logits[:, -1, :]
        delta = (logits[:, yes].float() - logits[:, no].float())
        probability = torch.sigmoid(delta)
    return list(zip(delta.detach().cpu().tolist(), probability.detach().cpu().tolist(), strict=True))


def _probe_batch_size(model: Any, tokenizer: Any, sample: Mapping[str, Any], args: argparse.Namespace, tokens: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    import torch
    attempts: list[dict[str, Any]] = []; accepted: list[int] = []
    for batch_size in sorted(set(int(value) for value in args.batch_candidates)):
        if batch_size <= 0:
            raise ValueError("batch candidates must be positive")
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            _score_batch(model, tokenizer, [sample] * batch_size, yes=int(tokens["yes_token_id"]), no=int(tokens["no_token_id"]), device=args.device)
            attempts.append({"batch_size": batch_size, "ok": True, "peak_allocated_bytes": int(torch.cuda.max_memory_allocated())})
            accepted.append(batch_size)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); attempts.append({"batch_size": batch_size, "ok": False})
            break
    if not accepted:
        raise RuntimeError("Qwen c200 cache OOM at batch size 1; choose a GPU with more free VRAM")
    return max(accepted), {"kind": "legalir_qwen_c200_gpu_probe", "batch_size": max(accepted), "attempts": attempts, "device": args.device, "precision": "bf16"}


def _recover_completed_qids(path: Path, universe: Any, variant: str) -> set[str]:
    """Make interrupted append-only cache files safe to resume.

    A QID is marked complete only if it has precisely the expected pool rows.
    Any partially appended QID is discarded and recalculated, avoiding silent
    duplicate/missing documents after an interruption or OOM.
    """

    if not path.is_file():
        return set()
    by_qid: dict[str, dict[str, dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise V27ContractError(f"{path}:{number}: cache row is not an object")
            qid, doc_id = str(row.get("qid", "")), str(row.get("doc_id", ""))
            if not qid or not doc_id or doc_id in by_qid.setdefault(qid, {}):
                raise V27ContractError(f"{path}:{number}: invalid or duplicate cache row")
            by_qid[qid][doc_id] = dict(row)
    complete: dict[str, list[dict[str, Any]]] = {}
    for qid, rows in by_qid.items():
        expected = list(pool_for(universe, qid, variant))
        if set(rows) == set(expected):
            if any(int(rows[doc_id].get("candidate_rank", -1)) != rank for rank, doc_id in enumerate(expected, 1)):
                raise V27ContractError(f"{qid}: cache candidate_rank drift")
            complete[qid] = [rows[doc_id] for doc_id in expected]
        elif qid not in {str(value) for value in universe.split.qids}:
            raise V27ContractError(f"{qid}: cache does not belong to complement candidate universe")
    # Rewriting only whole QID records makes resume idempotent after a failed
    # process.  The qwen cache is a derived artifact, not a source artifact.
    write_jsonl(path, (row for qid in sorted(complete) for row in complete[qid]), append=False)
    return set(complete)


def _records_for_qid(
    *,
    tokenizer: Any,
    universe: Any,
    docs: Mapping[str, Any],
    qid: str,
    query_index: int,
    variant: str,
    max_length: int,
) -> list[dict[str, Any]]:
    question = str(universe.split.questions[query_index])
    records: list[dict[str, Any]] = []
    for rank, doc_id in enumerate(pool_for(universe, qid, variant), 1):
        # ``_prompt_components`` knows the exact Qwen template/query reserve,
        # so first obtain its available document budget using an empty text.
        _, reserve = _prompt_components(tokenizer, question, "", max_length)
        packed, pack_meta = pack_document(
            tokenizer, question=question, document=docs[doc_id],
            max_tokens=int(reserve["document_token_budget"]),
        )
        input_ids, prompt_meta = _prompt_components(tokenizer, question, packed, max_length)
        records.append({
            "qid": qid, "doc_id": doc_id, "candidate_rank": rank,
            "input_ids": input_ids, "packed_text": packed, "pack": pack_meta, "prompt": prompt_meta,
        })
    return records


def preflight(args: argparse.Namespace) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    cap_cpu_threads(args.cpu_threads)
    universe = load_candidate_universe(stage1=args.stage1, stage2=args.stage2, anchor=args.anchor, reference=None)
    require_contexts_zip(args.contexts_zip)
    fingerprint = checkpoint_fingerprint(args.qwen_model)
    if set(args.variants) - {"A_c35_short", "D_c200_long"} or not args.variants:
        raise V27ContractError("Qwen cache must request A c35-short and/or D c200-long only; B/C derive from D")
    manifests: dict[str, Any] = {}
    for variant in args.variants:
        lineage = cache_lineage(universe, contexts_zip=args.contexts_zip, model_fingerprint=fingerprint, variant=variant, max_tokens=_variant_length(args, variant))
        manifests[variant] = {"lineage": lineage, "lineage_sha256": json_sha256(lineage), "shard_qids": _shard_qids(list(universe.split.qids), args.shard_index, args.shard_count)}
    report = {"kind": QWEN_BRANCH_KIND, "schema_version": 1, "stage": "label_blind_preflight", "split": "complement", "queries": len(universe.split.qids), "variants": manifests, "pool_oracles_unavailable_without_labels": True, "model_budget": {"v16b_parameters_billion": 1.09, "qwen_parameters_billion": 0.60, "selector_parameters_billion": 0.001, "total_parameters_billion": 1.691, "under_4b": True}, "holdout_opened": False, "public_labels_used": False}
    return report, universe, fingerprint


def score_variant(args: argparse.Namespace, universe: Any, fingerprint: Mapping[str, str], variant: str) -> dict[str, Any]:
    cache_path, meta_path, progress_path = _paths(args, variant)
    lineage = cache_lineage(universe, contexts_zip=args.contexts_zip, model_fingerprint=fingerprint, variant=variant, max_tokens=_variant_length(args, variant))
    lineage_sha = json_sha256(lineage)
    if meta_path.is_file():
        prior = json.loads(meta_path.read_text(encoding="utf-8"))
        if prior.get("lineage_sha256") != lineage_sha:
            raise V27ContractError(f"Qwen cache metadata differs from frozen lineage: {meta_path}")
    target_qids = _shard_qids(list(universe.split.qids), args.shard_index, args.shard_count)
    if cache_path.is_file() and not args.resume:
        raise FileExistsError(f"Refusing to append to an existing cache with --no-resume: {cache_path}")
    done = _recover_completed_qids(cache_path, universe, variant) if args.resume else set()
    unexpected = done - set(target_qids)
    if unexpected: raise V27ContractError("Qwen cache shard contains a QID assigned to another shard")
    model, tokenizer, token_info = _load_tokenizer_model(args)
    try:
        docs = load_raw_documents(args.contexts_zip, needed_document_ids(universe, limit=POOL_SIZES[variant]))
        todo = [(index, str(qid)) for index, qid in enumerate(universe.split.qids) if str(qid) in target_qids and str(qid) not in done]
        if todo:
            # Probe a genuinely heavy, fully templated record without keeping
            # all 239k c200 prompts in host memory.
            probe_candidates = sorted(
                todo,
                key=lambda item: max(len(docs[doc_id].body) + len(docs[doc_id].title) for doc_id in pool_for(universe, item[1], variant)),
                reverse=True,
            )[:16]
            sample = max(
                (
                    record
                    for index, qid in probe_candidates
                    for record in _records_for_qid(
                        tokenizer=tokenizer, universe=universe, docs=docs, qid=qid, query_index=index,
                        variant=variant, max_length=_variant_length(args, variant),
                    )
                ),
                key=lambda row: len(row["input_ids"]),
            )
            batch_size, probe = _probe_batch_size(model, tokenizer, sample, args, token_info)
        else:
            batch_size, probe = int(args.batch_size), {"kind": "cache_resume_no_probe", "batch_size": int(args.batch_size)}
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps({"kind": "legalir_qwen_c200_score_cache", "schema_version": 1, "lineage": lineage, "lineage_sha256": lineage_sha, "shard_index": args.shard_index, "shard_count": args.shard_count, "token_scoring": token_info, "runtime_probe": probe, "holdout_opened": False, "public_labels_used": False}, ensure_ascii=False, indent=2), encoding="utf-8")
        pairs_done = 0
        pairs_total = len(todo) * POOL_SIZES[variant]
        for qid_number, (index, qid) in enumerate(todo, 1):
            records = _records_for_qid(
                tokenizer=tokenizer, universe=universe, docs=docs, qid=qid, query_index=index,
                variant=variant, max_length=_variant_length(args, variant),
            )
            # Bucketing is query-local: it avoids padding while keeping an
            # interruption's recovery scope bounded to a single query.
            ordered = sorted(records, key=lambda row: (len(row["input_ids"]), int(row["candidate_rank"])))
            completed: list[dict[str, Any]] = []
            for start in range(0, len(ordered), batch_size):
                part = ordered[start:start + batch_size]
                values = _score_batch(model, tokenizer, part, yes=int(token_info["yes_token_id"]), no=int(token_info["no_token_id"]), device=args.device)
                for row, (logit, probability) in zip(part, values, strict=True):
                    completed.append({"qid": row["qid"], "doc_id": row["doc_id"], "candidate_rank": row["candidate_rank"], "yes_no_logit": float(logit), "yes_probability": float(probability), "pack": row["pack"], "prompt": row["prompt"], "lineage_sha256": lineage_sha})
            write_jsonl(cache_path, sorted(completed, key=lambda row: int(row["candidate_rank"])), append=True)
            done.add(qid)
            write_json(progress_path, {"kind": "legalir_qwen_c200_cache_progress", "lineage_sha256": lineage_sha, "completed_qids": sorted(done), "complete": set(done) == set(target_qids)})
            pairs_done += len(records)
            if qid_number % 10 == 0 or qid_number == len(todo):
                print(json.dumps({"event": "qwen_cache_progress", "variant": variant, "qids_done": qid_number, "qids_total": len(todo), "pairs_done": pairs_done, "pairs_total": pairs_total}, ensure_ascii=False), flush=True)
        return {"variant": variant, "cache_path": str(cache_path.resolve()), "metadata_path": str(meta_path.resolve()), "lineage_sha256": lineage_sha, "qids_complete": len(done), "qids_total": len(target_qids), "complete": set(done) == set(target_qids), "batch_size": batch_size}
    finally:
        del model, tokenizer; gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except ImportError:
            pass


def run(args: argparse.Namespace) -> dict[str, Any]:
    preflight_report, universe, fingerprint = preflight(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "qwen_c200_preflight.json", preflight_report)
    if args.mode == "preflight": return preflight_report
    reports = [score_variant(args, universe, fingerprint, variant) for variant in args.variants]
    return {**preflight_report, "stage": "gpu_cache_complete", "caches": reports}


def main() -> int:
    args = parse_args(); report = run(args); output = args.output_dir / ("qwen_c200_preflight.json" if args.mode == "preflight" else "qwen_c200_cache_report.json")
    write_json(output, report)
    print(json.dumps({"output": str(output.resolve()), "stage": report["stage"], "holdout_opened": False}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
