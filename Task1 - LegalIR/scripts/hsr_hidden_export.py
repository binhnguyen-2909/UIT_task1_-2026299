#!/usr/bin/env python3
"""GPU-only, resumable final-hidden-state exporter for the frozen c67 pool."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import qwen_c200_cache as legacy  # noqa: E402
import qwen_c200_common as qwen_common  # noqa: E402
from hsr_contract import (  # noqa: E402
    HIDDEN_DTYPE, HIDDEN_SIZE, HSR_KIND, HSR_VARIANT, MAX_TOKENS,
    PARAMETER_LIMIT, FROZEN_PLAN_PARAMETERS, HEAD_PARAMETERS, QWEN_MODEL_ID,
    QWEN_PLAN_PARAMETERS, POOL_SIZES, hidden_lineage, planned_parameter_budget,
)
from qwen_c200_common import (  # noqa: E402
    checkpoint_fingerprint, load_candidate_universe, load_raw_documents,
    needed_document_ids, pool_for, require_contexts_zip,
)
from v27_a_common import (  # noqa: E402
    CANDIDATE_LIMIT, V27ContractError, json_sha256, sha256_file, write_json,
)


NATIVE_PARITY_TOLERANCE = 1.0e-1  # bf16 autocast quantizes the hidden-state delta path at ~6e-2; same scale as the roundtrip gate below
HIDDEN_ROUNDTRIP_TOLERANCE = 1.0e-1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "pilot", "cache"), default="preflight")
    parser.add_argument("--stage1", type=Path, required=True); parser.add_argument("--stage2", type=Path, required=True); parser.add_argument("--anchor", type=Path, required=True); parser.add_argument("--contexts-zip", type=Path, required=True)
    parser.add_argument("--qwen-model", type=Path, required=True); parser.add_argument("--model-revision", required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0); parser.add_argument("--shard-count", type=int, default=1); parser.add_argument("--batch-candidates", nargs="+", type=int, default=[1, 2, 4, 8, 16]); parser.add_argument("--vram-reserve-gib", type=float, required=True); parser.add_argument("--pilot-qids", type=int, default=64); parser.add_argument("--pilot-seed", type=int, default=20_260_916); parser.add_argument("--cpu-threads", type=int, default=1); parser.add_argument("--hidden-dtype", choices=("float16", "float32"), default=HIDDEN_DTYPE)
    parser.add_argument("--resume", dest="resume", action="store_true"); parser.add_argument("--no-resume", dest="resume", action="store_false"); parser.set_defaults(resume=True)
    return parser.parse_args()


def _cap_threads(value: int) -> None:
    if int(value) < 1:
        raise ValueError("--cpu-threads must be positive")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
        os.environ[name] = str(int(value))


def _shard_qids(qids: Sequence[str], index: int, count: int) -> list[str]:
    if count < 1 or not 0 <= index < count:
        raise ValueError("invalid shard index/count")
    return [str(qid) for qid in qids if int(hashlib.sha256(str(qid).encode()).hexdigest(), 16) % count == index]


def _pilot_qids(qids: Sequence[str], count: int, seed: int) -> list[str]:
    return sorted(map(str, qids), key=lambda qid: hashlib.sha256(f"{seed}:{qid}".encode()).hexdigest())[:max(1, int(count))]


def _paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    root = args.output_dir / "hidden_state_cache"; stem = f"shard_{args.shard_index:03d}_of_{args.shard_count:03d}"
    return root / "features" / f"{stem}.npy", root / "rows" / f"{stem}.jsonl", root / "metadata" / f"{stem}.json", root / "progress" / f"{stem}.json"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise V27ContractError(f"{path}:{number}: hidden cache row is not an object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        sink.flush(); os.fsync(sink.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        sink.flush(); os.fsync(sink.fileno())


def _unique_parameter_count(model: Any) -> int:
    seen: set[int] = set(); total = 0
    for parameter in model.parameters():
        pointer = int(parameter.data_ptr())
        if pointer not in seen:
            seen.add(pointer); total += int(parameter.numel())
    return total


def _load_model(model_path: Path) -> tuple[Any, Any, dict[str, Any], int]:
    import torch
    model, tokenizer, token_info = legacy._load_tokenizer_model(argparse.Namespace(qwen_model=model_path, device="cuda", attention="sdpa"))
    hidden_size = int(getattr(model.config, "hidden_size", -1))
    if hidden_size != HIDDEN_SIZE:
        raise V27ContractError(f"Qwen hidden_size={hidden_size}; locked HSR plan requires {HIDDEN_SIZE}")
    if not hasattr(model, "lm_head"):
        raise V27ContractError("Qwen causal model has no lm_head for native parity")
    base = getattr(model, "model", None)
    if base is None and hasattr(model, "get_decoder"):
        base = model.get_decoder()
    if base is None:
        raise V27ContractError("Cannot access Qwen decoder for last_hidden_state")
    parameter_count = _unique_parameter_count(model)
    non_qwen_plan = FROZEN_PLAN_PARAMETERS - QWEN_PLAN_PARAMETERS
    combined = int(non_qwen_plan + parameter_count + HEAD_PARAMETERS)
    if combined >= PARAMETER_LIMIT:
        raise V27ContractError(f"HSR parameter budget exceeded: {combined} >= {PARAMETER_LIMIT}")
    token_info = {**dict(token_info), "hidden_size": hidden_size, "qwen_parameters": parameter_count, "combined_parameters": combined, "parameter_limit": PARAMETER_LIMIT, "under_4b": combined < PARAMETER_LIMIT}
    return model, tokenizer, token_info, parameter_count


def _batch_inputs(tokenizer: Any, rows: Sequence[Mapping[str, Any]], device: str) -> dict[str, Any]:
    batch = tokenizer.pad({"input_ids": [list(row["input_ids"]) for row in rows]}, padding=True, return_tensors="pt")
    return {key: value.to(device) for key, value in batch.items()}


def _hidden_and_delta(model: Any, tokenizer: Any, rows: Sequence[Mapping[str, Any]], *, yes: int, no: int, device: str, check_native: bool = False) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import torch
    batch = _batch_inputs(tokenizer, rows, device)
    if not bool(batch["attention_mask"][:, -1].all()):
        raise V27ContractError("Left-padding invariant failed: final position is padding")
    base = getattr(model, "model", None)
    if base is None and hasattr(model, "get_decoder"):
        base = model.get_decoder()
    if base is None:
        raise V27ContractError("Cannot access Qwen decoder for hidden-state export")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        try:
            output = base(**batch, use_cache=False, return_dict=True)
        except TypeError:
            output = base(**batch, return_dict=True)
        hidden = output.last_hidden_state[:, -1, :]
        weight = model.lm_head.weight
        bias = getattr(model.lm_head, "bias", None)
        delta_logits = hidden.matmul((weight[yes] - weight[no]).to(hidden.dtype))
        if bias is not None:
            delta_logits = delta_logits + (bias[yes] - bias[no]).to(delta_logits.dtype)
        native_error = None
        if check_native:
            native_logits = model(**batch, use_cache=False).logits[:, -1, :]
            native_delta = native_logits[:, yes].float() - native_logits[:, no].float()
            native_error = float(torch.max(torch.abs(native_delta - delta_logits.float())).item())
    return hidden.float().cpu().numpy(), delta_logits.float().cpu().numpy(), {"native_abs_error": native_error, "last_position": "last_non_padding_token", "base_output": "last_hidden_state_only"}


def _probe(model: Any, tokenizer: Any, sample: Mapping[str, Any], args: argparse.Namespace, token_info: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    import torch
    reserve = int(float(args.vram_reserve_gib) * 1024 ** 3); attempts: list[dict[str, Any]] = []; accepted: list[int] = []; parity_error: float | None = None; roundtrip_error: float | None = None
    for batch_size in sorted(set(int(value) for value in args.batch_candidates)):
        if batch_size < 1:
            raise ValueError("batch candidates must be positive")
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
            hidden, delta, check = _hidden_and_delta(model, tokenizer, [sample] * batch_size, yes=int(token_info["yes_token_id"]), no=int(token_info["no_token_id"]), device="cuda", check_native=True)
            parity_error = float(check["native_abs_error"] or 0.0); restored = hidden.astype(np.float16 if args.hidden_dtype == "float16" else np.float32).astype(np.float32)
            # Reconstructing with the native linear probe checks the locked
            # on-disk dtype before any labels are opened.
            import torch as _torch
            weight = model.lm_head.weight[int(token_info["yes_token_id"])] - model.lm_head.weight[int(token_info["no_token_id"])]
            roundtrip = _torch.from_numpy(restored).to("cuda").matmul(weight.to(_torch.float32)).detach().cpu().numpy()
            roundtrip_error = float(np.max(np.abs(roundtrip - delta)))
            free, total = torch.cuda.mem_get_info(); elapsed = time.monotonic() - started; ok = int(free) >= reserve and parity_error <= NATIVE_PARITY_TOLERANCE and roundtrip_error <= HIDDEN_ROUNDTRIP_TOLERANCE
            attempts.append({"batch_size": batch_size, "ok": ok, "elapsed_seconds": elapsed, "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()), "free_after_bytes": int(free), "total_bytes": int(total), "reserve_bytes": reserve, "native_abs_error": parity_error, "hidden_roundtrip_abs_error": roundtrip_error})
            if ok:
                accepted.append(batch_size)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); attempts.append({"batch_size": batch_size, "ok": False, "oom": True}); break
    if not accepted:
        raise RuntimeError("HSR exporter has no batch satisfying VRAM/parity reserve")
    return max(accepted), {"kind": f"{HSR_KIND}_gpu_probe", "batch_size": max(accepted), "attempts": attempts, "device": "cuda", "precision": "bf16", "native_parity_tolerance": NATIVE_PARITY_TOLERANCE, "hidden_roundtrip_tolerance": HIDDEN_ROUNDTRIP_TOLERANCE, "hidden_dtype": args.hidden_dtype}


def _recover_rows(rows_path: Path, *, expected_qids: Sequence[str], c67_by_qid: Mapping[str, Sequence[str]], lineage_sha256: str, feature_rows: int) -> tuple[dict[tuple[str, str], dict[str, Any]], set[str]]:
    rows = _read_jsonl(rows_path); result: dict[tuple[str, str], dict[str, Any]] = {}; by_qid: dict[str, set[str]] = {}; expected_qid_set = set(map(str, expected_qids))
    for row in rows:
        qid, doc_id = str(row.get("qid", "")), str(row.get("doc_id", "")); key = (qid, doc_id)
        if qid not in expected_qid_set or doc_id not in set(c67_by_qid.get(qid, ())) or key in result or row.get("lineage_sha256") != lineage_sha256:
            raise V27ContractError(f"Invalid/stale/duplicate HSR row: {key}")
        rank = list(c67_by_qid[qid]).index(doc_id) + 1
        index = int(row.get("state_index", -1))
        if int(row.get("candidate_rank", -1)) != rank or not 0 <= index < feature_rows:
            raise V27ContractError(f"HSR row index/rank drift: {key}")
        result[key] = row; by_qid.setdefault(qid, set()).add(doc_id)
    complete = {qid for qid, docs in by_qid.items() if docs == set(c67_by_qid[qid])}
    if len(result) != sum(len(c67_by_qid[qid]) for qid in complete):
        _write_jsonl(rows_path, [result[key] for qid in sorted(complete) for key in sorted(result) if key[0] == qid])
        result = {key: row for key, row in result.items() if key[0] in complete}
    return result, complete


def _pool_hash(c67_by_qid: Mapping[str, Sequence[str]]) -> str:
    return json_sha256({qid: list(c67_by_qid[qid]) for qid in sorted(c67_by_qid)})


def run(args: argparse.Namespace) -> dict[str, Any]:
    _cap_threads(args.cpu_threads); require_contexts_zip(args.contexts_zip)
    universe = load_candidate_universe(stage1=args.stage1, stage2=args.stage2, anchor=args.anchor, reference=None)
    c67_by_qid = {str(qid): list(pool_for(universe, str(qid), HSR_VARIANT)) for qid in universe.split.qids}
    target_qids = _shard_qids(list(c67_by_qid), args.shard_index, args.shard_count)
    if not args.qwen_model.is_dir():
        raise FileNotFoundError(f"Missing Qwen snapshot: {args.qwen_model}")
    fingerprint = checkpoint_fingerprint(args.qwen_model)
    lineage = hidden_lineage(stage1=args.stage1, stage2=args.stage2, anchor=args.anchor, contexts_zip=args.contexts_zip, model_fingerprint=fingerprint, model_revision=args.model_revision, hidden_dtype=args.hidden_dtype); lineage_sha = json_sha256(lineage)
    root = args.output_dir / "hidden_state_cache"; write_json(root / f"hsr_label_blind_manifest_shard_{args.shard_index:03d}_of_{args.shard_count:03d}.json", {"kind": f"{HSR_KIND}_manifest", "schema_version": 1, "lineage": lineage, "lineage_sha256": lineage_sha, "shard_index": args.shard_index, "shard_count": args.shard_count, "target_qids": target_qids, "planned_pairs": len(target_qids) * len(c67_by_qid[target_qids[0]]) if target_qids else 0, "pool_sha256": _pool_hash(c67_by_qid), "holdout_opened": False, "public_labels_used": False})
    if args.mode == "preflight":
        return {"kind": f"{HSR_KIND}_preflight", "stage": "label_blind_preflight", "lineage_sha256": lineage_sha, "queries": len(target_qids), "planned_pairs": len(target_qids) * POOL_SIZES[HSR_VARIANT], "hidden_size": HIDDEN_SIZE, "pool_size": POOL_SIZES[HSR_VARIANT], "parameter_budget": planned_parameter_budget(), "gpu_required": True, "holdout_opened": False, "public_labels_used": False}
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("HSR hidden exporter requires CUDA")
    model, tokenizer, token_info, qwen_parameters = _load_model(args.qwen_model)
    feature_path, rows_path, metadata_path, progress_path = _paths(args); pair_total = len(target_qids) * POOL_SIZES[HSR_VARIANT]
    feature_path.parent.mkdir(parents=True, exist_ok=True); dtype = np.float16 if args.hidden_dtype == "float16" else np.float32
    if feature_path.exists():
        states = np.load(feature_path, mmap_mode="r+", allow_pickle=False)
        if states.shape != (pair_total, HIDDEN_SIZE) or states.dtype != dtype:
            raise V27ContractError(f"Existing HSR feature array shape/dtype mismatch: {feature_path}")
    else:
        states = np.lib.format.open_memmap(feature_path, mode="w+", dtype=dtype, shape=(pair_total, HIDDEN_SIZE))
    meta = {"kind": f"{HSR_KIND}_cache", "schema_version": 1, "lineage": lineage, "lineage_sha256": lineage_sha, "shard_index": args.shard_index, "shard_count": args.shard_count, "hidden_size": HIDDEN_SIZE, "hidden_dtype": args.hidden_dtype, "feature_path": str(feature_path.resolve()), "rows_path": str(rows_path.resolve()), "pair_count": pair_total, "pool_size": POOL_SIZES[HSR_VARIANT], "pool_sha256": _pool_hash(c67_by_qid), "token_scoring": token_info, "model_sha256": json_sha256(fingerprint), "tokenizer_sha256": fingerprint.get("tokenizer.json") or fingerprint.get("tokenizer_config.json"), "packer_sha256": sha256_file(Path(qwen_common.__file__)), "parameter_ledger": {"frozen_plan_parameters": FROZEN_PLAN_PARAMETERS, "qwen_parameters": qwen_parameters, "head_parameters": HEAD_PARAMETERS, "qwen_plan_parameters": QWEN_PLAN_PARAMETERS, "combined_parameters": token_info["combined_parameters"], "parameter_limit": PARAMETER_LIMIT, "under_4b": token_info["under_4b"]}, "mode": args.mode, "holdout_opened": False, "public_labels_used": False}
    if metadata_path.is_file():
        old = json.loads(metadata_path.read_text(encoding="utf-8"))
        if old.get("lineage_sha256") != lineage_sha or int(old.get("pair_count", -1)) != pair_total:
            raise V27ContractError("Existing HSR metadata lineage/pair-count mismatch")
    write_json(metadata_path, meta)
    expected = {qid: {(qid, doc_id) for doc_id in c67_by_qid[qid]} for qid in target_qids}; existing, complete = _recover_rows(rows_path, expected_qids=target_qids, c67_by_qid=c67_by_qid, lineage_sha256=lineage_sha, feature_rows=pair_total) if args.resume else ({}, set())
    if rows_path.is_file() and not args.resume:
        raise FileExistsError(f"HSR cache exists with --no-resume: {rows_path}")
    docs = load_raw_documents(args.contexts_zip, needed_document_ids(universe, limit=POOL_SIZES[HSR_VARIANT])); qid_to_index = {str(qid): index for index, qid in enumerate(universe.split.qids)}
    work_qids = _pilot_qids(target_qids, args.pilot_qids, args.pilot_seed) if args.mode == "pilot" else list(target_qids); pending_qids = [qid for qid in work_qids if qid not in complete]
    if not pending_qids:
        report = {"kind": f"{HSR_KIND}_cache_report", "mode": args.mode, "score_path": str(feature_path.resolve()), "rows_path": str(rows_path.resolve()), "metadata_path": str(metadata_path.resolve()), "progress_path": str(progress_path.resolve()), "lineage_sha256": lineage_sha, "pairs": len(existing), "qids": len(complete), "complete": len(complete) == len(target_qids), "parameter_ledger": meta["parameter_ledger"], "holdout_opened": False, "public_labels_used": False}; write_json(root / f"hsr_{args.mode}_report_shard_{args.shard_index:03d}_of_{args.shard_count:03d}.json", report); return report
    probe_qids = sorted(pending_qids, key=lambda qid: max(len(docs[doc_id].body) + len(docs[doc_id].title) for doc_id in c67_by_qid[qid]), reverse=True)[:16]
    probe_records = [legacy._records_for_qid(tokenizer=tokenizer, universe=universe, docs=docs, qid=qid, query_index=qid_to_index[qid], variant=HSR_VARIANT, max_length=MAX_TOKENS) for qid in probe_qids]
    sample = max((row for records in probe_records for row in records), key=lambda row: len(row["input_ids"])); batch_size, probe = _probe(model, tokenizer, sample, args, token_info)
    pool_hash = _pool_hash(c67_by_qid); model_hash = json_sha256(fingerprint); started = time.monotonic(); pairs_done = len(existing); completed = set(complete); target_positions = {qid: index for index, qid in enumerate(target_qids)}
    try:
        for qid in pending_qids:
            records = legacy._records_for_qid(tokenizer=tokenizer, universe=universe, docs=docs, qid=qid, query_index=qid_to_index[qid], variant=HSR_VARIANT, max_length=MAX_TOKENS); hidden_blocks: list[np.ndarray] = []; delta_blocks: list[np.ndarray] = []
            for start in range(0, len(records), batch_size):
                block = records[start:start + batch_size]
                try:
                    hidden, delta, _ = _hidden_and_delta(model, tokenizer, block, yes=int(token_info["yes_token_id"]), no=int(token_info["no_token_id"]), device="cuda")
                except torch.cuda.OutOfMemoryError as exc:
                    torch.cuda.empty_cache(); raise RuntimeError(f"HSR OOM on QID {qid}; rerun with smaller --batch-candidates and --resume") from exc
                hidden_blocks.append(hidden); delta_blocks.append(delta)
            hidden_all = np.concatenate(hidden_blocks, axis=0); delta_all = np.concatenate(delta_blocks, axis=0); rows_out: list[dict[str, Any]] = []; base_offset = target_positions[qid] * POOL_SIZES[HSR_VARIANT]
            for offset, (record, vector, delta) in enumerate(zip(records, hidden_all, delta_all, strict=True)):
                packed_hash = hashlib.sha256(str(record["packed_text"]).encode("utf-8")).hexdigest()
                evidence_valid = bool(not docs[str(record["doc_id"])].empty and int(record["pack"].get("packed_tokens", 0)) > 0)
                stored = vector.astype(dtype, copy=False); states[base_offset + offset] = stored
                rows_out.append({"qid": qid, "doc_id": str(record["doc_id"]), "candidate_rank": int(record["candidate_rank"]), "state_index": base_offset + offset, "evidence_valid": evidence_valid, "packed_text_sha256": packed_hash, "input_ids_sha256": json_sha256({"input_ids": list(record["input_ids"])}), "native_yes_minus_no": float(delta), "model_sha256": model_hash, "tokenizer_sha256": meta["tokenizer_sha256"], "packer_sha256": meta["packer_sha256"], "pool_sha256": pool_hash, "lineage_sha256": lineage_sha})
            states.flush(); _append_jsonl(rows_path, rows_out); existing.update({(row["qid"], row["doc_id"]): row for row in rows_out}); completed.add(qid); pairs_done = len(existing); write_json(progress_path, {"kind": f"{HSR_KIND}_progress", "lineage_sha256": lineage_sha, "completed_qids": sorted(completed), "pairs_done": pairs_done, "pairs_total": pair_total, "complete": len(completed) == len(target_qids)}); print(json.dumps({"event": "hsr_hidden_progress", "mode": args.mode, "qid": qid, "qids_done": len(completed), "qids_total": len(target_qids), "pairs_done": pairs_done, "pairs_total": pair_total, "elapsed_seconds": time.monotonic() - started}, ensure_ascii=False), flush=True)
    finally:
        del model, tokenizer, states; gc.collect(); torch.cuda.empty_cache()
    complete_all = len(completed) == len(target_qids); report = {"kind": f"{HSR_KIND}_cache_report", "mode": args.mode, "score_path": str(feature_path.resolve()), "rows_path": str(rows_path.resolve()), "metadata_path": str(metadata_path.resolve()), "progress_path": str(progress_path.resolve()), "lineage_sha256": lineage_sha, "pairs": pairs_done, "qids": len(completed), "complete": complete_all, "batch_size": batch_size, "runtime_probe": probe, "parameter_ledger": meta["parameter_ledger"], "elapsed_seconds": time.monotonic() - started, "holdout_opened": False, "public_labels_used": False}
    write_json(root / f"hsr_{args.mode}_report_shard_{args.shard_index:03d}_of_{args.shard_count:03d}.json", report)
    if args.mode == "cache" and not complete_all:
        raise V27ContractError("HSR full cache did not cover all shard QIDs")
    return report


if __name__ == "__main__":
    raise SystemExit((lambda args: (print(json.dumps(run(args), ensure_ascii=False)), 0)[1])(parse_args()))
