#!/usr/bin/env python3
"""GPU-only, resumable Qwen3-Reranker-0.6B long cache over frozen c35.

The cache is label-blind and uses the already-validated F scorer semantics:
legal-boundary packing, the Qwen yes/no continuation, BF16 inference and a
4096-token context.  It does not read E3B reports, qrels or holdout labels.
"""

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

sys.path.insert(0, str(Path(__file__).resolve().parent))

import qwen_c200_cache as legacy  # noqa: E402
from qwen_c200_common import (  # noqa: E402
    checkpoint_fingerprint, load_candidate_universe, load_raw_documents,
    needed_document_ids, pack_document, pool_for, require_contexts_zip,
)
from r1_contract import FROZEN_UPSTREAM_PARAMETERS, PARAMETER_LIMIT, QWEN_MAX_TOKENS, R1_KIND, qwen_lineage  # noqa: E402
from v27_a_common import CANDIDATE_LIMIT, V27ContractError, json_sha256, sha256_file, write_json  # noqa: E402


# The existing module owns the prompt template and score implementation.  R1
# intentionally calls those exact helpers rather than making a second prompt.
R1_VARIANT = "A_c35_short"


def _unique_parameter_count(model: Any) -> int:
    """Count loaded tensor elements once, accounting for tied parameters."""
    seen: set[int] = set(); total = 0
    for parameter in model.parameters():
        pointer = int(parameter.data_ptr())
        if pointer in seen:
            continue
        seen.add(pointer); total += int(parameter.numel())
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "pilot", "cache"), default="preflight")
    parser.add_argument("--stage1", type=Path, required=True); parser.add_argument("--stage2", type=Path, required=True); parser.add_argument("--anchor", type=Path, required=True); parser.add_argument("--contexts-zip", type=Path, required=True)
    parser.add_argument("--qwen-model", type=Path, required=True); parser.add_argument("--model-revision", required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0); parser.add_argument("--shard-count", type=int, default=1); parser.add_argument("--batch-candidates", nargs="+", type=int, default=[1, 2, 4, 8, 16]); parser.add_argument("--vram-reserve-gib", type=float, required=True); parser.add_argument("--pilot-qids", type=int, default=64); parser.add_argument("--pilot-seed", type=int, default=20_260_916); parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--resume", dest="resume", action="store_true"); parser.add_argument("--no-resume", dest="resume", action="store_false"); parser.set_defaults(resume=True)
    return parser.parse_args()


def _cap_threads(value: int) -> None:
    if value < 1: raise ValueError("--cpu-threads must be positive")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_MAX_THREADS"): os.environ[name] = str(value)


def _shard_qids(qids: Sequence[str], index: int, count: int) -> list[str]:
    if count < 1 or not 0 <= index < count: raise ValueError("invalid shard index/count")
    return [str(qid) for qid in qids if int(hashlib.sha256(str(qid).encode("utf-8")).hexdigest(), 16) % count == index]


def _paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    root = args.output_dir / "r1_qwen_cache"; stem = f"shard_{args.shard_index:03d}_of_{args.shard_count:03d}"
    return root / "shards" / f"{stem}.jsonl", root / "metadata" / f"{stem}.json", root / "progress" / f"{stem}.json"


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows: target.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        target.flush(); os.fsync(target.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        target.flush(); os.fsync(target.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file(): return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip(): continue
            row = json.loads(line)
            if not isinstance(row, dict): raise V27ContractError(f"{path}:{number}: invalid R1 cache row")
            rows.append(row)
    return rows


def _records(tokenizer: Any, universe: Any, docs: Mapping[str, Any], qid: str, index: int) -> list[dict[str, Any]]:
    # `A_c35_short` identifies exactly the frozen c35 IDs; max_length is long.
    records = legacy._records_for_qid(tokenizer=tokenizer, universe=universe, docs=docs, qid=qid, query_index=index, variant=R1_VARIANT, max_length=QWEN_MAX_TOKENS)
    for row in records:
        packed = docs[str(row["doc_id"])]
        input_record = {"qid": str(qid), "doc_id": str(row["doc_id"]), "candidate_rank": int(row["candidate_rank"]), "input_ids": list(row["input_ids"]), "pack": row["pack"], "prompt": row["prompt"]}
        row["input_sha256"] = json_sha256(input_record); row["packed_source_tokens"] = int(row["pack"].get("source_tokens", 0)); row["packed_tokens"] = int(row["pack"].get("packed_tokens", 0)); row["packed_document_empty"] = bool(getattr(packed, "empty", False))
    return records


def _probe(model: Any, tokenizer: Any, sample: Mapping[str, Any], args: argparse.Namespace, token_info: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    import torch
    attempts: list[dict[str, Any]] = []; accepted: list[int] = []; reserve = int(args.vram_reserve_gib * 1024 ** 3)
    for batch_size in sorted(set(int(value) for value in args.batch_candidates)):
        if batch_size < 1: raise ValueError("batch candidates must be positive")
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
            legacy._score_batch(model, tokenizer, [sample] * batch_size, yes=int(token_info["yes_token_id"]), no=int(token_info["no_token_id"]), device="cuda")
            free, total = torch.cuda.mem_get_info(); elapsed = time.monotonic() - started; ok = int(free) >= reserve
            attempts.append({"batch_size": batch_size, "ok": ok, "elapsed_seconds": elapsed, "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()), "free_after_bytes": int(free), "total_bytes": int(total), "reserve_bytes": reserve})
            if ok: accepted.append(batch_size)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); attempts.append({"batch_size": batch_size, "ok": False, "oom": True}); break
    if not accepted: raise RuntimeError("R1 Qwen c35 cache OOM or VRAM reserve violation at batch 1")
    return max(accepted), {"kind": f"{R1_KIND}_gpu_probe", "attempts": attempts, "batch_size": max(accepted), "vram_reserve_gib": args.vram_reserve_gib, "device": "cuda", "precision": "bf16"}


def _load_cache(path: Path, *, expected_lineage: str, expected: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    rows = _read_jsonl(path); result: dict[tuple[str, str], dict[str, Any]] = {}
    by_qid: dict[str, set[str]] = {}
    for row in rows:
        key = (str(row.get("qid", "")), str(row.get("doc_id", "")))
        value = row.get("yes_no_logit")
        if key not in expected or key in result or row.get("lineage_sha256") != expected_lineage or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise V27ContractError(f"R1 cache has invalid/duplicate/stale row: {key}")
        result[key] = row; by_qid.setdefault(key[0], set()).add(key[1])
    expected_by_qid: dict[str, set[str]] = {}
    for qid, doc_id in expected: expected_by_qid.setdefault(qid, set()).add(doc_id)
    complete_qids = {qid for qid, docs in by_qid.items() if docs == expected_by_qid[qid]}
    kept = {key: row for key, row in result.items() if key[0] in complete_qids}
    if len(kept) != len(result): _write_jsonl(path, [kept[key] for key in sorted(kept)])
    return kept


def _pilot_qid_list(qids: Sequence[str], count: int, seed: int) -> list[str]:
    return sorted(map(str, qids), key=lambda qid: hashlib.sha256(f"{seed}:{qid}".encode("utf-8")).hexdigest())[:count]


def run(args: argparse.Namespace) -> dict[str, Any]:
    _cap_threads(args.cpu_threads); require_contexts_zip(args.contexts_zip)
    universe = load_candidate_universe(stage1=args.stage1, stage2=args.stage2, anchor=args.anchor, reference=None)
    qids = _shard_qids([str(qid) for qid in universe.split.qids], args.shard_index, args.shard_count)
    if args.mode == "pilot": qids = _pilot_qid_list(qids, args.pilot_qids, args.pilot_seed)
    if not args.qwen_model.is_dir(): raise FileNotFoundError(f"Missing Qwen model snapshot: {args.qwen_model}")
    fingerprint = checkpoint_fingerprint(args.qwen_model)
    lineage = qwen_lineage(stage1=args.stage1, stage2=args.stage2, anchor=args.anchor, contexts_zip=args.contexts_zip, model_fingerprint=fingerprint, model_revision=args.model_revision)
    lineage_sha = json_sha256(lineage)
    root = args.output_dir / "r1_qwen_cache"; write_json(root / "r1_qwen_label_blind_manifest.json", {"kind": f"{R1_KIND}_qwen_manifest", "schema_version": 1, "lineage": lineage, "lineage_sha256": lineage_sha, "shard_index": args.shard_index, "shard_count": args.shard_count, "qids": qids, "planned_pairs": len(qids) * CANDIDATE_LIMIT, "holdout_opened": False, "public_labels_used": False})
    if args.mode == "preflight": return {"kind": f"{R1_KIND}_qwen_preflight", "lineage": lineage, "lineage_sha256": lineage_sha, "queries": len(qids), "planned_pairs": len(qids) * CANDIDATE_LIMIT, "holdout_opened": False, "public_labels_used": False}
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("R1 Qwen cache requires CUDA")
    model, tokenizer, token_info = legacy._load_tokenizer_model(argparse.Namespace(qwen_model=args.qwen_model, device="cuda", attention="sdpa"))
    actual_qwen_parameters = _unique_parameter_count(model)
    combined_parameters = int(FROZEN_UPSTREAM_PARAMETERS + actual_qwen_parameters)
    if combined_parameters >= PARAMETER_LIMIT:
        raise V27ContractError(f"R1 parameter budget exceeded: {combined_parameters} >= {PARAMETER_LIMIT}")
    score_path, metadata_path, progress_path = _paths(args)
    docs = load_raw_documents(args.contexts_zip, needed_document_ids(universe, limit=CANDIDATE_LIMIT))
    qid_to_index = {str(qid): index for index, qid in enumerate(universe.split.qids)}
    expected = {(qid, str(doc_id)): {} for qid in qids for doc_id in pool_for(universe, qid, R1_VARIANT)}
    probe_qids = sorted(qids, key=lambda qid: max(len(docs[doc_id].body) + len(docs[doc_id].title) for doc_id in pool_for(universe, qid, R1_VARIANT)), reverse=True)[:16]
    probe_records = [_records(tokenizer, universe, docs, qid, qid_to_index[qid]) for qid in probe_qids]
    sample = max((record for records in probe_records for record in records), key=lambda row: len(row["input_ids"]))
    batch_size, probe = _probe(model, tokenizer, sample, args, token_info)
    parameter_ledger = {"frozen_upstream_parameters": int(FROZEN_UPSTREAM_PARAMETERS), "qwen_parameters": actual_qwen_parameters, "combined_parameters": combined_parameters, "limit": int(PARAMETER_LIMIT), "under_4b": combined_parameters < PARAMETER_LIMIT}
    score_path.parent.mkdir(parents=True, exist_ok=True); metadata = {"kind": f"{R1_KIND}_qwen_cache", "schema_version": 1, "lineage": lineage, "lineage_sha256": lineage_sha, "shard_index": args.shard_index, "shard_count": args.shard_count, "token_scoring": token_info, "runtime_probe": probe, "parameter_ledger": parameter_ledger, "mode": args.mode, "holdout_opened": False, "public_labels_used": False}
    if metadata_path.is_file():
        old = json.loads(metadata_path.read_text(encoding="utf-8"));
        if old.get("lineage_sha256") != lineage_sha: raise V27ContractError("R1 cache metadata lineage changed")
    write_json(metadata_path, metadata)
    existing = _load_cache(score_path, expected_lineage=lineage_sha, expected=expected) if args.resume else {}
    if score_path.is_file() and not args.resume: raise FileExistsError(f"R1 cache exists with --no-resume: {score_path}")
    complete_qids = [qid for qid in qids if {(qid, str(doc_id)) for doc_id in pool_for(universe, qid, R1_VARIANT)}.issubset(set(existing))]
    pending_qids = [qid for qid in qids if qid not in complete_qids]; started = time.monotonic(); written = dict(existing); pairs_done = len(written)
    try:
        for qid_number, qid in enumerate(pending_qids, 1):
            records = _records(tokenizer, universe, docs, qid, qid_to_index[qid])
            pending = [row for row in records if (str(row["qid"]), str(row["doc_id"])) not in written]
            completed_rows: list[dict[str, Any]] = []
            for start in range(0, len(pending), batch_size):
                batch = pending[start:start + batch_size]
                try:
                    values = legacy._score_batch(model, tokenizer, batch, yes=int(token_info["yes_token_id"]), no=int(token_info["no_token_id"]), device="cuda")
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache(); raise RuntimeError(f"R1 OOM while scoring QID {qid}; lower --batch-candidates or use resume")
                for row, (logit, probability) in zip(batch, values, strict=True):
                    material = {"qid": str(row["qid"]), "doc_id": str(row["doc_id"]), "candidate_rank": int(row["candidate_rank"]), "yes_no_logit": float(logit), "yes_probability": float(probability), "input_sha256": row["input_sha256"], "pack": row["pack"], "prompt": row["prompt"], "packed_source_tokens": row["packed_source_tokens"], "packed_tokens": row["packed_tokens"], "packed_document_empty": row["packed_document_empty"], "lineage_sha256": lineage_sha}
                    completed_rows.append(material); written[(material["qid"], material["doc_id"])] = material
            if len(completed_rows) != len(pending): raise V27ContractError(f"R1 Qwen QID {qid} was not fully scored")
            # Appending a whole QID is the resume boundary.  If the process
            # dies mid-QID, _load_cache discards that partial QID on restart.
            _append_jsonl(score_path, sorted(completed_rows, key=lambda row: int(row["candidate_rank"])))
            complete_qids.append(qid)
            write_json(progress_path, {"kind": f"{R1_KIND}_qwen_progress", "lineage_sha256": lineage_sha, "completed_qids": sorted(complete_qids), "complete": len(written) == len(expected)})
            pairs_done = len(written); print(json.dumps({"event": "r1_qwen_progress", "mode": args.mode, "qid": qid, "qids_done": len(complete_qids), "qids_total": len(qids), "pairs_done": pairs_done, "pairs_total": len(expected), "elapsed_seconds": time.monotonic() - started}, ensure_ascii=False), flush=True)
    finally:
        del model, tokenizer; gc.collect(); torch.cuda.empty_cache()
    if set(written) != set(expected): raise V27ContractError("R1 Qwen cache does not cover all c35 rows")
    report = {"kind": f"{R1_KIND}_qwen_cache_report", "mode": args.mode, "score_path": str(score_path.resolve()), "metadata_path": str(metadata_path.resolve()), "progress_path": str(progress_path.resolve()), "lineage_sha256": lineage_sha, "pairs": len(written), "qids": len(qids), "batch_size": batch_size, "runtime_probe": probe, "parameter_ledger": parameter_ledger, "elapsed_seconds": time.monotonic() - started, "holdout_opened": False, "public_labels_used": False}
    write_json(root / f"r1_qwen_{args.mode}_report_shard_{args.shard_index:03d}_of_{args.shard_count:03d}.json", report)
    return report


if __name__ == "__main__":
    raise SystemExit((lambda args: (print(json.dumps(run(args), ensure_ascii=False)), 0)[1])(parse_args()))
