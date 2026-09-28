#!/usr/bin/env python3
"""GPU scorer for PV1's frozen Qwen3-1.7B pairwise verifier.

This program consumes only a label-blind pair manifest.  It uses first-token
logits for A/B/C/D, scores both document orders, and writes append-safe cache
shards.  It never imports qrels, E/F reports, splits, or a promotion gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pv1_contract import (  # noqa: E402
    DEFAULT_EF_PARAMETERS, DEFAULT_V16B_PARAMETERS, KIND, MAX_TOTAL_PARAMETERS,
    PROMPT_SHA256, PROMPT_VERSION, SYSTEM_PROMPT, USER_TEMPLATE, load_pair_manifest,
)
from v27_a_common import V27ContractError, json_sha256, read_json, sha256_file, write_json  # noqa: E402


SCORE_KIND = f"{KIND}_score_cache"
SCORE_SCHEMA_VERSION = 1
LABELS = ("A", "B", "C", "D")


def pair_margin(logits: Mapping[str, float], challenger_label: str) -> float:
    """Compatibility helper: challenger logit minus logsumexp of the other three labels."""
    if challenger_label not in LABELS:
        raise ValueError(f"unknown PV1 label {challenger_label}")
    values = {label: float(logits[label]) for label in LABELS}
    others = [value for label, value in values.items() if label != challenger_label]
    maximum = max(others)
    logsumexp = maximum + math.log(sum(math.exp(value - maximum) for value in others))
    return float(values[challenger_label] - logsumexp)


def challenger_utility(forward: Mapping[str, float], reverse: Mapping[str, float]) -> float:
    """Compatibility helper: challenger-only wins in both orders, utility is min margin."""
    fwd = {label: float(forward[label]) for label in LABELS}
    rev = {label: float(reverse[label]) for label in LABELS}
    if max(fwd, key=fwd.get) != "B" or max(rev, key=rev.get) != "A":
        return float("-inf")
    return float(min(pair_margin(fwd, "B"), pair_margin(rev, "A")))


def decide_slot5(anchor: Sequence[str], scored: Sequence[tuple[str, int, float]], policy: Mapping[str, Any]) -> list[str]:
    """Compatibility helper: strict threshold decision preserving frozen top-4."""
    anchor_list = [str(value) for value in anchor]
    if len(anchor_list) != 5 or len(set(anchor_list)) != 5:
        raise V27ContractError("PV1 anchor must hold five unique document IDs")
    if policy.get("mode") == "abstain":
        return list(anchor_list)
    if not scored:
        return list(anchor_list)
    for challenger_id, _rank, _utility in scored:
        if str(challenger_id) in set(anchor_list):
            raise V27ContractError(f"PV1 challenger {challenger_id} collides with anchor")
    tau = float(policy.get("tau", 0.0))
    ranked = sorted(scored, key=lambda item: (-float(item[2]), int(item[1]), str(item[0])))
    best_id, _best_rank, best_utility = ranked[0]
    if float(best_utility) > tau:
        answer = anchor_list[:4] + [str(best_id)]
        if len(set(answer)) != 5 or answer[:4] != anchor_list[:4]:
            raise V27ContractError("PV1 violated frozen top-4/unique top-5 contract")
        return answer
    return list(anchor_list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "cache"), required=True)
    parser.add_argument("--pairs", type=Path, required=True); parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True); parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-input-tokens", type=int, default=12_288)
    parser.add_argument("--pilot-qids", type=int, default=64); parser.add_argument("--pilot-seed", type=int, default=20_260_915)
    parser.add_argument("--batch-candidates", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--vram-reserve-gib", type=float, required=True)
    parser.add_argument("--shard-index", type=int, default=0); parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=1); parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false"); parser.set_defaults(resume=True)
    return parser.parse_args()


def _cap_threads(value: int) -> None:
    if value < 1:
        raise ValueError("--cpu-threads must be positive")
    text = str(value)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
        os.environ[key] = text


def _score_root(args: argparse.Namespace) -> Path:
    return args.output_dir / "pv1_pair_scores"


def _paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    root = _score_root(args); suffix = f"shard-{args.shard_index:03d}-of-{args.shard_count:03d}"
    return root / "shards" / f"pv1_scores_{suffix}.jsonl", root / "metadata" / f"pv1_scores_{suffix}.json", root / "progress" / f"pv1_scores_{suffix}.json"


def _fingerprint_model(path: Path) -> dict[str, str]:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing local verifier model snapshot: {path}")
    names = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "model.safetensors.index.json")
    output = {name: sha256_file(path / name) for name in names if (path / name).is_file()}
    weights = sorted(path.glob("*.safetensors"))
    if "config.json" not in output or (not weights and "model.safetensors.index.json" not in output):
        raise FileNotFoundError(f"Incomplete Qwen3 verifier snapshot: {path}")
    # Hash individual weights only for an unsharded snapshot.  A sharded index
    # cryptographically binds the shard names and is preserved in the bundle.
    if len(weights) == 1:
        output[weights[0].name] = sha256_file(weights[0])
    return output


def _shard_for(qid: str, count: int) -> int:
    return int(hashlib.sha256(qid.encode("utf-8")).hexdigest()[:16], 16) % count


def _key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return tuple(str(row.get(name, "")) for name in ("qid", "incumbent_id", "challenger_id", "orientation"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise V27ContractError(f"{path}:{number}: invalid score JSONL") from exc
            if not isinstance(row, dict):
                raise V27ContractError(f"{path}:{number}: invalid score cache row")
            rows.append(row)
    return rows


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        target.flush(); os.fsync(target.fileno())
    os.replace(temporary, path)


def _recover(path: Path, *, expected: Sequence[Mapping[str, Any]], run_hash: str, resume: bool) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    existing = _read_jsonl(path)
    if existing and not resume:
        raise FileExistsError(f"PV1 score cache exists but --no-resume was selected: {path}")
    expected_by_key = {_key(row): row for row in expected}
    completed_by_qid: dict[str, set[tuple[str, str, str, str]]] = {}
    valid: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in existing:
        key = _key(row); logits = row.get("logits_abcd")
        if key not in expected_by_key or row.get("run_hash") != run_hash or row.get("input_sha256") != expected_by_key.get(key, {}).get("input_sha256") or not isinstance(logits, list) or len(logits) != 4:
            raise V27ContractError(f"{path}: score row does not match PV1 run contract")
        if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in logits):
            raise V27ContractError(f"{path}: non-finite PV1 score logit")
        if key in valid:
            raise V27ContractError(f"{path}: duplicate PV1 score key {key}")
        valid[key] = row; completed_by_qid.setdefault(key[0], set()).add(key)
    expected_qid_keys: dict[str, set[tuple[str, str, str, str]]] = {}
    for key in expected_by_key:
        expected_qid_keys.setdefault(key[0], set()).add(key)
    complete_qids = {qid for qid, keys in completed_by_qid.items() if keys == expected_qid_keys[qid]}
    # A process can die halfway through a QID.  Remove that entire QID so a
    # resumed job never calls a partially cached pair complete.
    kept = {key: row for key, row in valid.items() if key[0] in complete_qids}
    if len(kept) != len(valid):
        _write_jsonl_atomic(path, [kept[key] for key in sorted(kept)])
    return kept


def _messages(row: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(query=row["query"], document_1=row["document_1_text"], document_2=row["document_2_text"])},
    ]


def _render(tokenizer: Any, row: Mapping[str, Any]) -> str:
    try:
        return tokenizer.apply_chat_template(_messages(row), tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError as exc:
        raise V27ContractError("Installed tokenizer does not support Qwen3 enable_thinking=False; pin a compatible Transformers/Qwen snapshot") from exc


def _label_ids(tokenizer: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for label in LABELS:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise V27ContractError(f"PV1 label {label!r} is not a single continuation token: {ids}")
        result[label] = int(ids[0])
    if len(set(result.values())) != len(result):
        raise V27ContractError("PV1 A/B/C/D continuation tokens are not distinct")
    return result


def _prepare(tokenizer: Any, rows: Sequence[Mapping[str, Any]], max_tokens: int, device: str) -> tuple[Any, list[int], list[str]]:
    prompts = [_render(tokenizer, row) for row in rows]
    encoded_lengths = [len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) for prompt in prompts]
    if max(encoded_lengths, default=0) > max_tokens:
        over = next(row for row, length in zip(rows, encoded_lengths, strict=True) if length > max_tokens)
        raise V27ContractError(f"{over['qid']}/{over['challenger_id']}: PV1 prompt exceeds {max_tokens} tokens; truncation is forbidden")
    batch = tokenizer(prompts, add_special_tokens=False, padding=True, truncation=False, return_tensors="pt")
    return {key: value.to(device) for key, value in batch.items()}, encoded_lengths, prompts


def _forward(model: Any, tokenizer: Any, rows: Sequence[Mapping[str, Any]], label_ids: Mapping[str, int], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[int]]:
    import torch

    batch, lengths, _ = _prepare(tokenizer, rows, args.max_input_tokens, args.device)
    try:
        with torch.inference_mode():
            output = model(**batch, use_cache=False, logits_to_keep=1)
    except TypeError as exc:
        raise V27ContractError("Installed Transformers model does not implement logits_to_keep=1; do not silently change scorer semantics") from exc
    logits = output.logits[:, -1, :].float()
    token_ids = [label_ids[label] for label in LABELS]
    selected = logits[:, token_ids]
    probabilities = torch.softmax(logits, dim=-1)[:, token_ids].sum(dim=-1)
    rows_out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        values = [float(value) for value in selected[index].detach().cpu().tolist()]
        rows_out.append({
            "qid": str(row["qid"]), "incumbent_id": str(row["incumbent_id"]), "challenger_id": str(row["challenger_id"]),
            "orientation": str(row["orientation"]), "f_rank": int(row["f_rank"]), "f_score": float(row["f_score"]),
            "input_sha256": str(row["input_sha256"]), "num_tokens": int(lengths[index]),
            "label_token_ids": {label: int(label_ids[label]) for label in LABELS}, "logits_abcd": values,
            "label_mass": float(probabilities[index].detach().cpu()),
        })
    return rows_out, lengths


def _probe(model: Any, tokenizer: Any, rows: Sequence[Mapping[str, Any]], label_ids: Mapping[str, int], args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    import torch

    if not rows:
        return 1, {"attempts": [], "chosen_batch_size": 1, "reason": "no_pairs"}
    longest = sorted(rows, key=lambda row: len(str(row["query"])) + len(str(row["document_1_text"])) + len(str(row["document_2_text"])), reverse=True)
    attempts: list[dict[str, Any]] = []; accepted: list[int] = []
    for batch_size in sorted(set(args.batch_candidates)):
        if batch_size < 1:
            raise ValueError("--batch-candidates must be positive")
        trial = [longest[index % len(longest)] for index in range(batch_size)]
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            started = time.monotonic(); _, lengths = _forward(model, tokenizer, trial, label_ids, args)
            elapsed = time.monotonic() - started
            peak = int(torch.cuda.max_memory_allocated())
            free, total = torch.cuda.mem_get_info()
            reserve = int(args.vram_reserve_gib * 1024 ** 3)
            ok = free >= reserve
            attempts.append({"batch_size": batch_size, "ok": ok, "elapsed_seconds": elapsed, "max_tokens": max(lengths), "peak_allocated_bytes": peak, "free_after_bytes": int(free), "total_bytes": int(total), "reserve_bytes": reserve})
            if ok:
                accepted.append(batch_size)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); attempts.append({"batch_size": batch_size, "ok": False, "oom": True})
    if not accepted:
        raise RuntimeError("PV1 verifier OOM or violates reserve at batch size 1")
    return max(accepted), {"kind": f"{KIND}_gpu_probe", "attempts": attempts, "chosen_batch_size": max(accepted), "reserve_gib": args.vram_reserve_gib}


def _pilot_rows(rows: Sequence[Mapping[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    qids = sorted({str(row["qid"]) for row in rows}, key=lambda qid: hashlib.sha256(f"{seed}:{qid}".encode("utf-8")).hexdigest())[:count]
    chosen = set(qids)
    return [dict(row) for row in rows if str(row["qid"]) in chosen]


def _parameter_count(model: Any) -> int:
    seen: set[int] = set(); total = 0
    for parameter in model.parameters():
        pointer = int(parameter.data_ptr())
        if pointer not in seen:
            total += int(parameter.numel()); seen.add(pointer)
    return total


def _reload_verify(*, args: argparse.Namespace, target_by_key: Mapping[tuple[str, str, str, str], Mapping[str, Any]], written: Mapping[tuple[str, str, str, str], Mapping[str, Any]], label_ids: Mapping[str, int], model: Any) -> dict[str, Any]:
    """Reload from the immutable snapshot and reproduce one stored decision.

    This executes only after a complete cache shard.  It deliberately creates
    a fresh model object rather than trusting the in-memory scorer which wrote
    the cache.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    key = sorted(written)[0]; expected = written[key]; source = target_by_key[key]
    del model; torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    fresh_ids = _label_ids(tokenizer)
    if dict(fresh_ids) != dict(label_ids):
        raise V27ContractError("PV1 verifier reload changed A/B/C/D token IDs")
    fresh = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16).to(args.device).eval()
    actual, _ = _forward(fresh, tokenizer, [source], fresh_ids, args)
    del fresh; torch.cuda.empty_cache()
    got = actual[0]["logits_abcd"]; reference = expected["logits_abcd"]
    atol, rtol = 1.0e-4, 1.0e-4
    if not all(math.isclose(float(left), float(right), abs_tol=atol, rel_tol=rtol) for left, right in zip(got, reference, strict=True)):
        raise V27ContractError("PV1 verifier reload logits differ from the completed score cache")
    signature = {"argmax": LABELS[max(range(4), key=lambda index: float(got[index]))], "input_sha256": source["input_sha256"]}
    expected_signature = {"argmax": LABELS[max(range(4), key=lambda index: float(reference[index]))], "input_sha256": source["input_sha256"]}
    if signature != expected_signature:
        raise V27ContractError("PV1 verifier reload decision differs from the completed score cache")
    return {"complete": True, "key": {"qid": key[0], "incumbent_id": key[1], "challenger_id": key[2], "orientation": key[3]}, "expected_logits": reference, "reloaded_logits": got,
            "atol": atol, "rtol": rtol, "decision": signature}


def run(args: argparse.Namespace) -> dict[str, Any]:
    _cap_threads(args.cpu_threads)
    if args.device != "cuda" or args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("PV1 scorer requires CUDA and a valid shard index/count")
    pair_rows, pair_manifest = load_pair_manifest(args.pair_manifest, args.pairs)
    target = [row for row in pair_rows if _shard_for(str(row["qid"]), args.shard_count) == args.shard_index]
    if args.mode == "pilot":
        target = _pilot_rows(target, args.pilot_qids, args.pilot_seed)
    if not target:
        raise V27ContractError("This PV1 shard contains no pair rows")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PV1 pair scorer needs an available CUDA GPU")
    fingerprint = _fingerprint_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16).to(args.device).eval()
    label_ids = _label_ids(tokenizer)
    actual_parameters = _parameter_count(model)
    total_parameters = DEFAULT_V16B_PARAMETERS + DEFAULT_EF_PARAMETERS + actual_parameters
    if total_parameters >= MAX_TOTAL_PARAMETERS:
        raise V27ContractError(f"PV1 actual parameter ledger exceeds limit: {total_parameters}")
    run_contract = {"kind": SCORE_KIND, "schema_version": SCORE_SCHEMA_VERSION, "pair_lineage_sha256": pair_manifest["lineage_sha256"],
                    "model_path": str(args.model.resolve()), "model_revision": args.model_revision, "model_fingerprint": fingerprint,
                    "prompt_version": PROMPT_VERSION, "prompt_sha256": PROMPT_SHA256, "label_token_ids": label_ids,
                    "max_input_tokens": args.max_input_tokens, "dtype": "bfloat16", "use_cache": False, "logits_to_keep": 1,
                    "padding_side": "left", "enable_thinking": False, "shard_index": args.shard_index, "shard_count": args.shard_count}
    run_hash = json_sha256(run_contract)
    score_path, metadata_path, progress_path = _paths(args)
    if metadata_path.exists():
        previous = read_json(metadata_path)
        if not isinstance(previous, dict) or previous.get("run_hash") != run_hash:
            raise V27ContractError("Existing PV1 score metadata differs from current model/prompt/pair contract")
    score_path.parent.mkdir(parents=True, exist_ok=True); metadata_path.parent.mkdir(parents=True, exist_ok=True); progress_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"kind": SCORE_KIND, "schema_version": SCORE_SCHEMA_VERSION, "run_contract": run_contract, "run_hash": run_hash,
                "pair_manifest": str(args.pair_manifest.resolve()), "pair_count_target": len(target), "mode": args.mode,
                "parameter_ledger": {"v16b": DEFAULT_V16B_PARAMETERS, "ef_reranker": DEFAULT_EF_PARAMETERS, "verifier_actual": actual_parameters, "total": total_parameters, "limit_exclusive": MAX_TOTAL_PARAMETERS, "under_4b": True},
                "holdout_opened": False, "public_labels_used": False}
    write_json(metadata_path, metadata)
    batch_size, probe = _probe(model, tokenizer, target, label_ids, args)
    existing = _recover(score_path, expected=target, run_hash=run_hash, resume=args.resume)
    expected_keys = {_key(row) for row in target}; pending = [row for row in target if _key(row) not in existing]
    started = time.monotonic(); lengths: list[int] = []; written = dict(existing)
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        scored, batch_lengths = _forward(model, tokenizer, batch, label_ids, args)
        for row in scored:
            row["run_hash"] = run_hash; written[_key(row)] = row
        lengths.extend(batch_lengths)
        _write_jsonl_atomic(score_path, [written[key] for key in sorted(written)])
        complete_qids = sorted({key[0] for key in written if {candidate for candidate in expected_keys if candidate[0] == key[0]}.issubset(set(written))})
        write_json(progress_path, {"kind": f"{SCORE_KIND}_progress", "run_hash": run_hash, "completed_qids": complete_qids, "complete": len(written) == len(expected_keys)})
        print(json.dumps({"event": "pv1_score_progress", "mode": args.mode, "shard": f"{args.shard_index}/{args.shard_count}", "pairs_done": len(written), "pairs_total": len(expected_keys), "elapsed_seconds": round(time.monotonic() - started, 3)}, ensure_ascii=False), flush=True)
    if set(written) != expected_keys:
        raise V27ContractError("PV1 score cache did not cover every expected pair")
    reload_verification = None
    if args.mode == "cache":
        target_by_key = {_key(row): row for row in target}
        reload_verification = _reload_verify(args=args, target_by_key=target_by_key, written=written, label_ids=label_ids, model=model)
        verification_path = _score_root(args) / f"pv1_reload_verification_shard-{args.shard_index:03d}-of-{args.shard_count:03d}.json"
        write_json(verification_path, reload_verification)
    report = {"kind": f"{SCORE_KIND}_report", "mode": args.mode, "score_path": str(score_path.resolve()), "metadata_path": str(metadata_path.resolve()),
              "progress_path": str(progress_path.resolve()), "run_hash": run_hash, "pairs": len(written), "runtime_probe": probe,
              "token_lengths": {"minimum": min(lengths) if lengths else None, "maximum": max(lengths) if lengths else None, "observed_pairs": len(lengths)},
              "elapsed_seconds": time.monotonic() - started, "reload_verification": reload_verification,
              "holdout_opened": False, "public_labels_used": False}
    write_json(_score_root(args) / f"pv1_{args.mode}_score_report_shard-{args.shard_index:03d}-of-{args.shard_count:03d}.json", report)
    return report


def main() -> int:
    args = parse_args(); report = run(args)
    print(json.dumps({"output": report["score_path"], "stage": args.mode, "pairs": report["pairs"], "holdout_opened": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
