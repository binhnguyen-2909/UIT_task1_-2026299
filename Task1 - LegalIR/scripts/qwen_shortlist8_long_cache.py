#!/usr/bin/env python3
"""F_shortlist8_long 4096 cache: max 8 docs/query from deterministic shortlist.

Complement-only. Shortlist comes from qwen_shortlist8.select_shortlist8
(incumbent + 4 top-short + 3 from 36-67, dedup, fill by short score).
Long scoring reuses the Qwen checkpoint/instruction with 4096 packs.
Own manifest; never presents D_c200_long lineage as F.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwen_c200_cache import (  # noqa: E402  same scoring path as E short cache
    PREFIX,
    SUFFIX,
    _probe_batch_size,
    _score_batch,
)
from qwen_c200_common import (  # noqa: E402
    INSTRUCTION,
    QWEN_BRANCH_KIND,
    cap_cpu_threads,
    checkpoint_fingerprint,
    load_candidate_universe,
    load_raw_documents,
    pack_document,
    pool_for,
    read_score_cache,
    require_complement_file,
    validate_cache_coverage,
    write_jsonl,
)
from qwen_shortlist8 import select_shortlist8  # noqa: E402
from v27_a_common import V27ContractError, json_sha256, sha256_file, write_json  # noqa: E402

VARIANT = "F_shortlist8_long"
SHORT_VARIANT = "E_c67_short"
LONG_MAX_TOKENS = 4096
MAX_DOCS_PER_QUERY = 8
SHORTLIST_POLICY = "incumbent_plus_4top-short_plus_3tail36-67_dedup_short-score-fill"


def preflight(*, output_dir: Path, short_cache_dir: Path) -> dict:
    for path, label in ((short_cache_dir, "short_cache_dir"), (output_dir, "output_dir")):
        value = str(path).replace("\\", "/").casefold()
        if any(term in value for term in ("holdout", "public", "submission", "locked")):
            raise V27ContractError(f"F cache is complement-only; {label} may not refer to holdout/public: {path}")
    return {
        "variant": VARIANT,
        "max_tokens": LONG_MAX_TOKENS,
        "max_docs_per_query": MAX_DOCS_PER_QUERY,
        "holdout_opened": False,
        "public_labels_used": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "score"), default="preflight")
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--contexts-zip", type=Path, required=True)
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--short-cache-dir", type=Path, required=True,
                        help="completed E_c67_short qwen_cache dir (shards+metadata)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--batch-candidates", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--attention", choices=("sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def _shard_qids(qids: Sequence[str], index: int, count: int) -> list[str]:
    if count <= 0 or not 0 <= index < count:
        raise ValueError("shard-index must be within [0, shard-count)")
    return [qid for qid in qids if int(hashlib.sha256(qid.encode("utf-8")).hexdigest(), 16) % count == index]


def _paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    root = args.output_dir / "qwen_cache" / VARIANT
    stem = f"shard_{args.shard_index:03d}_of_{args.shard_count:03d}"
    return root / "shards" / f"{stem}.jsonl", root / "metadata" / f"{stem}.json", root / "progress" / f"{stem}.json"


def _short_scores(short_cache_dir: Path) -> dict[str, dict[str, float]]:
    paths = sorted(short_cache_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"E short cache has no shards: {short_cache_dir}")
    rows = read_score_cache(paths)
    return {qid: {doc: float(row["yes_no_logit"]) for doc, row in docs.items()} for qid, docs in rows.items()}


def _lineage(*, universe: Any, contexts_zip: Path, fingerprint: Mapping[str, str], short_lineage_sha: str) -> dict[str, Any]:
    return {
        "kind": QWEN_BRANCH_KIND, "schema_version": 1, "split": "complement",
        "stage1_sha256": universe.stage1_sha256, "stage2_sha256": universe.stage2_sha256,
        "anchor_sha256": universe.anchor_sha256, "contexts_zip_sha256": sha256_file(contexts_zip),
        "qwen_model": "Qwen/Qwen3-Reranker-0.6B", "qwen_checkpoint_fingerprint": dict(fingerprint),
        "variant": VARIANT, "short_scores_variant": SHORT_VARIANT, "short_scores_lineage_sha256": short_lineage_sha,
        "shortlist_policy": SHORTLIST_POLICY, "max_docs_per_query": MAX_DOCS_PER_QUERY,
        "max_tokens": LONG_MAX_TOKENS, "instruction": INSTRUCTION,
        "candidate_prefix_is_frozen_c35": True, "holdout_opened": False, "public_labels_used": False,
    }


def _shortlist_for_qid(*, universe: Any, scores: Mapping[str, Mapping[str, float]], qid: str) -> tuple[list[str], list[str], str]:
    index = list(universe.split.qids).index(str(qid))
    anchor = [str(v) for v in universe.split.anchor[index]]
    ranking = [str(v) for v in pool_for(universe, str(qid), SHORT_VARIANT)]
    if set(scores.get(str(qid), {})) != set(ranking):
        raise V27ContractError(f"{qid}: E short-score coverage differs from c67 pool; refuse shortlist")
    short = select_shortlist8(str(qid), anchor, ranking, {d: float(scores[str(qid)][d]) for d in ranking})
    if not short or len(short) > MAX_DOCS_PER_QUERY:
        raise V27ContractError(f"{qid}: shortlist must hold 1..8 docs, got {len(short)}")
    if str(anchor[4]) not in short:
        raise V27ContractError(f"{qid}: shortlist dropped the V16b incumbent")
    return short, anchor, str(universe.split.questions[index])


def _records_for_qid(*, tokenizer: Any, universe: Any, docs: Mapping[str, Any],
                     scores: Mapping[str, Mapping[str, float]], qid: str) -> list[dict[str, Any]]:
    short, _, question = _shortlist_for_qid(universe=universe, scores=scores, qid=str(qid))
    query = f"<Instruct>: {INSTRUCTION}\n<Query>: {question}\n<Document>: "
    prefix_ids = tokenizer.encode(PREFIX, add_special_tokens=False)
    query_ids = tokenizer.encode(query, add_special_tokens=False)
    suffix_ids = tokenizer.encode(SUFFIX, add_special_tokens=False)
    budget = LONG_MAX_TOKENS - len(prefix_ids) - len(query_ids) - len(suffix_ids)
    if budget <= 32:
        raise V27ContractError("Qwen prompt/template consumes the entire configured context length")
    template_tokens = len(prefix_ids) + len(query_ids) + len(suffix_ids)
    records = []
    for position, doc_id in enumerate(short, 1):
        packed, pack_meta = pack_document(tokenizer, question=question, document=docs[doc_id], max_tokens=budget)
        doc_ids = tokenizer(packed, add_special_tokens=False).input_ids[:budget]
        records.append({
            "qid": str(qid), "doc_id": str(doc_id), "shortlist_index": position,
            "input_ids": prefix_ids + query_ids + doc_ids + suffix_ids,
            "pack": pack_meta,
            "prompt": {"input_tokens": len(prefix_ids) + len(query_ids) + len(doc_ids) + len(suffix_ids),
                       "document_tokens_after_template": len(doc_ids),
                       "document_token_budget": budget, "template_tokens": template_tokens},
        })
    return records


def _recover_completed_qids(path: Path, *, universe: Any, scores: Mapping[str, Mapping[str, float]]) -> set[str]:
    if not path.is_file():
        return set()
    by_qid: dict[str, dict[str, dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            qid, doc_id = str(row.get("qid", "")), str(row.get("doc_id", ""))
            if not qid or not doc_id or doc_id in by_qid.setdefault(qid, {}):
                raise V27ContractError(f"{path}:{number}: invalid or duplicate F cache row")
            by_qid[qid][doc_id] = dict(row)
    complete = set()
    for qid in by_qid:
        short, _, _ = _shortlist_for_qid(universe=universe, scores=scores, qid=qid)
        if set(by_qid[qid]) == set(short):
            complete.add(qid)
    from qwen_c200_common import write_jsonl as _w
    ordered = []
    for qid in sorted(complete):
        short, _, _ = _shortlist_for_qid(universe=universe, scores=scores, qid=qid)
        ordered.extend(by_qid[qid][doc] for doc in short)
    _w(path, iter(ordered), append=False)
    return complete


def _load_tokenizer_model(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_model, local_files_only=True, padding_side="left", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {"local_files_only": True, "torch_dtype": torch.bfloat16, "attn_implementation": args.attention}
    try:
        model = AutoModelForCausalLM.from_pretrained(args.qwen_model, **kwargs).to(args.device).eval()
    except (ImportError, ValueError):
        if args.attention != "flash_attention_2":
            raise
        kwargs["attn_implementation"] = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(args.qwen_model, **kwargs).to(args.device).eval()
    yes = tokenizer("yes", add_special_tokens=False).input_ids
    no = tokenizer("no", add_special_tokens=False).input_ids
    if len(yes) != 1 or len(no) != 1:
        raise V27ContractError("Qwen reranker yes/no tokens are not single-token IDs")
    return model, tokenizer, {"yes_token_id": int(yes[0]), "no_token_id": int(no[0]), "attention": kwargs["attn_implementation"]}


def run(args: argparse.Namespace) -> dict[str, Any]:
    cap_cpu_threads(args.cpu_threads)
    preflight(output_dir=args.output_dir, short_cache_dir=args.short_cache_dir)
    universe = load_candidate_universe(stage1=args.stage1, stage2=args.stage2, anchor=args.anchor, reference=None)
    require_complement_file(args.contexts_zip, "contexts ZIP")
    fingerprint = checkpoint_fingerprint(args.qwen_model)
    scores = _short_scores(args.short_cache_dir)
    for qid in universe.split.qids:
        if set(scores.get(str(qid), {})) != set(pool_for(universe, str(qid), SHORT_VARIANT)):
            raise V27ContractError(f"{qid}: E short cache incomplete; finish E before F")
    short_lineage_sha = json_sha256({"e_variant": SHORT_VARIANT, "qids": len(scores),
                                     "pairs": sum(len(v) for v in scores.values())})
    lineage = _lineage(universe=universe, contexts_zip=args.contexts_zip, fingerprint=fingerprint,
                       short_lineage_sha=short_lineage_sha)
    lineage_sha = json_sha256(lineage)
    cache_path, meta_path, progress_path = _paths(args)
    if meta_path.is_file():
        prior = json.loads(meta_path.read_text(encoding="utf-8"))
        if prior.get("lineage_sha256") != lineage_sha:
            raise V27ContractError(f"F cache metadata differs from frozen lineage: {meta_path}")
    target_qids = _shard_qids([str(q) for q in universe.split.qids], args.shard_index, args.shard_count)
    if cache_path.is_file() and not args.resume:
        raise FileExistsError(f"Refusing to append with --no-resume: {cache_path}")
    done = _recover_completed_qids(cache_path, universe=universe, scores=scores) if args.resume else set()
    if done - set(target_qids):
        raise V27ContractError("F cache shard contains a QID assigned to another shard")
    model, tokenizer, token_info = _load_tokenizer_model(args)
    try:
        wanted: set[str] = set()
        for qid in target_qids:
            short, _, _ = _shortlist_for_qid(universe=universe, scores=scores, qid=qid)
            wanted.update(short)
        docs = load_raw_documents(args.contexts_zip, wanted)
        todo = [qid for qid in target_qids if qid not in done]
        if todo:
            probe_qids = sorted(todo, key=lambda q: max(len(docs[d].body) for d in _shortlist_for_qid(universe=universe, scores=scores, qid=q)[0]), reverse=True)[:8]
            sample = max((r for q in probe_qids for r in _records_for_qid(tokenizer=tokenizer, universe=universe, docs=docs, scores=scores, qid=q)),
                         key=lambda r: len(r["input_ids"]))
            batch_size, probe = _probe_batch_size(model, tokenizer, sample, args, token_info)
        else:
            batch_size, probe = 4, {"kind": "cache_resume_no_probe", "batch_size": 4}
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps({"kind": "legalir_qwen_c200_score_cache", "schema_version": 1, "lineage": lineage,
                                         "lineage_sha256": lineage_sha, "shard_index": args.shard_index,
                                         "shard_count": args.shard_count, "token_scoring": token_info,
                                         "runtime_probe": probe, "holdout_opened": False,
                                         "public_labels_used": False}, ensure_ascii=False, indent=2), encoding="utf-8")
        import time as _time
        pairs_done = 0
        t_pack = t_fwd = 0.0
        for n, qid in enumerate(todo, 1):
            t0 = _time.perf_counter()
            records = _records_for_qid(tokenizer=tokenizer, universe=universe, docs=docs, scores=scores, qid=qid)
            t1 = _time.perf_counter()
            ordered = sorted(records, key=lambda r: (len(r["input_ids"]), r["shortlist_index"]))
            completed = []
            for start in range(0, len(ordered), batch_size):
                part = ordered[start:start + batch_size]
                values = _score_batch(model, tokenizer, part, yes=int(token_info["yes_token_id"]),
                                      no=int(token_info["no_token_id"]), device=args.device)
                for row, (logit, prob) in zip(part, values, strict=True):
                    completed.append({"qid": row["qid"], "doc_id": row["doc_id"], "shortlist_index": row["shortlist_index"],
                                      "yes_no_logit": float(logit), "yes_probability": float(prob),
                                      "pack": row["pack"], "prompt": row["prompt"], "lineage_sha256": lineage_sha})
            t2 = _time.perf_counter()
            t_pack += t1 - t0
            t_fwd += t2 - t1
            write_jsonl(cache_path, sorted(completed, key=lambda r: r["shortlist_index"]), append=True)
            done.add(qid)
            pairs_done += len(completed)
            write_json(progress_path, {"kind": "legalir_qwen_f_cache_progress", "lineage_sha256": lineage_sha,
                                       "completed_qids": sorted(done), "complete": set(done) == set(target_qids)})
            if n % 10 == 0 or n == len(todo):
                print(json.dumps({"event": "qwen_f_progress", "qids_done": n, "qids_total": len(todo),
                                  "pairs_done": pairs_done, "pack_s_avg": t_pack / n, "fwd_s_avg": t_fwd / n},
                                 ensure_ascii=False), flush=True)
        return {"variant": VARIANT, "qids_complete": len(done), "qids_total": len(target_qids),
                "lineage_sha256": lineage_sha, "complete": set(done) == set(target_qids), "batch_size": batch_size}
    finally:
        del model, tokenizer
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass


def main() -> int:
    args = parse_args()
    if args.mode == "preflight":
        preflight(output_dir=args.output_dir, short_cache_dir=args.short_cache_dir)
        print(json.dumps({"stage": "label_blind_preflight", "holdout_opened": False}, ensure_ascii=False), flush=True)
        return 0
    report = run(args)
    out = args.output_dir / "qwen_f_cache_report.json"
    write_json(out, {**report, "holdout_opened": False, "public_labels_used": False})
    print(json.dumps({"output": str(out.resolve()), "stage": "gpu_cache_complete", "holdout_opened": False}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
