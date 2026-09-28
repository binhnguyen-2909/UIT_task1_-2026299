#!/usr/bin/env python3
"""Materialize PV1 F_SHORTLIST (PV1 schema) from local F_shortlist8_long cache.

Reproduces the EXACT long F packs with the same packer/template as
qwen_shortlist8_long_cache but uses the `tokenizers` library directly
(HF `transformers`/`torch` import hangs on this shared box right now).
Verifies pack metadata against the F cache shard. Complement-only.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from collections import defaultdict
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TASK_ROOT / "scripts"))

from qwen_c200_common import (  # noqa: E402
    INSTRUCTION,
    load_candidate_universe,
    load_raw_documents,
    pack_document,
)
from v27_a_common import V27ContractError, sha256_file  # noqa: E402

PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
LONG_MAX_TOKENS = 4096


class _TokAdapter:
    """Minimal HF-compatible wrapper around tokenizers.Tokenizer."""

    def __init__(self, path: Path) -> None:
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(str(path / "tokenizer.json"))

    def __call__(self, text, add_special_tokens=False):  # noqa: ANN001, ANN202
        if isinstance(text, list):
            return {"input_ids": [self._tok.encode(item).ids for item in text]}
        return {"input_ids": self._tok.encode(str(text)).ids}

    def encode(self, text: str, add_special_tokens=False) -> list[int]:  # noqa: ANN001
        return list(self._tok.encode(str(text)).ids)

    def decode(self, ids, skip_special_tokens=True) -> str:  # noqa: ANN001
        return str(self._tok.decode(list(ids), skip_special_tokens=bool(skip_special_tokens)))


def _packer_sha() -> str:
    import qwen_c200_common as common

    blob = (
        inspect.getsource(common.pack_document)
        + "\n---\n"
        + INSTRUCTION
        + "\n---\n"
        + PREFIX
        + "\n---\n"
        + SUFFIX
        + "\n---\n"
        + str(LONG_MAX_TOKENS)
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _read_f_shards(cache_dir: Path) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = defaultdict(list)
    for path in sorted((cache_dir / "shards").glob("*.jsonl")):
        with path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows[str(row["qid"])].append(row)
    for qid in rows:
        rows[qid].sort(key=lambda r: int(r["shortlist_index"]))
    return dict(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--contexts-zip", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--f-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-qids", type=int, default=0)
    args = parser.parse_args()

    print("loading universe...", flush=True)
    uni = load_candidate_universe(
        stage1=args.stage1, stage2=args.stage2, anchor=args.anchor, reference=args.reference
    )
    qids = [str(q) for q in uni.split.qids]
    if args.max_qids > 0:
        qids = qids[: args.max_qids]
    qid_to_index = {str(qid): i for i, qid in enumerate(uni.split.qids)}
    print(f"qids={len(qids)}", flush=True)
    f_rows = _read_f_shards(args.f_cache_dir)
    missing = set(qids) - set(f_rows)
    if missing:
        raise V27ContractError(f"F cache misses {len(missing)} qids: {sorted(missing)[:5]}")

    wanted: set[str] = set()
    for qid in qids:
        recs = f_rows[qid]
        if len(recs) != 8:
            raise V27ContractError(f"{qid}: F shortlist must hold 8 docs, got {len(recs)}")
        for rec in recs:
            wanted.add(str(rec["doc_id"]))
    print(f"loading {len(wanted)} raw docs...", flush=True)
    docs = load_raw_documents(args.contexts_zip, wanted)
    print("loading tokenizer...", flush=True)
    tokenizer = _TokAdapter(args.tokenizer_dir)
    print("tokenizer ok", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "frozen_f_shortlist_with_packs.jsonl"
    done: set[str] = set()
    if out_path.is_file():
        with out_path.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    done.add(str(json.loads(line)["qid"]))
        print(f"resume: {len(done)} qids already materialized", flush=True)

    packer_sha = _packer_sha()
    mismatches = 0
    written = 0
    with out_path.open("a", encoding="utf-8") as target:
        for position, qid in enumerate(qids):
            if qid in done:
                continue
            question = str(uni.split.questions[qid_to_index[qid]])
            query = f"<Instruct>: {INSTRUCTION}\n<Query>: {question}\n<Document>: "
            budget = (
                LONG_MAX_TOKENS
                - len(tokenizer.encode(PREFIX))
                - len(tokenizer.encode(query))
                - len(tokenizer.encode(SUFFIX))
            )
            shortlist = []
            packed = []
            for rec in f_rows[qid]:
                doc_id = str(rec["doc_id"])
                packed_text, meta = pack_document(
                    tokenizer, question=question, document=docs[doc_id], max_tokens=budget
                )
                expected = rec.get("pack", {})
                if (
                    meta.get("packed_tokens") != expected.get("packed_tokens")
                    or meta.get("kind") != expected.get("kind")
                ):
                    mismatches += 1
                    if mismatches <= 5:
                        print(
                            f"pack mismatch {qid}/{doc_id}: got={meta} expected={expected}",
                            flush=True,
                        )
                shortlist.append(
                    {
                        "doc_id": doc_id,
                        "f_rank": int(rec["shortlist_index"]),
                        "f_score": float(rec["yes_no_logit"]),
                    }
                )
                packed.append({"doc_id": doc_id, "packed_text": packed_text})
            target.write(
                json.dumps(
                    {"qid": qid, "shortlist": shortlist, "packed_documents": packed},
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
            if (position + 1) % 50 == 0 or (position + 1) == len(qids):
                print(
                    f"progress {position + 1}/{len(qids)} mismatches={mismatches}",
                    flush=True,
                )
    print(json.dumps({"written": written, "pack_mismatches": mismatches, "packer_sha256": packer_sha}))
    manifest = {
        "lineage": {
            "anchor_sha256": sha256_file(args.anchor),
            "stage1_sha256": sha256_file(args.stage1),
            "split": "complement",
            "packer_sha256": packer_sha,
            "f_cache_lineage": "F_shortlist8_long",
        }
    }
    manifest_path = args.output_dir / "frozen_f_source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest: {manifest_path}", flush=True)
    if mismatches:
        raise SystemExit(f"pack verification mismatches: {mismatches}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
