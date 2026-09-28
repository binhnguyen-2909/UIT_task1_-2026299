"""Materialize V25-SV label-free raw document blocks from the audited SQLite corpus.

This stage deliberately does not run a relevance model and does not write qrels or
edit labels into the evidence rows.  Candidate identities come from the frozen
V23-B0 edit beam; the actual evidence remains the raw document text tokenized by
the sealed BGE-Gemma tokenizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v25_sv_common import (  # noqa: E402
    iter_document_texts,
    load_and_verify_edits,
    load_config,
    load_metadata,
    make_blocks,
    sha256_file,
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    questions = load_metadata(config)
    rows_by_qid, force_identity = load_and_verify_edits(config, questions)

    model_dir = Path(str(config["model"]["local_dir"]))
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    tokenizer_path = model_dir / "tokenizer.json"
    if sha256_file(tokenizer_path) != config["tokenizer"]["tokenizer_json_sha256"]:
        raise ValueError("tokenizer.json checksum differs from resolved config")

    required_docs: set[str] = set()
    for qid, rows in rows_by_qid.items():
        if qid in force_identity:
            continue
        for row in rows:
            required_docs.update(str(doc) for doc in row["new_pair"])
        required_docs.update(questions[qid]["anchor"][3:])

    raw = config["raw_evidence"]
    block_path = args.output_dir / "v25_sv_raw_block_manifest.jsonl"
    if args.reuse_existing and block_path.is_file():
        digest = hashlib.sha256()
        document_counts: dict[str, int] = {}
        block_count = 0
        with block_path.open("rb") as source:
            for line_number, encoded in enumerate(source, 1):
                digest.update(encoded)
                row = json.loads(encoded)
                document_id = str(row.get("doc_id", ""))
                if document_id not in required_docs:
                    raise ValueError(f"existing manifest has an unexpected document at line {line_number}")
                if str(row.get("block_id", "")).split("::v25block::", 1)[0] != document_id:
                    raise ValueError(f"existing manifest block/document mismatch at line {line_number}")
                if int(row["token_count"]) != len(row["token_ids"]):
                    raise ValueError(f"existing manifest token count mismatch at line {line_number}")
                document_counts[document_id] = document_counts.get(document_id, 0) + 1
                block_count += 1
        if set(document_counts) != required_docs:
            raise ValueError("existing raw evidence document coverage mismatch")
        block_sha = digest.hexdigest()
        provenance = {
            "kind": "legalir_v25_sv_raw_block_manifest",
            "config_sha256": config["resolved_config_sha256"],
            "model_revision": config["model"]["revision"],
            "tokenizer_sha256": config["tokenizer"]["tokenizer_json_sha256"],
            "source_contexts_database_sha256": config["inputs"]["hashes"]["contexts_database"],
            "source_candidate_pool": "V23-B0 new_pair union plus frozen V16b slots 4-5",
            "qrels_loaded_for_feature_content": False,
            "labels_written": False,
            "holdout_opened": False,
            "public_labels_used": False,
            "force_identity_qids_without_required_packs": sorted(force_identity),
            "document_count": len(required_docs),
            "block_count": block_count,
            "document_block_counts": document_counts,
            "block_manifest_sha256": block_sha,
        }
        write_json(args.output_dir / "v25_sv_raw_block_manifest.provenance.json", provenance)
        print(json.dumps({
            "manifest": str(block_path.resolve()),
            "provenance": str((args.output_dir / "v25_sv_raw_block_manifest.provenance.json").resolve()),
            "documents": len(required_docs),
            "blocks": block_count,
            "force_identity_qids": len(force_identity),
            "sha256": block_sha,
            "holdout_opened": False,
        }, ensure_ascii=False))
        return 0

    document_counts: dict[str, int] = {}
    missing: list[str] = []
    seen_docs: set[str] = set()
    block_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = block_path.with_name(block_path.name + ".tmp")
    digest = hashlib.sha256()
    block_count = 0
    with temporary.open("wb") as target:
        for document_id, text in iter_document_texts(Path(str(config["inputs"]["contexts_database_path"])), required_docs):
            seen_docs.add(document_id)
            if not text:
                missing.append(document_id)
                continue
            blocks = make_blocks(
                tokenizer,
                document_id,
                text,
                int(raw["block_tokens"]),
                int(raw["block_overlap_tokens"]),
            )
            if not blocks:
                missing.append(document_id)
                continue
            document_counts[document_id] = len(blocks)
            for block in blocks:
                # Keep only deterministic, label-free evidence materialization fields.
                row = {
                    "block_id": block["block_id"],
                    "block_index": int(block["block_index"]),
                    "doc_id": document_id,
                    "end_token": int(block["end_token"]),
                    "start_token": int(block["start_token"]),
                    "text": block["text"],
                    "token_count": int(block["token_count"]),
                    "token_ids": [int(value) for value in block["token_ids"]],
                }
                encoded = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
                target.write(encoded)
                digest.update(encoded)
                block_count += 1

    not_seen = sorted(required_docs - seen_docs)
    missing = sorted(set(missing) | set(not_seen))
    if missing:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"raw evidence missing/empty for {len(missing)} candidate documents; no substitution is allowed: {missing[:10]}")
    if set(document_counts) != required_docs:
        temporary.unlink(missing_ok=True)
        raise ValueError("raw evidence document coverage mismatch")
    os.replace(temporary, block_path)
    block_sha = digest.hexdigest()

    provenance = {
        "kind": "legalir_v25_sv_raw_block_manifest",
        "config_sha256": config["resolved_config_sha256"],
        "model_revision": config["model"]["revision"],
        "tokenizer_sha256": config["tokenizer"]["tokenizer_json_sha256"],
        "source_contexts_database_sha256": config["inputs"]["hashes"]["contexts_database"],
        "source_candidate_pool": "V23-B0 new_pair union plus frozen V16b slots 4-5",
        "qrels_loaded_for_feature_content": False,
        "labels_written": False,
        "holdout_opened": False,
        "public_labels_used": False,
        "force_identity_qids_without_required_packs": sorted(force_identity),
        "document_count": len(required_docs),
        "block_count": block_count,
        "document_block_counts": document_counts,
        "block_manifest_sha256": block_sha,
    }
    write_json(args.output_dir / "v25_sv_raw_block_manifest.provenance.json", provenance)
    print(json.dumps({
        "manifest": str(block_path.resolve()),
        "provenance": str((args.output_dir / "v25_sv_raw_block_manifest.provenance.json").resolve()),
        "documents": len(required_docs),
        "blocks": block_count,
        "force_identity_qids": len(force_identity),
        "sha256": block_sha,
        "holdout_opened": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
