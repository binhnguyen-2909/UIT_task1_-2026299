"""Prepare the label-free, parity-locked input contract for V21-B MIL.

V21-B is conditional on a failed V21-A promotion.  This preflight deliberately
does not read references or fit a model: it proves that a representation-level
document-bag runner can consume the exact V18 query-conditioned evidence packs
without changing V16b's candidate identities or introducing structural-first
fallbacks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


KIND = "legalir_v21_b_mil_label_free_preflight"
PACK_CONFIG = "B2304-D60"
FORBIDDEN_TERMS = ("holdout", "public", "locked", "submission")


def _safe(path: Path, label: str) -> None:
    normalized = str(path.resolve()).replace("\\", "/").casefold()
    if any(term in normalized for term in FORBIDDEN_TERMS):
        raise ValueError(f"{label} must be clean-complement-only: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    _safe(path, label)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _required_bags(metadata: Mapping[str, Any]) -> tuple[dict[str, str], set[tuple[str, str]]]:
    if metadata.get("kind") != "legalir_v18b_evidence_pack_manifest" or metadata.get("split") != "clean_complement_only":
        raise ValueError("requires V18b clean-complement metadata")
    if metadata.get("holdout_opened") is not False or metadata.get("public_labels_used") is not False:
        raise ValueError("metadata does not attest unopened labels")
    raw_questions = metadata.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise ValueError("metadata questions are missing")
    queries: dict[str, str] = {}
    required: set[tuple[str, str]] = set()
    for question in raw_questions:
        if not isinstance(question, Mapping):
            raise ValueError("malformed metadata question")
        qid, query = str(question.get("qid", "")), str(question.get("query", ""))
        anchor = [str(value) for value in question.get("anchor_answer", [])]
        scored = [str(value) for value in question.get("e3b_scoring_boundary", [])]
        if not qid or not query.strip() or qid in queries:
            raise ValueError("empty or duplicate query identity")
        if len(anchor) != 5 or not scored or anchor[-1] not in scored:
            raise ValueError(f"{qid}: V16b slot-5/evidence boundary parity failure")
        queries[qid] = query
        required.update((qid, doc_id) for doc_id in scored)
    return queries, required


def build(metadata_path: Path, pack_manifest_path: Path, fold_manifest_path: Path) -> dict[str, Any]:
    for path, label in ((metadata_path, "metadata"), (pack_manifest_path, "pack manifest"), (fold_manifest_path, "fold manifest")):
        _safe(path, label)
    metadata = _read_json(metadata_path, "metadata")
    queries, required = _required_bags(metadata)
    folds = _read_json(fold_manifest_path, "fold manifest")
    if folds.get("kind") != "legalir_v21_group_crossfit_manifest" or folds.get("metadata_sha256") != _sha256(metadata_path):
        raise ValueError("fold manifest is not the audited V21 group manifest for this metadata")
    if set(str(value) for value in folds.get("qids", [])) != set(queries):
        raise ValueError("fold manifest qids differ from V18b pack metadata")
    found: dict[tuple[str, str], dict[str, Any]] = {}
    block_counts: Counter[int] = Counter()
    token_counts: list[int] = []
    source_signatures: Counter[str] = Counter()
    with pack_manifest_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping) or str(row.get("config")) != PACK_CONFIG:
                continue
            qid, doc_id = str(row.get("qid", "")), str(row.get("doc_id", ""))
            key = (qid, doc_id)
            if key not in required:
                continue
            if key in found:
                raise ValueError(f"duplicate {PACK_CONFIG} bag at {qid}/{doc_id}")
            if str(row.get("query", "")) != queries[qid]:
                raise ValueError(f"{qid}/{doc_id}: pack query differs from metadata query")
            pack = str(row.get("pack", ""))
            blocks = row.get("blocks")
            block_count = int(row.get("block_count", -1))
            token_count = int(row.get("pair_token_count", -1))
            if not pack.strip() or not isinstance(blocks, list) or len(blocks) != block_count or not 3 <= block_count <= 5:
                raise ValueError(f"{qid}/{doc_id}: malformed document bag")
            if not 1 <= token_count <= 2304:
                raise ValueError(f"{qid}/{doc_id}: token budget differs from V18 parity contract")
            block_ids = [str(block.get("block_id", "")) for block in blocks if isinstance(block, Mapping)]
            if len(block_ids) != block_count or len(set(block_ids)) != block_count or any(not value for value in block_ids):
                raise ValueError(f"{qid}/{doc_id}: invalid evidence block identity")
            found[key] = {
                "pack_sha256": str(row.get("pack_sha256", "")),
                "pair_token_count": token_count,
                "block_count": block_count,
                "preparation_signature": str(row.get("preparation_signature", "")),
            }
            block_counts[block_count] += 1
            token_counts.append(token_count)
            source_signatures[str(row.get("preparation_signature", ""))] += 1
    missing = required - set(found)
    if missing:
        raise ValueError(f"missing {len(missing)} required V21-B document bags; first={sorted(missing)[:3]}")
    return {
        "kind": KIND,
        "schema_version": 1,
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "labels_loaded": False,
        "training_started": False,
        "deployment_changed": False,
        "representation_contract": {
            "architecture": "hidden/token representations -> document-bag MIL/CELI/TIR; no scalar CATER feature selector",
            "pack_config": PACK_CONFIG,
            "blocks_per_bag": "3..5",
            "max_pair_tokens": 2304,
            "candidate_identity": "exact immutable V16b slot-5 incumbent plus frozen E3B boundary",
            "fallback": "force abstain on an invalid bag; never substitute a challenger document",
            "start_gate": "V21-A fails its complete promotion contract",
        },
        "counts": {
            "queries": len(queries), "document_bags": len(required),
            "block_counts": dict(sorted(block_counts.items())),
            "pair_token_min": min(token_counts), "pair_token_max": max(token_counts),
            "preparation_signatures": dict(sorted(source_signatures.items())),
        },
        "inputs": {
            "metadata": str(metadata_path.resolve()), "metadata_sha256": _sha256(metadata_path),
            "pack_manifest": str(pack_manifest_path.resolve()), "pack_manifest_sha256": _sha256(pack_manifest_path),
            "fold_manifest": str(fold_manifest_path.resolve()), "fold_manifest_sha256": _sha256(fold_manifest_path),
        },
        "sample_bag_fingerprints": [
            {"qid": qid, "doc_id": doc_id, **found[(qid, doc_id)]}
            for qid, doc_id in sorted(found)[:10]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--pack-manifest", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    _safe(args.output, "output")
    if args.output.exists() and not args.force:
        raise FileExistsError(args.output)
    payload = build(args.metadata, args.pack_manifest, args.fold_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "counts": payload["counts"], "labels_loaded": False}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
