"""Materialize the missing V23-A incumbent slot-4 E3B representation.

This is a label-free preparation step.  It reuses the archived V18b parser,
query-conditioned BQB proposer, frozen Stage-2 CE passage signal, and the
registered B2304-D60 pack builder.  It never substitutes a missing challenger
and never opens holdout/public labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = REPO_ROOT / "Task1 - LegalIR"
ARCHIVE_SCRIPTS = TASK_ROOT / "archive" / "2026-08-31_retired_experiments" / "Task1-scripts"
PACK_CONFIG = "B2304-D60"
PACK_BUDGET = 2304
PACK_DIVERSITY = 0.60
PASSAGES_PER_DOC = 3
PROPOSAL_LIMIT = 24
BQB_FUNNEL_PER_DOC = 16
MAX_CE_LENGTH = 2304
FORBIDDEN_TERMS = ("holdout", "public", "locked", "submission")

if str(ARCHIVE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(ARCHIVE_SCRIPTS))

from v18b_prepare_packs import (  # noqa: E402
    ZipDocumentStore,
    _masked_bqb_scores,
    _pair_token_length,
    _render_pack,
    _sha256_file,
    _sha256_path,
    _stable_hash,
    _tokenizer_hash,
    bm25_score,
    build_valid_pack,
    citation_keys,
    propose_blocks,
)
from v18_evidence_audit import (  # noqa: E402
    _bm25_statistics,
    load_passage_store,
    open_readonly,
    score_pairs,
)
from batch_stage2_rerank import hydrate  # noqa: E402
from reranker import Reranker  # noqa: E402


def _safe(path: Path, label: str) -> None:
    folded = str(path.resolve()).replace("\\", "/").casefold()
    if any(term in folded for term in FORBIDDEN_TERMS):
        raise ValueError(f"{label} must remain clean-complement-only: {path}")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _safe(path, label)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _metadata_questions(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    if metadata.get("kind") != "legalir_v18b_evidence_pack_manifest" or metadata.get("split") != "clean_complement_only":
        raise ValueError("V23-A slot-4 materialization requires V18b clean-complement metadata")
    if metadata.get("holdout_opened") is not False or metadata.get("public_labels_used") is not False:
        raise ValueError("metadata does not attest unopened labels")
    raw_questions = metadata.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise ValueError("metadata.questions is missing")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_questions:
        if not isinstance(raw, Mapping):
            raise ValueError("malformed metadata question")
        qid = str(raw.get("qid", ""))
        query = str(raw.get("query", ""))
        anchor = [str(value) for value in raw.get("anchor_answer", [])]
        scoring = [str(value) for value in raw.get("scoring_boundary", [])]
        e3b = [str(value) for value in raw.get("e3b_scoring_boundary", [])]
        unavailable = [str(value) for value in raw.get("e3b_source_unavailable", [])]
        if not qid or qid in seen or not query.strip() or len(anchor) != 5 or len(set(anchor)) != 5:
            raise ValueError(f"invalid V18b question: {qid!r}")
        if not scoring or scoring[0] != anchor[-1] or not e3b or e3b[0] != anchor[-1]:
            raise ValueError(f"{qid}: scoring boundary parity failure")
        if not set(e3b).issubset(set(scoring)) or set(unavailable) != set(scoring) - set(e3b):
            raise ValueError(f"{qid}: E3B unavailable-document declaration drift")
        slot4 = anchor[3]
        if slot4 in set(scoring) or slot4 in set(e3b):
            raise ValueError(f"{qid}: slot-4 identity overlaps the E3B challenger boundary")
        seen.add(qid)
        result.append({
            "qid": qid,
            "query": query,
            "anchor_answer": anchor,
            "scoring_boundary": scoring,
            "e3b_scoring_boundary": e3b,
            "e3b_source_unavailable": unavailable,
            "slot4": slot4,
        })
    return result


def _validate_existing_e3b(
    manifest_path: Path,
    questions: list[dict[str, Any]],
) -> tuple[set[tuple[str, str]], dict[str, Any]]:
    required = {
        (question["qid"], doc_id)
        for question in questions
        for doc_id in question["e3b_scoring_boundary"]
    }
    found: set[tuple[str, str]] = set()
    config_counts: defaultdict[str, int] = defaultdict(int)
    query_text = {question["qid"]: question["query"] for question in questions}
    with manifest_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping) or str(row.get("config")) != PACK_CONFIG:
                continue
            key = (str(row.get("qid", "")), str(row.get("doc_id", "")))
            if key not in required:
                continue
            if key in found:
                raise ValueError(f"duplicate E3B row at line {line_number}: {key}")
            if str(row.get("query", "")) != query_text[key[0]]:
                raise ValueError(f"E3B query drift at {key}")
            block_count = int(row.get("block_count", -1))
            pair_tokens = int(row.get("pair_token_count", -1))
            if not 3 <= block_count <= 5 or not 1 <= pair_tokens <= PACK_BUDGET:
                raise ValueError(f"invalid E3B pack contract at {key}")
            found.add(key)
            config_counts[PACK_CONFIG] += 1
    missing = required - found
    if missing:
        raise ValueError(f"existing E3B manifest misses {len(missing)} required bags; first={sorted(missing)[:3]}")
    return found, {"required_bags": len(required), "found_bags": len(found), "config_counts": dict(config_counts)}


def _load_slot4_manifest(
    manifest_path: Path,
    questions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
    expected = {(question["qid"], question["slot4"]) for question in questions}
    found: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    query_text = {question["qid"]: question["query"] for question in questions}
    with manifest_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"slot-4 manifest line {line_number} is not an object")
            key = (str(row.get("qid", "")), str(row.get("doc_id", "")))
            if key not in expected:
                raise ValueError(f"unexpected slot-4 manifest key at line {line_number}: {key}")
            if key in found:
                raise ValueError(f"duplicate slot-4 manifest key: {key}")
            if str(row.get("config")) != PACK_CONFIG or str(row.get("query", "")) != query_text[key[0]]:
                raise ValueError(f"slot-4 manifest provenance drift at {key}")
            block_count = int(row.get("block_count", -1))
            pair_tokens = int(row.get("pair_token_count", -1))
            blocks = row.get("blocks")
            if not 3 <= block_count <= 5 or not 1 <= pair_tokens <= PACK_BUDGET or not isinstance(blocks, list) or len(blocks) != block_count:
                raise ValueError(f"invalid slot-4 pack at {key}")
            found.add(key)
            rows.append(dict(row))
    if not found.issubset(expected):
        raise ValueError("slot-4 manifest contains an identity outside metadata")
    return rows, found


def _score_slot4_passages(
    questions: list[dict[str, Any]],
    prior_index: Path,
    reranker_model: Path,
    device: str,
    batch_size: int,
    max_length: int,
) -> dict[tuple[str, str], list[float]]:
    doc_ids = sorted({question["slot4"] for question in questions})
    connection = open_readonly(prior_index)
    try:
        passage_store = load_passage_store(connection, doc_ids)
    finally:
        connection.close()
    pairs: list[tuple[str, str]] = []
    owners: list[tuple[str, str]] = []
    counts: dict[tuple[str, str], int] = {}
    for question in questions:
        owner = (question["qid"], question["slot4"])
        hydrated = hydrate(question["query"], [question["slot4"]], PASSAGES_PER_DOC, passage_store)
        passages = [str(value) for value in hydrated[0].get("passages", [])] if hydrated else []
        if not passages or any(not value.strip() for value in passages):
            raise ValueError(f"{owner}: frozen passage hydration returned an empty passage")
        counts[owner] = len(passages)
        pairs.extend((question["query"], passage) for passage in passages)
        owners.extend([owner] * len(passages))
    scorer = Reranker(
        str(reranker_model.resolve()), device=device, batch_size=batch_size,
        max_length=max_length, passage_aggregation="max", use_fp16=device == "cuda",
    )
    values = score_pairs(scorer, pairs)
    result: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for owner, value in zip(owners, values, strict=True):
        result[owner].append(float(value))
    if any(len(result.get(owner, [])) != count for owner, count in counts.items()):
        raise ValueError("slot-4 frozen Stage-2 passage scoring is incomplete")
    del scorer
    try:
        import torch
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return dict(result)


def _build_slot4_rows(
    questions: list[dict[str, Any]],
    contexts_zip: Path,
    prior_index: Path,
    reranker_model: Path,
    bqb_model: Path,
    device: str,
    bqb_batch_size: int,
    batch_questions: int,
    reranker_batch_size: int,
    reranker_max_length: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    passage_scores = _score_slot4_passages(
        questions, prior_index, reranker_model, device, reranker_batch_size, reranker_max_length,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(reranker_model.resolve()), local_files_only=True)
    bqb = SentenceTransformer(str(bqb_model.resolve()), device=device, local_files_only=True)
    bqb.max_seq_length = min(int(getattr(bqb, "max_seq_length", 256)), 256)
    connection = open_readonly(prior_index)
    try:
        passage_store = load_passage_store(connection, sorted({question["slot4"] for question in questions}))
    finally:
        connection.close()
    packable_questions: list[dict[str, Any]] = []
    slot4_unavailable_qids: list[str] = []
    with ZipDocumentStore(contexts_zip) as audit_store:
        for question in questions:
            document = audit_store.packable(question["slot4"])
            if document is None or len(document.blocks) < 3:
                slot4_unavailable_qids.append(question["qid"])
            else:
                packable_questions.append(question)
    rows: list[dict[str, Any]] = []
    with ZipDocumentStore(contexts_zip) as store:
        for start in range(0, len(packable_questions), batch_questions):
            batch = packable_questions[start:start + batch_questions]
            parsed: dict[str, Any] = {}
            eligible: dict[str, set[str]] = {}
            missing_blocks: dict[str, Any] = {}
            for question in batch:
                qid = question["qid"]
                doc_id = question["slot4"]
                document = store.packable(doc_id)
                if document is None or len(document.blocks) < 3:
                    raise RuntimeError(f"slot-4 packability changed during materialization: {qid}/{doc_id}")
                parsed[qid] = document
                stats = _bm25_statistics(document.blocks)
                lexical = {block.block_id: bm25_score(question["query"], block, stats) for block in document.blocks}
                allowed = set(sorted(lexical, key=lambda block_id: (-lexical[block_id], block_id))[:BQB_FUNNEL_PER_DOC])
                keys = citation_keys(question["query"])
                if keys:
                    allowed.update(
                        block.block_id for block in document.blocks
                        if any(key in block.search_text for key in keys)
                    )
                eligible[qid] = allowed
                for block in document.blocks:
                    if block.block_id in allowed:
                        missing_blocks[block.block_id] = block
            vector_cache: dict[str, np.ndarray] = {}
            if missing_blocks:
                ordered = [missing_blocks[key] for key in sorted(missing_blocks)]
                vectors = np.asarray(bqb.encode(
                    [block.text for block in ordered], batch_size=bqb_batch_size,
                    show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True,
                ), dtype=np.float32)
                vector_cache.update({block.block_id: vectors[index] for index, block in enumerate(ordered)})
            qvecs = np.asarray(bqb.encode(
                [question["query"] for question in batch], batch_size=bqb_batch_size,
                show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True,
            ), dtype=np.float32)
            for position, question in enumerate(batch):
                qid = question["qid"]
                doc_id = question["slot4"]
                document = parsed[qid]
                hydrated = hydrate(question["query"], [doc_id], PASSAGES_PER_DOC, passage_store)
                passages = [str(value) for value in hydrated[0].get("passages", [])]
                bqb_scores = _masked_bqb_scores(qvecs[position], eligible[qid], vector_cache)
                selected, _ = propose_blocks(
                    question["query"], document.blocks, passages, bqb_scores,
                    passage_scores[(qid, doc_id)], PROPOSAL_LIMIT, diversity=PACK_DIVERSITY,
                )
                pack, trace, pair_tokens = build_valid_pack(
                    tokenizer, question["query"], document.title, selected, PACK_BUDGET,
                )
                row = {
                    "schema_version": 2,
                    "qid": qid,
                    "query": question["query"],
                    "doc_id": doc_id,
                    "config": PACK_CONFIG,
                    "budget": PACK_BUDGET,
                    "mmr_diversity": PACK_DIVERSITY,
                    "pack": pack,
                    "pack_sha256": hashlib.sha256(pack.encode("utf-8")).hexdigest(),
                    "pair_token_count": pair_tokens,
                    "block_count": len(trace),
                    "blocks": trace,
                    "ancestor_preserved": all(bool(item["ancestor_preserved"]) for item in trace),
                }
                rows.append(row)
            print(f"V23-A slot-4 materialization: {min(start + len(batch), len(questions))}/{len(questions)}", flush=True)
    del bqb
    return rows, {
        "questions": len(questions),
        "packable_questions": len(packable_questions),
        "slot4_unavailable_qids": sorted(slot4_unavailable_qids),
        "rows": len(rows),
        "tokenizer_sha256": _tokenizer_hash(tokenizer),
        "slot4_pack_config": PACK_CONFIG,
        "passages_per_doc": PASSAGES_PER_DOC,
        "proposal_limit": PROPOSAL_LIMIT,
        "bqb_funnel_per_doc": BQB_FUNNEL_PER_DOC,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path, label in (
        (args.metadata, "metadata"), (args.e3b_manifest, "E3B manifest"),
        (args.contexts_zip, "contexts"), (args.prior_index, "prior index"),
        (args.reranker_model, "reranker model"), (args.bqb_model, "BQB model"),
        (args.output_dir, "output"),
    ):
        # The released context corpus is stored under public_test, but it is
        # not a label artifact.  Metadata, indexes, and outputs remain guarded
        # by the clean-complement path contract.
        if label != "contexts":
            _safe(path, label)
    metadata = _read_json(args.metadata, "metadata")
    questions = _metadata_questions(metadata)
    _e3b_keys, e3b_stats = _validate_existing_e3b(args.e3b_manifest, questions)
    manifest_path = args.output_dir / "v23_a_slot4_pack_manifest.jsonl"
    report_path = args.output_dir / "v23_a_slot4_materialization_report.json"
    if manifest_path.exists() and report_path.exists() and not args.force:
        raise FileExistsError("V23-A slot-4 output exists; use --force only for an explicit rerun")
    if manifest_path.exists() and not report_path.exists() and not args.force:
        rows, actual = _load_slot4_manifest(manifest_path, questions)
        unavailable_qids = {question["qid"] for question in questions} - {qid for qid, _ in actual}
        build_stats = {
            "questions": len(questions),
            "packable_questions": len(rows),
            "slot4_unavailable_qids": sorted(unavailable_qids),
            "rows": len(rows),
            "slot4_pack_config": PACK_CONFIG,
            "passages_per_doc": PASSAGES_PER_DOC,
            "proposal_limit": PROPOSAL_LIMIT,
            "bqb_funnel_per_doc": BQB_FUNNEL_PER_DOC,
            "recovered_existing_manifest": True,
        }
    else:
        rows, build_stats = _build_slot4_rows(
            questions, args.contexts_zip, args.prior_index, args.reranker_model,
            args.bqb_model, args.device, args.bqb_batch_size, args.query_batch_size,
            args.reranker_batch_size, args.reranker_max_length,
        )
        unavailable_qids = set(build_stats["slot4_unavailable_qids"])
        expected = {
            (question["qid"], question["slot4"])
            for question in questions
            if question["qid"] not in unavailable_qids
        }
        actual = {(str(row["qid"]), str(row["doc_id"])) for row in rows}
        if actual != expected or len(rows) != len(expected):
            raise ValueError("slot-4 manifest key set is not a one-to-one metadata match")
    signature_payload = {
        "kind": "legalir_v23_a_slot4_e3b_materialization",
        "metadata_sha256": _sha256_file(args.metadata),
        "e3b_manifest_sha256": _sha256_file(args.e3b_manifest),
        "contexts_sha256": _sha256_file(args.contexts_zip),
        "prior_index_sha256": _sha256_file(args.prior_index),
        "reranker_sha256": _sha256_path(args.reranker_model),
        "bqb_sha256": _sha256_path(args.bqb_model),
        "pack_config": PACK_CONFIG,
        "pack_budget": PACK_BUDGET,
        "pack_diversity": PACK_DIVERSITY,
        "passages_per_doc": PASSAGES_PER_DOC,
        "proposal_limit": PROPOSAL_LIMIT,
        "bqb_funnel_per_doc": BQB_FUNNEL_PER_DOC,
        "reranker_max_length": args.reranker_max_length,
    }
    signature = _stable_hash(signature_payload)
    existing_signatures = {str(row.get("preparation_signature", "")) for row in rows}
    if manifest_path.exists() and not args.force and existing_signatures != {signature}:
        raise ValueError("recovered slot-4 manifest signature differs from the current recipe")
    for row in rows:
        row["preparation_signature"] = signature
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not (manifest_path.exists() and not args.force):
        temporary = manifest_path.with_name(manifest_path.name + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as target:
            for row in rows:
                target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, manifest_path)
    report = {
        "kind": "legalir_v23_a_slot4_e3b_materialization",
        "schema_version": 1,
        "stage": "label_free_slot4_pack_ready",
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "labels_loaded": False,
        "training_started": False,
        "candidate_identity": "V16b incumbent slot-4 only; E3B challenger identities unchanged",
        "fallback": "force abstain on missing/invalid E3B evidence or missing incumbent slot-4 pack; no challenger substitution",
        "e3b_existing": e3b_stats,
        "slot4": build_stats,
        "slot4_force_abstain_qids": sorted(unavailable_qids),
        "source_unavailable_qids": sorted(question["qid"] for question in questions if question["e3b_source_unavailable"]),
        "preparation_signature": signature,
        "signature_payload": signature_payload,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256_file(manifest_path),
        "metadata": str(args.metadata.resolve()),
        "metadata_sha256": _sha256_file(args.metadata),
    }
    _atomic_json(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--e3b-manifest", type=Path, required=True)
    parser.add_argument("--contexts-zip", type=Path, required=True)
    parser.add_argument("--prior-index", type=Path, required=True)
    parser.add_argument("--reranker-model", type=Path, required=True)
    parser.add_argument("--bqb-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--query-batch-size", type=int, default=2)
    parser.add_argument("--bqb-batch-size", type=int, default=64)
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--reranker-max-length", type=int, default=768)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if min(args.query_batch_size, args.bqb_batch_size, args.reranker_batch_size, args.reranker_max_length) <= 0:
        raise ValueError("batch sizes and max length must be positive")
    report = run(args)
    print(json.dumps({
        "output": report["manifest"],
        "rows": report["slot4"]["rows"],
        "holdout_opened": False,
        "labels_loaded": False,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
