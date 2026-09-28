"""Build one canonical, label-free E3B evidence pack for every c35 pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_SCRIPTS = SCRIPT_DIR.parents[1] / "scripts"
ARCHIVE_SCRIPTS = SCRIPT_DIR.parents[1] / "archive" / "2026-08-31_retired_experiments" / "Task1-scripts"
for path in (REPO_SCRIPTS, ARCHIVE_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from batch_stage2_rerank import hydrate, load_passage_store, open_readonly  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from v18_evidence_audit import EvidenceBlock, _bm25_statistics, bm25_score, citation_keys, lexical_tokens, normalize_text, propose_blocks  # noqa: E402
from v18b_prepare_packs import (  # noqa: E402
    ZipDocumentStore,
    _masked_bqb_scores,
    build_valid_pack,
)

from v27_common import (  # noqa: E402
    CANONICAL_E3B_CONFIG,
    MAX_LENGTH,
    load_metadata,
    load_sidecar,
    sha256_file,
    write_json,
)


PACK_BODY_WORD_CAP = 1000


def bounded_pack_blocks(blocks: list[object]) -> tuple[list[object], bool]:
    """Bound pathological raw blocks before repeated budget-tokenization.

    Selection still sees the complete parsed block.  Only the pack-builder's
    candidate text is bounded; its 2,304-token budget performs the final fit.
    """
    bounded = []
    truncated = False
    for block in blocks:
        words = str(block.body or "").split()
        if len(words) > PACK_BODY_WORD_CAP:
            truncated = True
            body = " ".join(words[:PACK_BODY_WORD_CAP])
            bounded.append(replace(block, body=body, full_text=body, normalized_text=body))
        else:
            bounded.append(block)
    return bounded, truncated


def make_passage_blocks(doc_id: str, title: str, passages: list[str]) -> list[EvidenceBlock]:
    """Represent hydrated passages as lightweight E3B evidence blocks."""
    blocks = []
    for ordinal, passage in enumerate(passages[:5]):
        text = " ".join(str(passage).split())
        if not text:
            continue
        tokens = tuple(lexical_tokens(text))
        blocks.append(EvidenceBlock(
            block_id=f"{doc_id}:passage:{ordinal}",
            doc_id=str(doc_id),
            unit_type="PASSAGE",
            title=str(title),
            ancestors=(),
            article_number="",
            clause_number="",
            point_label="",
            heading="",
            body=text,
            ordinal=ordinal,
            full_text=text,
            normalized_text=normalize_text(text),
            token_values=tokens,
            token_set_values=frozenset(tokens),
        ))
    return blocks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--contexts-zip", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--stage2-sidecar", type=Path, required=True)
    parser.add_argument("--pack-tokenizer", type=Path, required=True)
    parser.add_argument("--bqb-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bqb-batch-size", type=int, default=128)
    parser.add_argument("--bqb-funnel-per-doc", type=int, default=16)
    parser.add_argument("--proposal-limit", type=int, default=24)
    parser.add_argument("--cache-size", type=int, default=512)
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pack-mode", choices=("structural", "passages"), default="structural")
    args = parser.parse_args()
    questions = load_metadata(args.metadata)
    stage2 = load_sidecar(args.stage2_sidecar, questions)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "v27_e3b_manifest.jsonl"
    provenance_path = out / "v27_e3b_manifest.provenance.json"
    completed_qids: set[str] = set()
    existing_rows = 0
    docs: set[str] = set()
    existing_fallback_rows = 0
    existing_empty_rows = 0
    if manifest_path.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite existing E3B manifest: {manifest_path}")
    if manifest_path.exists() and args.resume:
        retained_rows: list[str] = []
        discarded_rows: list[str] = []
        with manifest_path.open(encoding="utf-8") as source:
            counts: dict[str, int] = {}
            raw_rows: list[tuple[str, str]] = []
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if str(row.get("config")) != CANONICAL_E3B_CONFIG:
                    raise ValueError(f"invalid existing E3B row at line {line_number}")
                qid = str(row["qid"])
                docs.add(str(row["doc_id"]))
                if str(row.get("evidence_mode")) == "structural_fallback":
                    existing_fallback_rows += 1
                if bool(row.get("empty")):
                    existing_empty_rows += 1
                counts[qid] = counts.get(qid, 0) + 1
                raw_rows.append((qid, line))
            completed_qids = {qid for qid, count in counts.items() if count == 35}
            incomplete = {qid: count for qid, count in counts.items() if count != 35}
            if incomplete:
                # The previous process may have been interrupted mid-query.
                # Keep those rows in a recoverable sidecar and compact the
                # canonical manifest before appending the remaining query.
                backup = manifest_path.with_name(manifest_path.name + ".partial.jsonl")
                with backup.open("w", encoding="utf-8") as target:
                    for qid, line in raw_rows:
                        (retained_rows if qid in completed_qids else discarded_rows).append(line)
                    target.writelines(discarded_rows)
                temporary = manifest_path.with_name(manifest_path.name + ".resume.tmp")
                temporary.write_text("".join(retained_rows), encoding="utf-8")
                os.replace(temporary, manifest_path)
            else:
                retained_rows = [line for _, line in raw_rows]
            existing_rows = len(retained_rows)
            docs = {str(json.loads(line)["doc_id"]) for line in retained_rows}

    tokenizer = AutoTokenizer.from_pretrained(str(args.pack_tokenizer.resolve()), local_files_only=True)
    bqb = SentenceTransformer(str(args.bqb_model.resolve()), device=args.device, local_files_only=True)
    bqb.max_seq_length = min(int(getattr(bqb, "max_seq_length", 256)), 256)
    connection = open_readonly(args.index)
    vector_cache: dict[str, np.ndarray] = {}
    rows = existing_rows
    q_count = len(completed_qids)
    fallback_rows = existing_fallback_rows
    empty_rows = existing_empty_rows
    try:
        mode = "a" if manifest_path.exists() and args.resume else "w"
        with ZipDocumentStore(args.contexts_zip, cache_size=args.cache_size) as store, manifest_path.open(mode, encoding="utf-8") as target:
            for qid, question in questions.items():
                if qid in completed_qids:
                    continue
                if args.max_questions and q_count >= args.max_questions:
                    break
                q_count += 1
                query = str(question["query"])
                c35 = list(question["c35"])
                passage_store = load_passage_store(connection, c35)
                parsed: dict[str, object] = {}
                eligible: dict[str, set[str]] = {}
                doc_modes: dict[str, str] = {}
                hydrated = {str(item["doc_id"]): item for item in hydrate(query, c35, 3, passage_store)}
                if args.pack_mode == "passages":
                    for doc_id in c35:
                        passages = [str(value) for value in hydrated.get(doc_id, {}).get("passages", []) if str(value).strip()]
                        blocks = make_passage_blocks(doc_id, "", passages)
                        if len(blocks) >= 3:
                            parsed[doc_id] = blocks
                            eligible[doc_id] = {block.block_id for block in blocks}
                            doc_modes[doc_id] = "passages"
                        else:
                            fallback = store.packable(doc_id)
                            if fallback is None or len(fallback.blocks) < 3:
                                parsed[doc_id] = None
                                eligible[doc_id] = set()
                                hydrated[doc_id] = {"passages": []}
                                doc_modes[doc_id] = "empty"
                                empty_rows += 1
                                continue
                            parsed[doc_id] = fallback
                            stats = _bm25_statistics(fallback.blocks)
                            lexical = {block.block_id: float(bm25_score(query, block, stats)) for block in fallback.blocks}
                            eligible[doc_id] = set(sorted(lexical, key=lambda block_id: (-lexical[block_id], block_id))[:args.bqb_funnel_per_doc])
                            hydrated[doc_id] = {"passages": [str(block.text) for block in fallback.blocks[:3]]}
                            doc_modes[doc_id] = "structural_fallback"
                            fallback_rows += 1
                else:
                    for doc_id in c35:
                        document = store.packable(doc_id)
                        if document is None or not document.blocks:
                            raise ValueError(f"{qid}/{doc_id}: missing or empty parsed document")
                        parsed[doc_id] = document
                        stats = _bm25_statistics(document.blocks)
                        lexical = {block.block_id: float(bm25_score(query, block, stats)) for block in document.blocks}
                        allowed = set(sorted(lexical, key=lambda block_id: (-lexical[block_id], block_id))[:args.bqb_funnel_per_doc])
                        keys = citation_keys(query)
                        if keys:
                            allowed.update(block.block_id for block in document.blocks if any(key in block.search_text for key in keys))
                        eligible[doc_id] = allowed
                missing = {}
                for doc_id in c35:
                    document = parsed[doc_id]
                    if doc_modes[doc_id] == "empty":
                        continue
                    document_blocks = document if doc_modes[doc_id] == "passages" else document.blocks
                    for block in document_blocks:
                        if block.block_id in eligible[doc_id] and block.block_id not in vector_cache:
                            missing[block.block_id] = block
                if missing:
                    ordered = [missing[key] for key in sorted(missing)]
                    vectors = np.asarray(bqb.encode([block.text for block in ordered], batch_size=args.bqb_batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)
                    vector_cache.update({block.block_id: vectors[index] for index, block in enumerate(ordered)})
                qvec = np.asarray(bqb.encode([query], batch_size=1, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)[0], dtype=np.float32)
                for doc_id in c35:
                    document = parsed[doc_id]
                    if doc_modes[doc_id] == "empty":
                        row = {
                            "schema_version": 1,
                            "qid": qid,
                            "doc_id": doc_id,
                            "config": CANONICAL_E3B_CONFIG,
                            "budget": 2304,
                            "mmr_diversity": 0.60,
                            "pack": "",
                            "pack_sha256": hashlib.sha256(b"").hexdigest(),
                            "token_count": 0,
                            "block_count": 0,
                            "blocks": [],
                            "proposal_trace": {},
                            "pack_body_word_cap": PACK_BODY_WORD_CAP,
                            "pack_body_truncated": False,
                            "evidence_mode": "empty",
                            "empty": True,
                            "empty_reason": "source_unavailable_or_fewer_than_three_packable_blocks",
                            "projection_fix": {"previous_preregistered": [512, 1024], "canonical_runtime": [768, 1536], "applied": True},
                            "holdout_opened": False,
                            "public_labels_used": False,
                        }
                        target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                        rows += 1
                        docs.add(doc_id)
                        continue
                    document_blocks = document if doc_modes[doc_id] == "passages" else document.blocks
                    passages = [str(value) for value in hydrated.get(doc_id, {}).get("passages", [])]
                    if not passages or not any(value.strip() for value in passages):
                        raise ValueError(f"{qid}/{doc_id}: no non-empty hydrated passage")
                    passage_scores = [stage2[(qid, doc_id)]] * len(passages)
                    bqb_scores = _masked_bqb_scores(qvec, eligible[doc_id], vector_cache)
                    selected, trace = propose_blocks(query, document_blocks, passages, bqb_scores, passage_scores, args.proposal_limit, diversity=0.60)
                    pack_blocks, pack_body_truncated = bounded_pack_blocks(selected)
                    title = "" if doc_modes[doc_id] == "passages" else document.title
                    pack, block_trace, pair_tokens = build_valid_pack(tokenizer, query, title, pack_blocks, 2304)
                    if not pack.strip() or len(block_trace) < 3 or pair_tokens > 2304:
                        raise ValueError(f"{qid}/{doc_id}: invalid canonical E3B pack")
                    row = {
                        "schema_version": 1,
                        "qid": qid,
                        "doc_id": doc_id,
                        "config": CANONICAL_E3B_CONFIG,
                        "budget": 2304,
                        "mmr_diversity": 0.60,
                        "pack": pack,
                        "pack_sha256": hashlib.sha256(pack.encode("utf-8")).hexdigest(),
                        "token_count": int(pair_tokens),
                        "block_count": len(block_trace),
                        "blocks": block_trace,
                        "proposal_trace": trace,
                        "pack_body_word_cap": PACK_BODY_WORD_CAP,
                        "pack_body_truncated": pack_body_truncated,
                        "evidence_mode": doc_modes[doc_id],
                        "empty": False,
                        "projection_fix": {"previous_preregistered": [512, 1024], "canonical_runtime": [768, 1536], "applied": True},
                        "holdout_opened": False,
                        "public_labels_used": False,
                    }
                    target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    rows += 1
                    docs.add(doc_id)
                if q_count % 10 == 0:
                    print(json.dumps({"questions": q_count, "rows": rows, "unique_docs": len(docs), "cached_block_vectors": len(vector_cache)}), flush=True)
    finally:
        connection.close()
        del bqb

    expected = q_count * 35
    if args.max_questions == 0 and rows != len(questions) * 35:
        raise RuntimeError(f"E3B manifest coverage mismatch: {rows} != {len(questions) * 35}")
    manifest_hash = sha256_file(manifest_path)
    write_json(provenance_path, {
        "kind": "legalir_v27_canonical_e3b_provenance",
        "schema_version": 1,
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "config": CANONICAL_E3B_CONFIG,
        "evidence_mode": args.pack_mode,
        "structural_fallback_rows": fallback_rows,
        "empty_rows": empty_rows,
        "budget": 2304,
        "pack_body_word_cap": PACK_BODY_WORD_CAP,
        "mmr_diversity": 0.60,
        "queries": q_count,
        "rows": rows,
        "unique_documents": len(docs),
        "manifest_sha256": manifest_hash,
        "inputs": {"metadata": sha256_file(args.metadata), "contexts_zip": sha256_file(args.contexts_zip), "index": sha256_file(args.index), "stage2_sidecar": sha256_file(args.stage2_sidecar), "pack_tokenizer": str(args.pack_tokenizer.resolve()), "bqb_model": str(args.bqb_model.resolve())},
        "projection_fix": {"previous_preregistered": [512, 1024], "canonical_runtime": [768, 1536], "selection_regenerated": True},
    })
    print(json.dumps({"manifest": str(manifest_path.resolve()), "provenance": str(provenance_path.resolve()), "queries": q_count, "rows": rows, "expected_rows": expected, "unique_documents": len(docs), "sha256": manifest_hash}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
