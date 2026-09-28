"""E1: label-blind evidence proposal followed by frozen cross-encoder scoring.

This is deliberately a *complement-only diagnostic*.  It keeps the V16b
candidate universe fixed at c35, proposes evidence with three unsupervised
signals (lexical BM25-style overlap, frozen BQB dense similarity, and legal
structure), then lets the existing Vietnamese reranker read at most four
passages per document.  Labels are not loaded until the evidence manifest is
sealed and frozen scores have been written.

It is not a submission generator and it never opens holdout or public data.
E2 may use the frozen E1 manifest only if this diagnostic demonstrates a real
complement gain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v27_a_common import (  # noqa: E402
    CANDIDATE_LIMIT,
    EvidenceBlock,
    V27ContractError,
    V27Document,
    legal_question_type,
    load_manifest,
    load_relevance_model,
    load_split,
    load_tokenizer,
    make_blocks,
    normalized_text,
    recall_value,
    sha256_file,
    source_family,
    temperature_logsumexp,
    token_list,
    transition_metrics,
    url_slug,
    write_json,
)

from legal_structure import iter_legal_chunks  # noqa: E402


E1_KIND = "legalir_e1_label_blind_evidence_frozen_ce"
E1_SCHEMA_VERSION = 1
E1_PROPOSALS_PER_SOURCE = 4
E1_PROPOSAL_LIMIT = 12
E1_MIN_READ_PASSAGES = 3
E1_READ_LIMIT = 4
E1_UNCERTAINTY_MARGIN = 0.35
E1_BLOCK_TOKENS = 384
E1_BLOCK_OVERLAP = 64
E1_MAX_SEQUENCE_LENGTH = 768
E1_BQB_QUERY_PREFIX = "query: "
E1_BQB_PASSAGE_PREFIX = "passage: "
_NUMBER_RE = re.compile(r"\d+(?:[./-]\d+)*")
_LEGAL_MARKERS = ("dieu", "khoan", "diem", "chuong", "muc", "phan", "phu luc")


@dataclass(frozen=True)
class E1Document:
    """Raw-corpus document kept separate from V11's normalized index view."""

    doc_id: str
    title: str
    body: str
    link: str

    @property
    def empty_body(self) -> bool:
        return not self.body.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("materialize", "score", "all"), default="all")
    parser.add_argument("--contexts-zip", type=Path, required=True)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--bqb-model", type=Path, required=True)
    parser.add_argument("--reranker-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--proposal-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--score-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--bqb-batch-size", type=int, default=128)
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--query-chunk-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=E1_MAX_SEQUENCE_LENGTH)
    parser.add_argument("--aggregation-temperature", type=float, default=0.5)
    parser.add_argument("--force-manifest", action="store_true")
    parser.add_argument("--force-score", action="store_true")
    return parser.parse_args()


def load_raw_documents(contexts_zip: Path, document_ids: Iterable[str]) -> dict[str, E1Document]:
    """Load only required corpus rows without normalizing away legal structure."""

    needed = {str(value) for value in document_ids if str(value)}
    documents: dict[str, E1Document] = {}
    with zipfile.ZipFile(contexts_zip) as archive:
        for info in archive.infolist():
            if not info.filename.endswith(".json"):
                continue
            raw = json.loads(archive.read(info))
            doc_id = str(raw.get("id", ""))
            if doc_id not in needed:
                continue
            title = raw.get("name", "")
            body = raw.get("passage", "")
            documents[doc_id] = E1Document(
                doc_id=doc_id,
                title=title if isinstance(title, str) else "",
                body=body if isinstance(body, str) else "",
                link=str(raw.get("link", "") or ""),
            )
    missing = sorted(needed - set(documents))
    if missing:
        raise V27ContractError(
            f"Raw corpus lacks {len(missing)} c35 documents: {missing[:10]}"
        )
    return documents


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    return [int(value) for value in encoded["input_ids"]]


def make_structural_token_blocks(
    tokenizer: Any,
    document: E1Document,
    *,
    block_tokens: int = E1_BLOCK_TOKENS,
    overlap_tokens: int = E1_BLOCK_OVERLAP,
) -> tuple[list[EvidenceBlock], dict[str, dict[str, str]]]:
    """Token-budget legal chunks with repeated title/unit parent context.

    ``iter_legal_chunks`` first preserves Điều/Khoản/Điểm and hierarchy using
    the original corpus text.  Each unit is then split by the *reranker*
    tokenizer rather than character count.  The compact parent prefix is
    repeated in every token window, so a continuation never loses document or
    legal-unit identity merely because it crossed a model boundary.
    """

    if document.empty_body:
        return [], {}
    if block_tokens <= overlap_tokens or overlap_tokens < 0:
        raise ValueError("require block_tokens > overlap_tokens >= 0")
    blocks: list[EvidenceBlock] = []
    metadata: dict[str, dict[str, str]] = {}
    # Character bounds only define an upper legal-unit packing limit.  The
    # actual model budget below is exclusively token based.
    for unit in iter_legal_chunks(document.body, chunk_size=3600, overlap=300):
        parent = "\n".join(part for part in (
            f"[TITLE] {document.title}" if document.title else "",
            f"[HIERARCHY] {unit.hierarchy}" if unit.hierarchy else "",
            f"[UNIT_TYPE] {unit.unit_type}",
            f"[ARTICLE] {unit.article_number}" if unit.article_number else "",
        ) if part)
        # An exceptionally long title must not consume the entire evidence
        # budget.  It remains in the manifest's title field unchanged.
        parent_ids = _token_ids(tokenizer, parent)[:96]
        body_ids = _token_ids(tokenizer, unit.text)
        body_budget = max(1, block_tokens - len(parent_ids))
        step = max(1, body_budget - min(overlap_tokens, max(0, body_budget - 1)))
        for start in range(0, len(body_ids), step):
            body_window = body_ids[start:start + body_budget]
            if not body_window:
                break
            token_ids = tuple([*parent_ids, *body_window])
            block_index = len(blocks)
            block_id = f"{document.doc_id}::e1legal::{block_index:04d}"
            blocks.append(EvidenceBlock(
                block_id=block_id,
                block_index=block_index,
                text=str(tokenizer.decode(
                    list(token_ids), skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )),
                token_ids=token_ids,
                selector="legal_structure_tokenizer",
                selector_score=0.0,
            ))
            metadata[block_id] = {
                "unit_type": str(unit.unit_type),
                "article_number": str(unit.article_number),
                "hierarchy": str(unit.hierarchy),
            }
            if start + body_budget >= len(body_ids):
                break
    return blocks, metadata


def _stable_sorted(items: Iterable[tuple[EvidenceBlock, float]]) -> list[tuple[EvidenceBlock, float]]:
    return sorted(
        items,
        key=lambda item: (-float(item[1]), item[0].block_index, item[0].block_id),
    )


def _overlap_score(left: str, right: str) -> float:
    """A deterministic BM25-like term-overlap signal without any qrels."""

    left_counts = Counter(token_list(left))
    right_counts = Counter(token_list(right))
    if not left_counts or not right_counts:
        return 0.0
    shared = left_counts.keys() & right_counts.keys()
    coverage = sum(min(left_counts[token], right_counts[token]) for token in shared)
    idf_like = sum(1.0 + math.log1p(left_counts[token]) for token in shared)
    phrase_bonus = 0.20 if normalized_text(left) in normalized_text(right) else 0.0
    return coverage / max(sum(left_counts.values()), 1) + 0.08 * idf_like / max(len(left_counts), 1) + phrase_bonus


def _structure_score(question: str, block: EvidenceBlock) -> float:
    """Prefer legal boundaries and matching citations, independently of labels."""

    text = normalized_text(block.text)
    if not text:
        return float("-inf")
    qtext = normalized_text(question)
    q_numbers = set(_NUMBER_RE.findall(qtext))
    block_numbers = set(_NUMBER_RE.findall(text))
    number_hits = len(q_numbers & block_numbers)
    q_markers = {marker for marker in _LEGAL_MARKERS if marker in qtext}
    block_markers = {marker for marker in _LEGAL_MARKERS if marker in text}
    marker_hits = len(q_markers & block_markers)
    any_legal_boundary = int(bool(block_markers))
    # The title occupies the first block.  It is useful for law/section
    # identity, but is intentionally a small tie-break rather than a ranker.
    title_bias = 0.08 if block.block_index == 0 else 0.0
    return 0.45 * number_hits + 0.18 * marker_hits + 0.04 * any_legal_boundary + title_bias


def _jaccard(left: str, right: str) -> float:
    left_tokens = set(token_list(left))
    right_tokens = set(token_list(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def propose_evidence_blocks(
    question: str,
    document: E1Document | V27Document,
    tokenizer: Any,
    bqb_scores: Mapping[str, float],
    *,
    precomputed_blocks: Sequence[EvidenceBlock] | None = None,
    block_metadata: Mapping[str, Mapping[str, str]] | None = None,
    proposals_per_source: int = E1_PROPOSALS_PER_SOURCE,
    proposal_limit: int = E1_PROPOSAL_LIMIT,
) -> list[dict[str, Any]]:
    """Union the top blocks from lexical, BQB-dense and legal-structure views.

    The input contains no reference/qrel fields.  A source rank is retained in
    the manifest so an E2 experiment can audit exactly why a passage entered
    the candidate funnel.
    """

    if document.empty_body:
        return []
    if precomputed_blocks is not None:
        blocks = list(precomputed_blocks)
    elif isinstance(document, E1Document):
        blocks, _ = make_structural_token_blocks(tokenizer, document)
    else:
        blocks = make_blocks(
            tokenizer,
            document,
            block_tokens=E1_BLOCK_TOKENS,
            overlap_tokens=E1_BLOCK_OVERLAP,
        )
    if not blocks:
        return []
    if proposals_per_source <= 0 or proposal_limit <= 0:
        raise ValueError("proposal limits must be positive")

    sources = {
        "bm25_lexical": [(block, _overlap_score(question, block.text)) for block in blocks],
        "bqb_dense": [(block, float(bqb_scores.get(block.block_id, float("-inf")))) for block in blocks],
        "legal_structure": [(block, _structure_score(question, block)) for block in blocks],
    }
    proposal_by_id: dict[str, dict[str, Any]] = {}
    seen_text: set[str] = set()
    for source_name, values in sources.items():
        for rank, (block, score) in enumerate(_stable_sorted(values)[:proposals_per_source], 1):
            if not math.isfinite(float(score)):
                continue
            text_key = normalized_text(block.text)
            if not text_key:
                continue
            row = proposal_by_id.get(block.block_id)
            if row is None:
                # Overlapping token windows can decode to the exact same text;
                # keep its earliest block deterministically.
                if text_key in seen_text:
                    continue
                row = {
                    "block_id": block.block_id,
                    "block_index": int(block.block_index),
                    "text": block.text,
                    "source_scores": {},
                    "source_ranks": {},
                }
                if block_metadata and block.block_id in block_metadata:
                    row["legal_unit"] = dict(block_metadata[block.block_id])
                proposal_by_id[block.block_id] = row
                seen_text.add(text_key)
            row["source_scores"][source_name] = float(score)
            row["source_ranks"][source_name] = int(rank)

    proposals = list(proposal_by_id.values())
    for row in proposals:
        # Reciprocal ranks normalize sources with incommensurable scores.
        row["proposal_score"] = float(sum(1.0 / rank for rank in row["source_ranks"].values()))
    proposals.sort(key=lambda row: (-float(row["proposal_score"]), int(row["block_index"]), str(row["block_id"])))
    return proposals[:proposal_limit]


def select_read_blocks(
    proposals: Sequence[Mapping[str, Any]],
    *,
    minimum_read: int = E1_MIN_READ_PASSAGES,
    read_limit: int = E1_READ_LIMIT,
    diversity_penalty: float = 0.35,
    uncertainty_margin: float = E1_UNCERTAINTY_MARGIN,
) -> list[dict[str, Any]]:
    """Choose three passages, and a fourth only near an evidence boundary."""

    if minimum_read <= 0 or read_limit < minimum_read:
        raise ValueError("require read_limit >= minimum_read > 0")
    if uncertainty_margin < 0:
        raise ValueError("uncertainty margin must be non-negative")
    remaining = [dict(row) for row in proposals]
    selected: list[dict[str, Any]] = []
    while remaining and len(selected) < read_limit:
        ranked: list[tuple[float, dict[str, Any], float]] = []
        for row in remaining:
            redundancy = max((_jaccard(str(row["text"]), str(item["text"])) for item in selected), default=0.0)
            mmr = float(row["proposal_score"]) - diversity_penalty * redundancy
            ranked.append((mmr, row, redundancy))
        ranked.sort(key=lambda item: (-item[0], int(item[1]["block_index"]), str(item[1]["block_id"])))
        mmr, row, redundancy = ranked[0]
        # The fourth deep-read passage is not unconditional.  Once three
        # complementary passages have been chosen, a clearly weaker next
        # option is treated as low uncertainty and is not sent to the CE.
        if len(selected) >= minimum_read:
            previous = float(selected[-1]["selection_score"])
            if previous - mmr > uncertainty_margin:
                break
        row["selection_score"] = float(mmr)
        row["max_jaccard_to_selected"] = float(redundancy)
        selected.append(row)
        remaining = [item for item in remaining if str(item["block_id"]) != str(row["block_id"])]
    selected.sort(key=lambda row: (int(row["block_index"]), str(row["block_id"])))
    return selected


def _manifest_row(
    *,
    qid: str,
    question: str,
    doc_id: str,
    candidate_rank: int,
    stage2_score: float,
    document: E1Document | V27Document,
    proposals: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "qid": qid,
        "query": question,
        "doc_id": doc_id,
        "candidate_rank": int(candidate_rank),
        "stage2_score": float(stage2_score),
        "empty_body": document.empty_body,
        "title": document.title,
        "link": document.link,
        "url_slug": url_slug(document.link),
        "question_type": legal_question_type(question),
        "source_family": source_family(document.link, document.title),
        "proposal_count": len(proposals),
        "proposals": [dict(item) for item in proposals],
        "blocks": [dict(item) for item in selected],
    }


def _check_complement_only(args: argparse.Namespace) -> None:
    """Reject the two locked targets before either phase opens an artifact."""

    for label, path in (("stage1", args.stage1), ("stage2", args.stage2), ("anchor", args.anchor), ("reference", args.reference)):
        lowered = str(path).replace("\\", "/").casefold()
        if "holdout" in lowered or "public" in lowered:
            raise ValueError(f"E1 is complement-only; {label} must not point to holdout/public: {path}")


def _read_manifest_meta(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"E1 manifest metadata is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("kind") != E1_KIND:
        raise V27ContractError(f"Invalid E1 manifest metadata: {path}")
    return payload


def _checkpoint_fingerprint(model_path: Path, names: Sequence[str]) -> dict[str, str]:
    """Fingerprint exact local files so same-path model replacement is visible."""

    result: dict[str, str] = {}
    for name in names:
        path = model_path / name
        if path.is_file():
            result[name] = sha256_file(path)
    if not result:
        raise FileNotFoundError(f"No fingerprintable checkpoint files in {model_path}")
    return result


def _validate_manifest(
    manifest: Mapping[tuple[str, str], Mapping[str, Any]],
    split: Any,
) -> None:
    expected = {
        (str(qid), str(doc_id))
        for qid, candidates in zip(split.qids, split.candidates, strict=True)
        for doc_id in candidates[:CANDIDATE_LIMIT]
    }
    if set(manifest) != expected:
        missing = sorted(expected - set(manifest))
        extra = sorted(set(manifest) - expected)
        raise V27ContractError(
            f"E1 evidence manifest/c35 mismatch; missing={missing[:3]} extra={extra[:3]}"
        )
    for key, row in manifest.items():
        blocks = row.get("blocks")
        proposals = row.get("proposals")
        if not isinstance(proposals, list) or len(proposals) > E1_PROPOSAL_LIMIT:
            raise V27ContractError(f"{key}: invalid proposal cap")
        if bool(row.get("empty_body")):
            if blocks:
                raise V27ContractError(f"{key}: EMPTY_BODY row may not contain evidence")
        elif not isinstance(blocks, list) or not 1 <= len(blocks) <= E1_READ_LIMIT:
            raise V27ContractError(f"{key}: invalid selected evidence cap")


def _load_bqb_vectors(
    *,
    bqb_model: Path,
    device: str,
    batch_size: int,
    split: Any,
    block_cache: Mapping[str, Sequence[EvidenceBlock]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Encode each unique block once, then score BQB similarity locally."""

    if batch_size <= 0:
        raise ValueError("bqb batch size must be positive")
    if not bqb_model.is_dir():
        raise FileNotFoundError(f"BQB checkpoint is missing: {bqb_model}")
    from sentence_transformers import SentenceTransformer

    ordered_blocks = [
        block
        for doc_id in sorted(block_cache)
        for block in block_cache[doc_id]
    ]
    if not ordered_blocks:
        return {}, {}
    model = SentenceTransformer(str(bqb_model), device=device, local_files_only=True)
    model.max_seq_length = 512
    print(json.dumps({
        "event": "e1_bqb_dense_start",
        "device": device,
        "queries": len(split.qids),
        "unique_blocks": len(ordered_blocks),
        "batch_size": batch_size,
    }, ensure_ascii=False), flush=True)
    block_vectors = np.asarray(model.encode(
        [f"{E1_BQB_PASSAGE_PREFIX}{block.text}" for block in ordered_blocks],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ), dtype=np.float32)
    query_vectors = np.asarray(model.encode(
        [f"{E1_BQB_QUERY_PREFIX}{question}" for question in split.questions],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ), dtype=np.float32)
    vectors_by_block = {
        block.block_id: block_vectors[index]
        for index, block in enumerate(ordered_blocks)
    }
    vectors_by_qid = {
        str(qid): query_vectors[index]
        for index, qid in enumerate(split.qids)
    }
    del model
    try:
        import torch

        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover - only a cleanup convenience
        pass
    return vectors_by_block, vectors_by_qid


def materialize_manifest(
    args: argparse.Namespace,
    manifest_path: Path,
    meta_path: Path,
    *,
    split_name: str = "complement",
) -> dict[str, Any]:
    """Create an auditable label-blind evidence manifest for a frozen c35.

    The command-line E1 experiment remains complement-only.  The explicit
    ``split_name`` hook exists so the separately authorized E4 protocol can
    build a holdout evidence pack *without passing a reference file here*.
    This function never opens qrels; callers must record any later label read
    in their own protocol ledger.
    """

    if manifest_path.exists() and not args.force_manifest:
        raise FileExistsError(f"Refusing to overwrite E1 manifest: {manifest_path}")
    split = load_split(split_name, args.stage1, args.stage2, args.anchor, None)
    needed = {
        str(doc_id)
        for candidates in split.candidates
        for doc_id in candidates[:CANDIDATE_LIMIT]
    }
    tokenizer = load_tokenizer(args.reranker_model)
    documents = load_raw_documents(args.contexts_zip, needed)
    block_cache: dict[str, list[EvidenceBlock]] = {}
    block_metadata: dict[str, dict[str, dict[str, str]]] = {}
    for doc_id, document in documents.items():
        if document.empty_body:
            continue
        blocks, metadata = make_structural_token_blocks(
            tokenizer,
            document,
            block_tokens=E1_BLOCK_TOKENS,
            overlap_tokens=E1_BLOCK_OVERLAP,
        )
        block_cache[doc_id] = blocks
        block_metadata[doc_id] = metadata
    vectors_by_block, vectors_by_qid = _load_bqb_vectors(
        bqb_model=args.bqb_model,
        device=args.proposal_device,
        batch_size=args.bqb_batch_size,
        split=split,
        block_cache=block_cache,
    )
    print(json.dumps({
        "event": "e1_manifest_start",
        "queries": len(split.qids),
        "expected_rows": len(split.qids) * CANDIDATE_LIMIT,
        "candidate_limit": CANDIDATE_LIMIT,
        "label_blind": True,
    }, ensure_ascii=False), flush=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    digest = hashlib.sha256()
    rows = empty_rows = proposal_total = selected_total = 0
    with temporary.open("wb") as target:
        for query_index, qid in enumerate(split.qids):
            qvec = vectors_by_qid[str(qid)]
            for candidate_rank, doc_id in enumerate(split.candidates[query_index][:CANDIDATE_LIMIT], 1):
                document = documents[doc_id]
                blocks = block_cache.get(doc_id, [])
                bqb_scores = {
                    block.block_id: float(np.dot(qvec, vectors_by_block[block.block_id]))
                    for block in blocks
                }
                proposals = propose_evidence_blocks(
                    split.questions[query_index],
                    document,
                    tokenizer,
                    bqb_scores,
                    precomputed_blocks=blocks,
                    block_metadata=block_metadata.get(doc_id),
                )
                selected = select_read_blocks(proposals)
                row = _manifest_row(
                    qid=str(qid),
                    question=split.questions[query_index],
                    doc_id=str(doc_id),
                    candidate_rank=candidate_rank,
                    stage2_score=float(split.neural_scores[query_index][doc_id]),
                    document=document,
                    proposals=proposals,
                    selected=selected,
                )
                encoded = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
                target.write(encoded)
                digest.update(encoded)
                rows += 1
                empty_rows += int(document.empty_body)
                proposal_total += len(proposals)
                selected_total += len(selected)
            if (query_index + 1) % 25 == 0 or query_index + 1 == len(split.qids):
                print(json.dumps({"event": "e1_manifest_progress", "queries": query_index + 1, "rows": rows}, ensure_ascii=False), flush=True)
    os.replace(temporary, manifest_path)
    meta = {
        "kind": E1_KIND,
        "schema_version": E1_SCHEMA_VERSION,
        "split": split_name,
        "candidate_limit": CANDIDATE_LIMIT,
        "queries": len(split.qids),
        "rows": rows,
        "empty_body_rows": empty_rows,
        "proposal_total": proposal_total,
        "selected_total": selected_total,
        "proposals_per_source": E1_PROPOSALS_PER_SOURCE,
        "proposal_limit": E1_PROPOSAL_LIMIT,
        "minimum_read_passages": E1_MIN_READ_PASSAGES,
        "read_limit": E1_READ_LIMIT,
        "fourth_passage_uncertainty_margin": E1_UNCERTAINTY_MARGIN,
        "block_tokens": E1_BLOCK_TOKENS,
        "block_overlap": E1_BLOCK_OVERLAP,
        "chunking": "raw_corpus_legal_units_then_reranker_tokenizer_windows",
        "proposal_sources": ["bm25_lexical", "bqb_dense", "legal_structure"],
        "selector": "label_blind_reciprocal_rank_union_then_mmr",
        "contexts_zip_sha256": sha256_file(args.contexts_zip),
        "bqb_model": str(args.bqb_model.resolve()),
        "bqb_checkpoint_fingerprint": _checkpoint_fingerprint(
            args.bqb_model, ("config.json", "modules.json", "model.safetensors")
        ),
        "reranker_tokenizer_fingerprint": _checkpoint_fingerprint(
            args.reranker_model, ("config.json", "tokenizer_config.json", "tokenizer.json")
        ),
        "evidence_manifest_sha256": digest.hexdigest(),
        "stage1_sha256": sha256_file(args.stage1),
        "stage2_sha256": sha256_file(args.stage2),
        "anchor_sha256": sha256_file(args.anchor),
        "reference_loaded": False,
        "holdout_opened": False,
        "public_labels_used": False,
    }
    write_json(meta_path, meta)
    return meta


def _raw_model_scores(logits: Any) -> Any:
    values = logits.logits if hasattr(logits, "logits") else logits
    if values.ndim == 1:
        return values
    if values.shape[-1] == 1:
        return values[..., 0]
    return values[..., 1] - values[..., 0]


def _row_passages(row: Mapping[str, Any]) -> list[str]:
    title = normalized_text(row.get("title", "")) or "__missing_title__"
    slug = normalized_text(row.get("url_slug", "")) or "__missing_url_slug__"
    question_type = str(row.get("question_type", "other"))
    prefix = f"[TITLE] {title} [URL_SLUG] {slug} [QUESTION_TYPE] {question_type} [PASSAGE]"
    return [
        f"{prefix} {str(block.get('text', ''))}"
        for block in row.get("blocks", [])
        if isinstance(block, dict) and str(block.get("text", ""))
    ]


def _score_records(
    model: Any,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    device: str,
    batch_size: int,
    max_length: int,
    aggregation_temperature: float,
) -> dict[tuple[str, str], float]:
    """Run frozen CE inference only; no gradients, labels or calibration."""

    if batch_size <= 0:
        raise ValueError("reranker batch size must be positive")
    if max_length <= 0 or max_length > 8192:
        raise ValueError("max length must be in 1..8192")
    import torch

    flattened: list[tuple[str, str, str, str]] = []
    for row in records:
        qid, doc_id = str(row["qid"]), str(row["doc_id"])
        flattened.extend((str(row["query"]), qid, doc_id, passage) for passage in _row_passages(row))
    per_document: dict[tuple[str, str], list[float]] = defaultdict(list)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(flattened), batch_size):
            batch = flattened[start:start + batch_size]
            encoded = tokenizer(
                [item[0] for item in batch],
                [item[3] for item in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {name: value.to(device) for name, value in encoded.items()}
            logits = _raw_model_scores(model(**encoded)).detach().float().cpu().tolist()
            for (_, qid, doc_id, _), score in zip(batch, logits, strict=True):
                per_document[(qid, doc_id)].append(float(score))
    return {
        key: temperature_logsumexp(values, temperature=aggregation_temperature)
        for key, values in per_document.items()
    }


def _rank_predictions(
    split: Any,
    scores: Mapping[tuple[str, str], float],
    manifest: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, dict[str, list[str]]]:
    predictions: dict[str, dict[str, list[str]]] = {}
    for query_index, qid in enumerate(split.qids):
        c35 = split.candidates[query_index][:CANDIDATE_LIMIT]
        ordered = sorted(
            enumerate(c35, 1),
            key=lambda item: (
                -float(scores.get((str(qid), str(item[1])), float("-inf"))),
                -float(split.neural_scores[query_index][item[1]]),
                item[0],
                str(item[1]),
            ),
        )
        # Empty bodies have no real semantic evidence.  E1 must neither add
        # an empty ID nor silently remove one already present in the frozen
        # anchor; the latter is a separately registered ablation.
        answers = [
            str(doc_id)
            for _, doc_id in ordered
            if not bool(manifest[(str(qid), str(doc_id))].get("empty_body"))
        ][:5]
        for position, doc_id in enumerate(split.anchor[query_index]):
            if not bool(manifest[(str(qid), str(doc_id))].get("empty_body")):
                continue
            answers = [value for value in answers if value != str(doc_id)]
            answers.insert(position, str(doc_id))
            answers = answers[:5]
        if len(answers) != 5 or len(set(answers)) != 5:
            raise V27ContractError(f"{qid}: E1 produced invalid top-5")
        predictions[str(qid)] = {"answer": answers}
    return predictions


def _mean_recall(predictions: Mapping[str, Mapping[str, Sequence[str]]], split: Any) -> float:
    if split.references is None:
        raise V27ContractError("E1 metric requires complement qrels")
    values = [
        recall_value(predictions[str(qid)]["answer"], split.references[index])
        for index, qid in enumerate(split.qids)
    ]
    return float(np.mean(values)) if values else 0.0


def score_and_evaluate(
    args: argparse.Namespace,
    manifest_path: Path,
    meta_path: Path,
    score_path: Path,
    prediction_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Score a sealed manifest, then open only complement qrels for reporting."""

    for path in (score_path, prediction_path, report_path):
        if path.exists() and not args.force_score:
            raise FileExistsError(f"Refusing to overwrite E1 score artifact: {path}")
    meta = _read_manifest_meta(meta_path)
    for key, artifact in (("stage1_sha256", args.stage1), ("stage2_sha256", args.stage2), ("anchor_sha256", args.anchor)):
        if str(meta.get(key, "")) != sha256_file(artifact):
            raise V27ContractError(f"E1 manifest provenance mismatch for {key}; re-materialize evidence")
    tokenizer_fingerprint = _checkpoint_fingerprint(
        args.reranker_model, ("config.json", "tokenizer_config.json", "tokenizer.json")
    )
    if meta.get("reranker_tokenizer_fingerprint") != tokenizer_fingerprint:
        raise V27ContractError(
            "E1 reranker tokenizer changed after materialization; re-materialize evidence"
        )
    # Qrels are deliberately loaded only after the manifest provenance check.
    split = load_split("complement", args.stage1, args.stage2, args.anchor, args.reference)
    manifest = load_manifest(manifest_path)
    _validate_manifest(manifest, split)
    model, tokenizer = load_relevance_model(
        args.reranker_model,
        args.score_device,
        train=False,
        fp16=args.score_device == "cuda",
    )
    score_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = score_path.with_name(score_path.name + ".tmp")
    digest = hashlib.sha256()
    scores: dict[tuple[str, str], float] = {}
    passage_total = 0
    with temporary.open("wb") as target:
        for start in range(0, len(split.qids), args.query_chunk_size):
            stop = min(start + args.query_chunk_size, len(split.qids))
            records = [
                manifest[(str(qid), str(doc_id))]
                for query_index, qid in enumerate(split.qids[start:stop], start)
                for doc_id in split.candidates[query_index][:CANDIDATE_LIMIT]
            ]
            passage_total += sum(len(_row_passages(row)) for row in records)
            partial = _score_records(
                model,
                tokenizer,
                records,
                device=args.score_device,
                batch_size=args.reranker_batch_size,
                max_length=args.max_length,
                aggregation_temperature=args.aggregation_temperature,
            )
            scores.update(partial)
            for row in records:
                key = (str(row["qid"]), str(row["doc_id"]))
                payload = {
                    "qid": key[0],
                    "doc_id": key[1],
                    "candidate_rank": int(row["candidate_rank"]),
                    "passage_count": len(_row_passages(row)),
                    "frozen_ce_score": scores.get(key),
                }
                encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
                target.write(encoded)
                digest.update(encoded)
            print(json.dumps({
                "event": "e1_frozen_ce_progress",
                "queries": stop,
                "total_queries": len(split.qids),
                "scored_documents": len(scores),
            }, ensure_ascii=False), flush=True)
    os.replace(temporary, score_path)
    predictions = _rank_predictions(split, scores, manifest)
    empty_anchor = empty_output = 0
    for index, qid in enumerate(split.qids):
        anchor_empty = {
            str(doc_id)
            for doc_id in split.anchor[index]
            if bool(manifest[(str(qid), str(doc_id))].get("empty_body"))
        }
        output_empty = {
            str(doc_id)
            for doc_id in predictions[str(qid)]["answer"]
            if bool(manifest[(str(qid), str(doc_id))].get("empty_body"))
        }
        if output_empty != anchor_empty:
            raise V27ContractError(f"{qid}: empty-body guard changed frozen anchor membership")
        empty_anchor += len(anchor_empty)
        empty_output += len(output_empty)
    write_json(prediction_path, predictions)
    stage2_predictions = {
        str(qid): {"answer": list(split.candidates[index][:5])}
        for index, qid in enumerate(split.qids)
    }
    candidate_oracle = float(np.mean([
        recall_value(split.candidates[index][:CANDIDATE_LIMIT], split.references[index])
        for index in range(len(split.qids))
    ]))
    transition = transition_metrics(split, predictions)
    report = {
        "kind": E1_KIND,
        "schema_version": E1_SCHEMA_VERSION,
        "status": "complement_diagnostic_only_not_promoted",
        "split": "complement",
        "candidate_limit": CANDIDATE_LIMIT,
        "queries": len(split.qids),
        "evidence_manifest": str(manifest_path.resolve()),
        "evidence_manifest_sha256": sha256_file(manifest_path),
        "manifest_meta_sha256": sha256_file(meta_path),
        "score_jsonl": str(score_path.resolve()),
        "score_jsonl_sha256": digest.hexdigest(),
        "predictions": str(prediction_path.resolve()),
        "metrics": {
            "v16b_anchor_recall_at_5": _mean_recall({str(qid): {"answer": split.anchor[index]} for index, qid in enumerate(split.qids)}, split),
            "stage2_c35_prefix_recall_at_5": _mean_recall(stage2_predictions, split),
            "e1_frozen_ce_recall_at_5": _mean_recall(predictions, split),
            "e1_delta_vs_v16b": float(transition["gain"]),
            "c35_candidate_oracle_recall": candidate_oracle,
        },
        "transition_vs_v16b": transition,
        "empty_body_guard": {
            "empty_ids_in_v16b_anchor": empty_anchor,
            "empty_ids_in_e1_output": empty_output,
            "new_empty_ids": 0,
            "anchor_empty_membership_preserved": True,
        },
        "frozen_model": {
            "reranker_model": str(args.reranker_model.resolve()),
            "reranker_checkpoint_fingerprint": _checkpoint_fingerprint(
                args.reranker_model, ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors")
            ),
            "device": args.score_device,
            "fp16": args.score_device == "cuda",
            "max_length": args.max_length,
            "aggregation": "temperature_logsumexp_fixed",
            "aggregation_temperature": args.aggregation_temperature,
            "passages_scored": passage_total,
        },
        "protocol": {
            "manifest_was_label_blind": not bool(meta.get("reference_loaded")),
            "labels_opened_after_manifest": True,
            "holdout_opened": False,
            "public_labels_used": False,
            "e2_training_started": False,
            "promotion_allowed": False,
        },
    }
    write_json(report_path, report)
    return report


def main() -> int:
    args = parse_args()
    _check_complement_only(args)
    if args.query_chunk_size <= 0:
        raise ValueError("query chunk size must be positive")
    if args.aggregation_temperature <= 0:
        raise ValueError("aggregation temperature must be positive")
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "e1_complement_evidence.jsonl"
    meta_path = output_dir / "e1_complement_evidence.meta.json"
    score_path = output_dir / "e1_complement_frozen_ce_scores.jsonl"
    prediction_path = output_dir / "e1_complement_frozen_ce_predictions.json"
    report_path = output_dir / "e1_complement_frozen_ce_report.json"
    if args.phase in ("materialize", "all"):
        if args.phase == "all" and manifest_path.exists() and meta_path.exists() and not args.force_manifest:
            _read_manifest_meta(meta_path)
            print(json.dumps({"event": "e1_manifest_reused", "manifest": str(manifest_path.resolve())}, ensure_ascii=False), flush=True)
        else:
            materialize_manifest(args, manifest_path, meta_path)
    if args.phase in ("score", "all"):
        if not manifest_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError("E1 score phase requires a sealed manifest and metadata")
        report = score_and_evaluate(args, manifest_path, meta_path, score_path, prediction_path, report_path)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
