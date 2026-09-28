"""Shared implementation for the V27-A evidence-boundary verifier.

The V27-A branch is intentionally downstream of the frozen V16b audit.  It
does not retrieve new documents and it never edits ranks 1--4.  This module
contains only deterministic evidence construction, leak-safe grouping,
cross-encoder scoring and the slot-5 policy used by the OOF runner.

The module is importable without loading torch or transformers.  Those heavy
dependencies are imported lazily by the training/inference helpers so that
the contract and evidence-manifest tests can run on a CPU-only environment.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import zipfile
from collections import Counter, defaultdict, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


V27_KIND = "legalir_v27_a_evidence_boundary_verifier"
V27_CONFIG_VERSION = 1
_RESOLVED_CONFIG_PATH = Path(__file__).resolve().parents[1] / "V27_A_RESOLVED_CONFIG.json"
try:
    V27_RESOLVED_CONFIG: dict[str, Any] = json.loads(
        _RESOLVED_CONFIG_PATH.read_text(encoding="utf-8")
    )
except (FileNotFoundError, json.JSONDecodeError) as exc:  # pragma: no cover - packaging guard
    raise RuntimeError(f"V27-A resolved config is unavailable or invalid: {_RESOLVED_CONFIG_PATH}") from exc

CANDIDATE_LIMIT = int(V27_RESOLVED_CONFIG["anchor"]["candidate_limit"])
FROZEN_TOP_K = int(V27_RESOLVED_CONFIG["anchor"]["frozen_top_k"])
EDITABLE_SLOT = int(V27_RESOLVED_CONFIG["anchor"]["editable_slot"])
MAX_PASSAGES_PER_DOCUMENT = int(V27_RESOLVED_CONFIG["evidence"]["max_passages_per_document"])
LEXICAL_TOP_K = int(V27_RESOLVED_CONFIG["evidence"]["lexical_top_k"])
SEMANTIC_TOP_K = int(V27_RESOLVED_CONFIG["evidence"]["semantic_top_k"])
BLOCK_TOKENS = int(V27_RESOLVED_CONFIG["evidence"]["block_tokens"])
BLOCK_OVERLAP = int(V27_RESOLVED_CONFIG["evidence"]["block_overlap"])
AGGREGATION_TEMPERATURE = float(V27_RESOLVED_CONFIG["evidence"]["aggregation_temperature"])
MODEL_PARAMETER_ESTIMATE = int(V27_RESOLVED_CONFIG["model"]["reported_parameter_estimate"])
PIPELINE_PARAMETER_ESTIMATE = int(V27_RESOLVED_CONFIG["model_budget"]["frozen_v16b_parameters"]) + MODEL_PARAMETER_ESTIMATE
PIPELINE_PARAMETER_LIMIT = int(V27_RESOLVED_CONFIG["model_budget"]["pipeline_parameter_limit"])
REGIMES = tuple(str(value) for value in V27_RESOLVED_CONFIG["regimes"])
DEFAULT_SEEDS = tuple(int(value) for value in V27_RESOLVED_CONFIG["training"]["seeds"])

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d+(?:[./-]\d+)*")
_SOURCE_NUMBER_RE = re.compile(r"(?:19|20)\d{2,}|\d{2,}")
_TEMPLATE_STOP = {
    "cho", "cua", "ve", "la", "gi", "bao", "nhieu", "nhu", "the",
    "nao", "co", "duoc", "khong", "nhung", "mot", "nhat", "trong",
}


@dataclass(frozen=True)
class V27Document:
    doc_id: str
    title: str
    body: str
    link: str
    source_family: str
    duplicate_family: str

    @property
    def empty_body(self) -> bool:
        return not self.body.strip()


@dataclass
class V27Split:
    name: str
    qids: list[str]
    questions: list[str]
    candidates: list[list[str]]
    anchor: list[list[str]]
    neural_scores: list[dict[str, float]]
    result_rows: list[dict[str, dict[str, Any]]]
    references: list[set[str]] | None
    source_candidates: list[list[str]]
    source_results: list[dict[str, dict[str, Any]]]


@dataclass(frozen=True)
class EvidenceBlock:
    block_id: str
    block_index: int
    text: str
    token_ids: tuple[int, ...]
    selector: str
    selector_score: float


class V27ContractError(ValueError):
    """Raised when a frozen V27 invariant is violated."""


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalized_text(value: Any) -> str:
    """Use the same accent-folded token normalization as the V11 audit."""

    try:
        from .v11_exact_heading_rescue import normalize
    except ImportError:
        from v11_exact_heading_rescue import normalize  # type: ignore
    return normalize(str(value))


def token_list(value: Any) -> list[str]:
    return _TOKEN_RE.findall(normalized_text(value))


_QUESTION_TYPE_MARKERS = (
    ("nhu the nao", "nhu_the_nao"),
    ("truong hop nao", "truong_hop"),
    ("gom nhung", "gom_nhung"),
    ("muc phat", "muc_phat"),
    ("dieu kien", "dieu_kien"),
    ("thoi han", "thoi_han"),
    ("bao nhieu", "bao_nhieu"),
    ("bao lau", "bao_lau"),
    ("ho so", "ho_so"),
    ("la gi", "la_gi"),
    ("co duoc", "co_duoc"),
    ("do ai", "do_ai"),
)


def legal_question_type(value: Any) -> str:
    question = normalized_text(value)
    for marker, label in _QUESTION_TYPE_MARKERS:
        if marker in question:
            return label
    return "other"


def url_slug(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "__missing_url_slug__"
    parsed = urlparse(raw)
    path = parsed.path.rstrip("/")
    slug = path.rsplit("/", 1)[-1] if path else parsed.netloc
    return normalized_text(slug) or "__missing_url_slug__"


def source_family(link: Any, title: Any = "") -> str:
    """Build a conservative law/source family key without reading labels."""

    raw = normalized_text(link)
    if not raw:
        raw = normalized_text(title)
    raw = _SOURCE_NUMBER_RE.sub(" ", raw)
    raw = " ".join(raw.split())
    return raw or "__missing_source__"


def duplicate_family(title: Any, body: Any) -> str:
    normalized = "\n".join((normalized_text(title), normalized_text(body))).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _load_v11_module() -> Any:
    try:
        from . import v11_tiny_ltr
    except ImportError:
        import v11_tiny_ltr  # type: ignore
        return v11_tiny_ltr
    return v11_tiny_ltr


def load_anchor(path: Path) -> dict[str, list[str]]:
    payload = read_json(path)
    if isinstance(payload, list):
        rows = payload
        result: dict[str, list[str]] = {}
        for row in rows:
            if not isinstance(row, dict) or "qid" not in row:
                raise V27ContractError(f"Invalid anchor row in {path}")
            item = row.get("answer")
            result[str(row["qid"])] = [str(value) for value in item]
    elif isinstance(payload, dict):
        result = {}
        for raw_qid, raw_item in payload.items():
            if not isinstance(raw_item, dict) or not isinstance(raw_item.get("answer"), list):
                raise V27ContractError(f"Invalid anchor row for {raw_qid} in {path}")
            result[str(raw_qid)] = [str(value) for value in raw_item["answer"]]
    else:
        raise V27ContractError(f"Anchor must be a JSON object or row list: {path}")
    for qid, answers in result.items():
        if len(answers) != 5 or len(set(answers)) != 5:
            raise V27ContractError(f"{qid}: anchor must contain five unique document IDs")
    return result


def load_split(
    name: str,
    stage1_path: Path,
    stage2_path: Path,
    anchor_path: Path,
    reference_path: Path | None,
) -> V27Split:
    """Load a frozen audit and replace only its baseline with the V16b anchor."""

    v11 = _load_v11_module()
    raw = v11.load_audit_split(name, stage1_path, stage2_path, reference_path)
    # V16b deliberately excluded the three exact-memory queries from its fit
    # split.  Preserve that exact boundary here; their absence in the frozen
    # complement anchor is evidence of the locked split, not a reason to
    # reconstruct an anchor for them.
    exact_memory_qids: set[str] = set()
    if name == "complement":
        stage1_payload = read_json(stage1_path)
        for row in stage1_payload.get("queries", []) if isinstance(stage1_payload, dict) else []:
            if isinstance(row, dict) and bool(row.get("metadata", {}).get("exact_memory")):
                exact_memory_qids.add(str(row["query_id"]))
    keep = [index for index, qid in enumerate(raw.qids) if str(qid) not in exact_memory_qids]
    if len(keep) != len(raw.qids):
        raw = v11.AuditSplit(
            name=raw.name,
            qids=[raw.qids[index] for index in keep],
            questions=[raw.questions[index] for index in keep],
            candidates=[raw.candidates[index] for index in keep],
            baseline=[raw.baseline[index] for index in keep],
            neural_scores=[raw.neural_scores[index] for index in keep],
            result_rows=[raw.result_rows[index] for index in keep],
            references=[raw.references[index] for index in keep]
            if raw.references is not None else None,
            stage1_path=raw.stage1_path,
        )
    anchors = load_anchor(anchor_path)
    if set(anchors) != set(raw.qids):
        missing = sorted(set(raw.qids) - set(anchors))
        extra = sorted(set(anchors) - set(raw.qids))
        raise V27ContractError(
            f"{name}: anchor QID mismatch; missing={missing[:5]} extra={extra[:5]}"
        )

    source_candidates, source_results, _ = v11.load_source_stage1_evidence(raw)
    split = V27Split(
        name=name,
        qids=list(raw.qids),
        questions=list(raw.questions),
        candidates=[list(row) for row in raw.candidates],
        anchor=[anchors[qid] for qid in raw.qids],
        neural_scores=[dict(row) for row in raw.neural_scores],
        result_rows=[dict(row) for row in raw.result_rows],
        references=[set(row) for row in raw.references] if raw.references is not None else None,
        source_candidates=source_candidates,
        source_results=source_results,
    )
    validate_split(split)
    return split


def validate_split(split: V27Split) -> None:
    if len({*split.qids}) != len(split.qids):
        raise V27ContractError(f"{split.name}: duplicate query IDs")
    if len(split.qids) != len(split.questions):
        raise V27ContractError(f"{split.name}: query/question count mismatch")
    for index, qid in enumerate(split.qids):
        candidates = split.candidates[index]
        if len(candidates) < CANDIDATE_LIMIT:
            raise V27ContractError(f"{split.name}/{qid}: fewer than c35 candidates")
        c35 = candidates[:CANDIDATE_LIMIT]
        if len(set(c35)) != CANDIDATE_LIMIT:
            raise V27ContractError(f"{split.name}/{qid}: c35 contains duplicate IDs")
        scores = split.neural_scores[index]
        if set(scores) != set(c35):
            raise V27ContractError(f"{split.name}/{qid}: Stage-2 scores are not exactly c35")
        anchor = split.anchor[index]
        if len(anchor) != 5 or len(set(anchor)) != 5:
            raise V27ContractError(f"{split.name}/{qid}: invalid five-document anchor")
        if not set(anchor).issubset(set(c35)):
            raise V27ContractError(f"{split.name}/{qid}: anchor escapes c35")
        if split.references is not None and not split.references[index]:
            raise V27ContractError(f"{split.name}/{qid}: empty qrels")


def load_documents(contexts_zip: Path, document_ids: Iterable[str]) -> dict[str, V27Document]:
    needed = {str(value) for value in document_ids if str(value)}
    documents: dict[str, V27Document] = {}
    with zipfile.ZipFile(contexts_zip) as archive:
        for info in archive.infolist():
            if not info.filename.endswith(".json"):
                continue
            raw = json.loads(archive.read(info))
            doc_id = str(raw.get("id", ""))
            if doc_id not in needed:
                continue
            title = normalized_text(raw.get("name", ""))
            body = normalized_text(raw.get("passage", ""))
            link = str(raw.get("link", "") or "")
            documents[doc_id] = V27Document(
                doc_id=doc_id,
                title=title,
                body=body,
                link=link,
                source_family=source_family(link, title),
                duplicate_family=duplicate_family(title, body),
            )
    missing = sorted(needed - set(documents))
    if missing:
        raise V27ContractError(
            f"contexts corpus is missing {len(missing)} required c35 documents: {missing[:10]}"
        )
    return documents


def _tokenizer_ids(tokenizer: Any, text: str) -> list[int]:
    if not text:
        return []
    encoded = tokenizer(text, add_special_tokens=False)
    return [int(value) for value in encoded["input_ids"]]


def make_blocks(
    tokenizer: Any,
    document: V27Document,
    *,
    block_tokens: int = BLOCK_TOKENS,
    overlap_tokens: int = BLOCK_OVERLAP,
) -> list[EvidenceBlock]:
    if block_tokens <= overlap_tokens or block_tokens <= 0:
        raise ValueError("block_tokens must be greater than overlap_tokens")
    text = "\n".join(value for value in (document.title, document.body) if value).strip()
    token_ids = _tokenizer_ids(tokenizer, text)
    blocks: list[EvidenceBlock] = []
    step = block_tokens - overlap_tokens
    for block_index, start in enumerate(range(0, len(token_ids), step)):
        ids = tuple(token_ids[start:start + block_tokens])
        if not ids:
            break
        block_text = tokenizer.decode(
            list(ids), skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        blocks.append(EvidenceBlock(
            block_id=f"{document.doc_id}::v27block::{block_index:04d}",
            block_index=block_index,
            text=str(block_text),
            token_ids=ids,
            selector="document_order",
            selector_score=0.0,
        ))
        if start + block_tokens >= len(token_ids):
            break
    return blocks


def _block_lexical_score(question: str, block: EvidenceBlock) -> float:
    q_tokens = Counter(token_list(question))
    p_tokens = Counter(token_list(block.text))
    if not q_tokens or not p_tokens:
        return 0.0
    overlap = sum((q_tokens & p_tokens).values())
    weighted = sum(1.0 + math.log1p(count) for token, count in q_tokens.items() if p_tokens[token])
    phrase_bonus = 0.25 if normalized_text(question) and normalized_text(question) in normalized_text(block.text) else 0.0
    return overlap / max(sum(q_tokens.values()), 1) + weighted / max(len(q_tokens), 1) * 0.05 + phrase_bonus


def _block_semantic_proxy(question: str, block: EvidenceBlock, stage2_score: float) -> float:
    """Label-blind fallback when a precomputed block-semantic score is absent.

    A production run may provide frozen block scores from the existing
    Stage-2 model.  The proxy keeps A0/tests runnable without loading the 760M
    model and is explicitly recorded in the manifest so it cannot be confused
    with a new learned signal.
    """

    q_tokens = set(token_list(question))
    p_tokens = set(token_list(block.text))
    coverage = len(q_tokens & p_tokens) / max(len(q_tokens), 1)
    number_match = len(set(_NUMBER_RE.findall(normalized_text(question))) & set(_NUMBER_RE.findall(normalized_text(block.text))))
    return coverage + 0.02 * number_match + 0.001 * math.tanh(float(stage2_score))


def select_evidence_blocks(
    question: str,
    document: V27Document,
    tokenizer: Any,
    *,
    stage2_score: float = 0.0,
    stage2_text: str = "",
    semantic_block_scores: Mapping[str, float] | None = None,
    precomputed_blocks: Sequence[EvidenceBlock] | None = None,
) -> list[EvidenceBlock]:
    """Select the union of top-2 lexical and top-2 semantic blocks.

    The selection is deterministic, label-blind and capped at four blocks.
    When frozen Stage‑2/BQB block scores are supplied, the selector uses only
    the prescribed top‑2 lexical + top‑2 semantic union.  Without a block
    score map, the Stage‑2 audit passage is mapped deterministically to its
    top-2 document blocks; a lexical proxy is used only when that passage is
    unavailable.
    """

    if document.empty_body:
        return []
    blocks = list(precomputed_blocks) if precomputed_blocks is not None else make_blocks(tokenizer, document)
    if not blocks:
        return []
    lexical = sorted(
        blocks,
        key=lambda block: (-_block_lexical_score(question, block), block.block_index, block.block_id),
    )[:LEXICAL_TOP_K]
    if semantic_block_scores is None:
        if stage2_text and normalized_text(stage2_text):
            # The frozen Stage-2/BQB audit already identifies the evidence
            # passage.  Map it to document blocks with a deterministic,
            # label-blind overlap score instead of invoking the large old
            # reranker once per block.  This preserves the preregistered
            # "top-2 Stage-2/BQB" selector while keeping A0 feasible.
            semantic_scored = [
                (block, _block_lexical_score(stage2_text, block))
                for block in blocks
            ]
            semantic_selector = "stage2_bqb"
        else:
            semantic_scored = [
                (block, _block_semantic_proxy(question, block, stage2_score))
                for block in blocks
            ]
            semantic_selector = "semantic_proxy"
    else:
        semantic_scored = [
            (block, float(semantic_block_scores.get(block.block_id, float("-inf"))))
            for block in blocks
        ]
        semantic_selector = "existing_semantic"
    semantic = [
        EvidenceBlock(
            block_id=block.block_id,
            block_index=block.block_index,
            text=block.text,
            token_ids=block.token_ids,
            selector=semantic_selector,
            selector_score=score,
        )
        for block, score in sorted(
            semantic_scored,
            key=lambda item: (-float(item[1]), item[0].block_index, item[0].block_id),
        )[:SEMANTIC_TOP_K]
        if math.isfinite(float(score))
    ]
    selected: list[EvidenceBlock] = []
    seen_text: set[str] = set()

    for block in [*lexical, *semantic]:
        key = normalized_text(block.text)
        if not key or key in seen_text:
            continue
        selected.append(block)
        seen_text.add(key)
        if len(selected) >= MAX_PASSAGES_PER_DOCUMENT:
            break
    selected.sort(key=lambda block: (block.block_index < 0, block.block_index, block.block_id))
    return selected[:MAX_PASSAGES_PER_DOCUMENT]


def document_manifest_row(
    split: V27Split,
    query_index: int,
    doc_id: str,
    document: V27Document,
    blocks: Sequence[EvidenceBlock],
    *,
    identity_ambiguous: bool = False,
) -> dict[str, Any]:
    c35 = split.candidates[query_index][:CANDIDATE_LIMIT]
    qid = split.qids[query_index]
    stage2_score = float(split.neural_scores[query_index][doc_id])
    stage2_row = split.result_rows[query_index].get(doc_id, {})
    return {
        "qid": qid,
        "query": split.questions[query_index],
        "doc_id": doc_id,
        "candidate_rank": c35.index(doc_id) + 1,
        "neural_rank": 1 + sorted(
            split.neural_scores[query_index],
            key=lambda value: (-split.neural_scores[query_index][value], value),
        ).index(doc_id),
        "neural_score": stage2_score,
        "source_rank": (
            split.source_candidates[query_index].index(doc_id) + 1
            if doc_id in split.source_candidates[query_index] else None
        ),
        "stage2_text": str(stage2_row.get("text", "") or ""),
        "empty_body": document.empty_body,
        "missing_title": not bool(document.title),
        "question_type": legal_question_type(split.questions[query_index]),
        "url_slug": url_slug(document.link),
        "identity_ambiguous": bool(identity_ambiguous),
        "source_family": document.source_family,
        "duplicate_family": document.duplicate_family,
        "title": document.title,
        "link": document.link,
        "blocks": [
            {
                "block_id": block.block_id,
                "block_index": block.block_index,
                "text": block.text,
                "token_ids": list(block.token_ids),
                "selector": block.selector,
                "selector_score": float(block.selector_score),
            }
            for block in blocks
        ],
    }


def load_manifest(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or "qid" not in row or "doc_id" not in row:
                raise V27ContractError(f"Invalid evidence manifest row {line_number}")
            key = (str(row["qid"]), str(row["doc_id"]))
            if key in rows:
                raise V27ContractError(f"Duplicate evidence manifest row: {key}")
            blocks = row.get("blocks")
            if bool(row.get("empty_body")):
                if blocks:
                    raise V27ContractError(f"EMPTY_BODY row has semantic evidence: {key}")
            elif not isinstance(blocks, list) or not 1 <= len(blocks) <= MAX_PASSAGES_PER_DOCUMENT:
                raise V27ContractError(f"Invalid evidence block count for {key}")
            rows[key] = dict(row)
    if not rows:
        raise V27ContractError(f"Evidence manifest is empty: {path}")
    return rows


def qrel_map(path: Path, qids: Iterable[str]) -> dict[str, set[str]]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise V27ContractError(f"Reference must be a keyed JSON object: {path}")
    wanted = {str(qid) for qid in qids}
    result: dict[str, set[str]] = {}
    for qid in wanted:
        if qid not in payload:
            raise V27ContractError(f"Reference is missing {qid}: {path}")
        row = payload[qid]
        answers = row.get("answer") if isinstance(row, dict) else None
        if not isinstance(answers, list) or not answers:
            raise V27ContractError(f"Reference has invalid answers for {qid}")
        result[qid] = {str(value) for value in answers}
    return result


def build_training_examples(
    split: V27Split,
    manifest: Mapping[tuple[str, str], Mapping[str, Any]],
    memory_features: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
    *,
    max_negative_per_category: int = 4,
) -> dict[str, list[dict[str, Any]]]:
    """Build dense relevance supervision with auditable hard-negative types."""

    if split.references is None:
        raise V27ContractError("Training examples require qrels")
    output: dict[str, list[dict[str, Any]]] = {}
    memory_features = memory_features or {}
    for index, qid in enumerate(split.qids):
        gold = set(split.references[index])
        c35 = split.candidates[index][:CANDIDATE_LIMIT]
        rows = [manifest[(qid, doc_id)] for doc_id in c35]
        by_doc = {str(row["doc_id"]): row for row in rows}
        positives = [doc_id for doc_id in c35 if doc_id in gold]
        exact_families = {
            str(by_doc[doc_id]["duplicate_family"])
            for doc_id in positives if doc_id in by_doc
        }
        selected: dict[str, set[str]] = defaultdict(set)

        def consider(category: str, doc_id: str) -> None:
            doc_id = str(doc_id)
            row = by_doc.get(doc_id)
            if row is None or doc_id in gold or bool(row.get("empty_body")):
                return
            if str(row.get("duplicate_family", "")) in exact_families:
                return
            if len(selected[category]) < max_negative_per_category:
                selected[category].add(doc_id)

        for doc_id in c35[:5]:
            consider("anchor_top5_false_positive", doc_id)
        neural_order = sorted(
            c35,
            key=lambda doc_id: (-float(split.neural_scores[index][doc_id]), doc_id),
        )
        for doc_id in neural_order:
            consider("high_neural_nonrelevant", doc_id)
        memory_row = memory_features.get(qid, {})
        memory_order = sorted(
            c35,
            key=lambda doc_id: (
                -float(memory_row.get(doc_id, {}).get("degree_corrected_memory_score", 0.0) or 0.0),
                doc_id,
            ),
        )
        for doc_id in memory_order:
            consider("memory_popularity_collision", doc_id)
        positive_families = {
            str(by_doc[doc_id].get("source_family", ""))
            for doc_id in positives if doc_id in by_doc
        }
        for doc_id in c35:
            if str(by_doc[doc_id].get("source_family", "")) in positive_families:
                consider("same_law_source_wrong_element", doc_id)

        chosen_negatives: dict[str, set[str]] = defaultdict(set)
        for category, values in selected.items():
            for doc_id in sorted(values):
                chosen_negatives[doc_id].add(category)
        records: list[dict[str, Any]] = []
        for doc_id in positives:
            row = by_doc[doc_id]
            if bool(row.get("empty_body")) or not row.get("blocks"):
                # EMPTY_BODY is excluded from semantic supervision and is a
                # hard veto at inference; it must never become a false
                # negative or an impossible positive training example.
                continue
            records.append({
                "qid": qid,
                "query": str(row.get("query", split.questions[index])),
                "doc_id": doc_id,
                "label": 1,
                "hard_negative_types": [],
                "blocks": row["blocks"],
            })
        for doc_id in sorted(chosen_negatives):
            row = by_doc[doc_id]
            records.append({
                "qid": qid,
                "query": str(row.get("query", split.questions[index])),
                "doc_id": doc_id,
                "label": 0,
                "hard_negative_types": sorted(chosen_negatives[doc_id]),
                "blocks": row["blocks"],
            })
        if records:
            output[qid] = records
    return output


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def _groups_from_union(union: _UnionFind) -> np.ndarray:
    roots = [union.find(index) for index in range(len(union.parent))]
    remap = {root: position for position, root in enumerate(sorted(set(roots)))}
    return np.asarray([remap[root] for root in roots], dtype=np.int32)


def query_template(value: str) -> str:
    tokens = token_list(value)
    if not tokens:
        return "__empty__"
    kept = [token for token in tokens if token not in _TEMPLATE_STOP and not _NUMBER_RE.fullmatch(token)]
    return " ".join(kept[:8]) or tokens[0]


def build_regime_groups(split: V27Split, manifest: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """Build the four independent, query-level transfer regimes."""

    if split.references is None:
        raise V27ContractError("Regime groups require qrels")
    size = len(split.qids)
    standard = _UnionFind(size)
    by_doc: dict[str, list[int]] = defaultdict(list)
    for index, answers in enumerate(split.references):
        for doc_id in answers:
            by_doc[str(doc_id)].append(index)
    for values in by_doc.values():
        for value in values[1:]:
            standard.union(values[0], value)
    near_duplicate = _UnionFind(size)
    vectors = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True
    ).fit_transform(split.questions)
    similarities = (vectors @ vectors.T).tocsr()
    similarities.setdiag(0.0)
    for row in range(size):
        start, stop = similarities.indptr[row], similarities.indptr[row + 1]
        for column, value in zip(similarities.indices[start:stop], similarities.data[start:stop]):
            if column > row and float(value) >= 0.75:
                near_duplicate.union(row, int(column))

    template = _UnionFind(size)
    by_template: dict[str, list[int]] = defaultdict(list)
    for index, question in enumerate(split.questions):
        by_template[query_template(question)].append(index)
    for values in by_template.values():
        for value in values[1:]:
            template.union(values[0], value)

    source = _UnionFind(size)
    by_source: dict[str, list[int]] = defaultdict(list)
    for index, answers in enumerate(split.references):
        families = {
            str(manifest[(split.qids[index], str(doc_id))].get("source_family", "__missing_source__"))
            for doc_id in answers
            if (split.qids[index], str(doc_id)) in manifest
        }
        for family in families:
            by_source[family].append(index)
    for values in by_source.values():
        for value in values[1:]:
            source.union(values[0], value)

    groups = {
        "standard": _groups_from_union(standard),
        "law_source_family": _groups_from_union(source),
        "query_template_family": _groups_from_union(template),
        "near_duplicate_family": _groups_from_union(near_duplicate),
    }
    for regime, values in groups.items():
        if len(np.unique(values)) < 2:
            raise V27ContractError(f"{regime}: fewer than two groups")
    return groups


def row_blocks(row: Mapping[str, Any]) -> list[str]:
    blocks = row.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        return []
    title = normalized_text(row.get("title", "")) or "__missing_title__"
    slug = normalized_text(row.get("url_slug", "")) or url_slug(row.get("link", ""))
    question_type = str(row.get("question_type", legal_question_type(row.get("query", ""))))
    missing_title = int(bool(row.get("missing_title", not normalized_text(row.get("title", "")))))
    metadata = (
        f"[TITLE] {title} [URL_SLUG] {slug} [QUESTION_TYPE] {question_type} "
        f"[MISSING_TITLE] {missing_title} [PASSAGE]"
    )
    return [
        f"{metadata} {str(item.get('text', ''))}"
        for item in blocks
        if isinstance(item, dict) and str(item.get("text", ""))
    ]


def temperature_logsumexp(values: Sequence[float], temperature: float = AGGREGATION_TEMPERATURE) -> float:
    if not values:
        return float("-inf")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    scaled = array / float(temperature)
    maximum = float(np.max(scaled))
    return float(temperature * (maximum + math.log(float(np.exp(scaled - maximum).sum()))))


def _raw_model_score(logits: Any) -> Any:
    """Convert sequence-classification output to one relevance logit."""

    values = logits.logits if hasattr(logits, "logits") else logits
    if values.ndim == 1:
        return values
    if values.shape[-1] == 1:
        return values[..., 0]
    return values[..., 1] - values[..., 0]


def enable_gradient_checkpointing(model: Any) -> None:
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()


def validate_local_checkpoint(model_path: Path, *, require_weights: bool) -> None:
    """Fail closed on an incomplete or accidentally network-resolved checkpoint."""

    if not model_path.is_dir():
        raise V27ContractError(f"V27-A checkpoint directory is missing: {model_path}")
    required_config = model_path / "config.json"
    if not required_config.is_file():
        raise V27ContractError(f"V27-A checkpoint has no config.json: {model_path}")
    tokenizer_files = (
        "tokenizer.json",
        "sentencepiece.bpe.model",
        "spiece.model",
        "vocab.txt",
        "vocab.json",
    )
    if not any((model_path / name).is_file() for name in tokenizer_files):
        raise V27ContractError(
            f"V27-A checkpoint has no usable tokenizer files: {model_path}"
        )
    if require_weights:
        weight_files = (
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
        if not any((model_path / name).is_file() for name in weight_files):
            raise V27ContractError(
                f"V27-A checkpoint has no complete model weights: {model_path}"
            )


def load_tokenizer(model_path: Path) -> Any:
    from transformers import AutoTokenizer

    validate_local_checkpoint(model_path, require_weights=False)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, use_fast=True
    )
    # Block construction intentionally tokenizes the whole document before
    # slicing it into 384-token passages.  The model's 512-token max length is
    # enforced later at every query-document forward pass, so do not emit a
    # misleading long-document warning during this label-blind preprocessing.
    tokenizer.model_max_length = 10**9
    return tokenizer


def load_relevance_model(
    model_path: Path,
    device: str,
    *,
    train: bool,
    fp16: bool = True,
    gradient_checkpointing: bool = False,
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification

    validate_local_checkpoint(model_path, require_weights=True)
    tokenizer = load_tokenizer(model_path)
    # GradScaler requires FP32 master parameters/gradients.  Use FP32 weights
    # for training and let autocast provide the preregistered FP16 forward
    # path; loading FP16 parameters for training produces the PyTorch
    # "Attempting to unscale FP16 gradients" failure.  Locked inference can
    # still load the student directly in FP16.
    dtype = (
        torch.float32
        if train
        else (torch.float16 if fp16 and device.startswith("cuda") else torch.float32)
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path), local_files_only=True, torch_dtype=dtype
    )
    model.to(device)
    if train and gradient_checkpointing:
        enable_gradient_checkpointing(model)
    model.train(train)
    return model, tokenizer


class PairEncodingCache:
    """CPU cache for exact query/passage tokenization pairs.

    The verifier sees the same evidence pairs in every epoch and again during
    calibration/validation.  Cache the CPU tokenizer result, while padding
    each forward batch only when it is sent to the model.  A bounded cache
    prevents a long fold from growing without limit.
    """

    def __init__(
        self,
        tokenizer: Any,
        max_length: int,
        *,
        max_entries: int = 80_000,
        shared_store: OrderedDict[tuple[str, str], dict[str, tuple[int, ...]]] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.max_entries = int(max_entries)
        self._cache = shared_store if shared_store is not None else OrderedDict()
        self.hits = 0
        self.misses = 0

    def encode(self, records: Sequence[tuple[str, str]], device: str) -> dict[str, Any]:
        import torch

        keys = [(str(query), str(passage)) for query, passage in records]
        missing = [key for key in dict.fromkeys(keys) if key not in self._cache]
        self.hits += len(keys) - len(missing)
        self.misses += len(missing)
        if missing:
            tokenized = self.tokenizer(
                [key[0] for key in missing],
                [key[1] for key in missing],
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            for index, key in enumerate(missing):
                self._cache[key] = {
                    name: tuple(int(value) for value in values[index])
                    for name, values in tokenized.items()
                }
        for key in keys:
            if key in self._cache:
                self._cache.move_to_end(key)
        while len(self._cache) > self.max_entries:
            # Keep a true bounded LRU across fold/model reloads. Eviction is
            # label-blind and cannot affect predictions, but avoids throwing
            # away the whole cache whenever one batch crosses the limit.
            self._cache.popitem(last=False)
        items = [
            {name: list(values) for name, values in self._cache[key].items()}
            for key in keys
        ]
        padded = self.tokenizer.pad(items, padding=True, return_tensors="pt")
        return {name: value.to(device) for name, value in padded.items()}

    def stats(self) -> dict[str, int]:
        return {
            "entries": int(len(self._cache)),
            "hits": int(self.hits),
            "misses": int(self.misses),
        }


def score_document_records(
    model: Any,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    device: str,
    *,
    max_length: int = BLOCK_TOKENS,
    batch_size: int = 2,
) -> dict[str, float]:
    """Score query-document rows by fixed temperature-logsumexp passage aggregation."""

    import torch

    if max_length != BLOCK_TOKENS:
        raise V27ContractError("V27-A max sequence length is fixed at 384")
    model.eval()
    pair_cache = getattr(tokenizer, "_v27_pair_encoding_cache", None)
    flattened: list[tuple[str, str, str, str]] = []
    for row in records:
        qid = str(row["qid"])
        doc_id = str(row["doc_id"])
        for passage in row_blocks(row):
            flattened.append((str(row.get("query", "")), qid, doc_id, passage))
    by_doc: dict[tuple[str, str], list[float]] = defaultdict(list)
    with torch.inference_mode():
        for start in range(0, len(flattened), max(1, int(batch_size))):
            batch = flattened[start:start + max(1, int(batch_size))]
            pairs = [(item[0], item[3]) for item in batch]
            if pair_cache is not None and pair_cache.max_length == max_length:
                encoded = pair_cache.encode(pairs, device)
            else:
                encoded = tokenizer(
                    [item[0] for item in batch],
                    [item[3] for item in batch],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = _raw_model_score(model(**encoded)).detach().float().cpu().tolist()
            for item, value in zip(batch, logits, strict=True):
                by_doc[(item[1], item[2])].append(float(value))
    return {
        f"{qid}\t{doc_id}": temperature_logsumexp(values)
        for (qid, doc_id), values in by_doc.items()
    }


class ScoreCalibrator:
    """A one-dimensional Platt calibrator fit on one fold's calibration set."""

    def __init__(self) -> None:
        self.model: LogisticRegression | None = None
        self.constant: float | None = None

    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> "ScoreCalibrator":
        values = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        target = np.asarray(labels, dtype=np.int8)
        if len(values) == 0 or len(set(target.tolist())) < 2:
            self.constant = float(np.mean(target)) if len(target) else 0.0
            self.model = None
        else:
            self.model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
            self.model.fit(values, target)
            self.constant = None
        return self

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScoreCalibrator":
        result = cls()
        if payload.get("kind") == "constant":
            result.constant = float(payload.get("value", 0.0))
            return result
        if payload.get("kind") != "platt_logistic":
            raise V27ContractError(f"Unknown calibrator kind: {payload.get('kind')}")
        result.model = LogisticRegression(C=1.0, solver="lbfgs")
        result.model.classes_ = np.asarray([0, 1], dtype=np.int8)
        result.model.coef_ = np.asarray([[float(payload["coef"])]], dtype=np.float64)
        result.model.intercept_ = np.asarray([float(payload["intercept"])], dtype=np.float64)
        result.model.n_features_in_ = 1
        result.model.n_iter_ = np.asarray([1], dtype=np.int32)
        return result

    def predict(self, scores: Sequence[float]) -> np.ndarray:
        values = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        if self.model is None:
            return np.full(len(values), float(self.constant or 0.0), dtype=np.float64)
        return self.model.predict_proba(values)[:, 1]

    def to_dict(self) -> dict[str, Any]:
        if self.model is None:
            return {"kind": "constant", "value": float(self.constant or 0.0)}
        return {
            "kind": "platt_logistic",
            "coef": float(self.model.coef_[0, 0]),
            "intercept": float(self.model.intercept_[0]),
        }


def predictions_from_probabilities(
    split: V27Split,
    probabilities: Mapping[str, Mapping[str, float]],
    policy: Mapping[str, Any],
    *,
    empty_body_by_doc: Mapping[tuple[str, str], bool] | None = None,
    memory_features: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
    eligibility_by_doc: Mapping[tuple[str, str], bool] | None = None,
) -> dict[str, dict[str, list[str]]]:
    """Apply the only permitted V27 action: a selective replacement at slot 5."""

    empty_body_by_doc = empty_body_by_doc or {}
    memory_features = memory_features or {}
    eligibility_by_doc = eligibility_by_doc or {}
    output: dict[str, dict[str, list[str]]] = {}
    for index, qid in enumerate(split.qids):
        anchor = list(split.anchor[index])
        if len(anchor) != 5:
            raise V27ContractError(f"{split.name}/{qid}: anchor is not five IDs")
        # The frozen prefix is copied, never sorted or reconstructed.
        answers = list(anchor)
        c35 = split.candidates[index][:CANDIDATE_LIMIT]
        if any(bool(empty_body_by_doc.get((qid, doc_id), False)) for doc_id in anchor[:FROZEN_TOP_K]):
            # The old anchor itself is known dirty on a few public rows.  Do
            # not silently repair it because that would violate the prefix
            # invariant; report/gate code will close the branch instead.
            output[qid] = {"answer": answers}
            continue
        challenger_ids = prefilter_challengers(
            split, index, policy, memory_features=memory_features.get(qid, {})
        )
        incumbent = anchor[EDITABLE_SLOT - 1]
        if not bool(eligibility_by_doc.get((qid, incumbent), True)):
            output[qid] = {"answer": answers}
            continue
        incumbent_probability = float(probabilities.get(qid, {}).get(incumbent, 0.0))
        absolute_min = float(policy.get("min_challenger_probability", 0.5))
        delta_min = float(policy.get("min_delta", 0.0))
        best: tuple[float, float, int, str] | None = None
        for candidate_rank in challenger_ids:
            doc_id = c35[candidate_rank - 1]
            if (
                doc_id in anchor
                or bool(empty_body_by_doc.get((qid, doc_id), False))
                or not bool(eligibility_by_doc.get((qid, doc_id), True))
            ):
                continue
            probability = float(probabilities.get(qid, {}).get(doc_id, 0.0))
            delta = probability - incumbent_probability
            if probability < absolute_min or delta < delta_min:
                continue
            candidate = (delta, probability, -candidate_rank, doc_id)
            if best is None or candidate > best:
                best = candidate
        if best is not None:
            answers[-1] = best[3]
        if len(answers) != 5 or len(set(answers)) != 5:
            raise V27ContractError(f"{split.name}/{qid}: invalid slot-5 output")
        if answers[:FROZEN_TOP_K] != anchor[:FROZEN_TOP_K]:
            raise V27ContractError(f"{split.name}/{qid}: top 1-4 invariant violated")
        if answers[-1] not in c35:
            raise V27ContractError(f"{split.name}/{qid}: replacement escapes c35")
        output[qid] = {"answer": answers}
    return output


def prefilter_challengers(
    split: V27Split,
    index: int,
    policy: Mapping[str, Any],
    *,
    memory_features: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[int]:
    c35 = split.candidates[index][:CANDIDATE_LIMIT]
    anchor = set(split.anchor[index])
    tail = [rank for rank in range(FROZEN_TOP_K + 2, CANDIDATE_LIMIT + 1) if c35[rank - 1] not in anchor]
    max_challengers = int(policy.get("max_prefiltered_challengers", 6))
    neural_order = sorted(
        tail,
        key=lambda rank: (-float(split.neural_scores[index][c35[rank - 1]]), rank),
    )
    source_rank = {
        doc_id: rank + 1 for rank, doc_id in enumerate(split.source_candidates[index])
    }
    lexical_order = sorted(
        tail,
        key=lambda rank: (source_rank.get(c35[rank - 1], 10_000), rank),
    )
    # The memory union is intentionally read only as a prefilter signal.  It
    # is not used as a semantic score or as a training label.
    memory_order = sorted(
        tail,
        key=lambda rank: (
            -float((memory_features or {}).get(
                c35[rank - 1], {}
            ).get("degree_corrected_memory_score", 0.0) or 0.0),
            rank,
        ),
    )
    union: list[int] = []
    for order in (neural_order, lexical_order, memory_order):
        for rank in order:
            if rank not in union:
                union.append(rank)
            if len(union) >= max_challengers:
                return union
    return union[:max_challengers]


def anchor_predictions(split: V27Split) -> dict[str, dict[str, list[str]]]:
    return {qid: {"answer": list(split.anchor[index])} for index, qid in enumerate(split.qids)}


def recall_value(prediction: Sequence[str], gold: set[str]) -> float:
    return len(set(str(value) for value in prediction) & gold) / len(gold) if gold else 0.0


def transition_metrics(
    split: V27Split,
    predictions: Mapping[str, Mapping[str, Sequence[str]]],
) -> dict[str, Any]:
    if split.references is None:
        raise V27ContractError("Transition metrics require qrels")
    deltas: list[float] = []
    improved = harmed = neutral = changed = destructive = successful_rescues = 0
    rows: list[dict[str, Any]] = []
    for index, qid in enumerate(split.qids):
        anchor = split.anchor[index]
        proposed = [str(value) for value in predictions[qid]["answer"]]
        gold = split.references[index]
        before = recall_value(anchor, gold)
        after = recall_value(proposed, gold)
        delta = after - before
        deltas.append(delta)
        is_changed = proposed != anchor
        changed += int(is_changed)
        if delta > 1e-12:
            improved += 1
        elif delta < -1e-12:
            harmed += 1
        elif is_changed:
            neutral += 1
        if (set(proposed) & gold) - set(anchor):
            successful_rescues += 1
        if (set(anchor) & gold) - set(proposed):
            destructive += 1
        rows.append({"qid": qid, "before": before, "after": after, "delta": delta, "changed": is_changed})
    recalls = [row["after"] for row in rows]
    gains = np.asarray(deltas, dtype=np.float64)
    return {
        "queries": len(rows),
        "recall": float(statistics.fmean(recalls)) if recalls else 0.0,
        "anchor_recall": float(statistics.fmean([row["before"] for row in rows])) if rows else 0.0,
        "gain": float(gains.mean()) if len(gains) else 0.0,
        "net_query_equivalents": float(gains.sum()),
        "changed_queries": changed,
        "improved_queries": improved,
        "harmed_queries": harmed,
        "neutral_changed_queries": neutral,
        "successful_rescues": successful_rescues,
        "destructive_changes": destructive,
        "outcome_changing_precision": float(
            successful_rescues / (successful_rescues + destructive)
        ) if successful_rescues + destructive else 1.0,
        "rows": rows,
    }


def select_boundary_policy(
    split: V27Split,
    probabilities: Mapping[str, Mapping[str, float]],
    *,
    min_probability: float = 0.5,
    minimum_precision: float = 0.90,
    maximum_destructive: int = 2,
    min_improvements: int = 5,
    memory_features: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
    empty_body_by_doc: Mapping[tuple[str, str], bool] | None = None,
    eligibility_by_doc: Mapping[tuple[str, str], bool] | None = None,
) -> dict[str, Any]:
    """Choose one selective slot-5 gate from calibration data only."""

    if split.references is None:
        raise V27ContractError("Boundary policy selection requires qrels")
    memory_features = memory_features or {}
    empty_body_by_doc = empty_body_by_doc or {}
    eligibility_by_doc = eligibility_by_doc or {}
    deltas: set[float] = {0.0}
    for index, qid in enumerate(split.qids):
        c35 = split.candidates[index][:CANDIDATE_LIMIT]
        incumbent = split.anchor[index][EDITABLE_SLOT - 1]
        incumbent_probability = float(probabilities.get(qid, {}).get(incumbent, 0.0))
        for rank in prefilter_challengers(
            split, index, {"max_prefiltered_challengers": 6},
            memory_features=memory_features.get(qid, {}),
        ):
            doc_id = c35[rank - 1]
            if doc_id not in probabilities.get(qid, {}):
                continue
            deltas.add(float(probabilities[qid][doc_id]) - incumbent_probability)
    candidates = sorted(deltas, reverse=True)
    best: dict[str, Any] | None = None
    for threshold in candidates:
        policy = {
            "kind": "v27_a_slot5",
            "pool": CANDIDATE_LIMIT,
            "max_prefiltered_challengers": 6,
            "min_challenger_probability": float(min_probability),
            "min_delta": float(threshold),
            "aggregation": "temperature_logsumexp_fixed",
            "empty_body_hard_veto": True,
        }
        predictions = predictions_from_probabilities(
            split, probabilities, policy,
            memory_features=memory_features,
            empty_body_by_doc=empty_body_by_doc,
            eligibility_by_doc=eligibility_by_doc,
        )
        metrics = transition_metrics(split, predictions)
        acceptable = (
            metrics["outcome_changing_precision"] >= minimum_precision
            and metrics["improved_queries"] >= min_improvements
            and metrics["destructive_changes"] <= maximum_destructive
        )
        if not acceptable:
            continue
        key = (
            metrics["gain"],
            metrics["outcome_changing_precision"],
            -metrics["neutral_changed_queries"],
            -threshold,
        )
        candidate = {"policy": policy, "calibration_metrics": metrics, "key": list(key)}
        if best is None or tuple(key) > tuple(best["key"]):
            best = candidate
    if best is None:
        return {
            "policy": {
                "kind": "v27_a_slot5",
                "pool": CANDIDATE_LIMIT,
                "max_prefiltered_challengers": 6,
                "min_challenger_probability": float(min_probability),
                "min_delta": float("inf"),
                "aggregation": "temperature_logsumexp_fixed",
                "empty_body_hard_veto": True,
                "abstain_all": True,
            },
            "calibration_metrics": transition_metrics(split, anchor_predictions(split)),
            "abstain": True,
        }
    best.pop("key", None)
    best["abstain"] = False
    return best


def top_prefix_identical(
    split: V27Split,
    predictions: Mapping[str, Mapping[str, Sequence[str]]],
) -> bool:
    return all(
        list(predictions[qid]["answer"])[:FROZEN_TOP_K] == split.anchor[index][:FROZEN_TOP_K]
        for index, qid in enumerate(split.qids)
    )


def bootstrap_gain(
    split: V27Split,
    predictions: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    samples: int = 10_000,
    seed: int = 20260901,
) -> dict[str, float]:
    if split.references is None:
        raise V27ContractError("Bootstrap requires qrels")
    per_query = np.asarray([
        recall_value(predictions[qid]["answer"], split.references[index])
        - recall_value(split.anchor[index], split.references[index])
        for index, qid in enumerate(split.qids)
    ], dtype=np.float64)
    if not len(per_query):
        return {"mean": 0.0, "lower_95": 0.0, "upper_95": 0.0}
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = float(rng.choice(per_query, size=len(per_query), replace=True).mean())
    return {
        "mean": float(per_query.mean()),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
    }


def duplicate_family_invariant(
    split: V27Split,
    predictions: Mapping[str, Mapping[str, Sequence[str]]],
    manifest: Mapping[tuple[str, str], Mapping[str, Any]],
) -> bool:
    """Reject outputs that emit two documents from one exact identity family."""

    for qid in split.qids:
        families = [
            str(manifest.get((qid, str(doc_id)), {}).get("duplicate_family", ""))
            for doc_id in predictions[qid]["answer"]
        ]
        if not all(families) or len(set(families)) != len(families):
            return False
    return True


def contract_report(
    split: V27Split,
    predictions: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    fold_gains: Sequence[float],
    robust_gain: float,
    bootstrap: Mapping[str, float],
    empty_body_outputs: int,
    manifest: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
    seed_reports: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    metrics = transition_metrics(split, predictions)
    gates_config = V27_RESOLVED_CONFIG["gates"]
    seed_gains = [float(item.get("gain", 0.0)) for item in seed_reports]
    gates = {
        "oof_recall_gain_ge_0_010": metrics["gain"] >= float(gates_config["oof_recall_gain_min"]),
        "robust_gain_ge_0_008": robust_gain >= float(gates_config["robust_gain_min"]),
        "bootstrap_lower_95_ge_0_003": float(bootstrap["lower_95"]) >= float(gates_config["bootstrap_lower_95_min"]),
        "four_regime_transfer_nonnegative": all(float(value) >= float(gates_config["all_regime_gain_min"]) for value in fold_gains),
        "net_query_equivalent_ge_12": metrics["net_query_equivalents"] >= float(gates_config["net_query_equivalent_min"]),
        "outcome_changing_precision_ge_0_90": metrics["outcome_changing_precision"] >= float(gates_config["outcome_changing_precision_min"]),
        "destructive_le_2": metrics["destructive_changes"] <= int(gates_config["destructive_max"]),
        "top_1_to_4_byte_identical": top_prefix_identical(split, predictions),
        "empty_body_outputs_eq_0": int(empty_body_outputs) <= int(gates_config["empty_body_outputs_max"]),
        "duplicate_family_invariant": (
            duplicate_family_invariant(split, predictions, manifest)
            if manifest is not None else True
        ),
        "seed_gain_difference_le_0_002": (
            max(seed_gains) - min(seed_gains) <= float(gates_config["seed_gain_difference_max"])
            if len(seed_gains) >= 2 else False
        ),
    }
    return {
        "metrics": metrics,
        "fold_gains": [float(value) for value in fold_gains],
        "robust_gain": float(robust_gain),
        "bootstrap": dict(bootstrap),
        "gates": gates,
        "passed": bool(all(gates.values())),
    }


def empty_body_count(
    split: V27Split,
    predictions: Mapping[str, Mapping[str, Sequence[str]]],
    manifest: Mapping[tuple[str, str], Mapping[str, Any]],
) -> int:
    count = 0
    for qid, row in predictions.items():
        for doc_id in row["answer"]:
            if bool(manifest.get((qid, str(doc_id)), {}).get("empty_body", False)):
                count += 1
    return count


def manifest_empty_map(
    manifest: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[tuple[str, str], bool]:
    return {key: bool(row.get("empty_body", False)) for key, row in manifest.items()}


def manifest_eligibility_map(
    manifest: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[tuple[str, str], bool]:
    """Return the conservative evidence eligibility gate for inference."""

    return {
        key: bool(
            not row.get("empty_body", False)
            and not row.get("missing_title", False)
            and not row.get("identity_ambiguous", False)
            and isinstance(row.get("blocks"), list)
            and bool(row.get("blocks"))
        )
        for key, row in manifest.items()
    }


def group_folds(groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = np.unique(groups)
    if len(unique) < 2:
        raise V27ContractError("Grouped cross-fit requires at least two groups")
    indices = np.arange(len(groups), dtype=np.int32)
    splitter = GroupKFold(n_splits=min(5, len(unique)))
    return [
        (train.astype(np.int32), validation.astype(np.int32))
        for train, validation in splitter.split(indices, groups=groups)
    ]


def inner_fit_calibration_indices(
    train_indices: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=int(seed))
    local = np.arange(len(train_indices), dtype=np.int32)
    fit_local, calibration_local = next(
        splitter.split(local, groups=groups[train_indices])
    )
    return train_indices[fit_local], train_indices[calibration_local]


def model_parameter_count(model: Any) -> int:
    return int(sum(int(parameter.numel()) for parameter in model.parameters()))


def model_compliance(model: Any | None = None) -> dict[str, Any]:
    actual = model_parameter_count(model) if model is not None else None
    model_parameters = int(actual or MODEL_PARAMETER_ESTIMATE)
    total = 1_090_010_000 + model_parameters
    return {
        "anchor_parameters": 1_090_010_000,
        "verifier_parameters": model_parameters,
        "total_parameters": total,
        "parameter_limit": PIPELINE_PARAMETER_LIMIT,
        "under_4b": total < PIPELINE_PARAMETER_LIMIT,
        "estimate_used": actual is None,
    }


def atomic_torch_save(payload: Mapping[str, Any], path: Path | str) -> str:
    """Atomically save a torch payload and return its SHA-256 hex digest."""

    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, target)
    return sha256_file(target)


def _torch_load_mapping(path: Path) -> dict[str, Any]:
    import torch

    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise V27ContractError(f"Torch checkpoint is not a mapping: {path}")
    return dict(payload)


def verify_torch_checkpoint(path: Path | str, required_keys: Sequence[str] = ()) -> dict[str, Any]:
    """Fail closed when a torch checkpoint is missing keys."""

    target = Path(path)
    if not target.is_file():
        raise V27ContractError(f"Torch checkpoint is missing: {target}")
    payload = _torch_load_mapping(target)
    missing = [str(key) for key in required_keys if key not in payload]
    if missing:
        raise V27ContractError(f"Torch checkpoint {target} is missing keys: {missing}")
    return payload


def verify_torch_state_dict(path: Path | str) -> dict[str, Any]:
    """Fail closed when a saved state dict cannot be reloaded as a mapping."""

    target = Path(path)
    if not target.is_file():
        raise V27ContractError(f"Torch state dict is missing: {target}")
    payload = _torch_load_mapping(target)
    if not payload:
        raise V27ContractError(f"Torch state dict is empty: {target}")
    return payload


def build_weight_manifest(
    *,
    regime: str,
    outer_fold: int,
    inner_seed: int,
    neural_seed: int,
    fit_seed: int,
    split_qids_sha256: str,
    state_sha256: str,
    retained_sha256: str,
    weight_path: Path | str,
) -> dict[str, Any]:
    """Build a lineage manifest for a retained E3-B neural weight file."""

    import datetime

    return {
        "kind": "legalir_e3b_neural_weight_manifest",
        "regime": str(regime),
        "outer_fold": int(outer_fold),
        "inner_seed": int(inner_seed),
        "neural_seed": int(neural_seed),
        "fit_seed": int(fit_seed),
        "split_qids_sha256": str(split_qids_sha256),
        "state_sha256": str(state_sha256),
        "retained_sha256": str(retained_sha256),
        "weight_path": str(weight_path),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
