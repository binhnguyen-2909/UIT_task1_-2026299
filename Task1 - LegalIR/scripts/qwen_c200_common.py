"""Shared, label-safe artifacts for the Qwen c35/c67/c200 research branch.

This branch is intentionally isolated from E4/E5.  It builds candidate pools
only from an already frozen Stage-1 audit, creates deterministic legal-text
packs without qrels, and validates reusable Qwen score caches.  Labels are
loaded only by the separate nested evaluator after its manifest is written.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from v27_a_common import CANDIDATE_LIMIT, V27ContractError, json_sha256, load_split, read_json, sha256_file


QWEN_BRANCH_KIND = "legalir_qwen_c200_pool_research"
QWEN_MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
SHORT_MAX_TOKENS = 768
LONG_MAX_TOKENS = 4096
POOL_SIZES = {"A_c35_short": 35, "B_c35_long": 35, "C_c67_long": 67, "D_c200_long": 200, "E_c67_short": 67, "F_shortlist8_long": 8}
INSTRUCTION = "Given a Vietnamese legal question, retrieve passages that directly answer it, including legal conditions, exceptions, articles, clauses, and document numbers."
FORBIDDEN_TERMS = ("holdout", "public", "submission", "locked")
_LEGAL_BOUNDARY = re.compile(r"(?im)(?=^(?:điều|chương|mục|khoản|điểm)\s+\d+|\n\s*\n)")
_WORD = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)
_NUMBER = re.compile(r"\d+[\w./-]*", re.UNICODE)


@dataclass(frozen=True)
class RawDocument:
    doc_id: str
    title: str
    body: str
    link: str

    @property
    def empty(self) -> bool:
        return not bool(self.title.strip() or self.body.strip())


@dataclass(frozen=True)
class CandidateUniverse:
    split: Any
    candidates: Mapping[str, tuple[str, ...]]
    stage1_sha256: str
    stage2_sha256: str
    anchor_sha256: str
    reference_sha256: str | None


def cap_cpu_threads(threads: int) -> None:
    if int(threads) < 1:
        raise ValueError("--cpu-threads must be positive")
    value = str(int(threads))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
        os.environ[name] = value
    try:
        import torch
        torch.set_num_threads(int(threads))
        torch.set_num_interop_threads(int(threads))
    except RuntimeError:
        pass


def require_complement_file(path: Path, label: str) -> None:
    normalized = str(path).replace("\\", "/").casefold()
    if any(term in normalized for term in FORBIDDEN_TERMS):
        raise V27ContractError(f"Qwen c200 branch is complement-only; {label} may not refer to holdout/public: {path}")
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def require_contexts_zip(path: Path) -> None:
    """Validate the raw, label-free corpus separately from split artifacts.

    The competition corpus is commonly stored below ``data/raw/public_test``;
    that directory name does not mean public labels were loaded.  Keep the
    strict guard for manifests/predictions, but do not reject this legitimate
    raw contexts ZIP merely because its parent contains the word ``public``.
    """

    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing contexts ZIP: {path}")
    normalized = str(path).replace("\\", "/").casefold()
    if any(term in normalized for term in ("holdout", "submission", "locked")):
        raise V27ContractError(f"Raw contexts ZIP cannot refer to sealed/output data: {path}")
    if Path(path).suffix.casefold() != ".zip":
        raise V27ContractError(f"Contexts input must be a ZIP: {path}")


def checkpoint_fingerprint(model_path: Path) -> dict[str, str]:
    files = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors", "model.safetensors.index.json")
    result = {name: sha256_file(model_path / name) for name in files if (model_path / name).is_file()}
    if "config.json" not in result or not {"model.safetensors", "model.safetensors.index.json"} & set(result):
        raise FileNotFoundError(f"Qwen checkpoint is incomplete: {model_path}")
    return result


def _raw_stage1_rows(stage1_path: Path) -> dict[str, tuple[tuple[str, ...], Mapping[str, Any]]]:
    payload = read_json(stage1_path)
    if not isinstance(payload, Mapping):
        raise V27ContractError("Qwen c200 branch requires a JSON-object Stage-1 audit")
    queries, rankings = payload.get("queries"), payload.get("stage1_candidates")
    if not isinstance(queries, list) or not isinstance(rankings, list) or len(queries) != len(rankings):
        raise V27ContractError("Qwen c200 branch requires a valid Stage-1 audit with stage1_candidates")
    result: dict[str, tuple[tuple[str, ...], Mapping[str, Any]]] = {}
    for query, ranking in zip(queries, rankings, strict=True):
        if not isinstance(query, Mapping) or not isinstance(ranking, list):
            raise V27ContractError("Invalid Stage-1 query/ranking row")
        qid = str(query.get("query_id", ""))
        unique = tuple(dict.fromkeys(str(value) for value in ranking if str(value)))
        if not qid or len(unique) < max(POOL_SIZES.values()) or qid in result:
            raise V27ContractError(f"{qid or '<missing>'}: Stage-1 does not provide a unique c200 pool")
        result[qid] = (unique, query)
    return result


def load_candidate_universe(
    *,
    stage1: Path,
    stage2: Path,
    anchor: Path,
    reference: Path | None,
) -> CandidateUniverse:
    """Load the V16b complement boundary and an exact Stage-1 c200 superset."""

    for path, label in ((stage1, "Stage-1"), (stage2, "Stage-2"), (anchor, "V16b anchor")):
        require_complement_file(path, label)
    if reference is not None:
        require_complement_file(reference, "complement reference")
    split = load_split("complement", stage1, stage2, anchor, reference)
    raw = _raw_stage1_rows(stage1)
    pools: dict[str, tuple[str, ...]] = {}
    for index, qid in enumerate(split.qids):
        ranking, _ = raw.get(str(qid), ((), {}))
        if len(ranking) < 200:
            raise V27ContractError(f"{qid}: c200 Stage-1 pool is incomplete")
        c35 = tuple(str(value) for value in split.candidates[index][:CANDIDATE_LIMIT])
        if tuple(ranking[:CANDIDATE_LIMIT]) != c35:
            raise V27ContractError(f"{qid}: Stage-1 c35 prefix drifted from sealed Stage-2/V16b universe")
        pools[str(qid)] = ranking[:200]
    return CandidateUniverse(
        split=split, candidates=pools,
        stage1_sha256=sha256_file(stage1), stage2_sha256=sha256_file(stage2), anchor_sha256=sha256_file(anchor),
        reference_sha256=None if reference is None else sha256_file(reference),
    )


def pool_for(universe: CandidateUniverse, qid: str, name: str) -> tuple[str, ...]:
    if name not in POOL_SIZES:
        raise ValueError(f"Unknown Qwen research configuration: {name}")
    return universe.candidates[str(qid)][:POOL_SIZES[name]]


def needed_document_ids(universe: CandidateUniverse, *, limit: int) -> set[str]:
    return {doc_id for qid in universe.split.qids for doc_id in universe.candidates[str(qid)][:int(limit)]}


def load_raw_documents(contexts_zip: Path, document_ids: Iterable[str]) -> dict[str, RawDocument]:
    require_contexts_zip(contexts_zip)
    wanted = {str(value) for value in document_ids}
    result: dict[str, RawDocument] = {}
    with zipfile.ZipFile(contexts_zip) as archive:
        for info in archive.infolist():
            if not info.filename.endswith(".json"):
                continue
            raw = json.loads(archive.read(info))
            doc_id = str(raw.get("id", ""))
            if doc_id in wanted:
                result[doc_id] = RawDocument(doc_id, str(raw.get("name", "") or ""), str(raw.get("passage", "") or ""), str(raw.get("link", "") or ""))
    missing = wanted - set(result)
    if missing:
        raise V27ContractError(f"Contexts ZIP misses {len(missing)} Stage-1 c200 documents: {sorted(missing)[:5]}")
    return result


def _terms(text: str) -> set[str]:
    return {value.casefold() for value in _WORD.findall(text) if len(value) > 1}


def _segments(text: str) -> list[str]:
    chunks = [value.strip() for value in _LEGAL_BOUNDARY.split(text) if value.strip()]
    return chunks or ([text.strip()] if text.strip() else [])


def pack_document(tokenizer: Any, *, question: str, document: RawDocument, max_tokens: int) -> tuple[str, dict[str, Any]]:
    """Label-blind legal-boundary pack retaining order after lexical selection."""

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    heading = f"[TITLE] {document.title.strip()}\n" if document.title.strip() else ""
    full = f"{heading}[DOCUMENT]\n{document.body.strip()}".strip()
    full_ids = tokenizer(full, add_special_tokens=False).get("input_ids", [])
    if len(full_ids) <= max_tokens:
        return full, {"kind": "full_document", "source_tokens": len(full_ids), "packed_tokens": len(full_ids), "segments": 1}
    q_terms, q_numbers = _terms(question), set(_NUMBER.findall(question))
    scored: list[tuple[int, float, str, int]] = []
    for index, segment in enumerate(_segments(document.body)):
        ids = tokenizer(segment, add_special_tokens=False).get("input_ids", [])
        if not ids:
            continue
        terms = _terms(segment)
        score = float(len(q_terms & terms)) / max(len(q_terms), 1) + 0.25 * len(q_numbers & set(_NUMBER.findall(segment)))
        scored.append((index, score, segment, len(ids)))
    # Select informative legal units, then restore statutory order.  A segment
    # is clipped only at a token boundary when a final budget remainder exists.
    selected: list[tuple[int, str, int]] = []
    used = len(tokenizer(heading + "[DOCUMENT]\n", add_special_tokens=False).get("input_ids", []))
    for index, _, segment, tokens in sorted(scored, key=lambda row: (-row[1], row[0])):
        if used >= max_tokens:
            break
        take = min(tokens, max_tokens - used)
        if take <= 0:
            continue
        if take < tokens:
            ids = tokenizer(segment, add_special_tokens=False).get("input_ids", [])[:take]
            segment = tokenizer.decode(ids, skip_special_tokens=True)
        selected.append((index, segment, take)); used += take
    selected.sort(key=lambda row: row[0])
    packed = f"{heading}[DOCUMENT]\n" + "\n".join(value for _, value, _ in selected)
    return packed.strip(), {"kind": "legal_boundary_lexical_order", "source_tokens": len(full_ids), "packed_tokens": used, "segments": len(selected)}


def cache_lineage(universe: CandidateUniverse, *, contexts_zip: Path, model_fingerprint: Mapping[str, str], variant: str, max_tokens: int) -> dict[str, Any]:
    return {
        "kind": QWEN_BRANCH_KIND, "schema_version": 1, "split": "complement",
        "stage1_sha256": universe.stage1_sha256, "stage2_sha256": universe.stage2_sha256,
        "anchor_sha256": universe.anchor_sha256, "contexts_zip_sha256": sha256_file(contexts_zip),
        "qwen_model": QWEN_MODEL_ID, "qwen_checkpoint_fingerprint": dict(model_fingerprint),
        "variant": variant, "pool_size": POOL_SIZES[variant], "max_tokens": int(max_tokens),
        "instruction": INSTRUCTION, "candidate_prefix_is_frozen_c35": True,
        "holdout_opened": False, "public_labels_used": False,
    }


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]], *, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            sink.write("\n")


def read_score_cache(paths: Sequence[Path], *, expected_lineage_sha256: str | None = None) -> dict[str, dict[str, dict[str, Any]]]:
    scores: dict[str, dict[str, dict[str, Any]]] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Qwen cache shard is missing: {path}")
        with path.open(encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise V27ContractError(f"{path}:{number}: Qwen cache row is not an object")
                qid, doc_id = str(row.get("qid", "")), str(row.get("doc_id", ""))
                if not qid or not doc_id or qid in scores and doc_id in scores[qid]:
                    raise V27ContractError(f"{path}:{number}: duplicate/invalid Qwen cache key")
                if expected_lineage_sha256 and row.get("lineage_sha256") != expected_lineage_sha256:
                    raise V27ContractError(f"{path}:{number}: Qwen cache lineage drift")
                value = row.get("yes_no_logit")
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise V27ContractError(f"{path}:{number}: Qwen cache contains no finite yes/no logit")
                scores.setdefault(qid, {})[doc_id] = dict(row)
    return scores


def validate_cache_coverage(
    universe: CandidateUniverse,
    scores: Mapping[str, Mapping[str, Any]],
    *,
    variant: str,
) -> None:
    for qid in universe.split.qids:
        expected = set(pool_for(universe, str(qid), variant))
        actual = set(scores.get(str(qid), {}))
        if expected != actual:
            raise V27ContractError(f"{qid}: Qwen cache coverage differs; missing={sorted(expected - actual)[:3]} extra={sorted(actual - expected)[:3]}")


def score_order(scores: Mapping[str, Mapping[str, Any]], qid: str, pool: Sequence[str]) -> list[str]:
    return sorted(
        (str(doc_id) for doc_id in pool),
        key=lambda doc_id: (-float(scores[str(qid)][doc_id]["yes_no_logit"]), list(pool).index(doc_id), doc_id),
    )


def oracle_slot5(split: Any, pools: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    if split.references is None:
        raise V27ContractError("Qwen oracle diagnostic requires complement qrels")
    rows: list[float] = []; absent = 0
    for index, qid in enumerate(split.qids):
        gold = set(split.references[index]); anchor = list(split.anchor[index]); pool = list(pools[str(qid)])
        candidates = [doc_id for doc_id in pool if doc_id not in set(anchor[:4])]
        best = max((float(doc_id in gold) - float(anchor[4] in gold)) / len(gold) for doc_id in candidates)
        before = len(set(anchor) & gold) / len(gold)
        rows.append(before + best); absent += int(not (set(pool) & gold))
    return {"slot5_oracle_recall": float(np.mean(rows)), "queries_without_gold_in_pool": absent, "queries": len(rows)}
