"""V14 Phase-B late-interaction rescue around the frozen v11h anchor.

This runner is deliberately narrower than a new retriever.  It never searches
outside the frozen Stage-1 c35 list and it never globally reranks v11h.  Jina
ColBERT v2 only supplies late-interaction features for the Phase-A replacement
proposals (Stage-1 ranks 6--35) and the frozen answer in slot five.  At most one
slot-five replacement can be made.

Selection and threshold calibration use the complement split only.  Standard,
law-family OOD, query-template OOD and worst-fold gates must all pass before the
locked holdout is encoded.  Public data is touched only after a promoted locked
holdout result.

The neural parameter budget is checked end to end:

* frozen v11h embedding model: 330M;
* frozen v11h cross encoder: 760M;
* Jina ColBERT v2: counted from the loaded model (conservative fallback 600M);
* the tiny replacement selector: counted exactly after fitting.

The current frozen deployment is 1.649497216B before any trainable selector;
the hard assertion enforces <4B.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np

try:
    from .legal_structure import iter_legal_chunks
    from .v14_replacement_pipeline import (
        MODEL_CONFIGS,
        PairDataset,
        RawDocument,
        _load_split,
        _sample_weights,
        evaluate_policy,
        fold_diagnostics,
        grouped_oof,
        law_family_groups,
        load_document_titles,
        load_raw_documents,
        model_parameter_count,
        predictions_from_probabilities,
        query_template_signature,
        select_threshold,
        standard_leak_groups,
        template_groups,
        write_json,
    )
    from .v14_safe_rescue import parse_constraints, validate_model_budget
except ImportError:  # pragma: no cover - direct script execution
    from legal_structure import iter_legal_chunks
    from v14_replacement_pipeline import (
        MODEL_CONFIGS,
        PairDataset,
        RawDocument,
        _load_split,
        _sample_weights,
        evaluate_policy,
        fold_diagnostics,
        grouped_oof,
        law_family_groups,
        load_document_titles,
        load_raw_documents,
        model_parameter_count,
        predictions_from_probabilities,
        query_template_signature,
        select_threshold,
        standard_leak_groups,
        template_groups,
        write_json,
    )
    from v14_safe_rescue import parse_constraints, validate_model_budget


SEED = 20260828
MODEL_ID = "jinaai/jina-colbert-v2"
MODEL_REVISION = "a9dc5cd7293d4c71dbbba04829923ba4d0e4f6ea"
BASE_CODE_ID = "jinaai/xlm-roberta-flash-implementation"
BASE_CODE_REVISION = "bd55a5ec8e6c0fb1d6c26efb4b6a4a74ce8a88d3"
MODEL_LICENSE = "CC-BY-NC-4.0"
CONSERVATIVE_COLBERT_PARAMETERS = 600_000_000
EMBEDDING_DIMENSION = 64

# Fixed parser keys. Missing elements receive zero and their presence is already
# represented in the reused Phase-A query features.
ELEMENT_KEYS = (
    "actor",
    "action",
    "object",
    "condition",
    "exception",
    "temporal",
    "modality",
    "number",
    "article",
    "clause",
    "point",
    "topic",
)

COLBERT_DOCUMENT_FEATURES = (
    "full_best_single",
    "full_top2_mean",
    "full_top3_mean",
    "full_top6_mean",
    "full_top6_std",
    "full_margin_1_2",
    "full_best_combo2",
    "full_best_combo3",
    "full_combo2_gain",
    "full_combo3_gain",
    "title_score",
    "element_mean",
    "element_min",
    "element_max",
    "element_std",
    "critical_mean",
    "critical_min",
    "critical_max",
    *(f"element_{key}" for key in ELEMENT_KEYS),
    "passage_count_log",
)

# Query-relative transforms remove the raw MaxSim scale caused by query type
# and length. These fields were fixed before reading Phase-B labels/results.
QUERY_NORMALIZED_FIELDS = (
    "full_best_single",
    "full_best_combo3",
    "title_score",
    "element_mean",
    "element_min",
    "critical_mean",
    "critical_min",
    "element_action",
    "element_object",
    "element_condition",
    "element_exception",
    "element_number",
    "element_topic",
)

GATE_MODES = (
    "phase_a_strict",
    "phase_a_or_colbert_dominance",
    "colbert_ce_rank25",
    "calibrated_all",
)

# This is a frozen, label-free guard selected on complement only after the
# first ColBERT pass: the candidate must beat the v11h tail on the existing
# cross-encoder margin and remain inside Stage-1 ranks 6--25.  It preserves
# the one-slot rescue policy while removing the noisy tail of rank 26--35.
COLBERT_CE_MARGIN_MIN = 0.0
COLBERT_STAGE1_MAX_RANK = 25

# Phase-A pointwise models remain in the comparison. These two groupwise
# LambdaMART variants are the next roadmap step after the ColBERT signal: a
# query's candidates are trained/ranked as one group, while the target remains
# the metric-aligned one-slot rescue label.
PHASE_B_MODEL_CONFIGS = (
    *MODEL_CONFIGS,
    {"kind": "lightgbm_ranker", "num_leaves": 7, "max_depth": 3, "n_estimators": 160},
    {"kind": "lightgbm_ranker", "num_leaves": 15, "max_depth": 4, "n_estimators": 220},
)


@dataclass(frozen=True)
class QuerySpec:
    full_index: int
    element_indices: Mapping[str, int]
    critical: tuple[str, ...]


@dataclass
class RaggedEmbeddingBank:
    texts: tuple[str, ...]
    flat: np.ndarray
    offsets: np.ndarray

    def get(self, index: int) -> np.ndarray:
        start, stop = int(self.offsets[index]), int(self.offsets[index + 1])
        return np.asarray(self.flat[start:stop], dtype=np.float16)


@dataclass
class LateInteractionCache:
    question_index: np.ndarray
    document_id: np.ndarray
    features: np.ndarray
    feature_names: tuple[str, ...]
    metadata: dict[str, Any]


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _source_stat(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".temporary.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def load_phase_a_pair_cache(path: Path) -> PairDataset:
    """Load an immutable Phase-A cache without recomputing its mtime signature."""

    with np.load(path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata"].item()))
        dataset = PairDataset(
            split_name=str(metadata["split_name"]),
            X=np.asarray(cache["X"], dtype=np.float32),
            target=np.asarray(cache["target"], dtype=np.int8),
            outcome=np.asarray(cache["outcome"]).astype(str),
            question_index=np.asarray(cache["question_index"], dtype=np.int32),
            candidate_id=np.asarray(cache["candidate_id"]).astype(str),
            safe_gate=np.asarray(cache["safe_gate"], dtype=np.bool_),
            feature_names=tuple(metadata["feature_names"]),
            prefilter=dict(metadata["prefilter"]),
        )
    if dataset.X.shape[1] != len(dataset.feature_names):
        raise ValueError("Phase-A pair cache feature schema is inconsistent")
    if int(dataset.prefilter.get("maximum_candidates", -1)) != 30:
        raise ValueError("Phase B requires the max-30 c35 complement cache")
    return dataset


def _query_text_plan(
    questions: Sequence[str],
) -> tuple[tuple[str, ...], tuple[QuerySpec, ...]]:
    texts: list[str] = []
    text_index: dict[str, int] = {}

    def register(value: str) -> int:
        value = " ".join(value.split())
        if value not in text_index:
            text_index[value] = len(texts)
            texts.append(value)
        return text_index[value]

    specs: list[QuerySpec] = []
    for question in questions:
        constraints = parse_constraints(question)
        full_index = register(question)
        elements = {
            key: register(value)
            for key, value in constraints.non_empty()
            if key in ELEMENT_KEYS and value
        }
        specs.append(QuerySpec(full_index, elements, constraints.critical))
    return tuple(texts), tuple(specs)


def _query_cache_signature(questions: Sequence[str]) -> str:
    return _json_hash(
        {
            "schema": 2,
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "base_code": BASE_CODE_REVISION,
            "dimension": EMBEDDING_DIMENSION,
            "questions": list(questions),
            "element_keys": ELEMENT_KEYS,
        }
    )


def _load_query_bank(path: Path, signature: str) -> RaggedEmbeddingBank | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata"].item()))
        if metadata.get("signature") != signature:
            return None
        return RaggedEmbeddingBank(
            tuple(np.asarray(cache["texts"]).astype(str)),
            np.asarray(cache["flat"], dtype=np.float16),
            np.asarray(cache["offsets"], dtype=np.int64),
        )


def _write_query_bank(
    path: Path,
    signature: str,
    bank: RaggedEmbeddingBank,
    device_trace: Mapping[str, Any],
) -> None:
    metadata = {
        "signature": signature,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "base_code_id": BASE_CODE_ID,
        "base_code_revision": BASE_CODE_REVISION,
        "dimension": EMBEDDING_DIMENSION,
        "dtype": "float16",
        "device_trace": dict(device_trace),
    }
    _atomic_npz(
        path,
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        texts=np.asarray(bank.texts),
        flat=bank.flat,
        offsets=bank.offsets,
    )


class ColBERTEncoder:
    """CUDA-first encoder with a non-destructive CPU fallback on CUDA OOM."""

    def __init__(self, requested_device: str) -> None:
        import torch
        from sentence_transformers import MultiVectorEncoder

        self.torch = torch
        self.requested_device = requested_device
        self.events: list[dict[str, Any]] = []
        if requested_device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device must be auto, cuda or cpu")
        preferred = "cuda" if requested_device != "cpu" and torch.cuda.is_available() else "cpu"
        kwargs = {
            "device": preferred,
            "trust_remote_code": True,
            "revision": MODEL_REVISION,
            "config_kwargs": {"code_revision": BASE_CODE_REVISION},
            "model_kwargs": {
                "dtype": torch.float16 if preferred == "cuda" else torch.float32,
                "attn_implementation": "eager",
            },
        }
        started = time.perf_counter()
        try:
            self.model = MultiVectorEncoder(MODEL_ID, **kwargs)
            self.device = preferred
        except RuntimeError as error:
            if preferred != "cuda" or not self.is_oom(error):
                raise
            torch.cuda.empty_cache()
            self.events.append(
                {
                    "event": "cuda_load_oom",
                    "action": "cpu_fallback_without_stopping_other_gpu_jobs",
                    "error": str(error)[:500],
                }
            )
            kwargs["device"] = "cpu"
            kwargs["model_kwargs"] = {
                "dtype": torch.float32,
                "attn_implementation": "eager",
            }
            self.model = MultiVectorEncoder(MODEL_ID, **kwargs)
            self.device = "cpu"
        self.parameter_count = int(sum(parameter.numel() for parameter in self.model.parameters()))
        self.events.append(
            {
                "event": "model_loaded",
                "device": self.device,
                "seconds": time.perf_counter() - started,
                "parameters": self.parameter_count,
                "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "cuda_free_total": list(torch.cuda.mem_get_info()) if self.device == "cuda" else None,
            }
        )

    @staticmethod
    def is_oom(error: BaseException) -> bool:
        return "out of memory" in str(error).lower()

    def switch_to_cpu(self, reason: str) -> None:
        if self.device == "cpu":
            return
        self.model.to("cpu")
        self.torch.cuda.empty_cache()
        self.device = "cpu"
        self.events.append(
            {
                "event": "cuda_runtime_oom",
                "action": "cpu_fallback_without_stopping_other_gpu_jobs",
                "reason": reason[:500],
            }
        )

    def _normalize_outputs(self, outputs: Any) -> list[np.ndarray]:
        torch = self.torch
        if isinstance(outputs, torch.Tensor):
            outputs = [item for item in outputs]
        rows: list[np.ndarray] = []
        for output in outputs:
            tensor = output[:, :EMBEDDING_DIMENSION]
            tensor = torch.nn.functional.normalize(tensor.float(), p=2, dim=-1)
            rows.append(tensor.detach().cpu().numpy().astype(np.float16, copy=False))
        return rows

    def encode_queries(self, texts: Sequence[str], batch_size: int) -> list[np.ndarray]:
        outputs = self.model.encode_query(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=False,
            device=self.device,
        )
        return self._normalize_outputs(outputs)

    def encode_documents(self, texts: Sequence[str], batch_size: int) -> list[np.ndarray]:
        outputs = self.model.encode_document(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=False,
            device=self.device,
        )
        return self._normalize_outputs(outputs)


def build_query_bank(
    path: Path,
    questions: Sequence[str],
    encoder: ColBERTEncoder,
    batch_size: int,
    checkpoint_texts: int = 256,
) -> tuple[RaggedEmbeddingBank, tuple[QuerySpec, ...]]:
    texts, specs = _query_text_plan(questions)
    signature = _query_cache_signature(questions)
    cached = _load_query_bank(path, signature)
    if cached is not None:
        if cached.texts != texts:
            raise RuntimeError("query bank text plan does not match its signature")
        print(f"ColBERT query cache hit: {path}", flush=True)
        return cached, specs

    rows: list[np.ndarray] = []
    for start in range(0, len(texts), checkpoint_texts):
        stop = min(start + checkpoint_texts, len(texts))
        try:
            rows.extend(encoder.encode_queries(texts[start:stop], batch_size))
        except RuntimeError as error:
            if encoder.device != "cuda" or not encoder.is_oom(error):
                raise
            encoder.torch.cuda.empty_cache()
            try:
                rows.extend(encoder.encode_queries(texts[start:stop], 1))
            except RuntimeError as retry_error:
                if not encoder.is_oom(retry_error):
                    raise
                encoder.switch_to_cpu(str(retry_error))
                rows.extend(encoder.encode_queries(texts[start:stop], 1))
        print(f"ColBERT query embeddings: {stop}/{len(texts)}", flush=True)
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    for index, row in enumerate(rows):
        offsets[index + 1] = offsets[index] + len(row)
    flat = np.concatenate(rows, axis=0) if rows else np.zeros((0, EMBEDDING_DIMENSION), dtype=np.float16)
    bank = RaggedEmbeddingBank(texts, flat, offsets)
    _write_query_bank(path, signature, bank, {"events": encoder.events})
    print(f"ColBERT query cache written: {path}", flush=True)
    return bank, specs


def _top_mean(values: Any, count: int) -> float:
    if int(values.numel()) == 0:
        return 0.0
    count = min(count, int(values.numel()))
    return float(values.topk(count).values.float().mean().item())


def late_interaction_feature_vector(
    full_passage_token_maxima: Sequence[Any],
    title_full_token_maxima: Any,
    element_passage_scores: Sequence[Mapping[str, float]],
    title_element_scores: Mapping[str, float],
    present_elements: Sequence[str],
    critical_elements: Sequence[str],
) -> np.ndarray:
    """Reduce MaxSim evidence to fixed, label-free document features.

    For the six strongest legal passages this computes every union of one,
    two and three passages. Union MaxSim takes the best document token for each
    query token across the selected passages, matching ColBERT late interaction.
    """

    import torch

    if not full_passage_token_maxima:
        full_passage_token_maxima = [torch.zeros_like(title_full_token_maxima)]
        element_passage_scores = [{}]
    token_rows = torch.stack([row.float() for row in full_passage_token_maxima])
    single_scores = token_rows.mean(dim=1)
    order = torch.argsort(single_scores, descending=True)
    shortlist = order[: min(6, len(order))].tolist()
    best_single = float(single_scores[order[0]].item())
    second = float(single_scores[order[1]].item()) if len(order) > 1 else best_single

    best_by_size = {1: best_single, 2: best_single, 3: best_single}
    for size in (2, 3):
        if len(shortlist) < size:
            continue
        best = -math.inf
        for combo in itertools.combinations(shortlist, size):
            union = torch.stack([token_rows[index] for index in combo]).max(dim=0).values
            best = max(best, float(union.mean().item()))
        best_by_size[size] = best

    element_best: dict[str, float] = {}
    for key in present_elements:
        values = [float(row.get(key, 0.0)) for row in element_passage_scores]
        values.append(float(title_element_scores.get(key, 0.0)))
        element_best[key] = max(values, default=0.0)
    present_values = list(element_best.values())
    critical_values = [element_best[key] for key in critical_elements if key in element_best]
    if not critical_values:
        critical_values = [element_best.get("topic", 0.0)]

    top6 = single_scores[order[: min(6, len(order))]].float()
    row = [
        best_single,
        _top_mean(single_scores, 2),
        _top_mean(single_scores, 3),
        float(top6.mean().item()),
        float(top6.std(unbiased=False).item()) if len(top6) > 1 else 0.0,
        best_single - second,
        best_by_size[2],
        best_by_size[3],
        best_by_size[2] - best_single,
        best_by_size[3] - best_single,
        float(title_full_token_maxima.float().mean().item()),
        statistics.fmean(present_values) if present_values else 0.0,
        min(present_values) if present_values else 0.0,
        max(present_values) if present_values else 0.0,
        statistics.pstdev(present_values) if len(present_values) > 1 else 0.0,
        statistics.fmean(critical_values),
        min(critical_values),
        max(critical_values),
        *(element_best.get(key, 0.0) for key in ELEMENT_KEYS),
        math.log1p(len(full_passage_token_maxima)),
    ]
    if len(row) != len(COLBERT_DOCUMENT_FEATURES):
        raise RuntimeError("late-interaction document feature schema mismatch")
    return np.asarray(row, dtype=np.float32)


def _document_inputs(raw: RawDocument) -> tuple[str, list[str]]:
    title = raw.title.strip() or raw.doc_id
    chunks = [chunk.text for chunk in iter_legal_chunks(raw.text, 1800, 180)]
    if not chunks:
        chunks = [raw.text]
    title_input = f"Tiêu đề văn bản pháp luật: {title}"
    passage_inputs = [f"Tiêu đề: {title}\nNội dung: {text}" for text in chunks]
    return title_input, passage_inputs


def _query_matrix(
    bank: RaggedEmbeddingBank,
    spec: QuerySpec,
) -> tuple[np.ndarray, dict[str, slice]]:
    components: list[np.ndarray] = [bank.get(spec.full_index)]
    spans: dict[str, slice] = {}
    offset = len(components[0])
    for key in ELEMENT_KEYS:
        if key not in spec.element_indices:
            continue
        row = bank.get(spec.element_indices[key])
        spans[key] = slice(offset, offset + len(row))
        components.append(row)
        offset += len(row)
    return np.concatenate(components, axis=0), spans


def score_document_queries(
    encoder: ColBERTEncoder,
    bank: RaggedEmbeddingBank,
    specs: Sequence[QuerySpec],
    raw: RawDocument,
    questions: Sequence[int],
    document_batch_size: int,
    score_query_batch_size: int,
) -> dict[int, np.ndarray]:
    """Encode one document once, then score every query that references it."""

    torch = encoder.torch
    title_text, passage_texts = _document_inputs(raw)
    encoded = encoder.encode_documents([title_text, *passage_texts], document_batch_size)
    score_dtype = torch.float16 if encoder.device == "cuda" else torch.float32
    title_embedding = torch.as_tensor(encoded[0], device=encoder.device, dtype=score_dtype)
    passage_embeddings = [
        torch.as_tensor(row, device=encoder.device, dtype=score_dtype)
        for row in encoded[1:]
    ]
    results: dict[int, np.ndarray] = {}
    for start in range(0, len(questions), score_query_batch_size):
        batch_questions = list(questions[start : start + score_query_batch_size])
        matrices: list[np.ndarray] = []
        batch_spans: list[dict[str, slice]] = []
        batch_offsets: list[int] = []
        offset = 0
        for question in batch_questions:
            matrix, spans = _query_matrix(bank, specs[question])
            matrices.append(matrix)
            batch_offsets.append(offset)
            batch_spans.append(spans)
            offset += len(matrix)
        query_matrix = torch.as_tensor(
            np.concatenate(matrices, axis=0), device=encoder.device, dtype=score_dtype
        )
        title_max = torch.matmul(query_matrix, title_embedding.T).max(dim=1).values
        passage_max = [
            torch.matmul(query_matrix, document.T).max(dim=1).values
            for document in passage_embeddings
        ]
        for local, question in enumerate(batch_questions):
            spec = specs[question]
            local_start = batch_offsets[local]
            full_length = len(bank.get(spec.full_index))
            full_slice = slice(local_start, local_start + full_length)
            element_spans = {
                key: slice(local_start + value.start, local_start + value.stop)
                for key, value in batch_spans[local].items()
            }
            full_rows = [row[full_slice].detach().cpu() for row in passage_max]
            title_full = title_max[full_slice].detach().cpu()
            element_rows = [
                {
                    key: float(row[value].float().mean().item())
                    for key, value in element_spans.items()
                }
                for row in passage_max
            ]
            title_elements = {
                key: float(title_max[value].float().mean().item())
                for key, value in element_spans.items()
            }
            results[question] = late_interaction_feature_vector(
                full_rows,
                title_full,
                element_rows,
                title_elements,
                tuple(element_spans),
                spec.critical,
            )
        del query_matrix, title_max, passage_max
    del title_embedding, passage_embeddings, encoded
    return results


def _score_pairs(
    split: Any,
    dataset: PairDataset,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[int]]]:
    keys = {
        (int(question), str(doc_id))
        for question, doc_id in zip(dataset.question_index, dataset.candidate_id)
    }
    for question, qid in enumerate(split.qids):
        keys.add((question, str(split.anchor[qid]["answer"][-1])))
    ordered = sorted(keys, key=lambda item: (item[1], item[0]))
    question_index = np.asarray([item[0] for item in ordered], dtype=np.int32)
    document_id = np.asarray([item[1] for item in ordered])
    uses: defaultdict[str, list[int]] = defaultdict(list)
    for question, doc_id in ordered:
        uses[doc_id].append(question)
    return question_index, document_id, dict(uses)


def _score_cache_signature(
    split: Any,
    dataset: PairDataset,
    pair_cache_path: Path,
    contexts_zip: Path,
) -> str:
    return _json_hash(
        {
            "schema": 4,
            "split": split.name,
            "qids": split.qids,
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "base_code_revision": BASE_CODE_REVISION,
            "dimension": EMBEDDING_DIMENSION,
            "features": COLBERT_DOCUMENT_FEATURES,
            "pair_cache": _source_stat(pair_cache_path),
            "contexts": _source_stat(contexts_zip),
            "rows": len(dataset.X),
        }
    )


def _load_score_cache(
    path: Path,
    signature: str,
    expected_questions: np.ndarray,
    expected_documents: np.ndarray,
) -> LateInteractionCache | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata"].item()))
        if metadata.get("signature") != signature:
            return None
        questions = np.asarray(cache["question_index"], dtype=np.int32)
        documents = np.asarray(cache["document_id"]).astype(str)
        if not np.array_equal(questions, expected_questions) or not np.array_equal(
            documents, expected_documents
        ):
            return None
        return LateInteractionCache(
            questions,
            documents,
            np.asarray(cache["features"], dtype=np.float32),
            tuple(metadata["feature_names"]),
            metadata,
        )


def _write_score_cache(path: Path, cache: LateInteractionCache) -> None:
    _atomic_npz(
        path,
        metadata=np.asarray(json.dumps(cache.metadata, ensure_ascii=False)),
        question_index=cache.question_index,
        document_id=cache.document_id,
        features=cache.features,
    )


def build_late_interaction_cache(
    split: Any,
    dataset: PairDataset,
    pair_cache_path: Path,
    contexts_zip: Path,
    output_path: Path,
    query_cache_path: Path,
    requested_device: str,
    query_batch_size: int,
    document_batch_size: int,
    score_query_batch_size: int,
    checkpoint_every: int,
    max_documents_this_run: int | None,
) -> tuple[LateInteractionCache, bool]:
    question_index, document_id, uses = _score_pairs(split, dataset)
    signature = _score_cache_signature(split, dataset, pair_cache_path, contexts_zip)
    cached = _load_score_cache(output_path, signature, question_index, document_id)
    if cached is None:
        metadata = {
            "signature": signature,
            "split": split.name,
            "feature_names": list(COLBERT_DOCUMENT_FEATURES),
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "base_code_id": BASE_CODE_ID,
            "base_code_revision": BASE_CODE_REVISION,
            "license": MODEL_LICENSE,
            "dimension": EMBEDDING_DIMENSION,
            "dtype": "float16 embeddings / float32 features",
            "chunking": "legal_structure.iter_legal_chunks(1800 chars, 180 overlap)",
            "evidence": "all chunks -> top6 by full-query MaxSim -> exhaustive unions of size 1,2,3",
            "device_events": [],
            "model_parameters": CONSERVATIVE_COLBERT_PARAMETERS,
        }
        cached = LateInteractionCache(
            question_index,
            document_id,
            np.full((len(question_index), len(COLBERT_DOCUMENT_FEATURES)), np.nan, dtype=np.float32),
            COLBERT_DOCUMENT_FEATURES,
            metadata,
        )
    elif cached.feature_names != COLBERT_DOCUMENT_FEATURES:
        raise RuntimeError("late-interaction cache feature schema differs")

    key_to_row = {
        (int(question), str(doc_id)): row
        for row, (question, doc_id) in enumerate(zip(cached.question_index, cached.document_id))
    }
    incomplete_documents = [
        doc_id
        for doc_id, questions in sorted(uses.items())
        if any(np.isnan(cached.features[key_to_row[(question, doc_id)]]).any() for question in questions)
    ]
    if not incomplete_documents:
        print(f"ColBERT score cache complete: {output_path}", flush=True)
        return cached, True

    encoder = ColBERTEncoder(requested_device)
    cached.metadata["model_parameters"] = encoder.parameter_count
    cached.metadata["device_events"].extend(encoder.events)
    bank, specs = build_query_bank(
        query_cache_path, split.questions, encoder, query_batch_size
    )
    selected_documents = (
        incomplete_documents[:max_documents_this_run]
        if max_documents_this_run is not None
        else incomplete_documents
    )
    raw_documents = load_raw_documents(contexts_zip, set(selected_documents))
    processed = 0
    active_document_batch = document_batch_size
    active_score_batch = score_query_batch_size
    for doc_id in selected_documents:
        questions = sorted(set(uses[doc_id]))
        try:
            values = score_document_queries(
                encoder,
                bank,
                specs,
                raw_documents[doc_id],
                questions,
                active_document_batch,
                active_score_batch,
            )
        except RuntimeError as error:
            if encoder.device != "cuda" or not encoder.is_oom(error):
                raise
            encoder.torch.cuda.empty_cache()
            if active_document_batch > 1 or active_score_batch > 1:
                active_document_batch = 1
                active_score_batch = 1
                encoder.events.append(
                    {
                        "event": "cuda_scoring_oom",
                        "action": "retry_cuda_with_document_and_score_query_batch_size_1",
                        "document": doc_id,
                    }
                )
                try:
                    values = score_document_queries(
                        encoder,
                        bank,
                        specs,
                        raw_documents[doc_id],
                        questions,
                        1,
                        1,
                    )
                except RuntimeError as retry_error:
                    if not encoder.is_oom(retry_error):
                        raise
                    encoder.switch_to_cpu(str(retry_error))
                    values = score_document_queries(
                        encoder,
                        bank,
                        specs,
                        raw_documents[doc_id],
                        questions,
                        1,
                        1,
                    )
            else:
                encoder.switch_to_cpu(str(error))
                values = score_document_queries(
                    encoder,
                    bank,
                    specs,
                    raw_documents[doc_id],
                    questions,
                    1,
                    1,
                )
        for question, row in values.items():
            cached.features[key_to_row[(question, doc_id)]] = row
        processed += 1
        if processed % checkpoint_every == 0 or processed == len(selected_documents):
            cached.metadata["device_events"] = encoder.events
            cached.metadata["completed_pairs"] = int(np.all(np.isfinite(cached.features), axis=1).sum())
            cached.metadata["total_pairs"] = len(cached.features)
            cached.metadata["last_checkpoint_utc"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            )
            _write_score_cache(output_path, cached)
            complete_docs = len(uses) - len(incomplete_documents) + processed
            print(
                f"ColBERT {split.name}: documents {complete_docs}/{len(uses)}; "
                f"device={encoder.device}; checkpoint={output_path}",
                flush=True,
            )
    complete = bool(np.all(np.isfinite(cached.features)))
    return cached, complete


def _descending_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    result = np.empty(len(values), dtype=np.float32)
    result[order] = np.arange(1, len(values) + 1, dtype=np.float32)
    return result


def augment_pair_dataset(
    split: Any,
    base: PairDataset,
    late: LateInteractionCache,
) -> tuple[PairDataset, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Join candidate/tail/delta MaxSim and query-normalized fields."""

    key_to_row = {
        (int(question), str(doc_id)): row
        for row, (question, doc_id) in enumerate(zip(late.question_index, late.document_id))
    }
    candidate_rows = np.asarray(
        [key_to_row[(int(question), str(doc_id))] for question, doc_id in zip(base.question_index, base.candidate_id)],
        dtype=np.int64,
    )
    tail_rows = np.asarray(
        [
            key_to_row[(int(question), str(split.anchor[split.qids[int(question)]]["answer"][-1]))]
            for question in base.question_index
        ],
        dtype=np.int64,
    )
    candidate = late.features[candidate_rows]
    tail = late.features[tail_rows]
    delta = candidate - tail
    if not np.all(np.isfinite(candidate)) or not np.all(np.isfinite(tail)):
        raise RuntimeError("cannot augment an incomplete late-interaction cache")

    normalized_rows: list[np.ndarray] = []
    normalized_names: list[str] = []
    columns = {name: index for index, name in enumerate(COLBERT_DOCUMENT_FEATURES)}
    for field in QUERY_NORMALIZED_FIELDS:
        values = candidate[:, columns[field]]
        inv_rank = np.zeros(len(values), dtype=np.float32)
        zscore = np.zeros(len(values), dtype=np.float32)
        minmax = np.zeros(len(values), dtype=np.float32)
        for question in range(len(split.qids)):
            rows = np.flatnonzero(base.question_index == question)
            if not len(rows):
                continue
            local = values[rows].astype(np.float64)
            ranks = _descending_rank(local)
            inv_rank[rows] = 1.0 / ranks
            mean, std = float(local.mean()), float(local.std())
            zscore[rows] = ((local - mean) / std).astype(np.float32) if std > 1e-9 else 0.0
            low, high = float(local.min()), float(local.max())
            minmax[rows] = ((local - low) / (high - low)).astype(np.float32) if high > low else 0.0
        normalized_rows.extend((inv_rank, zscore, minmax))
        normalized_names.extend(
            (
                f"candidate__colbert_query_inv_rank__{field}",
                f"candidate__colbert_query_zscore__{field}",
                f"candidate__colbert_query_minmax__{field}",
            )
        )
    normalized = np.column_stack(normalized_rows).astype(np.float32)
    matrix = np.column_stack(
        (
            base.X,
            base.safe_gate.astype(np.float32),
            candidate,
            tail,
            delta,
            normalized,
        )
    ).astype(np.float32)
    names = (
        *base.feature_names,
        "phase_a_safe_gate",
        *(f"candidate__colbert__{name}" for name in COLBERT_DOCUMENT_FEATURES),
        *(f"tail__colbert__{name}" for name in COLBERT_DOCUMENT_FEATURES),
        *(f"delta__colbert__{name}" for name in COLBERT_DOCUMENT_FEATURES),
        *normalized_names,
    )
    augmented = PairDataset(
        base.split_name,
        matrix,
        base.target.copy(),
        base.outcome.copy(),
        base.question_index.copy(),
        base.candidate_id.copy(),
        np.ones(len(base.X), dtype=np.bool_),
        tuple(names),
        dict(base.prefilter),
    )
    feature_arrays = {
        "candidate": candidate,
        "tail": tail,
        "delta": delta,
        "normalized": normalized,
    }
    direct = direct_score_arrays(candidate, tail, normalized, columns)
    return augmented, feature_arrays, direct


def gate_masks(
    base: PairDataset,
    feature_arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    columns = {name: index for index, name in enumerate(COLBERT_DOCUMENT_FEATURES)}
    delta = feature_arrays["delta"]
    full = delta[:, columns["full_best_combo3"]]
    critical = delta[:, columns["critical_mean"]]
    critical_min = delta[:, columns["critical_min"]]
    dominance = ((full >= 0.0) & (critical >= 0.0)) | (
        (full >= -0.005) & (critical_min >= 0.015)
    )
    base_columns = {name: index for index, name in enumerate(base.feature_names)}
    ce_margin = base.X[:, base_columns["delta__cross_encoder_score"]]
    stage1_rank = base.X[:, base_columns["candidate__stage1_rank"]]
    ce_rank25 = (
        (ce_margin >= COLBERT_CE_MARGIN_MIN)
        & (stage1_rank <= COLBERT_STAGE1_MAX_RANK)
    )
    return {
        "phase_a_strict": base.safe_gate.copy(),
        "phase_a_or_colbert_dominance": base.safe_gate | dominance,
        "colbert_ce_rank25": ce_rank25,
        "calibrated_all": np.ones(len(base.X), dtype=np.bool_),
    }


def make_ranker(config: Mapping[str, Any], seed: int) -> Any:
    from lightgbm import LGBMRanker

    return LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=[0.0, 1.0],
        n_estimators=int(config["n_estimators"]),
        num_leaves=int(config["num_leaves"]),
        max_depth=int(config["max_depth"]),
        learning_rate=0.03,
        min_child_samples=35,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=seed,
        n_jobs=-1,
        deterministic=True,
        verbosity=-1,
    )


def grouped_oof_ranker(
    dataset: PairDataset,
    groups: np.ndarray,
    config: Mapping[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    """OOF LambdaMART with one ranking group per query."""

    from sklearn.model_selection import StratifiedGroupKFold

    question_count = int(dataset.question_index.max()) + 1
    question_indices = np.arange(question_count, dtype=np.int32)
    question_target = np.asarray(
        [int(np.any(dataset.target[dataset.question_index == question])) for question in question_indices],
        dtype=np.int8,
    )
    unique_groups = len(set(map(int, groups)))
    positive_questions = int(question_target.sum())
    folds = min(5, unique_groups, positive_questions)
    if folds < 3:
        raise RuntimeError("insufficient grouped positives for LambdaMART OOF")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    predictions = np.full(len(dataset.X), np.nan, dtype=np.float64)
    fold_by_question = np.full(question_count, -1, dtype=np.int8)
    models: list[Any] = []
    for fold, (train_local, valid_local) in enumerate(
        splitter.split(question_indices, question_target, groups=groups)
    ):
        train_questions = question_indices[train_local]
        valid_questions = question_indices[valid_local]
        train_mask = np.isin(dataset.question_index, train_questions)
        train_rows = np.flatnonzero(train_mask)
        train_group_sizes = [
            int(np.sum(dataset.question_index[train_rows] == question))
            for question in train_questions
        ]
        if int(dataset.target[train_mask].sum()) == 0:
            raise RuntimeError("LambdaMART fold has no positive rescue rows")
        model = make_ranker(config, seed + fold)
        model.fit(
            dataset.X[train_rows],
            dataset.target[train_rows],
            group=train_group_sizes,
            sample_weight=_sample_weights(dataset, train_mask),
        )
        valid_mask = np.isin(dataset.question_index, valid_questions)
        predictions[valid_mask] = np.asarray(model.predict(dataset.X[valid_mask]), dtype=np.float64)
        fold_by_question[valid_questions] = fold
        models.append(model)
        print(
            f"replacement OOF/lightgbm_ranker fold {fold + 1}/{folds}: "
            f"train_q={len(train_questions)} valid_q={len(valid_questions)}",
            flush=True,
        )
    if np.isnan(predictions).any() or np.any(fold_by_question < 0):
        raise RuntimeError("LambdaMART OOF did not cover every row/question")
    return predictions, fold_by_question, models


def phase_b_grouped_oof(
    dataset: PairDataset,
    groups: np.ndarray,
    config: Mapping[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    if config["kind"] == "lightgbm_ranker":
        return grouped_oof_ranker(dataset, groups, config, seed)
    return grouped_oof(dataset, groups, config, seed)


def phase_b_predict_model(model: Any, matrix: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)
    return np.asarray(model.predict(matrix), dtype=np.float64)


def phase_b_ensemble_predict(models: Sequence[Any], matrix: np.ndarray) -> np.ndarray:
    return np.mean(np.vstack([phase_b_predict_model(model, matrix) for model in models]), axis=0)


def direct_score_arrays(
    candidate: np.ndarray,
    tail: np.ndarray,
    normalized: np.ndarray,
    columns: Mapping[str, int],
) -> dict[str, np.ndarray]:
    delta = candidate - tail
    full = delta[:, columns["full_best_combo3"]]
    critical = delta[:, columns["critical_mean"]]
    element = delta[:, columns["element_mean"]]
    critical_min = delta[:, columns["critical_min"]]

    field_position = {name: index for index, name in enumerate(QUERY_NORMALIZED_FIELDS)}

    def norm(field: str, transform: int) -> np.ndarray:
        return normalized[:, 3 * field_position[field] + transform]

    return {
        "frozen_full_delta": full.astype(np.float64),
        "frozen_balanced_delta": (
            0.50 * full + 0.25 * critical + 0.15 * critical_min + 0.10 * element
        ).astype(np.float64),
        "frozen_query_relative": (
            0.40 * norm("full_best_combo3", 1)
            + 0.25 * norm("critical_mean", 1)
            + 0.15 * norm("critical_min", 1)
            + 0.10 * norm("element_mean", 1)
            + 0.10 * full
        ).astype(np.float64),
    }


def _dataset_with_gate(dataset: PairDataset, mask: np.ndarray) -> PairDataset:
    return replace(dataset, safe_gate=np.asarray(mask, dtype=np.bool_))


def _validation_passes(evaluation: Mapping[str, Any], minimum_delta: float) -> bool:
    intervention = evaluation["intervention"]
    return bool(
        evaluation["delta"]["recall"] >= minimum_delta
        and intervention["intervention_precision"] >= 0.85
        and intervention["destructive_rate"] <= 0.03
        and intervention["successful_rescues"] > 0
    )


def group_fold_assignments(
    dataset: PairDataset,
    groups: np.ndarray,
    seed: int,
) -> np.ndarray:
    from sklearn.model_selection import StratifiedGroupKFold

    question_count = int(dataset.question_index.max()) + 1
    question_indices = np.arange(question_count, dtype=np.int32)
    question_target = np.asarray(
        [int(np.any(dataset.target[dataset.question_index == question])) for question in question_indices],
        dtype=np.int8,
    )
    folds = min(5, len(set(map(int, groups))), int(question_target.sum()))
    if folds < 3:
        raise RuntimeError("insufficient groups for direct-score fold diagnostics")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    result = np.full(question_count, -1, dtype=np.int8)
    for fold, (_, valid) in enumerate(splitter.split(question_indices, question_target, groups=groups)):
        result[question_indices[valid]] = fold
    if np.any(result < 0):
        raise RuntimeError("direct-score fold assignment is incomplete")
    return result


def _candidate_report(
    split: Any,
    dataset: PairDataset,
    probabilities_by_validation: Mapping[str, np.ndarray],
    folds_by_validation: Mapping[str, np.ndarray],
    gate_mode: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    threshold, standard_evaluation, threshold_leaderboard = select_threshold(
        split, dataset, probabilities_by_validation["standard"], 0.003
    )
    evaluations: dict[str, Any] = {}
    for name in ("standard", "law_family_ood", "query_template_ood"):
        evaluation, predictions, _ = evaluate_policy(
            split, dataset, probabilities_by_validation[name], threshold
        )
        evaluations[name] = {
            "evaluation": evaluation,
            "folds": fold_diagnostics(split, predictions, folds_by_validation[name]),
        }
    gates = {
        "standard_delta_at_least_0_003": _validation_passes(
            evaluations["standard"]["evaluation"], 0.003
        ),
        "law_family_delta_at_least_0_002": _validation_passes(
            evaluations["law_family_ood"]["evaluation"], 0.002
        ),
        "query_template_delta_at_least_0_002": _validation_passes(
            evaluations["query_template_ood"]["evaluation"], 0.002
        ),
        "worst_fold_not_below_minus_0_002": min(
            evaluations[name]["folds"]["worst_fold_delta"]
            for name in evaluations
        )
        >= -0.002,
    }
    return {
        "config": dict(config),
        "gate_mode": gate_mode,
        "threshold": threshold,
        **evaluations,
        "gates": gates,
        "eligible": all(gates.values()),
        "threshold_leaderboard": threshold_leaderboard,
    }


def validate_complement(
    split: Any,
    base_dataset: PairDataset,
    augmented: PairDataset,
    feature_arrays: Mapping[str, np.ndarray],
    direct_scores: Mapping[str, np.ndarray],
    titles: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups = {
        "standard": standard_leak_groups(split),
        "law_family_ood": law_family_groups(split, titles),
        "query_template_ood": template_groups(split),
    }
    group_report = {
        name: {
            "groups": len(set(map(int, values))),
            "largest_group": max(Counter(map(int, values)).values()),
        }
        for name, values in groups.items()
    }
    masks = gate_masks(base_dataset, feature_arrays)
    rows: list[dict[str, Any]] = []
    runtime: dict[str, Any] = {"supervised_models": {}}

    # Frozen zero-shot policies are independent of fold training. They still
    # receive each OOD fold assignment for exact worst-fold diagnostics.
    direct_folds = {
        name: group_fold_assignments(augmented, values, SEED + 700 + index * 31)
        for index, (name, values) in enumerate(groups.items())
    }
    for score_name, probabilities in direct_scores.items():
        for gate_mode in GATE_MODES:
            dataset = _dataset_with_gate(augmented, masks[gate_mode])
            rows.append(
                _candidate_report(
                    split,
                    dataset,
                    {name: probabilities for name in groups},
                    direct_folds,
                    gate_mode,
                    {"kind": "frozen_colbert", "score": score_name},
                )
            )

    for config_index, config in enumerate(PHASE_B_MODEL_CONFIGS):
        probabilities: dict[str, np.ndarray] = {}
        folds: dict[str, np.ndarray] = {}
        standard_models: list[Any] = []
        for validation_index, (name, group_values) in enumerate(groups.items()):
            output, fold_values, models = phase_b_grouped_oof(
                augmented,
                group_values,
                config,
                SEED + config_index * 100 + validation_index * 31,
            )
            probabilities[name] = output
            folds[name] = fold_values
            if name == "standard":
                standard_models = models
        runtime["supervised_models"][config_index] = standard_models
        for gate_mode in GATE_MODES:
            dataset = _dataset_with_gate(augmented, masks[gate_mode])
            rows.append(
                _candidate_report(
                    split,
                    dataset,
                    probabilities,
                    folds,
                    gate_mode,
                    {**dict(config), "model_config_index": config_index},
                )
            )
        best_for_config = max(
            rows[-len(GATE_MODES) :],
            key=lambda row: row["standard"]["evaluation"]["delta"]["recall"],
        )
        print(
            f"Phase-B config {config_index + 1}/{len(PHASE_B_MODEL_CONFIGS)} {config}: "
            f"best_standard_delta={best_for_config['standard']['evaluation']['delta']['recall']:+.6f}",
            flush=True,
        )
    return rows, {"groups": group_report, **runtime, "gate_masks": masks}


def _winner_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    deltas = [
        row[name]["evaluation"]["delta"]["recall"]
        for name in ("standard", "law_family_ood", "query_template_ood")
    ]
    return (
        bool(row["eligible"]),
        min(deltas),
        statistics.fmean(deltas),
        row["standard"]["evaluation"]["intervention"]["intervention_precision"],
        -row["standard"]["evaluation"]["intervention"]["changes"],
    )


def parameter_budget(colbert_parameters: int, selector_parameters: int = 0) -> dict[str, Any]:
    frozen = validate_model_budget()
    total = int(frozen["total_parameters"] + colbert_parameters + selector_parameters)
    if total >= 4_000_000_000:
        raise RuntimeError(f"end-to-end parameter ceiling violated: {total:,} >= 4,000,000,000")
    return {
        **frozen,
        "jina_colbert_v2_parameters": int(colbert_parameters),
        "replacement_selector_scalar_parameters": int(selector_parameters),
        "deployed_total_parameters": total,
        "deployed_total_billions": total / 1e9,
        "strictly_under_4b": True,
        "hard_ceiling_parameters": 4_000_000_000,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts-zip", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--score-cache", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--complement-stage1", type=Path, required=True)
    parser.add_argument("--complement-stage2", type=Path, required=True)
    parser.add_argument("--complement-reference", type=Path, required=True)
    parser.add_argument("--complement-pair-cache", type=Path, required=True)
    parser.add_argument("--holdout-stage1", type=Path, required=True)
    parser.add_argument("--holdout-stage2", type=Path, required=True)
    parser.add_argument("--holdout-reference", type=Path, required=True)
    parser.add_argument("--public-stage1", type=Path, required=True)
    parser.add_argument("--public-stage2", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--document-batch-size", type=int, default=1)
    parser.add_argument("--score-query-batch-size", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--max-documents-this-run", type=int)
    parser.add_argument("--minimum-promote-recall", type=float, default=0.925)
    parser.add_argument("--apply-public", action="store_true")
    return parser.parse_args()


def _base_pair_cache_for_split(
    split: Any,
    args: argparse.Namespace,
    name: str,
) -> tuple[PairDataset, Path]:
    if name == "complement":
        path = args.complement_pair_cache
        return load_phase_a_pair_cache(path), path
    # Imported lazily so complement failure can never trigger locked feature
    # extraction. The labels are used only after all complement gates pass.
    try:
        from .v14_replacement_pipeline import build_pair_dataset
    except ImportError:  # pragma: no cover
        from v14_replacement_pipeline import build_pair_dataset
    path = args.output_dir / "cache" / f"{name}_pairs.npz"
    return (
        build_pair_dataset(split, args.contexts_zip, args.index, path, 35, 30),
        path,
    )


def _score_split(
    split: Any,
    base: PairDataset,
    pair_path: Path,
    args: argparse.Namespace,
) -> tuple[LateInteractionCache, bool]:
    return build_late_interaction_cache(
        split,
        base,
        pair_path,
        args.contexts_zip,
        args.output_dir / "cache" / f"{split.name}_colbert_scores.npz",
        args.output_dir / "cache" / f"{split.name}_colbert_queries.npz",
        args.device,
        args.query_batch_size,
        args.document_batch_size,
        args.score_query_batch_size,
        args.checkpoint_every,
        args.max_documents_this_run,
    )


def _predict_winner(
    winner: Mapping[str, Any],
    augmented: PairDataset,
    direct_scores: Mapping[str, np.ndarray],
    models: Sequence[Any],
) -> np.ndarray:
    if winner["config"]["kind"] == "frozen_colbert":
        return np.asarray(direct_scores[winner["config"]["score"]], dtype=np.float64)
    return phase_b_ensemble_predict(models, augmented.X)


def main() -> int:
    args = parse_args()
    if min(args.query_batch_size, args.document_batch_size, args.score_query_batch_size, args.checkpoint_every) < 1:
        raise ValueError("batch sizes and checkpoint interval must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    parameter_budget(CONSERVATIVE_COLBERT_PARAMETERS)

    complement = _load_split(
        "complement",
        args.complement_stage1,
        args.complement_stage2,
        args.complement_reference,
        args.score_cache,
        args.feature_cache,
        35,
    )
    base, pair_path = _base_pair_cache_for_split(complement, args, "complement")
    if len(complement.qids) != int(base.question_index.max()) + 1:
        raise RuntimeError("complement split and Phase-A pair cache do not align")
    late, complete = _score_split(complement, base, pair_path, args)
    report_path = args.output_dir / "v14_colbert_report.json"
    if not complete:
        report = {
            "kind": "legalir_v14_phase_b_colbert_late_interaction",
            "status": "checkpoint_incomplete",
            "completion": {
                "completed_pairs": int(np.all(np.isfinite(late.features), axis=1).sum()),
                "total_pairs": len(late.features),
            },
            "model_budget": parameter_budget(int(late.metadata["model_parameters"])),
            "device_events": late.metadata.get("device_events", []),
            "holdout": {"opened": False},
            "public": {"applied": False},
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(report_path, report)
        print(json.dumps({"report": str(report_path.resolve()), **report}, ensure_ascii=False, indent=2), flush=True)
        return 0

    augmented, feature_arrays, direct_scores = augment_pair_dataset(complement, base, late)
    titles = load_document_titles(args.index)
    candidates, runtime = validate_complement(
        complement, base, augmented, feature_arrays, direct_scores, titles
    )
    winner = max(candidates, key=_winner_key)
    winner_models: list[Any] = []
    if winner["config"]["kind"] != "frozen_colbert":
        winner_models = runtime["supervised_models"][winner["config"]["model_config_index"]]
    selector_parameters = sum(model_parameter_count(model) for model in winner_models)
    budget = parameter_budget(int(late.metadata["model_parameters"]), selector_parameters)

    report: dict[str, Any] = {
        "kind": "legalir_v14_phase_b_colbert_late_interaction",
        "status": "complement_gate_passed" if winner["eligible"] else "complement_gate_failed",
        "model_budget": budget,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "base_code_id": BASE_CODE_ID,
            "base_code_revision": BASE_CODE_REVISION,
            "license": MODEL_LICENSE,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "weights_are_not_reduced_by_matryoshka_dimension": True,
        },
        "device_events": late.metadata.get("device_events", []),
        "validation_groups": runtime["groups"],
        "winner": winner,
        "model_leaderboard": sorted(candidates, key=_winner_key, reverse=True),
        "architecture": {
            "frozen_candidate_pool": "v11h safe-tail Stage-1 c35",
            "candidate_ranks": [6, 35],
            "prefilter_maximum": 30,
            "replacement_slot": 5,
            "max_replacements": 1,
            "legal_chunking": "1800 chars with 180 overlap",
            "late_interaction": "full-query and typed-element token MaxSim",
            "evidence_search": "all chunks -> strongest six -> every union of one, two and three passages",
            "safety_modes": list(GATE_MODES),
            "validation": [
                "standard grouped OOF",
                "law-family OOD grouped OOF",
                "query-template OOD grouped OOF",
                "worst-fold",
            ],
        },
        "reused_logic": {
            "v11h_anchor": "Exact frozen grouped-OOF v11 selector plus longest_phrase_tokens t10 heading rescue reconstructed from v11h_scores.npz and v2_query_norm_c80.npz.",
            "candidate_generation": "Frozen v11h PRF safe-tail Stage-1 ranks 1-35; only non-anchor ranks 6-35 are replacement proposals.",
            "cross_encoder": "Frozen 760M Vietnamese cross-encoder scores from v11h Stage-2; no fine-tuning or global reranking.",
            "phase_a_features": "All 255 candidate-vs-tail features from the immutable complement_pairs.npz cache, including 56 curated v11 columns, deterministic parser and exhaustive evidence features.",
            "legal_parser": "v14_safe_rescue.parse_constraints actor/action/object/condition/exception/temporal/modality/numeric/article/clause/point/topic logic is reused verbatim.",
            "legal_chunker": "legal_structure.iter_legal_chunks is reused at 1800 characters and 180 overlap; no document is truncated before chunk scoring.",
            "new_phase_b_logic": "Frozen Jina-ColBERT-v2 token MaxSim at Matryoshka dimension 64, full-query plus per-element completeness, exhaustive one-to-three-passage union MaxSim, candidate/tail/delta and query-normalized features.",
            "safety": "Phase-A contradiction gate is retained as an explicit feature and strict policy; two pre-registered alternatives allow ColBERT dominance or calibrated all-candidate selection while preserving the one-slot rule.",
            "labels": "Complement references only for model/threshold selection. Locked holdout is not encoded until every complement gate passes; public is not encoded until holdout promotion.",
        },
        "artifacts": {
            "phase_a_pair_cache": str(pair_path.resolve()),
            "complement_colbert_scores": str((args.output_dir / "cache" / "complement_colbert_scores.npz").resolve()),
            "report": str(report_path.resolve()),
        },
        "holdout": {"opened": False},
        "public": {"applied": False},
    }
    if not winner["eligible"]:
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(report_path, report)
        print(
            json.dumps(
                {
                    "report": str(report_path.resolve()),
                    "status": report["status"],
                    "winner": winner,
                    "elapsed_seconds": report["elapsed_seconds"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return 2

    model_path = args.output_dir / "v14_colbert_replacement_ensemble.joblib"
    joblib.dump(
        {
            "models": winner_models,
            "winner": winner,
            "feature_names": augmented.feature_names,
            "colbert_document_feature_names": COLBERT_DOCUMENT_FEATURES,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "base_code_revision": BASE_CODE_REVISION,
            "embedding_dimension": EMBEDDING_DIMENSION,
        },
        model_path,
    )

    holdout = _load_split(
        "holdout",
        args.holdout_stage1,
        args.holdout_stage2,
        args.holdout_reference,
        args.score_cache,
        args.feature_cache,
        35,
    )
    holdout_base, holdout_pair_path = _base_pair_cache_for_split(holdout, args, "holdout")
    holdout_late, holdout_complete = _score_split(holdout, holdout_base, holdout_pair_path, args)
    if not holdout_complete:
        report["status"] = "holdout_checkpoint_incomplete"
        report["holdout"] = {"opened": True, "evaluation": False, "checkpoint_complete": False}
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(report_path, report)
        return 0
    holdout_augmented, holdout_arrays, holdout_direct = augment_pair_dataset(
        holdout, holdout_base, holdout_late
    )
    holdout_masks = gate_masks(holdout_base, holdout_arrays)
    holdout_policy = _dataset_with_gate(
        holdout_augmented, holdout_masks[winner["gate_mode"]]
    )
    holdout_probabilities = _predict_winner(
        winner, holdout_augmented, holdout_direct, winner_models
    )
    holdout_evaluation, holdout_predictions, holdout_trace = evaluate_policy(
        holdout, holdout_policy, holdout_probabilities, float(winner["threshold"])
    )
    observed_baseline = float(holdout_evaluation["baseline"]["recall"])
    if abs(observed_baseline - 0.9188333333333334) > 1e-12:
        raise RuntimeError(f"locked v11h baseline mismatch: {observed_baseline:.15f}")
    intervention = holdout_evaluation["intervention"]
    promoted = bool(
        holdout_evaluation["rescued"]["recall"] >= args.minimum_promote_recall
        and holdout_evaluation["delta"]["recall"] > 0.0
        and intervention["intervention_precision"] >= 0.85
        and intervention["destructive_rate"] <= 0.03
    )
    report["status"] = "holdout_promoted" if promoted else "holdout_rejected"
    report["holdout"] = {
        "opened": True,
        "frozen_threshold": winner["threshold"],
        "frozen_gate_mode": winner["gate_mode"],
        "evaluation": holdout_evaluation,
        "minimum_promote_recall": args.minimum_promote_recall,
        "promoted": promoted,
        "trace": holdout_trace,
    }
    report["artifacts"]["model"] = str(model_path.resolve())

    if promoted and args.apply_public:
        public = _load_split(
            "public",
            args.public_stage1,
            args.public_stage2,
            None,
            args.score_cache,
            args.feature_cache,
            35,
        )
        public_base, public_pair_path = _base_pair_cache_for_split(public, args, "public")
        public_late, public_complete = _score_split(public, public_base, public_pair_path, args)
        if not public_complete:
            report["status"] = "public_checkpoint_incomplete"
            report["public"] = {"applied": False, "checkpoint_complete": False}
            report["elapsed_seconds"] = time.perf_counter() - started
            write_json(report_path, report)
            return 0
        public_augmented, public_arrays, public_direct = augment_pair_dataset(
            public, public_base, public_late
        )
        public_masks = gate_masks(public_base, public_arrays)
        public_policy = _dataset_with_gate(
            public_augmented, public_masks[winner["gate_mode"]]
        )
        public_probabilities = _predict_winner(
            winner, public_augmented, public_direct, winner_models
        )
        public_predictions, public_trace = predictions_from_probabilities(
            public, public_policy, public_probabilities, float(winner["threshold"])
        )
        submission_path = args.output_dir / "v14_colbert_safe_rescue.json"
        write_json(submission_path, public_predictions)
        report["public"] = {
            "applied": True,
            "submission": str(submission_path.resolve()),
            "queries_changed": sum(
                public_predictions[qid]["answer"] != public.anchor[qid]["answer"]
                for qid in public.qids
            ),
            "trace": public_trace,
        }

    report["elapsed_seconds"] = time.perf_counter() - started
    write_json(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path.resolve()),
                "status": report["status"],
                "holdout": report["holdout"],
                "public": report["public"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0 if promoted else 3


if __name__ == "__main__":
    raise SystemExit(main())
