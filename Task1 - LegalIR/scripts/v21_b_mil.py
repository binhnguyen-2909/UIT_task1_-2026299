"""V21-B: frozen hidden representations with document-bag MIL/TIR aggregation.

This is the conditional successor to V21-A.  It keeps V16b's candidates and
V18's query-conditioned B2304-D60 evidence blocks fixed, but replaces the
scalar CATER feature gate with an attention-based multiple-instance model over
cross-encoder hidden representations.  Frozen representation extraction is
label-free; rescue/destructive outcomes are read only after the cache exists
and each MIL head is fitted under the four audited group partitions.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REGIMES = (
    "standard_grouped",
    "law_family_ood",
    "query_template_ood",
    "near_duplicate_ood",
)
PACK_CONFIG = "B2304-D60"
PRECHECK_KIND = "legalir_v21_b_mil_label_free_preflight"
KIND = "legalir_v21_b_representation_mil"
FORBIDDEN_TERMS = ("holdout", "public", "locked", "submission")
SEED = 20_260_833
EVIDENCE_MARKER = re.compile(r"(?m)^\[EVIDENCE_\d+\]\s*")


@dataclass(frozen=True)
class PairRow:
    qid: str
    candidate_doc_id: str
    tail_doc_id: str
    candidate_rank: int


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


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _safe(path, label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _load_module(path: Path, name: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    for candidate in (str(Path(__file__).resolve().parent), str(path.resolve().parent)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def _load_questions(metadata_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _read_json(metadata_path, "metadata")
    if payload.get("kind") != "legalir_v18b_evidence_pack_manifest" or payload.get("split") != "clean_complement_only":
        raise ValueError("V21-B requires V18b clean-complement metadata")
    if payload.get("holdout_opened") is not False or payload.get("public_labels_used") is not False:
        raise ValueError("metadata does not attest unopened labels")
    raw_questions = payload.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise ValueError("metadata.questions is missing")
    questions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_questions:
        if not isinstance(raw, Mapping):
            raise ValueError("malformed metadata question")
        qid = str(raw.get("qid", ""))
        query = str(raw.get("query", ""))
        anchor = [str(value) for value in raw.get("anchor_answer", [])]
        scored = [str(value) for value in raw.get("e3b_scoring_boundary", [])]
        if not qid or qid in seen or not query.strip() or len(anchor) != 5 or anchor[-1] not in scored:
            raise ValueError(f"invalid V18b question boundary: {qid!r}")
        seen.add(qid)
        # References remain opaque until after frozen bags have been built.
        questions.append({**raw, "qid": qid, "query": query, "anchor_answer": anchor, "scored": scored})
    return payload, questions


def _required_keys(questions: Sequence[Mapping[str, Any]]) -> set[tuple[str, str]]:
    return {(str(question["qid"]), str(doc_id)) for question in questions for doc_id in question["scored"]}


def _bag_texts(pack: str, block_count: int) -> list[str]:
    fragments = EVIDENCE_MARKER.split(pack)
    if len(fragments) < 2:
        raise ValueError("V18 pack has no evidence markers")
    title = fragments[0].strip()
    values = [f"{title}\n{fragment.strip()}".strip() for fragment in fragments[1:] if fragment.strip()]
    if len(values) != block_count:
        raise ValueError(f"V18 evidence marker count differs from block_count ({len(values)} != {block_count})")
    return values


def _cache_meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".provenance.json")


def _load_cache(cache_path: Path, expected: Mapping[str, Any]) -> tuple[Any, Any, list[tuple[str, str]]] | None:
    import numpy as np

    meta_path = _cache_meta_path(cache_path)
    if not cache_path.is_file() or not meta_path.is_file():
        return None
    meta = _read_json(meta_path, "representation cache provenance")
    if any(meta.get(key) != value for key, value in expected.items()):
        return None
    data = np.load(cache_path, allow_pickle=False)
    embeddings, mask, raw_keys = data["embeddings"], data["mask"], data["keys"]
    keys = [tuple(str(part) for part in value.split("\t", 1)) for value in raw_keys.tolist()]
    if embeddings.ndim != 3 or mask.shape != embeddings.shape[:2] or len(keys) != len(embeddings):
        raise ValueError("representation cache dimensions are malformed")
    return embeddings, mask, keys


def _extract_cache(
    cache_path: Path, metadata_path: Path, pack_manifest_path: Path, model_path: Path,
    questions: Sequence[Mapping[str, Any]], *, max_length: int, batch_size: int, device: str,
) -> tuple[Any, Any, list[tuple[str, str]], dict[str, Any]]:
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    expected_keys = _required_keys(questions)
    qmap = {str(question["qid"]): str(question["query"]) for question in questions}
    expected_meta = {
        "kind": "legalir_v21_b_frozen_bag_representations",
        "metadata_sha256": _sha256(metadata_path),
        "pack_manifest_sha256": _sha256(pack_manifest_path),
        "model_config_sha256": _sha256(model_path / "config.json"),
        "pack_config": PACK_CONFIG,
        "max_length": int(max_length),
    }
    cached = _load_cache(cache_path, expected_meta)
    if cached is not None:
        embeddings, mask, keys = cached
        return embeddings, mask, keys, {**expected_meta, "cache_reused": True, "bags": len(keys)}
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    torch_dtype = torch.float16 if device == "cuda" and torch.cuda.is_available() else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True, torch_dtype=torch_dtype)
    actual_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    model.to(actual_device)
    model.eval()
    hidden_size = int(getattr(model.config, "hidden_size", 0))
    if hidden_size <= 0:
        raise ValueError("cross-encoder config does not expose hidden_size")
    ordered_keys = sorted(expected_keys)
    key_index = {key: index for index, key in enumerate(ordered_keys)}
    embeddings = np.zeros((len(ordered_keys), 5, hidden_size), dtype=np.float16)
    mask = np.zeros((len(ordered_keys), 5), dtype=np.uint8)
    found: set[tuple[str, str]] = set()
    pending: list[tuple[int, int, str, str]] = []

    def flush() -> None:
        if not pending:
            return
        queries = [item[2] for item in pending]
        passages = [item[3] for item in pending]
        encoded = tokenizer(queries, passages, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        encoded = {name: value.to(actual_device) for name, value in encoded.items()}
        with torch.inference_mode():
            output = model(**encoded, output_hidden_states=True, return_dict=True)
            states = output.hidden_states[-1][:, 0, :].detach().float().cpu().numpy().astype(np.float16, copy=False)
        for (bag_index, slot, _, _), value in zip(pending, states, strict=True):
            embeddings[bag_index, slot] = value
            mask[bag_index, slot] = 1
        pending.clear()

    with pack_manifest_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping) or str(row.get("config")) != PACK_CONFIG:
                continue
            key = (str(row.get("qid", "")), str(row.get("doc_id", "")))
            if key not in expected_keys:
                continue
            if key in found:
                raise ValueError(f"duplicate {PACK_CONFIG} bag at {key}")
            if str(row.get("query", "")) != qmap[key[0]]:
                raise ValueError(f"pack query drift at {key}")
            block_count = int(row.get("block_count", -1))
            if not 3 <= block_count <= 5:
                raise ValueError(f"invalid block count at {key}")
            passages = _bag_texts(str(row.get("pack", "")), block_count)
            bag_index = key_index[key]
            for slot, passage in enumerate(passages):
                pending.append((bag_index, slot, qmap[key[0]], passage))
                if len(pending) >= batch_size:
                    flush()
            found.add(key)
    flush()
    missing = expected_keys - found
    if missing or not mask.any(axis=1).all():
        raise ValueError(f"representation cache misses {len(missing)} required bags; first={sorted(missing)[:3]}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(cache_path.name + ".tmp")
    with temporary.open("wb") as target:
        np.savez_compressed(target, embeddings=embeddings, mask=mask, keys=np.asarray([f"{qid}\t{doc_id}" for qid, doc_id in ordered_keys]))
    os.replace(temporary, cache_path)
    provenance = {
        **expected_meta, "cache_reused": False, "bags": len(ordered_keys), "hidden_size": hidden_size,
        "blocks_encoded": int(mask.sum()), "dtype": "float16", "device": str(actual_device),
        "labels_loaded": False, "holdout_opened": False, "public_labels_used": False,
    }
    _cache_meta_path(cache_path).write_text(json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embeddings, mask, ordered_keys, provenance


def _fold_data(manifest: Mapping[str, Any], regime: str, qids: set[str]) -> tuple[dict[str, int], dict[str, str]]:
    raw = manifest.get("regimes", {}).get(regime)
    if not isinstance(raw, Mapping):
        raise ValueError(f"manifest misses {regime}")
    assignments = {str(qid): str(group) for qid, group in dict(raw.get("group_assignments", {})).items()}
    if set(assignments) != qids:
        raise ValueError(f"{regime}: assignment coverage drift")
    fold_by_qid: dict[str, int] = {}
    for item in raw.get("folds", []):
        if not isinstance(item, Mapping):
            raise ValueError(f"{regime}: malformed fold")
        fold = int(item.get("fold", -1))
        train = {str(value) for value in item.get("train_qids", [])}
        validation = {str(value) for value in item.get("validation_qids", [])}
        if fold < 0 or not validation or train & validation:
            raise ValueError(f"{regime}: malformed fold {fold}")
        if {assignments[qid] for qid in train} & {assignments[qid] for qid in validation}:
            raise ValueError(f"{regime}: group overlap in fold {fold}")
        for qid in validation:
            if qid in fold_by_qid:
                raise ValueError(f"{regime}: qid duplicated across validation folds")
            fold_by_qid[qid] = fold
    if set(fold_by_qid) != qids:
        raise ValueError(f"{regime}: invalid validation coverage")
    return fold_by_qid, assignments


def _pair_rows(questions: Sequence[Mapping[str, Any]]) -> list[PairRow]:
    rows: list[PairRow] = []
    for question in questions:
        qid = str(question["qid"])
        anchor = [str(value) for value in question["anchor_answer"]]
        tail = anchor[-1]
        for doc_id in question["scored"]:
            candidate = str(doc_id)
            if candidate == tail:
                continue
            rows.append(PairRow(qid, candidate, tail, int(question["scored"].index(candidate) + 1)))
    rows.sort(key=lambda row: (row.qid, row.candidate_rank, row.candidate_doc_id))
    return rows


def _labels(rows: Sequence[PairRow], questions: Sequence[Mapping[str, Any]]) -> list[int]:
    references = {str(question["qid"]): {str(value) for value in question["references"]} for question in questions}
    result: list[int] = []
    for row in rows:
        gold = references[row.qid]
        result.append(1 if row.candidate_doc_id in gold and row.tail_doc_id not in gold else 0 if row.tail_doc_id in gold and row.candidate_doc_id not in gold else -1)
    return result


class BagMIL:
    """Small learned attention pool over frozen CE block representations."""

    def __init__(self, hidden_size: int, latent_size: int) -> None:
        import torch

        self.torch = torch
        self.module = torch.nn.ModuleDict({
            "project": torch.nn.Sequential(torch.nn.LayerNorm(hidden_size), torch.nn.Linear(hidden_size, latent_size), torch.nn.Tanh()),
            "attention": torch.nn.Linear(latent_size, 1, bias=False),
            "rescue": torch.nn.Linear(latent_size, 1),
            "destructive": torch.nn.Linear(latent_size, 1),
        })

    def to(self, device: Any) -> "BagMIL":
        self.module.to(device)
        return self

    def parameters(self) -> Any:
        return self.module.parameters()

    def train(self) -> None:
        self.module.train()

    def eval(self) -> None:
        self.module.eval()

    def _pool(self, bags: Any, mask: Any) -> Any:
        torch = self.torch
        projected = self.module["project"](bags)
        logits = self.module["attention"](projected).squeeze(-1).masked_fill(~mask.bool(), -1e4)
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        return (weights * projected).sum(dim=1)

    def forward(self, candidate: Any, candidate_mask: Any, tail: Any, tail_mask: Any) -> tuple[Any, Any]:
        difference = self._pool(candidate, candidate_mask) - self._pool(tail, tail_mask)
        return self.module["rescue"](difference).squeeze(-1), self.module["destructive"](difference).squeeze(-1)


def _class_weight(values: Sequence[int], positive: int) -> float:
    target = [int(value == positive) for value in values]
    positives = sum(target)
    negatives = len(target) - positives
    return float(max(1.0, negatives / max(1, positives)))


def _fit_and_score_fold(
    rows: Sequence[PairRow], labels: Sequence[int], embeddings: Any, mask: Any, key_index: Mapping[tuple[str, str], int],
    train_indices: Sequence[int], validation_indices: Sequence[int], groups: Sequence[str], *, latent_size: int, epochs: int, batch_size: int, device: str, seed: int,
) -> tuple[list[float], list[float], list[float], dict[str, Any]]:
    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupShuffleSplit

    torch.manual_seed(seed)
    actual_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    hidden_size = int(embeddings.shape[-1])
    model = BagMIL(hidden_size, latent_size).to(actual_device)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    train_groups = np.asarray([str(groups[index]) for index in train_indices], dtype=object)
    fit_indices, calibration_indices = train_indices, np.asarray([], dtype=np.int64)
    if len(set(train_groups.tolist())) >= 4:
        split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        fit_local, cal_local = next(split.split(train_indices, groups=train_groups))
        candidate_fit, candidate_cal = train_indices[fit_local], train_indices[cal_local]
        # Retain a calibration split only when both outcome heads remain fit-able.
        if all(len({int(labels[index] == outcome) for index in candidate_fit}) == 2 for outcome in (1, 0)):
            fit_indices, calibration_indices = candidate_fit, candidate_cal
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rescue_weight = torch.tensor(_class_weight([labels[index] for index in fit_indices], 1), device=actual_device)
    destructive_weight = torch.tensor(_class_weight([labels[index] for index in fit_indices], 0), device=actual_device)
    loss_rescue = torch.nn.BCEWithLogitsLoss(pos_weight=rescue_weight)
    loss_destructive = torch.nn.BCEWithLogitsLoss(pos_weight=destructive_weight)

    def tensors(indices: Sequence[int]) -> tuple[Any, Any, Any, Any]:
        candidate = np.stack([embeddings[key_index[(rows[index].qid, rows[index].candidate_doc_id)]] for index in indices]).astype(np.float32, copy=False)
        candidate_mask = np.stack([mask[key_index[(rows[index].qid, rows[index].candidate_doc_id)]] for index in indices])
        tail = np.stack([embeddings[key_index[(rows[index].qid, rows[index].tail_doc_id)]] for index in indices]).astype(np.float32, copy=False)
        tail_mask = np.stack([mask[key_index[(rows[index].qid, rows[index].tail_doc_id)]] for index in indices])
        return (torch.from_numpy(candidate).to(actual_device), torch.from_numpy(candidate_mask).to(actual_device), torch.from_numpy(tail).to(actual_device), torch.from_numpy(tail_mask).to(actual_device))

    order = np.asarray(fit_indices, dtype=np.int64)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(epochs):
        permutation = torch.randperm(len(order), generator=generator).numpy()
        for start in range(0, len(order), batch_size):
            selected = order[permutation[start:start + batch_size]]
            candidate, candidate_mask, tail, tail_mask = tensors(selected)
            y = torch.tensor([labels[index] for index in selected], dtype=torch.int64, device=actual_device)
            model.train()
            rescue_logits, destructive_logits = model.forward(candidate, candidate_mask, tail, tail_mask)
            loss = loss_rescue(rescue_logits, (y == 1).float()) + loss_destructive(destructive_logits, (y == 0).float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
            optimizer.step()

    def raw_logits(indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        output_rescue: list[float] = []
        output_destructive: list[float] = []
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(indices), batch_size):
                candidate, candidate_mask, tail, tail_mask = tensors(indices[start:start + batch_size])
                rescue_logits, destructive_logits = model.forward(candidate, candidate_mask, tail, tail_mask)
                output_rescue.extend(rescue_logits.detach().float().cpu().tolist())
                output_destructive.extend(destructive_logits.detach().float().cpu().tolist())
        return np.asarray(output_rescue, dtype=np.float64), np.asarray(output_destructive, dtype=np.float64)

    rescue_calibrator = destructive_calibrator = None
    if len(calibration_indices) >= 8:
        cal_rescue, cal_destructive = raw_logits(calibration_indices)
        for target, values, name in (([int(labels[index] == 1) for index in calibration_indices], cal_rescue, "rescue"), ([int(labels[index] == 0) for index in calibration_indices], cal_destructive, "destructive")):
            if len(set(target)) == 2:
                fitted = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500, random_state=seed)
                fitted.fit(values.reshape(-1, 1), target)
                if name == "rescue":
                    rescue_calibrator = fitted
                else:
                    destructive_calibrator = fitted
    raw_rescue, raw_destructive = raw_logits(validation_indices)
    sigmoid = lambda values: 1.0 / (1.0 + np.exp(-values))
    rescue = rescue_calibrator.predict_proba(raw_rescue.reshape(-1, 1))[:, 1] if rescue_calibrator is not None else sigmoid(raw_rescue)
    destructive = destructive_calibrator.predict_proba(raw_destructive.reshape(-1, 1))[:, 1] if destructive_calibrator is not None else sigmoid(raw_destructive)
    utility = rescue - 3.0 * destructive
    stats = {
        "fit_rows": int(len(fit_indices)), "calibration_rows": int(len(calibration_indices)), "validation_rows": int(len(validation_indices)),
        "latent_size": latent_size, "epochs": epochs, "calibrated_rescue": rescue_calibrator is not None,
        "calibrated_destructive": destructive_calibrator is not None,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return utility.tolist(), rescue.tolist(), destructive.tolist(), stats


def _crossfit(
    rows: Sequence[PairRow], labels: Sequence[int], embeddings: Any, mask: Any, key_index: Mapping[tuple[str, str], int],
    fold_by_qid: Mapping[str, int], group_map: Mapping[str, str], *, latent_size: int, epochs: int, batch_size: int, device: str, seed: int,
) -> tuple[list[float], list[float], list[float], dict[str, Any]]:
    import numpy as np

    folds = sorted(set(fold_by_qid.values()))
    utilities, rescue, destructive = (np.full(len(rows), np.nan, dtype=np.float64) for _ in range(3))
    stats: dict[str, Any] = {"outer_folds": {}, "group_overlap_all_folds": True}
    for fold in folds:
        validation = [index for index, row in enumerate(rows) if fold_by_qid[row.qid] == fold]
        training = [index for index, row in enumerate(rows) if fold_by_qid[row.qid] != fold]
        train_groups = {group_map[rows[index].qid] for index in training}
        validation_groups = {group_map[rows[index].qid] for index in validation}
        if train_groups & validation_groups:
            raise ValueError(f"MIL outer group leakage in fold {fold}")
        utility, p_rescue, p_destructive, fold_stats = _fit_and_score_fold(
            rows, labels, embeddings, mask, key_index, training, validation,
            [group_map[row.qid] for row in rows], latent_size=latent_size, epochs=epochs, batch_size=batch_size, device=device, seed=seed + fold,
        )
        utilities[validation], rescue[validation], destructive[validation] = utility, p_rescue, p_destructive
        stats["outer_folds"][str(fold)] = {"train_qids": len({rows[index].qid for index in training}), "validation_qids": len({rows[index].qid for index in validation}), "train_groups": len(train_groups), "validation_groups": len(validation_groups), "group_overlap": False, **fold_stats}
    if not np.isfinite(utilities).all():
        raise RuntimeError("MIL OOF prediction coverage failure")
    return utilities.tolist(), rescue.tolist(), destructive.tolist(), stats


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path, label in ((args.metadata, "metadata"), (args.pack_manifest, "pack manifest"), (args.fold_manifest, "fold manifest"), (args.preflight, "MIL preflight"), (args.model, "reranker model"), (args.legacy_cater, "metric helpers"), (args.output_dir, "output")):
        _safe(path, label)
    metadata, questions = _load_questions(args.metadata)
    qids = {str(question["qid"]) for question in questions}
    preflight = _read_json(args.preflight, "MIL preflight")
    if preflight.get("kind") != PRECHECK_KIND or preflight.get("labels_loaded") is not False or preflight.get("training_started") is not False:
        raise ValueError("V21-B needs the label-free preflight artifact")
    if preflight.get("inputs", {}).get("metadata_sha256") != _sha256(args.metadata) or preflight.get("inputs", {}).get("pack_manifest_sha256") != _sha256(args.pack_manifest):
        raise ValueError("MIL preflight provenance differs from inputs")
    manifest = _read_json(args.fold_manifest, "fold manifest")
    if manifest.get("kind") != "legalir_v21_group_crossfit_manifest" or manifest.get("metadata_sha256") != _sha256(args.metadata) or manifest.get("all_registered_regimes_zero_overlap") is not True:
        raise ValueError("V21-B needs the audited V21 group manifest")
    if set(str(value) for value in manifest.get("qids", [])) != qids:
        raise ValueError("fold manifest qid drift")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "frozen_bag_representations.npz"
    embeddings, mask, keys, cache_report = _extract_cache(cache_path, args.metadata, args.pack_manifest, args.model, questions, max_length=args.max_length, batch_size=args.extract_batch_size, device=args.device)
    key_index = {key: index for index, key in enumerate(keys)}
    expected = _required_keys(questions)
    if set(key_index) != expected:
        raise ValueError("representation cache key set differs from V18 boundary")
    rows = _pair_rows(questions)
    if args.dry_run:
        return {"kind": KIND, "stage": "representation_cache_ready", "cache": cache_report, "pair_rows": len(rows), "labels_loaded": False, "holdout_opened": False, "public_labels_used": False}

    # Labels are intentionally constructed only after label-free cache parity.
    labels = _labels(rows, questions)
    utilities: dict[str, list[float]] = {}
    rescue_probs: dict[str, list[float]] = {}
    destructive_probs: dict[str, list[float]] = {}
    groups: dict[str, dict[str, str]] = {}
    crossfit: dict[str, Any] = {}
    for offset, regime in enumerate(REGIMES):
        fold_by_qid, groups[regime] = _fold_data(manifest, regime, qids)
        utility, rescue, destructive, audit = _crossfit(rows, labels, embeddings, mask, key_index, fold_by_qid, groups[regime], latent_size=args.latent_size, epochs=args.epochs, batch_size=args.train_batch_size, device=args.device, seed=SEED + offset * 101)
        utilities[regime], rescue_probs[regime], destructive_probs[regime], crossfit[regime] = utility, rescue, destructive, audit

    from v21_winner_policy import _consensus_winners, _nested_winner_thresholds, _predictions
    legacy = _load_module(args.legacy_cater, "v21_b_metric_helpers")
    rows_by_regime = {regime: rows for regime in REGIMES}
    winners, candidate_traces = _consensus_winners(questions, rows_by_regime, utilities)
    thresholds, threshold_audit = _nested_winner_thresholds(winners, manifest, qids)
    predictions, winner_traces = _predictions(questions, winners, thresholds)
    baseline = {str(question["qid"]): list(question["anchor_answer"]) for question in questions}
    primary = legacy._regime_metrics(questions, predictions, baseline, groups["standard_grouped"])
    regime_metrics = {regime: legacy._regime_metrics(questions, predictions, baseline, groups[regime]) for regime in REGIMES}
    bootstrap = legacy._bootstrap(predictions, baseline, questions, samples=args.bootstrap_samples, seed=SEED)
    promotion = legacy._promotion(primary, bootstrap, regime_metrics, top_four_frozen=all(predictions[str(question["qid"])][:4] == question["anchor_answer"][:4] for question in questions), max_one_swap=True, data_safe=True)
    report = {
        "kind": KIND, "schema_version": 1, "stage": "V21_B_complete_clean_complement", "split": "clean_complement_only",
        "holdout_opened": False, "public_labels_used": False, "selection_performed": False,
        "architecture": {"encoder": "frozen existing CE hidden CLS representations", "instances": "3..5 query-conditioned evidence blocks per document", "aggregation": "learned attention document-bag MIL", "pair_head": "two calibrated rescue/destructive heads over candidate-minus-incumbent bag representation", "utility": "P(rescue) - 3 * P(destructive)", "not_scalar_cater_selector": True},
        "cache": cache_report, "cross_fit": crossfit, "winner_level_thresholds": threshold_audit,
        "metrics_vs_v16b_anchor": primary, "regimes": regime_metrics, "paired_bootstrap_vs_v16b": bootstrap, "promotion": promotion,
        "trace_summary": {"pair_rows": len(rows), "candidate_rows": len(candidate_traces), "winner_rows": len(winner_traces), "accepted_queries": sum(bool(item["accepted"]) for item in winner_traces), "consensus_eligible_winners": sum(bool(item["consensus_eligible"]) for item in winners.values())},
        "winner_traces": winner_traces,
        "parameter_ledger": {"base_pipeline_parameters": 1_090_010_000, "frozen_encoder_added_parameters": 0, "mil_head_deployment_upper_bound": int(next(iter(crossfit.values()))["outer_folds"]["0"]["parameter_count"]), "end_to_end_upper_bound": 1_091_000_000, "under_4b": True},
        "provenance": {"metadata": str(args.metadata.resolve()), "metadata_sha256": _sha256(args.metadata), "pack_manifest": str(args.pack_manifest.resolve()), "pack_manifest_sha256": _sha256(args.pack_manifest), "fold_manifest": str(args.fold_manifest.resolve()), "fold_manifest_sha256": _sha256(args.fold_manifest), "preflight": str(args.preflight.resolve()), "preflight_sha256": _sha256(args.preflight), "labels_used_after_representation_cache": True, "holdout_opened": False, "public_labels_used": False},
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--pack-manifest", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--legacy-cater", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--extract-batch-size", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--latent-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output_dir / "v21_b_mil_report.json"
    _safe(output, "output")
    if output.exists() and not args.force:
        raise FileExistsError(output)
    report = run(args)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"output": str(output.resolve()), "stage": report["stage"], "promoted": report.get("promotion", {}).get("promoted"), "holdout_opened": False}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
