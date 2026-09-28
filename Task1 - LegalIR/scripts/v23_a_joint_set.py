"""V23-A: joint two-slot set reranking on frozen E3B representations.

The runner keeps V16b slots 1--3 immutable and learns the final two slots from
full relevance supervision.  It first completes a label-free representation
cache; labels are loaded only after that cache and all manifest key checks have
passed.  Four independent group-cross-fitted models produce the OOF pair
scores.  Acceptance uses a nested, training-only margin threshold and otherwise
retains the V16b anchor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REGIMES = (
    "standard_grouped",
    "law_family_ood",
    "query_template_ood",
    "near_duplicate_ood",
)
PACK_CONFIG = "B2304-D60"
HIDDEN_MAX_LENGTH = 512
SEED = 2301
FORBIDDEN_TERMS = ("holdout", "public", "locked", "submission")


def _safe(path: Path, label: str) -> None:
    folded = str(path.resolve()).replace("\\", "/").casefold()
    if any(term in folded for term in FORBIDDEN_TERMS):
        raise ValueError(f"{label} must remain clean-complement-only: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _load_questions_label_free(path: Path) -> dict[str, dict[str, Any]]:
    metadata = _read_json(path, "V18b metadata")
    if metadata.get("kind") != "legalir_v18b_evidence_pack_manifest" or metadata.get("split") != "clean_complement_only":
        raise ValueError("V23-A requires V18b clean-complement metadata")
    if metadata.get("holdout_opened") is not False or metadata.get("public_labels_used") is not False:
        raise ValueError("metadata does not attest unopened labels")
    raw_questions = metadata.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise ValueError("metadata.questions is missing")
    result: dict[str, dict[str, Any]] = {}
    for raw in raw_questions:
        if not isinstance(raw, Mapping):
            raise ValueError("malformed metadata question")
        qid = str(raw.get("qid", ""))
        query = str(raw.get("query", ""))
        anchor = [str(value) for value in raw.get("anchor_answer", [])]
        e3b = [str(value) for value in raw.get("e3b_scoring_boundary", [])]
        unavailable = [str(value) for value in raw.get("e3b_source_unavailable", [])]
        candidates = [str(value) for value in raw.get("candidates_c35", [])]
        scores = [float(value) for value in raw.get("anchor_scores", [])]
        if not qid or qid in result or not query.strip() or len(anchor) != 5 or len(set(anchor)) != 5:
            raise ValueError(f"invalid V18b question: {qid!r}")
        if not e3b or e3b[0] != anchor[-1] or len(e3b) != len(set(e3b)):
            raise ValueError(f"{qid}: E3B boundary is invalid")
        if set(unavailable) != set(raw.get("scoring_boundary", [])) - set(e3b):
            raise ValueError(f"{qid}: E3B unavailable declaration drift")
        if len(candidates) != 35 or len(scores) != 35 or len(set(candidates)) != 35:
            raise ValueError(f"{qid}: V16b c35 teacher-score matrix is invalid")
        teacher = {doc_id: score for doc_id, score in zip(candidates, scores, strict=True)}
        result[qid] = {
            "qid": qid,
            "query": query,
            "anchor": anchor,
            "slot4": anchor[3],
            "e3b": e3b,
            "unavailable": unavailable,
            "teacher": teacher,
            "fold": int(raw.get("fold", -1)),
        }
    return result


def _load_questions_with_labels(path: Path, label_free: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    metadata = _read_json(path, "V18b metadata for relevance labels")
    raw_questions = metadata.get("questions")
    if not isinstance(raw_questions, list):
        raise ValueError("metadata.questions is missing")
    result: dict[str, dict[str, Any]] = {}
    for raw in raw_questions:
        if not isinstance(raw, Mapping):
            raise ValueError("malformed labelled metadata question")
        qid = str(raw.get("qid", ""))
        if qid not in label_free:
            raise ValueError(f"label metadata qid is outside the label-free manifest: {qid}")
        references = {str(value) for value in raw.get("references", [])}
        if not references:
            raise ValueError(f"{qid}: empty relevance reference set")
        result[qid] = {**label_free[qid], "references": references}
    if set(result) != set(label_free):
        raise ValueError("label metadata qid coverage differs from label-free metadata")
    return result


def _load_fold_manifest(path: Path, metadata_path: Path, qids: set[str]) -> dict[str, Any]:
    manifest = _read_json(path, "V21 group fold manifest")
    if manifest.get("kind") != "legalir_v21_group_crossfit_manifest" or manifest.get("split") != "clean_complement_only":
        raise ValueError("V23-A requires the audited V21 group fold manifest")
    if manifest.get("metadata_sha256") != _sha256(metadata_path) or manifest.get("all_registered_regimes_zero_overlap") is not True:
        raise ValueError("V21 group manifest provenance or overlap audit failed")
    if set(str(value) for value in manifest.get("qids", [])) != qids:
        raise ValueError("fold manifest qids differ from metadata")
    return manifest


def _fold_data(manifest: Mapping[str, Any], regime: str, qids: set[str]) -> tuple[dict[str, int], dict[str, str]]:
    raw = manifest.get("regimes", {}).get(regime)
    if not isinstance(raw, Mapping):
        raise ValueError(f"fold manifest lacks {regime}")
    assignments = {str(qid): str(group) for qid, group in dict(raw.get("group_assignments", {})).items()}
    if set(assignments) != qids:
        raise ValueError(f"{regime}: group assignment coverage drift")
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
                raise ValueError(f"{regime}: duplicate validation qid {qid}")
            fold_by_qid[qid] = fold
    if set(fold_by_qid) != qids:
        raise ValueError(f"{regime}: validation folds do not cover every qid")
    return fold_by_qid, assignments


def _load_manifest_rows(path: Path, expected: set[tuple[str, str]], queries: Mapping[str, str], label: str, *, allow_missing: bool = False) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"{label} line {line_number} is not an object")
            if str(row.get("config")) != PACK_CONFIG:
                continue
            key = (str(row.get("qid", "")), str(row.get("doc_id", "")))
            if key not in expected:
                raise ValueError(f"{label} contains unexpected key {key}")
            if key in rows or str(row.get("query", "")) != queries[key[0]]:
                raise ValueError(f"{label} duplicate/query drift at {key}")
            blocks = row.get("blocks")
            count = int(row.get("block_count", -1))
            tokens = int(row.get("pair_token_count", -1))
            pack = str(row.get("pack", ""))
            if not isinstance(blocks, list) or len(blocks) != count or not 3 <= count <= 5 or not 1 <= tokens <= 2304 or not pack.strip():
                raise ValueError(f"{label} malformed pack at {key}")
            rows[key] = dict(row)
    missing = expected - set(rows)
    if missing and not allow_missing:
        raise ValueError(f"{label} misses {len(missing)} keys; first={sorted(missing)[:3]}")
    return rows


def _load_npz_cache(path: Path, expected_meta: Mapping[str, Any], expected_keys: set[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]], dict[str, Any]]:
    meta_path = path.with_suffix(path.suffix + ".provenance.json")
    meta = _read_json(meta_path, "representation cache provenance")
    for key, value in expected_meta.items():
        if meta.get(key) != value:
            raise ValueError(f"representation cache provenance mismatch at {key}")
    data = np.load(path, allow_pickle=False)
    embeddings = data["embeddings"]
    mask = data["mask"]
    raw_keys = data["keys"].tolist()
    keys = [tuple(str(part) for part in str(value).split("\t", 1)) for value in raw_keys]
    if embeddings.ndim != 3 or embeddings.shape[1] != 5 or mask.shape != embeddings.shape[:2] or len(keys) != embeddings.shape[0]:
        raise ValueError("representation cache dimensions are malformed")
    if set(keys) != expected_keys or len(set(keys)) != len(keys):
        raise ValueError("representation cache key set differs from manifest")
    return embeddings, mask, keys, meta


def _extract_slot4(
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
    questions: Mapping[str, Mapping[str, Any]],
    model_path: Path,
    max_length: int,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]], dict[str, Any]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ordered_keys = sorted(rows)
    key_index = {key: index for index, key in enumerate(ordered_keys)}
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    dtype = torch.float16 if device == "cuda" and torch.cuda.is_available() else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True, torch_dtype=dtype)
    actual_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    model.to(actual_device)
    model.eval()
    hidden_size = int(getattr(model.config, "hidden_size", 0))
    if hidden_size <= 0:
        raise ValueError("cross-encoder config does not expose hidden_size")
    embeddings = np.zeros((len(ordered_keys), 5, hidden_size), dtype=np.float16)
    mask = np.zeros((len(ordered_keys), 5), dtype=np.uint8)
    pending: list[tuple[int, int, str, str]] = []

    def flush() -> None:
        if not pending:
            return
        encoded = tokenizer(
            [item[2] for item in pending], [item[3] for item in pending],
            truncation=True, max_length=max_length, padding=True, return_tensors="pt",
        )
        encoded = {name: value.to(actual_device) for name, value in encoded.items()}
        with torch.inference_mode():
            output = model(**encoded, output_hidden_states=True, return_dict=True)
            states = output.hidden_states[-1][:, 0, :].detach().float().cpu().numpy().astype(np.float16, copy=False)
        for (bag_index, slot, _, _), value in zip(pending, states, strict=True):
            embeddings[bag_index, slot] = value
            mask[bag_index, slot] = 1
        pending.clear()

    for key in ordered_keys:
        row = rows[key]
        pack = str(row["pack"])
        passages = [fragment.strip() for fragment in re.split(r"(?m)^\[EVIDENCE_\d+\]\s*", pack)[1:] if fragment.strip()]
        if len(passages) != int(row["block_count"]):
            raise ValueError(f"slot-4 pack marker count differs at {key}")
        query = str(questions[key[0]]["query"])
        for slot, passage in enumerate(passages):
            pending.append((key_index[key], slot, query, passage))
            if len(pending) >= batch_size:
                flush()
        if key_index[key] % 100 == 0:
            print(f"V23-A slot-4 hidden extraction: {key_index[key]}/{len(ordered_keys)}", flush=True)
    flush()
    if not mask.any(axis=1).all():
        raise ValueError("slot-4 hidden extraction produced an empty bag")
    meta = {
        "kind": "legalir_v23_a_slot4_hidden_bags",
        "model_config_sha256": _sha256(model_path / "config.json"),
        "pack_config": PACK_CONFIG,
        "max_length": max_length,
        "bags": len(ordered_keys),
        "hidden_size": hidden_size,
        "blocks_encoded": int(mask.sum()),
        "dtype": "float16",
        "device": str(actual_device),
        "labels_loaded": False,
        "holdout_opened": False,
        "public_labels_used": False,
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embeddings, mask, ordered_keys, meta


def _combine_cache(
    cache_path: Path,
    e3b_cache: Path,
    e3b_meta: Mapping[str, Any],
    e3b_keys: set[tuple[str, str]],
    slot4_rows: Mapping[tuple[str, str], Mapping[str, Any]],
    questions: Mapping[str, Mapping[str, Any]],
    model_path: Path,
    metadata_path: Path,
    e3b_manifest_path: Path,
    slot4_manifest_path: Path,
    device: str,
    extract_batch_size: int,
) -> dict[str, Any]:
    expected_keys = e3b_keys | set(slot4_rows)
    combined_meta = {
        "kind": "legalir_v23_a_frozen_bag_representations",
        "metadata_sha256": _sha256(metadata_path),
        "e3b_manifest_sha256": _sha256(e3b_manifest_path),
        "slot4_manifest_sha256": _sha256(slot4_manifest_path),
        "model_config_sha256": _sha256(model_path / "config.json"),
        "pack_config": PACK_CONFIG,
        "max_length": HIDDEN_MAX_LENGTH,
        "e3b_cache_sha256": _sha256(e3b_cache),
    }
    provenance_path = cache_path.with_suffix(cache_path.suffix + ".provenance.json")
    if cache_path.is_file() and provenance_path.is_file():
        meta = _read_json(provenance_path, "V23-A combined cache provenance")
        if all(meta.get(key) == value for key, value in combined_meta.items()):
            data = np.load(cache_path, allow_pickle=False)
            keys = [tuple(str(part) for part in str(value).split("\t", 1)) for value in data["keys"].tolist()]
            if set(keys) == expected_keys and len(keys) == len(set(keys)):
                return {**meta, "cache_reused": True, "bags": len(keys), "keys": keys}
    e3b_embeddings, e3b_mask, e3b_ordered, _ = _load_npz_cache(
        e3b_cache,
        {
            "metadata_sha256": e3b_meta["metadata_sha256"],
            "pack_manifest_sha256": e3b_meta["pack_manifest_sha256"],
            "model_config_sha256": e3b_meta["model_config_sha256"],
            "pack_config": PACK_CONFIG,
            "max_length": HIDDEN_MAX_LENGTH,
        },
        e3b_keys,
    )
    slot4_embeddings, slot4_mask, slot4_ordered, slot4_meta = _extract_slot4(
        slot4_rows, questions, model_path, HIDDEN_MAX_LENGTH, extract_batch_size, device,
    )
    all_keys = sorted(expected_keys)
    hidden_size = int(e3b_embeddings.shape[-1])
    embeddings = np.zeros((len(all_keys), 5, hidden_size), dtype=np.float16)
    mask = np.zeros((len(all_keys), 5), dtype=np.uint8)
    e3b_index = {key: index for index, key in enumerate(e3b_ordered)}
    slot4_index = {key: index for index, key in enumerate(slot4_ordered)}
    for index, key in enumerate(all_keys):
        if key in e3b_index:
            embeddings[index] = e3b_embeddings[e3b_index[key]]
            mask[index] = e3b_mask[e3b_index[key]]
        else:
            embeddings[index] = slot4_embeddings[slot4_index[key]]
            mask[index] = slot4_mask[slot4_index[key]]
    temporary = cache_path.with_name(cache_path.name + ".tmp")
    with temporary.open("wb") as target:
        np.savez_compressed(target, embeddings=embeddings, mask=mask, keys=np.asarray([f"{qid}\t{doc}" for qid, doc in all_keys]))
    os.replace(temporary, cache_path)
    provenance = {
        **combined_meta,
        "cache_reused": False,
        "bags": len(all_keys),
        "hidden_size": hidden_size,
        "blocks_encoded": int(mask.sum()),
        "dtype": "float16",
        "device": slot4_meta["device"],
        "e3b_bags": len(e3b_ordered),
        "slot4_bags": len(slot4_ordered),
        "slot4_force_abstain_qids": sorted(set(questions) - {qid for qid, _ in slot4_ordered}),
        "labels_loaded": False,
        "holdout_opened": False,
        "public_labels_used": False,
    }
    _atomic_json(provenance_path, provenance)
    return {**provenance, "keys": all_keys}


class V23ASetPairHead:
    """Permutation-aware set encoder followed by ordered slot pair scoring."""

    def __init__(self, hidden_size: int, projection: int = 768, heads: int = 8, feedforward: int = 1536, layers: int = 2, pair_hidden: int = 512) -> None:
        import torch

        self.torch = torch
        self.module = torch.nn.ModuleDict({
            "block_norm": torch.nn.LayerNorm(hidden_size),
            "block_project": torch.nn.Linear(hidden_size, projection),
            "block_attention": torch.nn.Linear(projection, 1, bias=False),
            "set_norm": torch.nn.LayerNorm(projection),
            "set_encoder": torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(
                    d_model=projection, nhead=heads, dim_feedforward=feedforward,
                    dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
                ), layers,
                norm=torch.nn.LayerNorm(projection),
            ),
            "pair": torch.nn.Sequential(
                torch.nn.LayerNorm(projection * 4),
                torch.nn.Linear(projection * 4, pair_hidden),
                torch.nn.GELU(),
                torch.nn.Linear(pair_hidden, 2),
            ),
        })

    def to(self, device: Any) -> "V23ASetPairHead":
        self.module.to(device)
        return self

    def parameters(self) -> Any:
        return self.module.parameters()

    def train(self) -> None:
        self.module.train()

    def eval(self) -> None:
        self.module.eval()

    def __call__(self, bags: Any, block_mask: Any, candidate_mask: Any) -> tuple[Any, Any]:
        torch = self.torch
        candidate_mask = candidate_mask.bool()
        projected = torch.tanh(self.module["block_project"](self.module["block_norm"](bags)))
        attention = self.module["block_attention"](projected).squeeze(-1).masked_fill(~block_mask.bool(), -1e4)
        weights = torch.softmax(attention, dim=2).unsqueeze(-1)
        candidates = (weights * projected).sum(dim=2)
        candidates = self.module["set_norm"](candidates)
        set_encoded = self.module["set_encoder"](candidates, src_key_padding_mask=~candidate_mask.bool())
        left = set_encoded.unsqueeze(2).expand(-1, -1, set_encoded.shape[1], -1)
        right = set_encoded.unsqueeze(1).expand(-1, set_encoded.shape[1], -1, -1)
        pair_features = torch.cat((left, right, left * right, left - right), dim=-1)
        outputs = self.module["pair"](pair_features)
        valid_pairs = candidate_mask.unsqueeze(2) & candidate_mask.unsqueeze(1)
        diagonal = torch.eye(outputs.shape[1], dtype=torch.bool, device=outputs.device).unsqueeze(0)
        valid_pairs = valid_pairs & ~diagonal
        outputs = outputs.masked_fill(~valid_pairs.unsqueeze(-1), -1e4)
        return outputs[..., 0], outputs[..., 1]


def _batch_arrays(qids: Sequence[str], records: Mapping[str, Mapping[str, Any]], embeddings: np.ndarray, masks: np.ndarray, key_index: Mapping[tuple[str, str], int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hidden = embeddings.shape[-1]
    max_candidates = max(len(records[qid]["docs"]) for qid in qids)
    bag = np.zeros((len(qids), max_candidates, 5, hidden), dtype=np.float32)
    block_mask = np.zeros((len(qids), max_candidates, 5), dtype=np.uint8)
    candidate_mask = np.zeros((len(qids), max_candidates), dtype=np.uint8)
    for row, qid in enumerate(qids):
        for column, doc_id in enumerate(records[qid]["docs"]):
            index = key_index[(qid, doc_id)]
            bag[row, column] = embeddings[index]
            block_mask[row, column] = masks[index]
            candidate_mask[row, column] = 1
    return bag, block_mask, candidate_mask


def _pair_targets(records: Mapping[str, Mapping[str, Any]], qids: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_candidates = max(len(records[qid]["docs"]) for qid in qids)
    coverage = np.zeros((len(qids), max_candidates, max_candidates), dtype=np.float32)
    preserve = np.zeros_like(coverage)
    teacher = np.zeros_like(coverage)
    valid = np.zeros_like(coverage, dtype=np.uint8)
    for row, qid in enumerate(qids):
        record = records[qid]
        refs = record["references"]
        docs = record["docs"]
        incumbent = set(docs[:2])
        scores = record["teacher"]
        for left, left_doc in enumerate(docs):
            for right, right_doc in enumerate(docs):
                if left == right:
                    continue
                valid[row, left, right] = 1
                answer = set(record["anchor"][:3]) | {left_doc, right_doc}
                coverage[row, left, right] = len(answer & refs) / len(refs)
                preserve[row, left, right] = len((incumbent & refs) - {left_doc, right_doc}) / len(refs)
                teacher[row, left, right] = (float(scores.get(left_doc, 0.0)) + float(scores.get(right_doc, 0.0))) / 2.0
    return coverage, preserve, teacher, valid


def _fit_model(
    qids: Sequence[str],
    records: Mapping[str, Mapping[str, Any]],
    embeddings: np.ndarray,
    masks: np.ndarray,
    key_index: Mapping[tuple[str, str], int],
    *,
    device: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    gradient_clip: float,
    seed: int,
) -> tuple[V23ASetPairHead, dict[str, Any]]:
    import torch

    torch.manual_seed(seed)
    random.seed(seed)
    actual_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    model = V23ASetPairHead(embeddings.shape[-1]).to(actual_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    order = list(qids)
    rng = random.Random(seed)
    losses: list[float] = []
    for _ in range(epochs):
        rng.shuffle(order)
        epoch_losses: list[float] = []
        for start in range(0, len(order), batch_size):
            batch_qids = order[start:start + batch_size]
            bag, block_mask, candidate_mask = _batch_arrays(batch_qids, records, embeddings, masks, key_index)
            coverage, preserve, teacher, valid = _pair_targets(records, batch_qids)
            bags = torch.from_numpy(bag).to(actual_device)
            block_masks = torch.from_numpy(block_mask).to(actual_device)
            candidate_masks = torch.from_numpy(candidate_mask).to(actual_device)
            coverage_target = torch.from_numpy(coverage).to(actual_device)
            preserve_target = torch.from_numpy(preserve).to(actual_device)
            teacher_target = torch.from_numpy(teacher).to(actual_device)
            valid_pairs = torch.from_numpy(valid.astype(bool)).to(actual_device)
            model.train()
            ranking, coverage_logits = model(bags, block_masks, candidate_masks)
            terms: list[Any] = []
            for row in range(len(batch_qids)):
                local = valid_pairs[row]
                values = ranking[row][local]
                target = coverage_target[row][local]
                positives = values[target >= target.max() - 1e-8]
                if positives.numel():
                    terms.append(torch.logsumexp(values, dim=0) - torch.logsumexp(positives, dim=0))
                predicted_coverage = torch.sigmoid(coverage_logits[row][local])
                terms.append(torch.nn.functional.smooth_l1_loss(predicted_coverage, target))
                removed = preserve_target[row][local]
                terms.append(2.0 * (predicted_coverage * removed).mean())
                teacher_values = teacher_target[row][local]
                teacher_values = (teacher_values - teacher_values.mean()) / (teacher_values.std(unbiased=False) + 1e-4)
                rank_values = values
                rank_values = (rank_values - rank_values.mean()) / (rank_values.std(unbiased=False) + 1e-4)
                terms.append(0.10 * torch.nn.functional.smooth_l1_loss(rank_values, teacher_values))
            loss = torch.stack(terms).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), gradient_clip)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(statistics.fmean(epoch_losses) if epoch_losses else float("nan"))
    model.eval()
    return model, {"parameter_count": parameter_count, "epochs": epochs, "loss_by_epoch": losses, "device": str(actual_device), "queries": len(qids)}


def _predict_model(model: V23ASetPairHead, qids: Sequence[str], records: Mapping[str, Mapping[str, Any]], embeddings: np.ndarray, masks: np.ndarray, key_index: Mapping[tuple[str, str], int], device: str, batch_size: int) -> dict[str, dict[tuple[int, int], float]]:
    import torch

    actual_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    result: dict[str, dict[tuple[int, int], float]] = {}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(qids), batch_size):
            batch_qids = qids[start:start + batch_size]
            bag, block_mask, candidate_mask = _batch_arrays(batch_qids, records, embeddings, masks, key_index)
            ranking, _ = model(torch.from_numpy(bag).to(actual_device), torch.from_numpy(block_mask).to(actual_device), torch.from_numpy(candidate_mask).to(actual_device))
            values = ranking.detach().float().cpu().numpy()
            for row, qid in enumerate(batch_qids):
                n = len(records[qid]["docs"])
                result[qid] = {(left, right): float(values[row, left, right]) for left in range(n) for right in range(n) if left != right}
    return result


def _standardized_ensemble(regime_scores: Mapping[str, Mapping[str, Mapping[tuple[int, int], float]]], records: Mapping[str, Mapping[str, Any]], qid: str) -> dict[tuple[int, int], float]:
    pairs = [(left, right) for left in range(len(records[qid]["docs"])) for right in range(len(records[qid]["docs"])) if left != right]
    normalized: list[dict[tuple[int, int], float]] = []
    for regime in REGIMES:
        values = [float(regime_scores[regime][qid][pair]) for pair in pairs]
        mean = statistics.fmean(values)
        scale = statistics.pstdev(values) or 1.0
        normalized.append({pair: (float(regime_scores[regime][qid][pair]) - mean) / scale for pair in pairs})
    return {pair: statistics.median(source[pair] for source in normalized) for pair in pairs}


def _best_pair(scores: Mapping[tuple[int, int], float], record: Mapping[str, Any]) -> tuple[tuple[int, int], float]:
    baseline = (0, 1)
    pair = max(scores, key=lambda item: (float(scores[item]), item == baseline, -item[0], -item[1]))
    return pair, float(scores[pair] - scores.get(baseline, 0.0))


def _is_baseline_pair(pair: tuple[int, int]) -> bool:
    return set(pair) == {0, 1}


def _delta(record: Mapping[str, Any], pair: tuple[int, int]) -> float:
    refs = record["references"]
    before = len(set(record["anchor"]) & refs) / len(refs)
    docs = record["docs"]
    after = len((set(record["anchor"][:3]) | {docs[pair[0]], docs[pair[1]]}) & refs) / len(refs)
    return after - before


def _select_threshold(qids: Sequence[str], ensemble: Mapping[str, Mapping[tuple[int, int], float]], records: Mapping[str, Mapping[str, Any]]) -> tuple[float, dict[str, Any]]:
    margins: dict[str, tuple[tuple[int, int], float]] = {qid: _best_pair(ensemble[qid], records[qid]) for qid in qids}
    candidates = [0.0, *sorted({margin for _, margin in margins.values() if math.isfinite(margin) and margin > 0.0})]
    candidates.append(float("inf"))
    best_threshold = float("inf")
    best_key = (float("-inf"), float("-inf"), float("-inf"), float("-inf"))
    best_detail: dict[str, Any] = {}
    for threshold in candidates:
        accepted = [qid for qid, (pair, margin) in margins.items() if not _is_baseline_pair(pair) and margin > 0.0 and margin >= threshold]
        deltas = [_delta(records[qid], margins[qid][0]) for qid in accepted]
        improved = sum(value > 1e-12 for value in deltas)
        worsened = sum(value < -1e-12 for value in deltas)
        net = improved - worsened
        precision = improved / (improved + worsened) if improved + worsened else 0.0
        key = (float(net), float(precision), -float(worsened), float(threshold))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_detail = {"training_queries": len(qids), "accepted_queries": len(accepted), "improved": improved, "worsened": worsened, "net_query_equivalents": net, "precision": precision}
    return best_threshold, best_detail


def _predictions(
    records: Mapping[str, Mapping[str, Any]],
    ensemble: Mapping[str, Mapping[tuple[int, int], float]],
    thresholds: Mapping[str, float],
    force_abstain: set[str],
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    predictions: dict[str, list[str]] = {}
    traces: list[dict[str, Any]] = []
    for qid in sorted(records):
        record = records[qid]
        pair, margin = _best_pair(ensemble[qid], record)
        threshold = float(thresholds.get(qid, float("inf")))
        accepted = qid not in force_abstain and not _is_baseline_pair(pair) and margin > 0.0 and margin >= threshold
        docs = record["docs"]
        answer = [*record["anchor"][:3], docs[pair[0]], docs[pair[1]]] if accepted else list(record["anchor"])
        predictions[qid] = answer
        traces.append({
            "qid": qid,
            "candidate_docs": docs,
            "selected_pair_indices": list(pair),
            "selected_pair": [docs[pair[0]], docs[pair[1]]],
            "baseline_pair": [docs[0], docs[1]],
            "ensemble_margin": margin,
            "nested_threshold": threshold if math.isfinite(threshold) else None,
            "accepted": accepted,
            "forced_abstain": qid in force_abstain,
            "top3_frozen": answer[:3] == record["anchor"][:3],
        })
    return predictions, traces


def _regime_metrics(qids: Sequence[str], before: Mapping[str, float], after: Mapping[str, float], fold_by_qid: Mapping[str, int]) -> dict[str, Any]:
    folds = []
    for fold in sorted(set(fold_by_qid.values())):
        selected = [qid for qid in qids if fold_by_qid[qid] == fold]
        base = statistics.fmean(before[qid] for qid in selected)
        candidate = statistics.fmean(after[qid] for qid in selected)
        folds.append({"fold": fold, "queries": len(selected), "baseline_recall": base, "recall": candidate, "gain": candidate - base})
    base = statistics.fmean(before.values())
    candidate = statistics.fmean(after.values())
    robust_base = base - 0.25 * statistics.pstdev(row["baseline_recall"] for row in folds)
    robust_candidate = candidate - 0.25 * statistics.pstdev(row["recall"] for row in folds)
    return {
        "baseline_recall": base,
        "recall": candidate,
        "recall_gain": candidate - base,
        "baseline_robust_recall": robust_base,
        "robust_recall": robust_candidate,
        "robust_gain": robust_candidate - robust_base,
        "worst_fold_delta": min(row["gain"] for row in folds),
        "non_negative": candidate - base >= -1e-12 and robust_candidate - robust_base >= -1e-12 and min(row["gain"] for row in folds) >= -1e-12,
        "folds": folds,
    }


def _bootstrap(qids: Sequence[str], before: Mapping[str, float], after: Mapping[str, float], samples: int) -> dict[str, Any]:
    rng = random.Random(SEED)
    deltas: list[float] = []
    for _ in range(samples):
        deltas.append(statistics.fmean(after[qid] - before[qid] for qid in (qids[rng.randrange(len(qids))] for _ in qids)))
    deltas.sort()
    lower = deltas[math.floor(0.025 * (len(deltas) - 1))]
    upper = deltas[math.ceil(0.975 * (len(deltas) - 1))]
    return {"samples": samples, "mean": statistics.fmean(deltas), "lower_95": lower, "upper_95": upper}


def _build_records(questions: Mapping[str, Mapping[str, Any]], embeddings: np.ndarray, masks: np.ndarray, keys: Sequence[tuple[str, str]]) -> tuple[dict[str, dict[str, Any]], set[str]]:
    key_set = set(keys)
    records: dict[str, dict[str, Any]] = {}
    force_abstain: set[str] = set()
    for qid, question in questions.items():
        docs = [question["slot4"], *question["e3b"]]
        required = {(qid, doc_id) for doc_id in docs}
        if not required.issubset(key_set):
            force_abstain.add(qid)
            continue
        records[qid] = {**question, "docs": docs, "references": question["references"]}
    return records, force_abstain


def _validate_preregistration(path: Path) -> dict[str, Any]:
    payload = _read_json(path, "V23-A pre-registration")
    if payload.get("kind") != "legalir_v23_a_joint_two_slot_set_reranker_preregistration" or payload.get("split") != "clean_complement_only":
        raise ValueError("wrong V23-A pre-registration")
    if payload.get("holdout_opened") is not False or payload.get("public_labels_used") is not False:
        raise ValueError("V23-A pre-registration must keep external labels closed")
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    prereg = _validate_preregistration(args.pre_registration)
    for path, label in (
        (args.metadata, "metadata"), (args.e3b_manifest, "E3B manifest"),
        (args.e3b_cache, "E3B representation cache"), (args.slot4_manifest, "slot-4 manifest"),
        (args.fold_manifest, "fold manifest"), (args.p0_report, "V23-P0 report"),
        (args.model, "reranker model"), (args.output_dir, "output"),
    ):
        _safe(path, label)
    p0 = _read_json(args.p0_report, "V23-P0 report")
    if p0.get("kind") != "legalir_v23_p0_two_slot_feasibility" or p0.get("registered_v23_a_gate", {}).get("passed") is not True or p0.get("holdout_opened") is not False:
        raise ValueError("V23-A requires the passed V23-P0 gate with holdout closed")
    questions = _load_questions_label_free(args.metadata)
    qids = set(questions)
    queries = {qid: str(row["query"]) for qid, row in questions.items()}
    e3b_expected = {(qid, doc_id) for qid, row in questions.items() for doc_id in row["e3b"]}
    e3b_rows = _load_manifest_rows(args.e3b_manifest, e3b_expected, queries, "E3B manifest")
    slot4_expected_all = {(qid, row["slot4"]) for qid, row in questions.items()}
    slot4_rows = _load_manifest_rows(args.slot4_manifest, slot4_expected_all, queries, "slot-4 manifest", allow_missing=True)
    slot4_force_abstain = {qid for qid, _ in slot4_expected_all - set(slot4_rows)}
    fold_manifest = _load_fold_manifest(args.fold_manifest, args.metadata, qids)
    e3b_cache_meta = _read_json(args.e3b_cache.with_suffix(args.e3b_cache.suffix + ".provenance.json"), "V21 E3B cache provenance")
    e3b_cache_keys = set(e3b_expected)
    combined_cache_path = args.output_dir / "v23_a_frozen_bag_representations.npz"
    cache_report = _combine_cache(
        combined_cache_path, args.e3b_cache, e3b_cache_meta, e3b_cache_keys, slot4_rows,
        questions, args.model, args.metadata, args.e3b_manifest, args.slot4_manifest,
        args.device, args.extract_batch_size,
    )
    preflight = {
        "kind": "legalir_v23_a_representation_preflight",
        "schema_version": 1,
        "stage": "label_free_combined_cache_ready",
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "labels_loaded": False,
        "training_started": False,
        "selection_performed": False,
        "candidate_contract": "V16b top-3 frozen; slot-4 incumbent plus E3B-aligned six challenger boundary; source-unavailable challengers retained out of E3B and never substituted",
        "e3b_manifest_bags": len(e3b_rows),
        "slot4_manifest_bags": len(slot4_rows),
        "slot4_force_abstain_qids": sorted(slot4_force_abstain),
        "combined_cache": {key: value for key, value in cache_report.items() if key != "keys"},
        "key_bijection": True,
        "inputs": {
            "metadata": str(args.metadata.resolve()),
            "metadata_sha256": _sha256(args.metadata),
            "e3b_manifest": str(args.e3b_manifest.resolve()),
            "e3b_manifest_sha256": _sha256(args.e3b_manifest),
            "slot4_manifest": str(args.slot4_manifest.resolve()),
            "slot4_manifest_sha256": _sha256(args.slot4_manifest),
            "e3b_cache": str(args.e3b_cache.resolve()),
            "e3b_cache_sha256": _sha256(args.e3b_cache),
            "fold_manifest": str(args.fold_manifest.resolve()),
            "fold_manifest_sha256": _sha256(args.fold_manifest),
        },
    }
    preflight_path = args.output_dir / "v23_a_representation_preflight.json"
    if preflight_path.exists() and not args.force:
        old = _read_json(preflight_path, "existing V23-A representation preflight")
        old_compare = json.loads(json.dumps(old))
        new_compare = json.loads(json.dumps(preflight))
        old_compare.get("combined_cache", {}).pop("cache_reused", None)
        new_compare.get("combined_cache", {}).pop("cache_reused", None)
        if old_compare != new_compare:
            raise ValueError("existing V23-A preflight differs from current cache/provenance")
    else:
        _atomic_json(preflight_path, preflight)
    if args.preflight_only:
        return preflight

    # Labels are intentionally loaded only after the label-free cache and
    # preflight artifact are complete.
    labelled = _load_questions_with_labels(args.metadata, questions)
    data = np.load(combined_cache_path, allow_pickle=False)
    embeddings, masks = data["embeddings"], data["mask"]
    keys = [tuple(str(part) for part in str(value).split("\t", 1)) for value in data["keys"].tolist()]
    records, cache_force_abstain = _build_records(labelled, embeddings, masks, keys)
    force_abstain = slot4_force_abstain | cache_force_abstain
    if not records:
        raise ValueError("V23-A has no trainable query records after representation checks")
    key_index = {key: index for index, key in enumerate(keys)}
    fold_data = {regime: _fold_data(fold_manifest, regime, qids) for regime in REGIMES}
    optimization = prereg["optimisation"]
    regime_scores: dict[str, dict[str, dict[tuple[int, int], float]]] = {}
    crossfit_audit: dict[str, Any] = {}
    parameter_count = 0
    ready_qids = sorted(records)
    for regime_index, regime in enumerate(REGIMES):
        fold_by_qid, assignments = fold_data[regime]
        by_fold: dict[int, list[str]] = defaultdict(list)
        for qid in ready_qids:
            by_fold[fold_by_qid[qid]].append(qid)
        scores: dict[str, dict[tuple[int, int], float]] = {}
        fold_audit: dict[str, Any] = {}
        for fold in sorted(by_fold):
            validation = sorted(by_fold[fold])
            training = sorted(qid for qid in ready_qids if qid not in set(validation))
            train_groups = {assignments[qid] for qid in training}
            validation_groups = {assignments[qid] for qid in validation}
            if train_groups & validation_groups:
                raise ValueError(f"{regime}/{fold}: outer group overlap")
            model, fit_audit = _fit_model(
                training, records, embeddings, masks, key_index,
                device=args.device, epochs=int(optimization["epochs"]), learning_rate=float(optimization["learning_rate"]),
                weight_decay=float(optimization["weight_decay"]), batch_size=int(optimization["query_batch_size"]),
                gradient_clip=float(optimization["gradient_clip"]), seed=SEED + regime_index * 100 + fold,
            )
            parameter_count = max(parameter_count, int(fit_audit["parameter_count"]))
            scores.update(_predict_model(model, validation, records, embeddings, masks, key_index, args.device, int(optimization["query_batch_size"])))
            fold_audit[str(fold)] = {
                "train_qids": len(training), "validation_qids": len(validation),
                "train_groups": len(train_groups), "validation_groups": len(validation_groups),
                "group_overlap": False, "fit": fit_audit,
            }
            del model
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            print(f"V23-A {regime}: outer fold {fold} complete", flush=True)
        if set(scores) != set(ready_qids):
            raise ValueError(f"{regime}: OOF score coverage is incomplete")
        regime_scores[regime] = scores
        crossfit_audit[regime] = {"outer_folds": fold_audit, "ready_qids": len(ready_qids)}

    ensemble: dict[str, dict[tuple[int, int], float]] = {qid: _standardized_ensemble(regime_scores, records, qid) for qid in ready_qids}
    thresholds: dict[str, float] = {}
    threshold_by_regime: dict[str, dict[str, float]] = {}
    threshold_audit: dict[str, Any] = {}
    for regime in REGIMES:
        fold_by_qid, assignments = fold_data[regime]
        regime_thresholds: dict[str, float] = {}
        fold_details: dict[str, Any] = {}
        for fold in sorted(set(fold_by_qid.values())):
            validation = {qid for qid in ready_qids if fold_by_qid[qid] == fold}
            training = sorted(qid for qid in ready_qids if qid not in validation)
            train_groups = {assignments[qid] for qid in training}
            validation_groups = {assignments[qid] for qid in validation}
            if train_groups & validation_groups:
                raise ValueError(f"{regime}/{fold}: nested threshold group overlap")
            threshold, detail = _select_threshold(training, ensemble, records)
            for qid in validation:
                regime_thresholds[qid] = threshold
            fold_details[str(fold)] = {"threshold": threshold if math.isfinite(threshold) else None, "explicit_abstain": not math.isfinite(threshold), "group_overlap": False, **detail}
        threshold_audit[regime] = fold_details
        threshold_by_regime[regime] = regime_thresholds
    for qid in ready_qids:
        thresholds[qid] = statistics.median([float(threshold_by_regime[regime][qid]) for regime in REGIMES])
    for qid in force_abstain:
        thresholds[qid] = float("inf")
    predictions, winner_traces = _predictions(records, ensemble, thresholds, force_abstain)
    # Add forced queries back as immutable V16b predictions for complete metrics.
    for qid in sorted(set(labelled) - set(predictions)):
        predictions[qid] = list(labelled[qid]["anchor"])
        winner_traces.append({"qid": qid, "candidate_docs": [], "selected_pair_indices": [0, 1], "selected_pair": labelled[qid]["anchor"][3:], "baseline_pair": labelled[qid]["anchor"][3:], "ensemble_margin": 0.0, "nested_threshold": None, "accepted": False, "forced_abstain": True, "top3_frozen": True})
    all_qids = sorted(labelled)
    before = {qid: len(set(labelled[qid]["anchor"]) & labelled[qid]["references"]) / len(labelled[qid]["references"]) for qid in all_qids}
    after = {qid: len(set(predictions[qid]) & labelled[qid]["references"]) / len(labelled[qid]["references"]) for qid in all_qids}
    primary = _regime_metrics(all_qids, before, after, fold_data["standard_grouped"][0])
    regimes = {regime: _regime_metrics(all_qids, before, after, fold_data[regime][0]) for regime in REGIMES}
    bootstrap = _bootstrap(all_qids, before, after, args.bootstrap_samples)
    changed = [qid for qid in all_qids if predictions[qid] != labelled[qid]["anchor"]]
    deltas = [after[qid] - before[qid] for qid in changed]
    improved = sum(value > 1e-12 for value in deltas)
    worsened = sum(value < -1e-12 for value in deltas)
    precision = improved / (improved + worsened) if improved + worsened else 0.0
    net_equivalents = improved - worsened
    checks = {
        "oof_gain_at_least_0_010": primary["recall_gain"] >= 0.010,
        "robust_gain_at_least_0_008": primary["robust_gain"] >= 0.008,
        "bootstrap_lower_at_least_0_004": bootstrap["lower_95"] >= 0.004,
        "net_query_equivalents_at_least_12": net_equivalents >= 12,
        "improved_precision_at_least_0_90": precision >= 0.90,
        "all_four_regimes_non_negative": all(value["non_negative"] for value in regimes.values()),
        "top3_byte_identical": all(predictions[qid][:3] == labelled[qid]["anchor"][:3] for qid in all_qids),
        "slots_4_5_only": all(predictions[qid][:3] == labelled[qid]["anchor"][:3] and len(predictions[qid]) == 5 for qid in all_qids),
        "holdout_closed": True,
    }
    report = {
        "kind": "legalir_v23_a_joint_two_slot_set_reranker",
        "schema_version": 1,
        "stage": "V23-A_complete_clean_complement_oof",
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "labels_loaded_after_representation_preflight": True,
        "pre_registration": {"path": str(args.pre_registration.resolve()), "sha256": _sha256(args.pre_registration), "config": prereg},
        "preflight": {"path": str(preflight_path.resolve()), "sha256": _sha256(preflight_path)},
        "inputs": preflight["inputs"],
        "architecture": {"name": "V23ASetPairHead", "frozen_block_hidden_size": int(embeddings.shape[-1]), "set_attention_layers": 2, "candidate_projection": 768, "pair_hidden": 512, "parameter_count": parameter_count, "deployment_parameter_limit": 50_000_000, "under_4b": True},
        "data_contract": {"metadata_queries": len(all_qids), "trainable_queries": len(ready_qids), "forced_abstain_qids": sorted(force_abstain), "frozen_slots": [1, 2, 3], "mutable_slots": [4, 5], "candidate_set": "slot4 incumbent + V16b slot5 + E3B-aligned challengers", "e3b_challenger_count": 6, "no_challenger_substitution": True},
        "representation_cache": {key: value for key, value in cache_report.items() if key != "keys"},
        "cross_fit": crossfit_audit,
        "nested_thresholds": threshold_audit,
        "metrics_vs_v16b_anchor": primary,
        "regimes": regimes,
        "paired_bootstrap_vs_v16b": bootstrap,
        "interventions": {"changed_queries": len(changed), "improved_queries": improved, "worsened_queries": worsened, "neutral_changed_queries": len(changed) - improved - worsened, "improved_precision": precision, "net_query_equivalents": net_equivalents},
        "promotion": {"checks": checks, "passed": all(checks.values())},
        "decision": {"holdout_may_open": False, "if_passed": "freeze recipe, full-fit only after review, then at most one public submission; do not tune on leaderboard", "if_failed": "close V23-A, retain V16b, do not open holdout or start V24"},
        "winner_traces": sorted(winner_traces, key=lambda item: str(item["qid"])),
    }
    output_path = args.output_dir / "v23_a_report.json"
    if output_path.exists() and not args.force:
        raise FileExistsError(output_path)
    _atomic_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--e3b-manifest", type=Path, required=True)
    parser.add_argument("--e3b-cache", type=Path, required=True)
    parser.add_argument("--slot4-manifest", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--p0-report", type=Path, required=True)
    parser.add_argument("--pre-registration", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--extract-batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.extract_batch_size <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("extract batch size and bootstrap sample count must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = run(args)
    print(json.dumps({
        "output": str((args.output_dir / ("v23_a_representation_preflight.json" if args.preflight_only else "v23_a_report.json")).resolve()),
        "stage": report["stage"],
        "promotion_passed": report.get("promotion", {}).get("passed"),
        "holdout_opened": False,
        "labels_loaded": report.get("labels_loaded_after_representation_preflight", False),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
