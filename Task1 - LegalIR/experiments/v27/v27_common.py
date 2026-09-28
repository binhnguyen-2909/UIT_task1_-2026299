"""Shared, holdout-blind utilities for the V27-A CrossEncoder pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
CANONICAL_E3B_CONFIG = "B2304-D60"
MAX_LENGTH = 384
SEED = 42
REGIMES = ("standard_grouped", "law_family_ood", "query_template_ood", "near_duplicate_ood")

CONTEXT_RE = re.compile(r"(?:^|/)context_(?P<doc_id>[^/]+)\.json$", re.IGNORECASE)
LAW_RE = re.compile(
    r"\b(?:bộ\s+luật|luật|nghị\s+định|thông\s+tư|quyết\s+định|pháp\s+lệnh)\b[^\n]{0,100}",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update((canonical_json(row) + "\n").encode("utf-8"))
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("wb") as target:
        for row in rows:
            encoded = (canonical_json(row) + "\n").encode("utf-8")
            target.write(encoded)
            digest.update(encoded)
    os.replace(temporary, path)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_text(value: str) -> str:
    return " ".join(TOKEN_RE.findall(str(value).casefold()))


def text_tokens(value: str) -> set[str]:
    return set(TOKEN_RE.findall(str(value).casefold()))


def law_family_key(title: str) -> str:
    match = LAW_RE.search(str(title))
    if not match:
        return ""
    return normalized_text(match.group(0))


def load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    if payload.get("kind") != "legalir_v18b_evidence_pack_manifest" or payload.get("split") != "clean_complement_only":
        raise ValueError("V27 requires the clean-complement V18b metadata")
    if payload.get("holdout_opened") is not False or payload.get("public_labels_used") is not False:
        raise ValueError("V27 refuses metadata that opened holdout/public labels")
    result: dict[str, dict[str, Any]] = {}
    for raw in payload.get("questions", []):
        qid = str(raw["qid"])
        c35 = [str(value) for value in raw["candidates_c35"]]
        anchor = [str(value) for value in raw["anchor_answer"]]
        refs = {str(value) for value in raw["references"]}
        if len(c35) != 35 or len(set(c35)) != 35 or len(anchor) != 5 or len(set(anchor)) != 5 or not refs:
            raise ValueError(f"{qid}: invalid c35/anchor/qrel shape")
        if not set(anchor).issubset(set(c35)):
            raise ValueError(f"{qid}: frozen anchor escaped c35")
        result[qid] = {
            "qid": qid,
            "query": str(raw["query"]),
            "c35": c35,
            "anchor": anchor,
            "references": refs,
            "anchor_scores": [float(value) for value in raw.get("anchor_scores", [])],
        }
    if not result:
        raise ValueError("V27 metadata is empty")
    return result


def load_sidecar(path: Path, questions: Mapping[str, Mapping[str, Any]]) -> dict[tuple[str, str], float]:
    payload = read_json(path)
    metadata = payload.get("metadata", {})
    kind = payload.get("kind", metadata.get("kind"))
    holdout_opened = payload.get("holdout_opened", metadata.get("holdout_opened"))
    public_labels_used = payload.get("public_labels_used", metadata.get("public_labels_used"))
    labels_loaded = payload.get("labels_loaded", metadata.get("labels_loaded"))
    label_free = payload.get("label_free", metadata.get("label_free"))
    if kind != "legalir_v20_stage2_ce_label_free_sidecar" or not label_free:
        raise ValueError("Stage2 sidecar is not the expected label-free artifact")
    if holdout_opened is not False or public_labels_used is not False or labels_loaded is not False:
        raise ValueError("Stage2 sidecar is not label-safe")
    scores: dict[tuple[str, str], float] = {}
    for row in payload.get("rows", []):
        key = (str(row["qid"]), str(row["doc_id"]))
        if key in scores:
            raise ValueError(f"duplicate Stage2 score {key}")
        scores[key] = float(row["ce_score"])
    expected = {(qid, doc) for qid, row in questions.items() for doc in row["c35"]}
    if set(scores) != expected:
        raise ValueError(f"Stage2 sidecar coverage mismatch: got={len(scores)} expected={len(expected)}")
    return scores


def load_document_info(contexts_zip: Path, doc_ids: set[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(contexts_zip) as archive:
        members: dict[str, str] = {}
        for name in archive.namelist():
            match = CONTEXT_RE.search(name)
            if match and match.group("doc_id") in doc_ids:
                members[match.group("doc_id")] = name
        for doc_id in sorted(doc_ids):
            name = members.get(doc_id)
            if name is None:
                raise ValueError(f"missing context document {doc_id}")
            raw = json.loads(archive.read(name))
            body = str(raw.get("passage", ""))
            title = " ".join(str(raw.get("name", "")).split())
            normalized_body = normalized_text(body)
            empty_body = not bool(normalized_body)
            result[doc_id] = {
                "doc_id": doc_id,
                "title": title,
                "body": body,
                "body_hash": sha256_bytes(normalized_body.encode("utf-8")),
                "canonical_key": sha256_bytes((f"EMPTY:{doc_id}" if empty_body else normalized_body).encode("utf-8")),
                "empty_body": empty_body,
                "title_tokens": text_tokens(title),
                "law_family": law_family_key(title),
            }
    return result


def make_canonical_map(doc_info: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    groups: dict[str, list[str]] = defaultdict(list)
    for doc_id, row in doc_info.items():
        groups[str(row["canonical_key"])].append(str(doc_id))
    result: dict[str, str] = {}
    for members in groups.values():
        canonical = min(members, key=lambda value: (int(value) if value.isdigit() else math.inf, value))
        for doc_id in members:
            result[doc_id] = canonical
    return result


def recall(answer: Sequence[str], references: set[str]) -> float:
    return len(set(answer) & references) / max(len(references), 1)


def baseline_recall(questions: Mapping[str, Mapping[str, Any]]) -> float:
    return float(np.mean([recall(row["anchor"], row["references"]) for row in questions.values()]))


class UnionFind:
    def __init__(self, values: Iterable[str]):
        self.parent = {str(value): str(value) for value in values}

    def find(self, value: str) -> str:
        value = str(value)
        parent = self.parent[value]
        while parent != self.parent[parent]:
            self.parent[parent] = self.parent[self.parent[parent]]
            parent = self.parent[parent]
        self.parent[value] = parent
        return parent

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def build_grouped_folds(qids: Sequence[str], source_manifest: Path, n_splits: int = 4, primary_regime: str = "standard_grouped") -> dict[str, Any]:
    """Make balanced outer folds on the registered standard query grouping.

    The four V21 regimes are deliberately retained as independent fold maps for
    transfer reporting.  Their transitive intersection is not used as the
    training split because it collapses most of this corpus into one component
    and makes four-way OOF estimates unusably imbalanced.
    """
    from sklearn.model_selection import GroupKFold

    payload = read_json(source_manifest)
    if payload.get("kind") != "legalir_v21_group_crossfit_manifest" or payload.get("holdout_opened") is not False:
        raise ValueError("invalid V21 group manifest")
    qid_set = {str(qid) for qid in qids}
    assignments_by_regime: dict[str, dict[str, str]] = {}
    for regime in REGIMES:
        assignments = {str(k): str(v) for k, v in payload["regimes"][regime]["group_assignments"].items()}
        if set(assignments) != qid_set:
            raise ValueError(f"{regime}: group manifest qids differ")
        assignments_by_regime[regime] = assignments
    if primary_regime not in assignments_by_regime:
        raise ValueError(f"unknown primary grouping regime: {primary_regime}")
    primary_assignments = assignments_by_regime[primary_regime]
    primary_values = [primary_assignments[qid] for qid in qids]
    if len(set(primary_values)) < n_splits:
        raise ValueError("primary grouping has too few groups")
    folds: list[dict[str, Any]] = []
    indices = np.arange(len(qids))
    for fold, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=n_splits).split(indices, groups=primary_values)):
        train_qids = [str(qids[int(i)]) for i in train_idx]
        test_qids = [str(qids[int(i)]) for i in test_idx]
        calibration_qids = sorted(
            train_qids,
            key=lambda qid: hashlib.sha256(f"{qid}|{SEED}|{fold}|calibration".encode("utf-8")).hexdigest(),
        )[:max(1, math.ceil(len(train_qids) * 0.10))]
        calibration_set = set(calibration_qids)
        fit_qids = [qid for qid in train_qids if qid not in calibration_set]
        if not fit_qids or calibration_set & set(test_qids):
            raise ValueError(f"fold {fold}: invalid nested calibration split")
        folds.append({
            "fold": fold,
            "train_qids": train_qids,
            "fit_qids": fit_qids,
            "calibration_qids": calibration_qids,
            "test_qids": test_qids,
            "train_groups": sorted({primary_assignments[q] for q in train_qids}),
            "test_groups": sorted({primary_assignments[q] for q in test_qids}),
            "calibration_fraction": len(calibration_qids) / len(train_qids),
        })
    regime_folds: dict[str, dict[str, int]] = {}
    for regime, assignments in assignments_by_regime.items():
        regime_group_values = [assignments[qid] for qid in qids]
        regime_map: dict[str, int] = {}
        for fold, (_, test_idx) in enumerate(GroupKFold(n_splits=n_splits).split(indices, groups=regime_group_values)):
            for index in test_idx:
                regime_map[str(qids[int(index)])] = fold
        if set(regime_map) != qid_set:
            raise ValueError(f"{regime}: failed to assign every qid to a regime fold")
        regime_folds[regime] = regime_map
    return {
        "kind": "legalir_v27_grouped_4fold_manifest",
        "schema_version": 1,
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "qids": list(map(str, qids)),
        "qids_sha256": sha256_bytes(canonical_json(list(map(str, qids))).encode("utf-8")),
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest),
        "primary_grouping_regime": primary_regime,
        "primary_group_assignments": primary_assignments,
        "folds": folds,
        "regime_folds": regime_folds,
        "regimes": {regime: {"group_assignments": assignments} for regime, assignments in assignments_by_regime.items()},
        "contract": {
            "primary_group_isolated": True,
            "all_registered_regimes_group_isolated": False,
            "regime_fold_maps_saved_for_transfer_diagnostics": True,
            "outer_folds": 4,
            "inner_calibration_excludes_outer_test": True,
        },
    }


def fold_qids(split: Mapping[str, Any], fold: int) -> tuple[list[str], list[str]]:
    row = next(item for item in split["folds"] if int(item["fold"]) == int(fold))
    return list(map(str, row["train_qids"])), list(map(str, row["test_qids"]))


def label_for_doc(row: Mapping[str, Any], doc_id: str, canonical: Mapping[str, str]) -> int:
    gold = {canonical.get(str(value), str(value)) for value in row["references"]}
    return int(canonical.get(str(doc_id), str(doc_id)) in gold)


def candidate_swap(answer: Sequence[str], qid: str, scores: Mapping[tuple[str, str], float], questions: Mapping[str, Mapping[str, Any]], *, tau_abs: float, tau_delta: float, empty_docs: set[tuple[str, str]] | None = None) -> dict[str, Any]:
    empty_docs = empty_docs or set()
    row = questions[qid]
    c35 = row["c35"]
    incumbent = row["anchor"][4]
    if (qid, incumbent) in empty_docs:
        return {"answer": list(answer), "changed": False, "challenger": None, "reason": "empty_incumbent"}
    challenger_docs = [doc for doc in c35[4:] if doc not in row["anchor"][:4] and doc != incumbent]
    viable = [(doc, float(scores[(qid, doc)])) for doc in challenger_docs if (qid, doc) not in empty_docs]
    if not viable:
        return {"answer": list(answer), "changed": False, "challenger": None, "reason": "no_viable_challenger"}
    challenger, challenger_score = max(viable, key=lambda item: (item[1], -c35.index(item[0])))
    incumbent_score = float(scores[(qid, incumbent)])
    if challenger_score < tau_abs or challenger_score - incumbent_score < tau_delta:
        return {"answer": list(answer), "changed": False, "challenger": challenger, "reason": "threshold"}
    updated = list(row["anchor"][:4]) + [challenger]
    return {"answer": updated, "changed": updated != list(answer), "challenger": challenger, "reason": "swap"}


def metric_summary(rows: Sequence[Mapping[str, Any]], baseline: Mapping[str, float], bootstrap_samples: int = 10000) -> dict[str, Any]:
    deltas = np.asarray([float(row["recall"]) - float(baseline[str(row["qid"])]) for row in rows], dtype=np.float64)
    recalls = np.asarray([float(row["recall"]) for row in rows], dtype=np.float64)
    base = np.asarray([float(baseline[str(row["qid"])]) for row in rows], dtype=np.float64)
    fold_values: dict[int, list[float]] = defaultdict(list)
    for row, delta in zip(rows, deltas, strict=True):
        fold_values[int(row["fold"])].append(float(row["recall"]))
    fold_recall = [float(np.mean(fold_values[key])) for key in sorted(fold_values)]
    fold_base = [float(np.mean([baseline[str(row["qid"])] for row in rows if int(row["fold"]) == key])) for key in sorted(fold_values)]
    fold_gain = [recall_value - base_value for recall_value, base_value in zip(fold_recall, fold_base, strict=True)]
    rng = np.random.default_rng(SEED)
    if len(deltas) and bootstrap_samples:
        bootstrap = np.asarray([np.mean(deltas[rng.integers(0, len(deltas), len(deltas))]) for _ in range(bootstrap_samples)], dtype=np.float64)
        lb = float(np.quantile(bootstrap, 0.025))
    else:
        lb = 0.0
    rescues = sum(delta > 1e-12 for delta in deltas)
    destructive = sum(delta < -1e-12 for delta in deltas)
    changed = sum(bool(row.get("changed")) for row in rows)
    return {
        "queries": len(rows),
        "baseline_recall": float(np.mean(base)) if len(base) else 0.0,
        "recall": float(np.mean(recalls)) if len(recalls) else 0.0,
        "gain": float(np.mean(deltas)) if len(deltas) else 0.0,
        "fold_recalls": fold_recall,
        "fold_baseline_recalls": fold_base,
        "fold_gains": fold_gain,
        "robust_recall": float(np.mean(fold_recall) - np.std(fold_recall)) if fold_recall else 0.0,
        "robust_baseline_recall": float(np.mean(fold_base) - np.std(fold_base)) if fold_base else 0.0,
        "robust_gain": float((np.mean(fold_recall) - np.std(fold_recall)) - (np.mean(fold_base) - np.std(fold_base))) if fold_recall else 0.0,
        "bootstrap_lower_95": lb,
        "rescues": int(rescues),
        "destructive": int(destructive),
        "neutral": int(len(rows) - rescues - destructive),
        "changed_queries": int(changed),
        "net_query_equivalent": int(rescues - destructive),
        "rescue_precision": float(rescues / max(rescues + destructive, 1)),
    }
