"""V23-B0: counterfactual set-edit audit, CPU-only and holdout-closed.

This audit enumerates every legal identity/one-slot/two-slot edit over the
registered c35/E3B boundary.  Labels are official clean-complement qrels,
but every edit is weighted as part of its query; edit rows are never treated
as independent observations.  A single pooled selection is produced from
four independently cross-fitted CPU diagnostic models.  No neural model is
trained and no holdout/public labels are touched.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REGIMES = (
    "standard_grouped",
    "law_family_ood",
    "query_template_ood",
    "near_duplicate_ood",
)
CONFIGS = (
    "B1024-D30",
    "B1024-D60",
    "B1536-D30",
    "B1536-D60",
    "B2304-D30",
    "B2304-D60",
)
PACK_CONFIG = "B2304-D60"
SEED = 2302
BOOTSTRAP_SAMPLES = 5000
FORBIDDEN_TERMS = ("holdout", "public", "locked", "submission")
EDIT_ORDER = {"identity": 0, "one_slot4": 1, "one_slot5": 2, "two_slot": 3}
FEATURE_NAMES = (
    "is_identity",
    "is_one_slot4",
    "is_one_slot5",
    "is_two_slot",
    "num_changes",
    "rank_mean",
    "rank_min",
    "rank_max",
    "challenger_fraction",
    "new_slot4_is_challenger",
    "new_slot5_is_challenger",
    "e3b_z_mean",
    "e3b_z_min",
    "e3b_z_max",
    "e3b_z_consensus",
    "e3b_delta_vs_slot5_mean",
    "e3b_positive_vs_slot5",
    "hidden_cosine_target_mean",
    "hidden_cosine_target_min",
    "hidden_distance_target_mean",
    "hidden_norm_delta_mean",
    "new_pair_cosine",
    "pair_cosine_delta",
    "new_pair_diversity",
    "pair_hidden_mean_norm",
    "slot4_rep_available",
    "slot5_rep_available",
    "both_new_reps_available",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe(path: Path, label: str) -> None:
    folded = str(path.resolve()).replace("\\", "/").casefold()
    if any(term in folded for term in FORBIDDEN_TERMS):
        raise ValueError(f"{label} must remain clean-complement-only: {path}")


def _read_json(path: Path, label: str, *, safe: bool = True) -> dict[str, Any]:
    if safe:
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


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _ids(value: Any, field: str, *, length: int | None = None) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a document-ID list")
    result = [str(item) for item in value]
    if length is not None and len(result) != length:
        raise ValueError(f"{field} must contain {length} IDs")
    if not result or len(result) != len(set(result)) or any(not item for item in result):
        raise ValueError(f"{field} must contain unique non-empty IDs")
    return result


def _load_metadata_label_free(path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(path, "V18b metadata")
    if payload.get("kind") != "legalir_v18b_evidence_pack_manifest" or payload.get("split") != "clean_complement_only":
        raise ValueError("metadata is not the registered clean-complement manifest")
    if payload.get("holdout_opened") is not False or payload.get("public_labels_used") is not False:
        raise ValueError("metadata does not attest unopened holdout/public labels")
    raw_questions = payload.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise ValueError("metadata.questions is missing")
    result: dict[str, dict[str, Any]] = {}
    for raw in raw_questions:
        if not isinstance(raw, Mapping):
            raise ValueError("malformed metadata question")
        qid = str(raw.get("qid", ""))
        if not qid or qid in result:
            raise ValueError("metadata has empty or duplicate qid")
        anchor = _ids(raw.get("anchor_answer"), f"{qid}.anchor_answer", length=5)
        e3b = _ids(raw.get("e3b_scoring_boundary"), f"{qid}.e3b_scoring_boundary")
        scoring = _ids(raw.get("scoring_boundary"), f"{qid}.scoring_boundary", length=7)
        unavailable = [str(value) for value in raw.get("e3b_source_unavailable", [])]
        if scoring[0] != anchor[-1] or e3b[0] != anchor[-1] or len(set(unavailable)) != len(unavailable):
            raise ValueError(f"{qid}: boundary/anchor contract violated")
        if set(unavailable) != set(scoring) - set(e3b):
            raise ValueError(f"{qid}: E3B unavailable declaration drift")
        result[qid] = {
            "qid": qid,
            "query": str(raw.get("query", "")),
            "anchor": anchor,
            "e3b": e3b,
            "unavailable": unavailable,
            "fold": int(raw.get("fold", -1)),
        }
    return result


def _load_references(path: Path, qids: set[str]) -> dict[str, set[str]]:
    payload = _read_json(path, "clean-complement qrels")
    result: dict[str, set[str]] = {}
    for raw in payload.get("questions", []):
        if not isinstance(raw, Mapping):
            raise ValueError("malformed qrel question")
        qid = str(raw.get("qid", ""))
        if qid in qids:
            refs = _ids(raw.get("references"), f"{qid}.references")
            result[qid] = set(refs)
    if set(result) != qids:
        raise ValueError("clean-complement qrel coverage differs from metadata")
    return result


def _load_p0(path: Path, metadata_path: Path, qids: set[str]) -> dict[str, list[str]]:
    report = _read_json(path, "V23-P0 report")
    if report.get("kind") != "legalir_v23_p0_two_slot_feasibility" or report.get("split") != "clean_complement_only":
        raise ValueError("V23-P0 report is not the registered clean-complement report")
    if report.get("holdout_opened") is not False or report.get("public_labels_used") is not False or report.get("training_performed") is not False:
        raise ValueError("V23-P0 report does not attest CPU-only unopened execution")
    if report.get("registered_v23_a_gate", {}).get("passed") is not True:
        raise ValueError("V23-B0 requires the passed V23-P0 feasibility gate")
    metadata_input = report.get("inputs", {}).get("metadata", {})
    if isinstance(metadata_input, Mapping) and metadata_input.get("sha256") != _sha256(metadata_path):
        raise ValueError("V23-P0 metadata provenance mismatch")
    traces = report.get("per_query_oracle_traces", {}).get("e3b_aligned_shortlist6")
    if not isinstance(traces, list):
        raise ValueError("V23-P0 lacks the selected E3B shortlist traces")
    result: dict[str, list[str]] = {}
    for raw in traces:
        if not isinstance(raw, Mapping):
            raise ValueError("malformed V23-P0 trace")
        qid = str(raw.get("qid", ""))
        pool = _ids(raw.get("pool"), f"{qid}.pool")
        if qid in result or not qid:
            raise ValueError("duplicate/empty V23-P0 qid")
        result[qid] = pool
    if set(result) != qids:
        raise ValueError("V23-P0 shortlist coverage differs from metadata")
    return result


def _load_fold_manifest(path: Path, metadata_path: Path, qids: set[str]) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, str]]]:
    payload = _read_json(path, "V21 group fold manifest")
    if payload.get("kind") != "legalir_v21_group_crossfit_manifest" or payload.get("split") != "clean_complement_only":
        raise ValueError("fold manifest is not the registered clean-complement manifest")
    if payload.get("metadata_sha256") != _sha256(metadata_path) or payload.get("all_registered_regimes_zero_overlap") is not True:
        raise ValueError("fold manifest provenance/group overlap audit failed")
    result: dict[str, dict[str, int]] = {}
    group_result: dict[str, dict[str, str]] = {}
    groups_root = payload.get("regimes", {})
    for regime in REGIMES:
        raw = groups_root.get(regime)
        if not isinstance(raw, Mapping):
            raise ValueError(f"fold manifest lacks {regime}")
        assignments = {str(key): str(value) for key, value in dict(raw.get("group_assignments", {})).items()}
        if set(assignments) != qids:
            raise ValueError(f"{regime}: group assignment coverage drift")
        fold_by_qid: dict[str, int] = {}
        for fold_raw in raw.get("folds", []):
            if not isinstance(fold_raw, Mapping):
                raise ValueError(f"{regime}: malformed fold")
            fold = int(fold_raw.get("fold", -1))
            train = {str(value) for value in fold_raw.get("train_qids", [])}
            validation = {str(value) for value in fold_raw.get("validation_qids", [])}
            if fold < 0 or not validation or train & validation:
                raise ValueError(f"{regime}/{fold}: malformed split")
            if {assignments[qid] for qid in train} & {assignments[qid] for qid in validation}:
                raise ValueError(f"{regime}/{fold}: group overlap")
            for qid in validation:
                if qid in fold_by_qid:
                    raise ValueError(f"{regime}: duplicate validation qid {qid}")
                fold_by_qid[qid] = fold
        if set(fold_by_qid) != qids:
            raise ValueError(f"{regime}: fold coverage drift")
        result[regime] = fold_by_qid
        group_result[regime] = assignments
    return result, group_result


def _load_e3b_scores(path: Path, manifest_path: Path, questions: Mapping[str, Mapping[str, Any]]) -> dict[tuple[str, str], np.ndarray]:
    manifest_sha = _sha256(manifest_path)
    expected = {(qid, doc, config) for qid, row in questions.items() for doc in row["e3b"] for config in CONFIGS}
    seen: set[tuple[str, str, str]] = set()
    values: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    run_signatures: set[str] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                raise ValueError(f"E3B score line {line_number} is not an object")
            qid, doc, config = str(raw.get("qid", "")), str(raw.get("doc_id", "")), str(raw.get("config", ""))
            key = (qid, doc, config)
            if key not in expected or key in seen or raw.get("manifest_sha256") != manifest_sha:
                raise ValueError(f"E3B score provenance/key error at line {line_number}: {key}")
            seen.add(key)
            values[(qid, doc)][config] = float(raw.get("score"))
            run_signatures.add(str(raw.get("run_signature", "")))
    if seen != expected or len(run_signatures) != 1:
        raise ValueError("E3B score key set or run signature drift")
    return {key: np.asarray([values[key][config] for config in CONFIGS], dtype=np.float64) for key in values}


def _load_rep_cache(path: Path, questions: Mapping[str, Mapping[str, Any]], slot4_force: set[str]) -> dict[tuple[str, str], dict[str, Any]]:
    provenance = _read_json(path.with_suffix(path.suffix + ".provenance.json"), "V23-A representation provenance")
    if provenance.get("kind") != "legalir_v23_a_frozen_bag_representations" or provenance.get("pack_config") != PACK_CONFIG:
        raise ValueError("representation cache kind/pack config drift")
    if provenance.get("labels_loaded") is not False or provenance.get("holdout_opened") is not False or provenance.get("public_labels_used") is not False:
        raise ValueError("representation cache does not attest label-free preparation")
    data = np.load(path, allow_pickle=False)
    embeddings = data["embeddings"]
    mask = data["mask"]
    raw_keys = data["keys"].tolist()
    if embeddings.ndim != 3 or embeddings.shape[1] != 5 or mask.shape != embeddings.shape[:2] or len(raw_keys) != embeddings.shape[0]:
        raise ValueError("representation cache dimensions are malformed")
    keys = [tuple(str(part) for part in str(value).split("\t", 1)) for value in raw_keys]
    expected = {(qid, doc) for qid, row in questions.items() for doc in row["e3b"]}
    expected |= {(qid, row["anchor"][3]) for qid, row in questions.items() if qid not in slot4_force}
    if set(keys) != expected or len(keys) != len(set(keys)):
        raise ValueError("representation cache key set differs from c35/E3B contract")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for index, key in enumerate(keys):
        valid = mask[index].astype(bool)
        if not valid.any():
            raise ValueError(f"empty representation bag: {key}")
        vec = np.asarray(embeddings[index, valid], dtype=np.float32).mean(axis=0)
        result[key] = {
            "vector": vec,
            "norm": float(np.linalg.norm(vec)),
            "blocks": int(valid.sum()),
            "available": True,
        }
    return result


def _cosine(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 0.0
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _generate_edits(qid: str, anchor: Sequence[str], pool: Sequence[str], force_identity: bool) -> list[dict[str, Any]]:
    old4, old5 = str(anchor[3]), str(anchor[4])
    universe = list(dict.fromkeys([old4, old5, *[str(doc) for doc in pool]]))
    if any(doc in anchor[:3] for doc in universe) or len(universe) != len(set(universe)):
        raise ValueError(f"{qid}: candidate universe overlaps frozen top-3 or has duplicates")
    if force_identity:
        return [{"qid": qid, "edit_id": f"{qid}:identity", "edit_type": "identity", "changed_slots": [], "new_pair": [old4, old5]}]
    edits: list[dict[str, Any]] = [{"qid": qid, "edit_id": f"{qid}:identity", "edit_type": "identity", "changed_slots": [], "new_pair": [old4, old5]}]
    challengers = [doc for doc in universe if doc not in {old4, old5}]
    for doc in challengers:
        edits.append({"qid": qid, "edit_id": f"{qid}:s4:{doc}", "edit_type": "one_slot4", "changed_slots": [4], "new_pair": [doc, old5]})
        edits.append({"qid": qid, "edit_id": f"{qid}:s5:{doc}", "edit_type": "one_slot5", "changed_slots": [5], "new_pair": [old4, doc]})
    for new4, new5 in itertools.permutations(universe, 2):
        if new4 == old4 or new5 == old5:
            continue
        edits.append({"qid": qid, "edit_id": f"{qid}:s45:{new4}:{new5}", "edit_type": "two_slot", "changed_slots": [4, 5], "new_pair": [new4, new5]})
    if len({row["edit_id"] for row in edits}) != len(edits):
        raise ValueError(f"{qid}: duplicate edit generated")
    return edits


def _doc_zscores(qid: str, docs: Sequence[str], score_vectors: Mapping[tuple[str, str], np.ndarray]) -> dict[str, np.ndarray]:
    available = [score_vectors[(qid, doc)] for doc in docs if (qid, doc) in score_vectors]
    if not available:
        return {}
    matrix = np.vstack(available)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return {doc: (score_vectors[(qid, doc)] - mean) / scale for doc in docs if (qid, doc) in score_vectors}


def _edit_features(
    row: Mapping[str, Any],
    question: Mapping[str, Any],
    zscores: Mapping[str, np.ndarray],
    score_vectors: Mapping[tuple[str, str], np.ndarray],
    reps: Mapping[tuple[str, str], Mapping[str, Any]],
    rank_by_doc: Mapping[str, int],
) -> tuple[list[float], dict[str, Any]]:
    qid = str(row["qid"])
    old4, old5 = question["anchor"][3], question["anchor"][4]
    new4, new5 = row["new_pair"]
    changed = [(4, new4, old4), (5, new5, old5)]
    changed = [(slot, new_doc, old_doc) for slot, new_doc, old_doc in changed if new_doc != old_doc]
    changed_docs = [new_doc for _, new_doc, _ in changed]
    rank_values = [float(rank_by_doc.get(doc, 0)) for doc in changed_docs]
    challenger_values = [doc not in {old4, old5} for doc in changed_docs]
    e3b_z_values: list[float] = []
    e3b_consensus: list[float] = []
    e3b_delta_values: list[float] = []
    e3b_positive_values: list[float] = []
    cosine_values: list[float] = []
    distance_values: list[float] = []
    norm_delta_values: list[float] = []
    sources: list[str] = []
    for slot, new_doc, old_doc in changed:
        source = "e3b_challenger" if new_doc not in {old4, old5} else f"incumbent_slot{4 if new_doc == old4 else 5}"
        sources.append(source)
        zvec = zscores.get(new_doc)
        if zvec is not None:
            e3b_z_values.append(float(np.mean(zvec)))
            e3b_consensus.append(float(np.mean(zvec > 0.0)))
            candidate_scores = score_vectors[(qid, new_doc)]
            baseline_scores = score_vectors.get((qid, old5))
            if baseline_scores is not None:
                e3b_delta_values.append(float(np.mean(candidate_scores - baseline_scores)))
                e3b_positive_values.append(float(np.mean(candidate_scores > baseline_scores)))
        candidate_rep = reps.get((qid, new_doc))
        target_rep = reps.get((qid, old4 if slot == 4 else old5))
        candidate_vec = candidate_rep["vector"] if candidate_rep else None
        target_vec = target_rep["vector"] if target_rep else None
        if candidate_vec is not None and target_vec is not None:
            cosine_values.append(_cosine(candidate_vec, target_vec))
            distance_values.append(float(np.linalg.norm(candidate_vec - target_vec)))
            norm_delta_values.append(abs(float(candidate_rep["norm"]) - float(target_rep["norm"])))
    old4_rep = reps.get((qid, old4))
    old5_rep = reps.get((qid, old5))
    new4_rep = reps.get((qid, new4))
    new5_rep = reps.get((qid, new5))
    old_pair_cos = _cosine(old4_rep["vector"], old5_rep["vector"]) if old4_rep and old5_rep else 0.0
    new_pair_cos = _cosine(new4_rep["vector"], new5_rep["vector"]) if new4_rep and new5_rep else 0.0
    pair_vecs = [rep["vector"] for rep in (new4_rep, new5_rep) if rep]
    pair_mean_norm = float(np.linalg.norm(np.mean(np.vstack(pair_vecs), axis=0))) if pair_vecs else 0.0
    edit_type = str(row["edit_type"])
    values = [
        float(edit_type == "identity"),
        float(edit_type == "one_slot4"),
        float(edit_type == "one_slot5"),
        float(edit_type == "two_slot"),
        float(len(changed)),
        float(statistics.fmean(rank_values) / 6.0) if rank_values else 0.0,
        float(min(rank_values) / 6.0) if rank_values else 0.0,
        float(max(rank_values) / 6.0) if rank_values else 0.0,
        float(statistics.fmean(challenger_values)) if challenger_values else 0.0,
        float(new4 not in {old4, old5}),
        float(new5 not in {old4, old5}),
        float(statistics.fmean(e3b_z_values)) if e3b_z_values else 0.0,
        float(min(e3b_z_values)) if e3b_z_values else 0.0,
        float(max(e3b_z_values)) if e3b_z_values else 0.0,
        float(statistics.fmean(e3b_consensus)) if e3b_consensus else 0.0,
        float(statistics.fmean(e3b_delta_values)) if e3b_delta_values else 0.0,
        float(statistics.fmean(e3b_positive_values)) if e3b_positive_values else 0.0,
        float(statistics.fmean(cosine_values)) if cosine_values else 0.0,
        float(min(cosine_values)) if cosine_values else 0.0,
        float(statistics.fmean(distance_values)) if distance_values else 0.0,
        float(statistics.fmean(norm_delta_values)) if norm_delta_values else 0.0,
        float(new_pair_cos),
        float(new_pair_cos - old_pair_cos),
        float(1.0 - new_pair_cos) if pair_vecs else 0.0,
        pair_mean_norm,
        float(new4_rep is not None),
        float(new5_rep is not None),
        float(new4_rep is not None and new5_rep is not None),
    ]
    if len(values) != len(FEATURE_NAMES):
        raise AssertionError("feature schema count drift")
    diagnostics = {
        "candidate_ranks": [int(rank_by_doc.get(doc, 0)) for doc in changed_docs],
        "candidate_sources": sources,
        "e3b_consensus": float(statistics.fmean(e3b_consensus)) if e3b_consensus else None,
        "hidden_cosine_target": float(statistics.fmean(cosine_values)) if cosine_values else None,
    }
    return values, diagnostics


def _recall(pair: Sequence[str], anchor: Sequence[str], refs: set[str]) -> float:
    return len((set(anchor[:3]) | {str(pair[0]), str(pair[1])}) & refs) / len(refs)


def _label_edits(edits_by_qid: Mapping[str, Sequence[dict[str, Any]]], questions: Mapping[str, Mapping[str, Any]], references: Mapping[str, set[str]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    per_query: dict[str, dict[str, Any]] = {}
    for qid in sorted(edits_by_qid):
        anchor = questions[qid]["anchor"]
        refs = references[qid]
        baseline = _recall([anchor[3], anchor[4]], anchor, refs)
        qrows: list[dict[str, Any]] = []
        for edit in edits_by_qid[qid]:
            after = _recall(edit["new_pair"], anchor, refs)
            delta = after - baseline
            outcome = "improve" if delta > 1e-12 else "harm" if delta < -1e-12 else "neutral"
            row = {**edit, "baseline_recall": baseline, "after_recall": after, "delta_recall": delta, "outcome": outcome, "query_weight": 1.0 / len(edits_by_qid[qid])}
            qrows.append(row)
            rows.append(row)
        best = max(qrows, key=lambda item: (float(item["delta_recall"]), -EDIT_ORDER[str(item["edit_type"])]))
        per_query[qid] = {
            "baseline_recall": baseline,
            "oracle_edit_beam_recall": float(best["after_recall"]),
            "oracle_edit_beam_delta": float(best["delta_recall"]),
            "oracle_edit_type": str(best["edit_type"]),
            "valid_edits": len(qrows),
            "force_identity": len(qrows) == 1,
        }
    return rows, per_query


def _fit_cpu_models(
    all_rows: Sequence[Mapping[str, Any]],
    rows_by_qid: Mapping[str, Sequence[Mapping[str, Any]]],
    fold_by_qid: Mapping[str, int],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression

    feature_matrix = np.asarray([row["features"] for row in all_rows], dtype=np.float64)
    if not np.isfinite(feature_matrix).all():
        raise ValueError("counterfactual feature matrix contains a non-finite value")
    targets = np.asarray([{ "harm": 0, "neutral": 1, "improve": 2 }[str(row["outcome"])] for row in all_rows], dtype=np.int64)
    row_index = {str(row["edit_id"]): index for index, row in enumerate(all_rows)}
    probabilities: dict[str, np.ndarray] = {}
    fold_details: dict[str, Any] = {}
    for fold in sorted(set(fold_by_qid.values())):
        train_qids = sorted(qid for qid, value in fold_by_qid.items() if value != fold)
        validation_qids = sorted(qid for qid, value in fold_by_qid.items() if value == fold)
        train_indices = [row_index[str(row["edit_id"])] for qid in train_qids for row in rows_by_qid[qid]]
        validation_indices = [row_index[str(row["edit_id"])] for qid in validation_qids for row in rows_by_qid[qid]]
        if len(set(targets[train_indices])) != 3:
            raise ValueError(f"CPU cross-fit fold {fold} lacks one of the three edit states")
        weights = np.asarray([1.0 / len(rows_by_qid[qid]) for qid in train_qids for _ in rows_by_qid[qid]], dtype=np.float64)
        if not np.allclose(np.asarray([sum(1.0 / len(rows_by_qid[qid]) for _ in rows_by_qid[qid]) for qid in train_qids]), 1.0):
            raise AssertionError("query weight does not sum to one")
        # Stable two-pass weighted StandardScaler.  sklearn's incremental
        # weighted variance can round a near-constant feature below zero when
        # thousands of edit rows share a per-query weight; this is the same
        # weighted standardization, evaluated in float64 without cancellation.
        train_features = feature_matrix[train_indices]
        weight_sum = float(weights.sum())
        weighted_mean = np.sum(train_features * weights[:, None], axis=0) / weight_sum
        weighted_var = np.sum(((train_features - weighted_mean) ** 2) * weights[:, None], axis=0) / weight_sum
        weighted_scale = np.sqrt(np.maximum(weighted_var, 1e-12))

        def standardize(values: np.ndarray) -> np.ndarray:
            return (values - weighted_mean) / weighted_scale

        classifier = LogisticRegression(solver="lbfgs", C=1.0, max_iter=500, class_weight=None, random_state=SEED)
        classifier.fit(standardize(train_features), targets[train_indices], sample_weight=weights)
        predicted = classifier.predict_proba(standardize(feature_matrix[validation_indices]))
        class_to_column = {int(value): index for index, value in enumerate(classifier.classes_)}
        for index, vector in zip(validation_indices, predicted, strict=True):
            probabilities[str(all_rows[index]["edit_id"])] = np.asarray([float(vector[class_to_column[0]]), float(vector[class_to_column[1]]), float(vector[class_to_column[2]])], dtype=np.float64)
        fold_details[str(fold)] = {
            "train_queries": len(train_qids),
            "validation_queries": len(validation_qids),
            "train_edits": len(train_indices),
            "validation_edits": len(validation_indices),
            "group_overlap": False,
            "classes": ["harm", "neutral", "improve"],
        }
    if set(probabilities) != set(row_index):
        raise ValueError("CPU cross-fit probability coverage drift")
    return probabilities, {"folds": fold_details, "model": "stable two-pass weighted StandardScaler + multinomial LogisticRegression", "features": list(FEATURE_NAMES)}


def _select_one(qid: str, rows: Sequence[Mapping[str, Any]], probabilities: Mapping[str, np.ndarray]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if row["edit_type"] == "identity":
            continue
        probs = probabilities[str(row["edit_id"])]
        p_harm, p_neutral, p_improve = (float(value) for value in probs)
        if p_harm > 0.15 or p_improve - p_harm < 0.05:
            continue
        candidates.append({
            **row,
            "p_harm": p_harm,
            "p_neutral": p_neutral,
            "p_improve": p_improve,
            "lower_confidence_gain": p_improve - p_harm,
            "gain_rank_score": p_improve - 2.0 * p_harm,
        })
    if not candidates:
        identity = next(row for row in rows if row["edit_type"] == "identity")
        return {**identity, "accepted": False, "p_harm": None, "p_neutral": None, "p_improve": None, "lower_confidence_gain": 0.0, "gain_rank_score": 0.0}
    return max(candidates, key=lambda row: (float(row["gain_rank_score"]), float(row["lower_confidence_gain"]), -EDIT_ORDER[str(row["edit_type"])], str(row["edit_id"]))) | {"accepted": True}


def _metrics(qids: Sequence[str], selected: Mapping[str, Mapping[str, Any]], fold_by_qid: Mapping[str, int]) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    before = {qid: float(selected[qid]["baseline_recall"]) for qid in qids}
    after = {qid: float(selected[qid]["after_recall"]) for qid in qids}
    for fold in sorted(set(fold_by_qid.values())):
        subset = [qid for qid in qids if fold_by_qid[qid] == fold]
        base = statistics.fmean(before[qid] for qid in subset)
        candidate = statistics.fmean(after[qid] for qid in subset)
        folds.append({"fold": fold, "queries": len(subset), "baseline_recall": base, "recall": candidate, "gain": candidate - base})
    base = statistics.fmean(before.values())
    candidate = statistics.fmean(after.values())
    robust_base = base - 0.25 * statistics.pstdev(row["baseline_recall"] for row in folds)
    robust_candidate = candidate - 0.25 * statistics.pstdev(row["recall"] for row in folds)
    changed = [selected[qid] for qid in qids if selected[qid]["accepted"]]
    improved = sum(float(row["delta_recall"]) > 1e-12 for row in changed)
    worsened = sum(float(row["delta_recall"]) < -1e-12 for row in changed)
    neutral = sum(abs(float(row["delta_recall"])) <= 1e-12 for row in changed)
    precision = improved / (improved + worsened) if improved + worsened else 0.0
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
        "changed_queries": len(changed),
        "improved_queries": improved,
        "worsened_queries": worsened,
        "neutral_changed_queries": neutral,
        "precision": precision,
        "net_query_equivalents": improved - worsened,
    }


def _bootstrap(qids: Sequence[str], selected: Mapping[str, Mapping[str, Any]], samples: int = BOOTSTRAP_SAMPLES) -> dict[str, Any]:
    rng = random.Random(SEED)
    values: list[float] = []
    deltas = [float(selected[qid]["delta_recall"]) for qid in qids]
    for _ in range(samples):
        values.append(statistics.fmean(deltas[rng.randrange(len(deltas))] for _ in qids))
    values.sort()
    low_index = max(0, math.floor(0.025 * (len(values) - 1)))
    high_index = min(len(values) - 1, math.ceil(0.975 * (len(values) - 1)))
    return {"samples": samples, "mean": statistics.fmean(values), "lower_95": values[low_index], "upper_95": values[high_index], "unit": "query_resampling"}


def _top_counts(rows: Sequence[Mapping[str, Any]], key_fn: Any, *, limit: int = 20) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[str(key_fn(row))] += 1
    return {key: count for key, count in counts.most_common(limit)}


def _diagnostics(rows: Sequence[Mapping[str, Any]], selected: Mapping[str, Mapping[str, Any]], groups: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    harm_rows = [row for row in rows if row["outcome"] == "harm"]
    selected_changed = [row for row in selected.values() if row["accepted"]]
    selected_harm = [row for row in selected_changed if row["outcome"] == "harm"]
    bucket = lambda value: "none" if value is None else "0.00-0.49" if float(value) < 0.5 else "0.50-0.99" if float(value) < 1.0 else "1.00"
    cosine_bucket = lambda value: "none" if value is None else "<0.25" if float(value) < 0.25 else "0.25-0.49" if float(value) < 0.5 else "0.50-0.74" if float(value) < 0.75 else ">=0.75"
    harm_by_type = Counter(str(row["edit_type"]) for row in harm_rows)
    selected_by_type = Counter(str(row["edit_type"]) for row in selected_changed)
    group_stats: dict[str, Any] = {}
    for regime in REGIMES:
        assignment = groups[regime]
        per_group: dict[str, dict[str, int]] = defaultdict(lambda: {"harm_edits": 0, "all_edits": 0, "selected_harm_queries": 0})
        for row in rows:
            group = assignment[str(row["qid"])]
            per_group[group]["all_edits"] += 1
            if row["outcome"] == "harm":
                per_group[group]["harm_edits"] += 1
        for row in selected_harm:
            per_group[assignment[str(row["qid"])]] ["selected_harm_queries"] += 1
        group_stats[regime] = dict(sorted(per_group.items(), key=lambda item: (-item[1]["selected_harm_queries"], -item[1]["harm_edits"], item[0]))[:20])
    return {
        "all_harm_edits_by_type": dict(harm_by_type),
        "all_harm_edits_by_changed_slot_count": _top_counts(harm_rows, lambda row: len(row["changed_slots"])),
        "all_harm_edits_by_candidate_rank_signature": _top_counts(harm_rows, lambda row: ",".join(str(value) for value in row.get("candidate_ranks", [])) or "identity"),
        "all_harm_edits_by_source_signature": _top_counts(harm_rows, lambda row: "+".join(row.get("candidate_sources", [])) or "identity"),
        "all_harm_edits_by_e3b_consensus": _top_counts(harm_rows, lambda row: bucket(row.get("e3b_consensus"))),
        "all_harm_edits_by_hidden_cosine_target": _top_counts(harm_rows, lambda row: cosine_bucket(row.get("hidden_cosine_target"))),
        "selected_changes_by_type": dict(selected_by_type),
        "selected_harm_by_type": dict(Counter(str(row["edit_type"]) for row in selected_harm)),
        "selected_harm_by_changed_slot_count": _top_counts(selected_harm, lambda row: len(row["changed_slots"])),
        "group_diagnostics_top20": group_stats,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    pre = _read_json(args.pre_registration, "V23-B0 pre-registration")
    if pre.get("kind") != "legalir_v23_b0_counterfactual_set_edit_audit_preregistration" or pre.get("split") != "clean_complement_only":
        raise ValueError("invalid V23-B0 pre-registration")
    if pre.get("holdout_opened") is not False or pre.get("public_labels_used") is not False:
        raise ValueError("V23-B0 pre-registration must keep holdout/public labels closed")
    if len(FEATURE_NAMES) != int(pre["model"]["feature_count"]):
        raise ValueError("feature count differs from the resolved pre-registration")
    questions = _load_metadata_label_free(args.metadata)
    qids = set(questions)
    pools = _load_p0(args.p0_report, args.metadata, qids)
    folds, group_assignments = _load_fold_manifest(args.fold_manifest, args.metadata, qids)
    slot4_materialization = _read_json(args.slot4_report, "slot-4 materialization report")
    if slot4_materialization.get("labels_loaded") is not False or slot4_materialization.get("holdout_opened") is not False or slot4_materialization.get("public_labels_used") is not False:
        raise ValueError("slot-4 materialization is not label-free")
    slot4_force = {str(value) for value in slot4_materialization.get("slot4_force_abstain_qids", [])}
    if slot4_force != {"63986", "98186"}:
        raise ValueError("slot-4 force-abstain declaration drift")
    e3b_scores = _load_e3b_scores(args.e3b_scores, args.e3b_manifest, questions)
    reps = _load_rep_cache(args.representation_cache, questions, slot4_force)

    # This is the label-free preflight boundary. Only now are clean-complement qrels read.
    references = _load_references(args.metadata, qids)
    edits_by_qid: dict[str, list[dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []
    for qid in sorted(qids):
        pool = pools[qid]
        if any(doc not in questions[qid]["e3b"] or doc in questions[qid]["unavailable"] for doc in pool):
            raise ValueError(f"{qid}: P0 pool is not an E3B-aligned available shortlist")
        force_identity = qid in slot4_force
        edits = _generate_edits(qid, questions[qid]["anchor"], pool, force_identity)
        zscores = _doc_zscores(qid, [questions[qid]["anchor"][4], *pool], e3b_scores)
        rank_by_doc = {doc: index + 1 for index, doc in enumerate(pool)}
        rank_by_doc[questions[qid]["anchor"][3]] = 0
        rank_by_doc[questions[qid]["anchor"][4]] = 0
        enriched: list[dict[str, Any]] = []
        for edit in edits:
            features, diagnostic = _edit_features(edit, questions[qid], zscores, e3b_scores, reps, rank_by_doc)
            enriched.append({**edit, "features": features, **diagnostic})
        if not math.isclose(sum(float(row["query_weight"]) if "query_weight" in row else 1.0 / len(enriched) for row in enriched), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError(f"{qid}: query edit weights do not sum to one")
        edits_by_qid[qid] = enriched
    labelled_rows, oracle = _label_edits(edits_by_qid, questions, references)
    rows_by_qid = {qid: [row for row in labelled_rows if str(row["qid"]) == qid] for qid in sorted(qids)}
    # Re-assert the query-level weighting after labels are attached.
    for qid, rows in rows_by_qid.items():
        if not math.isclose(sum(float(row["query_weight"]) for row in rows), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError(f"{qid}: query weights do not sum to one")

    regime_probabilities: dict[str, dict[str, np.ndarray]] = {}
    cross_fit: dict[str, Any] = {}
    for regime in REGIMES:
        probs, detail = _fit_cpu_models(labelled_rows, rows_by_qid, folds[regime])
        regime_probabilities[regime] = probs
        cross_fit[regime] = detail
    pooled_probabilities: dict[str, np.ndarray] = {}
    for row in labelled_rows:
        edit_id = str(row["edit_id"])
        pooled_probabilities[edit_id] = np.mean([regime_probabilities[regime][edit_id] for regime in REGIMES], axis=0)
    selected: dict[str, dict[str, Any]] = {}
    for qid in sorted(qids):
        selected[qid] = _select_one(qid, rows_by_qid[qid], pooled_probabilities)

    overall = _metrics(sorted(qids), selected, folds["standard_grouped"])
    regime_metrics = {regime: _metrics(sorted(qids), selected, folds[regime]) for regime in REGIMES}
    improved = overall["improved_queries"]
    worsened = overall["worsened_queries"]
    changed = overall["changed_queries"]
    precision = overall["precision"]
    oracle_deltas = [float(oracle[qid]["oracle_edit_beam_delta"]) for qid in sorted(qids)]
    oracle_gain = statistics.fmean(oracle_deltas)
    gate = {
        "improved_queries_at_least_18": improved >= 18,
        "worsened_queries_at_most_2": worsened <= 2,
        "precision_at_least_0_90": precision >= 0.90,
        "net_query_equivalents_at_least_16": overall["net_query_equivalents"] >= 16,
        "neutral_selected_at_most_4": overall["neutral_changed_queries"] <= 4,
        "four_regimes_non_negative": all(regime_metrics[regime]["non_negative"] for regime in REGIMES),
        "oracle_edit_beam_gain_at_least_0_025": oracle_gain >= 0.025,
    }

    edit_output = []
    for row in labelled_rows:
        edit_output.append({
            "qid": row["qid"],
            "edit_id": row["edit_id"],
            "edit_type": row["edit_type"],
            "changed_slots": row["changed_slots"],
            "new_pair": row["new_pair"],
            "candidate_ranks": row.get("candidate_ranks", []),
            "candidate_sources": row.get("candidate_sources", []),
            "e3b_consensus": row.get("e3b_consensus"),
            "hidden_cosine_target": row.get("hidden_cosine_target"),
            "query_weight": row["query_weight"],
            "baseline_recall": row["baseline_recall"],
            "after_recall": row["after_recall"],
            "delta_recall": row["delta_recall"],
            "outcome": row["outcome"],
        })
    selected_output = []
    for qid in sorted(selected):
        row = selected[qid]
        selected_output.append({
            "qid": qid,
            "edit_id": row["edit_id"],
            "edit_type": row["edit_type"],
            "new_pair": row["new_pair"],
            "accepted": row["accepted"],
            "outcome": row["outcome"],
            "delta_recall": row["delta_recall"],
            "p_harm": row["p_harm"],
            "p_neutral": row["p_neutral"],
            "p_improve": row["p_improve"],
            "lower_confidence_gain": row["lower_confidence_gain"],
            "gain_rank_score": row["gain_rank_score"],
            "candidate_ranks": row.get("candidate_ranks", []),
            "candidate_sources": row.get("candidate_sources", []),
        })
    report = {
        "kind": "legalir_v23_b0_counterfactual_set_edit_audit",
        "schema_version": 1,
        "stage": "V23-B0_complete_cpu_only_no_v23_b_training_no_holdout",
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "clean_complement_qrels_loaded_after_label_free_preflight": True,
        "neural_training_performed": False,
        "cpu_baseline_fit_performed": True,
        "pre_registration": {"path": str(args.pre_registration.resolve()), "sha256": _sha256(args.pre_registration), "config": pre},
        "resolved_hashes": {
            "config_sha256": _sha256(args.pre_registration),
            "model_graph_sha256": _json_hash(pre["model"]["model_graph"]),
            "tensor_shapes_sha256": _json_hash(pre["model"]["tensor_shapes"]),
            "feature_schema_sha256": _json_hash(list(FEATURE_NAMES)),
        },
        "inputs": {
            "metadata": {"path": str(args.metadata.resolve()), "sha256": _sha256(args.metadata)},
            "p0_report": {"path": str(args.p0_report.resolve()), "sha256": _sha256(args.p0_report)},
            "fold_manifest": {"path": str(args.fold_manifest.resolve()), "sha256": _sha256(args.fold_manifest)},
            "e3b_manifest": {"path": str(args.e3b_manifest.resolve()), "sha256": _sha256(args.e3b_manifest)},
            "e3b_scores": {"path": str(args.e3b_scores.resolve()), "sha256": _sha256(args.e3b_scores)},
            "representation_cache": {"path": str(args.representation_cache.resolve()), "sha256": _sha256(args.representation_cache)},
            "slot4_report": {"path": str(args.slot4_report.resolve()), "sha256": _sha256(args.slot4_report)},
        },
        "candidate_contract": {
            "queries": len(qids),
            "frozen_slots": [1, 2, 3],
            "mutable_slots": [4, 5],
            "pool": "V23-P0 e3b_aligned_shortlist6",
            "slot4_force_abstain_qids": sorted(slot4_force),
            "no_challenger_substitution": True,
            "candidate_pool_sizes": dict(sorted(Counter(len(pools[qid]) for qid in qids).items())),
        },
        "edit_space": {
            "edit_types": ["identity", "one_slot4", "one_slot5", "two_slot"],
            "total_edits": len(labelled_rows),
            "edits_by_type": dict(Counter(str(row["edit_type"]) for row in labelled_rows)),
            "identity_edits": len(qids),
            "all_variants_same_query_fold": True,
            "query_total_weight_min": min(sum(float(row["query_weight"]) for row in rows) for rows in rows_by_qid.values()),
            "query_total_weight_max": max(sum(float(row["query_weight"]) for row in rows) for rows in rows_by_qid.values()),
        },
        "oracle_edit_beam": {
            "gain": oracle_gain,
            "recall": statistics.fmean(float(oracle[qid]["oracle_edit_beam_recall"]) for qid in sorted(qids)),
            "beneficial_queries": sum(float(oracle[qid]["oracle_edit_beam_delta"]) > 1e-12 for qid in qids),
            "harmful_queries": sum(float(oracle[qid]["oracle_edit_beam_delta"]) < -1e-12 for qid in qids),
            "one_slot_best": sum(oracle[qid]["oracle_edit_type"] in {"one_slot4", "one_slot5"} for qid in qids),
            "two_slot_best": sum(oracle[qid]["oracle_edit_type"] == "two_slot" for qid in qids),
        },
        "cross_fit": cross_fit,
        "selection": {
            "regime_combination": "arithmetic mean of four cross-fitted regime probabilities; no independent veto",
            "harm_rejector_max_probability": 0.15,
            "gain_rank_score": "p_improve - 2 * p_harm",
            "lower_confidence_gain_minimum": 0.05,
            "accepted_queries": changed,
            "improved_queries": improved,
            "worsened_queries": worsened,
            "neutral_changed_queries": overall["neutral_changed_queries"],
            "precision": precision,
            "net_query_equivalents": overall["net_query_equivalents"],
        },
        "overall_metrics": overall,
        "regime_metrics": regime_metrics,
        "paired_bootstrap": _bootstrap(sorted(qids), selected),
        "diagnostics": _diagnostics(labelled_rows, selected, group_assignments),
        "promotion_gate": {"checks": gate, "passed": all(gate.values())},
        "decision": {
            "holdout_may_open": False,
            "if_passed": "freeze one V23-B predictor-rejector recipe and register its neural model before training",
            "if_failed": "close V23-B0 and the small-head/E3B edit family; do not train V23-B, open V24/K32, open holdout, or submit",
        },
    }
    _atomic_json(args.output_dir / "v23_b0_report.json", report)
    _atomic_jsonl(args.output_dir / "v23_b0_edit_rows.jsonl", edit_output)
    _atomic_jsonl(args.output_dir / "v23_b0_selected_edits.jsonl", selected_output)
    _atomic_jsonl(args.output_dir / "v23_b0_oracle_per_query.jsonl", [{"qid": qid, **oracle[qid]} for qid in sorted(oracle)])
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--p0-report", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--e3b-manifest", type=Path, required=True)
    parser.add_argument("--e3b-scores", type=Path, required=True)
    parser.add_argument("--representation-cache", type=Path, required=True)
    parser.add_argument("--slot4-report", type=Path, required=True)
    parser.add_argument("--pre-registration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    result = run(arguments)
    print(json.dumps({"output": str((arguments.output_dir / "v23_b0_report.json").resolve()), "promotion_passed": result["promotion_gate"]["passed"], "holdout_opened": False, "stage": result["stage"]}, ensure_ascii=False))
