"""Build and audit V21-A's four independently group-isolated fold manifests.

The V20 P3 score matrix used hash folds, which were clean only for exact
normalized duplicates.  V21-A instead creates a separate GroupKFold partition
for each registered deployment regime.  The manifest is a label-safe
pre-training artifact: it uses question text, clean-complement references only
for the registered grouping definition, and document titles; it never reads
holdout or public data.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
from sklearn.model_selection import GroupKFold


REGIMES = (
    "standard_grouped",
    "law_family_ood",
    "query_template_ood",
    "near_duplicate_ood",
)
KIND = "legalir_v21_group_crossfit_manifest"
FORBIDDEN_TERMS = ("holdout", "public", "locked", "submission")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_clean_path(path: Path, label: str) -> None:
    normalized = str(path.resolve()).replace("\\", "/").casefold()
    if any(term in normalized for term in FORBIDDEN_TERMS):
        raise ValueError(f"{label} must be clean-complement-only: {path}")


def _load_grouping_code(path: Path) -> Any:
    """Load the frozen V16 registered group definitions without running it."""

    if not path.is_file():
        raise FileNotFoundError(path)
    active_scripts = Path(__file__).resolve().parent
    legacy_scripts = path.resolve().parent
    for candidate in (str(active_scripts), str(legacy_scripts)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    spec = importlib.util.spec_from_file_location("v21_legacy_v16_groups", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import V16 grouping code: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fold_rows(groups: Any, qids: list[str], fold_builder: Any) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    assignments = {qid: str(int(groups[index])) for index, qid in enumerate(qids)}
    for fold_id, (train_indices, validation_indices) in enumerate(fold_builder(groups)):
        train_qids = [qids[int(index)] for index in train_indices.tolist()]
        validation_qids = [qids[int(index)] for index in validation_indices.tolist()]
        train_groups = {assignments[qid] for qid in train_qids}
        validation_groups = {assignments[qid] for qid in validation_qids}
        overlap = sorted(train_groups & validation_groups)
        if overlap:
            raise RuntimeError(f"fold {fold_id} leaks registered group(s): {overlap[:5]}")
        rows.append({
            "fold": int(fold_id),
            "train_qids": train_qids,
            "validation_qids": validation_qids,
            "train_groups": sorted(train_groups),
            "validation_groups": sorted(validation_groups),
            "validation_overlap_groups": overlap,
            "train_qids_sha256": _sha256_bytes(_canonical(train_qids).encode("utf-8")),
            "validation_qids_sha256": _sha256_bytes(_canonical(validation_qids).encode("utf-8")),
        })
    validation_seen = [qid for row in rows for qid in row["validation_qids"]]
    if len(validation_seen) != len(qids) or set(validation_seen) != set(qids) or len(set(validation_seen)) != len(qids):
        raise RuntimeError("group folds do not assign every query exactly once to validation")
    summary = {
        "groups": len(set(assignments.values())),
        "largest_group": max(sum(value == group for value in assignments.values()) for group in set(assignments.values())),
        "folds": len(rows),
        "validation_qids": len(validation_seen),
        "zero_overlap": True,
        "assignment_sha256": _sha256_bytes(_canonical(assignments).encode("utf-8")),
    }
    return rows, assignments, summary


def _group_isolated_folds(groups: Any) -> list[tuple[np.ndarray, np.ndarray]]:
    """Mirror V16's deterministic GroupKFold(max_folds=5) protocol."""

    values = np.asarray(groups, dtype=np.int32)
    unique_groups = np.unique(values)
    if values.ndim != 1 or len(unique_groups) < 2:
        raise ValueError("Grouped validation requires one-dimensional values with at least two groups")
    indices = np.arange(len(values), dtype=np.int32)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for train, validation in GroupKFold(n_splits=min(5, len(unique_groups))).split(indices, groups=values):
        if set(values[train].tolist()) & set(values[validation].tolist()):
            raise RuntimeError("GroupKFold unexpectedly overlaps train and validation groups")
        folds.append((train.astype(np.int32), validation.astype(np.int32)))
    return folds


def build_manifest(metadata_path: Path, grouping_code: Path) -> dict[str, Any]:
    _safe_clean_path(metadata_path, "metadata")
    _safe_clean_path(grouping_code, "grouping code")
    groups_module = _load_grouping_code(grouping_code)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("kind") != "legalir_v18b_evidence_pack_manifest":
        raise ValueError("V21 group manifest requires V18b evidence-pack metadata")
    if metadata.get("split") != "clean_complement_only":
        raise ValueError("V21 group manifest requires clean_complement_only metadata")
    if bool(metadata.get("holdout_opened")) or bool(metadata.get("public_labels_used")):
        raise ValueError("metadata declares unavailable external labels")
    questions = metadata.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("metadata questions must be a non-empty list")
    qids = [str(question["qid"]) for question in questions]
    if len(qids) != len(set(qids)):
        raise ValueError("metadata qids must be unique")
    queries = [str(question["query"]) for question in questions]
    references = [[str(doc_id) for doc_id in question["references"]] for question in questions]
    if any(not query.strip() for query in queries) or any(not docs for docs in references):
        raise ValueError("metadata questions must have non-empty queries and references")
    titles_raw = metadata.get("document_titles")
    if not isinstance(titles_raw, Mapping):
        raise ValueError("metadata document_titles must be a mapping")
    # Empty title strings occurred in the original V18 pack metadata.  V16's
    # law-family definition intentionally falls back to the document id for
    # these entries; rejecting them would silently change the registered split.
    titles = {str(doc_id): str(title) for doc_id, title in titles_raw.items()}
    split = SimpleNamespace(qids=qids, questions=queries, references=references)
    derived = {
        "standard_grouped": groups_module.build_leak_groups(split),
        "law_family_ood": groups_module.law_family_groups(split, titles),
        "query_template_ood": groups_module.template_groups(queries),
        "near_duplicate_ood": groups_module.near_duplicate_groups(queries),
    }
    regimes: dict[str, Any] = {}
    for regime in REGIMES:
        if regime not in derived:
            raise ValueError(f"missing registered regime: {regime}")
        rows, assignments, summary = _fold_rows(derived[regime], qids, _group_isolated_folds)
        regimes[regime] = {"group_assignments": assignments, "folds": rows, "audit": summary}
    return {
        "kind": KIND,
        "schema_version": 1,
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "selection_performed": False,
        "metadata": str(metadata_path.resolve()),
        "metadata_sha256": _sha256_file(metadata_path),
        "grouping_code": str(grouping_code.resolve()),
        "grouping_code_sha256": _sha256_file(grouping_code),
        "questions": len(qids),
        "qids": qids,
        "qids_sha256": _sha256_bytes(_canonical(qids).encode("utf-8")),
        "regimes": regimes,
        "all_registered_regimes_zero_overlap": all(
            bool(regimes[name]["audit"]["zero_overlap"]) for name in REGIMES
        ),
        "contract": {
            "ce_adapter_fit_excludes_validation_group": True,
            "feature_fit_excludes_validation_group": True,
            "threshold_fit_excludes_validation_group": True,
            "outer_regimes_are_independent": True,
        },
    }


def _write_atomic(path: Path, payload: Mapping[str, Any], *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--grouping-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_manifest(args.metadata, args.grouping_code)
    _safe_clean_path(args.output, "output")
    _write_atomic(args.output, payload, force=args.force)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "questions": payload["questions"],
        "all_registered_regimes_zero_overlap": payload["all_registered_regimes_zero_overlap"],
        "regimes": {name: payload["regimes"][name]["audit"] for name in REGIMES},
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
