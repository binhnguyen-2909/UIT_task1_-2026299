"""Fail-closed model compliance checks for DSC 2026 offline pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional


PARAMETER_LIMIT = 4_000_000_000


def _registered_path_candidates(registry_path: Path, value: str) -> list[Path]:
    """Resolve registry paths written either from repo root or registry dir.

    Older local registries used ``models/foo`` while living inside ``models/``;
    resolving only relative to the registry directory turned that into
    ``models/models/foo``. Keep both conventions valid, but still require the
    selected path to be an existing local directory.
    """
    path = Path(value)
    if path.is_absolute():
        return [path]
    registry_path = registry_path.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        registry_path.parent / path,
        registry_path.parent.parent / path,
        repo_root / path,
        Path.cwd() / path,
    ]
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def validate_model_plan(
    registry_path: Optional[Path],
    selected: Dict[str, Optional[str]],
) -> Dict[str, object]:
    """Validate local paths, BTC approval flags and summed model parameters.

    ``selected`` maps component names (embedding/reranker/generator) to local
    model directories.  No neural model is allowed when the registry is absent.
    Models that share one registry ``id`` are counted once.
    """
    active = {component: path for component, path in selected.items() if path}
    if not active:
        return {"models": [], "total_parameters": 0, "limit": PARAMETER_LIMIT}
    if registry_path is None:
        raise ValueError("--model-registry is required when a neural model is enabled")
    registry_path = Path(registry_path)
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    records = data.get("models") if isinstance(data, dict) else None
    if not isinstance(records, list):
        raise ValueError("Model registry must contain a 'models' list")

    selected_records = []
    for component, local_path in active.items():
        resolved = Path(str(local_path)).resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"{component} model is not a local directory: {resolved}")
        matches = []
        for record in records:
            if not isinstance(record, dict) or record.get("component") != component:
                continue
            registered_value = str(record.get("local_path", ""))
            candidates = _registered_path_candidates(registry_path, registered_value)
            if any(candidate == resolved for candidate in candidates):
                matches.append(record)
        if not matches:
            marker_path = resolved / "derived_model.json"
            if marker_path.is_file():
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                if marker.get("schema") != 1 or marker.get("component") != component:
                    raise ValueError(
                        f"Invalid derived model provenance for {component}: {marker_path}"
                    )
                base_model_id = marker.get("base_model_id")
                training_method = marker.get("training_method")
                if not isinstance(training_method, str) or not training_method.strip():
                    raise ValueError(f"Derived model is missing training_method: {marker_path}")
                matches = [
                    record
                    for record in records
                    if isinstance(record, dict)
                    and record.get("component") == component
                    and record.get("id") == base_model_id
                ]
        if len(matches) != 1:
            raise ValueError(f"{component} model is not uniquely registered: {resolved}")
        record = matches[0]
        if not isinstance(record.get("id"), str) or not record["id"].strip():
            raise ValueError(f"{component} registry record is missing a model id")
        if record.get("registered") is not True or record.get("btc_approved") is not True:
            raise ValueError(f"{record.get('id', resolved)} is not registered and BTC-approved")
        if not isinstance(record.get("parameters"), int) or record["parameters"] <= 0:
            raise ValueError(f"{record.get('id', resolved)} has invalid parameter count")
        if not isinstance(record.get("license"), str) or not record["license"].strip():
            raise ValueError(f"{record.get('id', resolved)} is missing license evidence")
        if not isinstance(record.get("approval_evidence"), str) or not record["approval_evidence"].strip():
            raise ValueError(f"{record.get('id', resolved)} is missing BTC approval evidence")
        selected_records.append(record)

    unique = {str(record["id"]): record for record in selected_records}
    total = sum(record["parameters"] for record in unique.values())
    limit = int(data.get("parameter_limit", PARAMETER_LIMIT))
    if limit > PARAMETER_LIMIT:
        raise ValueError("Registry parameter_limit may not exceed 4,000,000,000")
    if total >= limit:
        raise ValueError(f"Selected models total {total:,} parameters; must be below {limit:,}")
    return {
        "models": sorted(unique),
        "total_parameters": total,
        "limit": limit,
    }
