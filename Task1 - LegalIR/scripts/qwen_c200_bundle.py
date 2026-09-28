#!/usr/bin/env python3
"""Package a completed Qwen c200 research run with reproducibility artifacts.

This produces a research reproduction bundle, not a private submission.  It
contains every retained S2 fit (last, A-stop selected and backup), Qwen score
caches, reports, policy/feature schemas and immutable complement provenance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
from pathlib import Path
from typing import Any, Iterable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwen_c200_common import checkpoint_fingerprint, require_complement_file  # noqa: E402
from v27_a_common import V27ContractError, read_json, sha256_file, write_json  # noqa: E402


KIND = "legalir_qwen_c200_research_reproduction_bundle"
SCHEMA_VERSION = 1
REPORT_KIND = "legalir_qwen_c200_nested_selector_research"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result-report", type=Path, required=True)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--e1-manifest", type=Path, required=True)
    parser.add_argument("--e3b-standard-report", type=Path, required=True)
    parser.add_argument("--e3b-stress-report", type=Path, required=True)
    parser.add_argument("--qwen-model", type=Path, required=True, help="frozen shared Qwen model/tokenizer directory")
    parser.add_argument("--output-path", type=Path)
    return parser.parse_args()


def _required(path: Path, label: str) -> Path:
    require_complement_file(path, label)
    return Path(path).resolve()


def _relative_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Missing Qwen output directory: {root}")
    return sorted(path for path in root.rglob("*") if path.is_file() and path.name != "qwen_c200_reproduction_bundle.tar.gz")


def _model_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Missing shared Qwen model/tokenizer directory: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise V27ContractError("Shared Qwen model/tokenizer directory is empty")
    return files


def _fit_artifact_paths(report: Mapping[str, Any]) -> set[Path]:
    result: set[Path] = set()
    regimes = report.get("regimes")
    if not isinstance(regimes, Mapping):
        raise V27ContractError("Qwen result report has no nested regime reports")
    for runs in regimes.values():
        if not isinstance(runs, list):
            raise V27ContractError("Qwen result report has invalid nested runs")
        for run in runs:
            if not isinstance(run, Mapping) or not isinstance(run.get("folds"), list):
                raise V27ContractError("Qwen result report has invalid fold reports")
            for fold in run["folds"]:
                trace = fold.get("calibration_configuration_trace") if isinstance(fold, Mapping) else None
                if not isinstance(trace, list):
                    raise V27ContractError("Qwen fold is missing configuration trace")
                for trial in trace:
                    fit = trial.get("fit") if isinstance(trial, Mapping) else None
                    if not isinstance(fit, Mapping):
                        continue
                    artifacts = fit.get("artifacts")
                    if fit.get("checkpoint_verified") is not True or not isinstance(artifacts, Mapping):
                        raise V27ContractError("Qwen S2 fit was not checkpoint-verified")
                    for name in ("last", "selected"):
                        item = artifacts.get(name)
                        if not isinstance(item, Mapping) or item.get("pass") is not True:
                            raise V27ContractError(f"Qwen S2 {name} checkpoint lacks a successful reload verification")
                        path = Path(str(item.get("checkpoint", "")))
                        if not path.is_file() or sha256_file(path) != item.get("sha256"):
                            raise V27ContractError(f"Qwen S2 {name} checkpoint is missing or hash-drifted")
                        result.add(path.resolve())
                    backup = artifacts.get("selected_backup")
                    marker = artifacts.get("complete_marker")
                    if not isinstance(backup, Mapping) or not isinstance(marker, str):
                        raise V27ContractError("Qwen S2 fit omits selected backup or completion marker")
                    backup_path, marker_path = Path(str(backup.get("path", ""))), Path(marker)
                    if not backup_path.is_file() or sha256_file(backup_path) != backup.get("sha256") or not marker_path.is_file():
                        raise V27ContractError("Qwen S2 backup/marker is missing or hash-drifted")
                    result.update((backup_path.resolve(), marker_path.resolve()))
    if not result:
        raise V27ContractError("Qwen report contains no retained S2 artifacts")
    return result


def _add_file(archive: tarfile.TarFile, path: Path, arcname: str, rows: list[dict[str, Any]]) -> None:
    archive.add(path, arcname=arcname, recursive=False)
    rows.append({"archive_path": arcname, "source_path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    result_report = _required(args.result_report, "Qwen result report")
    report = read_json(result_report)
    if not isinstance(report, dict) or report.get("kind") != REPORT_KIND or report.get("holdout_opened") is not False or report.get("public_labels_used") is not False:
        raise V27ContractError("Only a complement-only completed Qwen research report can be bundled")
    if report.get("checkpoint", {}).get("resumable") is not True:
        raise V27ContractError("Qwen result report is missing its resumable checkpoint contract")
    retained = _fit_artifact_paths(report)
    provenance = {
        "stage1": _required(args.stage1, "Stage-1"), "stage2": _required(args.stage2, "Stage-2"),
        "anchor": _required(args.anchor, "V16b anchor"), "reference": _required(args.reference, "complement reference"),
        "e1_manifest": _required(args.e1_manifest, "E1 evidence manifest"),
        "e3b_standard_report": _required(args.e3b_standard_report, "E3-B standard report"),
        "e3b_stress_report": _required(args.e3b_stress_report, "E3-B stress report"),
    }
    qwen_model = Path(args.qwen_model).resolve()
    qwen_fingerprint = checkpoint_fingerprint(qwen_model)
    expected_qwen = report.get("lineage", {}).get("qwen_model_fingerprint")
    if expected_qwen != qwen_fingerprint:
        raise V27ContractError("Shared Qwen model/tokenizer fingerprint differs from the completed score cache/report")
    output_path = Path(args.output_path or output_dir / "qwen_c200_reproduction_bundle.tar.gz").resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite reproduction bundle: {output_path}")
    output_files = _relative_files(output_dir)
    rows: list[dict[str, Any]] = []
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    with tarfile.open(temporary, mode="w:gz") as archive:
        for path in output_files:
            _add_file(archive, path, f"run_output/{path.relative_to(output_dir).as_posix()}", rows)
        for name, path in provenance.items():
            _add_file(archive, path, f"frozen_inputs/{name}{path.suffix}", rows)
        for path in _model_files(qwen_model):
            _add_file(archive, path, f"qwen_model/{path.relative_to(qwen_model).as_posix()}", rows)
        source_root = Path(__file__).resolve().parent
        for name in ("qwen_c200_common.py", "qwen_c200_cache.py", "qwen_c200_nested_eval.py", "qwen_c200_bundle.py", "gpu_transition_metrics.py", "v27_a_common.py"):
            _add_file(archive, source_root / name, f"code/{name}", rows)
        manifest = {
            "kind": KIND, "schema_version": SCHEMA_VERSION, "result_report": str(result_report),
            "result_report_sha256": sha256_file(result_report), "lineage_sha256": report.get("lineage_sha256"),
            "retained_s2_artifacts": len(retained), "files": rows,
            "qwen_checkpoint": qwen_fingerprint,
            "note": "One shared Qwen backbone/tokenizer is retained in qwen_model/, never duplicated per fold. This remains a complement-only research reproduction bundle, not a private submission.",
            "holdout_opened": False, "public_labels_used": False,
        }
        payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        info = tarfile.TarInfo("bundle_manifest.json"); info.size = len(payload)
        archive.addfile(info, fileobj=__import__("io").BytesIO(payload))
    os.replace(temporary, output_path)
    bundle_sha = sha256_file(output_path)
    report_path = output_dir / "qwen_c200_reproduction_bundle_report.json"
    bundle_report = {"kind": KIND, "schema_version": SCHEMA_VERSION, "bundle": str(output_path), "sha256": bundle_sha, "bytes": output_path.stat().st_size, "retained_s2_artifacts": len(retained), "files": len(rows) + 1, "holdout_opened": False, "public_labels_used": False}
    write_json(report_path, bundle_report)
    return bundle_report


def main() -> int:
    report = run(parse_args())
    print(json.dumps({"output": report["bundle"], "sha256": report["sha256"], "holdout_opened": False}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
