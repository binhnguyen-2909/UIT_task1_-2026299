"""Run V21-A / P3-GX LoRA r=4 with true group-isolated CE score matrices.

This wrapper preserves the fixed P3 recipe (one epoch, semi-hard boundary
pairs, anchor preservation, B2304-D60 evidence packs) while replacing the old
query-hash folds with four independent, audited group partitions. Each output
matrix is valid only for its named regime. A later winner-policy evaluator must
calibrate abstention on selected winners, never pooled candidate pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping


REGIMES = (
    "standard_grouped",
    "law_family_ood",
    "query_template_ood",
    "near_duplicate_ood",
)
MANIFEST_KIND = "legalir_v21_group_crossfit_manifest"
FORBIDDEN_TERMS = ("holdout", "public", "locked", "submission")


def _safe_clean_path(path: Path, label: str) -> None:
    normalized = str(path.resolve()).replace("\\", "/").casefold()
    if any(term in normalized for term in FORBIDDEN_TERMS):
        raise ValueError(f"{label} must be clean-complement-only: {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_legacy_trainer(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    active_scripts = Path(__file__).resolve().parent
    legacy_scripts = path.resolve().parent
    for candidate in (str(active_scripts), str(legacy_scripts)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    spec = importlib.util.spec_from_file_location("v21_legacy_p3_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import P3 trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves postponed annotations through sys.modules while the
    # legacy trainer is executing, so its dynamically loaded module must be
    # registered before exec_module.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def _load_manifest(path: Path) -> dict[str, Any]:
    _safe_clean_path(path, "fold manifest")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("kind") != MANIFEST_KIND:
        raise ValueError("not a V21 group-crossfit manifest")
    if payload.get("split") != "clean_complement_only" or bool(payload.get("holdout_opened")) or bool(payload.get("public_labels_used")):
        raise ValueError("fold manifest does not attest clean-complement-only execution")
    if not bool(payload.get("all_registered_regimes_zero_overlap")):
        raise ValueError("fold manifest failed zero-overlap audit")
    return payload


def _regime_fold_map(payload: Mapping[str, Any], regime: str, expected_qids: set[str]) -> dict[str, int]:
    raw_regime = payload.get("regimes", {}).get(regime)
    if not isinstance(raw_regime, Mapping):
        raise ValueError(f"fold manifest misses regime: {regime}")
    rows = raw_regime.get("folds")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError(f"{regime}: fewer than two outer folds")
    folds: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{regime}: malformed fold row")
        fold = int(row.get("fold", -1))
        train = {str(value) for value in row.get("train_qids", [])}
        validation = [str(value) for value in row.get("validation_qids", [])]
        train_groups = {str(value) for value in row.get("train_groups", [])}
        validation_groups = {str(value) for value in row.get("validation_groups", [])}
        if fold < 0 or not validation or train_groups & validation_groups:
            raise ValueError(f"{regime}: malformed or leaking fold {fold}")
        if set(validation) & train:
            raise ValueError(f"{regime}: query identity leakage in fold {fold}")
        for qid in validation:
            if qid in folds:
                raise ValueError(f"{regime}: qid assigned to multiple validation folds: {qid}")
            folds[qid] = fold
    if set(folds) != expected_qids:
        raise ValueError(f"{regime}: validation qid coverage differs from metadata")
    return folds


def _namespace(args: argparse.Namespace, output_dir: Path, folds: int) -> argparse.Namespace:
    return argparse.Namespace(
        metadata=args.metadata,
        current_scores=args.current_scores,
        index_dir=None,
        manifest=args.pack_manifest,
        pack_config=args.pack_config,
        model=args.model,
        output_dir=output_dir,
        passage_cache=None,
        lora_r=4,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        anchor_weight=args.anchor_weight,
        epochs=1,
        batch_size=args.batch_size,
        grad_accumulation=args.grad_accumulation,
        max_grad_norm=args.max_grad_norm,
        max_length=args.max_length,
        inference_batch_size=args.inference_batch_size,
        negative_per_positive=2,
        folds=folds,
        seed=args.seed,
        device=args.device,
        dry_run=args.dry_run,
        force=args.force or args.resume,
        resume=args.resume,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path, label in (
        (args.metadata, "metadata"), (args.current_scores, "frozen CE scores"),
        (args.pack_manifest, "evidence pack manifest"), (args.model, "reranker model"),
        (args.fold_manifest, "fold manifest"), (args.legacy_trainer, "legacy trainer"),
        (args.output_dir, "output"),
    ):
        _safe_clean_path(path, label)
    manifest = _load_manifest(args.fold_manifest)
    trainer = _load_legacy_trainer(args.legacy_trainer)
    _, by_qid, _ = trainer.load_inputs(args.metadata, args.current_scores)
    expected_qids = set(by_qid)
    manifest_qids = {str(value) for value in manifest.get("qids", [])}
    if manifest_qids != expected_qids:
        raise ValueError("fold-manifest qids differ from P3 metadata")
    if manifest.get("metadata_sha256") != _sha256_file(args.metadata):
        raise ValueError("fold manifest was built from different metadata")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original_qid_fold = trainer._qid_fold
    regimes: dict[str, Any] = {}
    try:
        for regime in REGIMES:
            fold_map = _regime_fold_map(manifest, regime, expected_qids)
            fold_count = len(set(fold_map.values()))
            if fold_count < 2:
                raise ValueError(f"{regime}: insufficient folds")

            def group_fold(qid: str, folds: int, *, mapping: Mapping[str, int] = fold_map, expected: int = fold_count) -> int:
                if folds != expected:
                    raise ValueError(f"{regime}: trainer fold count drift ({folds} != {expected})")
                return int(mapping[str(qid)])

            trainer._qid_fold = group_fold
            regime_dir = args.output_dir / regime
            result = trainer.run(_namespace(args, regime_dir, fold_count))
            report_path = regime_dir / "lora_training_report.json"
            score_path = regime_dir / "adapted_current_passage_scores.json"
            if not args.dry_run and not score_path.is_file():
                raise RuntimeError(f"{regime}: missing group-cross-fitted score matrix")
            regimes[regime] = {
                "folds": fold_count,
                "score_path": str(score_path.resolve()) if score_path.exists() else None,
                "score_sha256": _sha256_file(score_path) if score_path.exists() else None,
                "training_report": str(report_path.resolve()),
                "training_report_sha256": _sha256_file(report_path),
                "examples": result["training"]["examples"],
                "fold_stats": result.get("folds", {}),
            }
    finally:
        trainer._qid_fold = original_qid_fold
    payload = {
        "kind": "legalir_v21_p3_gx_group_crossfit",
        "schema_version": 1,
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "selection_performed": False,
        "recipe": {
            "lora_r": 4,
            "epochs": 1,
            "negative_policy": "semi-hard boundary negatives; skip top-1 hardest when possible; near-duplicate-title exclusions",
            "anchor_preservation": True,
            "pack_config": args.pack_config,
            "winner_level_calibration_required_next": True,
        },
        "inputs": {
            "metadata": str(args.metadata.resolve()), "metadata_sha256": _sha256_file(args.metadata),
            "current_scores": str(args.current_scores.resolve()), "current_scores_sha256": _sha256_file(args.current_scores),
            "pack_manifest": str(args.pack_manifest.resolve()), "pack_manifest_sha256": _sha256_file(args.pack_manifest),
            "fold_manifest": str(args.fold_manifest.resolve()), "fold_manifest_sha256": _sha256_file(args.fold_manifest),
            "legacy_trainer": str(args.legacy_trainer.resolve()), "legacy_trainer_sha256": _sha256_file(args.legacy_trainer),
        },
        "regimes": regimes,
        "parameter_ledger": {
            "base_pipeline_parameters": 1_090_010_000,
            "shared_deployed_lora_upper_bound": 20_000_000,
            "four_tiny_heads_upper_bound": 1_000_000,
            "end_to_end_upper_bound": 1_111_010_000,
            "under_4b": True,
            "note": "The four fold adapters are temporary validation artifacts; deployment uses one shared full-data adapter only after promotion.",
        },
    }
    output = args.output_dir / "v21_p3_gx_report.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--current-scores", type=Path, required=True)
    parser.add_argument("--pack-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--legacy-trainer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pack-config", default="B2304-D60")
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--anchor-weight", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--inference-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(args)
    print(json.dumps({
        "output": str((args.output_dir / "v21_p3_gx_report.json").resolve()),
        "dry_run": args.dry_run,
        "regimes": {name: {"folds": report["regimes"][name]["folds"], "scores": report["regimes"][name]["score_path"]} for name in REGIMES},
        "under_4b": report["parameter_ledger"]["under_4b"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
