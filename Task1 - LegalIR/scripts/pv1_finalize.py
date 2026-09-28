#!/usr/bin/env python3
"""Freeze the promoted PV1 policy and create a reproducible complement bundle.

This is intentionally *not* a submission generator.  It accepts only a
passing PV1 stress report, chooses the locked three-way policy on all
complement labels, verifies the pinned verifier snapshot fingerprint, and
packages the policy/cache/manifest/code needed for later public inference.
"""

from __future__ import annotations

import argparse
import json
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any

from pv1_contract import KIND, load_pair_manifest, load_universe
from pv1_nested_eval import EVAL_KIND, _choose_policy, _load_scores, _metrics, _predict
from pv1_score_pairs import _fingerprint_model
from v27_a_common import V27ContractError, sha256_file, write_json


def finalize_policy(*, policy: dict[str, Any], model_revision: str, tokenizer_hash: str, prompt_hash: str, packer_hash: str, input_manifest_sha: str, predictions: dict[str, list[str]], anchor_map: dict[str, list[str]], promotion_pass: bool, output_dir: Path) -> str:
    """Compatibility helper for unit tests: validate predictions and freeze a policy file."""
    if not promotion_pass:
        raise V27ContractError("PV1 finalization is blocked: promotion gate has not passed")
    if set(predictions) != set(anchor_map):
        raise V27ContractError("PV1 predictions cover a different QID set than the anchor")
    for qid, answer in predictions.items():
        answer_ids = [str(value) for value in answer]
        anchor_ids = [str(value) for value in anchor_map[str(qid)]]
        if len(answer_ids) != 5 or len(set(answer_ids)) != 5:
            raise V27ContractError(f"{qid}: PV1 prediction must hold five unique document IDs")
        if answer_ids[:4] != anchor_ids[:4]:
            raise V27ContractError(f"{qid}: PV1 violated frozen top-4 contract")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    policy_path = output_path / "pv1_final_policy.json"
    payload = {
        "policy": dict(policy),
        "model_revision": str(model_revision),
        "tokenizer_hash": str(tokenizer_hash),
        "prompt_hash": str(prompt_hash),
        "packer_hash": str(packer_hash),
        "input_manifest_sha": str(input_manifest_sha),
        "predictions": {str(qid): [str(value) for value in answer] for qid, answer in predictions.items()},
        "promotion_pass": bool(promotion_pass),
        "not_a_submission": True,
    }
    write_json(policy_path, payload)
    return str(policy_path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path, required=True); parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True); parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True); parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--score-dir", type=Path, required=True); parser.add_argument("--stress-report", type=Path, required=True)
    parser.add_argument("--verifier-model", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--deadline", required=True, help="official deadline in ISO-8601 including timezone")
    parser.add_argument("--remaining-work-seconds", type=float, required=True, help="measured remaining public/bundle ETA including reserve")
    parser.add_argument("--bundle-path", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _deadline_ok(value: str, remaining_seconds: float) -> dict[str, Any]:
    if remaining_seconds < 0.0:
        raise ValueError("--remaining-work-seconds must be non-negative")
    try:
        deadline = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("--deadline must be ISO-8601 with timezone") from exc
    if deadline.tzinfo is None:
        raise ValueError("--deadline must include an explicit timezone")
    now = datetime.now(deadline.tzinfo); seconds = (deadline - now).total_seconds()
    return {"deadline": deadline.isoformat(), "checked_at": now.isoformat(), "seconds_until_deadline": seconds,
            "remaining_work_seconds": remaining_seconds, "enough_time": seconds >= remaining_seconds}


def _verify_promotion(report_path: Path, lineage_sha: str) -> dict[str, Any]:
    if not report_path.is_file(): raise FileNotFoundError(f"Missing PV1 stress report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("kind") != EVAL_KIND or report.get("mode") != "stress":
        raise V27ContractError("PV1 finalization requires its stress report")
    if report.get("lineage_sha256") != lineage_sha or report.get("promotion_to_public_allowed") is not True:
        raise V27ContractError("PV1 finalization is blocked: stress gate has not passed for this exact lineage")
    if report.get("holdout_opened") is not False or report.get("public_labels_used") is not False:
        raise V27ContractError("PV1 finalization refuses a report that opened holdout/public labels")
    return report


def _bundle(bundle_path: Path, paths: list[Path], *, output_dir: Path, include_model: Path | None) -> dict[str, Any]:
    if bundle_path.exists(): raise FileExistsError(f"Refusing to overwrite PV1 bundle: {bundle_path}")
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle_path, "w:gz") as archive:
        for path in sorted({item.resolve() for item in paths if item.is_file()}):
            archive.add(path, arcname=str(Path("pv1") / path.name), recursive=False)
        if include_model is not None:
            archive.add(include_model.resolve(), arcname="pv1/verifier_snapshot", recursive=True)
    return {"path": str(bundle_path.resolve()), "sha256": sha256_file(bundle_path), "bytes": bundle_path.stat().st_size,
            "includes_verifier_snapshot": include_model is not None}


def run(args: argparse.Namespace) -> dict[str, Any]:
    pair_rows, pair_manifest = load_pair_manifest(args.pair_manifest, args.pairs)
    scores, score_info = _load_scores(args.score_dir, pair_rows, pair_manifest)
    # Recreate exactly the evaluator's lineage preimage before permitting a
    # final policy.  The stress report's hash is authoritative for the rest.
    stress = json.loads(args.stress_report.read_text(encoding="utf-8"))
    if not isinstance(stress, dict): raise V27ContractError("Invalid PV1 stress report")
    lineage_sha = str(stress.get("lineage_sha256", "")); promotion = _verify_promotion(args.stress_report, lineage_sha)
    report_lineage = stress.get("lineage")
    if not isinstance(report_lineage, dict) or report_lineage.get("pair_lineage_sha256") != pair_manifest.get("lineage_sha256"):
        raise V27ContractError("PV1 stress report is not bound to this pair-manifest lineage")
    timing = _deadline_ok(args.deadline, args.remaining_work_seconds)
    if not timing["enough_time"]:
        raise V27ContractError("PV1 finalize is blocked: measured remaining work would exceed official deadline")
    fingerprint = _fingerprint_model(args.verifier_model)
    expected_fingerprint = score_info["contracts"][0].get("model_fingerprint")
    if fingerprint != expected_fingerprint or any(contract.get("model_fingerprint") != expected_fingerprint for contract in score_info["contracts"]):
        raise V27ContractError("Verifier snapshot fingerprint differs from the one that produced PV1 logits")
    root = Path(score_info["root"])
    reload_checks = sorted(root.glob("pv1_reload_verification_shard-*.json"))
    if len(reload_checks) != len(score_info["contracts"]):
        raise V27ContractError("PV1 finalization requires a verified fresh-model reload for every cache shard")
    for path in reload_checks:
        check = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(check, dict) or check.get("complete") is not True:
            raise V27ContractError(f"PV1 verifier reload check failed: {path}")
    split = load_universe(stage1=args.stage1, stage2=args.stage2, anchor=args.anchor, reference=args.reference).split
    indices = list(range(len(split.qids))); policy, trace = _choose_policy(split, indices, pair_manifest, scores)
    predictions, decisions = _predict(split, indices=indices, pair_manifest=pair_manifest, scores=scores, policy=policy)
    metrics = _metrics(split, predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy_path, predictions_path = args.output_dir / "pv1_final_policy.json", args.output_dir / "pv1_complement_final_predictions.json"
    if not args.force and (policy_path.exists() or predictions_path.exists()):
        raise FileExistsError("PV1 final policy/predictions exist; refuse to overwrite without --force")
    policy_payload = {"kind": f"{KIND}_final_policy", "schema_version": 1, "policy": policy, "selection": "all complement labels after passing stress; in-sample, not OOF",
                      "selection_trace": trace, "metrics": metrics, "pair_lineage_sha256": pair_manifest["lineage_sha256"], "evaluator_lineage_sha256": lineage_sha,
                      "verifier_fingerprint": fingerprint, "promotion_report_sha256": sha256_file(args.stress_report), "deadline": timing,
                      "holdout_opened": False, "public_labels_used": False, "not_a_submission": True}
    write_json(policy_path, policy_payload); write_json(predictions_path, {"kind": f"{KIND}_complement_predictions", "policy_sha256": sha256_file(policy_path), "predictions": predictions, "decisions": decisions, "not_a_submission": True})
    bundle_info = None
    if args.bundle_path is not None:
        files = [args.pairs, args.pair_manifest, args.stress_report, policy_path, predictions_path]
        files.extend(sorted((root / "metadata").glob("*.json"))); files.extend(sorted((root / "shards").glob("*.jsonl")))
        files.extend(reload_checks)
        for name in ("pv1_contract.py", "pv1_build_pairs.py", "pv1_score_pairs.py", "pv1_nested_eval.py", "pv1_finalize.py"):
            files.append(Path(__file__).with_name(name))
        bundle_info = _bundle(args.bundle_path, files, output_dir=args.output_dir, include_model=args.verifier_model)
    return {"kind": f"{KIND}_finalize_report", "policy": str(policy_path.resolve()), "predictions": str(predictions_path.resolve()), "metrics": metrics,
            "timing": timing, "promotion_report": str(args.stress_report.resolve()), "bundle": bundle_info, "holdout_opened": False, "public_labels_used": False,
            "next_action": "Use this frozen policy only with a separately preflighted, label-free public PV1 pair/cache manifest; do not treat complement predictions as a submission."}


def main() -> int:
    args = parse_args(); report_path = args.output_dir / "pv1_finalize_report.json"
    if report_path.exists() and not args.force: raise FileExistsError(f"PV1 finalizer report exists: {report_path}")
    report = run(args); write_json(report_path, report)
    print(json.dumps({"output": str(report_path.resolve()), "policy": report["policy"], "holdout_opened": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
