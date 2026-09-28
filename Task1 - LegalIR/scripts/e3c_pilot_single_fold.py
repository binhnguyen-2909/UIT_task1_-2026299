#!/usr/bin/env python3
"""E3-C pilot single-fold: complement-only, exploratory, ETA <40p.

Uses exactly one retained E3-B weight (standard, inner 20260910, fold 1,
neural 20260908) from the retain dir. Trains one LoRA duo comparator for
1 epoch, scores C/V for that fold only, selects a fixed-margin policy on C
and reports V. Never opens holdout/public, never authorizes E4, never
touches the locked E3-B policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from v27_a_common import V27ContractError

PILOT_REGIME = "standard"
PILOT_INNER_SEED = 20260910
PILOT_OUTER_FOLD = 0
PILOT_EPOCHS = 1
PILOT_KIND = "legalir_e3c_pilot_single_fold"
PILOT_SOURCE_NEURAL_SEED = 20260908
FORBIDDEN_TERMS = ("holdout", "public", "submission", "locked")


def _safe_path(path: Path, label: str) -> None:
    value = str(path).replace("\\", "/").casefold()
    if any(term in value for term in FORBIDDEN_TERMS):
        raise V27ContractError(
            f"E3-C pilot is complement-only; {label} may not refer to holdout/public: {path}"
        )


def pilot_weight_path(e3b_root: Path) -> Path:
    return (
        e3b_root
        / "e3b_weights"
        / "standard"
        / f"inner_{PILOT_INNER_SEED}_fold_{PILOT_OUTER_FOLD + 1}_neural_{PILOT_SOURCE_NEURAL_SEED}.pt"
    )


def pilot_preflight(*, e3b_root: Path, output_dir: Path) -> dict:
    _safe_path(e3b_root, "e3b_root")
    _safe_path(output_dir, "output_dir")
    weight = pilot_weight_path(e3b_root)
    if not weight.is_file():
        raise FileNotFoundError(
            f"E3-C pilot needs retained E3-B single-fold weight but cannot find {weight}. "
            "Run E3B retain for standard inner_20260910 fold_1 first; do not substitute a base checkpoint."
        )
    progress = e3b_root / "e3b_standard_progress.json"
    if not progress.is_file():
        raise FileNotFoundError(f"E3-C pilot needs E3B retain progress: {progress}")
    return {
        "kind": PILOT_KIND,
        "regime": PILOT_REGIME,
        "inner_seed": PILOT_INNER_SEED,
        "outer_fold": PILOT_OUTER_FOLD,
        "epochs": PILOT_EPOCHS,
        "weight": str(weight.resolve()),
        "holdout_opened": False,
        "public_labels_used": False,
        "exploratory_only": True,
        "promotion_to_e4_allowed": False,
    }


def pilot_lineage(*, e3b_weight_sha256: str, e3b_state_sha256: str) -> dict:
    return {
        "kind": PILOT_KIND,
        "regime": PILOT_REGIME,
        "inner_seed": PILOT_INNER_SEED,
        "outer_fold": PILOT_OUTER_FOLD,
        "epochs": PILOT_EPOCHS,
        "e3b_weight_sha256": e3b_weight_sha256,
        "e3b_state_sha256": e3b_state_sha256,
        "holdout_opened": False,
        "public_labels_used": False,
        "exploratory_only": True,
        "promotion_to_e4_allowed": False,
        "promotion_to_stress_allowed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--e1-manifest", type=Path, required=True)
    parser.add_argument("--e1-manifest-meta", type=Path, required=True)
    parser.add_argument("--e1-frozen-scores", type=Path, required=True)
    parser.add_argument("--reranker-model", type=Path, required=True)
    parser.add_argument("--e3b-retain-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=PILOT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    import e3c_nested_duo_lora as e3c
    import e2_mil_finetune_oof as e2
    import e3b_nested_mil_scores as e3b
    from e3c_duo_gate import apply_policy as apply_duo_policy
    from e3c_duo_gate import select_calibration_policy
    from v27_a_common import group_folds, json_sha256, sha256_file, write_json

    if int(args.epochs) != PILOT_EPOCHS:
        raise V27ContractError(f"E3-C pilot pins epochs={PILOT_EPOCHS} for <40p ETA")
    pre = pilot_preflight(e3b_root=args.e3b_retain_dir, output_dir=args.output_dir)
    for path, label in (
        (args.stage1, "stage1"),
        (args.stage2, "stage2"),
        (args.anchor, "anchor"),
        (args.reference, "reference"),
        (args.e1_manifest, "E1 manifest"),
        (args.e1_manifest_meta, "E1 metadata"),
        (args.e1_frozen_scores, "E1 scores"),
    ):
        _safe_path(path, label)
        if not path.is_file():
            raise FileNotFoundError(f"E3-C pilot required input is missing: {path}")
    if not args.reranker_model.is_dir():
        raise FileNotFoundError("E3-C pilot reranker model is missing")

    weight_path = Path(pre["weight"])
    file_sha = sha256_file(weight_path)

    # Verify source against retain progress record (same-fold condition).
    progress = json.loads((args.e3b_retain_dir / "e3b_standard_progress.json").read_text(encoding="utf-8"))
    fold_key = f"standard|inner={PILOT_INNER_SEED}|fold={PILOT_OUTER_FOLD}"
    fold_report = progress["completed_folds"][fold_key]["fold_report"]
    source_state = None
    for model in fold_report["neural_models"]:
        if int(model.get("neural_seed", -1)) == PILOT_SOURCE_NEURAL_SEED:
            source_state = model["weights"]
            break
    if source_state is None:
        raise V27ContractError("E3-C pilot: retain progress has no source neural seed")
    if source_state.get("retained_sha256") != file_sha:
        raise V27ContractError("E3-C pilot: retained weight hash mismatch")
    source = {
        "path": str(weight_path.resolve()),
        "file_sha256": file_sha,
        "state_sha256": source_state["state_sha256"],
        "neural_seed": PILOT_SOURCE_NEURAL_SEED,
    }

    # Build E2 data with pilot args mapped onto e3c/e2 namespaces.
    e2_args = argparse.Namespace(
        stage1=args.stage1,
        stage2=args.stage2,
        anchor=args.anchor,
        reference=args.reference,
        e1_manifest=args.e1_manifest,
        e1_manifest_meta=args.e1_manifest_meta,
        e1_frozen_scores=args.e1_frozen_scores,
        reranker_model=args.reranker_model,
    )
    data = e2.load_e2_data(e2_args)
    groups = data.groups_by_regime[PILOT_REGIME]
    outer_folds = group_folds(groups)
    outer_train, validation = outer_folds[PILOT_OUTER_FOLD]
    partitions, partition_metadata = e3b.make_partitions(
        split=data.split,
        groups=groups,
        outer_train=outer_train,
        validation=validation,
        inner_seed=PILOT_INNER_SEED,
        outer_fold=PILOT_OUTER_FOLD,
    )

    e3c_args = argparse.Namespace(
        epochs=PILOT_EPOCHS,
        learning_rate=float(args.learning_rate),
        weight_decay=0.01,
        micro_batch=4,
        gradient_accumulation=int(args.gradient_accumulation),
        score_batch_size=16,
        micro_batch_candidates=[16, 8, 4, 2, 1],
        score_batch_candidates=[512, 256, 128, 64, 32, 16, 8],
        auto_probe=True,
        runtime_contract=None,
        vram_cap_gib=None,
        cpu_threads=int(args.cpu_threads),
        gradient_checkpointing=False,
        device=args.device,
        resume=True,
        output_dir=args.output_dir,
        mode="standard",
    )
    e3c._configure_cpu_threads(e3c_args)
    runtime, runtime_report = e3c._runtime_from_contract(e3c_args, data, {f"pilot|{PILOT_OUTER_FOLD}": source})
    e3c._apply_runtime(e3c_args, runtime)

    pairs_fit = e3c._eligible_pairs(data, partitions["a_fit"].tolist(), with_targets=True)
    pairs_stop = e3c._eligible_pairs(data, partitions["a_stop"].tolist(), with_targets=True)
    pairs_c = e3c._eligible_pairs(data, partitions["calibration"].tolist(), with_targets=False)
    pairs_v = e3c._eligible_pairs(data, partitions["validation"].tolist(), with_targets=False)

    model, tokenizer, training = e3c._train_comparator(
        args=e3c_args,
        pairs_fit=pairs_fit,
        pairs_stop=pairs_stop,
        source=source,
        regime=PILOT_REGIME,
        inner_seed=PILOT_INNER_SEED,
        outer_fold=PILOT_OUTER_FOLD,
    )
    try:
        lineage_sha = json_sha256(pilot_lineage(e3b_weight_sha256=file_sha, e3b_state_sha256=source["state_sha256"]))
        c_scores, c_artifact = e3c._export_scores(
            args=e3c_args, model=model, tokenizer=tokenizer, pairs=pairs_c, data=data,
            indices=partitions["calibration"].tolist(), regime=PILOT_REGIME, partition="calibration",
            inner_seed=PILOT_INNER_SEED, outer_fold=PILOT_OUTER_FOLD, lineage_sha=lineage_sha,
        )
        v_scores, v_artifact = e3c._export_scores(
            args=e3c_args, model=model, tokenizer=tokenizer, pairs=pairs_v, data=data,
            indices=partitions["validation"].tolist(), regime=PILOT_REGIME, partition="validation",
            inner_seed=PILOT_INNER_SEED, outer_fold=PILOT_OUTER_FOLD, lineage_sha=lineage_sha,
        )
    finally:
        import gc

        del model, tokenizer
        gc.collect()
        if args.device == "cuda":
            import torch

            torch.cuda.empty_cache()

    from v27_a_common import json_sha256 as _sha

    c_split = e3c._subset(data.split, partitions["calibration"])
    v_split = e3c._subset(data.split, partitions["validation"])
    policy, calibration, trace = select_calibration_policy(c_split, c_scores)
    predictions, decisions = apply_duo_policy(v_split, list(range(len(v_split.qids))), v_scores, policy)
    report = {
        "kind": PILOT_KIND,
        "regime": PILOT_REGIME,
        "inner_seed": PILOT_INNER_SEED,
        "outer_fold": PILOT_OUTER_FOLD,
        "split": "complement",
        "lineage": pilot_lineage(e3b_weight_sha256=file_sha, e3b_state_sha256=source["state_sha256"]),
        "partitions": partition_metadata,
        "training": training,
        "score_artifacts": [c_artifact, v_artifact],
        "selected_policy": {"kind": "v16b_anchor_abstain"} if policy is None else policy.as_dict(),
        "calibration": calibration,
        "policy_trace": trace,
        "validation_vs_v16b": e3c.compact_metrics(
            __import__("gpu_transition_metrics", fromlist=["transition_metrics_device"]).transition_metrics_device(
                v_split, predictions, device=args.device
            )
        ),
        "decisions": decisions,
        "runtime": runtime,
        "runtime_report": runtime_report,
        "holdout_opened": False,
        "public_labels_used": False,
        "exploratory_only": True,
        "promotion_to_e4_allowed": False,
        "promotion_to_stress_allowed": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "e3c_pilot_single_fold_report.json"
    if out.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite E3-C pilot report: {out}")
    write_json(out, report)
    return report


def main() -> int:
    args = parse_args()
    report = run(args)
    print(json.dumps({"output": str((args.output_dir / "e3c_pilot_single_fold_report.json").resolve()), "kind": report["kind"], "holdout_opened": False, "promotion_to_e4_allowed": False}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
