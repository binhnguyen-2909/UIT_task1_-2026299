"""Run the V16b memory-feature gate on the frozen v11h c35 candidate set.

This experiment deliberately does not expand the candidate pool and does not
rerun the CrossEncoder.  It appends leak-safe QREL-memory features to the
existing c35 rows, trains the existing tiny selector, and evaluates a
conservative policy that can replace only slot 5 of the locked v11h answer.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v11_tiny_ltr import (  # noqa: E402
    AuditSplit,
    apply_policy,
    baseline_predictions,
    build_leak_groups,
    build_query_normalized_features,
    evaluate_policy,
    load_audit_split,
    load_documents,
    load_priors,
    model_scores,
    row_indices,
    select_model_and_policy,
    train_model,
    write_json,
)


MEMORY_FEATURE_NAMES = (
    "memory_score",
    "prototype_max",
    "prototype_top3_mean",
    "neighbor_vote_count",
    "unique_neighbor_count",
    "neighbor_similarity_max",
    "neighbor_similarity_gap_1_2",
    "neighbor_entropy",
    "reciprocal_neighbor_score",
    "dense_lexical_neighbor_agreement",
    "document_qrel_degree",
    "degree_corrected_memory_score",
    "memory_rank",
    "is_memory_outside_c35",
    "memory_vs_anchor_margin",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_memory_feature_map(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    payload = read_json(path)
    rows = payload.get("queries") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"Invalid V16 memory feature artifact: {path}")
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict) or "query_id" not in row:
            raise ValueError(f"Invalid V16 memory query row: {path}")
        qid = str(row["query_id"])
        memory = row.get("memory", {})
        documents = memory.get("documents") if isinstance(memory, dict) else None
        if not isinstance(documents, list):
            raise ValueError(f"Memory documents missing for {qid}: {path}")
        by_doc: dict[str, dict[str, Any]] = {}
        for item in documents:
            if not isinstance(item, dict) or "doc_id" not in item:
                raise ValueError(f"Invalid memory document for {qid}: {path}")
            doc_id = str(item["doc_id"])
            if doc_id in by_doc:
                raise ValueError(f"Duplicate memory document {doc_id} for {qid}")
            by_doc[doc_id] = dict(item)
        output[qid] = by_doc
    if len(output) != len(rows):
        raise ValueError(f"Memory feature artifact contains duplicate query IDs: {path}")
    return output


def replace_baselines_from_predictions(
    split: AuditSplit,
    path: Path,
) -> AuditSplit:
    """Use an existing locked selector output as the inference anchor."""

    payload = read_json(path)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("queries", payload.get("results", payload.get("items")))
        if rows is None and all(isinstance(value, dict) for value in payload.values()):
            rows = [dict(value, qid=str(key)) for key, value in payload.items()]
    else:
        rows = None
    if not isinstance(rows, list):
        raise ValueError(f"Invalid locked selector output: {path}")
    by_qid: dict[str, list[str]] = {}
    for row in rows:
        if not isinstance(row, dict) or "qid" not in row or not isinstance(row.get("answer"), list):
            raise ValueError(f"Invalid locked selector row: {path}")
        answers = list(dict.fromkeys(str(value) for value in row["answer"]))
        if len(answers) != 5:
            raise ValueError(f"Locked selector row does not contain five answers: {row.get('qid')}")
        by_qid[str(row["qid"])] = answers
    if set(by_qid) != set(split.qids):
        missing = sorted(set(split.qids) - set(by_qid))
        extra = sorted(set(by_qid) - set(split.qids))
        raise ValueError(f"Locked selector qids differ; missing={missing[:5]}, extra={extra[:5]}")
    output = replace(
        split,
        baseline=[by_qid[qid] for qid in split.qids],
    )
    for qid, candidates, baseline in zip(
        output.qids, output.candidates, output.baseline, strict=True
    ):
        if not set(baseline).issubset(set(candidates[:35])):
            raise ValueError(f"{output.name}/{qid}: locked anchor escapes c35")
    return output


def load_v16_split(
    name: str,
    stage1_path: Path,
    stage2_path: Path,
    reference: Path | None,
) -> AuditSplit:
    split = load_audit_split(name, stage1_path, stage2_path, reference)
    for qid, candidates, neural_scores, baseline in zip(
        split.qids, split.candidates, split.neural_scores, split.baseline, strict=True
    ):
        c35 = set(candidates[:35])
        score_ids = set(neural_scores)
        if len(candidates) < 35 or len(score_ids) != 35:
            raise ValueError(f"{name}/{qid}: expected a full c35 candidate row")
        if not score_ids.issubset(c35):
            raise ValueError(f"{name}/{qid}: Stage-2 scores escape the Stage-1 c35")
        if not set(baseline).issubset(score_ids):
            raise ValueError(f"{name}/{qid}: baseline answer is absent from c35 scores")
    return split


def subset_split(split: AuditSplit, indices: Sequence[int]) -> AuditSplit:
    selected = [int(index) for index in indices]
    return AuditSplit(
        name=split.name,
        qids=[split.qids[index] for index in selected],
        questions=[split.questions[index] for index in selected],
        candidates=[split.candidates[index] for index in selected],
        baseline=[split.baseline[index] for index in selected],
        neural_scores=[split.neural_scores[index] for index in selected],
        result_rows=[split.result_rows[index] for index in selected],
        references=[split.references[index] for index in selected]
        if split.references is not None else None,
        stage1_path=split.stage1_path,
    )


def remove_exact_memory_queries(split: AuditSplit) -> tuple[AuditSplit, list[str]]:
    if split.stage1_path is None:
        return split, []
    payload = read_json(split.stage1_path)
    rows = payload.get("queries", [])
    exact_ids = {
        str(row["query_id"])
        for row in rows
        if isinstance(row, dict) and bool(row.get("metadata", {}).get("exact_memory"))
    }
    keep = [index for index, qid in enumerate(split.qids) if qid not in exact_ids]
    return subset_split(split, keep), sorted(set(split.qids) - set(split.qids[index] for index in keep))


def augment_features(
    split: AuditSplit,
    documents: Mapping[str, Any],
    document_frequency: Mapping[str, int],
    memory_by_qid: Mapping[str, Mapping[str, Mapping[str, Any]]],
    candidate_limit: int,
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    base, target, base_names = build_query_normalized_features(
        split, documents, document_frequency, candidate_limit
    )
    rows: list[list[float]] = []
    defaults = {name: 0.0 for name in MEMORY_FEATURE_NAMES}
    for query_index, qid in enumerate(split.qids):
        by_doc = memory_by_qid.get(qid, {})
        candidates = list(split.candidates[query_index][:candidate_limit])
        candidates.extend([""] * (candidate_limit - len(candidates)))
        for doc_id in candidates:
            item = by_doc.get(str(doc_id), defaults)
            rows.append([
                float(item.get(name, 0.0) or 0.0)
                for name in MEMORY_FEATURE_NAMES
            ])
    extra = np.asarray(rows, dtype=np.float32)
    if extra.shape[0] != base.shape[0]:
        raise RuntimeError(f"{split.name}: memory feature row count mismatch")
    return np.hstack((base, extra)), target, [*base_names, *MEMORY_FEATURE_NAMES]


def transition_report(
    split: AuditSplit,
    predictions: Mapping[str, Mapping[str, Sequence[str]]],
    question_indices: Sequence[int],
) -> dict[str, int]:
    if split.references is None:
        return {"changed_queries": 0, "successful_rescues": 0, "destructive_changes": 0}
    changed = successes = destructive = 0
    for index in question_indices:
        qid = split.qids[index]
        anchor = set(split.baseline[index])
        proposed = set(str(value) for value in predictions[qid]["answer"])
        gold = set(split.references[index])
        if proposed != anchor:
            changed += 1
        if (proposed & gold) - anchor:
            successes += 1
        if (anchor & gold) - proposed:
            destructive += 1
    return {
        "changed_queries": changed,
        "successful_rescues": successes,
        "destructive_changes": destructive,
    }


def memory_slot5_policy_grid(base_grid: Any) -> list[dict[str, Any]]:
    """Return only conservative memory-gated slot-5 policies.

    The ordinary v11h prefix/RRF policies are intentionally excluded.  This
    keeps the experiment attributable to the agreed V16b intervention and
    prevents a broad selector policy from bypassing the slot-5 safety gate.
    """

    del base_grid
    policies: list[dict[str, Any]] = []
    for memory_floor in (0.005, 0.01, 0.02, 0.05):
        for memory_margin in (0.0, 0.005, 0.01, 0.02):
            for model_margin in (0.0, 0.02, 0.05):
                for exact in (False, True):
                    policies.append({
                        "kind": "memory_slot5",
                        "pool": 35,
                        "memory_feature": "degree_corrected_memory_score",
                        "memory_floor": memory_floor,
                        "memory_margin": memory_margin,
                        "model_margin": model_margin,
                        "exact_rescue": exact,
                    })
    return policies


def _policy_grid_with_memory_pools(
    base_grid: Any,
) -> list[dict[str, Any]]:
    return [
        policy
        for policy in base_grid()
        if int(policy["pool"]) in {35, 40}
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts-zip", type=Path, required=True)
    parser.add_argument("--prior-index", type=Path, required=True)
    parser.add_argument("--complement-stage1", type=Path, required=True)
    parser.add_argument("--complement-stage2", type=Path, required=True)
    parser.add_argument("--complement-reference", type=Path, required=True)
    parser.add_argument("--complement-memory-features", type=Path, required=True)
    parser.add_argument("--holdout-stage1", type=Path, required=True)
    parser.add_argument("--holdout-stage2", type=Path, required=True)
    parser.add_argument("--holdout-reference", type=Path, required=True)
    parser.add_argument("--holdout-memory-features", type=Path, required=True)
    parser.add_argument("--holdout-anchor-json", type=Path, required=True)
    parser.add_argument("--public-stage1", type=Path, required=True)
    parser.add_argument("--public-stage2", type=Path, required=True)
    parser.add_argument("--public-memory-features", type=Path, required=True)
    parser.add_argument("--public-anchor-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-limit", type=int, default=35)
    parser.add_argument("--model-profile", default="pointwise-v2-winner")
    parser.add_argument("--minimum-holdout-delta", type=float, default=0.0)
    parser.add_argument("--submission-name", default="v16b_memory_ltr.json")
    parser.add_argument(
        "--holdout-predictions",
        type=Path,
        help="optional report-only export of the locked holdout predictions",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.candidate_limit != 35:
        raise ValueError("V16b is intentionally fixed to the existing c35 candidate set")
    if Path(args.submission_name).name != args.submission_name:
        raise ValueError("submission-name must be a plain filename")
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    complement = load_v16_split(
        "complement", args.complement_stage1, args.complement_stage2,
        args.complement_reference,
    )
    complement, excluded_exact = remove_exact_memory_queries(complement)
    holdout = load_v16_split(
        "holdout", args.holdout_stage1, args.holdout_stage2,
        args.holdout_reference,
    )
    holdout = replace_baselines_from_predictions(holdout, args.holdout_anchor_json)
    public = load_v16_split(
        "public", args.public_stage1, args.public_stage2, None,
    )
    public = replace_baselines_from_predictions(public, args.public_anchor_json)
    memory_maps = {
        "complement": load_memory_feature_map(args.complement_memory_features),
        "holdout": load_memory_feature_map(args.holdout_memory_features),
        "public": load_memory_feature_map(args.public_memory_features),
    }
    if set(complement.qids) - set(memory_maps["complement"]):
        raise ValueError("Complement memory features do not cover the cleaned split")
    if set(holdout.qids) - set(memory_maps["holdout"]):
        raise ValueError("Holdout memory features do not cover the split")
    if set(public.qids) - set(memory_maps["public"]):
        raise ValueError("Public memory features do not cover the split")

    needed_ids = {
        doc_id
        for split in (complement, holdout, public)
        for row in split.candidates
        for doc_id in row[:args.candidate_limit]
        if doc_id
    }
    needed_ids.update(
        doc_id
        for split in (complement, holdout, public)
        for row in split.baseline
        for doc_id in row
        if doc_id
    )
    priors = load_priors(args.prior_index)
    documents, document_frequency = load_documents(args.contexts_zip, needed_ids, priors)
    complement_matrix, complement_target, feature_names = augment_features(
        complement, documents, document_frequency, memory_maps["complement"], args.candidate_limit
    )
    holdout_matrix, holdout_target, holdout_names = augment_features(
        holdout, documents, document_frequency, memory_maps["holdout"], args.candidate_limit
    )
    public_matrix, _, public_names = augment_features(
        public, documents, document_frequency, memory_maps["public"], args.candidate_limit
    )
    if complement_target is None or holdout_target is None:
        raise RuntimeError("V16b requires complement and holdout references")
    if feature_names != holdout_names or feature_names != public_names:
        raise RuntimeError("V16b feature schema mismatch")

    # Keep the original selector implementation, but expose only c35 policies
    # plus the explicitly conservative memory slot-5 gate.
    import v11_tiny_ltr as ltr_module

    original_policy_grid = ltr_module.policy_grid
    ltr_module.policy_grid = lambda: memory_slot5_policy_grid(original_policy_grid)
    try:
        groups = build_leak_groups(complement)
        complement_indices = np.arange(len(complement.qids), dtype=np.int32)
        config, policy, leaderboard, oof_scores, fold_by_question = select_model_and_policy(
            complement, complement_matrix, complement_target, feature_names,
            complement_indices, groups, args.candidate_limit, args.model_profile,
        )
    finally:
        ltr_module.policy_grid = original_policy_grid

    complement_predictions = apply_policy(
        complement, oof_scores, complement_indices, complement_matrix,
        feature_names, args.candidate_limit, policy,
    )
    complement_ltr = evaluate_policy(
        complement, complement_predictions, complement_indices, fold_by_question
    )
    complement_anchor = evaluate_policy(
        complement,
        baseline_predictions(
            complement, complement_indices, complement_matrix, feature_names,
            args.candidate_limit, False,
        ),
        complement_indices,
    )
    complement_exact = evaluate_policy(
        complement,
        baseline_predictions(
            complement, complement_indices, complement_matrix, feature_names,
            args.candidate_limit, True,
        ),
        complement_indices,
    )

    locked_model = train_model(
        config, complement_matrix, complement_target, complement_indices, args.candidate_limit
    )
    holdout_indices = np.arange(len(holdout.qids), dtype=np.int32)
    holdout_scores = model_scores(locked_model, holdout_matrix).reshape(
        len(holdout.qids), args.candidate_limit
    )
    holdout_predictions = apply_policy(
        holdout, holdout_scores, holdout_indices, holdout_matrix,
        feature_names, args.candidate_limit, policy,
    )
    if args.holdout_predictions is not None:
        write_json(args.holdout_predictions, holdout_predictions)
    holdout_ltr = evaluate_policy(holdout, holdout_predictions, holdout_indices)
    holdout_anchor = evaluate_policy(
        holdout,
        baseline_predictions(
            holdout, holdout_indices, holdout_matrix, feature_names,
            args.candidate_limit, False,
        ),
        holdout_indices,
    )
    holdout_exact = evaluate_policy(
        holdout,
        baseline_predictions(
            holdout, holdout_indices, holdout_matrix, feature_names,
            args.candidate_limit, True,
        ),
        holdout_indices,
    )
    complement_delta = float(complement_ltr["recall"] - complement_exact["recall"])
    holdout_delta = float(holdout_ltr["recall"] - holdout_exact["recall"])
    promoted = (
        complement_delta >= 0.0
        and holdout_delta >= args.minimum_holdout_delta
    )

    public_output: Path | None = None
    public_changes = 0
    public_predictions: dict[str, dict[str, list[str]]] | None = None
    if promoted:
        combined_matrix = np.vstack((complement_matrix, holdout_matrix))
        combined_target = np.concatenate((complement_target, holdout_target))
        combined_split_indices = np.arange(
            len(complement.qids) + len(holdout.qids), dtype=np.int32
        )
        final_model = train_model(
            config, combined_matrix, combined_target,
            combined_split_indices, args.candidate_limit,
        )
        public_indices = np.arange(len(public.qids), dtype=np.int32)
        public_scores = model_scores(final_model, public_matrix).reshape(
            len(public.qids), args.candidate_limit
        )
        public_predictions = apply_policy(
            public, public_scores, public_indices, public_matrix,
            feature_names, args.candidate_limit, policy,
        )
        public_output = args.output_dir / args.submission_name
        write_json(public_output, public_predictions)
        public_changes = sum(
            public_predictions[qid]["answer"] != public.baseline[index]
            for index, qid in enumerate(public.qids)
        )

    report = {
        "kind": "legalir_v16b_memory_feature_ltr",
        "leakage_policy": {
            "memory_features": "built from fold-isolated labelled questions only",
            "complement_selection": "grouped OOF after removing exact-memory queries",
            "holdout": "opened after model/policy lock",
            "public_labels_used": False,
            "excluded_exact_memory_qids": excluded_exact,
            "memory_labels_used_for_selector": False,
        },
        "model_compliance": {
            "retrieval_model_parameters": 330_000_000,
            "reranker_model_parameters": 760_000_000,
            "selector_parameters_upper_bound": 10_000,
            "graph_teacher_parameters": 0,
            "total_parameters_upper_bound": 1_090_010_000,
            "parameter_limit": 4_000_000_000,
            "under_4b": True,
        },
        "candidate_limit": args.candidate_limit,
        "memory_top_k": 5,
        "features": feature_names,
        "locked_config": config,
        "locked_policy": policy,
        "selection": {
            "anchor_c35": complement_anchor,
            "anchor_c35_exact": complement_exact,
            "memory_ltr_oof": complement_ltr,
            "delta_vs_anchor_exact_recall": complement_delta,
            "transition": transition_report(complement, complement_predictions, complement_indices),
        },
        "holdout": {
            "anchor_c35": holdout_anchor,
            "anchor_c35_exact": holdout_exact,
            "memory_ltr": holdout_ltr,
            "delta_vs_anchor_exact_recall": holdout_delta,
            "transition": transition_report(holdout, holdout_predictions, holdout_indices),
        },
        "promotion": {
            "minimum_holdout_delta": args.minimum_holdout_delta,
            "promoted": promoted,
            "public_submission": str(public_output.resolve()) if public_output else None,
            "public_queries_changed_vs_anchor": public_changes,
        },
        "model_leaderboard": leaderboard,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = args.output_dir / "v16b_memory_ltr_report.json"
    write_json(report_path, report)
    print(json.dumps({
        "report": str(report_path.resolve()),
        "selection": report["selection"],
        "holdout": report["holdout"],
        "promotion": report["promotion"],
        "under_4b": report["model_compliance"]["under_4b"],
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
