#!/usr/bin/env python3
"""Nested OOF trainer/evaluator for a 1025-parameter hidden-state head.

Qwen is never loaded here.  The frozen c67 hidden-state cache is read from
disk; only the linear head is fit on CUDA.  Labels are opened only after the
label-blind evaluator manifest has been written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e3b_residual_score_gate as e3b_gate  # noqa: E402
from hsr_contract import (  # noqa: E402
    EPSILON, HIDDEN_SIZE, HSR_KIND, HSR_VARIANT, MAX_TOKENS, HEAD_PARAMETERS,
    MIN_STANDARD_GAIN_VS_E3B, PARAMETER_LIMIT, REGULARISATIONS, TAUS, E3BFold, HiddenCache,
    load_e3b_folds, load_hidden_cache,
)
from qwen_c200_common import load_candidate_universe  # noqa: E402
from v27_a_common import (  # noqa: E402
    CANDIDATE_LIMIT, V27ContractError, build_regime_groups, json_sha256,
    load_manifest, read_json, recall_value, sha256_file, transition_metrics,
    write_json,
)


EVAL_KIND = f"{HSR_KIND}_nested_oof"
CHECKPOINT_VERSION = 1
BOOTSTRAP_SAMPLES = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "standard", "stress"), default="preflight")
    parser.add_argument("--stage1", type=Path, required=True); parser.add_argument("--stage2", type=Path, required=True); parser.add_argument("--anchor", type=Path, required=True); parser.add_argument("--reference", type=Path, required=True); parser.add_argument("--contexts-zip", type=Path, required=True)
    parser.add_argument("--e1-manifest", type=Path, required=True); parser.add_argument("--e3b-standard-report", type=Path, required=True); parser.add_argument("--e3b-stress-report", type=Path, required=True); parser.add_argument("--hidden-cache-root", type=Path, required=True); parser.add_argument("--gate-source", type=Path, required=True); parser.add_argument("--gate-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--standard-report", type=Path); parser.add_argument("--checkpoint-path", type=Path); parser.add_argument("--report-path", type=Path); parser.add_argument("--device", choices=("cuda",), default="cuda"); parser.add_argument("--max-lbfgs-iterations", type=int, default=200); parser.add_argument("--resume", dest="resume", action="store_true"); parser.add_argument("--no-resume", dest="resume", action="store_false"); parser.set_defaults(resume=True); parser.add_argument("--force", action="store_true")
    return parser.parse_args()


class SplitView:
    pass


def _view(split: Any, qids: Sequence[str], candidates_by_qid: Mapping[str, Sequence[str]] | None = None) -> SplitView:
    positions = {str(qid): index for index, qid in enumerate(split.qids)}; indices = [positions[str(qid)] for qid in qids]
    result = SplitView(); result.qids = [str(qid) for qid in qids]; result.questions = [split.questions[index] for index in indices]; result.anchor = [split.anchor[index] for index in indices]; result.references = [split.references[index] for index in indices]
    result.candidates = [list(candidates_by_qid[str(qid)]) if candidates_by_qid is not None else list(split.candidates[index]) for qid, index in zip(qids, indices, strict=True)]
    return result


def _c67_by_qid(universe: Any) -> dict[str, list[str]]:
    return {str(qid): list(universe.candidates[str(qid)][:67]) for qid in universe.split.qids}


def _base_predictions(split_c35: Any, fold: E3BFold, qids: Sequence[str]) -> dict[str, dict[str, list[str]]]:
    view = _view(split_c35, qids)
    predictions, decisions = e3b_gate.apply_policy(view, list(range(len(qids))), fold.calibration_by_seed if set(qids) == set(fold.calibration_qids) else fold.validation_by_seed, {}, fold.selected_policy)
    for qid in qids:
        decision = decisions[str(qid)]
        if decision.get("replacement") is not None and decision["replacement"] != predictions[str(qid)]["answer"][4]:
            raise V27ContractError(f"E3B baseline reconstruction mismatch for {qid}")
    return predictions


def _base_from_frozen_e3b_decisions(split_c35: Any, fold: E3BFold) -> dict[str, dict[str, list[str]]]:
    """Recreate validation baseline from the immutable E3B decision ledger."""
    positions = {str(qid): index for index, qid in enumerate(split_c35.qids)}
    if set(fold.validation_qids) != set(fold.reported_validation_decisions):
        raise V27ContractError("Frozen E3B validation decisions differ from the fold manifest")
    predictions: dict[str, dict[str, list[str]]] = {}
    for qid in fold.validation_qids:
        anchor = list(split_c35.anchor[positions[qid]])
        decision = fold.reported_validation_decisions[qid]
        replacement = decision["replacement"] if bool(decision["changed"]) else anchor[4]
        answer = anchor[:4] + [str(replacement)]
        if len(answer) != 5 or len(set(answer)) != 5:
            raise V27ContractError(f"{qid}: frozen E3B decision violates the five-document contract")
        predictions[qid] = {"answer": answer}
    return predictions


def _base_for_partition(split_c35: Any, fold: E3BFold, partition: str) -> dict[str, dict[str, list[str]]]:
    if partition == "validation":
        return _base_from_frozen_e3b_decisions(split_c35, fold)
    qids = fold.calibration_qids if partition == "calibration" else fold.validation_qids
    view = _view(split_c35, qids)
    scores = fold.calibration_by_seed if partition == "calibration" else fold.validation_by_seed
    predictions, _ = e3b_gate.apply_policy(view, list(range(len(qids))), scores, {}, fold.selected_policy)
    return predictions


def _training_examples(split: Any, hidden: HiddenCache, qids: Sequence[str], c67_by_qid: Mapping[str, Sequence[str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[list[int], list[int], float]], dict[str, Any]]:
    vectors: list[np.ndarray] = []; native: list[float] = []; labels: list[float] = []; groups: list[tuple[list[int], list[int], float]] = []
    positions = {str(qid): index for index, qid in enumerate(split.qids)}; audit = {"qids_total": len(qids), "qids_with_positive": 0, "qids_without_positive": 0, "qids_used": 0, "qids_skipped_no_valid_examples": 0, "positives": 0, "negatives": 0}
    for qid in map(str, qids):
        gold = set(map(str, split.references[positions[qid]])); pool = list(c67_by_qid[qid]); rows = hidden.scores[qid]; valid = [doc for doc in pool if bool(rows[doc].get("evidence_valid"))]
        positives = [doc for doc in valid if doc in gold]; negative_pool = [doc for doc in valid if doc not in gold]
        if positives:
            audit["qids_with_positive"] += 1
        else:
            audit["qids_without_positive"] += 1
        negative_pool.sort(key=lambda doc: (-float(rows[doc]["native_yes_minus_no"]), pool.index(doc), doc)); top = negative_pool[:4]; remaining = negative_pool[4:]
        if remaining:
            center = len(remaining) // 2; middle = remaining[max(0, center - 2):center + 2]
        else:
            middle = []
        negatives = list(dict.fromkeys(top + middle))[:8]
        if not positives and not negatives:
            audit["qids_skipped_no_valid_examples"] += 1; continue
        pos_indices: list[int] = []; neg_indices: list[int] = []
        for doc in positives:
            row = rows[doc]; vectors.append(np.asarray(row["hidden"], dtype=np.float32)); native.append(float(row["native_yes_minus_no"])); labels.append(1.0); pos_indices.append(len(labels) - 1)
        for doc in negatives:
            row = rows[doc]; vectors.append(np.asarray(row["hidden"], dtype=np.float32)); native.append(float(row["native_yes_minus_no"])); labels.append(0.0); neg_indices.append(len(labels) - 1)
        groups.append((pos_indices, neg_indices, 1.0 / max(len(gold), 1))); audit["qids_used"] += 1; audit["positives"] += len(positives); audit["negatives"] += len(negatives)
    if not vectors:
        raise V27ContractError("HSR T has no valid hidden-state training examples")
    return np.stack(vectors).astype(np.float32), np.asarray(native, dtype=np.float32), np.asarray(labels, dtype=np.float32), groups, audit


@dataclass(frozen=True)
class GroupedLossLayout:
    """Fixed GPU indexing tensors for the query-weighted HSR objective."""

    positive_indices: Any
    positive_mask: Any
    negative_indices: Any
    negative_mask: Any
    sample_indices: Any
    sample_targets: Any
    sample_mask: Any
    query_weights: Any
    denominator: float


def _build_grouped_loss_layout(groups: Sequence[tuple[list[int], list[int], float]], *, device: str) -> GroupedLossLayout:
    """Pad query groups once so every L-BFGS closure uses batched GPU kernels."""
    import torch

    if not groups:
        raise V27ContractError("HSR training loss has no query groups")
    group_count = len(groups)
    positive_width = max(1, max(len(positive) for positive, _, _ in groups))
    negative_width = max(1, max(len(negative) for _, negative, _ in groups))
    sample_width = positive_width + negative_width
    positive_indices = torch.zeros((group_count, positive_width), dtype=torch.long, device=device)
    negative_indices = torch.zeros((group_count, negative_width), dtype=torch.long, device=device)
    sample_indices = torch.zeros((group_count, sample_width), dtype=torch.long, device=device)
    positive_mask = torch.zeros((group_count, positive_width), dtype=torch.bool, device=device)
    negative_mask = torch.zeros((group_count, negative_width), dtype=torch.bool, device=device)
    sample_mask = torch.zeros((group_count, sample_width), dtype=torch.bool, device=device)
    sample_targets = torch.zeros((group_count, sample_width), dtype=torch.float32, device=device)
    query_weights = torch.empty(group_count, dtype=torch.float32, device=device)
    for index, (positive, negative, query_weight) in enumerate(groups):
        if positive:
            positive_values = torch.as_tensor(positive, dtype=torch.long, device=device)
            positive_indices[index, :len(positive)] = positive_values
            positive_mask[index, :len(positive)] = True
            sample_indices[index, :len(positive)] = positive_values
            sample_targets[index, :len(positive)] = 1.0
            sample_mask[index, :len(positive)] = True
        if negative:
            negative_values = torch.as_tensor(negative, dtype=torch.long, device=device)
            negative_indices[index, :len(negative)] = negative_values
            negative_mask[index, :len(negative)] = True
            start = len(positive)
            sample_indices[index, start:start + len(negative)] = negative_values
            sample_mask[index, start:start + len(negative)] = True
        query_weights[index] = float(query_weight)
    denominator = float(query_weights.sum().item())
    if denominator <= 0.0:
        raise V27ContractError("HSR training loss has no positive query weight")
    return GroupedLossLayout(
        positive_indices=positive_indices,
        positive_mask=positive_mask,
        negative_indices=negative_indices,
        negative_mask=negative_mask,
        sample_indices=sample_indices,
        sample_targets=sample_targets,
        sample_mask=sample_mask,
        query_weights=query_weights,
        denominator=denominator,
    )


def _grouped_loss(scores: Any, layout: GroupedLossLayout, weight: Any, bias: Any, regularisation: float) -> Any:
    """Exact batched form of the locked HSR pointwise/pairwise objective."""
    import torch

    positive_scores = scores[layout.positive_indices]
    negative_scores = scores[layout.negative_indices]
    pair_mask = layout.negative_mask[:, :, None] & layout.positive_mask[:, None, :]
    pair_values = torch.nn.functional.softplus(negative_scores[:, :, None] - positive_scores[:, None, :])
    pair_count = pair_mask.sum(dim=(1, 2))
    pairwise = (pair_values * pair_mask).sum(dim=(1, 2)) / pair_count.clamp_min(1)
    pairwise = torch.where(pair_count > 0, pairwise, torch.zeros_like(pairwise))
    sample_scores = scores[layout.sample_indices]
    point_values = torch.nn.functional.binary_cross_entropy_with_logits(
        sample_scores, layout.sample_targets.to(dtype=scores.dtype), reduction="none"
    )
    sample_count = layout.sample_mask.sum(dim=1)
    pointwise = (point_values * layout.sample_mask).sum(dim=1) / sample_count.clamp_min(1)
    weighted = layout.query_weights.to(dtype=scores.dtype) * (0.5 * pairwise + 0.5 * pointwise)
    return weighted.sum() / layout.denominator + float(regularisation) * (weight.square().sum() + bias.square())


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_name(f".{path.name}.tmp")
    torch.save(dict(payload), temporary); os.replace(temporary, path)


def _move_optimizer_state(optimizer: Any, device: str) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if hasattr(value, "to"):
                state[key] = value.to(device)


def _fit_head(*, h: np.ndarray, z0: np.ndarray, y: np.ndarray, groups: list[tuple[list[int], list[int], float]], regularisation: float, path: Path, lineage_sha256: str, train_qid_hash: str, device: str, max_iterations: int, resume: bool) -> dict[str, Any]:
    import torch
    if not torch.cuda.is_available() or device != "cuda":
        raise RuntimeError("HSR head fits are configured for CUDA")
    h_tensor = torch.as_tensor(h, dtype=torch.float32, device=device); z_tensor = torch.as_tensor(z0, dtype=torch.float32, device=device); params = torch.nn.Parameter(torch.zeros(HIDDEN_SIZE, dtype=torch.float32, device=device)); bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32, device=device)); optimizer = torch.optim.LBFGS([params, bias], lr=1.0, max_iter=1, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1.0e-8, tolerance_change=1.0e-10)
    start_iteration = 0; history: list[dict[str, Any]] = []
    if path.is_file():
        if not resume:
            raise FileExistsError(f"HSR head checkpoint exists with --no-resume: {path}")
        state = torch.load(path, map_location=device, weights_only=False)
        if state.get("kind") != f"{HSR_KIND}_head_checkpoint" or state.get("lineage_sha256") != lineage_sha256 or state.get("train_qid_hash") != train_qid_hash or abs(float(state.get("regularisation", -1.0)) - regularisation) > EPSILON:
            raise V27ContractError(f"HSR head checkpoint lineage/config mismatch: {path}")
        params.data.copy_(state["weight"].to(device)); bias.data.copy_(state["bias"].to(device)); optimizer.load_state_dict(state["optimizer"]); _move_optimizer_state(optimizer, device); start_iteration = int(state.get("iteration", 0)); history = list(state.get("history", []))
        if bool(state.get("complete")):
            return {"weight": params.detach().cpu().numpy(), "bias": float(bias.detach().cpu().item()), "regularisation": regularisation, "iteration": start_iteration, "checkpoint": str(path.resolve()), "history": history, "complete": True, "train_qid_hash": train_qid_hash, "lineage_sha256": lineage_sha256}
    x_norm = h_tensor / torch.clamp(torch.linalg.vector_norm(h_tensor, dim=1, keepdim=True), min=1.0e-8)
    layout = _build_grouped_loss_layout(groups, device=device)

    def objective() -> Any:
        score = z_tensor + x_norm.matmul(params) + bias
        return _grouped_loss(score, layout, params, bias, regularisation)

    def closure():
        optimizer.zero_grad(set_to_none=True); loss = objective()
        if not torch.isfinite(loss):
            raise V27ContractError("HSR head produced non-finite loss")
        loss.backward()
        if not torch.isfinite(params.grad).all() or not torch.isfinite(bias.grad):
            raise V27ContractError("HSR head produced non-finite gradient")
        return loss

    def logged_loss() -> float:
        with torch.no_grad():
            loss = objective()
        if not torch.isfinite(loss):
            raise V27ContractError("HSR head produced non-finite loss")
        return float(loss.detach().cpu().item())

    for iteration in range(start_iteration + 1, max_iterations + 1):
        before = logged_loss(); optimizer.step(closure); after = logged_loss(); history.append({"iteration": iteration, "loss_before": before, "loss_after": after})
        if iteration % 10 == 0 or iteration == max_iterations:
            _atomic_torch_save({"kind": f"{HSR_KIND}_head_checkpoint", "schema_version": CHECKPOINT_VERSION, "lineage_sha256": lineage_sha256, "train_qid_hash": train_qid_hash, "regularisation": regularisation, "iteration": iteration, "complete": iteration == max_iterations, "weight": params.detach().cpu(), "bias": bias.detach().cpu(), "optimizer": optimizer.state_dict(), "history": history}, path)
    return {"weight": params.detach().cpu().numpy(), "bias": float(bias.detach().cpu().item()), "regularisation": regularisation, "iteration": max_iterations, "checkpoint": str(path.resolve()), "history": history, "complete": True, "train_qid_hash": train_qid_hash, "lineage_sha256": lineage_sha256}


def _head_scores(hidden: HiddenCache, qids: Sequence[str], c67_by_qid: Mapping[str, Sequence[str]], fit: Mapping[str, Any] | None) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}; weight = None if fit is None else np.asarray(fit["weight"], dtype=np.float32); bias = 0.0 if fit is None else float(fit["bias"])
    for qid in map(str, qids):
        rows = hidden.scores[qid]; current: dict[str, float] = {}
        for doc in c67_by_qid[qid]:
            row = rows[doc]
            if not bool(row.get("evidence_valid")):
                continue
            value = float(row["native_yes_minus_no"])
            if weight is not None:
                vector = np.asarray(row["hidden"], dtype=np.float32); vector /= max(float(np.linalg.norm(vector)), 1.0e-8); value += float(np.dot(weight, vector) + bias)
            current[doc] = value
        result[qid] = current
    return result


def _select_slot5(split67: Any, qids: Sequence[str], base: Mapping[str, Mapping[str, Sequence[str]]], scores: Mapping[str, Mapping[str, float]], c67_by_qid: Mapping[str, Sequence[str]], tau: float) -> tuple[dict[str, dict[str, list[str]]], dict[str, Any]]:
    predictions: dict[str, dict[str, list[str]]] = {}; decisions: dict[str, Any] = {}
    for qid in map(str, qids):
        anchor = [str(value) for value in split67.anchor[list(map(str, split67.qids)).index(qid)]]; answer = [str(value) for value in base[qid]["answer"]]; incumbent = answer[4]
        if answer[:4] != anchor[:4] or len(answer) != 5 or len(set(answer)) != 5 or not set(answer).issubset(set(c67_by_qid[qid])):
            raise V27ContractError(f"{qid}: E3B incumbent violates frozen c67/top-4 contract")
        incumbent_score = scores.get(qid, {}).get(incumbent); candidates = [(doc, float(scores[qid][doc])) for doc in c67_by_qid[qid] if doc not in set(anchor[:4]) | {incumbent} and doc in scores.get(qid, {})]
        candidates.sort(key=lambda item: (-item[1], c67_by_qid[qid].index(item[0]), item[0])); winner = candidates[0] if candidates else None; reason = "ok"
        if incumbent_score is None:
            reason = "incumbent_evidence_missing"
        elif winner is None:
            reason = "no_valid_challenger"
        elif not (winner[1] > 0.0 and winner[1] - incumbent_score > float(tau) + EPSILON):
            reason = "threshold_abstain"
        else:
            answer[4] = winner[0]
        predictions[qid] = {"answer": answer}; decisions[qid] = {"incumbent": incumbent, "replacement": None if answer[4] == incumbent else answer[4], "changed": answer != base[qid]["answer"], "tau": float(tau), "reason": reason, "challenger_score": None if winner is None else winner[1], "incumbent_score": incumbent_score}
    return predictions, decisions


def _choose_native(split67: Any, qids: Sequence[str], base: Mapping[str, Mapping[str, Sequence[str]]], native_scores: Mapping[str, Mapping[str, float]], c67_by_qid: Mapping[str, Sequence[str]]) -> tuple[float, list[dict[str, Any]]]:
    trials: list[dict[str, Any]] = []
    for tau in TAUS:
        pred, _ = _select_slot5(split67, qids, base, native_scores, c67_by_qid, tau); metrics = transition_metrics(split67, pred); trials.append({"tau": tau, "metrics": {key: value for key, value in metrics.items() if key != "rows"}})
    selected = max(trials, key=lambda row: (float(row["metrics"]["recall"]), -int(row["metrics"]["destructive_changes"]), -int(row["metrics"]["changed_queries"]), float(row["tau"])))
    return float(selected["tau"]), trials


def _choose_head(split67: Any, qids: Sequence[str], base: Mapping[str, Mapping[str, Sequence[str]]], scores_by_lambda: Mapping[float, Mapping[str, Mapping[str, float]]], c67_by_qid: Mapping[str, Sequence[str]]) -> tuple[tuple[float, float] | None, list[dict[str, Any]]]:
    baseline_metrics = transition_metrics(split67, base); trials: list[dict[str, Any]] = [{"kind": "e3b_baseline", "lambda": None, "tau": None, "metrics": {key: value for key, value in baseline_metrics.items() if key != "rows"}}]
    for regularisation, scores in scores_by_lambda.items():
        for tau in TAUS:
            pred, _ = _select_slot5(split67, qids, base, scores, c67_by_qid, tau); metrics = transition_metrics(split67, pred); trials.append({"kind": "hidden_state_head", "lambda": regularisation, "tau": tau, "metrics": {key: value for key, value in metrics.items() if key != "rows"}})
    def key(row: Mapping[str, Any]) -> tuple[float, int, int, int, float, float]:
        is_base = 1 if row["kind"] == "e3b_baseline" else 0; lam = -1.0 if row["lambda"] is None else float(row["lambda"]); tau = -1.0 if row["tau"] is None else float(row["tau"])
        return (float(row["metrics"]["recall"]), -int(row["metrics"]["destructive_changes"]), -int(row["metrics"]["changed_queries"]), is_base, lam, tau)
    selected = max(trials, key=key)
    return (None if selected["kind"] == "e3b_baseline" else (float(selected["lambda"]), float(selected["tau"]))), trials


def _group_bootstrap_lower(split: Any, qids: Sequence[str], candidate: Mapping[str, Mapping[str, Sequence[str]]], baseline: Mapping[str, Mapping[str, Sequence[str]]], groups: Mapping[str, int], seed: int) -> float:
    positions = {str(qid): index for index, qid in enumerate(split.qids)}; values_by_group: dict[int, list[float]] = {}
    for qid in map(str, qids):
        gold = split.references[positions[qid]]; after = recall_value(candidate[qid]["answer"], gold); before = recall_value(baseline[qid]["answer"], gold); values_by_group.setdefault(int(groups[qid]), []).append(float(after - before))
    if not values_by_group:
        return 0.0
    group_values = np.asarray([float(np.mean(values)) for values in values_by_group.values()], dtype=np.float64); rng = np.random.default_rng(int(seed)); samples = rng.choice(group_values, size=(BOOTSTRAP_SAMPLES, len(group_values)), replace=True).mean(axis=1); return float(np.quantile(samples, 0.025))


def _fold_run(*, split_c35: Any, split67: Any, hidden: HiddenCache, fold: E3BFold, c67_by_qid: Mapping[str, Sequence[str]], output_dir: Path, lineage_sha256: str, device: str, max_iterations: int, resume: bool, groups: Mapping[str, int]) -> dict[str, Any]:
    train_h, train_z, train_y, train_groups, train_audit = _training_examples(split67, hidden, fold.train_qids, c67_by_qid); train_hash = json_sha256(list(fold.train_qids)); fit_info: dict[float, dict[str, Any]] = {}
    for regularisation in REGULARISATIONS:
        checkpoint = output_dir / "head_weights" / fold.regime / f"inner_{fold.inner_seed}_fold_{fold.outer_fold + 1}" / f"head_lambda_{regularisation:g}.pt"; fit_info[regularisation] = _fit_head(h=train_h, z0=train_z, y=train_y, groups=train_groups, regularisation=regularisation, path=checkpoint, lineage_sha256=lineage_sha256, train_qid_hash=train_hash, device=device, max_iterations=max_iterations, resume=resume)
    c_qids, v_qids = list(fold.calibration_qids), list(fold.validation_qids); base_c = _base_for_partition(split_c35, fold, "calibration"); base_v = _base_for_partition(split_c35, fold, "validation"); native_scores_c = _head_scores(hidden, c_qids, c67_by_qid, None); native_scores_v = _head_scores(hidden, v_qids, c67_by_qid, None); native_tau, native_trace = _choose_native(_view(split67, c_qids, c67_by_qid), c_qids, base_c, native_scores_c, c67_by_qid); native_c, _ = _select_slot5(_view(split67, c_qids, c67_by_qid), c_qids, base_c, native_scores_c, c67_by_qid, native_tau); native_v, native_v_decisions = _select_slot5(_view(split67, v_qids, c67_by_qid), v_qids, base_v, native_scores_v, c67_by_qid, native_tau); scores_by_lambda_c = {float(lam): _head_scores(hidden, c_qids, c67_by_qid, info) for lam, info in fit_info.items()}; scores_by_lambda_v = {float(lam): _head_scores(hidden, v_qids, c67_by_qid, info) for lam, info in fit_info.items()}; chosen, head_trace = _choose_head(_view(split67, c_qids, c67_by_qid), c_qids, base_c, scores_by_lambda_c, c67_by_qid)
    if chosen is None:
        selected_v, selected_decisions = base_v, {qid: {"changed": False, "policy": "e3b_baseline"} for qid in v_qids}; selected_c = base_c; selected_policy = {"kind": "e3b_baseline"}
    else:
        lam, tau = chosen; selected_c, _ = _select_slot5(_view(split67, c_qids, c67_by_qid), c_qids, base_c, scores_by_lambda_c[lam], c67_by_qid, tau); selected_v, selected_decisions = _select_slot5(_view(split67, v_qids, c67_by_qid), v_qids, base_v, scores_by_lambda_v[lam], c67_by_qid, tau); selected_policy = {"kind": "hidden_state_head", "regularisation": lam, "tau": tau}
    baseline_c_metrics = transition_metrics(_view(split67, c_qids, c67_by_qid), base_c); selected_c_metrics = transition_metrics(_view(split67, c_qids, c67_by_qid), selected_c); baseline_v_metrics = transition_metrics(_view(split67, v_qids, c67_by_qid), base_v); selected_v_metrics = transition_metrics(_view(split67, v_qids, c67_by_qid), selected_v); native_v_metrics = transition_metrics(_view(split67, v_qids, c67_by_qid), native_v)
    baseline_report_match = not fold.reported_validation_metrics or (abs(float(fold.reported_validation_metrics.get("recall", baseline_v_metrics["recall"])) - float(baseline_v_metrics["recall"])) <= 1.0e-12 and int(fold.reported_validation_metrics.get("changed_queries", baseline_v_metrics["changed_queries"])) == int(baseline_v_metrics["changed_queries"]))
    if not baseline_report_match:
        raise V27ContractError(f"{fold.regime}/seed={fold.inner_seed}/fold={fold.outer_fold}: reconstructed E3B baseline differs from report")
    return {"regime": fold.regime, "inner_seed": fold.inner_seed, "outer_fold": fold.outer_fold, "train_qids": list(fold.train_qids), "calibration_qids": c_qids, "validation_qids": v_qids, "train_audit": train_audit, "heads": {str(lam): {key: value for key, value in info.items() if key not in {"weight"}} for lam, info in fit_info.items()}, "baseline_calibration": {key: value for key, value in baseline_c_metrics.items() if key != "rows"}, "selected_calibration": {key: value for key, value in selected_c_metrics.items() if key != "rows"}, "baseline_validation": {key: value for key, value in baseline_v_metrics.items() if key != "rows"}, "validation": {key: value for key, value in selected_v_metrics.items() if key != "rows"}, "native_validation": {key: value for key, value in native_v_metrics.items() if key != "rows"}, "native_tau": native_tau, "native_trace": native_trace, "selected_policy": selected_policy, "head_trace": head_trace, "baseline_report_match": baseline_report_match, "paired_bootstrap_lower": _group_bootstrap_lower(split67, v_qids, selected_v, base_v, groups, seed=fold.inner_seed + fold.outer_fold), "selected_predictions": selected_v, "native_predictions": native_v, "decisions": selected_decisions, "native_decisions": native_v_decisions}


def _historical(report: Mapping[str, Any], regime: str, seed: int) -> Mapping[str, Any] | None:
    runs = report.get("regimes", {}).get(regime, []) if isinstance(report.get("regimes"), Mapping) else []
    return next((run for run in runs if isinstance(run, Mapping) and int(run.get("inner_seed", -1)) == int(seed)), None)


def _gate(runs: Sequence[Mapping[str, Any]], *, standard: bool) -> dict[str, Any]:
    deltas = [float(run["metrics"]["recall"]) - float(run["baseline_metrics"]["recall"]) for run in runs]; lower = [float(run["paired_bootstrap_lower"]) for run in runs]; minima = min(deltas) if deltas else -math.inf; rescues = sum(int(run["metrics"]["successful_rescues"]) for run in runs); destructive = sum(int(run["metrics"]["destructive_changes"]) for run in runs); required = MIN_STANDARD_GAIN_VS_E3B if standard else 0.0; per_seed = [(delta >= required - EPSILON and ci > 0.0 and bool(run.get("not_below_historical", True)) and int(run["metrics"]["successful_rescues"]) > int(run["metrics"]["destructive_changes"])) for delta, ci, run in zip(deltas, lower, runs, strict=True)]
    return {"per_seed_pass": per_seed, "pass": bool(per_seed and all(per_seed)), "minimum_gain_vs_e3b": minima, "required_gain_vs_e3b": required, "minimum_paired_bootstrap_lower": min(lower) if lower else -math.inf, "successful_rescues": rescues, "destructive_changes": destructive, "rescue_minus_destructive": rescues - destructive}


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path, label in ((args.stage1, "Stage-1"), (args.stage2, "Stage-2"), (args.anchor, "V16b anchor"), (args.contexts_zip, "contexts ZIP"), (args.e1_manifest, "E1 manifest"), (args.e3b_standard_report, "E3B standard report"), (args.e3b_stress_report, "E3B stress report"), (args.gate_source, "gate source"), (args.gate_config, "gate config")):
        if label == "contexts ZIP":
            if not path.is_file(): raise FileNotFoundError(path)
        elif not path.is_file() or any(term in str(path).replace("\\", "/").casefold() for term in ("holdout", "submission", "locked", "public") ):
            raise V27ContractError(f"HSR complement input invalid: {label} {path}")
    label_blind = load_candidate_universe(stage1=args.stage1, stage2=args.stage2, anchor=args.anchor, reference=None); c67_by_qid = _c67_by_qid(label_blind); hidden = load_hidden_cache(args.hidden_cache_root, expected_qids=label_blind.split.qids, c67_by_qid=c67_by_qid); current_stage_hashes = {"stage1_sha256": sha256_file(args.stage1), "stage2_sha256": sha256_file(args.stage2), "anchor_sha256": sha256_file(args.anchor), "contexts_zip_sha256": sha256_file(args.contexts_zip)}
    if any(hidden.lineage.get(key) != value for key, value in current_stage_hashes.items()) or hidden.lineage.get("variant") != HSR_VARIANT or int(hidden.lineage.get("max_tokens", -1)) != MAX_TOKENS or int(hidden.lineage.get("hidden_size", -1)) != HIDDEN_SIZE:
        raise V27ContractError("Hidden-state cache lineage does not match current frozen inputs")
    standard_folds, standard_info = load_e3b_folds(report_path=args.e3b_standard_report, expected_mode="standard", split=label_blind.split); stress_folds, stress_info = load_e3b_folds(report_path=args.e3b_stress_report, expected_mode="stress", split=label_blind.split)
    lineage = {"kind": EVAL_KIND, "schema_version": 1, "mode": args.mode, "stage1_sha256": sha256_file(args.stage1), "stage2_sha256": sha256_file(args.stage2), "anchor_sha256": sha256_file(args.anchor), "contexts_zip_sha256": sha256_file(args.contexts_zip), "e1_manifest_sha256": sha256_file(args.e1_manifest), "e3b_standard_report": standard_info, "e3b_stress_report": stress_info, "hidden_cache_lineage_sha256": hidden.lineage_sha256, "hidden_cache_metadata": list(hidden.metadata_paths), "gate_source_sha256": sha256_file(args.gate_source), "gate_config_sha256": sha256_file(args.gate_config), "regularisations": list(REGULARISATIONS), "taus": list(TAUS), "head_parameters": HEAD_PARAMETERS, "max_lbfgs_iterations": args.max_lbfgs_iterations, "new_policy_holdout_validated": False, "holdout_opened": False, "public_labels_used": False}
    experiment_lineage = {key: value for key, value in lineage.items() if key != "mode"}; lineage_sha = json_sha256(lineage); experiment_sha = json_sha256(experiment_lineage); args.output_dir.mkdir(parents=True, exist_ok=True); manifest_path = args.output_dir / f"hsr_{args.mode}_label_blind_manifest.json"; write_json(manifest_path, {"kind": f"{HSR_KIND}_eval_manifest", "schema_version": 1, "lineage": lineage, "lineage_sha256": lineage_sha, "experiment_lineage_sha256": experiment_sha, "labels_opened_after_manifest": False, "holdout_opened": False, "public_labels_used": False})
    if args.mode == "preflight":
        return {"kind": f"{EVAL_KIND}_preflight", "stage": "label_blind_preflight", "lineage": lineage, "lineage_sha256": lineage_sha, "experiment_lineage_sha256": experiment_sha, "hidden_cache": {"pairs": hidden.pairs, "hidden_size": hidden.hidden_size, "hidden_dtype": hidden.hidden_dtype, "parameter_ledger": dict(hidden.parameter_ledger)}, "e3b_folds": {"standard": {key: len(value) for key, value in standard_folds.items()}, "stress": {key: len(value) for key, value in stress_folds.items()}}, "parameter_budget": {"head_parameters": HEAD_PARAMETERS, "parameter_limit": PARAMETER_LIMIT, "under_4b": hidden.parameter_ledger.get("under_4b") is True}, "labels_opened_after_manifest": False, "holdout_opened": False, "public_labels_used": False, "next_action": "Run MODE=standard only after verifying the hidden cache is complete."}
    if args.max_lbfgs_iterations < 1 or args.max_lbfgs_iterations > 200:
        raise ValueError("--max-lbfgs-iterations must be in [1,200]")
    if args.mode == "stress":
        if args.standard_report is None: raise ValueError("HSR stress requires --standard-report")
        prior = read_json(args.standard_report)
        if not isinstance(prior, Mapping) or prior.get("promotion_to_stress_allowed") is not True or prior.get("experiment_lineage_sha256") != experiment_sha:
            raise V27ContractError("HSR stress is blocked by the standard gate or lineage")
    split = load_candidate_universe(stage1=args.stage1, stage2=args.stage2, anchor=args.anchor, reference=args.reference).split; split67 = _view(split, split.qids, c67_by_qid); evidence = load_manifest(args.e1_manifest); group_arrays = build_regime_groups(split, evidence); regime_names = ("standard",) if args.mode == "standard" else ("law_source_family", "query_template_family", "near_duplicate_family"); report_source = read_json(args.e3b_standard_report if args.mode == "standard" else args.e3b_stress_report); fold_map = standard_folds if args.mode == "standard" else stress_folds; checkpoint_path = args.checkpoint_path or args.output_dir / f"hsr_{args.mode}_progress.json"; checkpoint = {"kind": f"{EVAL_KIND}_checkpoint", "schema_version": CHECKPOINT_VERSION, "lineage_sha256": lineage_sha, "experiment_lineage_sha256": experiment_sha, "mode": args.mode, "completed_folds": {}, "complete": False}
    if checkpoint_path.exists():
        if not args.resume: raise FileExistsError(f"HSR checkpoint exists with --no-resume: {checkpoint_path}")
        checkpoint = read_json(checkpoint_path)
        if not isinstance(checkpoint, Mapping) or checkpoint.get("kind") != f"{EVAL_KIND}_checkpoint" or checkpoint.get("lineage_sha256") != lineage_sha:
            raise V27ContractError("HSR nested checkpoint lineage mismatch")
        checkpoint = dict(checkpoint)
    reports: dict[str, list[dict[str, Any]]] = {}
    for regime in regime_names:
        seeds = sorted({int(fold.inner_seed) for fold in fold_map.get(regime, [])}); regime_reports: list[dict[str, Any]] = []
        for seed in seeds:
            folds = sorted([fold for fold in fold_map[regime] if int(fold.inner_seed) == seed], key=lambda item: item.outer_fold); fold_reports: list[dict[str, Any]] = []; selected_oof: dict[str, dict[str, list[str]]] = {}; baseline_oof: dict[str, dict[str, list[str]]] = {}; native_oof: dict[str, dict[str, list[str]]] = {}
            for fold in folds:
                key = f"{regime}|inner={seed}|fold={fold.outer_fold}"; saved = checkpoint["completed_folds"].get(key)
                if saved is not None:
                    fold_report = saved["fold_report"]
                else:
                    fold_report = _fold_run(split_c35=split, split67=split67, hidden=hidden, fold=fold, c67_by_qid=c67_by_qid, output_dir=args.output_dir / args.mode, lineage_sha256=lineage_sha, device=args.device, max_iterations=args.max_lbfgs_iterations, resume=args.resume, groups={str(qid): int(group_arrays[regime][index]) for index, qid in enumerate(split.qids)}); checkpoint["completed_folds"][key] = {"fold_report": fold_report}; write_json(checkpoint_path, checkpoint)
                fold_reports.append(fold_report); qids = fold_report["validation_qids"]
                selected_oof.update(fold_report["selected_predictions"])
                baseline_oof.update(_base_for_partition(split, fold, "validation"))
                native_oof.update(fold_report["native_predictions"])
            if set(selected_oof) != set(map(str, split.qids)):
                raise V27ContractError(f"HSR {regime}/seed={seed} OOF does not cover complement")
            selected_metrics = transition_metrics(split67, selected_oof); baseline_metrics = transition_metrics(split67, baseline_oof); native_metrics = transition_metrics(split67, native_oof); aggregate_groups = {str(qid): int(group_arrays[regime][index]) for index, qid in enumerate(split.qids)}; aggregate_lower = _group_bootstrap_lower(split67, split.qids, selected_oof, baseline_oof, aggregate_groups, seed=seed); historical = _historical(report_source, regime, seed); historical_metrics = None if historical is None else historical.get("metrics"); not_below_history = historical_metrics is None or float(selected_metrics["recall"]) >= float(historical_metrics.get("recall", -math.inf)) - EPSILON; regime_reports.append({"inner_seed": seed, "folds": fold_reports, "metrics": {key: value for key, value in selected_metrics.items() if key != "rows"}, "baseline_metrics": {key: value for key, value in baseline_metrics.items() if key != "rows"}, "native_metrics": {key: value for key, value in native_metrics.items() if key != "rows"}, "gain_vs_e3b": float(selected_metrics["recall"] - baseline_metrics["recall"]), "paired_bootstrap_lower": aggregate_lower, "historical_e3b_metrics": historical_metrics, "not_below_historical": not_below_history, "predictions": selected_oof})
        reports[regime] = regime_reports
    checkpoint["complete"] = True; write_json(checkpoint_path, checkpoint); gates = {regime: _gate(runs, standard=(regime == "standard")) for regime, runs in reports.items()}; standard_pass = bool(args.mode == "standard" and gates.get("standard", {}).get("pass")); stress_pass = bool(args.mode == "stress" and all(value.get("pass") is True for value in gates.values())); return {"kind": EVAL_KIND, "schema_version": 1, "status": "complement_nested_oof_only", "mode": args.mode, "split": "complement", "lineage": lineage, "lineage_sha256": lineage_sha, "experiment_lineage_sha256": experiment_sha, "label_blind_manifest": str(manifest_path.resolve()), "labels_opened_after_manifest": True, "holdout_opened": False, "public_labels_used": False, "checkpoint": {"path": str(checkpoint_path.resolve()), "completed_folds": len(checkpoint["completed_folds"]), "resumable": True}, "regimes": reports, "gates": gates, "checks": {"new_policy_holdout_validated": False, "stress_seen_during_design": True}, "promotion_to_stress_allowed": standard_pass, "promotion_to_public_allowed": stress_pass, "new_policy_holdout_validated": False, "next_action": "Do not overwrite accepted E3B/E4/E5 until a separate release contract is satisfied."}


def read_head_weight(path: Path) -> np.ndarray:
    import torch
    state = torch.load(path, map_location="cpu", weights_only=False); return np.asarray(state["weight"], dtype=np.float32)


def main() -> int:
    args = parse_args(); output = args.report_path or args.output_dir / f"hsr_{args.mode}_report.json"
    if output.exists() and not args.force:
        raise FileExistsError(f"HSR report exists: {output}")
    report = run(args); write_json(output, report); print(json.dumps({"output": str(output.resolve()), "mode": args.mode, "promotion_to_stress_allowed": report.get("promotion_to_stress_allowed", False), "promotion_to_public_allowed": report.get("promotion_to_public_allowed", False), "holdout_opened": False}, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
