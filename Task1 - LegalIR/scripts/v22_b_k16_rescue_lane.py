"""Registered V22-B: K16 complementary rescue lane.

This is one deliberately small, K16-only experiment after V22-A failed its
unchanged promotion contract.  It never loads or fine-tunes a CE, leaves the
V16b top four frozen, uses only the preflight's fixed E3B top-four shortlist,
and performs nested group-OOF calibration before any threshold decision.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIGS = ("B1024-D30", "B1024-D60", "B1536-D30", "B1536-D60", "B2304-D30", "B2304-D60")
REGIMES = ("standard_grouped", "law_family_ood", "query_template_ood", "near_duplicate_ood")
SEED = 22119
OUTER_FOLDS = 5

PREREGISTRATION = {
    "schema_version": 1,
    "stage": "V22-B_K16_complementary_rescue_lane",
    "scope": "the 598-query preflight-approved K16 subset only; no whole-cohort, K32, K64, slot-four, holdout, or submission claim",
    "candidate_policy": "immutable K16 E3B top-four shortlist; no candidate replacement; incumbent V16b slot five remains the fallback",
    "features": "six E3B pack scores, challenger-minus-incumbent six-pack deltas, pack-score dispersion, source rank, and E3B shortlist rank",
    "model": {"kind": "two-layer setwise micro-head", "hidden_dim": 32, "parameters_expected_under": 2_000},
    "semi_hard_negatives": {
        "definition": "for queries with a rescue retain non-positive shortlisted candidates below the best rescue and within a fixed score margin; otherwise retain ranks 2-3 rather than the top-ranked negative",
        "score_margin": 0.20,
        "maximum_per_query": 2,
        "false_negative_check": "every selected negative is checked against clean-complement QREL before loss construction",
    },
    "objective": {
        "classes": ["neutral", "destructive", "rescue"],
        "class_weights": [1.0, 3.0, 8.0],
        "cross_entropy_weight": 1.0,
        "signed_utility_huber_weight": 0.50,
        "multi_positive_lse_weight": 0.50,
        "utility": "P(rescue)-3*P(destructive)",
    },
    "optimisation": {"epochs": 96, "learning_rate": 0.005, "weight_decay": 0.0001, "grad_clip": 1.0, "cpu_threads": 4},
    "crossfit": "five outer group folds independently for each of four regimes; every outer threshold is fit from an inner group-OOF prediction on only that outer training partition",
    "calibration": {"temperatures": [0.5, 3.0, 0.05], "risk_caps": [0.05, 0.10, 0.15, 0.20], "objective": "max(rescues-3*destructive) subject to precision>=0.90 and at least one rescue"},
    "consensus": "at least three of four head winners, all four utilities positive, and every head passes its own nested threshold/risk gate",
    "promotion": {"recall_gain_min": 0.003, "robust_gain_min": 0.002, "bootstrap_lower_positive": True, "intervention_precision_min": 0.90, "all_four_regimes_non_negative": True, "at_least_one_rescue": True, "rescue_vs_neutral_pr_auc_bootstrap_lower_positive": True, "top4_byte_identical": True, "at_most_one_slot5_swap": True},
    "holdout_opened": False,
    "ce_fine_tuned": False,
}


class K16LaneError(ValueError):
    """Raised when a V22-B K16 contract does not hold."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_unsafe(path: Path, label: str) -> None:
    normalized = str(path.resolve()).replace("\\", "/").casefold()
    if any(term in normalized for term in ("holdout", "public", "submission", "locked")):
        raise K16LaneError(f"V22-B is clean-complement-only; refusing {label}: {path}")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise K16LaneError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise K16LaneError(f"{label} must be a JSON object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to overwrite immutable V22-B artifact: {path}")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to overwrite immutable V22-B trace: {path}")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise K16LaneError(f"cannot import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _input_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {"b0_report": args.b0_report, "b0_script": args.b0_script, "p0_script": args.p0_script, "base_metadata": args.base_metadata, "k16_metadata": args.k16_metadata, "k16_scores": args.k16_scores, "fold_manifest": args.fold_manifest}
    for name, path in paths.items():
        _reject_unsafe(path, name)
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")
    return paths


def _pre_registration(args: argparse.Namespace) -> dict[str, Any]:
    paths = _input_paths(args)
    script = Path(__file__).resolve()
    return {"kind": "legalir_v22_b_k16_pre_registration", "split": "clean_complement_only", "holdout_opened": False, "public_labels_used": False, "config": PREREGISTRATION, "script": {"path": str(script), "sha256": _sha256(script)}, "inputs": {name: {"path": str(path.resolve()), "sha256": _sha256(path)} for name, path in paths.items()}}


def _validate_pre_registration(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    observed = _read_json(path, "V22-B pre-registration")
    expected = _pre_registration(args)
    if observed.get("kind") != expected["kind"]:
        raise K16LaneError("wrong V22-B pre-registration kind")
    for field in ("split", "holdout_opened", "public_labels_used", "config", "script", "inputs"):
        if observed.get(field) != expected.get(field):
            raise K16LaneError(f"pre-registration drift in {field}; refuse post-hoc K16 training")
    return observed


def _softmax(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    scaled = [value / temperature for value in logits]
    shift = max(scaled)
    values = [math.exp(value - shift) for value in scaled]
    normalizer = sum(values)
    return [value / normalizer for value in values]


def _utility(probability: Sequence[float]) -> float:
    return probability[2] - 3.0 * probability[1]


def _temperature_values() -> list[float]:
    start, stop, step = PREREGISTRATION["calibration"]["temperatures"]
    return [round(start + index * step, 8) for index in range(int(round((stop - start) / step)) + 1)]


def _winner_rows(predictions: Mapping[str, Sequence[Sequence[float]]], labels: Mapping[str, Sequence[int]]) -> list[dict[str, Any]]:
    rows = []
    for qid, logits_rows in predictions.items():
        winner = max(range(len(logits_rows)), key=lambda index: (_utility(_softmax(logits_rows[index])), -index))
        rows.append({"qid": qid, "winner": winner, "logits": logits_rows[winner], "label": labels[qid][winner]})
    return rows


def _fit_temperature(winners: Sequence[Mapping[str, Any]]) -> float:
    best_temperature = 1.0
    best_loss = math.inf
    for temperature in _temperature_values():
        loss = 0.0
        for row in winners:
            probability = _softmax(row["logits"], temperature)
            loss -= math.log(max(probability[row["label"]], 1e-12))
        loss /= len(winners)
        if loss < best_loss - 1e-12 or (abs(loss - best_loss) <= 1e-12 and temperature < best_temperature):
            best_loss = loss
            best_temperature = temperature
    return best_temperature


def _select_policy(winners: Sequence[Mapping[str, Any]], temperature: float) -> dict[str, float | int]:
    transformed = []
    for row in winners:
        probability = _softmax(row["logits"], temperature)
        transformed.append({"utility": _utility(probability), "risk": probability[1], "label": row["label"]})
    thresholds = sorted({row["utility"] for row in transformed} | {float("inf")})
    best: tuple[float, float, float, int, int] | None = None
    selected_policy: dict[str, float | int] = {"threshold": float("inf"), "risk_cap": 0.0, "rescues": 0, "destructive": 0, "precision": 1.0}
    for risk_cap in PREREGISTRATION["calibration"]["risk_caps"]:
        for threshold in thresholds:
            selected = [row for row in transformed if row["utility"] >= threshold and row["risk"] <= risk_cap]
            rescues = sum(row["label"] == 2 for row in selected)
            destructive = sum(row["label"] == 1 for row in selected)
            precision = rescues / (rescues + destructive) if rescues + destructive else 0.0
            if rescues < 1 or precision < 0.90:
                continue
            key = (rescues - 3.0 * destructive, threshold, -risk_cap, rescues, -destructive)
            if best is None or key > best:
                best = key
                selected_policy = {"threshold": threshold, "risk_cap": risk_cap, "rescues": rescues, "destructive": destructive, "precision": precision}
    return selected_policy


def _semi_hard_mask(scores: Sequence[float], labels: Sequence[int], e3b_ranks: Sequence[int]) -> list[bool]:
    selected = [label in (1, 2) for label in labels]
    positive = [index for index, label in enumerate(labels) if label == 2]
    neutral = [index for index, label in enumerate(labels) if label == 0]
    if positive:
        best_positive = max(scores[index] for index in positive)
        candidates = [index for index in neutral if 0.0 < best_positive - scores[index] <= PREREGISTRATION["semi_hard_negatives"]["score_margin"]]
        if not candidates:
            candidates = [index for index in neutral if scores[index] < best_positive]
        candidates.sort(key=lambda index: (best_positive - scores[index], e3b_ranks[index], index))
    else:
        candidates = sorted((index for index in neutral if e3b_ranks[index] >= 2), key=lambda index: (e3b_ranks[index], index))
    for index in candidates[:PREREGISTRATION["semi_hard_negatives"]["maximum_per_query"]]:
        selected[index] = True
    if not any(selected):
        raise K16LaneError("semi-hard construction produced an empty training query")
    return selected


def _build_dataset(paths: Mapping[str, Path], b0: Any, p0: Any) -> dict[str, Any]:
    b0_report = _read_json(paths["b0_report"], "V22-B0 report")
    gate = b0_report.get("registered_k16_training_gate", {})
    if gate.get("passed") is not True or b0_report.get("model_trained") is not False or b0_report.get("holdout_opened") is not False:
        raise K16LaneError("V22-B requires a passing no-training V22-B0 preflight")
    base_questions = p0._load_questions(paths["base_metadata"], require_references=True)
    k16 = b0._load_expanded_questions(paths["k16_metadata"], "K16 metadata")
    scores = b0._load_e3b(paths["k16_scores"])
    expected = int(b0_report.get("e3b_integrity", {}).get("expected_keys", -1))
    if len(scores) != expected:
        raise K16LaneError("K16 score matrix count differs from its passing V22-B0 preflight")
    folds = p0._load_regime_folds(paths["fold_manifest"], set(base_questions))
    qids = sorted(k16, key=lambda value: (int(value) if value.isdigit() else value))
    features: dict[str, list[list[float]]] = {}
    labels: dict[str, list[int]] = {}
    docs: dict[str, list[str]] = {}
    outcomes: dict[str, list[str]] = {}
    training_mask: dict[str, list[bool]] = {}
    false_negative_checked = 0
    for qid in qids:
        row = k16[qid]
        anchor = list(row["anchor"])
        incumbent = anchor[-1]
        gold = set(base_questions[qid]["references"])
        candidates = []
        incumbent_scores = [scores[(qid, incumbent, config)] for config in CONFIGS]
        for source_rank, doc in enumerate(row["scoring"][1:], 1):
            raw = [scores[(qid, doc, config)] for config in CONFIGS]
            candidates.append({"doc_id": doc, "source_rank": source_rank, "raw": raw, "mean": statistics.fmean(raw), "outcome": p0._outcome(doc, incumbent, gold)})
        ranked = sorted(candidates, key=lambda item: (-item["mean"], item["source_rank"], item["doc_id"]))[:4]
        for e3b_rank, item in enumerate(ranked, 1):
            item["e3b_rank"] = e3b_rank
        outcome_to_label = {"neutral_00": 0, "neutral_11": 0, "destructive": 1, "rescue": 2}
        labels[qid] = [outcome_to_label[item["outcome"]] for item in ranked]
        docs[qid] = [item["doc_id"] for item in ranked]
        outcomes[qid] = [item["outcome"] for item in ranked]
        feature_rows = []
        for item in ranked:
            raw = item["raw"]
            deltas = [score - incumbent_score for score, incumbent_score in zip(raw, incumbent_scores, strict=True)]
            feature_rows.append([*raw, *deltas, statistics.pstdev(raw), item["source_rank"] / 16.0, item["e3b_rank"] / 4.0])
        features[qid] = feature_rows
        training_mask[qid] = _semi_hard_mask([item["mean"] for item in ranked], labels[qid], [item["e3b_rank"] for item in ranked])
        for doc, label, included in zip(docs[qid], labels[qid], training_mask[qid], strict=True):
            if included and label != 2:
                if doc in gold:
                    raise K16LaneError(f"{qid}/{doc}: selected negative is QREL-positive")
                false_negative_checked += 1
    feature_dim = len(next(iter(features.values()))[0])
    return {"qids": qids, "questions": base_questions, "folds": folds, "features": features, "labels": labels, "docs": docs, "outcomes": outcomes, "training_mask": training_mask, "feature_dim": feature_dim, "false_negative_checked": false_negative_checked}


def _tensors(dataset: Mapping[str, Any], qids: Sequence[str], device: str) -> tuple[Any, Any, Any]:
    import torch

    features = torch.tensor([dataset["features"][qid] for qid in qids], dtype=torch.float32, device=device)
    labels = torch.tensor([dataset["labels"][qid] for qid in qids], dtype=torch.long, device=device)
    mask = torch.tensor([dataset["training_mask"][qid] for qid in qids], dtype=torch.bool, device=device)
    return features, labels, mask


def _fit_predict(dataset: Mapping[str, Any], train_qids: Sequence[str], predict_qids: Sequence[str], *, seed: int) -> tuple[dict[str, list[list[float]]], int]:
    import torch

    torch.manual_seed(seed)
    random.seed(seed)
    feature_dim = int(dataset["feature_dim"])

    class MicroHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.Sequential(torch.nn.Linear(feature_dim, PREREGISTRATION["model"]["hidden_dim"]), torch.nn.GELU(), torch.nn.Linear(PREREGISTRATION["model"]["hidden_dim"], 3))

        def forward(self, value: Any) -> Any:
            return self.layers(value)

    model = MicroHead().to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=PREREGISTRATION["optimisation"]["learning_rate"], weight_decay=PREREGISTRATION["optimisation"]["weight_decay"])
    class_weight = torch.tensor(PREREGISTRATION["objective"]["class_weights"], dtype=torch.float32)
    features, labels, mask = _tensors(dataset, train_qids, "cpu")
    for _ in range(PREREGISTRATION["optimisation"]["epochs"]):
        model.train()
        logits = model(features)
        flat_logits = logits[mask]
        flat_labels = labels[mask]
        ce = torch.nn.functional.cross_entropy(flat_logits, flat_labels, weight=class_weight)
        probability = torch.softmax(logits, dim=-1)
        utility = probability[..., 2] - 3.0 * probability[..., 1]
        target = torch.where(labels == 2, torch.ones_like(utility), torch.where(labels == 1, -torch.ones_like(utility), torch.zeros_like(utility)))
        huber = torch.nn.functional.smooth_l1_loss(utility[mask], target[mask])
        terms = []
        for index in range(labels.shape[0]):
            allowed = mask[index]
            positive = utility[index][(labels[index] == 2) & allowed]
            if positive.numel():
                terms.append(torch.logsumexp(utility[index][allowed], dim=0) - torch.logsumexp(positive, dim=0))
        lse = torch.stack(terms).mean() if terms else torch.zeros((), dtype=torch.float32)
        loss = ce + PREREGISTRATION["objective"]["signed_utility_huber_weight"] * huber + PREREGISTRATION["objective"]["multi_positive_lse_weight"] * lse
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), PREREGISTRATION["optimisation"]["grad_clip"])
        optimizer.step()
    model.eval()
    with torch.no_grad():
        values = model(_tensors(dataset, predict_qids, "cpu")[0]).tolist()
    return {qid: [[float(item) for item in row] for row in per_query] for qid, per_query in zip(predict_qids, values, strict=True)}, sum(parameter.numel() for parameter in model.parameters())


def _nested_crossfit(dataset: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
    import torch

    torch.set_num_threads(PREREGISTRATION["optimisation"]["cpu_threads"])
    all_qids = set(dataset["qids"])
    labels = dataset["labels"]
    result: dict[str, Any] = {}
    parameters = 0
    for regime_index, regime in enumerate(REGIMES):
        assignments = dataset["folds"][regime]
        by_fold: dict[int, list[str]] = defaultdict(list)
        for qid in dataset["qids"]:
            by_fold[assignments[qid]].append(qid)
        if set(by_fold) != set(range(OUTER_FOLDS)):
            raise K16LaneError(f"{regime}: not all registered group folds occur in K16 scope")
        calibrated: dict[str, Any] = {}
        audit: dict[str, Any] = {}
        for outer in range(OUTER_FOLDS):
            validation = sorted(by_fold[outer])
            outer_train = sorted(all_qids - set(validation))
            if {assignments[qid] for qid in validation} & {assignments[qid] for qid in outer_train}:
                raise K16LaneError(f"{regime}/{outer}: outer group overlap")
            raw_validation, count = _fit_predict(dataset, outer_train, validation, seed=SEED + regime_index * 10_000 + outer)
            parameters = max(parameters, count)
            inner_raw: dict[str, list[list[float]]] = {}
            inner_audit = {}
            for inner in range(OUTER_FOLDS):
                if inner == outer:
                    continue
                inner_validation = sorted(by_fold[inner])
                inner_train = sorted(set(outer_train) - set(inner_validation))
                if not inner_train or not inner_validation:
                    raise K16LaneError(f"{regime}/{outer}/{inner}: empty nested partition")
                raw_inner, count = _fit_predict(dataset, inner_train, inner_validation, seed=SEED + regime_index * 10_000 + 100 + outer * 10 + inner)
                parameters = max(parameters, count)
                inner_raw.update(raw_inner)
                inner_audit[str(inner)] = {"train_queries": len(inner_train), "validation_queries": len(inner_validation), "group_overlap": False, "training_rescue_rows": sum(label == 2 for qid in inner_train for label in labels[qid])}
            if set(inner_raw) != set(outer_train):
                raise K16LaneError(f"{regime}/{outer}: nested OOF coverage incomplete")
            temperature = _fit_temperature(_winner_rows(inner_raw, labels))
            policy = _select_policy(_winner_rows(inner_raw, labels), temperature)
            for qid, logits_rows in raw_validation.items():
                rows = []
                for logits in logits_rows:
                    probability = _softmax(logits, temperature)
                    rows.append({"logits": logits, "probability": probability, "utility": _utility(probability), "risk": probability[1]})
                calibrated[qid] = {"candidates": rows, "threshold": policy["threshold"], "risk_cap": policy["risk_cap"], "outer_fold": outer}
            audit[str(outer)] = {"outer_train_queries": len(outer_train), "outer_validation_queries": len(validation), "outer_group_overlap": False, "temperature": temperature, "nested_policy": policy, "inner_audit": inner_audit}
        if set(calibrated) != all_qids:
            raise K16LaneError(f"{regime}: outer OOF coverage incomplete")
        result[regime] = {"calibrated": calibrated, "audit": audit}
    return result, parameters


def _combine(dataset: Mapping[str, Any], heads: Mapping[str, Any]) -> tuple[dict[str, list[str]], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    predictions: dict[str, list[str]] = {}
    winners: list[dict[str, Any]] = []
    candidate_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for qid in dataset["qids"]:
        anchor = list(dataset["questions"][qid]["anchor"])
        head_winners = {regime: max(range(4), key=lambda index: (heads[regime]["calibrated"][qid]["candidates"][index]["utility"], -index)) for regime in REGIMES}
        rows = []
        for index, doc in enumerate(dataset["docs"][qid]):
            utilities = {regime: heads[regime]["calibrated"][qid]["candidates"][index]["utility"] for regime in REGIMES}
            risks = {regime: heads[regime]["calibrated"][qid]["candidates"][index]["risk"] for regime in REGIMES}
            thresholds = {regime: heads[regime]["calibrated"][qid]["threshold"] for regime in REGIMES}
            caps = {regime: heads[regime]["calibrated"][qid]["risk_cap"] for regime in REGIMES}
            row = {"qid": qid, "candidate_doc_id": doc, "shortlist_rank": index + 1, "outcome": dataset["outcomes"][qid][index], "utilities": utilities, "risks": risks, "thresholds": thresholds, "risk_caps": caps, "winner_count": sum(head_winners[regime] == index for regime in REGIMES)}
            row["median_utility"] = statistics.median(utilities.values())
            row["all_four_positive"] = all(value > 0.0 for value in utilities.values())
            row["all_head_gates"] = all(utilities[regime] >= thresholds[regime] and risks[regime] <= caps[regime] for regime in REGIMES)
            rows.append(row)
            candidate_rows[qid].append(row)
        selected = max(rows, key=lambda row: (row["winner_count"], row["median_utility"], -row["shortlist_rank"], row["candidate_doc_id"]))
        accepted = selected["winner_count"] >= 3 and selected["all_four_positive"] and selected["all_head_gates"]
        predictions[qid] = [*anchor[:4], selected["candidate_doc_id"]] if accepted else anchor
        winners.append({**selected, "head_winners": {regime: dataset["docs"][qid][index] for regime, index in head_winners.items()}, "accepted": accepted, "top_four_frozen": True})
    return predictions, winners, candidate_rows


def _bootstrap_gain(before: Sequence[float], after: Sequence[float], samples: int = 10_000) -> dict[str, Any]:
    generator = random.Random(SEED)
    deltas = []
    for _ in range(samples):
        indices = [generator.randrange(len(before)) for _ in before]
        deltas.append(statistics.fmean(after[index] - before[index] for index in indices))
    deltas.sort()
    return {"samples": samples, "mean": statistics.fmean(deltas), "lower_95": deltas[math.floor(0.025 * (samples - 1))], "upper_95": deltas[math.ceil(0.975 * (samples - 1))]}


def _regime_metrics(qids: Sequence[str], before: Mapping[str, float], after: Mapping[str, float], assignments: Mapping[str, int]) -> dict[str, Any]:
    folds = []
    for fold in range(OUTER_FOLDS):
        selected = [qid for qid in qids if assignments[qid] == fold]
        base = statistics.fmean(before[qid] for qid in selected)
        candidate = statistics.fmean(after[qid] for qid in selected)
        folds.append({"fold": fold, "queries": len(selected), "baseline_recall": base, "recall": candidate, "gain": candidate - base})
    base = statistics.fmean(before[qid] for qid in qids)
    candidate = statistics.fmean(after[qid] for qid in qids)
    robust_base = base - 0.25 * statistics.pstdev(row["baseline_recall"] for row in folds)
    robust_candidate = candidate - 0.25 * statistics.pstdev(row["recall"] for row in folds)
    gain = candidate - base
    robust_gain = robust_candidate - robust_base
    return {"baseline_recall": base, "recall": candidate, "recall_gain": gain, "baseline_robust_recall": robust_base, "robust_recall": robust_candidate, "robust_gain": robust_gain, "worst_fold_delta": min(row["gain"] for row in folds), "non_negative": gain >= -1e-12 and robust_gain >= -1e-12 and min(row["gain"] for row in folds) >= -1e-12, "folds": folds}


def _average_precision(items: Sequence[tuple[float, int]]) -> float | None:
    ordered = sorted(items, key=lambda item: -item[0])
    positives = sum(label for _, label in ordered)
    if not positives:
        return None
    hits = 0
    total = 0.0
    for index, (_, label) in enumerate(ordered, 1):
        if label:
            hits += 1
            total += hits / index
    return total / positives


def _bootstrap_pr_lift(rows_by_qid: Mapping[str, Sequence[tuple[float, int]]], samples: int = 2_000) -> dict[str, Any]:
    qids = sorted(rows_by_qid)
    generator = random.Random(SEED)
    values = []
    for _ in range(samples):
        rows = [item for _ in qids for item in rows_by_qid[qids[generator.randrange(len(qids))]]]
        prevalence = sum(label for _, label in rows) / len(rows)
        ap = _average_precision(rows)
        if ap is not None:
            values.append(ap - prevalence)
    values.sort()
    return {"samples": len(values), "mean": statistics.fmean(values) if values else None, "lower_95": values[math.floor(0.025 * (len(values) - 1))] if values else None, "upper_95": values[math.ceil(0.975 * (len(values) - 1))] if values else None}


def _evaluate(dataset: Mapping[str, Any], predictions: Mapping[str, Sequence[str]], winners: Sequence[Mapping[str, Any]], candidate_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    before = {qid: len(set(dataset["questions"][qid]["anchor"]) & set(dataset["questions"][qid]["references"])) / len(dataset["questions"][qid]["references"]) for qid in dataset["qids"]}
    after = {qid: len(set(predictions[qid]) & set(dataset["questions"][qid]["references"])) / len(dataset["questions"][qid]["references"]) for qid in dataset["qids"]}
    metrics = _regime_metrics(dataset["qids"], before, after, dataset["folds"]["standard_grouped"])
    regimes = {regime: _regime_metrics(dataset["qids"], before, after, dataset["folds"][regime]) for regime in REGIMES}
    changed = [row for row in winners if row["accepted"]]
    rescues = sum(row["outcome"] == "rescue" for row in changed)
    destructive = sum(row["outcome"] == "destructive" for row in changed)
    neutral = sum(row["outcome"].startswith("neutral") for row in changed)
    precision = rescues / (rescues + destructive) if rescues + destructive else 0.0
    pr_rows: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for qid, rows in candidate_rows.items():
        for row in rows:
            if row["outcome"] in {"rescue", "neutral_00"}:
                pr_rows[qid].append((float(row["median_utility"]), int(row["outcome"] == "rescue")))
    pr_items = [item for rows in pr_rows.values() for item in rows]
    prevalence = sum(label for _, label in pr_items) / len(pr_items)
    pr_auc = _average_precision(pr_items)
    pr_bootstrap = _bootstrap_pr_lift(pr_rows)
    bootstrap = _bootstrap_gain([before[qid] for qid in dataset["qids"]], [after[qid] for qid in dataset["qids"]])
    top_four_frozen = all(list(predictions[qid])[:4] == list(dataset["questions"][qid]["anchor"][:4]) for qid in dataset["qids"])
    max_one_swap = all(list(predictions[qid]) == list(dataset["questions"][qid]["anchor"]) or list(predictions[qid])[:4] == list(dataset["questions"][qid]["anchor"][:4]) for qid in dataset["qids"])
    checks = {"recall_gain_at_least_0_003": metrics["recall_gain"] >= PREREGISTRATION["promotion"]["recall_gain_min"], "robust_gain_at_least_0_002": metrics["robust_gain"] >= PREREGISTRATION["promotion"]["robust_gain_min"], "paired_bootstrap_lower_positive": bootstrap["lower_95"] > 0.0, "intervention_precision_at_least_0_90": precision >= PREREGISTRATION["promotion"]["intervention_precision_min"], "all_four_regimes_non_negative": all(value["non_negative"] for value in regimes.values()), "at_least_one_rescue": rescues >= 1, "rescue_vs_neutral_pr_auc_lift_bootstrap_lower_positive": pr_bootstrap["lower_95"] is not None and pr_bootstrap["lower_95"] > 0.0, "top_four_byte_identical": top_four_frozen, "at_most_one_slot5_swap": max_one_swap, "data_safety": True}
    return {"metrics_vs_v16b_anchor": metrics, "regimes": regimes, "paired_bootstrap_vs_v16b": bootstrap, "interventions": {"changed_queries": len(changed), "rescues": rescues, "destructive": destructive, "neutral": neutral, "intervention_precision": precision}, "rescue_vs_neutral": {"candidate_rows": len(pr_items), "prevalence": prevalence, "pr_auc": pr_auc, "pr_auc_lift_vs_prevalence": pr_auc - prevalence if pr_auc is not None else None, "query_bootstrap_pr_auc_lift": pr_bootstrap}, "promotion": {"checks": checks, "passed": all(checks.values())}}


def run(args: argparse.Namespace) -> dict[str, Any]:
    pre = _validate_pre_registration(args.pre_registration, args)
    paths = _input_paths(args)
    b0 = _load_module(paths["b0_script"], "v22_b0_shared")
    p0 = _load_module(paths["p0_script"], "v22_p0_shared")
    dataset = _build_dataset(paths, b0, p0)
    heads, parameters = _nested_crossfit(dataset)
    predictions, winners, candidate_rows = _combine(dataset, heads)
    evaluation = _evaluate(dataset, predictions, winners, candidate_rows)
    report = {"kind": "legalir_v22_b_k16_complementary_rescue_lane", "schema_version": 1, "split": "clean_complement_only", "holdout_opened": False, "public_labels_used": False, "ce_loaded": False, "ce_fine_tuned": False, "v16b_top4_changed": False, "pre_registration": {"path": str(args.pre_registration.resolve()), "sha256": _sha256(args.pre_registration), "config": pre["config"]}, "inputs": pre["inputs"], "architecture": {"name": "K16_E3B_vector_setwise_micro_head", "parameter_count": parameters, "under_2k": parameters < PREREGISTRATION["model"]["parameters_expected_under"], "base_pipeline_parameters": 1_090_010_000, "end_to_end_upper_bound": 1_090_010_000 + parameters, "under_4b": True}, "data_contract": {"queries": len(dataset["qids"]), "shortlist_per_query": 4, "false_negative_checked_selected_negative_rows": dataset["false_negative_checked"], "fallback": "retain original V16b top-five identity; no automatic candidate substitution"}, "crossfit": {regime: details["audit"] for regime, details in heads.items()}, **evaluation, "winner_traces": winners, "decision": {"holdout_may_open": False, "if_local_contract_passes": "K16 is a subset-only result: authorize the separately registered K32 preflight, not holdout or submission", "if_failed": "close V22-B K16 without threshold/model retuning; do not open K32, K64, or holdout"}}
    report_path = args.output_dir / "v22_b_k16_report.json"
    trace_path = args.output_dir / "v22_b_k16_candidate_traces.jsonl"
    if report_path.exists() or trace_path.exists():
        raise FileExistsError("V22-B output exists; refusing to replace evidence")
    _atomic_json(report_path, report)
    _atomic_jsonl(trace_path, [row for rows in candidate_rows.values() for row in rows])
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0-report", type=Path, required=True)
    parser.add_argument("--b0-script", type=Path, required=True)
    parser.add_argument("--p0-script", type=Path, required=True)
    parser.add_argument("--base-metadata", type=Path, required=True)
    parser.add_argument("--k16-metadata", type=Path, required=True)
    parser.add_argument("--k16-scores", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--pre-registration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pre-register-only", action="store_true")
    args = parser.parse_args()
    if not args.pre_register_only and args.output_dir is None:
        parser.error("--output-dir is required for V22-B execution")
    return args


def main() -> int:
    args = parse_args()
    _reject_unsafe(args.pre_registration, "pre-registration")
    if args.pre_register_only:
        if args.pre_registration.exists():
            raise FileExistsError(f"pre-registration already exists: {args.pre_registration}")
        _atomic_json(args.pre_registration, _pre_registration(args))
        print(json.dumps({"pre_registered": str(args.pre_registration.resolve()), "config": PREREGISTRATION}, ensure_ascii=False, sort_keys=True))
        return 0
    assert args.output_dir is not None
    _reject_unsafe(args.output_dir, "output directory")
    report = run(args)
    print(json.dumps({"output": str((args.output_dir / "v22_b_k16_report.json").resolve()), "local_contract_passed": report["promotion"]["passed"], "holdout_opened": False}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
