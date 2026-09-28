"""V22-A: frozen-representation candidate--incumbent verifier.

This is the only c35 representation arm opened by a passing V22-P0 audit.  It
keeps V16b's top four immutable, uses the existing query-conditioned frozen
block cache, and learns a <10M CAP-DBA/TIR-style head for the six rank-6..12
challengers versus the incumbent slot 5.  The existing CE is never loaded or
fine-tuned.

The runner has two phases.  ``--pre-register-only`` records fixed architecture,
loss, calibration, and promotion conditions.  The regular run verifies that
pre-registration byte-for-byte before it loads QRELs or starts training.
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


K = 5
SEED = 22117
REGIMES = ("standard_grouped", "law_family_ood", "query_template_ood", "near_duplicate_ood")
CONFIGS = ("B1024-D30", "B1024-D60", "B1536-D30", "B1536-D60", "B2304-D30", "B2304-D60")

# Fixed before labels/training.  It is deliberately one small arm, with no
# architecture or threshold sweep following its OOF outcome.
PREREGISTRATION = {
    "schema_version": 1,
    "representation": "V21-B frozen B2304-D60 query-conditioned CLS block cache",
    "context": "V16b incumbent slot-5 bag; every block is already query-conditioned",
    "challengers": "immutable c35 ranks 6..12; invalid/missing bags force abstention",
    "e3b_artifact_contract": "each packable qid/document/config key occurs exactly once in both the E3B manifest and score matrix; only metadata-declared source-unavailable documents may be absent and their queries force-abstain",
    "head": {"projection_dim": 128, "attention_heads": 4, "tir_residual_bound": 0.10, "dropout": 0.0},
    "objective": {
        "classes": ["neutral", "destructive", "rescue"],
        "class_weights": [1.0, 3.0, 8.0],
        "cross_entropy_weight": 1.0,
        "signed_utility_huber_weight": 0.50,
        "multi_positive_lse_weight": 0.50,
        "utility": "P(rescue)-3*P(destructive)",
    },
    "optimisation": {"epochs": 16, "batch_queries": 96, "learning_rate": 0.001, "weight_decay": 0.0001, "grad_clip": 1.0, "cpu_threads": 4},
    "outer_crossfit": "five group-isolated folds independently for each of standard, law-family, query-template, near-duplicate",
    "calibration": {
        "kind": "winner_after_argmax_temperature_then_nested_threshold",
        "temperature_grid": [0.5, 3.0, 0.05],
        "risk_caps": [0.05, 0.10, 0.15, 0.20],
        "inner_predictions": "other four outer OOF folds only; current validation fold excluded",
        "threshold_objective": "max(rescues-3*destructive) subject to precision>=0.90 and at least one rescue",
    },
    "consensus": "at least 3/4 argmax winners, all four calibrated utilities positive, each head passes its nested utility/risk gate",
    "promotion": {
        "recall_gain_min": 0.003,
        "robust_gain_min": 0.002,
        "bootstrap_lower_positive": True,
        "intervention_precision_min": 0.90,
        "all_four_regimes_non_negative": True,
        "at_least_one_rescue": True,
        "rescue_coverage_nonzero": True,
        "rescue_vs_neutral_pr_auc_bootstrap_lower_positive": True,
        "top4_byte_identical": True,
        "at_most_one_slot5_swap": True,
    },
    "holdout_opened": False,
    "ce_fine_tuned": False,
    "training_allowed": True,
}


class V22AError(ValueError):
    """Raised when a V22-A condition or immutable input contract is violated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V22AError(f"cannot read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise V22AError(f"{label} must be a JSON object")
    return payload


def _reject_unsafe(path: Path, label: str) -> None:
    normal = str(path.resolve()).replace("\\", "/").casefold()
    if any(term in normal for term in ("holdout", "public", "submission", "locked")):
        raise V22AError(f"V22-A must remain clean-complement only; refusing {label}: {path}")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to overwrite V22-A evidence: {path}")
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
            raise FileExistsError(f"refusing to overwrite V22-A evidence: {path}")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_p0_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("v22_p0_shared", path)
    if spec is None or spec.loader is None:
        raise V22AError(f"cannot import V22-P0 helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _input_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
        "p0_report": args.p0_report,
        "p0_script": args.p0_script,
        "metadata": args.metadata,
        "fold_manifest": args.fold_manifest,
        "bag_cache": args.bag_cache,
        "bag_cache_provenance": args.bag_cache.with_suffix(args.bag_cache.suffix + ".provenance.json"),
        "current_scores": args.current_scores,
        "e3b_scores": args.e3b_scores,
    }
    for name, path in paths.items():
        _reject_unsafe(path, name)
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")
    return paths


def _pre_registration(args: argparse.Namespace) -> dict[str, Any]:
    paths = _input_paths(args)
    script = Path(__file__).resolve()
    return {
        "kind": "legalir_v22_a_cap_dba_tir_pre_registration",
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "config": PREREGISTRATION,
        "script": {"path": str(script), "sha256": _sha256(script)},
        "inputs": {name: {"path": str(path.resolve()), "sha256": _sha256(path)} for name, path in paths.items()},
    }


def _validate_pre_registration(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    observed = _read_json(path, "V22-A pre-registration")
    expected = _pre_registration(args)
    if observed.get("kind") != expected["kind"]:
        raise V22AError("wrong V22-A pre-registration kind")
    for field in ("split", "holdout_opened", "public_labels_used", "config", "script", "inputs"):
        if observed.get(field) != expected.get(field):
            raise V22AError(f"pre-registration drift in {field}; refuse post-hoc V22-A run")
    return observed


def _assert_p0_pass(report: Mapping[str, Any]) -> None:
    if report.get("kind") != "legalir_v22_p0_rescue_neutral_closure_audit":
        raise V22AError("V22-A requires the V22-P0 closure-audit report")
    if report.get("holdout_opened") is not False or report.get("public_labels_used") is not False:
        raise V22AError("V22-P0 report must attest no holdout/public-label access")
    gate = report.get("registered_c35_rankability_gate")
    if not isinstance(gate, Mapping) or gate.get("passed") is not True:
        raise V22AError("V22-A is closed because the registered V22-P0 rankability gate did not pass")


def _mask_mean(tensor: Any, mask: Any) -> Any:
    import torch

    weights = mask.unsqueeze(-1).to(dtype=tensor.dtype)
    return (tensor * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _temperature_grid() -> list[float]:
    low, high, step = PREREGISTRATION["calibration"]["temperature_grid"]
    values: list[float] = []
    value = float(low)
    while value <= float(high) + 1e-12:
        values.append(round(value, 8))
        value += float(step)
    return values


def _softmax(logits: Sequence[float], temperature: float) -> list[float]:
    scaled = [float(value) / temperature for value in logits]
    ceiling = max(scaled)
    values = [math.exp(value - ceiling) for value in scaled]
    total = sum(values)
    return [value / total for value in values]


def _utility(probability: Sequence[float]) -> float:
    return float(probability[2] - 3.0 * probability[1])


def _fit_temperature(winners: Sequence[Mapping[str, Any]]) -> float:
    if not winners:
        return 1.0
    best = (float("inf"), 1.0)
    for temperature in _temperature_grid():
        loss = 0.0
        for row in winners:
            probability = _softmax(row["logits"], temperature)
            loss -= math.log(max(probability[int(row["label"])], 1e-12))
        key = (loss / len(winners), temperature)
        if key < best:
            best = key
    return best[1]


def _winner_rows(predictions: Mapping[str, Sequence[Sequence[float]]], labels: Mapping[str, Sequence[int]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for qid, per_candidate in predictions.items():
        winner = max(
            range(len(per_candidate)),
            key=lambda index: (_utility(_softmax(per_candidate[index], 1.0)), -index),
        )
        rows.append({"qid": qid, "candidate_index": winner, "logits": list(per_candidate[winner]), "label": int(labels[qid][winner])})
    return rows


def _select_calibration_policy(winners: Sequence[Mapping[str, Any]], temperature: float) -> dict[str, Any]:
    calibrated = [
        {**row, "probability": _softmax(row["logits"], temperature)}
        for row in winners
    ]
    for row in calibrated:
        row["utility"] = _utility(row["probability"])
        row["risk"] = float(row["probability"][1])
    thresholds = [0.0, *sorted({float(row["utility"]) for row in calibrated if float(row["utility"]) > 0.0}), float("inf")]
    best_key: tuple[float, float, int, float, float] = (0.0, 0.0, 0, float("-inf"), float("-inf"))
    best = {"threshold": float("inf"), "risk_cap": min(PREREGISTRATION["calibration"]["risk_caps"]), "accepted": 0, "rescues": 0, "destructive": 0, "neutral": 0, "precision": 0.0, "net_utility": 0.0}
    for risk_cap in PREREGISTRATION["calibration"]["risk_caps"]:
        for threshold in thresholds:
            accepted = [row for row in calibrated if row["utility"] >= threshold and row["risk"] <= risk_cap]
            rescues = sum(row["label"] == 2 for row in accepted)
            destructive = sum(row["label"] == 1 for row in accepted)
            neutral = len(accepted) - rescues - destructive
            precision = rescues / (rescues + destructive) if rescues + destructive else 0.0
            net = float(rescues - 3 * destructive)
            valid = rescues >= 1 and precision >= PREREGISTRATION["promotion"]["intervention_precision_min"] and net > 0.0
            if not valid:
                continue
            key = (net, precision, rescues, float(threshold), -float(risk_cap))
            if key > best_key:
                best_key = key
                best = {"threshold": float(threshold), "risk_cap": float(risk_cap), "accepted": len(accepted), "rescues": rescues, "destructive": destructive, "neutral": neutral, "precision": precision, "net_utility": net}
    return best


def _average_precision(items: Sequence[tuple[float, int]]) -> float | None:
    positives = sum(label for _, label in items)
    if positives == 0:
        return None
    ordered = sorted(items, key=lambda item: -item[0])
    found = 0
    total = 0.0
    for rank, (_, label) in enumerate(ordered, 1):
        if label:
            found += 1
            total += found / rank
    return total / positives


def _bootstrap_pr_lift(rows_by_qid: Mapping[str, Sequence[Mapping[str, Any]]], *, samples: int = 2_000) -> dict[str, Any]:
    qids = sorted(rows_by_qid)
    rng = random.Random(SEED + 91)
    values: list[float] = []
    for _ in range(samples):
        items: list[tuple[float, int]] = []
        for _ in qids:
            for row in rows_by_qid[qids[rng.randrange(len(qids))]]:
                if row["outcome"] in {"rescue", "neutral_00"}:
                    items.append((float(row["utility"]), int(row["outcome"] == "rescue")))
        ap = _average_precision(items)
        if ap is not None and items:
            values.append(ap - sum(label for _, label in items) / len(items))
    if not values:
        return {"samples": samples, "lower_95": None, "upper_95": None, "mean": None}
    values.sort()
    return {
        "samples": samples,
        "lower_95": values[math.floor(0.025 * (len(values) - 1))],
        "upper_95": values[math.ceil(0.975 * (len(values) - 1))],
        "mean": statistics.fmean(values),
        "unit": "query",
    }


def _recall(answer: Sequence[str], references: Sequence[str]) -> float:
    gold = set(str(value) for value in references)
    return len(set(str(value) for value in answer[:K]) & gold) / len(gold)


def _bootstrap_gain(anchor: Sequence[float], candidate: Sequence[float], *, samples: int = 10_000) -> dict[str, Any]:
    import numpy as np

    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(anchor, dtype=np.float64)
    rng = np.random.default_rng(SEED + 17)
    values = []
    for start in range(0, samples, 250):
        count = min(250, samples - start)
        indices = rng.integers(0, len(delta), size=(count, len(delta)))
        values.extend(delta[indices].mean(axis=1).tolist())
    values.sort()
    return {"samples": samples, "mean": float(statistics.fmean(values)), "lower_95": float(values[math.floor(0.025 * (len(values) - 1))]), "upper_95": float(values[math.ceil(0.975 * (len(values) - 1))])}


def _regime_metrics(
    qids: Sequence[str],
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    folds: Mapping[str, int],
) -> dict[str, Any]:
    fold_rows = []
    for fold in sorted(set(folds[qid] for qid in qids)):
        selected = [qid for qid in qids if folds[qid] == fold]
        before = statistics.fmean(baseline[qid] for qid in selected)
        after = statistics.fmean(candidate[qid] for qid in selected)
        fold_rows.append({"fold": fold, "queries": len(selected), "baseline_recall": before, "recall": after, "gain": after - before})
    before = statistics.fmean(baseline[qid] for qid in qids)
    after = statistics.fmean(candidate[qid] for qid in qids)
    before_std = statistics.pstdev(row["baseline_recall"] for row in fold_rows)
    after_std = statistics.pstdev(row["recall"] for row in fold_rows)
    robust_gain = (after - 0.25 * after_std) - (before - 0.25 * before_std)
    return {"baseline_recall": before, "recall": after, "recall_gain": after - before, "baseline_robust_recall": before - 0.25 * before_std, "robust_recall": after - 0.25 * after_std, "robust_gain": robust_gain, "worst_fold_delta": min(row["gain"] for row in fold_rows), "non_negative": after - before >= -1e-12 and robust_gain >= -1e-12 and min(row["gain"] for row in fold_rows) >= -1e-12, "folds": fold_rows}


def _build_dataset(args: argparse.Namespace, p0: Any, paths: Mapping[str, Path]) -> dict[str, Any]:
    import numpy as np

    questions = p0._load_questions(paths["metadata"], require_references=True)
    qids = sorted(questions, key=lambda item: (int(item) if item.isdigit() else item))
    raw_metadata = _read_json(paths["metadata"], "E3B metadata")
    raw_questions = raw_metadata.get("questions")
    if not isinstance(raw_questions, list):
        raise V22AError("E3B metadata.questions must be a list")
    e3b_metadata = {str(row.get("qid", "")): row for row in raw_questions if isinstance(row, Mapping)}
    if set(e3b_metadata) != set(qids):
        raise V22AError("E3B metadata question IDs disagree with the normalized P0 question contract")
    regime_folds = p0._load_regime_folds(paths["fold_manifest"], set(qids))
    cache = np.load(paths["bag_cache"], allow_pickle=False)
    if set(cache.files) != {"embeddings", "mask", "keys"}:
        raise V22AError("frozen bag cache schema drift")
    embeddings = cache["embeddings"]
    mask = cache["mask"]
    keys = [str(value) for value in cache["keys"].tolist()]
    if embeddings.shape != (len(keys), 5, 1024) or mask.shape != (len(keys), 5):
        raise V22AError("frozen cache shape is not the V21-B 3..5 block representation contract")
    index = {key: position for position, key in enumerate(keys)}
    if len(index) != len(keys):
        raise V22AError("frozen cache repeats a qid/document bag")
    current = p0._load_nested_scores(paths["current_scores"], "frozen current CE", set(qids))
    e3b = p0._load_e3b_scores(paths["e3b_scores"], set(qids))
    active: list[str] = []
    forced: dict[str, str] = {}
    per_qid: dict[str, dict[str, Any]] = {}
    for qid in qids:
        question = questions[qid]
        incumbent = question["anchor"][-1]
        candidates = list(question["boundary"][1:])
        all_docs = [incumbent, *candidates]
        raw_question = e3b_metadata[qid]
        unavailable = [str(doc) for doc in raw_question.get("e3b_source_unavailable", [])]
        unavailable_set = set(unavailable)
        expected_e3b_docs = [doc for doc in all_docs if doc not in unavailable_set]
        if len(candidates) != 6:
            raise V22AError(f"{qid}: c35 must contain exactly six rank-6..12 challengers")
        if [str(doc) for doc in raw_question.get("e3b_scoring_boundary", [])] != expected_e3b_docs:
            raise V22AError(f"{qid}: E3B scoring boundary disagrees with source-unavailable metadata")
        if any((qid, doc, config) in e3b for doc in unavailable for config in CONFIGS):
            raise V22AError(f"{qid}: a source-unavailable document unexpectedly has E3B scores")
        valid_packable = all(f"{qid}\t{doc}" in index for doc in expected_e3b_docs)
        valid_packable = valid_packable and all((qid, doc) in current for doc in expected_e3b_docs)
        valid_packable = valid_packable and all((qid, doc, config) in e3b for doc in expected_e3b_docs for config in CONFIGS)
        if not valid_packable:
            raise V22AError(f"{qid}: E3B/cache is incomplete outside a metadata-declared source-unavailable document")
        if unavailable:
            forced[qid] = "metadata-declared E3B source-unavailable document; original V16b identity retained"
            continue
        gold = set(question["references"])
        label_for_doc = {"neutral_00": 0, "neutral_11": 0, "destructive": 1, "rescue": 2}
        outcomes = [p0._outcome(doc, incumbent, gold) for doc in candidates]
        e3b_tail = statistics.fmean(e3b[(qid, incumbent, config)] for config in CONFIGS)
        ce_tail = current[(qid, incumbent)]
        per_qid[qid] = {
            "candidate_docs": candidates,
            "candidate_indices": [index[f"{qid}\t{doc}"] for doc in candidates],
            "incumbent_index": index[f"{qid}\t{incumbent}"],
            "labels": [label_for_doc[outcome] for outcome in outcomes],
            "outcomes": outcomes,
            "static": [[statistics.fmean(e3b[(qid, doc, config)] for config in CONFIGS) - e3b_tail, current[(qid, doc)] - ce_tail, (rank + 1) / 6.0] for rank, doc in enumerate(candidates)],
        }
        active.append(qid)
    if not active:
        raise V22AError("no query has complete frozen cache and score evidence")
    return {"questions": questions, "all_qids": qids, "active_qids": active, "force_abstain_qids": sorted(forced), "force_abstain_reasons": forced, "regime_folds": regime_folds, "embeddings": embeddings, "mask": mask, "per_qid": per_qid}


def _make_model() -> Any:
    import torch

    class CapDbaTir(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dim = PREREGISTRATION["head"]["projection_dim"]
            self.project = torch.nn.Sequential(torch.nn.Linear(1024, dim, bias=False), torch.nn.LayerNorm(dim), torch.nn.GELU())
            self.attention = torch.nn.MultiheadAttention(dim, PREREGISTRATION["head"]["attention_heads"], batch_first=True, dropout=0.0)
            self.tir = torch.nn.Sequential(torch.nn.Linear(dim * 3 + 4, dim), torch.nn.GELU(), torch.nn.Linear(dim, 1))
            self.output = torch.nn.Sequential(torch.nn.Linear(dim * 3 + 4, dim), torch.nn.GELU(), torch.nn.Linear(dim, 3))
            self.residual_bound = float(PREREGISTRATION["head"]["tir_residual_bound"])

        def forward(self, incumbent: Any, incumbent_mask: Any, challenger: Any, challenger_mask: Any, static: Any) -> Any:
            batch, candidates, blocks, _ = challenger.shape
            context = _mask_mean(self.project(incumbent), incumbent_mask)
            candidate_hidden = self.project(challenger.reshape(batch * candidates, blocks, -1))
            candidate_mask = challenger_mask.reshape(batch * candidates, blocks).bool()
            query = context.unsqueeze(1).repeat_interleave(candidates, dim=0)
            pooled, _ = self.attention(query, candidate_hidden, candidate_hidden, key_padding_mask=~candidate_mask, need_weights=False)
            pooled = pooled.squeeze(1)
            context_flat = query.squeeze(1)
            cosine = torch.nn.functional.cosine_similarity(context_flat, pooled, dim=-1).unsqueeze(-1)
            static_flat = static.reshape(batch * candidates, -1)
            features = torch.cat([context_flat, pooled, pooled - context_flat, cosine, static_flat], dim=-1)
            logits = self.output(features)
            correction = self.residual_bound * torch.tanh(self.tir(features))
            # Bounded TIR residual adjusts rescue/destructive evidence in
            # opposite directions without changing the frozen encoder.
            logits = logits + torch.cat([-0.5 * correction, -correction, correction], dim=-1)
            return logits.reshape(batch, candidates, 3)

    return CapDbaTir()


def _tensors(dataset: Mapping[str, Any], qids: Sequence[str], device: str) -> tuple[Any, Any, Any, Any, Any, Any]:
    import numpy as np
    import torch

    info = dataset["per_qid"]
    incumbent_indices = np.asarray([info[qid]["incumbent_index"] for qid in qids], dtype=np.int64)
    candidate_indices = np.asarray([info[qid]["candidate_indices"] for qid in qids], dtype=np.int64)
    incumbent = torch.from_numpy(np.asarray(dataset["embeddings"][incumbent_indices], dtype=np.float32)).to(device)
    incumbent_mask = torch.from_numpy(np.asarray(dataset["mask"][incumbent_indices], dtype=np.uint8)).to(device)
    challenger = torch.from_numpy(np.asarray(dataset["embeddings"][candidate_indices], dtype=np.float32)).to(device)
    challenger_mask = torch.from_numpy(np.asarray(dataset["mask"][candidate_indices], dtype=np.uint8)).to(device)
    static = torch.tensor([info[qid]["static"] for qid in qids], dtype=torch.float32, device=device)
    labels = torch.tensor([info[qid]["labels"] for qid in qids], dtype=torch.long, device=device)
    return incumbent, incumbent_mask, challenger, challenger_mask, static, labels


def _fit_and_predict(dataset: Mapping[str, Any], train_qids: Sequence[str], validation_qids: Sequence[str], *, device: str, seed: int) -> tuple[dict[str, list[list[float]]], int]:
    import torch

    torch.manual_seed(seed)
    random.seed(seed)
    model = _make_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PREREGISTRATION["optimisation"]["learning_rate"], weight_decay=PREREGISTRATION["optimisation"]["weight_decay"])
    class_weight = torch.tensor(PREREGISTRATION["objective"]["class_weights"], dtype=torch.float32, device=device)
    train = _tensors(dataset, train_qids, device)
    batch_size = PREREGISTRATION["optimisation"]["batch_queries"]
    for _ in range(PREREGISTRATION["optimisation"]["epochs"]):
        model.train()
        order = list(range(len(train_qids)))
        random.shuffle(order)
        for start in range(0, len(order), batch_size):
            chosen = order[start:start + batch_size]
            incumbent, incumbent_mask, challenger, challenger_mask, static, labels = (value[chosen] for value in train)
            logits = model(incumbent, incumbent_mask, challenger, challenger_mask, static)
            ce = torch.nn.functional.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1), weight=class_weight)
            probability = torch.softmax(logits, dim=-1)
            utility = probability[..., 2] - 3.0 * probability[..., 1]
            target = torch.where(labels == 2, torch.ones_like(utility), torch.where(labels == 1, -torch.ones_like(utility), torch.zeros_like(utility)))
            huber = torch.nn.functional.smooth_l1_loss(utility, target)
            pair_losses = []
            for index in range(labels.shape[0]):
                positive = utility[index][labels[index] == 2]
                if positive.numel():
                    pair_losses.append(torch.logsumexp(utility[index], dim=0) - torch.logsumexp(positive, dim=0))
            lse = torch.stack(pair_losses).mean() if pair_losses else torch.zeros((), device=device)
            loss = ce + PREREGISTRATION["objective"]["signed_utility_huber_weight"] * huber + PREREGISTRATION["objective"]["multi_positive_lse_weight"] * lse
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), PREREGISTRATION["optimisation"]["grad_clip"])
            optimizer.step()
    model.eval()
    validation = _tensors(dataset, validation_qids, device)
    with torch.no_grad():
        logits = model(*validation[:5]).cpu().tolist()
    return {qid: [[float(value) for value in row] for row in per_query] for qid, per_query in zip(validation_qids, logits, strict=True)}, sum(parameter.numel() for parameter in model.parameters())


def _crossfit(dataset: Mapping[str, Any], *, device: str) -> tuple[dict[str, dict[str, Any]], int]:
    active = set(dataset["active_qids"])
    outputs: dict[str, dict[str, Any]] = {}
    parameter_count = 0
    for regime_index, regime in enumerate(REGIMES):
        folds = dataset["regime_folds"][regime]
        by_fold: dict[int, list[str]] = defaultdict(list)
        for qid in dataset["active_qids"]:
            by_fold[folds[qid]].append(qid)
        if len(by_fold) != 5 or sum(len(value) for value in by_fold.values()) != len(active):
            raise V22AError(f"{regime}: active qids do not span the five registered outer folds")
        raw: dict[str, list[list[float]]] = {}
        train_audit: dict[str, Any] = {}
        for fold in sorted(by_fold):
            validation = sorted(by_fold[fold])
            training = sorted(active - set(validation))
            overlap = {folds[qid] for qid in training} & {folds[qid] for qid in validation}
            if overlap:
                raise V22AError(f"{regime}/{fold}: group-isolated outer fold overlap")
            predicted, parameters = _fit_and_predict(dataset, training, validation, device=device, seed=SEED + regime_index * 100 + fold)
            parameter_count = max(parameter_count, parameters)
            raw.update(predicted)
            train_audit[str(fold)] = {"train_queries": len(training), "validation_queries": len(validation), "group_overlap": False}
        if set(raw) != active:
            raise V22AError(f"{regime}: outer OOF coverage incomplete")
        labels = {qid: dataset["per_qid"][qid]["labels"] for qid in active}
        calibrated: dict[str, Any] = {}
        threshold_audit: dict[str, Any] = {}
        for fold in sorted(by_fold):
            validation = set(by_fold[fold])
            calibration_predictions = {qid: raw[qid] for qid in active - validation}
            winners = _winner_rows(calibration_predictions, labels)
            temperature = _fit_temperature(winners)
            policy = _select_calibration_policy(winners, temperature)
            threshold_audit[str(fold)] = {"calibration_queries": len(calibration_predictions), "validation_queries_excluded": len(validation), "temperature": temperature, **policy}
            for qid in validation:
                rows = []
                for logits in raw[qid]:
                    probability = _softmax(logits, temperature)
                    rows.append({"logits": logits, "probability": probability, "utility": _utility(probability), "risk": probability[1]})
                calibrated[qid] = {"candidates": rows, "threshold": policy["threshold"], "risk_cap": policy["risk_cap"], "fold": fold}
        outputs[regime] = {"raw": raw, "calibrated": calibrated, "outer_fold_audit": train_audit, "nested_winner_calibration": threshold_audit}
    return outputs, parameter_count


def _combine(dataset: Mapping[str, Any], heads: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, list[str]], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    predictions: dict[str, list[str]] = {}
    winner_traces: list[dict[str, Any]] = []
    candidate_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for qid in dataset["all_qids"]:
        question = dataset["questions"][qid]
        anchor = list(question["anchor"])
        if qid in dataset["force_abstain_qids"]:
            predictions[qid] = anchor
            winner_traces.append({"qid": qid, "accepted": False, "force_abstain": True, "reason": dataset["force_abstain_reasons"][qid]})
            continue
        candidates = dataset["per_qid"][qid]["candidate_docs"]
        head_winners: dict[str, int] = {}
        for regime in REGIMES:
            per_candidate = heads[regime]["calibrated"][qid]["candidates"]
            head_winners[regime] = max(range(len(per_candidate)), key=lambda index: (per_candidate[index]["utility"], -index))
        rows = []
        for index, doc in enumerate(candidates):
            utilities = {regime: heads[regime]["calibrated"][qid]["candidates"][index]["utility"] for regime in REGIMES}
            risks = {regime: heads[regime]["calibrated"][qid]["candidates"][index]["risk"] for regime in REGIMES}
            thresholds = {regime: heads[regime]["calibrated"][qid]["threshold"] for regime in REGIMES}
            caps = {regime: heads[regime]["calibrated"][qid]["risk_cap"] for regime in REGIMES}
            row = {"qid": qid, "candidate_doc_id": doc, "candidate_rank": index + 6, "outcome": dataset["per_qid"][qid]["outcomes"][index], "utilities": utilities, "risks": risks, "thresholds": thresholds, "risk_caps": caps, "winner_count": sum(head_winners[regime] == index for regime in REGIMES)}
            row["median_utility"] = statistics.median(utilities.values())
            row["all_four_positive"] = all(value > 0.0 for value in utilities.values())
            row["all_head_gates"] = all(utilities[regime] >= thresholds[regime] and risks[regime] <= caps[regime] for regime in REGIMES)
            rows.append(row)
            candidate_rows[qid].append(row)
        selected = max(rows, key=lambda row: (row["winner_count"], row["median_utility"], -row["candidate_rank"], row["candidate_doc_id"]))
        accepted = selected["winner_count"] >= 3 and selected["all_four_positive"] and selected["all_head_gates"]
        predictions[qid] = [*anchor[:4], selected["candidate_doc_id"]] if accepted else anchor
        winner_traces.append({**selected, "head_winners": {regime: candidates[index] for regime, index in head_winners.items()}, "accepted": accepted, "force_abstain": False, "top_four_frozen": True})
    return predictions, winner_traces, candidate_rows


def _evaluate(dataset: Mapping[str, Any], predictions: Mapping[str, Sequence[str]], winners: Sequence[Mapping[str, Any]], candidate_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    baseline = {qid: _recall(dataset["questions"][qid]["anchor"], dataset["questions"][qid]["references"]) for qid in dataset["all_qids"]}
    final = {qid: _recall(predictions[qid], dataset["questions"][qid]["references"]) for qid in dataset["all_qids"]}
    primary = _regime_metrics(dataset["all_qids"], baseline, final, dataset["regime_folds"]["standard_grouped"])
    regimes = {regime: _regime_metrics(dataset["all_qids"], baseline, final, dataset["regime_folds"][regime]) for regime in REGIMES}
    changed = [row for row in winners if row.get("accepted")]
    rescues = sum(row.get("outcome") == "rescue" for row in changed)
    destructive = sum(row.get("outcome") == "destructive" for row in changed)
    neutral00 = sum(row.get("outcome") == "neutral_00" for row in changed)
    neutral11 = sum(row.get("outcome") == "neutral_11" for row in changed)
    precision = rescues / (rescues + destructive) if rescues + destructive else 0.0
    pr_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for qid, rows in candidate_rows.items():
        for row in rows:
            pr_rows[qid].append({"outcome": row["outcome"], "utility": row["median_utility"]})
    items = [(float(row["utility"]), int(row["outcome"] == "rescue")) for rows in pr_rows.values() for row in rows if row["outcome"] in {"rescue", "neutral_00"}]
    prevalence = sum(label for _, label in items) / len(items) if items else None
    pr_auc = _average_precision(items) if items else None
    pr_bootstrap = _bootstrap_pr_lift(pr_rows)
    bootstrap = _bootstrap_gain([baseline[qid] for qid in dataset["all_qids"]], [final[qid] for qid in dataset["all_qids"]])
    top_four_frozen = all(list(predictions[qid])[:4] == list(dataset["questions"][qid]["anchor"][:4]) for qid in dataset["all_qids"])
    max_one_swap = all(list(predictions[qid]) == list(dataset["questions"][qid]["anchor"]) or list(predictions[qid])[:4] == list(dataset["questions"][qid]["anchor"][:4]) for qid in dataset["all_qids"])
    checks = {
        "recall_gain_at_least_0_003": primary["recall_gain"] >= PREREGISTRATION["promotion"]["recall_gain_min"],
        "robust_gain_at_least_0_002": primary["robust_gain"] >= PREREGISTRATION["promotion"]["robust_gain_min"],
        "paired_bootstrap_lower_positive": bootstrap["lower_95"] > 0.0,
        "intervention_precision_at_least_0_90": precision >= PREREGISTRATION["promotion"]["intervention_precision_min"],
        "all_four_regimes_non_negative": all(regimes[regime]["non_negative"] for regime in REGIMES),
        "at_least_one_rescue": rescues >= 1,
        "rescue_coverage_nonzero": rescues > 0,
        "rescue_vs_neutral_pr_auc_lift_bootstrap_lower_positive": pr_bootstrap["lower_95"] is not None and pr_bootstrap["lower_95"] > 0.0,
        "top_four_byte_identical": top_four_frozen,
        "at_most_one_slot5_swap": max_one_swap,
        "data_safety": True,
    }
    return {"metrics_vs_v16b_anchor": primary, "regimes": regimes, "paired_bootstrap_vs_v16b": bootstrap, "interventions": {"changed_queries": len(changed), "rescues": rescues, "destructive": destructive, "neutral_00": neutral00, "neutral_11": neutral11, "intervention_precision": precision}, "rescue_vs_neutral": {"candidate_rows": len(items), "prevalence": prevalence, "pr_auc": pr_auc, "pr_auc_lift_vs_prevalence": pr_auc - prevalence if pr_auc is not None and prevalence is not None else None, "query_bootstrap_pr_auc_lift": pr_bootstrap}, "promotion": {"checks": checks, "promoted": all(checks.values()), "thresholds": PREREGISTRATION["promotion"]}}


def run(args: argparse.Namespace) -> dict[str, Any]:
    pre = _validate_pre_registration(args.pre_registration, args)
    paths = _input_paths(args)
    p0_report = _read_json(paths["p0_report"], "V22-P0 report")
    _assert_p0_pass(p0_report)
    p0 = _load_p0_module(paths["p0_script"])
    import torch

    device = args.device
    if device != "cpu":
        raise V22AError("V22-A is explicitly run CPU-only here to avoid contending with user-owned GPU processes; it loads no CE backbone")
    torch.set_num_threads(PREREGISTRATION["optimisation"]["cpu_threads"])
    dataset = _build_dataset(args, p0, paths)
    heads, parameter_count = _crossfit(dataset, device=device)
    predictions, winners, candidate_rows = _combine(dataset, heads)
    evaluation = _evaluate(dataset, predictions, winners, candidate_rows)
    expected_under_10m = parameter_count < 10_000_000
    report = {
        "kind": "legalir_v22_a_cap_dba_tir_signed_utility",
        "schema_version": 1,
        "stage": "V22-A_complete_clean_complement_crossfit",
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "device": device,
        "ce_loaded": False,
        "ce_fine_tuned": False,
        "v16b_top4_changed": False,
        "pre_registration": {"path": str(args.pre_registration.resolve()), "sha256": _sha256(args.pre_registration), "config": pre["config"]},
        "inputs": pre["inputs"],
        "v22_p0_gate": {"report": str(paths["p0_report"].resolve()), "sha256": _sha256(paths["p0_report"]), "passed": True},
        "architecture": {"name": "query_conditioned_incumbent_context_CAP_DBA_TIR", "context": PREREGISTRATION["context"], "static_deltas": ["six-pack E3B challenger-minus-incumbent", "frozen-CE challenger-minus-incumbent", "candidate rank"], "parameter_count": parameter_count, "under_10m": expected_under_10m, "base_pipeline_parameters": 1_090_010_000, "end_to_end_upper_bound": 1_090_010_000 + parameter_count, "under_4b": 1_090_010_000 + parameter_count < 4_000_000_000},
        "data_contract": {"all_clean_complement_queries": len(dataset["all_qids"]), "active_complete_cache_queries": len(dataset["active_qids"]), "forced_abstain_metadata_declared_source_unavailable_queries": len(dataset["force_abstain_qids"]), "forced_abstain_qids": dataset["force_abstain_qids"], "forced_abstain_reasons": dataset["force_abstain_reasons"], "fallback": "preserve original V16b answer identity; never substitute another document"},
        "crossfit": {regime: {"outer_fold_audit": details["outer_fold_audit"], "nested_winner_calibration": details["nested_winner_calibration"]} for regime, details in heads.items()},
        **evaluation,
        "winner_traces": winners,
        "decision": {"holdout_may_open": evaluation["promotion"]["promoted"], "if_promoted": "freeze one full-fit V22-A head and run the locked holdout once without retuning", "if_failed": "close frozen c35 representation family; do not tune another threshold/MLP; only then consider V22-B K16 under its separate registered gate", "v21_closed": True, "v21_c_locked": True},
    }
    report_path = args.output_dir / "v22_a_report.json"
    trace_path = args.output_dir / "v22_a_candidate_traces.jsonl"
    if report_path.exists() or trace_path.exists():
        raise FileExistsError("V22-A output exists; refusing to replace evidence")
    _atomic_json(report_path, report)
    flattened = [row for rows in candidate_rows.values() for row in rows]
    _atomic_jsonl(trace_path, flattened)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0-report", type=Path, required=True)
    parser.add_argument("--p0-script", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--bag-cache", type=Path, required=True)
    parser.add_argument("--current-scores", type=Path, required=True)
    parser.add_argument("--e3b-scores", type=Path, required=True)
    parser.add_argument("--pre-registration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--pre-register-only", action="store_true")
    args = parser.parse_args()
    if not args.pre_register_only and args.output_dir is None:
        parser.error("--output-dir is required for V22-A execution")
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
    print(json.dumps({"output": str((args.output_dir / "v22_a_report.json").resolve()), "promoted": report["promotion"]["promoted"], "recall_gain": report["metrics_vs_v16b_anchor"]["recall_gain"], "holdout_opened": False}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
