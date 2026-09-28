"""CPU-only E3-B residual gate over two full-c35 E2 MIL score exports.

This module intentionally contains no model loading or fit code.  It makes the
slot-five policy independently testable and prevents calibration labels from
leaking into score construction.  Callers provide score files made by models
whose A-fit/A-stop dependencies were validated separately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from v27_a_common import CANDIDATE_LIMIT, EDITABLE_SLOT, V27ContractError, transition_metrics


EPSILON = 1e-12
DISAGREEMENT_WEIGHT = 0.25
E3B_MEMORY_WEIGHTS = (0.0, 0.10)
E3B_THRESHOLDS = (0.0, 0.25, 0.50, 1.00)


@dataclass(frozen=True)
class E3BPolicy:
    """A preregistered threshold policy; ``None`` is the V16b abstention."""

    memory_weight: float
    threshold: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "e3b_full_c35_mil_gap",
            "candidate_limit": CANDIDATE_LIMIT,
            "editable_slot": EDITABLE_SLOT,
            "frozen_top_k": EDITABLE_SLOT - 1,
            "disagreement_weight": DISAGREEMENT_WEIGHT,
            "memory_weight": self.memory_weight,
            "threshold": self.threshold,
            "default": "v16b_anchor_abstain",
        }


ABSTAIN_POLICY: E3BPolicy | None = None


def candidate_policies(*, memory_prior_allowed: bool) -> list[E3BPolicy]:
    weights = E3B_MEMORY_WEIGHTS if memory_prior_allowed else (0.0,)
    return [
        E3BPolicy(memory_weight=weight, threshold=threshold)
        for weight in weights
        for threshold in E3B_THRESHOLDS
    ]


def _score_value(row: Mapping[str, Any], *, qid: str, doc_id: str) -> float | None:
    empty = bool(row.get("empty_body"))
    value = row.get("document_mil_score")
    passages = row.get("passage_logits")
    if empty:
        if value is not None or passages not in ([], None):
            raise V27ContractError(f"{qid}/{doc_id}: EMPTY_BODY score row must be null with no passages")
        return None
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise V27ContractError(f"{qid}/{doc_id}: non-empty candidate has no finite E2 MIL score")
    if not isinstance(passages, list) or not passages:
        raise V27ContractError(f"{qid}/{doc_id}: non-empty candidate is missing passage logits")
    if not all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in passages):
        raise V27ContractError(f"{qid}/{doc_id}: passage logits must be finite")
    return float(value)


def validate_full_c35_scores(
    rows: Sequence[Mapping[str, Any]],
    *,
    c35_by_qid: Mapping[str, Sequence[str]],
    expected_qids: Sequence[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Validate a seed/partition score export against exact frozen c35 IDs."""

    expected = {str(qid): [str(doc_id) for doc_id in c35_by_qid[str(qid)][:CANDIDATE_LIMIT]] for qid in expected_qids}
    if set(expected) != {str(qid) for qid in expected_qids}:
        raise V27ContractError("E3-B expected score QIDs are not unique")
    parsed: dict[str, dict[str, dict[str, Any]]] = {qid: {} for qid in expected}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise V27ContractError("E3-B full score export contains a non-object row")
        qid, doc_id = str(raw.get("qid", "")), str(raw.get("doc_id", ""))
        if qid not in parsed or doc_id not in set(expected[qid]):
            raise V27ContractError(f"E3-B score escapes expected c35: {qid}/{doc_id}")
        if doc_id in parsed[qid]:
            raise V27ContractError(f"E3-B duplicate score row: {qid}/{doc_id}")
        expected_rank = expected[qid].index(doc_id) + 1
        if raw.get("candidate_rank") != expected_rank:
            raise V27ContractError(f"{qid}/{doc_id}: score candidate_rank differs from frozen c35")
        _score_value(raw, qid=qid, doc_id=doc_id)
        parsed[qid][doc_id] = dict(raw)
    for qid, c35 in expected.items():
        if set(parsed[qid]) != set(c35):
            missing = sorted(set(c35) - set(parsed[qid]))
            extra = sorted(set(parsed[qid]) - set(c35))
            raise V27ContractError(f"{qid}: full-c35 score coverage differs; missing={missing[:3]} extra={extra[:3]}")
    return parsed


def _normalised_scores(by_doc: Mapping[str, Mapping[str, Any]], c35: Sequence[str], *, qid: str) -> dict[str, float]:
    values = [
        _score_value(by_doc[doc_id], qid=qid, doc_id=doc_id)
        for doc_id in c35
        if _score_value(by_doc[doc_id], qid=qid, doc_id=doc_id) is not None
    ]
    if not values:
        return {}
    median = float(np.median(values))
    q75, q25 = np.percentile(values, [75.0, 25.0])
    scale = max(float(q75 - q25), 1.0e-6)
    return {
        doc_id: (float(value) - median) / scale
        for doc_id in c35
        if (value := _score_value(by_doc[doc_id], qid=qid, doc_id=doc_id)) is not None
    }


def _memory_rank(memory_by_qid: Mapping[str, Mapping[str, int]], qid: str, doc_id: str) -> int | None:
    value = memory_by_qid.get(qid, {}).get(doc_id)
    return int(value) if isinstance(value, int) and 1 <= int(value) <= CANDIDATE_LIMIT else None


def _memory_prior(memory_by_qid: Mapping[str, Mapping[str, int]], qid: str, doc_id: str) -> float:
    rank = _memory_rank(memory_by_qid, qid, doc_id)
    return 0.0 if rank is None else 1.0 / float(rank)


def ranked_challengers(
    *,
    qid: str,
    anchor: Sequence[str],
    c35: Sequence[str],
    score_by_seed: Sequence[Mapping[str, Mapping[str, Mapping[str, Any]]]],
    memory_by_qid: Mapping[str, Mapping[str, int]],
    policy: E3BPolicy,
    disagreement_weight: float = DISAGREEMENT_WEIGHT,
    require_consensus: bool = False,
) -> list[dict[str, Any]]:
    """Rank E3-B challengers without applying the abstention threshold.

    The incumbent must be scoreable by both E2 seeds.  Candidates need a
    positive mean gap; with ``require_consensus`` both seed gaps must be
    positive.  A soft memory prior can only break a neural comparison after
    both seeds put the candidate above the incumbent on average.
    """

    if len(anchor) != 5 or len(set(anchor)) != 5:
        raise V27ContractError(f"{qid}: E3-B requires a five-ID V16b anchor")
    c35 = [str(doc_id) for doc_id in c35[:CANDIDATE_LIMIT]]
    if len(c35) != CANDIDATE_LIMIT or len(set(c35)) != CANDIDATE_LIMIT:
        raise V27ContractError(f"{qid}: E3-B requires exact unique c35")
    anchor = [str(doc_id) for doc_id in anchor]
    if not set(anchor).issubset(set(c35)):
        raise V27ContractError(f"{qid}: E3-B anchor escapes c35")
    if len(score_by_seed) != 2:
        raise V27ContractError("E3-B requires exactly two neural E2 score exports")

    normalised: list[dict[str, float]] = []
    for seed_scores in score_by_seed:
        by_doc = seed_scores.get(qid)
        if by_doc is None:
            raise V27ContractError(f"{qid}: missing full score export for a neural seed")
        normalised.append(_normalised_scores(by_doc, c35, qid=qid))
    frozen = set(anchor[: EDITABLE_SLOT - 1])
    incumbent = anchor[EDITABLE_SLOT - 1]
    if incumbent not in normalised[0] or incumbent not in normalised[1]:
        return []

    memory_incumbent = _memory_prior(memory_by_qid, qid, incumbent)
    proposals: list[dict[str, Any]] = []
    for doc_id in c35:
        if doc_id in frozen or doc_id == incumbent:
            continue
        if doc_id not in normalised[0] or doc_id not in normalised[1]:
            continue
        gaps = [scores[doc_id] - scores[incumbent] for scores in normalised]
        mean_gap = float(sum(gaps) / len(gaps))
        if mean_gap <= 0.0:
            continue
        if require_consensus and not (gaps[0] > 0.0 and gaps[1] > 0.0):
            continue
        disagreement = abs(gaps[0] - gaps[1])
        memory_delta = _memory_prior(memory_by_qid, qid, doc_id) - memory_incumbent
        utility = mean_gap - float(disagreement_weight) * disagreement + policy.memory_weight * memory_delta
        proposals.append({
            "doc_id": doc_id,
            "mean_gap": mean_gap,
            "seed_gaps": gaps,
            "disagreement": disagreement,
            "memory_delta": memory_delta,
            "utility": float(utility),
        })
    proposals.sort(key=lambda row: (-float(row["utility"]), -float(row["mean_gap"]), str(row["doc_id"])))
    return proposals


def propose_slot5(
    *,
    qid: str,
    anchor: Sequence[str],
    c35: Sequence[str],
    score_by_seed: Sequence[Mapping[str, Mapping[str, Mapping[str, Any]]]],
    memory_by_qid: Mapping[str, Mapping[str, int]],
    policy: E3BPolicy,
) -> dict[str, Any] | None:
    """Return the best E3-B challenger and diagnostic, or abstain.

    The incumbent must be scoreable by both E2 seeds.  A soft memory prior can
    only break a neural comparison after both seeds put the candidate above the
    incumbent on average; it cannot rescue a neural loser.
    """

    proposals = ranked_challengers(
        qid=qid,
        anchor=anchor,
        c35=c35,
        score_by_seed=score_by_seed,
        memory_by_qid=memory_by_qid,
        policy=policy,
    )
    if not proposals:
        return None
    winner = proposals[0]
    # A tie with the keep action or threshold is always an abstention.
    if float(winner["utility"]) <= float(policy.threshold) + EPSILON:
        return None
    return winner


def apply_policy(
    split: Any,
    indices: Sequence[int],
    score_by_seed: Sequence[Mapping[str, Mapping[str, Mapping[str, Any]]]],
    memory_by_qid: Mapping[str, Mapping[str, int]],
    policy: E3BPolicy | None,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, dict[str, Any]]]:
    """Produce c35-constrained slot-five outputs and label-free diagnostics."""

    predictions: dict[str, dict[str, list[str]]] = {}
    decisions: dict[str, dict[str, Any]] = {}
    for index in indices:
        qid = str(split.qids[index])
        anchor = [str(doc_id) for doc_id in split.anchor[index]]
        c35 = [str(doc_id) for doc_id in split.candidates[index][:CANDIDATE_LIMIT]]
        answer = list(anchor)
        proposal = None if policy is None else propose_slot5(
            qid=qid,
            anchor=anchor,
            c35=c35,
            score_by_seed=score_by_seed,
            memory_by_qid=memory_by_qid,
            policy=policy,
        )
        if proposal is not None:
            answer[EDITABLE_SLOT - 1] = str(proposal["doc_id"])
        if answer[: EDITABLE_SLOT - 1] != anchor[: EDITABLE_SLOT - 1]:
            raise V27ContractError(f"{qid}: E3-B mutated the frozen V16b top-4")
        if len(answer) != 5 or len(set(answer)) != 5 or not set(answer).issubset(set(c35)):
            raise V27ContractError(f"{qid}: E3-B output violates five-ID/c35 contract")
        predictions[qid] = {"answer": answer}
        decisions[qid] = {
            "changed": proposal is not None,
            "incumbent": anchor[EDITABLE_SLOT - 1],
            "replacement": None if proposal is None else proposal["doc_id"],
            "proposal": proposal,
        }
    return predictions, decisions


def compact_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "rows"}


def select_calibration_policy(
    calibration: Any,
    score_by_seed: Sequence[Mapping[str, Mapping[str, Mapping[str, Any]]]],
    memory_by_qid: Mapping[str, Mapping[str, int]],
    *,
    memory_prior_allowed: bool,
) -> tuple[E3BPolicy | None, dict[str, Any], list[dict[str, Any]]]:
    """Select only from the fixed eight-policy family using calibration qrels."""

    indices = list(range(len(calibration.qids)))
    baseline, _ = apply_policy(calibration, indices, score_by_seed, memory_by_qid, ABSTAIN_POLICY)
    trials: list[tuple[E3BPolicy | None, dict[str, Any]]] = [
        (ABSTAIN_POLICY, transition_metrics(calibration, baseline))
    ]
    for policy in candidate_policies(memory_prior_allowed=memory_prior_allowed):
        prediction, _ = apply_policy(calibration, indices, score_by_seed, memory_by_qid, policy)
        trials.append((policy, transition_metrics(calibration, prediction)))
    eligible = [
        (policy, metrics)
        for policy, metrics in trials[1:]
        if float(metrics["gain"]) > EPSILON and int(metrics["successful_rescues"]) > int(metrics["destructive_changes"])
    ]
    if not eligible:
        selected, selected_metrics = ABSTAIN_POLICY, trials[0][1]
    else:
        def key(item: tuple[E3BPolicy | None, Mapping[str, Any]]) -> tuple[float, int, int, float, float]:
            policy, metrics = item
            assert policy is not None
            # Maximise gain.  The preregistered ties then minimise harm,
            # prefer stronger abstention, and minimise use of memory.
            return (
                float(metrics["gain"]),
                -int(metrics["destructive_changes"]),
                int(metrics["successful_rescues"]),
                float(policy.threshold),
                -float(policy.memory_weight),
            )
        selected, selected_metrics = max(eligible, key=key)
    trace = [
        {
            "policy": {"kind": "v16b_anchor_abstain"} if policy is None else policy.as_dict(),
            "metrics": compact_metrics(metrics),
            "eligible": bool(
                policy is not None
                and float(metrics["gain"]) > EPSILON
                and int(metrics["successful_rescues"]) > int(metrics["destructive_changes"])
            ),
        }
        for policy, metrics in trials
    ]
    return selected, compact_metrics(selected_metrics), trace


def paired_group_confidence_interval(
    split: Any,
    predictions: Mapping[str, Mapping[str, Sequence[str]]],
    groups: Sequence[Any],
    *,
    seed: int,
    repetitions: int = 1_000,
) -> dict[str, Any]:
    """Cluster bootstrap for paired macro-Recall delta; groups are resampled."""

    if split.references is None or len(groups) != len(split.qids):
        raise V27ContractError("E3-B paired group interval requires qrels and one group per query")
    by_group: dict[str, list[float]] = {}
    for index, qid in enumerate(split.qids):
        answer = predictions.get(str(qid), {}).get("answer")
        if not isinstance(answer, Sequence):
            raise V27ContractError(f"{qid}: E3-B interval is missing a prediction")
        gold = set(str(doc_id) for doc_id in split.references[index])
        before = len(set(map(str, split.anchor[index])) & gold) / len(gold)
        after = len(set(map(str, answer)) & gold) / len(gold)
        by_group.setdefault(str(groups[index]), []).append(float(after - before))
    group_values = list(by_group.values())
    if len(group_values) < 2:
        raise V27ContractError("E3-B paired group interval needs at least two groups")
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(repetitions), dtype=np.float64)
    for iteration in range(int(repetitions)):
        selected = rng.integers(0, len(group_values), size=len(group_values))
        flattened = [value for group_index in selected for value in group_values[int(group_index)]]
        samples[iteration] = float(np.mean(flattened))
    return {
        "kind": "paired_group_bootstrap_macro_recall_delta",
        "groups": len(group_values),
        "queries": len(split.qids),
        "repetitions": int(repetitions),
        "seed": int(seed),
        "point_estimate": float(np.mean([value for values in group_values for value in values])),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
    }
