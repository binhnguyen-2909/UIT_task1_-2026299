"""Pure, testable policy code for E3-C's incumbent-aware duo comparator.

E3-C never reorders a query.  It may replace only V16b slot five, keeps the
top four byte-for-byte, and abstains whenever its comparator does not clear a
fixed positive margin.  This module has no torch/model dependency so that the
policy and its calibration can be audited independently of GPU fitting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from v27_a_common import CANDIDATE_LIMIT, EDITABLE_SLOT, V27ContractError, transition_metrics


EPSILON = 1.0e-12
E3C_THRESHOLDS = (0.00, 0.01, 0.02, 0.05, 0.10)


@dataclass(frozen=True)
class E3CPolicy:
    """Strict positive predicted-delta margin; ``None`` means V16b keep."""

    threshold: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "e3c_duo_delta_margin",
            "candidate_limit": CANDIDATE_LIMIT,
            "editable_slot": EDITABLE_SLOT,
            "frozen_top_k": EDITABLE_SLOT - 1,
            "threshold": float(self.threshold),
            "default": "v16b_anchor_abstain",
        }


ABSTAIN_POLICY: E3CPolicy | None = None


def soft_delta_target(*, gold: Sequence[str], incumbent: str, candidate: str) -> tuple[float, float]:
    """Return exact Recall@5 delta and its BCE target in [0, 1]."""

    truth = {str(value) for value in gold}
    if not truth:
        raise V27ContractError("E3-C duo target requires at least one qrel")
    delta = (float(str(candidate) in truth) - float(str(incumbent) in truth)) / float(len(truth))
    if not -1.0 <= delta <= 1.0:
        raise V27ContractError("E3-C duo delta escaped [-1, 1]")
    return delta, (1.0 + delta) / 2.0


def _finite_score(value: Any, *, qid: str, doc_id: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise V27ContractError(f"{qid}/{doc_id}: E3-C comparator score must be finite")
    return float(value)


def validate_duo_scores(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: Any,
    expected_indices: Sequence[int],
) -> dict[str, dict[str, float]]:
    """Check exact challenger coverage for a C/V E3-C score export."""

    expected: dict[str, set[str]] = {}
    incumbents: dict[str, str] = {}
    for index in expected_indices:
        qid = str(split.qids[int(index)])
        anchor = [str(value) for value in split.anchor[int(index)]]
        c35 = [str(value) for value in split.candidates[int(index)][:CANDIDATE_LIMIT]]
        if len(anchor) != 5 or len(set(anchor)) != 5 or not set(anchor).issubset(c35):
            raise V27ContractError(f"{qid}: E3-C requires an in-c35 five-ID anchor")
        expected[qid] = set(c35) - set(anchor[:EDITABLE_SLOT])
        incumbents[qid] = anchor[EDITABLE_SLOT - 1]

    parsed: dict[str, dict[str, float]] = {qid: {} for qid in expected}
    seen: dict[str, set[str]] = {qid: set() for qid in expected}
    for row in rows:
        if not isinstance(row, Mapping):
            raise V27ContractError("E3-C score export contains a non-object row")
        qid, doc_id = str(row.get("qid", "")), str(row.get("candidate_doc_id", ""))
        if qid not in expected or doc_id not in expected[qid] or doc_id in seen[qid]:
            raise V27ContractError(f"E3-C score export escapes exact competitor set: {qid}/{doc_id}")
        seen[qid].add(doc_id)
        if row.get("incumbent_doc_id") != incumbents[qid]:
            raise V27ContractError(f"{qid}/{doc_id}: E3-C incumbent provenance drift")
        eligible = row.get("eligible")
        if not isinstance(eligible, bool):
            raise V27ContractError(f"{qid}/{doc_id}: E3-C score row lacks boolean eligibility")
        if eligible:
            parsed[qid][doc_id] = _finite_score(row.get("predicted_delta"), qid=qid, doc_id=doc_id)
        elif row.get("predicted_delta") is not None:
            raise V27ContractError(f"{qid}/{doc_id}: ineligible E3-C score must be null")
    for qid, competitors in expected.items():
        if seen[qid] != competitors:
            missing = sorted(competitors - seen[qid])
            extra = sorted(seen[qid] - competitors)
            raise V27ContractError(f"{qid}: E3-C score coverage differs; missing={missing[:3]} extra={extra[:3]}")
        # ``parsed`` intentionally holds only scoreable pairs so an EMPTY_BODY
        # document cannot create a fake score or override default abstention.
    return parsed


def apply_policy(
    split: Any,
    indices: Sequence[int],
    score_by_qid: Mapping[str, Mapping[str, float]],
    policy: E3CPolicy | None,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, dict[str, Any]]]:
    """Apply default-abstain E3-C to frozen c35 slot five only."""

    predictions: dict[str, dict[str, list[str]]] = {}
    decisions: dict[str, dict[str, Any]] = {}
    for index in indices:
        qid = str(split.qids[int(index)])
        anchor = [str(value) for value in split.anchor[int(index)]]
        c35 = [str(value) for value in split.candidates[int(index)][:CANDIDATE_LIMIT]]
        if len(anchor) != 5 or len(set(anchor)) != 5 or not set(anchor).issubset(c35):
            raise V27ContractError(f"{qid}: E3-C anchor violates c35/top-5 contract")
        incumbent = anchor[EDITABLE_SLOT - 1]
        scores = score_by_qid.get(qid, {})
        replacement: str | None = None
        predicted_delta: float | None = None
        if policy is not None:
            valid = [
                (str(doc_id), _finite_score(value, qid=qid, doc_id=str(doc_id)))
                for doc_id, value in scores.items()
                if str(doc_id) in set(c35) - set(anchor[:EDITABLE_SLOT])
            ]
            if valid:
                valid.sort(key=lambda item: (-item[1], item[0]))
                candidate, value = valid[0]
                if value > float(policy.threshold) + EPSILON:
                    replacement, predicted_delta = candidate, value
        answer = list(anchor)
        if replacement is not None:
            answer[EDITABLE_SLOT - 1] = replacement
        if answer[:EDITABLE_SLOT - 1] != anchor[:EDITABLE_SLOT - 1]:
            raise V27ContractError(f"{qid}: E3-C mutated a frozen top-four ID")
        if len(answer) != 5 or len(set(answer)) != 5 or not set(answer).issubset(c35):
            raise V27ContractError(f"{qid}: E3-C violated five-ID/c35 output contract")
        predictions[qid] = {"answer": answer}
        decisions[qid] = {
            "changed": replacement is not None,
            "incumbent": incumbent,
            "replacement": replacement,
            "predicted_delta": predicted_delta,
        }
    return predictions, decisions


def compact_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "rows"}


def select_calibration_policy(
    calibration: Any,
    scores: Mapping[str, Mapping[str, float]],
    *,
    metrics_fn: Callable[[Any, Mapping[str, Mapping[str, Sequence[str]]]], Mapping[str, Any]] = transition_metrics,
) -> tuple[E3CPolicy | None, dict[str, Any], list[dict[str, Any]]]:
    """Select a preregistered strict margin using C labels only."""

    indices = list(range(len(calibration.qids)))
    trials: list[tuple[E3CPolicy | None, dict[str, Any]]] = []
    for policy in [ABSTAIN_POLICY, *[E3CPolicy(value) for value in E3C_THRESHOLDS]]:
        prediction, _ = apply_policy(calibration, indices, scores, policy)
        trials.append((policy, compact_metrics(metrics_fn(calibration, prediction))))
    eligible = [
        item for item in trials[1:]
        if float(item[1]["gain"]) > EPSILON
        and int(item[1]["successful_rescues"]) > int(item[1]["destructive_changes"])
    ]
    if eligible:
        selected, metrics = max(
            eligible,
            key=lambda item: (
                float(item[1]["gain"]),
                -int(item[1]["destructive_changes"]),
                int(item[1]["successful_rescues"]),
                float(item[0].threshold),  # stronger abstention on ties
            ),
        )
    else:
        selected, metrics = trials[0]
    trace = [
        {
            "policy": {"kind": "v16b_anchor_abstain"} if policy is None else policy.as_dict(),
            "metrics": row,
            "eligible": bool(policy is not None and float(row["gain"]) > EPSILON and int(row["successful_rescues"]) > int(row["destructive_changes"])),
        }
        for policy, row in trials
    ]
    return selected, metrics, trace
