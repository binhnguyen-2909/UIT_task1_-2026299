"""Exact Recall@5 transition metrics with membership arithmetic on a device.

Document IDs and JSON inevitably originate on the host.  Once converted to
compact integer IDs, all set-membership, recall and promotion statistics are
evaluated by torch on the selected device.  Float64 preserves the established
CPU metric values for one-time parity decisions.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from v27_a_common import V27ContractError


def transition_metrics_device(
    split: Any,
    predictions: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    device: str,
    baseline_predictions: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
) -> dict[str, Any]:
    if split.references is None:
        raise V27ContractError("Transition metrics require qrels")
    import torch

    qids = [str(qid) for qid in split.qids]
    if not qids:
        return {
            "queries": 0, "recall": 0.0, "anchor_recall": 0.0, "gain": 0.0,
            "net_query_equivalents": 0.0, "changed_queries": 0, "improved_queries": 0,
            "harmed_queries": 0, "neutral_changed_queries": 0, "successful_rescues": 0,
            "destructive_changes": 0, "outcome_changing_precision": 1.0, "rows": [],
        }
    universe: dict[str, int] = {}
    def encode(doc_id: str) -> int:
        key = str(doc_id)
        if key not in universe:
            universe[key] = len(universe)
        return universe[key]
    anchors, proposed, golds = [], [], []
    for index, qid in enumerate(qids):
        anchor_raw = split.anchor[index] if baseline_predictions is None else baseline_predictions.get(qid, {}).get("answer")
        if not isinstance(anchor_raw, Sequence):
            raise V27ContractError(f"{qid}: transition metric baseline is missing")
        anchor = [str(value) for value in anchor_raw]
        answer = predictions.get(qid, {}).get("answer")
        if not isinstance(answer, Sequence):
            raise V27ContractError(f"{qid}: transition metric prediction is missing")
        answer = [str(value) for value in answer]
        gold = [str(value) for value in split.references[index]]
        if len(anchor) != 5 or len(answer) != 5 or not gold:
            raise V27ContractError(f"{qid}: transition metric needs five IDs and non-empty qrels")
        anchors.append([encode(value) for value in anchor])
        proposed.append([encode(value) for value in answer])
        golds.append([encode(value) for value in gold])
    max_gold = max(len(values) for values in golds)
    padded = [values + [-1] * (max_gold - len(values)) for values in golds]
    anchor_t = torch.tensor(anchors, dtype=torch.long, device=device)
    proposed_t = torch.tensor(proposed, dtype=torch.long, device=device)
    gold_t = torch.tensor(padded, dtype=torch.long, device=device)
    gold_count = (gold_t >= 0).sum(dim=1).to(dtype=torch.float64)
    anchor_hit = (anchor_t.unsqueeze(2) == gold_t.unsqueeze(1)).any(dim=2)
    proposed_hit = (proposed_t.unsqueeze(2) == gold_t.unsqueeze(1)).any(dim=2)
    before = anchor_hit.sum(dim=1).to(dtype=torch.float64) / gold_count
    after = proposed_hit.sum(dim=1).to(dtype=torch.float64) / gold_count
    delta = after - before
    changed = (anchor_t != proposed_t).any(dim=1)
    # Exact equivalents of `(set(proposed) & gold) - set(anchor)` and its
    # destructive counterpart. Predictions are guaranteed unique upstream.
    proposal_in_anchor = (proposed_t.unsqueeze(2) == anchor_t.unsqueeze(1)).any(dim=2)
    anchor_in_proposal = (anchor_t.unsqueeze(2) == proposed_t.unsqueeze(1)).any(dim=2)
    rescue = (proposed_hit & ~proposal_in_anchor).any(dim=1)
    destructive = (anchor_hit & ~anchor_in_proposal).any(dim=1)
    eps = 1.0e-12
    rows = [
        {"qid": qid, "before": float(before[index].cpu()), "after": float(after[index].cpu()),
         "delta": float(delta[index].cpu()), "changed": bool(changed[index].cpu())}
        for index, qid in enumerate(qids)
    ]
    rescues, harms = int(rescue.sum().cpu()), int(destructive.sum().cpu())
    return {
        "queries": len(rows), "recall": float(after.mean().cpu()), "anchor_recall": float(before.mean().cpu()),
        "gain": float(delta.mean().cpu()), "net_query_equivalents": float(delta.sum().cpu()),
        "changed_queries": int(changed.sum().cpu()), "improved_queries": int((delta > eps).sum().cpu()),
        "harmed_queries": int((delta < -eps).sum().cpu()),
        "neutral_changed_queries": int(((delta.abs() <= eps) & changed).sum().cpu()),
        "successful_rescues": rescues, "destructive_changes": harms,
        "outcome_changing_precision": float(rescues / (rescues + harms)) if rescues + harms else 1.0,
        "rows": rows,
    }
