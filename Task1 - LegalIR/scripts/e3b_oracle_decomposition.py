"""E3-B oracle decomposition R0/R1/R3/R35 (CPU-only, measurement only).

For every validation query, compare four slot-5 decision levels that all keep
the frozen V16b top-4 and always allow keeping the incumbent:

- R0  — the actual E3-B gate choice recorded in the fold report;
- R1  — oracle keep/swap between the incumbent and the single challenger the
  gate nominated before its threshold (recomputed threshold-free);
- R3  — oracle keep/swap among the incumbent and the top-3 ranked challengers;
- R35 — oracle keep/swap among the incumbent and every eligible c35
  challenger (scoreable by both seeds, not frozen, not the incumbent).

Oracles use complement gold labels only to *measure* where recall is lost
(gate rejection vs nomination depth vs candidate coverage).  They must never
become per-QID or per-doc rules, and every fit stays inside its inner split.

Buckets for queries with headroom (R35 > R0):
- destructive_blockable — the gate swapped away recall oracle-keep would fix;
- gate_reject — the nominated challenger rescues but the gate abstained;
- rank_2_3 — the rescue sits at rank 2-3 while the winner does not rescue;
- no_nomination — no challenger cleared mean_gap > 0 yet the pool rescues;
- deep_or_filtered — rescue only beyond rank 3 (depth or validity filter);
- no_gain_possible — the oracle cannot beat the gate choice on this query.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import e3b_nested_mil_scores as e3b  # noqa: E402
from e3b_residual_score_gate import E3BPolicy, ranked_challengers  # noqa: E402
from v27_a_common import (  # noqa: E402
    CANDIDATE_LIMIT,
    EDITABLE_SLOT,
    V27ContractError,
    recall_value,
    read_json,
    write_json,
)


EPS = 1e-12
REGIMES = ("standard", "law_source_family", "query_template_family", "near_duplicate_family")


def _swap(anchor: Sequence[str], doc_id: str) -> list[str]:
    out = [str(v) for v in anchor]
    out[EDITABLE_SLOT - 1] = str(doc_id)
    return out


def _best_of(anchor: Sequence[str], docs: Sequence[str], gold: set[str]) -> tuple[str | None, float]:
    base = recall_value(anchor, gold)
    best_doc: str | None = None
    best = base
    for doc in docs:
        value = recall_value(_swap(anchor, doc), gold)
        if value > best + EPS:
            best, best_doc = value, str(doc)
    return best_doc, best


def decompose_queries(queries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compute R-levels and miss buckets for preassembled query rows."""
    rows: list[dict[str, Any]] = []
    for q in queries:
        anchor = [str(v) for v in q["anchor"]]
        gold = set(str(v) for v in q.get("gold", set()))
        ranked = [str(v) for v in q.get("ranked", [])]
        eligible = [str(v) for v in q.get("eligible", [])]
        nominated = q.get("nominated")
        nominated = None if nominated is None else str(nominated)
        gate_choice = [str(v) for v in q["gate_choice"]]

        anchor_r = recall_value(anchor, gold)
        r0 = recall_value(gate_choice, gold)
        cands1 = [nominated] if nominated is not None else []
        _, r1 = _best_of(anchor, cands1, gold)
        _, r3 = _best_of(anchor, ranked[:3], gold)
        best_doc, r35 = _best_of(anchor, eligible, gold)
        ranked_gaps = list(q.get("ranked_gaps", []))
        c35 = [str(v) for v in q.get("c35", [])]
        unscoreable = [str(v) for v in q.get("unscoreable", [])]

        rescued_by = lambda doc: recall_value(_swap(anchor, doc), gold) > anchor_r + EPS  # noqa: E731
        if r35 <= r0 + EPS:
            bucket = "no_gain_possible"
        elif r0 < anchor_r - EPS:
            bucket = "destructive_blockable"
        elif nominated is None:
            bucket = "no_nomination"
        elif rescued_by(nominated):
            bucket = "gate_reject"
        elif any(rescued_by(doc) for doc in ranked[1:3]):
            bucket = "rank_2_3"
        else:
            bucket = "deep_or_filtered"

        sub: str | None = None
        if bucket == "deep_or_filtered" and best_doc is not None:
            if best_doc in ranked[3:]:
                sub = "deep_ranked"
            else:
                sub = "signal_miss"
        pool_gold = set(gold) & set(eligible)
        c35_gold = set(gold) & set(c35)
        rows.append({
            "anchor_r": anchor_r, "r0": r0, "r1": r1, "r3": r3, "r35": r35,
            "bucket": bucket, "sub": sub,
            "gold_absent_c35": bool(gold) and not c35_gold,
            "filter_blocked": bool(gold) and not pool_gold and bool(c35_gold)
            and any(d in unscoreable for d in c35_gold),
        })
    return rows


def _load_scores(score_dir: Path, regime: str, partition: str, inner_seed: int, fold: int, neural: int) -> dict[str, dict[str, dict[str, Any]]]:
    name = f"{partition}_inner_{inner_seed}_fold_{fold + 1}_neural_{neural}.jsonl"
    by_doc: dict[str, dict[str, dict[str, Any]]] = {}
    with open(score_dir / regime / name, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_doc.setdefault(str(row["qid"]), {})[str(row["doc_id"])] = row
    return by_doc


def _c35_from_scores(by_doc: dict[str, dict[str, Any]]) -> list[str]:
    ordered = sorted(by_doc.values(), key=lambda r: int(r["candidate_rank"]))
    return [str(r["doc_id"]) for r in ordered[:CANDIDATE_LIMIT]]


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    out: dict[str, Any] = {"queries": n, "buckets": dict(Counter(r["bucket"] for r in rows))}
    for key in ("anchor_r", "r0", "r1", "r3", "r35"):
        out[key] = float(sum(r[key] for r in rows) / n) if n else 0.0
    out["d_r1_r0"] = out["r1"] - out["r0"]
    out["d_r3_r1"] = out["r3"] - out["r1"]
    out["d_r35_r3"] = out["r35"] - out["r3"]
    out["rescuable_queries"] = sum(1 for r in rows if r["r35"] > r["r0"] + EPS)
    out["sub_buckets"] = dict(Counter(r["sub"] for r in rows if r.get("sub")))
    out["gold_absent_c35_queries"] = sum(1 for r in rows if r.get("gold_absent_c35"))
    out["filter_blocked_queries"] = sum(1 for r in rows if r.get("filter_blocked"))
    return out


def _fold_qids(score_dir: Path, regime: str, inner_seed: int, fold_idx: int) -> set[str]:
    qids: set[str] = set()
    for neural in (20260908, 20260909):
        name = f"validation_inner_{inner_seed}_fold_{fold_idx + 1}_neural_{neural}.jsonl"
        with open(score_dir / regime / name, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    qids.add(str(json.loads(line)["qid"]))
    return qids


def run(args: argparse.Namespace) -> dict[str, Any]:
    e3b_dir = Path(args.e3b_output_dir)
    score_dir = e3b_dir / "e3b_full_scores"
    reference = read_json(args.reference)
    anchor_raw = read_json(args.anchor)
    anchor_all = {
        str(k): [str(v) for v in (vals["answer"] if isinstance(vals, dict) else vals)]
        for k, vals in anchor_raw.items()
    }
    memory_path = Path(args.memory_features)
    reports = {
        "standard": read_json(args.e3b_standard_report),
        "stress": read_json(args.e3b_stress_report),
    }
    expected: set[str] = set()
    plan: list[tuple[str, dict[str, Any]]] = []
    for regime in REGIMES:
        report = reports["standard"] if regime == "standard" else reports["stress"]
        for entry in report["regimes"][regime]:
            inner_seed = int(entry["inner_seed"])
            for fold in entry["folds"]:
                expected |= _fold_qids(score_dir, regime, inner_seed, int(fold["outer_fold"]))
            plan.append((regime, entry))
    memory_all, _ = e3b._load_memory_prior(memory_path, expected)
    summary: dict[str, Any] = {"kind": "legalir_e3b_oracle_decomposition", "regimes": {}}
    for regime, entry in plan:
        inner_seed = int(entry["inner_seed"])
        seed_rows: list[dict[str, Any]] = []
        for fold in entry["folds"]:
            fold_idx = int(fold["outer_fold"])
            policy_raw = fold.get("selected_policy") or {}
            if policy_raw.get("kind") == "e3b_full_c35_mil_gap":
                policy = E3BPolicy(memory_weight=float(policy_raw["memory_weight"]), threshold=float(policy_raw["threshold"]))
                policy_note = "selected"
            else:
                # Abstain folds have no gate: rank with the locked neutral
                # memory_weight so nomination depth stays measurable.
                policy = E3BPolicy(memory_weight=0.0, threshold=float("inf"))
                policy_note = "abstain_rank_mem0"
            seeds = [
                _load_scores(score_dir, regime, "validation", inner_seed, fold_idx, neural)
                for neural in (20260908, 20260909)
            ]
            decisions = fold.get("decisions", {})
            for qid in sorted(set(seeds[0]) & set(seeds[1])):
                c35 = _c35_from_scores(seeds[0][qid])
                if len(c35) != CANDIDATE_LIMIT or len(set(c35)) != CANDIDATE_LIMIT:
                    raise V27ContractError(f"{regime}/{qid}: cannot rebuild exact c35")
                anchor = anchor_all[qid]
                ranked = ranked_challengers(
                    qid=qid, anchor=anchor, c35=c35, score_by_seed=seeds,
                    memory_by_qid=memory_all, policy=policy,
                )
                eligible = [p["doc_id"] for p in ranked] + _unranked_eligible(
                    qid, anchor, c35, seeds,
                )
                frozen = set(anchor[: EDITABLE_SLOT - 1])
                eligible_set = set(eligible)
                unscoreable = sorted(
                    d for d in c35
                    if d not in frozen and d != anchor[EDITABLE_SLOT - 1] and d not in eligible_set
                )
                dec = decisions.get(qid, {})
                gate_answer = list(anchor)
                if dec.get("changed") and dec.get("replacement"):
                    gate_answer = _swap(anchor, dec["replacement"])
                gold = set(str(v) for v in (reference.get(qid, {}) or {}).get("answer", []))
                seed_rows.append({
                    "anchor": anchor,
                    "nominated": ranked[0]["doc_id"] if ranked else None,
                    "ranked": [p["doc_id"] for p in ranked],
                    "ranked_gaps": [float(p["mean_gap"]) for p in ranked],
                    "eligible": eligible,
                    "c35": c35,
                    "unscoreable": unscoreable,
                    "gold": gold,
                    "gate_choice": gate_answer,
                })
        rows = decompose_queries(seed_rows)
        summary["regimes"].setdefault(regime, {})[str(inner_seed)] = _summarize(rows)
    return summary


def _unranked_eligible(
    qid: str,
    anchor: list[str],
    c35: list[str],
    score_by_seed: Sequence[Mapping[str, Mapping[str, Mapping[str, Any]]]],
) -> list[str]:
    """Eligible challengers the utility ranking drops (mean_gap <= 0).

    These stay valid oracle candidates for R35: the oracle is allowed to swap
    in any scoreable non-frozen non-incumbent c35 doc, even when the neural
    comparison does not favour it.
    """
    from e3b_residual_score_gate import _normalised_scores  # noqa: E402

    frozen = set(anchor[: EDITABLE_SLOT - 1])
    incumbent = anchor[EDITABLE_SLOT - 1]
    norms = []
    for seed_scores in score_by_seed:
        by_doc = seed_scores.get(qid, {})
        norms.append(_normalised_scores(by_doc, c35, qid=qid))
    if incumbent not in norms[0] or incumbent not in norms[1]:
        return []
    ranked_set: set[str] = set()
    out: list[str] = []
    for doc_id in c35:
        if doc_id in frozen or doc_id == incumbent:
            continue
        if doc_id not in norms[0] or doc_id not in norms[1]:
            continue
        gaps = [scores[doc_id] - scores[incumbent] for scores in norms]
        if sum(gaps) / len(gaps) > 0.0:
            ranked_set.add(doc_id)
            continue
        out.append(doc_id)
    return sorted(set(out) - ranked_set)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e3b-output-dir", type=Path, required=True)
    parser.add_argument("--e3b-standard-report", type=Path, required=True)
    parser.add_argument("--e3b-stress-report", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--memory-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args)
    write_json(args.output, summary)
    for regime, seeds in summary["regimes"].items():
        for seed, stats in sorted(seeds.items()):
            print(
                f"{regime} inner={seed} n={stats['queries']} "
                f"anchor={stats['anchor_r']:.6f} R0={stats['r0']:.6f} "
                f"R1={stats['r1']:.6f} R3={stats['r3']:.6f} R35={stats['r35']:.6f} "
                f"d10={stats['d_r1_r0']:+.6f} d31={stats['d_r3_r1']:+.6f} "
                f"d353={stats['d_r35_r3']:+.6f} rescuable={stats['rescuable_queries']} "
                f"buckets={json.dumps(stats['buckets'], sort_keys=True)}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
