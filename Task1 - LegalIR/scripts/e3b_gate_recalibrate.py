"""Recalibrate the E3-B slot-5 gate on the C partition (CPU-only).

The oracle decomposition showed the gate leaves +0.015..0.017 recall on the
table (rejected rescues + blockable harm).  This study searches an extended
gate family — finer thresholds, consensus requirement, disagreement weights —
selects exactly like E3-B (gain > 0 and rescues > destructive on C, then max
gain, min harm, stronger abstention, less memory), and reports the winner on
the untouched V partition.  No training, no new weights, no holdout/public.

A reproduction check runs first: the preregistered E3-B policy evaluated here
must match the fold report V recall, otherwise the study stops.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import e3b_nested_mil_scores as e3b  # noqa: E402
import e3b_oracle_decomposition as dec  # noqa: E402
from e3b_residual_score_gate import E3BPolicy, ranked_challengers  # noqa: E402
from e3b_residual_score_gate import EPSILON  # noqa: E402
from v27_a_common import (  # noqa: E402
    EDITABLE_SLOT,
    V27ContractError,
    recall_value,
    read_json,
    write_json,
)


THRESHOLDS = [round(v * 0.05, 2) for v in range(21)]
MEMORY_WEIGHTS = (0.0, 0.1)
DISAGREEMENT_WEIGHTS = (0.25, 0.5, 1.0)
CONSENSUS = (False, True)


def apply_threshold(ranked: Sequence[Mapping[str, Any]], threshold: float) -> str | None:
    """Nominate the top-ranked challenger or abstain (ties abstain)."""
    if not ranked:
        return None
    if float(ranked[0]["utility"]) <= float(threshold) + EPSILON:
        return None
    return str(ranked[0]["doc_id"])


def select_best(
    trials: Sequence[tuple[dict[str, Any] | None, dict[str, Any]]],
    abstain_metrics: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Mirror the E3-B calibration rule over an extended config family."""
    eligible = [
        (config, metrics)
        for config, metrics in trials
        if config is not None
        and float(metrics["gain"]) > EPSILON
        and int(metrics["successful_rescues"]) > int(metrics["destructive_changes"])
    ]
    if not eligible:
        return None, dict(abstain_metrics)

    def key(item: tuple[dict[str, Any], dict[str, Any]]) -> tuple[float, int, int, float, float, float, int]:
        config, metrics = item
        return (
            float(metrics["gain"]),
            -int(metrics["destructive_changes"]),
            int(metrics["successful_rescues"]),
            float(config["threshold"]),
            -float(config["memory_weight"]),
            -float(config["disagreement_weight"]),
            -int(bool(config["consensus"])),
        )

    return max(eligible, key=key)


def _fold_queries(
    score_dir: Path, regime: str, partition: str, inner_seed: int, fold_idx: int,
    anchor_all: Mapping[str, Sequence[str]], reference: Mapping[str, Any],
) -> list[dict[str, Any]]:
    seeds = [
        dec._load_scores(score_dir, regime, partition, inner_seed, fold_idx, neural)
        for neural in (20260908, 20260909)
    ]
    out: list[dict[str, Any]] = []
    for qid in sorted(set(seeds[0]) & set(seeds[1])):
        c35 = dec._c35_from_scores(seeds[0][qid])
        anchor = [str(v) for v in anchor_all[qid]]
        gold = set(str(v) for v in (reference.get(qid, {}) or {}).get("answer", []))
        out.append({"qid": qid, "anchor": anchor, "c35": c35, "gold": gold, "seeds": seeds})
    return out


def _metrics(queries: Sequence[Mapping[str, Any]], choices: Mapping[str, Sequence[str] | None]) -> dict[str, Any]:
    recalls, rescues, destructive, changed = [], 0, 0, 0
    for q in queries:
        anchor = [str(v) for v in q["anchor"]]
        gold = set(q["gold"])
        before = recall_value(anchor, gold)
        pick = choices[q["qid"]]
        after = recall_value(_answer(anchor, pick), gold) if pick else before
        recalls.append(after)
        if _answer(anchor, pick) != anchor:
            changed += 1
        if (set(_answer(anchor, pick)) & gold) - set(anchor):
            rescues += 1
        if (set(anchor) & gold) - set(_answer(anchor, pick)):
            destructive += 1
    mean = float(sum(recalls) / len(recalls)) if recalls else 0.0
    anchor_mean = float(sum(recall_value([str(v) for v in q["anchor"]], set(q["gold"])) for q in queries) / len(queries)) if queries else 0.0
    return {
        "queries": len(queries), "recall": mean, "gain": mean - anchor_mean,
        "successful_rescues": rescues, "destructive_changes": destructive,
        "changed_queries": changed,
    }


def _answer(anchor: Sequence[str], pick: Sequence[str] | str | None) -> list[str]:
    if pick is None:
        return [str(v) for v in anchor]
    if isinstance(pick, str):
        return dec._swap(anchor, pick)
    return [str(v) for v in pick]


def run(args: argparse.Namespace) -> dict[str, Any]:
    e3b_dir = Path(args.e3b_output_dir)
    score_dir = e3b_dir / "e3b_full_scores"
    reference = read_json(args.reference)
    anchor_raw = read_json(args.anchor)
    anchor_all = {
        str(k): [str(v) for v in (vals["answer"] if isinstance(vals, dict) else vals)]
        for k, vals in anchor_raw.items()
    }
    reports = {
        "standard": read_json(args.e3b_standard_report),
        "stress": read_json(args.e3b_stress_report),
    }
    expected: set[str] = set()
    plan: list[tuple[str, dict[str, Any]]] = []
    for regime in dec.REGIMES:
        report = reports["standard"] if regime == "standard" else reports["stress"]
        for entry in report["regimes"][regime]:
            inner_seed = int(entry["inner_seed"])
            for fold in entry["folds"]:
                expected |= dec._fold_qids(score_dir, regime, inner_seed, int(fold["outer_fold"]))
            plan.append((regime, entry))
    memory_all, _ = e3b._load_memory_prior(Path(args.memory_features), expected)

    summary: dict[str, Any] = {"kind": "legalir_e3b_gate_recalibration", "regimes": {}}
    for regime, entry in plan:
        inner_seed = int(entry["inner_seed"])
        v_rows: list[dict[str, Any]] = []
        v_report_recalls: list[float] = []
        for fold in entry["folds"]:
            fold_idx = int(fold["outer_fold"])
            c_queries = _fold_queries(score_dir, regime, "calibration", inner_seed, fold_idx, anchor_all, reference)
            v_queries = _fold_queries(score_dir, regime, "validation", inner_seed, fold_idx, anchor_all, reference)
            # Rank once per (mem_w, dis_w, consensus); thresholds are post-hoc.
            ranked_c: dict[tuple[float, float, bool], dict[str, list[dict[str, Any]]]] = {}
            ranked_v: dict[tuple[float, float, bool], dict[str, list[dict[str, Any]]]] = {}
            policy0 = E3BPolicy(memory_weight=0.0, threshold=0.0)
            for mem_w, dis_w, consensus in itertools.product(MEMORY_WEIGHTS, DISAGREEMENT_WEIGHTS, CONSENSUS):
                key = (mem_w, dis_w, consensus)
                pol = E3BPolicy(memory_weight=mem_w, threshold=0.0)
                ranked_c[key] = {
                    q["qid"]: ranked_challengers(
                        qid=q["qid"], anchor=q["anchor"], c35=q["c35"], score_by_seed=q["seeds"],
                        memory_by_qid=memory_all, policy=pol,
                        disagreement_weight=dis_w, require_consensus=consensus,
                    )
                    for q in c_queries
                }
                ranked_v[key] = {
                    q["qid"]: ranked_challengers(
                        qid=q["qid"], anchor=q["anchor"], c35=q["c35"], score_by_seed=q["seeds"],
                        memory_by_qid=memory_all, policy=pol,
                        disagreement_weight=dis_w, require_consensus=consensus,
                    )
                    for q in v_queries
                }
            del policy0
            trials: list[tuple[dict[str, Any] | None, dict[str, Any]]] = []
            for (mem_w, dis_w, consensus), rank_map in ranked_c.items():
                for threshold in THRESHOLDS:
                    choices = {qid: apply_threshold(ranked, threshold) for qid, ranked in rank_map.items()}
                    metrics = _metrics(c_queries, choices)
                    trials.append((
                        {"threshold": threshold, "memory_weight": mem_w,
                         "disagreement_weight": dis_w, "consensus": consensus},
                        metrics,
                    ))
            c_abstain = _metrics(c_queries, {q["qid"]: None for q in c_queries})
            best_config, best_c = select_best(trials, c_abstain)
            # Reproduction check: E3-B preregistered policy must match the report.
            rep_raw = fold.get("selected_policy") or {}
            if rep_raw.get("kind") == "e3b_full_c35_mil_gap":
                rep_ranked = {
                    q["qid"]: ranked_challengers(
                        qid=q["qid"], anchor=q["anchor"], c35=q["c35"], score_by_seed=q["seeds"],
                        memory_by_qid=memory_all,
                        policy=E3BPolicy(float(rep_raw["memory_weight"]), float(rep_raw["threshold"])),
                    )
                    for q in v_queries
                }
                rep_choices = {qid: apply_threshold(r, float(rep_raw["threshold"])) for qid, r in rep_ranked.items()}
                rep_metrics = _metrics(v_queries, rep_choices)
                reported = float(fold["validation"]["recall"])
                if abs(rep_metrics["recall"] - reported) > 1e-9:
                    raise V27ContractError(
                        f"{regime}/{inner_seed}/fold{fold_idx}: repro {rep_metrics['recall']} != report {reported}"
                    )
                v_report_recalls.append(reported)
            if best_config is None:
                v_choices = {q["qid"]: None for q in v_queries}
            else:
                key = (best_config["memory_weight"], best_config["disagreement_weight"], best_config["consensus"])
                v_choices = {qid: apply_threshold(ranked, best_config["threshold"]) for qid, ranked in ranked_v[key].items()}
            v_metrics = _metrics(v_queries, v_choices)
            for q in v_queries:
                pick = v_choices[q["qid"]]
                v_rows.append({"r": recall_value(_answer(q["anchor"], pick), set(q["gold"]))})
            summary["regimes"].setdefault(regime, {}).setdefault(str(inner_seed), {}).setdefault("folds", []).append({
                "fold": fold_idx,
                "best_config": best_config,
                "c_metrics": best_c,
                "v_metrics": v_metrics,
            })
        for seed_key, bucket in summary["regimes"][regime].items():
            if seed_key != str(inner_seed):
                continue
            folds = bucket["folds"]
            bucket["v_recall"] = float(sum(f["v_metrics"]["recall"] * f["v_metrics"]["queries"] for f in folds)
                                       / sum(f["v_metrics"]["queries"] for f in folds))
            bucket["v_report_recall"] = float(sum(v_report_recalls) / len(v_report_recalls)) if v_report_recalls else None
            bucket["v_rescues"] = sum(f["v_metrics"]["successful_rescues"] for f in folds)
            bucket["v_destructive"] = sum(f["v_metrics"]["destructive_changes"] for f in folds)
    return summary


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
            cfgs = {}
            for f in stats["folds"]:
                c = f["best_config"]
                k = "abstain" if c is None else (
                    f"t={c['threshold']}/m={c['memory_weight']}/d={c['disagreement_weight']}/c={int(c['consensus'])}")
                cfgs[k] = cfgs.get(k, 0) + 1
            print(
                f"{regime} inner={seed} V_report={stats['v_report_recall']:.6f} "
                f"V_recal={stats['v_recall']:.6f} d={stats['v_recall'] - (stats['v_report_recall'] or 0):+.6f} "
                f"rescues={stats['v_rescues']} destructive={stats['v_destructive']} cfgs={cfgs}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
