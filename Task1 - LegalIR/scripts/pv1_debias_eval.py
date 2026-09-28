#!/usr/bin/env python3
"""PV1 debiased eval-only re-analysis on frozen cache. Complement-only, CPU-only.

Reads pv1_v1 frozen pairs + score cache, computes position-bias-cancelled
score = 0.5*[(fwd_B-fwd_A)+(rev_A-rev_B)], runs the same nested standard
OOF (2 inner seeds, same partitions via e3b.make_partitions) with the same
3 policies (tau=0, tau=ln3, abstain), and writes a NEW lineage pv1_v2_debias.
Never overwrites pv1_v1. Never opens holdout/public.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e3b_nested_mil_scores as e3b  # noqa: E402
from pv1_contract import load_pair_manifest, load_universe  # noqa: E402
from pv1_debias import debiased_challenger_score  # noqa: E402
from pv1_nested_eval import (  # noqa: E402
    INNER_SEEDS,
    POLICIES,
    _load_scores,
    _metrics,
    _subset,
)
from v27_a_common import (  # noqa: E402
    V27ContractError,
    build_regime_groups,
    group_folds,
    json_sha256,
    load_manifest,
    read_json,
    sha256_file,
    write_json,
)

EVAL_KIND = "legalir_pv1_pairwise_abstain_debias_nested_eval"


def _candidate_rows_debiased(qid, pair_manifest, scores):
    check = pair_manifest["shortlist_checks"].get(qid)
    if not isinstance(check, dict):
        raise V27ContractError(f"PV1 pair manifest lacks QID {qid}")
    incumbent = str(check["incumbent_id"])
    out = []
    for challenger in check.get("challengers", []):
        challenger = str(challenger)
        fwd = scores.get((qid, incumbent, challenger, "incumbent_first"))
        rev = scores.get((qid, incumbent, challenger, "challenger_first"))
        if fwd is None or rev is None:
            raise V27ContractError(f"{qid}/{challenger}: missing orientation in cache")
        f_map = {l: float(v) for l, v in zip(("A", "B", "C", "D"), fwd["logits_abcd"], strict=True)}
        r_map = {l: float(v) for l, v in zip(("A", "B", "C", "D"), rev["logits_abcd"], strict=True)}
        util = debiased_challenger_score(f_map, r_map)
        out.append({"doc_id": challenger, "f_rank": int(fwd["f_rank"]),
                    "f_score": float(fwd["f_score"]), "utility": float(util)})
    return out


def _predict(split, *, indices, pair_manifest, scores, policy, top_k=3):
    preds = {}
    for index in indices:
        qid = str(split.qids[index])
        anchor = [str(v) for v in split.anchor[index]]
        cands = _candidate_rows_debiased(qid, pair_manifest, scores)[:top_k]
        ranked = sorted(cands, key=lambda d: (-float(d["utility"]), int(d["f_rank"]), str(d["doc_id"])))
        sel = ranked[0] if ranked else None
        changed = False
        if policy["mode"] == "threshold" and sel is not None and float(sel["utility"]) > float(policy["tau"]):
            answer = anchor[:4] + [str(sel["doc_id"])]
            changed = True
        else:
            answer = anchor.copy()
        if len(answer) != 5 or len(set(answer)) != 5 or answer[:4] != anchor[:4]:
            raise V27ContractError(f"{qid}: violated frozen top-4 contract")
        preds[qid] = {"answer": answer}
    return preds


def _choose_policy(split, calibration, pair_manifest, scores):
    trials = []
    for policy in POLICIES:
        preds = _predict(split, indices=calibration, pair_manifest=pair_manifest, scores=scores, policy=policy)
        m = _metrics(_subset(split, calibration), preds)
        trials.append({"policy": dict(policy), "metrics": m})
    eligible = [t for t in trials if float(t["metrics"]["gain"]) > 1e-12]
    if not eligible:
        return {"mode": "abstain", "tau": None}, trials
    sel = max(eligible, key=lambda t: (float(t["metrics"]["gain"]), -int(t["metrics"]["destructive_changes"]), -int(t["metrics"]["changed_queries"]), float(t["policy"]["tau"])))
    return dict(sel["policy"]), trials


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--pair-manifest", type=Path, required=True)
    ap.add_argument("--score-dir", type=Path, required=True)
    ap.add_argument("--stage1", type=Path, required=True)
    ap.add_argument("--stage2", type=Path, required=True)
    ap.add_argument("--anchor", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--e1-manifest", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    for p, label in ((args.stage1, "stage1"), (args.stage2, "stage2"), (args.anchor, "anchor"),
                     (args.reference, "reference"), (args.e1_manifest, "e1"), (args.pairs, "pairs"),
                     (args.pair_manifest, "pair-manifest")):
        s = str(p).replace("\\", "/").casefold()
        if any(t in s for t in ("holdout", "submission", "locked")):
            raise V27ContractError(f"debias eval refuses {label} outside complement: {p}")
    pair_rows, pair_manifest = load_pair_manifest(args.pair_manifest, args.pairs)
    scores, score_info = _load_scores(args.score_dir, pair_rows, pair_manifest)
    universe = load_universe(stage1=args.stage1, stage2=args.stage2, anchor=args.anchor, reference=args.reference)
    split = universe.split
    evidence = load_manifest(args.e1_manifest)
    groups = build_regime_groups(split, evidence)["standard"]

    lineage = {"kind": EVAL_KIND, "pair_manifest_sha256": sha256_file(args.pair_manifest),
               "pairs_sha256": sha256_file(args.pairs), "score_root": str(args.score_dir.resolve()),
               "policies": list(POLICIES), "inner_seeds": list(INNER_SEEDS),
               "method": "debiased_challenger_score mean-across-orders",
               "holdout_opened": False, "public_labels_used": False}
    lineage_sha = json_sha256(lineage)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "pv1_debias_label_blind_manifest.json", lineage)

    import importlib.util
    spec = importlib.util.spec_from_file_location("pv1_bound_gate", str(Path(__file__).resolve().parent / "e3b_nested_mil_scores.py"))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    gate = mod._gate

    runs = []
    for inner_seed in INNER_SEEDS:
        oof = {}
        folds = []
        for outer_fold, (outer_train, validation) in enumerate(group_folds(groups)):
            parts, pmeta = e3b.make_partitions(split=split, groups=groups, outer_train=outer_train,
                                                validation=validation, inner_seed=inner_seed, outer_fold=outer_fold)
            policy, trace = _choose_policy(split, parts["calibration"].tolist(), pair_manifest, scores)
            preds = _predict(split, indices=parts["validation"].tolist(), pair_manifest=pair_manifest,
                             scores=scores, policy=policy)
            m = _metrics(_subset(split, parts["validation"].tolist()), preds)
            folds.append({"outer_fold": int(outer_fold), "selected_policy": policy,
                          "calibration_policy_trace": trace, "validation": m})
            oof.update(preds)
        metrics = _metrics(split, {str(q): oof[str(q)] for q in split.qids})
        runs.append({"inner_seed": int(inner_seed), "folds": folds, "metrics": metrics, "predictions": oof})
    g = dict(gate(runs))
    report = {"kind": EVAL_KIND, "lineage": lineage, "lineage_sha256": lineage_sha,
              "holdout_opened": False, "public_labels_used": False,
              "runs": [{k: v for k, v in r.items() if k != "predictions"} for r in runs],
              "gate": g, "promotion_to_stress_allowed": False,
              "next_action": "debias is diagnostic only; promotion needs new protocol + stress."}
    # compact per-seed summary to stdout
    for r in runs:
        m = r["metrics"]
        print(f"seed={r['inner_seed']} gain={m['gain']:.6f} rescue={m['successful_rescues']} destructive={m['destructive_changes']} changed={m['changed_queries']}")
    print(f"gate={g}")
    write_json(args.output_dir / "pv1_debias_standard_report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
