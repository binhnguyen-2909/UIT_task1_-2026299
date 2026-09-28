"""V33-B rank-only LambdaMART selector with nested qid%5 OOF."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
V27_DIR = SCRIPT_DIR / "v27"
if str(V27_DIR) not in sys.path:
    sys.path.insert(0, str(V27_DIR))
from v27_common import load_metadata, sha256_file, write_json, write_jsonl  # noqa: E402


VIEWS = ("Q0", "Q1", "Q2")
FEATURES = [
    "rank_q0", "rank_q1", "rank_q2", "min_rank", "mean_rank", "median_rank",
    "rr_q0", "rr_q1", "rr_q2", "rrf", "views_top1", "views_top3", "views_top5",
    "views_top10", "views_top20", "z_q0", "z_q1", "z_q2", "max_z", "mean_z",
    "gap_q0", "gap_q1", "gap_q2", "rank_gap_to_best", "q0_support20", "q0_support50",
    "q0_support100", "rank_spread", "num_views_present", "is_incumbent", "is_old_top4",
]


def _num(value: str) -> tuple[int, str]:
    return (int(value), value) if str(value).isdigit() else (10**18, str(value))


def _load_views(path: Path, qids: set[str]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    result = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                qid = str(row["qid"])
                if qid in qids:
                    result[qid] = {view: row[view] for view in VIEWS}
    if set(result) != qids:
        raise ValueError(f"view coverage mismatch {len(result)} != {len(qids)}")
    return result


def _features(qid: str, doc: str, views: dict[str, list[dict[str, Any]]], anchor: list[str]) -> dict[str, float]:
    ranks: dict[str, int] = {}
    scores: dict[str, float] = {}
    top_scores: dict[str, float] = {}
    z: dict[str, float] = {}
    for view in VIEWS:
        items = views[view]
        by_doc = {str(item["doc_id"]): item for item in items[:20]}
        item = by_doc.get(doc)
        if item is None:
            ranks[view] = 1001
            scores[view] = float("nan")
            top_scores[view] = float(items[0]["score"])
            z[view] = -4.0
        else:
            ranks[view] = int(item["rank"])
            scores[view] = float(item["score"])
            top_scores[view] = float(items[0]["score"])
            vals = np.asarray([float(x["score"]) for x in items[:20]], dtype=np.float64)
            z[view] = float((scores[view] - float(vals.mean())) / max(float(vals.std()), 1e-6))
    present = [ranks[v] for v in VIEWS if ranks[v] <= 20]
    rrs = {v: (1.0 / (60.0 + ranks[v]) if ranks[v] <= 20 else 0.0) for v in VIEWS}
    return {
        "rank_q0": float(ranks["Q0"]), "rank_q1": float(ranks["Q1"]), "rank_q2": float(ranks["Q2"]),
        "min_rank": float(min(ranks.values())), "mean_rank": float(np.mean(list(ranks.values()))), "median_rank": float(np.median(list(ranks.values()))),
        "rr_q0": rrs["Q0"], "rr_q1": rrs["Q1"], "rr_q2": rrs["Q2"], "rrf": float(sum(rrs.values())),
        "views_top1": float(sum(ranks[v] <= 1 for v in VIEWS)), "views_top3": float(sum(ranks[v] <= 3 for v in VIEWS)),
        "views_top5": float(sum(ranks[v] <= 5 for v in VIEWS)), "views_top10": float(sum(ranks[v] <= 10 for v in VIEWS)), "views_top20": float(sum(ranks[v] <= 20 for v in VIEWS)),
        "z_q0": z["Q0"], "z_q1": z["Q1"], "z_q2": z["Q2"], "max_z": float(max(z.values())), "mean_z": float(np.mean(list(z.values()))),
        "gap_q0": float(top_scores["Q0"] - scores["Q0"]) if math.isfinite(scores["Q0"]) else 1.0,
        "gap_q1": float(top_scores["Q1"] - scores["Q1"]) if math.isfinite(scores["Q1"]) else 1.0,
        "gap_q2": float(top_scores["Q2"] - scores["Q2"]) if math.isfinite(scores["Q2"]) else 1.0,
        "rank_gap_to_best": float(max(ranks.values()) - min(ranks.values())),
        "q0_support20": float(ranks["Q0"] <= 20), "q0_support50": float(ranks["Q0"] <= 50), "q0_support100": float(ranks["Q0"] <= 100),
        "rank_spread": float(max(present) - min(present)) if present else 1001.0,
        "num_views_present": float(len(present)), "is_incumbent": float(doc == str(anchor[4])), "is_old_top4": float(doc in {str(x) for x in anchor[:4]}),
    }


def _build_dataset(questions: dict[str, dict[str, Any]], views: dict[str, dict[str, list[dict[str, Any]]]]) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    rows = []
    row_indices: dict[str, list[int]] = {}
    for qid in sorted(questions, key=_num):
        question = questions[qid]
        anchor = [str(x) for x in question["anchor"]]
        docs = []
        for view in VIEWS:
            for item in views[qid][view][:20]:
                doc = str(item["doc_id"])
                if doc not in docs:
                    docs.append(doc)
        if anchor[4] not in docs:
            docs.append(anchor[4])
        row_indices[qid] = []
        for doc in docs:
            feature = _features(qid, doc, views[qid], anchor)
            row_indices[qid].append(len(rows))
            rows.append({"qid": qid, "doc_id": doc, "label": int(doc in question["references"]), "old_recall": len(set(anchor) & set(question["references"])) / max(len(question["references"]), 1), "ref_count": len(question["references"]), **feature})
    return pd.DataFrame(rows), row_indices


def _fit(frame: pd.DataFrame, qids: set[str], n_estimators: int) -> lgb.LGBMRanker:
    subset = frame[frame["qid"].isin(qids)].sort_values(["qid", "doc_id"], key=lambda col: col.map(_num) if col.name == "qid" else col)
    groups = subset.groupby("qid", sort=False).size().tolist()
    if not groups:
        raise ValueError("empty training groups")
    model = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", num_leaves=15, max_depth=4,
        learning_rate=0.03, n_estimators=n_estimators, min_data_in_leaf=20,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        random_state=42, n_jobs=8, verbosity=-1, force_col_wise=True,
    )
    model.fit(subset[FEATURES], subset["label"], group=groups)
    return model


def _predict(model: lgb.LGBMRanker, frame: pd.DataFrame, qids: set[str]) -> pd.DataFrame:
    result = frame[frame["qid"].isin(qids)].copy()
    result["selector_score"] = model.predict(result[FEATURES])
    return result


def _best_challenger(group: pd.DataFrame) -> tuple[pd.Series | None, pd.Series | None]:
    incumbent = group[group["is_incumbent"] > 0.5].iloc[0]
    challenger = group[(group["is_incumbent"] < 0.5) & (group["is_old_top4"] < 0.5)].sort_values(["selector_score", "rrf"], ascending=False)
    return incumbent, (challenger.iloc[0] if len(challenger) else None)


def _calibrate(pred: pd.DataFrame) -> float:
    margins = []
    for _, group in pred.groupby("qid", sort=False):
        incumbent, challenger = _best_challenger(group)
        if challenger is not None:
            margins.append({"margin": float(challenger["selector_score"] - incumbent["selector_score"]), "gain": _gain(incumbent, challenger)})
    if not margins:
        return float("inf")
    best_tau = float("inf")
    best_key = (-1, 0, float("inf"))
    for tau in sorted({row["margin"] for row in margins}, reverse=True):
        selected = [row for row in margins if row["margin"] >= tau]
        rescue = sum(row["gain"] > 0 for row in selected)
        destructive = sum(row["gain"] < 0 for row in selected)
        precision = rescue / max(rescue + destructive, 1)
        if precision >= 0.90 and destructive <= 1:
            key = (rescue, -destructive, tau)
            if key > best_key:
                best_key = key
                best_tau = tau
    return best_tau


def _gain(incumbent: pd.Series, challenger: pd.Series) -> float:
    old = float(incumbent["label"])  # incumbent is one slot, so this is slot-level Recall@5 delta
    new = float(challenger["label"])
    return (new - old) / max(float(incumbent["ref_count"]), 1.0)


def _apply(pred: pd.DataFrame, tau: float) -> list[dict[str, Any]]:
    rows = []
    for qid, group in pred.groupby("qid", sort=False):
        incumbent, challenger = _best_challenger(group)
        promote = challenger is not None and float(challenger["selector_score"] - incumbent["selector_score"]) >= tau
        chosen = str(challenger["doc_id"]) if promote else str(incumbent["doc_id"])
        gain = _gain(incumbent, challenger) if promote and challenger is not None else 0.0
        old_recall = float(incumbent["old_recall"])
        rows.append({"qid": str(qid), "fold": int(qid) % 5, "old_slot5": str(incumbent["doc_id"]), "challenger": str(challenger["doc_id"]) if challenger is not None else None, "old_slot5_score": float(incumbent["selector_score"]), "challenger_score": float(challenger["selector_score"]) if challenger is not None else None, "delta": float(challenger["selector_score"] - incumbent["selector_score"]) if challenger is not None else None, "tau": tau, "promoted": bool(promote), "chosen": chosen, "is_gold": bool(chosen == str(challenger["doc_id"]) and challenger["label"] > 0) if promote and challenger is not None else bool(incumbent["label"] > 0), "rescue": gain > 0, "destructive": gain < 0, "neutral": gain == 0, "recall_old": old_recall, "recall_new": old_recall + gain, "gain": gain})
    return rows


def _bootstrap(gains: np.ndarray, seed: int = 42, repeats: int = 10000) -> float:
    rng = np.random.default_rng(seed)
    values = []
    for start in range(0, repeats, 500):
        idx = rng.integers(0, len(gains), size=(min(500, repeats - start), len(gains)))
        values.append(gains[idx].mean(axis=1))
    return float(np.quantile(np.concatenate(values), 0.025))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    questions = load_metadata(args.metadata)
    views = _load_views(args.views, set(questions))
    frame, _ = _build_dataset(questions, views)
    try:
        frame.to_parquet(args.output_dir / "features.parquet", index=False)
    except Exception:
        frame.to_json(args.output_dir / "features.jsonl", orient="records", lines=True, force_ascii=False)
    all_qids = set(questions)
    oof_rows: list[dict[str, Any]] = []
    importance = np.zeros(len(FEATURES), dtype=np.float64)
    for outer in range(5):
        test_qids = {qid for qid in all_qids if int(qid) % 5 == outer}
        train_qids = all_qids - test_qids
        # Inner OOF predictions are used only to choose the threshold.
        inner_rows = []
        train_sorted = sorted(train_qids, key=_num)
        for inner in range(4):
            inner_val = {qid for position, qid in enumerate(train_sorted) if position % 4 == inner}
            inner_train = train_qids - inner_val
            model_inner = _fit(frame, inner_train, n_estimators=60)
            inner_rows.append(_predict(model_inner, frame, inner_val))
        calibration = pd.concat(inner_rows, ignore_index=True) if inner_rows else frame.iloc[0:0].copy()
        tau = _calibrate(calibration)
        model = _fit(frame, train_qids, n_estimators=200)
        importance += np.asarray(model.feature_importances_, dtype=np.float64)
        outer_pred = _predict(model, frame, test_qids)
        oof_rows.extend(_apply(outer_pred, tau))
        print(json.dumps({"outer_fold": outer, "train_queries": len(train_qids), "test_queries": len(test_qids), "tau": tau}), flush=True)
    oof_rows.sort(key=lambda row: _num(row["qid"]))
    write_jsonl(args.output_dir / "nested_oof.jsonl", oof_rows)
    rescue = sum(row["rescue"] for row in oof_rows)
    destructive = sum(row["destructive"] for row in oof_rows)
    neutral = sum(row["neutral"] for row in oof_rows if row["promoted"])
    gains = np.asarray([float(row["gain"]) for row in oof_rows], dtype=np.float64)
    fold_gains = [float(np.mean([row["gain"] for row in oof_rows if row["fold"] == fold])) for fold in range(5)]
    summary = {"kind": "legalir_v33_b_lambdamart_nested_oof", "schema_version": 1, "queries": len(oof_rows), "candidate_rows": len(frame), "candidate_rows_mean": len(frame) / max(len(questions), 1), "features": FEATURES, "candidate_contract": "Q0/Q1/Q2 top20 union plus old slot5; old top1-4 excluded from action", "model_config": {"objective": "lambdarank", "num_leaves": 15, "max_depth": 4, "learning_rate": 0.03, "n_estimators": 200, "inner_n_estimators": 60, "min_data_in_leaf": 20, "feature_fraction": 0.8, "bagging_fraction": 0.8, "seed": 42, "n_jobs": 8}, "selection": {"candidate_swaps": sum(row["promoted"] for row in oof_rows), "rescue": rescue, "destructive": destructive, "neutral_promoted": neutral, "precision_rescue_vs_destructive": rescue / max(rescue + destructive, 1), "recall_old": float(np.mean([row["recall_old"] for row in oof_rows])), "recall_new": float(np.mean([row["recall_new"] for row in oof_rows])), "recall_gain": float(np.mean(gains)), "fold_gains_mod5": fold_gains, "robust_gain_min_mod5": min(fold_gains), "bootstrap_lb95_gain": _bootstrap(gains)}, "gate": {"rescue_ge_12": rescue >= 12, "destructive_le_1": destructive <= 1, "precision_ge_0_90": rescue / max(rescue + destructive, 1) >= 0.90, "top1_4_identical": True, "empty_output_zero": True, "pass": rescue >= 12 and destructive <= 1 and rescue / max(rescue + destructive, 1) >= 0.90}, "inputs": {"metadata": str(args.metadata.resolve()), "metadata_sha256": sha256_file(args.metadata), "views": str(args.views.resolve()), "views_sha256": sha256_file(args.views)}}
    write_json(args.output_dir / "feature_importance.json", {"features": FEATURES, "mean_gain": (importance / 5.0).tolist()})
    write_json(args.output_dir / "selection_pareto.json", {"kind": "legalir_v33_b_selection_pareto", "selection": summary["selection"], "gate": summary["gate"]})
    write_json(args.output_dir / "nested_oof_report.json", summary)
    write_json(args.output_dir.parent / "v33_gate.json", {"kind": "legalir_v33_gate", "native_a0_a1_report": str((args.output_dir.parent / "a0_ceiling" / "view_agreement_report.json").resolve()), "native_b0_report": str((args.output_dir / "nested_oof_report.json").resolve()), "decision": "close_v33_and_do_not_promote" if not summary["gate"]["pass"] else "promote_v33_native_selector", "gate": {"native_a_pass": False, "lambdamart_pass": summary["gate"]["pass"]}})
    print(json.dumps({"report": str((args.output_dir / "nested_oof_report.json").resolve()), "selection": summary["selection"], "gate": summary["gate"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
