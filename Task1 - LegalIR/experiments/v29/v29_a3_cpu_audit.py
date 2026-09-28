"""V29-A3 CPU-only selection ceiling, Pareto, and fusion audit."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
V27_EXPERIMENTS = SCRIPT_DIR.parent / "v27"
if str(V27_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(V27_EXPERIMENTS))

from v27_common import load_metadata, load_sidecar, read_json, sha256_file, write_json, write_jsonl  # noqa: E402


RRF_K = 60.0
FOLDS = tuple(range(5))


def _numeric_key(value: str) -> tuple[int, str]:
    return (int(value), value) if str(value).isdigit() else (10**18, str(value))


def _fold(qid: str) -> int:
    return int(qid) % 5


def _json_tau(value: float) -> float | None:
    return None if math.isinf(value) else float(value)


def _load_dense(path: Path, qids: set[str]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            raw = json.loads(line)
            qid = str(raw["qid"])
            if qid not in qids:
                continue
            if qid in result:
                raise ValueError(f"duplicate dense qid {qid}")
            candidates = []
            seen: set[str] = set()
            for expected_rank, item in enumerate(raw["candidates"][:1000], 1):
                docid = str(item["doc_id"])
                rank = int(item["rank"])
                score = float(item["score"])
                if rank != expected_rank or docid in seen or not math.isfinite(score):
                    raise ValueError(f"{qid}: malformed dense rank")
                seen.add(docid)
                candidates.append({"docid": docid, "dense_rank": rank, "dense_score": score})
            if len(candidates) < 500:
                raise ValueError(f"{qid}: fewer than 500 dense candidates")
            result[qid] = candidates
    if set(result) != qids:
        raise ValueError(f"dense coverage mismatch: got={len(result)} expected={len(qids)}")
    return result


def _load_frozen(path: Path, qids: set[str]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {qid: [] for qid in qids}
    seen: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            qid, docid = str(row["qid"]), str(row["docid"])
            if qid not in qids:
                raise ValueError(f"frozen cache has unexpected qid {qid}")
            key = (qid, docid)
            if key in seen:
                raise ValueError(f"duplicate frozen tuple {key}")
            score = float(row["stage2_score"])
            if not math.isfinite(score):
                raise ValueError(f"non-finite frozen score {key}")
            result[qid].append({
                "docid": docid,
                "dense_rank": int(row["dense_rank"]),
                "dense_score": float(row["dense_score"]),
                "stage2_score": score,
            })
            seen.add(key)
    if any(not rows for rows in result.values()):
        raise ValueError("frozen cache has an empty qid")
    return result


def _stage2_order(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=lambda row: (-float(row["stage2_score"]), int(row["dense_rank"]), _numeric_key(str(row["docid"]))))


def _best(items: list[dict[str, Any]], score_key: str) -> dict[str, Any] | None:
    if not items:
        return None
    return sorted(items, key=lambda row: (-float(row[score_key]), int(row["dense_rank"]), _numeric_key(str(row["docid"]))))[0]


def _choice_fields(row: dict[str, Any], name: str, choice: dict[str, Any] | None, incumbent_stage2: float, references: set[str], incumbent_recall: float) -> None:
    candidate = None if choice is None else str(choice["docid"])
    score = None if choice is None else float(choice.get("stage2_score", float("nan")))
    margin = float("-inf") if choice is None else float(choice["policy_score"]) - float(choice["incumbent_policy_score"])
    answer = row["old_answer"][:4] + ([candidate] if candidate is not None else [row["incumbent_doc"]])
    recall = len(set(answer) & references) / max(len(references), 1)
    row[f"{name}_candidate"] = candidate
    row[f"{name}_score"] = score
    row[f"{name}_margin"] = margin
    row[f"{name}_is_relevant"] = bool(candidate is not None and candidate in references)
    row[f"{name}_gain"] = recall - incumbent_recall
    row[f"{name}_stage2_beats_slot5"] = bool(choice is not None and score > incumbent_stage2)


def _with_policy_score(item: dict[str, Any], policy_score: float, incumbent_policy_score: float) -> dict[str, Any]:
    result = dict(item)
    result["policy_score"] = float(policy_score)
    result["incumbent_policy_score"] = float(incumbent_policy_score)
    return result


def _build_rows(questions: Mapping[str, Mapping[str, Any]], dense: Mapping[str, list[dict[str, Any]]], frozen: Mapping[str, list[dict[str, Any]]], sidecar: Mapping[tuple[str, str], float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for qid in sorted(questions, key=_numeric_key):
        question = questions[qid]
        references = set(question["references"])
        incumbent_doc = str(question["anchor"][4])
        incumbent_stage2 = float(sidecar[(qid, incumbent_doc)])
        dense_map = {str(item["docid"]): item for item in dense[qid]}
        incumbent_dense = dense_map.get(incumbent_doc)
        incumbent_dense_rank = int(incumbent_dense["dense_rank"]) if incumbent_dense else 1001
        incumbent_dense_score = float(incumbent_dense["dense_score"]) if incumbent_dense else float(dense[qid][-1]["dense_score"])
        new = [dict(item) for item in frozen[qid]]
        for item in new:
            item["stage2_score"] = float(item["stage2_score"])
        all_items = [*new, {"docid": incumbent_doc, "dense_rank": incumbent_dense_rank, "dense_score": incumbent_dense_score, "stage2_score": incumbent_stage2}]
        stage2_ranked = _stage2_order(all_items)
        stage2_rank = {str(item["docid"]): rank for rank, item in enumerate(stage2_ranked, 1)}
        for item in all_items:
            item["stage2_rank"] = stage2_rank[str(item["docid"])]
            item["rrf_score"] = 1.0 / (RRF_K + int(item["dense_rank"])) + 1.0 / (RRF_K + int(item["stage2_rank"]))
        incumbent_item = next(item for item in all_items if str(item["docid"]) == incumbent_doc)
        incumbent_recall = len(set(question["anchor"]) & references) / max(len(references), 1)
        row: dict[str, Any] = {
            "qid": qid,
            "fold": _fold(qid),
            "old_answer": list(question["anchor"]),
            "incumbent_doc": incumbent_doc,
            "incumbent_is_gold": incumbent_doc in references,
            "incumbent_stage2": incumbent_stage2,
            "incumbent_dense_rank": incumbent_dense_rank,
            "incumbent_dense_score": incumbent_dense_score,
            "incumbent_dense_observed": incumbent_dense is not None,
            "recall_old": incumbent_recall,
            "references": sorted(references, key=_numeric_key),
        }

        stage2_choice = _stage2_order(new)[0] if new else None
        if stage2_choice is not None:
            stage2_choice = _with_policy_score(stage2_choice, float(stage2_choice["stage2_score"]), incumbent_stage2)
        _choice_fields(row, "stage2", stage2_choice, incumbent_stage2, references, incumbent_recall)

        row["best_new_challenger"] = row["stage2_candidate"]
        row["best_new_stage2"] = row["stage2_score"]
        row["best_new_dense_rank"] = None if stage2_choice is None else int(stage2_choice["dense_rank"])
        row["best_new_dense_score"] = None if stage2_choice is None else float(stage2_choice["dense_score"])
        row["best_new_is_gold"] = bool(row["stage2_is_relevant"])
        gold = [item for item in new if str(item["docid"]) in references]
        gold_by_stage2 = _stage2_order(gold)
        best_gold = gold_by_stage2[0] if gold_by_stage2 else None
        row["new_gold_best_rank"] = min((int(item["dense_rank"]) for item in gold), default=None)
        row["new_gold_stage2"] = None if best_gold is None else float(best_gold["stage2_score"])
        row["new_gold_dense_rank"] = None if best_gold is None else int(best_gold["dense_rank"])
        row["new_gold_found_count"] = len(gold)
        row["new_gold_beats_slot5_count"] = sum(float(item["stage2_score"]) > incumbent_stage2 for item in gold)
        row["new_gold_is_best_new_challenger"] = bool(stage2_choice is not None and str(stage2_choice["docid"]) in references)
        row["new_gold_is_final_unthresholded_choice"] = row["new_gold_is_best_new_challenger"]
        row["stage2_margin"] = row["stage2_margin"]
        row["dense_rank_gap"] = None if stage2_choice is None else incumbent_dense_rank - int(stage2_choice["dense_rank"])
        row["dense_score_gap"] = None if stage2_choice is None else float(stage2_choice["dense_score"]) - incumbent_dense_score

        for cap in (25, 50, 100, 200, 500):
            eligible = [item for item in new if int(item["dense_rank"]) <= cap]
            choice = _stage2_order(eligible)[0] if eligible else None
            if choice is not None:
                choice = _with_policy_score(choice, float(choice["stage2_score"]), incumbent_stage2)
            _choice_fields(row, f"dense_rank_le_{cap}", choice, incumbent_stage2, references, incumbent_recall)

        rrf_choice = _best(new, "rrf_score")
        if rrf_choice is not None:
            rrf_choice = _with_policy_score(rrf_choice, float(rrf_choice["rrf_score"]), float(incumbent_item["rrf_score"]))
        _choice_fields(row, "rrf", rrf_choice, incumbent_stage2, references, incumbent_recall)
        row["rrf_incumbent"] = float(incumbent_item["rrf_score"])
        row["rrf_best_new"] = None if rrf_choice is None else float(rrf_choice["rrf_score"])

        dense_values = np.asarray([float(item["dense_score"]) for item in all_items], dtype=np.float64)
        stage2_values = np.asarray([float(item["stage2_score"]) for item in all_items], dtype=np.float64)
        dense_mean, dense_std = float(dense_values.mean()), float(dense_values.std())
        stage2_mean, stage2_std = float(stage2_values.mean()), float(stage2_values.std())
        for item in all_items:
            item["z_dense"] = 0.0 if dense_std == 0.0 else (float(item["dense_score"]) - dense_mean) / dense_std
            item["z_stage2"] = 0.0 if stage2_std == 0.0 else (float(item["stage2_score"]) - stage2_mean) / stage2_std
        for alpha in (0.25, 0.50, 0.75):
            tag = f"fusion_alpha_{int(alpha * 100):02d}"
            for item in all_items:
                item["fusion_score"] = alpha * float(item["z_stage2"]) + (1.0 - alpha) * float(item["z_dense"])
            choice = _best(new, "fusion_score")
            incumbent_fusion = next(item for item in all_items if str(item["docid"]) == incumbent_doc)["fusion_score"]
            if choice is not None:
                choice = _with_policy_score(choice, float(choice["fusion_score"]), float(incumbent_fusion))
            _choice_fields(row, tag, choice, incumbent_stage2, references, incumbent_recall)
            row[f"{tag}_incumbent"] = float(incumbent_fusion)
            row[f"{tag}_best_new"] = None if choice is None else float(choice["fusion_score"])
        rows.append(row)
    return rows


def _evaluate(rows: list[dict[str, Any]], candidate_field: str, margin_field: str, tau: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for row in rows:
        candidate = row.get(candidate_field)
        margin = float(row[margin_field])
        promoted = candidate is not None and margin >= tau
        answer = list(row["old_answer"][:4]) + ([candidate] if promoted else [row["incumbent_doc"]])
        refs = set(row["references"])
        recall_new = len(set(answer) & refs) / max(len(refs), 1)
        relevant = bool(promoted and candidate in refs)
        decisions.append({
            "qid": row["qid"],
            "fold": row["fold"],
            "answer": answer,
            "old_answer": row["old_answer"],
            "candidate": candidate,
            "margin": margin,
            "tau": _json_tau(tau),
            "promoted": bool(promoted),
            "promotion_is_relevant": relevant,
            "recall_old": float(row["recall_old"]),
            "recall_new": recall_new,
            "new_recall_gain": recall_new - float(row["recall_old"]),
            "top1_4_identical": answer[:4] == row["old_answer"][:4],
            "threshold_survived": bool(promoted),
        })
    metrics = _metrics(decisions)
    return decisions, metrics


def _bootstrap_lb(gains: np.ndarray, seed: int = 42, repeats: int = 10000) -> float:
    if len(gains) == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    values: list[np.ndarray] = []
    for start in range(0, repeats, 500):
        size = min(500, repeats - start)
        indices = rng.integers(0, len(gains), size=(size, len(gains)))
        values.append(gains[indices].mean(axis=1))
    return float(np.quantile(np.concatenate(values), 0.025))


def _metrics(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    old = np.asarray([float(row["recall_old"]) for row in decisions], dtype=np.float64)
    new = np.asarray([float(row["recall_new"]) for row in decisions], dtype=np.float64)
    gains = new - old
    promotions = [row for row in decisions if row["promoted"]]
    relevant = [row for row in promotions if row["promotion_is_relevant"]]
    fold_rows = []
    for fold in FOLDS:
        subset = [row for row in decisions if int(row["fold"]) == fold]
        fold_rows.append({"fold": fold, "queries": len(subset), "gain": float(np.mean([float(row["new_recall_gain"]) for row in subset])) if subset else 0.0, "promotions": sum(bool(row["promoted"]) for row in subset), "destructive": sum(bool(row["promoted"] and not row["promotion_is_relevant"]) for row in subset)})
    return {
        "queries": len(decisions),
        "baseline_recall": float(np.mean(old)) if len(old) else 0.0,
        "new_recall": float(np.mean(new)) if len(new) else 0.0,
        "gain": float(np.mean(gains)) if len(gains) else 0.0,
        "robust_gain_min_fold": min((float(row["gain"]) for row in fold_rows), default=0.0),
        "bootstrap_lb95_gain": _bootstrap_lb(gains),
        "promotions": len(promotions),
        "relevant_promotions": len(relevant),
        "rescue": sum(bool(row["promotion_is_relevant"] and float(row["new_recall_gain"]) > 0.0) for row in promotions),
        "destructive": len(promotions) - len(relevant),
        "neutral": sum(float(row["new_recall_gain"]) == 0.0 for row in decisions),
        "promotion_precision": len(relevant) / max(len(promotions), 1),
        "top1_4_identical": all(bool(row["top1_4_identical"]) for row in decisions),
        "empty_output": sum(not row["answer"] for row in decisions),
        "folds": fold_rows,
    }


def _choose_tau(rows: list[dict[str, Any]], candidate_field: str, margin_field: str) -> dict[str, Any]:
    candidate_mask = np.asarray([row.get(candidate_field) is not None for row in rows], dtype=bool)
    margin_values = np.asarray([float(row[margin_field]) if row.get(candidate_field) is not None else float("-inf") for row in rows], dtype=np.float64)
    prefix = candidate_field.removesuffix("_candidate")
    gain_field = f"{prefix}_gain"
    gains = np.asarray([float(row[gain_field]) for row in rows], dtype=np.float64)
    relevant = np.asarray([row.get(candidate_field) is not None and str(row[candidate_field]) in set(row["references"]) for row in rows], dtype=bool)
    margins = sorted(set(margin_values[candidate_mask & np.isfinite(margin_values)].tolist()))
    thresholds = [float("inf"), *margins]
    accepted: list[tuple[float, float, dict[str, Any]]] = []
    for tau in thresholds:
        promoted = candidate_mask & (margin_values >= tau)
        promotion_count = int(promoted.sum())
        relevant_count = int((promoted & relevant).sum())
        destructive = promotion_count - relevant_count
        precision = relevant_count / max(promotion_count, 1)
        gain = float(gains[promoted].sum() / max(len(rows), 1))
        if precision >= 0.90 and destructive <= 2:
            accepted.append((gain, tau, {"promotion_precision": precision, "destructive": destructive, "promotions": promotion_count}))
    if not accepted:
        return {"tau": float("inf"), "selection": "conservative_abstain_no_train_threshold_met", "train_rows": len(rows), "train_gain": 0.0, "train_precision": 0.0, "train_destructive": 0, "train_promotions": 0, "candidate_thresholds": len(thresholds)}
    gain, tau, metrics = max(accepted, key=lambda item: (item[0], item[1]))
    return {"tau": tau, "selection": "max_train_gain_precision_ge_0.90_destructive_le_2", "train_rows": len(rows), "train_gain": gain, "train_precision": metrics["promotion_precision"], "train_destructive": metrics["destructive"], "train_promotions": metrics["promotions"], "candidate_thresholds": len(thresholds)}


def _run_nested(rows: list[dict[str, Any]], configs: list[dict[str, str]]) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    outer: list[dict[str, Any]] = []
    for held_fold in FOLDS:
        train = [row for row in rows if int(row["fold"]) != held_fold]
        test = [row for row in rows if int(row["fold"]) == held_fold]
        selections = []
        for index, config in enumerate(configs):
            selection = _choose_tau(train, config["candidate_field"], config["margin_field"])
            selections.append((float(selection["train_gain"]), -index, selection, config))
        _, _, selection, config = max(selections, key=lambda item: (item[0], item[1]))
        test_decisions, test_metrics = _evaluate(test, config["candidate_field"], config["margin_field"], float(selection["tau"]))
        for decision in test_decisions:
            decision["chosen_config"] = config["name"]
            decision["outer_held_out_fold"] = held_fold
        decisions.extend(test_decisions)
        outer.append({"held_out_fold": held_fold, "chosen_config": config["name"], "tau": _json_tau(float(selection["tau"])), "selection": {key: value for key, value in selection.items() if key != "tau"} | {"tau": _json_tau(float(selection["tau"]))}, "test_metrics": test_metrics})
    decisions.sort(key=lambda row: _numeric_key(str(row["qid"])))
    return {"metrics": _metrics(decisions), "outer_folds": outer, "decisions": decisions}


def _ceiling_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = ("qid", "fold", "incumbent_doc", "incumbent_is_gold", "incumbent_stage2", "incumbent_dense_rank", "incumbent_dense_score", "best_new_challenger", "best_new_stage2", "best_new_dense_rank", "best_new_dense_score", "best_new_is_gold", "new_gold_best_rank", "new_gold_stage2", "new_gold_dense_rank", "stage2_margin", "dense_rank_gap", "dense_score_gap")
    return [{field: row.get(field) for field in fields} for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--dense-pools", type=Path, required=True)
    parser.add_argument("--frozen-cache", type=Path, required=True)
    parser.add_argument("--stage2-sidecar", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    questions = load_metadata(args.metadata)
    qids = set(questions)
    dense = _load_dense(args.dense_pools, qids)
    frozen = _load_frozen(args.frozen_cache, qids)
    sidecar = load_sidecar(args.stage2_sidecar, questions)
    rows = _build_rows(questions, dense, frozen, sidecar)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    ceiling = {
        "new_gold_found": sum(int(row["new_gold_found_count"]) for row in rows),
        "new_gold_beats_slot5": sum(int(row["new_gold_beats_slot5_count"]) for row in rows),
        "new_gold_is_best_new_challenger": sum(bool(row["new_gold_is_best_new_challenger"]) for row in rows),
        "new_gold_is_final_unthresholded_choice": sum(bool(row["new_gold_is_final_unthresholded_choice"]) for row in rows),
        "semantics": {"first_two": "gold-document counts", "last_two": "query counts; tau=-inf means always choose best new Stage2 challenger"},
    }
    write_jsonl(out / "selection_ceiling.jsonl", _ceiling_rows(rows))
    write_json(out / "selection_ceiling.json", {"kind": "legalir_v29_a3_selection_ceiling", "split": "clean_complement_only", "fold": "qid % 5", "counts": ceiling})

    thresholds = [float("-inf"), *sorted({float(row["stage2_margin"]) for row in rows}), float("inf")]
    pareto = []
    for tau in thresholds:
        decisions, metrics = _evaluate(rows, "stage2_candidate", "stage2_margin", tau)
        pareto.append({"tau": _json_tau(tau), "candidate_swaps": metrics["promotions"], "rescue": metrics["rescue"], "destructive": metrics["destructive"], "neutral": metrics["neutral"], "precision": metrics["promotion_precision"], "net_query_equivalent": metrics["gain"], "recall_at_5_gain": metrics["gain"], "robust_gain_min_fold": metrics["robust_gain_min_fold"]})
    frontier = {}
    for limit in (0, 1, 2):
        eligible = [row for row in pareto if int(row["destructive"]) <= limit]
        best = max(eligible, key=lambda row: (int(row["rescue"]), float(row["net_query_equivalent"]), float(row["precision"]))) if eligible else None
        frontier[str(limit)] = best
    write_json(out / "stage2_margin_pareto.json", {"kind": "legalir_v29_a3_stage2_margin_pareto", "split": "clean_complement_only", "threshold_count": len(pareto), "thresholds": pareto, "frontier_max_rescue_by_destructive": frontier})

    stage2_configs = [{"name": "stage2_margin", "candidate_field": "stage2_candidate", "margin_field": "stage2_margin"}]
    dense_configs = [{"name": f"dense_rank_le_{cap}", "candidate_field": f"dense_rank_le_{cap}_candidate", "margin_field": f"dense_rank_le_{cap}_margin"} for cap in (25, 50, 100, 200, 500)]
    rrf_configs = [{"name": "rrf_k60", "candidate_field": "rrf_candidate", "margin_field": "rrf_margin"}]
    fusion_configs = [{"name": f"normalized_alpha_{int(alpha * 100):02d}", "candidate_field": f"fusion_alpha_{int(alpha * 100):02d}_candidate", "margin_field": f"fusion_alpha_{int(alpha * 100):02d}_margin"} for alpha in (0.25, 0.50, 0.75)]
    nested = {}
    for name, configs in (("stage2_margin", stage2_configs), ("dense_rank_gate", dense_configs), ("rrf_k60", rrf_configs), ("normalized_fusion", fusion_configs)):
        result = _run_nested(rows, configs)
        nested[name] = {"metrics": result["metrics"], "outer_folds": result["outer_folds"]}
        write_jsonl(out / f"nested_{name}_predictions.jsonl", result["decisions"])

    gates = {}
    for name, result in nested.items():
        metrics = result["metrics"]
        gates[name] = {
            "pass": bool(metrics["rescue"] >= 12 and metrics["destructive"] <= 1 and metrics["promotion_precision"] >= 0.90 and metrics["gain"] >= 0.010),
            "near_pass": bool(8 <= metrics["rescue"] <= 11 and metrics["destructive"] <= 1),
            "rescue_ge_12": metrics["rescue"] >= 12,
            "destructive_le_1": metrics["destructive"] <= 1,
            "precision_ge_0_90": metrics["promotion_precision"] >= 0.90,
            "gain_ge_0_010": metrics["gain"] >= 0.010,
        }
    report = {
        "kind": "legalir_v29_a3_cpu_audit",
        "schema_version": 1,
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "gpu_rerun": False,
        "fold": "qid % 5",
        "contracts": {"rrf_k": 60, "dense_rank_caps": [25, 50, 100, 200, 500], "normalized_fusion_alpha": [0.25, 0.50, 0.75], "incumbent_dense_fallback": "rank1001 and rank1000 dense score when incumbent is outside top1000"},
        "inputs": {"metadata": str(args.metadata.resolve()), "metadata_sha256": sha256_file(args.metadata), "dense_pools": str(args.dense_pools.resolve()), "dense_pools_sha256": sha256_file(args.dense_pools), "frozen_cache": str(args.frozen_cache.resolve()), "frozen_cache_sha256": sha256_file(args.frozen_cache), "stage2_sidecar": str(args.stage2_sidecar.resolve()), "stage2_sidecar_sha256": sha256_file(args.stage2_sidecar)},
        "selection_ceiling": ceiling,
        "pareto_frontier_max_rescue_by_destructive": frontier,
        "nested": nested,
        "gates": gates,
        "artifacts": {"selection_ceiling": str((out / "selection_ceiling.json").resolve()), "selection_ceiling_rows": str((out / "selection_ceiling.jsonl").resolve()), "stage2_margin_pareto": str((out / "stage2_margin_pareto.json").resolve())},
    }
    write_json(out / "v29_a3_report.json", report)
    print(json.dumps({"report": str((out / "v29_a3_report.json").resolve()), "selection_ceiling": ceiling, "frontier": frontier, "nested_metrics": {name: result["metrics"] for name, result in nested.items()}, "gates": gates}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
