"""V22-P0 rescue--neutral closure audit (CPU-only, clean-complement only).

The audit is intentionally diagnostic.  It never trains, loads a model, opens
the locked holdout, or creates a prediction/submission.  It first records a
pre-registration containing the exact rankability gate and input hashes, then
uses the already-cached V18/V21 score matrices and clean-complement QRELs to
answer two separate questions:

* Are c35 rescue candidates rankable before a threshold is involved?
* Does the independently materialised complementary K16 -> K32 -> K64 lane
  create additional one-/two-swap oracle opportunity?

The expanded-lane metadata covers only 598 of the 1,197 clean-complement
queries.  The script keeps that scope explicit and never promotes it to a
whole-cohort result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


K = 5
SEED = 22091
BOOTSTRAP_SAMPLES = 2_000
REGIMES = ("standard_grouped", "law_family_ood", "query_template_ood", "near_duplicate_ood")
CONFIGS = ("B1024-D30", "B1024-D60", "B1536-D30", "B1536-D60", "B2304-D30", "B2304-D60")

# These values are deliberately fixed before any P0 result is read.  The
# gate uses the frozen six-pack mean only: it does not select a lucky LoRA or
# individual pack after labels have been inspected.  It is a branch decision,
# not a promotion rule.
PREREGISTRATION = {
    "schema_version": 1,
    "rankability_head": "e3b_mean_sixpacks",
    "rescue_recall_at_3_min": 0.20,
    "rescue_opportunity_queries_min": 10,
    "rescue_vs_neutral00_pr_auc_lift_bootstrap_lower_must_be_positive": True,
    "bootstrap_samples": BOOTSTRAP_SAMPLES,
    "bootstrap_seed": SEED,
    "bootstrap_unit": "query",
    "score_coverage_policy": "for_each_fixed_score_head_exclude_an_entire_query_when_any_c35_P0_challenger_score_is_missing;_for_E3B_require_all_six_configs;_never_filter_on_QREL",
    "branch_if_pass": "V22-A_candidate_incumbent_CAP_DBA_TIR_signed_utility",
    "branch_if_fail": "V22-B_K16_complementary_rescue_lane_audit_only",
    "holdout_opened": False,
    "training_allowed": False,
}


class AuditError(ValueError):
    """Raised when an artifact cannot safely support this audit."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"{label} must be a JSON object")
    return value


def _reject_unsafe_path(path: Path, label: str) -> None:
    normalized = str(path.resolve()).replace("\\", "/").casefold()
    if any(term in normalized for term in ("holdout", "public", "submission", "locked")):
        raise AuditError(f"V22-P0 is clean-complement only; refusing {label}: {path}")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to overwrite immutable V22 artifact: {path}")
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to overwrite immutable V22 artifact: {path}")
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _as_ids(value: Any, field: str, *, allow_empty: bool = False) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AuditError(f"{field} must be a list")
    result = [str(item) for item in value]
    if (not allow_empty and not result) or len(set(result)) != len(result) or any(not item for item in result):
        raise AuditError(f"{field} must contain unique non-empty document IDs")
    return result


def _validate_clean(payload: Mapping[str, Any], label: str) -> None:
    if payload.get("split") != "clean_complement_only":
        raise AuditError(f"{label} must declare clean_complement_only")
    if payload.get("holdout_opened") is not False or payload.get("public_labels_used") is not False:
        raise AuditError(f"{label} must attest no holdout/public-label access")


def _load_questions(path: Path, *, require_references: bool) -> dict[str, dict[str, Any]]:
    payload = _read_json(path, "metadata")
    _validate_clean(payload, "metadata")
    raw = payload.get("questions")
    if not isinstance(raw, list) or not raw:
        raise AuditError("metadata.questions must be non-empty")
    result: dict[str, dict[str, Any]] = {}
    for row in raw:
        if not isinstance(row, Mapping):
            raise AuditError("metadata question must be an object")
        qid = str(row.get("qid", ""))
        if not qid or qid in result:
            raise AuditError("metadata has empty or duplicate qid")
        anchor = _as_ids(row.get("anchor_answer"), f"{qid}.anchor_answer")
        if len(anchor) != K:
            raise AuditError(f"{qid}: V16b anchor must have exactly {K} documents")
        scoring_boundary = _as_ids(row.get("scoring_boundary"), f"{qid}.scoring_boundary")
        full_boundary = _as_ids(row.get("boundary"), f"{qid}.boundary")
        if scoring_boundary[0] != anchor[-1]:
            raise AuditError(f"{qid}: scoring boundary must start with incumbent slot 5")
        if full_boundary[:K] != anchor:
            raise AuditError(f"{qid}: full boundary must start with frozen V16b top-5")
        references = _as_ids(row.get("references"), f"{qid}.references") if require_references else []
        result[qid] = {
            "qid": qid,
            "anchor": anchor,
            "boundary": scoring_boundary,
            "candidate_boundary": full_boundary,
            "references": references,
            "fold": int(row.get("fold", -1)),
            "c35": _as_ids(row.get("candidates_c35"), f"{qid}.candidates_c35"),
        }
    return result


def _score_value(value: Any, field: str) -> float:
    values = value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else [value]
    try:
        numeric = [float(item) for item in values]
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{field} must contain numeric scores") from exc
    if not numeric or not all(math.isfinite(item) for item in numeric):
        raise AuditError(f"{field} must contain finite scores")
    return max(numeric)


def _load_nested_scores(path: Path, label: str, qids: set[str]) -> dict[tuple[str, str], float]:
    payload = _read_json(path, label)
    _validate_clean(payload, label)
    raw = payload.get("scores")
    if not isinstance(raw, Mapping):
        raise AuditError(f"{label}.scores is missing")
    result: dict[tuple[str, str], float] = {}
    for raw_qid, docs in raw.items():
        qid = str(raw_qid)
        if qid not in qids:
            continue
        if not isinstance(docs, Mapping):
            raise AuditError(f"{label}.scores[{qid}] must be an object")
        for raw_doc, value in docs.items():
            doc = str(raw_doc)
            result[(qid, doc)] = _score_value(value, f"{label}[{qid}][{doc}]")
    return result


def _load_e3b_scores(path: Path, qids: set[str]) -> dict[tuple[str, str, str], float]:
    result: dict[tuple[str, str, str], float] = {}
    try:
        lines = path.open(encoding="utf-8")
    except OSError as exc:
        raise AuditError(f"cannot read E3B score matrix: {path}") from exc
    with lines:
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditError(f"E3B score line {number} is invalid JSON") from exc
            if not isinstance(row, Mapping):
                raise AuditError(f"E3B score line {number} must be an object")
            qid, doc, config = str(row.get("qid", "")), str(row.get("doc_id", "")), str(row.get("config", ""))
            if qid not in qids:
                continue
            if config not in CONFIGS or not doc:
                raise AuditError(f"E3B score line {number} has invalid qid/doc/config")
            key = (qid, doc, config)
            if key in result:
                raise AuditError(f"duplicate E3B score for {key}")
            result[key] = _score_value(row.get("score"), f"E3B[{qid}][{doc}][{config}]")
    return result


def _load_regime_folds(path: Path, qids: set[str]) -> dict[str, dict[str, int]]:
    payload = _read_json(path, "four-regime fold manifest")
    _validate_clean(payload, "four-regime fold manifest")
    raw = payload.get("regimes")
    if not isinstance(raw, Mapping):
        raise AuditError("fold manifest lacks regimes")
    result: dict[str, dict[str, int]] = {}
    for regime in REGIMES:
        details = raw.get(regime)
        if not isinstance(details, Mapping) or not isinstance(details.get("folds"), list):
            raise AuditError(f"fold manifest lacks {regime}")
        assignment: dict[str, int] = {}
        for fold_row in details["folds"]:
            if not isinstance(fold_row, Mapping):
                raise AuditError(f"{regime}: invalid fold row")
            fold = int(fold_row.get("fold", -1))
            validation = fold_row.get("validation_qids")
            if not isinstance(validation, list):
                raise AuditError(f"{regime}/{fold}: validation_qids missing")
            for raw_qid in validation:
                qid = str(raw_qid)
                if qid in assignment:
                    raise AuditError(f"{regime}: qid {qid} appears in two validation folds")
                assignment[qid] = fold
        if set(assignment) != qids:
            raise AuditError(f"{regime}: validation fold qids do not match metadata")
        result[regime] = assignment
    return result


def _outcome(candidate: str, incumbent: str, gold: set[str]) -> str:
    candidate_relevant = candidate in gold
    incumbent_relevant = incumbent in gold
    if candidate_relevant and not incumbent_relevant:
        return "rescue"
    if incumbent_relevant and not candidate_relevant:
        return "destructive"
    if candidate_relevant:
        return "neutral_11"
    return "neutral_00"


def _average_precision(items: Sequence[tuple[float, int]]) -> float | None:
    positives = sum(label for _, label in items)
    if positives == 0:
        return None
    ordered = sorted(items, key=lambda pair: -pair[0])
    hit_count = 0
    total = 0.0
    for rank, (_, label) in enumerate(ordered, 1):
        if label:
            hit_count += 1
            total += hit_count / rank
    return total / positives


def _roc_auc(items: Sequence[tuple[float, int]]) -> float | None:
    positives = sum(label for _, label in items)
    negatives = len(items) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(items, key=lambda pair: pair[0])
    rank_sum = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        rank_sum += average_rank * sum(label for _, label in ordered[start:end])
        start = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _metric_for_rows(rows: Sequence[Mapping[str, Any]], *, comparison: str) -> dict[str, Any]:
    if comparison == "rescue_vs_neutral00":
        selected = [row for row in rows if row["outcome"] in {"rescue", "neutral_00"}]
        positive = "rescue"
    elif comparison == "rescue_vs_all_nonrescue":
        selected = list(rows)
        positive = "rescue"
    elif comparison == "destructive_vs_safe":
        selected = list(rows)
        positive = "destructive"
    else:
        raise AuditError(f"unknown comparison: {comparison}")
    items = [(float(row["score"]), int(row["outcome"] == positive)) for row in selected]
    prevalence = sum(label for _, label in items) / len(items) if items else None
    pr_auc = _average_precision(items) if items else None
    return {
        "comparison": comparison,
        "candidate_rows": len(items),
        "positive_rows": sum(label for _, label in items),
        "prevalence": prevalence,
        "pr_auc": pr_auc,
        "pr_auc_lift_vs_prevalence": pr_auc - prevalence if pr_auc is not None and prevalence is not None else None,
        "roc_auc": _roc_auc(items) if items else None,
    }


def _bootstrap_ap_lift(rows_by_qid: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    qids = sorted(rows_by_qid)
    rng = random.Random(SEED)
    values: list[float] = []
    skipped = 0
    for _ in range(BOOTSTRAP_SAMPLES):
        selected: list[tuple[float, int]] = []
        for _ in qids:
            qid = qids[rng.randrange(len(qids))]
            for row in rows_by_qid[qid]:
                if row["outcome"] in {"rescue", "neutral_00"}:
                    selected.append((float(row["score"]), int(row["outcome"] == "rescue")))
        if not selected:
            skipped += 1
            continue
        ap = _average_precision(selected)
        prevalence = sum(label for _, label in selected) / len(selected)
        if ap is None:
            skipped += 1
            continue
        values.append(ap - prevalence)
    if not values:
        return {"samples": BOOTSTRAP_SAMPLES, "valid_samples": 0, "lower_95": None, "upper_95": None}
    values.sort()
    lower = values[max(0, math.floor(0.025 * (len(values) - 1)))]
    upper = values[min(len(values) - 1, math.ceil(0.975 * (len(values) - 1)))]
    return {
        "samples": BOOTSTRAP_SAMPLES,
        "valid_samples": len(values),
        "skipped_samples": skipped,
        "lower_95": lower,
        "upper_95": upper,
        "mean": statistics.fmean(values),
        "unit": "query_resampling_with_all_its_candidate_rows",
    }


def _head_diagnostics(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    regime_folds: Mapping[str, Mapping[str, int]],
    *,
    bootstrap: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw_row in rows:
        row = dict(raw_row)
        by_qid[str(row["qid"])].append(row)
    query_rows: list[dict[str, Any]] = []
    for qid, candidates in by_qid.items():
        ordered = sorted(candidates, key=lambda row: (-float(row["score"]), str(row["doc_id"])))
        rescue = [index + 1 for index, row in enumerate(ordered) if row["outcome"] == "rescue"]
        neutral = [row for row in ordered if str(row["outcome"]).startswith("neutral_")]
        winner = ordered[0]
        query_rows.append({
            "qid": qid,
            "rescue_opportunity": bool(rescue),
            "best_rescue_rank": min(rescue) if rescue else None,
            "winner_outcome": winner["outcome"],
            "winner_doc_id": winner["doc_id"],
            "winner_score": winner["score"],
            "best_rescue_score": max((float(row["score"]) for row in ordered if row["outcome"] == "rescue"), default=None),
            "best_neutral_score": max((float(row["score"]) for row in neutral), default=None),
        })
    rescue_queries = [row for row in query_rows if row["rescue_opportunity"]]
    recall_at = {
        f"recall_at_{cutoff}": (
            sum(int(int(row["best_rescue_rank"]) <= cutoff) for row in rescue_queries) / len(rescue_queries)
            if rescue_queries else None
        )
        for cutoff in (1, 2, 3, 6)
    }
    margins = [
        float(row["best_rescue_score"]) - float(row["best_neutral_score"])
        for row in rescue_queries
        if row["best_neutral_score"] is not None and row["best_rescue_score"] is not None
    ]
    winner_outcomes = Counter(str(row["winner_outcome"]) for row in query_rows)
    candidate_outcomes = Counter(str(row["outcome"]) for row in rows)
    metrics = {
        comparison: _metric_for_rows(rows, comparison=comparison)
        for comparison in ("rescue_vs_neutral00", "rescue_vs_all_nonrescue", "destructive_vs_safe")
    }
    per_regime: dict[str, dict[str, Any]] = {}
    for regime in REGIMES:
        fold_rows: dict[str, dict[str, Any]] = {}
        for fold in sorted(set(regime_folds[regime].values())):
            qid_set = {qid for qid, assigned in regime_folds[regime].items() if assigned == fold}
            subset = [row for row in rows if str(row["qid"]) in qid_set]
            subset_rescue = [row for row in query_rows if row["qid"] in qid_set and row["rescue_opportunity"]]
            fold_rows[str(fold)] = {
                "queries_in_regime_fold": len(qid_set),
                "scored_queries": len({str(row["qid"]) for row in subset}),
                "rescue_opportunity_queries": len(subset_rescue),
                "rescue_recall_at_3": (
                    sum(int(int(row["best_rescue_rank"]) <= 3) for row in subset_rescue) / len(subset_rescue)
                    if subset_rescue else None
                ),
                "rescue_vs_neutral00": _metric_for_rows(subset, comparison="rescue_vs_neutral00"),
            }
        per_regime[regime] = fold_rows
    diagnostic = {
        "head": name,
        "queries": len(query_rows),
        "candidate_rows": len(rows),
        "candidate_outcome_counts": dict(sorted(candidate_outcomes.items())),
        "winner_outcome_counts_pre_threshold": dict(sorted(winner_outcomes.items())),
        "rescue_opportunity_queries": len(rescue_queries),
        "rescue_candidate_ranking": {
            **recall_at,
            "best_rescue_rank_median": statistics.median([int(row["best_rescue_rank"]) for row in rescue_queries]) if rescue_queries else None,
            "best_rescue_rank_mean": statistics.fmean([int(row["best_rescue_rank"]) for row in rescue_queries]) if rescue_queries else None,
            "rescue_minus_neutral_winner_margin_mean": statistics.fmean(margins) if margins else None,
            "rescue_minus_neutral_winner_margin_median": statistics.median(margins) if margins else None,
        },
        "classification": metrics,
        "per_regime_outer_fold": per_regime,
        "post_argmax_calibration": {
            "status": "not_evaluable_from_raw_score_heads",
            "reason": "P0 scores are ranking scores, not independently calibrated winner probabilities; no threshold is fit or tuned in P0.",
        },
    }
    if bootstrap:
        diagnostic["classification"]["rescue_vs_neutral00"]["query_bootstrap_pr_auc_lift"] = _bootstrap_ap_lift(by_qid)
    trace_rows = [{"head": name, **row} for row in query_rows]
    return diagnostic, trace_rows


def _recall(answer: Sequence[str], gold: set[str]) -> float:
    return len(set(answer[:K]) & gold) / len(gold)


def _oracle(prefix: Sequence[str], pool: Sequence[str], gold: set[str]) -> list[str]:
    fixed = [str(item) for item in prefix]
    unique_pool = [str(item) for item in pool if str(item) not in set(fixed)]
    if len(fixed) > K or len(set(fixed)) != len(fixed) or len(set(unique_pool)) != len(unique_pool):
        raise AuditError("invalid oracle prefix or candidate pool")
    return [*fixed, *(doc for doc in unique_pool if doc in gold), *(doc for doc in unique_pool if doc not in gold)][:K]


def _oracle_summary(questions: Mapping[str, Mapping[str, Any]], pools: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    values: dict[str, list[float]] = {"candidate_coverage_oracle": [], "o1_top4_plus_one": [], "o2_top3_plus_two": []}
    rescue_queries = 0
    rescue_candidates = 0
    for qid, question in questions.items():
        pool = [str(value) for value in pools[qid]]
        anchor = [str(value) for value in question["anchor"]]
        gold = set(str(value) for value in question["references"])
        if not set(anchor).issubset(set(pool)):
            raise AuditError(f"{qid}: candidate pool does not retain V16b anchor")
        values["candidate_coverage_oracle"].append(_recall(_oracle((), pool, gold), gold))
        values["o1_top4_plus_one"].append(_recall(_oracle(anchor[:4], pool, gold), gold))
        values["o2_top3_plus_two"].append(_recall(_oracle(anchor[:3], pool, gold), gold))
        if anchor[-1] not in gold:
            rescues = [doc for doc in pool if doc not in anchor[:4] and doc in gold]
            rescue_queries += int(bool(rescues))
            rescue_candidates += len(rescues)
    return {
        "queries": len(questions),
        "candidate_coverage_oracle": statistics.fmean(values["candidate_coverage_oracle"]),
        "o1_top4_plus_one": statistics.fmean(values["o1_top4_plus_one"]),
        "o2_top3_plus_two": statistics.fmean(values["o2_top3_plus_two"]),
        "rescue_opportunity_queries": rescue_queries,
        "rescue_candidate_incidences": rescue_candidates,
    }


def _expanded_oracles(
    full_questions: Mapping[str, Mapping[str, Any]],
    lane_metadata_path: Path,
    regime_folds: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    lane_questions = _load_questions(lane_metadata_path, require_references=False)
    if not set(lane_questions).issubset(full_questions):
        raise AuditError("expanded K metadata contains qids outside clean-complement metadata")
    joined: dict[str, dict[str, Any]] = {}
    for qid, lane in lane_questions.items():
        base = full_questions[qid]
        if lane["anchor"] != base["anchor"]:
            raise AuditError(f"{qid}: expanded lane anchor differs from frozen V16b")
        joined[qid] = {**lane, "references": base["references"]}
    # boundary = V16b top-five followed by a label-free lexical unseen lane.
    # We restore the frozen c35 to form the direct complementary union used by
    # the V20 K64 coverage oracle; no label is used for construction.
    c35_pools: dict[str, list[str]] = {}
    pools: dict[str, dict[str, list[str]]] = {"K16": {}, "K32": {}, "K64": {}}
    for qid, question in joined.items():
        c35 = list(question["c35"])
        lane = list(question["candidate_boundary"])
        if lane[:K] != question["anchor"] or len(lane) != K + 64:
            raise AuditError(f"{qid}: K64 metadata must contain V16b top-5 plus exactly 64 lane documents")
        unseen = lane[K:]
        if set(unseen) & set(c35):
            raise AuditError(f"{qid}: complementary lane overlaps c35")
        c35_pools[qid] = c35
        for label, size in (("K16", 16), ("K32", 32), ("K64", 64)):
            pools[label][qid] = [*c35, *unseen[:size]]
    results: dict[str, Any] = {"scope": {"queries": len(joined), "kind": "598-query materialised complementary-lane subset", "not_whole_clean_complement": True}}
    c35_result = _oracle_summary(joined, c35_pools)
    results["c35_same_subset"] = c35_result
    previous_pool = c35_pools
    previous_oracle = c35_result
    c35_rescue_qids = {
        qid for qid, question in joined.items()
        if question["anchor"][-1] not in set(question["references"])
        and any(doc in set(question["references"]) for doc in c35_pools[qid] if doc not in question["anchor"][:4])
    }
    for label in ("K16", "K32", "K64"):
        current = _oracle_summary(joined, pools[label])
        rescue_qids = {
            qid for qid, question in joined.items()
            if question["anchor"][-1] not in set(question["references"])
            and any(doc in set(question["references"]) for doc in pools[label][qid] if doc not in question["anchor"][:4])
        }
        added = {doc for qid in joined for doc in pools[label][qid] if doc not in previous_pool[qid]}
        result = {
            **current,
            "candidate_pool_size": 35 + int(label[1:]),
            "o1_gain_vs_previous_pool": current["o1_top4_plus_one"] - previous_oracle["o1_top4_plus_one"],
            "o1_gain_vs_c35_same_subset": current["o1_top4_plus_one"] - c35_result["o1_top4_plus_one"],
            "rescue_query_gain_vs_previous_pool": len(rescue_qids) - (len(c35_rescue_qids) if label == "K16" else results["K" + str(int(label[1:]) // 2)]["rescue_opportunity_queries"]),
            "rescue_query_gain_vs_c35_same_subset": len(rescue_qids) - len(c35_rescue_qids),
            "new_rescue_queries_vs_c35": len(rescue_qids - c35_rescue_qids),
            "rescue_overlap_with_c35_queries": len(rescue_qids & c35_rescue_qids),
            "marginal_unique_documents_from_previous_pool": len(added),
            "candidate_construction": "frozen_c35_plus_prefix_of_label_free_K64_lane",
        }
        per_regime: dict[str, Any] = {}
        for regime in REGIMES:
            folds: dict[str, Any] = {}
            for fold in sorted(set(regime_folds[regime][qid] for qid in joined)):
                subset_ids = {qid for qid in joined if regime_folds[regime][qid] == fold}
                subset_questions = {qid: joined[qid] for qid in subset_ids}
                subset_pools = {qid: pools[label][qid] for qid in subset_ids}
                folds[str(fold)] = _oracle_summary(subset_questions, subset_pools)
            per_regime[regime] = folds
        result["per_regime_outer_fold"] = per_regime
        results[label] = result
        previous_pool = pools[label]
        previous_oracle = current
    k16 = results["K16"]
    results["v22_b_k16_stop_before_training_gate"] = {
        "o1_increases_vs_c35": k16["o1_gain_vs_c35_same_subset"] > 0.0,
        "rescue_opportunity_increases_vs_c35": k16["rescue_query_gain_vs_c35_same_subset"] > 0,
        "passed": k16["o1_gain_vs_c35_same_subset"] > 0.0 and k16["rescue_query_gain_vs_c35_same_subset"] > 0,
        "action_if_pass": "K16_lane_training_may_be_pre_registered_after_P0; do_not_open_K32_yet",
        "action_if_fail": "stop_retrieval_branch_before_training_and_do_not_open_K32_or_K64",
    }
    return results


def _input_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
        "metadata": args.metadata,
        "current_scores": args.current_scores,
        "e3b_scores": args.e3b_scores,
        "fold_manifest": args.fold_manifest,
        "p3_standard": args.p3_root / "standard_grouped" / "adapted_current_passage_scores.json",
        "p3_law_family": args.p3_root / "law_family_ood" / "adapted_current_passage_scores.json",
        "p3_query_template": args.p3_root / "query_template_ood" / "adapted_current_passage_scores.json",
        "p3_near_duplicate": args.p3_root / "near_duplicate_ood" / "adapted_current_passage_scores.json",
        "k64_metadata": args.k64_metadata,
    }
    for name, path in paths.items():
        _reject_unsafe_path(path, name)
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")
    return paths


def _pre_registration_payload(args: argparse.Namespace) -> dict[str, Any]:
    paths = _input_paths(args)
    script = Path(__file__).resolve()
    return {
        "kind": "legalir_v22_p0_pre_registration",
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "cpu_only": True,
        "selector_trained": False,
        "config": PREREGISTRATION,
        "script": {"path": str(script), "sha256": _sha256(script)},
        "inputs": {name: {"path": str(path.resolve()), "sha256": _sha256(path)} for name, path in paths.items()},
    }


def _validate_pre_registration(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    payload = _read_json(path, "V22-P0 pre-registration")
    if payload.get("kind") != "legalir_v22_p0_pre_registration":
        raise AuditError("wrong V22-P0 pre-registration kind")
    expected = _pre_registration_payload(args)
    for key in ("split", "holdout_opened", "public_labels_used", "cpu_only", "selector_trained", "config", "script", "inputs"):
        if payload.get(key) != expected.get(key):
            raise AuditError(f"pre-registration drift in {key}; refuse post-hoc P0 execution")
    return payload


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    pre_registration = _validate_pre_registration(args.pre_registration, args)
    paths = _input_paths(args)
    questions = _load_questions(paths["metadata"], require_references=True)
    qids = set(questions)
    regime_folds = _load_regime_folds(paths["fold_manifest"], qids)
    current_scores = _load_nested_scores(paths["current_scores"], "frozen current CE scores", qids)
    e3b_scores = _load_e3b_scores(paths["e3b_scores"], qids)
    p3_scores = {
        "p3_lora_standard_grouped": _load_nested_scores(paths["p3_standard"], "P3 standard scores", qids),
        "p3_lora_law_family_ood": _load_nested_scores(paths["p3_law_family"], "P3 law-family scores", qids),
        "p3_lora_query_template_ood": _load_nested_scores(paths["p3_query_template"], "P3 query-template scores", qids),
        "p3_lora_near_duplicate_ood": _load_nested_scores(paths["p3_near_duplicate"], "P3 near-duplicate scores", qids),
    }
    rows_by_head: dict[str, list[dict[str, Any]]] = {"frozen_ce_max_passage": [], "e3b_mean_sixpacks": []}
    for config in CONFIGS:
        rows_by_head[f"e3b_{config}"] = []
    for name in p3_scores:
        rows_by_head[name] = []
    label_rows: list[dict[str, Any]] = []
    complete_qids_by_head: dict[str, set[str]] = {name: set() for name in rows_by_head}
    for qid, question in questions.items():
        incumbent = str(question["anchor"][-1])
        gold = set(str(item) for item in question["references"])
        challengers = list(question["boundary"][1:])
        if len(challengers) != 6:
            raise AuditError(f"{qid}: c35 P0 expects exactly six rank-6..12 challengers")
        complete_qids_by_head["frozen_ce_max_passage"].update(
            {qid} if all((qid, doc) in current_scores for doc in challengers) else set()
        )
        e3b_complete = all((qid, doc, config) in e3b_scores for doc in challengers for config in CONFIGS)
        if e3b_complete:
            complete_qids_by_head["e3b_mean_sixpacks"].add(qid)
            for config in CONFIGS:
                complete_qids_by_head[f"e3b_{config}"].add(qid)
        for name, score_map in p3_scores.items():
            if all((qid, doc) in score_map for doc in challengers):
                complete_qids_by_head[name].add(qid)
        for offset, doc in enumerate(challengers, 6):
            outcome = _outcome(doc, incumbent, gold)
            row = {"qid": qid, "doc_id": doc, "candidate_rank": offset, "outcome": outcome}
            label_rows.append(row)
            if qid in complete_qids_by_head["frozen_ce_max_passage"]:
                rows_by_head["frozen_ce_max_passage"].append({**row, "score": current_scores[(qid, doc)]})
            if e3b_complete:
                config_scores = [e3b_scores[(qid, doc, config)] for config in CONFIGS]
                rows_by_head["e3b_mean_sixpacks"].append({**row, "score": statistics.fmean(config_scores)})
                for config, score in zip(CONFIGS, config_scores, strict=True):
                    rows_by_head[f"e3b_{config}"].append({**row, "score": score})
            for name, score_map in p3_scores.items():
                if qid in complete_qids_by_head[name]:
                    rows_by_head[name].append({**row, "score": score_map[(qid, doc)]})
    head_reports: dict[str, Any] = {}
    trace_rows: list[dict[str, Any]] = []
    for name, rows in rows_by_head.items():
        diagnostic, head_trace = _head_diagnostics(name, rows, regime_folds, bootstrap=name == PREREGISTRATION["rankability_head"])
        head_reports[name] = diagnostic
        trace_rows.extend(head_trace)
    registered = head_reports[PREREGISTRATION["rankability_head"]]
    registered_bootstrap = registered["classification"]["rescue_vs_neutral00"].get("query_bootstrap_pr_auc_lift", {})
    rankability_gate = {
        "head": PREREGISTRATION["rankability_head"],
        "rescue_opportunity_queries": registered["rescue_opportunity_queries"],
        "rescue_recall_at_3": registered["rescue_candidate_ranking"]["recall_at_3"],
        "pr_auc_lift_bootstrap_lower_95": registered_bootstrap.get("lower_95"),
        "conditions": {
            "opportunity_queries_at_least_min": registered["rescue_opportunity_queries"] >= PREREGISTRATION["rescue_opportunity_queries_min"],
            "rescue_recall_at_3_at_least_min": (
                registered["rescue_candidate_ranking"]["recall_at_3"] is not None
                and registered["rescue_candidate_ranking"]["recall_at_3"] >= PREREGISTRATION["rescue_recall_at_3_min"]
            ),
            "pr_auc_lift_bootstrap_lower_positive": (
                registered_bootstrap.get("lower_95") is not None and registered_bootstrap["lower_95"] > 0.0
            ),
        },
    }
    rankability_gate["passed"] = all(rankability_gate["conditions"].values())
    expanded = _expanded_oracles(questions, paths["k64_metadata"], regime_folds)
    branch = (
        PREREGISTRATION["branch_if_pass"]
        if rankability_gate["passed"]
        else PREREGISTRATION["branch_if_fail"]
    )
    report = {
        "kind": "legalir_v22_p0_rescue_neutral_closure_audit",
        "schema_version": 1,
        "stage": "V22-P0_complete_cpu_only_no_training_no_holdout",
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "cpu_only": True,
        "gpu_used": False,
        "model_loaded": False,
        "selector_trained": False,
        "selection_performed": False,
        "pre_registration": {"path": str(args.pre_registration.resolve()), "sha256": _sha256(args.pre_registration), "config": pre_registration["config"]},
        "scope": {
            "c35_queries": len(questions),
            "c35_challengers_per_query": 6,
            "score_complete_queries_by_head": {name: len(qid_set) for name, qid_set in sorted(complete_qids_by_head.items())},
            "score_excluded_queries_label_free_by_head": {name: len(questions) - len(qid_set) for name, qid_set in sorted(complete_qids_by_head.items())},
            "score_exclusion_policy": PREREGISTRATION["score_coverage_policy"],
            "expanded_oracle_queries": expanded["scope"]["queries"],
            "expanded_oracle_scope_warning": "K16/K32/K64 are a fixed 598-query candidate-lane subset; no whole-cohort claim is valid.",
        },
        "neutral_outcome_definition": {
            "rescue": "challenger relevant and V16b slot-5 incumbent non-relevant",
            "destructive": "challenger non-relevant and V16b slot-5 incumbent relevant",
            "neutral_00": "both challenger and incumbent non-relevant",
            "neutral_11": "both challenger and incumbent relevant",
            "label_rows": len(label_rows),
            "counts": dict(sorted(Counter(str(row["outcome"]) for row in label_rows).items())),
            "unit": "candidate--incumbent comparison; all bootstrap resampling remains at query level",
        },
        "c35_pre_threshold_rankability": head_reports,
        "registered_c35_rankability_gate": rankability_gate,
        "expanded_candidate_oracles": expanded,
        "decision": {
            "next_branch": branch,
            "v21_closed": True,
            "v21_c_locked": True,
            "holdout_may_open": False,
            "if_v22_a": "Pre-register one CAP-DBA/TIR signed-utility arm using query + incumbent slot-5 + incumbent evidence; no LoRA.",
            "if_v22_b": "Only the K16 stop-before-training gate may authorize a separately pre-registered complementary rescue-lane training arm. K32/K64 stay closed until K16 policy OOF passes.",
        },
        "provenance": {"inputs": pre_registration["inputs"], "script": pre_registration["script"]},
    }
    report_path = args.output_dir / "v22_p0_report.json"
    trace_path = args.output_dir / "v22_p0_pre_threshold_winner_traces.jsonl"
    if report_path.exists() or trace_path.exists():
        raise FileExistsError("V22-P0 output already exists; refusing to replace audit evidence")
    _atomic_json(report_path, report)
    _atomic_jsonl(trace_path, trace_rows)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--current-scores", type=Path, required=True)
    parser.add_argument("--e3b-scores", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--p3-root", type=Path, required=True)
    parser.add_argument("--k64-metadata", type=Path, required=True)
    parser.add_argument("--pre-registration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pre-register-only", action="store_true")
    args = parser.parse_args()
    if not args.pre_register_only and args.output_dir is None:
        parser.error("--output-dir is required for the audit run")
    return args


def main() -> int:
    args = parse_args()
    _reject_unsafe_path(args.pre_registration, "pre-registration")
    if args.pre_register_only:
        if args.pre_registration.exists():
            raise FileExistsError(f"pre-registration already exists: {args.pre_registration}")
        payload = _pre_registration_payload(args)
        _atomic_json(args.pre_registration, payload)
        print(json.dumps({"pre_registered": str(args.pre_registration.resolve()), "config": PREREGISTRATION}, ensure_ascii=False, sort_keys=True))
        return 0
    assert args.output_dir is not None
    _reject_unsafe_path(args.output_dir, "output directory")
    report = run_audit(args)
    print(json.dumps({
        "output": str((args.output_dir / "v22_p0_report.json").resolve()),
        "next_branch": report["decision"]["next_branch"],
        "rankable": report["registered_c35_rankability_gate"]["passed"],
        "holdout_opened": False,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
