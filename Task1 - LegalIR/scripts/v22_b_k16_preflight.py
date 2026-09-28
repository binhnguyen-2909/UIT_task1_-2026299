"""V22-B K16 complementary-rescue preflight.

This is a registered, clean-complement-only gate.  It does not train a model,
load a cross-encoder, inspect holdout data, or choose a submission.  It proves
that the existing K16 E3B score matrix is complete, pool-aligned with K64, and
contains enough rankable rescue evidence to justify exactly one later K16-only
multi-positive arm.  K32 and K64 policy training remain explicitly closed.
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


CONFIGS = ("B1024-D30", "B1024-D60", "B1536-D30", "B1536-D60", "B2304-D30", "B2304-D60")
REGIMES = ("standard_grouped", "law_family_ood", "query_template_ood", "near_duplicate_ood")
SEED = 22118
BOOTSTRAP_SAMPLES = 2_000

PREREGISTRATION = {
    "schema_version": 1,
    "stage": "V22-B0_K16_complementary_rescue_preflight",
    "candidate_lane": "immutable label-free K16 prefix of registered K64 lane; V16b top-four and slot-five incumbent retained",
    "evidence": "existing query-conditioned six-pack E3B scores only; manifest and score matrix must be a one-to-one key set",
    "ranker_shortlist": "rank all 16 K16 challengers only by mean of the six registered E3B pack scores, then retain exactly top four for a possible later lane",
    "no_model_training": True,
    "no_ce_loaded": True,
    "no_holdout": True,
    "no_public_labels": True,
    "closed_lanes": ["K32", "K64", "slot4", "V22-A retuning", "CE fine-tuning"],
    "gate": {
        "p0_k16_o1_gain_positive": True,
        "p0_k16_new_rescue_queries_positive": True,
        "e3b_rescue_recall_at_4_min": 0.75,
        "e3b_rescue_pr_auc_lift_bootstrap_lower_positive": True,
        "minimum_selected_rescue_rows_in_every_outer_training_partition": 5,
        "evidence_key_bijection": True,
        "k16_is_exact_k64_prefix": True,
    },
}


class PreflightError(ValueError):
    """Raised when the registered K16 lane contract does not hold."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_unsafe(path: Path, label: str) -> None:
    normalized = str(path.resolve()).replace("\\", "/").casefold()
    if any(term in normalized for term in ("holdout", "public", "submission", "locked")):
        raise PreflightError(f"V22-B0 is clean-complement only; refusing {label}: {path}")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"{label} must be a JSON object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to overwrite immutable V22-B0 artifact: {path}")
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
        raise PreflightError(f"cannot import V22-P0 helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _input_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
        "v22_a_report": args.v22_a_report,
        "p0_report": args.p0_report,
        "p0_script": args.p0_script,
        "base_metadata": args.base_metadata,
        "k16_metadata": args.k16_metadata,
        "k64_metadata": args.k64_metadata,
        "k16_manifest": args.k16_manifest,
        "k16_scores": args.k16_scores,
        "fold_manifest": args.fold_manifest,
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
        "kind": "legalir_v22_b0_k16_pre_registration",
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "config": PREREGISTRATION,
        "script": {"path": str(script), "sha256": _sha256(script)},
        "inputs": {name: {"path": str(path.resolve()), "sha256": _sha256(path)} for name, path in paths.items()},
    }


def _validate_pre_registration(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    observed = _read_json(path, "V22-B0 pre-registration")
    expected = _pre_registration(args)
    if observed.get("kind") != expected["kind"]:
        raise PreflightError("wrong V22-B0 pre-registration kind")
    for field in ("split", "holdout_opened", "public_labels_used", "config", "script", "inputs"):
        if observed.get(field) != expected.get(field):
            raise PreflightError(f"pre-registration drift in {field}; refuse post-hoc K16 evaluation")
    return observed


def _as_ids(value: Any, field: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PreflightError(f"{field} must be a document-ID list")
    result = [str(item) for item in value]
    if not result or len(result) != len(set(result)) or any(not item for item in result):
        raise PreflightError(f"{field} must have unique non-empty document IDs")
    return result


def _load_expanded_questions(path: Path, label: str) -> dict[str, dict[str, Any]]:
    payload = _read_json(path, label)
    if payload.get("split") != "clean_complement_only" or payload.get("holdout_opened") is not False or payload.get("public_labels_used") is not False:
        raise PreflightError(f"{label} must attest clean-complement-only construction")
    rows = payload.get("questions")
    if not isinstance(rows, list) or not rows:
        raise PreflightError(f"{label}.questions must be a non-empty list")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise PreflightError(f"{label}.questions must contain objects")
        qid = str(row.get("qid", ""))
        if not qid or qid in result:
            raise PreflightError(f"{label} has an empty or duplicate qid")
        anchor = _as_ids(row.get("anchor_answer"), f"{label}.{qid}.anchor_answer")
        boundary = _as_ids(row.get("boundary"), f"{label}.{qid}.boundary")
        scoring = _as_ids(row.get("scoring_boundary"), f"{label}.{qid}.scoring_boundary")
        e3b = _as_ids(row.get("e3b_scoring_boundary"), f"{label}.{qid}.e3b_scoring_boundary")
        unavailable = [str(item) for item in row.get("e3b_source_unavailable", [])]
        if len(anchor) != 5 or boundary[:5] != anchor or scoring[0] != anchor[-1]:
            raise PreflightError(f"{label}.{qid}: frozen V16b boundary contract violated")
        if len(unavailable) != len(set(unavailable)) or any(doc not in scoring for doc in unavailable):
            raise PreflightError(f"{label}.{qid}: invalid source-unavailable declaration")
        if e3b != [doc for doc in scoring if doc not in set(unavailable)]:
            raise PreflightError(f"{label}.{qid}: E3B boundary disagrees with source-unavailable declaration")
        result[qid] = {"anchor": anchor, "boundary": boundary, "scoring": scoring, "e3b_scoring": e3b, "source_unavailable": unavailable}
    return result


def _load_e3b(path: Path) -> dict[tuple[str, str, str], float]:
    scores: dict[tuple[str, str, str], float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
                key = (str(row["qid"]), str(row["doc_id"]), str(row["config"]))
                value = float(row["score"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise PreflightError(f"invalid K16 E3B score at line {line_number}") from exc
            if key in scores or key[2] not in CONFIGS or not math.isfinite(value):
                raise PreflightError(f"invalid or duplicate K16 E3B score key at line {line_number}")
            scores[key] = value
    return scores


def _load_manifest_keys(path: Path) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
                key = (str(row["qid"]), str(row["doc_id"]), str(row["config"]))
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise PreflightError(f"invalid K16 manifest row at line {line_number}") from exc
            if key in keys or key[2] not in CONFIGS:
                raise PreflightError(f"invalid or duplicate K16 manifest key at line {line_number}")
            keys.add(key)
    return keys


def _average_precision(items: Sequence[tuple[float, int]]) -> float | None:
    ordered = sorted(items, key=lambda item: (-item[0],))
    positives = sum(label for _, label in ordered)
    if positives == 0:
        return None
    hits = 0
    total = 0.0
    for index, (_, label) in enumerate(ordered, 1):
        if label:
            hits += 1
            total += hits / index
    return total / positives


def _bootstrap_pr_lift(rows_by_qid: Mapping[str, Sequence[tuple[float, int]]]) -> dict[str, Any]:
    qids = sorted(rows_by_qid)
    generator = random.Random(SEED)
    values: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        selected = [qids[generator.randrange(len(qids))] for _ in qids]
        rows = [item for qid in selected for item in rows_by_qid[qid]]
        if not rows:
            continue
        prevalence = sum(label for _, label in rows) / len(rows)
        ap = _average_precision(rows)
        if ap is not None:
            values.append(ap - prevalence)
    values.sort()
    if not values:
        return {"samples": 0, "lower_95": None, "mean": None, "upper_95": None}
    return {
        "samples": len(values),
        "lower_95": values[math.floor(0.025 * (len(values) - 1))],
        "mean": statistics.fmean(values),
        "upper_95": values[math.ceil(0.975 * (len(values) - 1))],
    }


def _outcome(candidate: str, incumbent: str, gold: set[str]) -> str:
    if candidate in gold and incumbent not in gold:
        return "rescue"
    if candidate not in gold and incumbent in gold:
        return "destructive"
    return "neutral"


def _recall(answer: Sequence[str], gold: set[str]) -> float:
    return len(set(answer) & gold) / len(gold)


def _assert_prior_stage(p0_report: Mapping[str, Any], v22a_report: Mapping[str, Any]) -> dict[str, Any]:
    if p0_report.get("kind") != "legalir_v22_p0_rescue_neutral_closure_audit":
        raise PreflightError("V22-B0 requires the V22-P0 closure report")
    k16 = p0_report.get("expanded_candidate_oracles", {}).get("K16")
    if not isinstance(k16, Mapping):
        raise PreflightError("V22-P0 has no registered K16 oracle")
    if not (float(k16.get("o1_gain_vs_c35_same_subset", 0.0)) > 0.0 and int(k16.get("new_rescue_queries_vs_c35", 0)) > 0):
        raise PreflightError("K16 stop-before-training gate did not show added rescue opportunity")
    if v22a_report.get("kind") != "legalir_v22_a_cap_dba_tir_signed_utility":
        raise PreflightError("V22-B0 requires the completed V22-A report")
    if v22a_report.get("promotion", {}).get("promoted") is not False:
        raise PreflightError("V22-B0 is only reached after V22-A fails its unchanged promotion contract")
    return {"k16_o1_gain_vs_c35_same_subset": float(k16["o1_gain_vs_c35_same_subset"]), "k16_new_rescue_queries_vs_c35": int(k16["new_rescue_queries_vs_c35"]), "k16_p0_queries": int(k16["queries"])}


def run(args: argparse.Namespace) -> dict[str, Any]:
    pre = _validate_pre_registration(args.pre_registration, args)
    paths = _input_paths(args)
    p0_report = _read_json(paths["p0_report"], "V22-P0 report")
    v22a_report = _read_json(paths["v22_a_report"], "V22-A report")
    prior = _assert_prior_stage(p0_report, v22a_report)
    p0 = _load_p0_module(paths["p0_script"])
    base_questions = p0._load_questions(paths["base_metadata"], require_references=True)
    k16 = _load_expanded_questions(paths["k16_metadata"], "K16 metadata")
    k64 = _load_expanded_questions(paths["k64_metadata"], "K64 metadata")
    if len(k16) != prior["k16_p0_queries"]:
        raise PreflightError("K16 metadata query count differs from V22-P0 registered scope")
    if set(k16) - set(base_questions) or set(k16) - set(k64):
        raise PreflightError("K16 contains a query outside the clean-complement base/K64 scope")
    for qid, row in k16.items():
        if row["anchor"] != base_questions[qid]["anchor"]:
            raise PreflightError(f"{qid}: K16 anchor differs from frozen V16b anchor")
        full = k64[qid]
        if row["boundary"] != full["boundary"][:len(row["boundary"])] or row["scoring"] != full["scoring"][:len(row["scoring"])]:
            raise PreflightError(f"{qid}: K16 is not an exact prefix of the registered K64 lane")
        if row["source_unavailable"] or row["e3b_scoring"] != row["scoring"] or len(row["scoring"]) != 17:
            raise PreflightError(f"{qid}: K16 must be incumbent plus exactly 16 challengers")
    manifest_keys = _load_manifest_keys(paths["k16_manifest"])
    scores = _load_e3b(paths["k16_scores"])
    expected_keys = {(qid, doc, config) for qid, row in k16.items() for doc in row["scoring"] for config in CONFIGS}
    if manifest_keys != expected_keys or set(scores) != expected_keys:
        raise PreflightError("K16 E3B manifest/score matrix is not exactly the registered complete candidate boundary")
    folds = p0._load_regime_folds(paths["fold_manifest"], set(base_questions))
    rows_by_qid: dict[str, list[dict[str, Any]]] = {}
    all_pr_rows: dict[str, list[tuple[float, int]]] = {}
    all_outcomes: Counter[str] = Counter()
    rescue_best_ranks: list[int] = []
    baseline_values: list[float] = []
    oracle_values: list[float] = []
    selected_by_regime: dict[str, dict[int, Counter[str]]] = {regime: defaultdict(Counter) for regime in REGIMES}
    for qid in sorted(k16, key=lambda value: (int(value) if value.isdigit() else value)):
        row = k16[qid]
        incumbent = row["anchor"][-1]
        gold = set(base_questions[qid]["references"])
        candidates = row["scoring"][1:]
        candidate_rows = []
        for source_rank, doc in enumerate(candidates, 1):
            six_scores = [scores[(qid, doc, config)] for config in CONFIGS]
            candidate_rows.append({"doc_id": doc, "source_rank": source_rank, "e3b_mean": statistics.fmean(six_scores), "outcome": _outcome(doc, incumbent, gold)})
        ranked = sorted(candidate_rows, key=lambda item: (-item["e3b_mean"], item["source_rank"], item["doc_id"]))
        for rank, item in enumerate(ranked, 1):
            item["e3b_rank"] = rank
        rescue_ranks = [item["e3b_rank"] for item in ranked if item["outcome"] == "rescue"]
        if rescue_ranks:
            rescue_best_ranks.append(min(rescue_ranks))
        all_outcomes.update(item["outcome"] for item in candidate_rows)
        all_pr_rows[qid] = [(item["e3b_mean"], int(item["outcome"] == "rescue")) for item in candidate_rows if item["outcome"] in {"rescue", "neutral"}]
        selected = ranked[:4]
        for regime in REGIMES:
            selected_by_regime[regime][folds[regime][qid]].update(item["outcome"] for item in selected)
        baseline_values.append(_recall(row["anchor"], gold))
        rescue_docs = [item["doc_id"] for item in candidate_rows if item["outcome"] == "rescue"]
        oracle_values.append(_recall([*row["anchor"][:4], rescue_docs[0]] if rescue_docs else row["anchor"], gold))
        rows_by_qid[qid] = ranked
    rescue_queries = len(rescue_best_ranks)
    rankability = {f"rescue_recall_at_{cutoff}": sum(rank <= cutoff for rank in rescue_best_ranks) / rescue_queries if rescue_queries else 0.0 for cutoff in (1, 2, 3, 4, 6, 16)}
    pr_items = [item for rows in all_pr_rows.values() for item in rows]
    prevalence = sum(label for _, label in pr_items) / len(pr_items)
    pr_auc = _average_precision(pr_items)
    pr_bootstrap = _bootstrap_pr_lift(all_pr_rows)
    support: dict[str, dict[str, Any]] = {}
    support_ok = True
    for regime in REGIMES:
        total_rescues = sum(counter["rescue"] for counter in selected_by_regime[regime].values())
        by_fold = {}
        for fold in range(5):
            validation = selected_by_regime[regime][fold]
            train_rescues = total_rescues - validation["rescue"]
            by_fold[str(fold)] = {"validation": dict(validation), "training_rescue_rows": train_rescues}
            support_ok = support_ok and train_rescues >= PREREGISTRATION["gate"]["minimum_selected_rescue_rows_in_every_outer_training_partition"]
        support[regime] = by_fold
    o1_gain = statistics.fmean(oracle_values) - statistics.fmean(baseline_values)
    checks = {
        "p0_k16_o1_gain_positive": prior["k16_o1_gain_vs_c35_same_subset"] > 0.0,
        "p0_k16_new_rescue_queries_positive": prior["k16_new_rescue_queries_vs_c35"] > 0,
        "e3b_rescue_recall_at_4": rankability["rescue_recall_at_4"] >= PREREGISTRATION["gate"]["e3b_rescue_recall_at_4_min"],
        "e3b_rescue_pr_auc_lift_bootstrap_lower_positive": pr_bootstrap["lower_95"] is not None and pr_bootstrap["lower_95"] > 0.0,
        "minimum_training_rescue_support": support_ok,
        "evidence_key_bijection": True,
        "k16_is_exact_k64_prefix": True,
        "k32_k64_policy_training_closed": True,
        "holdout_closed": True,
    }
    report = {
        "kind": "legalir_v22_b0_k16_complementary_rescue_preflight",
        "schema_version": 1,
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "ce_loaded": False,
        "model_trained": False,
        "pre_registration": {"path": str(args.pre_registration.resolve()), "sha256": _sha256(args.pre_registration), "config": pre["config"]},
        "inputs": pre["inputs"],
        "prior_stage": prior,
        "e3b_integrity": {"manifest_keys": len(manifest_keys), "score_keys": len(scores), "expected_keys": len(expected_keys), "key_bijection": True, "configs": list(CONFIGS)},
        "scope": {"queries": len(k16), "challengers_per_query": 16, "inference_shortlist_per_query": 4, "k32_k64_policy_training": "closed"},
        "outcomes_over_full_k16": dict(all_outcomes),
        "e3b_rankability": {**rankability, "rescue_opportunity_queries": rescue_queries, "best_rescue_rank_histogram": dict(sorted(Counter(rescue_best_ranks).items())), "rescue_vs_neutral_pr_auc": pr_auc, "rescue_vs_neutral_prevalence": prevalence, "pr_auc_lift_vs_prevalence": pr_auc - prevalence if pr_auc is not None else None, "query_bootstrap_pr_auc_lift": pr_bootstrap},
        "o1_oracle_vs_v16b_on_k16_subset": {"baseline_recall": statistics.fmean(baseline_values), "o1_recall": statistics.fmean(oracle_values), "gain": o1_gain},
        "outer_crossfit_top4_support": support,
        "registered_k16_training_gate": {"checks": checks, "passed": all(checks.values())},
        "decision": {"holdout_may_open": False, "if_passed": "pre-register exactly one K16-only four-regime cross-fitted multi-positive rescue lane; select only the fixed E3B top-four shortlist and retain V16b unless the unchanged promotion contract passes", "if_failed": "close K16 without tuning; do not open K32/K64 or holdout"},
    }
    report_path = args.output_dir / "v22_b0_k16_preflight_report.json"
    if report_path.exists():
        raise FileExistsError(f"V22-B0 output already exists: {report_path}")
    _atomic_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v22-a-report", type=Path, required=True)
    parser.add_argument("--p0-report", type=Path, required=True)
    parser.add_argument("--p0-script", type=Path, required=True)
    parser.add_argument("--base-metadata", type=Path, required=True)
    parser.add_argument("--k16-metadata", type=Path, required=True)
    parser.add_argument("--k64-metadata", type=Path, required=True)
    parser.add_argument("--k16-manifest", type=Path, required=True)
    parser.add_argument("--k16-scores", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--pre-registration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pre-register-only", action="store_true")
    args = parser.parse_args()
    if not args.pre_register_only and args.output_dir is None:
        parser.error("--output-dir is required for V22-B0 execution")
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
    print(json.dumps({"output": str((args.output_dir / "v22_b0_k16_preflight_report.json").resolve()), "passed": report["registered_k16_training_gate"]["passed"], "holdout_opened": False}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
