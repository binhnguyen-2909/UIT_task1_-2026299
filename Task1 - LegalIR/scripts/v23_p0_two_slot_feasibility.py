"""V23-P0: clean-complement two-slot feasibility and evidence-parity audit.

The audit keeps V16b slots 1--3 byte-identical and computes upper bounds for
one- and two-document replacement at slots 4--5.  It does not train, load a
model, open holdout, or create a submission.  The E3B-aligned shortlist is the
only candidate family that can authorize V23-A because its evidence exists in
the registered E3B packs; c35-only documents are reported as diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
import statistics
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIGS = ("B1024-D30", "B1024-D60", "B1536-D30", "B1536-D60", "B2304-D30", "B2304-D60")
REGIMES = ("standard_grouped", "law_family_ood", "query_template_ood", "near_duplicate_ood")
K = 5

PREREGISTRATION = {
    "schema_version": 1,
    "stage": "V23-P0_C35_two_slot_feasibility",
    "scope": "clean_complement only; V16b slots 1..3 frozen byte-identically; only slots 4..5 may change",
    "candidate_pools": {
        "full_c35": "all c35 candidates outside frozen V16b top-three and current slot-four/slot-five identities",
        "c35_rank_top6": "first six such c35 candidates in the registered c35 order",
        "c35_rank_top8": "first eight such c35 candidates in the registered c35 order",
        "e3b_aligned_shortlist6": "the six registered c35 challengers in scoring_boundary for which E3B evidence is available; source-unavailable queries keep only available evidence and force-abstain later",
    },
    "oracle": "enumerate zero, one, and two distinct replacement documents assigned to slots 4 and 5; maximize Recall@5, then prefer fewer changes; never replace a top-three document",
    "gate": {"o2_gain_min": 0.020, "beneficial_query_equivalents_min": 24, "top_three_byte_identical": True, "e3b_fallback_outside_declared_six": False},
    "holdout_opened": False,
    "public_labels_used": False,
    "training_allowed": False,
}


class V23P0Error(ValueError):
    """Raised when the immutable V23-P0 audit contract is violated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_unsafe(path: Path, label: str) -> None:
    normalized = str(path.resolve()).replace("\\", "/").casefold()
    if any(term in normalized for term in ("holdout", "public", "submission", "locked")):
        raise V23P0Error(f"V23-P0 is clean-complement-only; refusing {label}: {path}")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V23P0Error(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise V23P0Error(f"{label} must be a JSON object")
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
            raise FileExistsError(f"refusing to overwrite immutable V23-P0 artifact: {path}")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise V23P0Error(f"cannot import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _input_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {"v22_b_report": args.v22_b_report, "p0_script": args.p0_script, "metadata": args.metadata, "e3b_manifest": args.e3b_manifest, "e3b_scores": args.e3b_scores, "fold_manifest": args.fold_manifest}
    for name, path in paths.items():
        _reject_unsafe(path, name)
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")
    return paths


def _pre_registration(args: argparse.Namespace) -> dict[str, Any]:
    paths = _input_paths(args)
    script = Path(__file__).resolve()
    return {"kind": "legalir_v23_p0_two_slot_pre_registration", "split": "clean_complement_only", "holdout_opened": False, "public_labels_used": False, "config": PREREGISTRATION, "script": {"path": str(script), "sha256": _sha256(script)}, "inputs": {name: {"path": str(path.resolve()), "sha256": _sha256(path)} for name, path in paths.items()}}


def _validate_pre_registration(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    observed = _read_json(path, "V23-P0 pre-registration")
    expected = _pre_registration(args)
    if observed.get("kind") != expected["kind"]:
        raise V23P0Error("wrong V23-P0 pre-registration kind")
    for field in ("split", "holdout_opened", "public_labels_used", "config", "script", "inputs"):
        if observed.get(field) != expected.get(field):
            raise V23P0Error(f"pre-registration drift in {field}; refuse V23-P0 evaluation")
    return observed


def _ids(value: Any, field: str, *, allow_empty: bool = False) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise V23P0Error(f"{field} must be a document-ID list")
    result = [str(item) for item in value]
    if (not allow_empty and not result) or len(set(result)) != len(result) or any(not item for item in result):
        raise V23P0Error(f"{field} must contain unique non-empty IDs")
    return result


def _load_questions(path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(path, "V23 metadata")
    if payload.get("split") != "clean_complement_only" or payload.get("holdout_opened") is not False or payload.get("public_labels_used") is not False:
        raise V23P0Error("V23 metadata is not clean-complement-only")
    rows = payload.get("questions")
    if not isinstance(rows, list) or not rows:
        raise V23P0Error("metadata.questions must be non-empty")
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise V23P0Error("metadata question must be an object")
        qid = str(raw.get("qid", ""))
        if not qid or qid in result:
            raise V23P0Error("metadata has empty or duplicate qid")
        anchor = _ids(raw.get("anchor_answer"), f"{qid}.anchor_answer")
        c35 = _ids(raw.get("candidates_c35"), f"{qid}.candidates_c35")
        scoring = _ids(raw.get("scoring_boundary"), f"{qid}.scoring_boundary")
        e3b = _ids(raw.get("e3b_scoring_boundary"), f"{qid}.e3b_scoring_boundary")
        unavailable = [str(item) for item in raw.get("e3b_source_unavailable", [])]
        references = _ids(raw.get("references"), f"{qid}.references")
        if len(anchor) != K or len(c35) != 35 or len(scoring) != 7 or scoring[0] != anchor[-1]:
            raise V23P0Error(f"{qid}: anchor/c35/scoring boundary contract violated")
        if len(unavailable) != len(set(unavailable)) or any(doc not in scoring for doc in unavailable):
            raise V23P0Error(f"{qid}: invalid E3B source-unavailable declaration")
        if e3b != [doc for doc in scoring if doc not in set(unavailable)]:
            raise V23P0Error(f"{qid}: E3B boundary does not match declared unavailable documents")
        result[qid] = {"anchor": anchor, "c35": c35, "scoring": scoring, "e3b": e3b, "unavailable": unavailable, "references": references, "fold": int(raw.get("fold", -1))}
    return result


def _load_e3b_keys(path: Path, field: str) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
                key = (str(row["qid"]), str(row["doc_id"]), str(row["config"]))
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise V23P0Error(f"invalid {field} row {number}") from exc
            if key in keys or key[2] not in CONFIGS:
                raise V23P0Error(f"duplicate or unknown {field} key at row {number}")
            keys.add(key)
    return keys


def _recall(answer: Sequence[str], references: set[str]) -> float:
    return len(set(answer) & references) / len(references)


def _oracle(anchor: Sequence[str], candidates: Sequence[str], references: set[str], max_swaps: int) -> tuple[list[str], int]:
    baseline = _recall(anchor, references)
    best_value = baseline
    best_swaps = 0
    best_answer = list(anchor)
    for count in range(1, max_swaps + 1):
        for slot_indices in itertools.combinations((3, 4), count):
            for chosen in itertools.permutations(candidates, count):
                if len(set(chosen)) != count or any(doc in anchor[:3] for doc in chosen):
                    continue
                answer = list(anchor)
                for index, doc in zip(slot_indices, chosen, strict=True):
                    answer[index] = doc
                value = _recall(answer, references)
                if value > best_value + 1e-12 or (abs(value - best_value) <= 1e-12 and count < best_swaps):
                    best_value, best_swaps, best_answer = value, count, answer
    return best_answer, best_swaps


def _pool_for(row: Mapping[str, Any], name: str) -> list[str]:
    anchor = list(row["anchor"])
    c35_challengers = [doc for doc in row["c35"] if doc not in set(anchor)]
    if name == "full_c35":
        return c35_challengers
    if name == "c35_rank_top6":
        return c35_challengers[:6]
    if name == "c35_rank_top8":
        return c35_challengers[:8]
    if name == "e3b_aligned_shortlist6":
        return list(row["e3b"])[1:] if row["e3b"] and row["e3b"][0] == anchor[-1] else list(row["e3b"])
    raise V23P0Error(f"unknown candidate pool {name}")


def _pool_summary(questions: Mapping[str, Mapping[str, Any]], pool_name: str, assignments: Mapping[str, Mapping[str, int]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_query = []
    for qid, row in questions.items():
        anchor = list(row["anchor"])
        references = set(row["references"])
        pool = _pool_for(row, pool_name)
        one, one_swaps = _oracle(anchor, pool, references, 1)
        two, two_swaps = _oracle(anchor, pool, references, 2)
        base = _recall(anchor, references)
        one_value = _recall(one, references)
        two_value = _recall(two, references)
        lost_relevant = len((set(anchor[3:]) - set(two[3:])) & references)
        per_query.append({"qid": qid, "baseline_recall": base, "o1_recall": one_value, "o2_recall": two_value, "o1_gain": one_value - base, "o2_gain": two_value - base, "o1_swaps": one_swaps, "o2_swaps": two_swaps, "o2_lost_relevant_incumbents": lost_relevant, "pool_size": len(pool), "pool": pool})
    baseline = statistics.fmean(row["baseline_recall"] for row in per_query)
    o1 = statistics.fmean(row["o1_recall"] for row in per_query)
    o2 = statistics.fmean(row["o2_recall"] for row in per_query)
    summary = {"queries": len(per_query), "pool_size_distribution": dict(sorted(Counter(row["pool_size"] for row in per_query).items())), "baseline_recall": baseline, "o1_recall": o1, "o1_gain": o1 - baseline, "o2_recall": o2, "o2_gain": o2 - baseline, "o1_improved_queries": sum(row["o1_gain"] > 1e-12 for row in per_query), "o2_improved_queries": sum(row["o2_gain"] > 1e-12 for row in per_query), "o2_worsened_queries": sum(row["o2_gain"] < -1e-12 for row in per_query), "o2_one_swap_improvements": sum(row["o2_gain"] > 1e-12 and row["o2_swaps"] == 1 for row in per_query), "o2_two_swap_improvements": sum(row["o2_gain"] > 1e-12 and row["o2_swaps"] == 2 for row in per_query), "o2_beneficial_query_equivalents": sum(row["o2_gain"] > 1e-12 for row in per_query), "o2_recall_equivalents": sum(row["o2_gain"] for row in per_query), "o2_lost_relevant_incumbent_cases": sum(row["o2_lost_relevant_incumbents"] > 0 for row in per_query)}
    regime_summary = {}
    for regime, regime_assignments in assignments.items():
        regime_qids = [qid for qid in questions if qid in regime_assignments]
        regime_summary[regime] = {"queries": len(regime_qids), "baseline_recall": statistics.fmean(next(row["baseline_recall"] for row in per_query if row["qid"] == qid) for qid in regime_qids), "o1_gain": statistics.fmean(next(row["o1_gain"] for row in per_query if row["qid"] == qid) for qid in regime_qids), "o2_gain": statistics.fmean(next(row["o2_gain"] for row in per_query if row["qid"] == qid) for qid in regime_qids), "o2_improved_queries": sum(next(row["o2_gain"] for row in per_query if row["qid"] == qid) > 1e-12 for qid in regime_qids)}
    summary["per_regime"] = regime_summary
    return summary, per_query


def run(args: argparse.Namespace) -> dict[str, Any]:
    pre = _validate_pre_registration(args.pre_registration, args)
    paths = _input_paths(args)
    prior = _read_json(paths["v22_b_report"], "V22-B report")
    if prior.get("promotion", {}).get("passed") is not False or prior.get("holdout_opened") is not False:
        raise V23P0Error("V23-P0 requires the completed, failed V22-B report with holdout closed")
    p0 = _load_module(paths["p0_script"], "v22_p0_shared")
    questions = _load_questions(paths["metadata"])
    folds = p0._load_regime_folds(paths["fold_manifest"], set(questions))
    e3b_manifest = _load_e3b_keys(paths["e3b_manifest"], "E3B manifest")
    e3b_scores = _load_e3b_keys(paths["e3b_scores"], "E3B score")
    expected_e3b = {(qid, doc, config) for qid, row in questions.items() for doc in row["e3b"] for config in CONFIGS}
    if e3b_manifest != expected_e3b or e3b_scores != expected_e3b:
        raise V23P0Error("E3B manifest/score keys do not exactly match metadata-declared packable boundaries")
    unavailable_qids = sorted(qid for qid, row in questions.items() if row["unavailable"])
    if len(unavailable_qids) != 6 or any(len(questions[qid]["unavailable"]) != 1 for qid in unavailable_qids):
        raise V23P0Error("the six declared E3B source-unavailable query cases changed")
    assignments = {regime: folds[regime] for regime in REGIMES}
    pools = {}
    traces = {}
    for name in ("full_c35", "c35_rank_top6", "c35_rank_top8", "e3b_aligned_shortlist6"):
        pools[name], traces[name] = _pool_summary(questions, name, assignments)
    selected = pools["e3b_aligned_shortlist6"]
    checks = {"o2_gain_at_least_0_020": selected["o2_gain"] >= PREREGISTRATION["gate"]["o2_gain_min"], "beneficial_query_equivalents_at_least_24": selected["o2_beneficial_query_equivalents"] >= PREREGISTRATION["gate"]["beneficial_query_equivalents_min"], "top_three_byte_identical": True, "e3b_fallback_outside_declared_six": len(unavailable_qids) == 6, "holdout_closed": True, "training_not_performed": True}
    report = {"kind": "legalir_v23_p0_two_slot_feasibility", "schema_version": 1, "stage": "V23-P0_complete_cpu_only_no_training_no_holdout", "split": "clean_complement_only", "holdout_opened": False, "public_labels_used": False, "model_loaded": False, "training_performed": False, "pre_registration": {"path": str(args.pre_registration.resolve()), "sha256": _sha256(args.pre_registration), "config": pre["config"]}, "inputs": pre["inputs"], "scope": {"queries": len(questions), "frozen_slots": [1, 2, 3], "mutable_slots": [4, 5], "e3b_manifest_keys": len(e3b_manifest), "e3b_score_keys": len(e3b_scores), "e3b_key_bijection": True, "declared_source_unavailable_qids": unavailable_qids, "no_e3b_fallback_outside_declared_six": True}, "candidate_pool_summaries": pools, "selected_training_pool": "e3b_aligned_shortlist6", "registered_v23_a_gate": {"checks": checks, "passed": all(checks.values()), "selected_pool_summary": selected}, "decision": {"holdout_may_open": False, "if_passed": "freeze E3B-aligned six-challenger shortlist and pre-register one V23-A joint two-slot set reranker using relevance coverage supervision", "if_failed": "close V23-A; do not expand candidate pool or open holdout"}, "per_query_oracle_traces": traces}
    report_path = args.output_dir / "v23_p0_report.json"
    if report_path.exists():
        raise FileExistsError(f"V23-P0 output already exists: {report_path}")
    _atomic_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v22-b-report", type=Path, required=True)
    parser.add_argument("--p0-script", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--e3b-manifest", type=Path, required=True)
    parser.add_argument("--e3b-scores", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--pre-registration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pre-register-only", action="store_true")
    args = parser.parse_args()
    if not args.pre_register_only and args.output_dir is None:
        parser.error("--output-dir is required for V23-P0 execution")
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
    print(json.dumps({"output": str((args.output_dir / "v23_p0_report.json").resolve()), "passed": report["registered_v23_a_gate"]["passed"], "holdout_opened": False}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
