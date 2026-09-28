"""V36-R: build clean P/N/U/C supervision from the sealed BTC complement.

Only official qrels create positives.  Negative certification uses explicit,
deterministic structural contradictions; semantic similarity, same-law overlap,
and Qwen scores never certify a negative.  Ambiguous rows are quarantined.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


SUBJECT_TERMS = (
    "người sử dụng lao động", "người lao động", "người nộp thuế", "cá nhân",
    "tổ chức", "doanh nghiệp", "hộ gia đình", "cơ quan nhà nước",
    "người vi phạm", "chủ đầu tư", "người sử dụng",
)
NUMBER_UNIT_RE = re.compile(r"\d+(?:[.,]\d+)?\s*(?:ngày|tháng|năm|tuổi|căn hộ|%|đồng|người|lần)", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def norm(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).casefold()).strip()


def load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != "legalir_v18b_evidence_pack_manifest" or payload.get("split") != "clean_complement_only":
        raise ValueError("metadata is not clean-complement")
    if payload.get("holdout_opened") is not False or payload.get("public_labels_used") is not False:
        raise ValueError("metadata does not attest unopened labels")
    questions = {}
    for raw in payload["questions"]:
        qid = str(raw["qid"])
        questions[qid] = {
            "qid": qid,
            "query": str(raw["query"]),
            "references": [str(x) for x in raw["references"]],
            "fold": int(raw.get("fold", -1)),
        }
    if len(questions) != 1197:
        raise ValueError(f"metadata size drift: {len(questions)}")
    return questions


def open_readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def hydrate(path: Path, doc_ids: list[str]) -> dict[str, str]:
    unique = list(dict.fromkeys(doc_ids))
    connection = open_readonly(path)
    try:
        placeholders = ",".join("?" for _ in unique)
        rows = connection.execute(
            f"SELECT doc_id, chunk_id, title, passage FROM chunks WHERE doc_id IN ({placeholders}) ORDER BY doc_id, chunk_id",
            unique,
        ).fetchall()
    finally:
        connection.close()
    result: dict[str, str] = {}
    for row in rows:
        doc = str(row["doc_id"])
        if doc not in result:
            result[doc] = "\n".join(x for x in (row["title"], row["passage"]) if x)
    return result


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    # Iterate by physical LF records.  ``str.splitlines()`` also treats some
    # legal-text control characters (for example U+0085) as line breaks even
    # though they are valid JSON string content.
    rows = []
    with path.open(encoding="utf-8", newline="\n") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: malformed JSONL record") from exc
    return rows


def certify_negative(row: Mapping[str, Any]) -> tuple[bool, list[str]]:
    query = norm(row["query"])
    gold = norm(row["gold_text"])
    hard = norm(row["hard_text"])
    reasons: list[str] = []
    query_subjects = [term for term in SUBJECT_TERMS if term in query]
    gold_subjects = [term for term in SUBJECT_TERMS if term in gold]
    hard_subjects = [term for term in SUBJECT_TERMS if term in hard]
    # A subject mismatch is considered certifiable only when the gold passage
    # contains the query subject and the hard passage contains another known
    # subject while omitting it.
    if query_subjects and any(term in gold_subjects for term in query_subjects) and not any(term in hard_subjects for term in query_subjects) and hard_subjects:
        reasons.append("explicit_subject_mismatch")
    # A completely unrelated source is only certified when all independent
    # structural cues disagree; lexical closeness alone can never certify N.
    if (
        float(row["query_token_coverage"]) < 0.30
        and float(row["gold_hard_sequence_ratio"]) < 0.45
        and not row["shared_law_references"]
        and not row["shared_articles"]
    ):
        reasons.append("structurally_unrelated_source")
    query_numbers = set(NUMBER_UNIT_RE.findall(query))
    hard_numbers = set(NUMBER_UNIT_RE.findall(hard))
    if query_numbers and hard_numbers and query_numbers.isdisjoint(hard_numbers):
        units_in_query = {value.split()[-1] for value in query_numbers}
        units_in_hard = {value.split()[-1] for value in hard_numbers}
        if units_in_query & units_in_hard:
            reasons.append("explicit_incompatible_numeric_time")
    return bool(reasons), reasons


def is_certified_cf(row: Mapping[str, Any]) -> tuple[bool, str]:
    factor = str(row["factor"])
    if factor in {"condition", "exception"}:
        return False, "CF_UNCERTAIN_condition_or_exception"
    if factor not in {"negation", "modality", "numeric_time", "subject", "authority"}:
        return False, "CF_UNCERTAIN_unknown_factor"
    if int(row.get("changed_token_count", 99)) > 8 or float(row.get("sequence_ratio", 0.0)) < 0.88:
        return False, "validator_fail_edit_distance"
    if norm(row["original_query"]) == norm(row["counterfactual_query"]):
        return False, "validator_fail_no_change"
    if re.search(r"\b(không không|được được|có có|khi khi)\b", norm(row["counterfactual_query"])):
        return False, "validator_fail_double_negation"
    return True, "C_CERTIFIED_COUNTERFACTUAL"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--counterfactuals", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--teacher-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    questions = load_metadata(args.metadata)
    audit = load_jsonl(args.audit)
    counterfactuals = load_jsonl(args.counterfactuals)
    teacher_report = json.loads(args.teacher_report.read_text(encoding="utf-8")) if args.teacher_report.exists() else None
    audit_by_pair = {(str(row["qid"]), str(row["hard_doc"])): row for row in audit}

    certified_negative: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    for row in audit:
        ok, reasons = certify_negative(row)
        result = {
            "qid": str(row["qid"]),
            "fold": row["fold"],
            "query": row["query"],
            "doc_id": str(row["hard_doc"]),
            "premise": row["hard_text"],
            "grade": 0,
            "bucket": "N" if ok else "U",
            "certification_reasons": reasons,
            "qrel_excludes_doc": bool(row["qrel_excludes_hard"]),
            "qwen_second_opinion": {"gold_wins": row.get("qwen_gold_wins_second_opinion"), "margin": row.get("qwen_margin_second_opinion")},
        }
        (certified_negative if ok else quarantined).append(result)

    all_gold_docs = [doc for question in questions.values() for doc in question["references"]]
    gold_texts = hydrate(args.index, all_gold_docs)
    certified_positive: list[dict[str, Any]] = []
    for qid in sorted(questions, key=lambda value: (int(value) if value.isdigit() else 10**18, value)):
        question = questions[qid]
        for doc_id in question["references"]:
            premise = gold_texts.get(doc_id, "")
            if not premise:
                raise ValueError(f"missing gold premise: {qid}/{doc_id}")
            certified_positive.append({
                "qid": qid,
                "fold": question["fold"],
                "query": question["query"],
                "doc_id": doc_id,
                "premise": premise,
                "grade": 2,
                "bucket": "P",
                "certification_reasons": ["official_clean_complement_qrel"],
            })

    certified_cf: list[dict[str, Any]] = []
    uncertain_cf: list[dict[str, Any]] = []
    clean_cf_seen: set[tuple[str, str, str]] = set()
    clean_cf_duplicates_collapsed = 0
    cf_by_factor: dict[str, dict[str, int]] = defaultdict(lambda: {"generated": 0, "validator_pass": 0, "validator_fail": 0, "teacher_agree": 0})
    teacher_agree_ids: set[str] = set()
    if args.teacher_report.exists():
        teacher_rows = load_jsonl(args.teacher_report.with_name("v36_b_qwen_teacher_validation.jsonl"))
        teacher_agree_ids = {str(row["counterfactual_id"]) for row in teacher_rows if row.get("teacher_supports_flip")}
    for row in counterfactuals:
        factor = str(row["factor"])
        cf_by_factor[factor]["generated"] += 1
        ok, reason = is_certified_cf(row)
        if ok:
            clean_key = (str(row["qid"]), str(row["gold_doc"]), norm(row["counterfactual_query"]))
            if clean_key in clean_cf_seen:
                clean_cf_duplicates_collapsed += 1
                continue
            clean_cf_seen.add(clean_key)
            cf_by_factor[factor]["validator_pass"] += 1
            certified_cf.append({
                "qid": str(row["qid"]),
                "fold": row["fold"],
                "query": row["counterfactual_query"],
                "original_query": row["original_query"],
                "counterfactual_query": row["counterfactual_query"],
                "doc_id": str(row["gold_doc"]),
                "premise": row["premise"],
                "grade": 1,
                "bucket": "C",
                "counterfactual_id": row["counterfactual_id"],
                "factor": factor,
                "certification_reasons": [reason, "single_deterministic_legal_factor"],
            })
        else:
            if reason.startswith("validator_fail"):
                cf_by_factor[factor]["validator_fail"] += 1
            uncertain_cf.append({**row, "bucket": "CF_UNCERTAIN", "validator_status": reason})
        if str(row["counterfactual_id"]) in teacher_agree_ids:
            cf_by_factor[factor]["teacher_agree"] += 1

    relations = [*certified_positive, *certified_negative, *certified_cf]
    keys = {(str(row["qid"]), str(row["doc_id"]), str(row["query"]), str(row["bucket"])) for row in relations}
    if len(keys) != len(relations):
        raise ValueError("duplicate clean relation")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    relations_path = args.output_dir / "v36_r_clean_relations.jsonl"
    quarantine_path = args.output_dir / "v36_r_quarantine.jsonl"
    cf_uncertain_path = args.output_dir / "v36_r_cf_uncertain.jsonl"
    write_jsonl(relations_path, relations)
    write_jsonl(quarantine_path, quarantined)
    write_jsonl(cf_uncertain_path, uncertain_cf)
    report = {
        "kind": "legalir_v36_r_clean_supervision",
        "schema_version": 1,
        "stage": "V36-R0_complete_no_manual_label_no_training",
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "training_performed": False,
        "qrels_authoritative": True,
        "qwen_used_as_second_opinion_only": True,
        "certified_positive": len(certified_positive),
        "certified_negative": len(certified_negative),
        "ambiguous_quarantined": len(quarantined),
        "negative_certification_reasons": dict(sorted(Counter(reason for row in certified_negative for reason in row["certification_reasons"]).items())),
        "counterfactual_by_type": dict(sorted(cf_by_factor.items())),
        "counterfactual_certified_C": len(certified_cf),
        "counterfactual_uncertain": len(uncertain_cf),
        "counterfactual_clean_duplicates_collapsed": clean_cf_duplicates_collapsed,
        "counterfactual_teacher_agree_total": sum(value["teacher_agree"] for value in cf_by_factor.values()),
        "counterfactual_teacher_report": teacher_report,
        "relations_for_pilot": {"P": len(certified_positive), "N": len(certified_negative), "C": len(certified_cf), "total": len(relations)},
        "gate": {
            "ambiguous_excluded_from_loss": True,
            "no_manual_label": True,
            "certified_cf_at_least_400": len(certified_cf) >= 400,
            "pilot_dataset_ready": len(certified_positive) > 0 and len(certified_negative) > 0 and len(certified_cf) >= 100,
        },
        "inputs": {
            "metadata": {"path": str(args.metadata.resolve()), "sha256": sha256_file(args.metadata)},
            "audit": {"path": str(args.audit.resolve()), "sha256": sha256_file(args.audit)},
            "counterfactuals": {"path": str(args.counterfactuals.resolve()), "sha256": sha256_file(args.counterfactuals)},
            "teacher_report": {"path": str(args.teacher_report.resolve()), "sha256": sha256_file(args.teacher_report) if args.teacher_report.exists() else None},
            "index": {"path": str(args.index.resolve()), "sha256": sha256_file(args.index)},
        },
        "outputs": {
            "clean_relations": {"path": str(relations_path.resolve()), "sha256": sha256_file(relations_path)},
            "quarantine": {"path": str(quarantine_path.resolve()), "sha256": sha256_file(quarantine_path)},
            "cf_uncertain": {"path": str(cf_uncertain_path.resolve()), "sha256": sha256_file(cf_uncertain_path)},
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = args.output_dir / "v36_r_clean_supervision_report.json"
    write_json(report_path, report)
    print(json.dumps({"report": str(report_path.resolve()), "P": len(certified_positive), "N": len(certified_negative), "U": len(quarantined), "C": len(certified_cf), "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
