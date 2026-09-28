"""V36-0/V36-B: CPU supervision audit and deterministic counterfactual candidates.

This stage deliberately does not train a model and does not use holdout/public
labels.  Existing qrels remain authoritative.  The Qwen scores are retained as
a second-opinion field only; they never override a qrel or automatically turn
an item into a positive.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import shutil
import sqlite3
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


STOPWORDS = {
    "là", "gì", "nào", "những", "các", "việc", "cho", "của", "và",
    "hoặc", "trong", "khi", "có", "được", "bị", "như", "thế", "nào",
    "ra", "sao", "về", "với", "từ", "theo", "phải", "cần", "ai",
    "bao", "gồm", "thực", "hiện", "quy", "định", "đối", "với",
}
TOKEN_RE = re.compile(r"[0-9a-zà-ỹđ]+", re.IGNORECASE)
ARTICLE_RE = re.compile(r"\bđiều\s+\d+[a-z]?", re.IGNORECASE)
LAW_REF_RE = re.compile(
    r"\b(?:luật|nghị\s+định|thông\s+tư|quyết\s+định|nghị\s+quyết|chỉ\s+thị|công\s+văn|pháp\s+lệnh)\s+(?:số\s+)?[0-9a-z./-]+",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(
    r"(?P<num>\d+(?:[.,]\d+)?)\s*(?P<unit>ngày|tháng|năm|tuổi|căn hộ|%|đồng|người|lần|số)",
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def norm_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    value = re.sub(r"[^0-9a-zà-ỹđ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def tokens(value: str, *, remove_stopwords: bool = False) -> set[str]:
    result = set(TOKEN_RE.findall(norm_text(value)))
    return result - STOPWORDS if remove_stopwords else result


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / max(len(union), 1)


def numeric_key(value: str) -> tuple[int, str]:
    return (int(value), value) if str(value).isdigit() else (10**18, str(value))


def load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != "legalir_v18b_evidence_pack_manifest" or payload.get("split") != "clean_complement_only":
        raise ValueError("metadata is not the clean-complement manifest")
    if payload.get("holdout_opened") is not False or payload.get("public_labels_used") is not False:
        raise ValueError("metadata does not attest unopened holdout/public labels")
    questions = {}
    for row in payload.get("questions", []):
        qid = str(row["qid"])
        if qid in questions:
            raise ValueError(f"duplicate qid: {qid}")
        refs = [str(x) for x in row["references"]]
        if not refs:
            raise ValueError(f"empty references: {qid}")
        questions[qid] = {
            "qid": qid,
            "query": str(row["query"]),
            "references": refs,
            "anchor": [str(x) for x in row.get("anchor_answer", [])],
            "fold": int(row.get("fold", -1)),
        }
    if len(questions) != 1197:
        raise ValueError(f"unexpected metadata size: {len(questions)}")
    return questions


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: expected object")
                rows.append(row)
    return rows


def document_structure(text: str) -> dict[str, Any]:
    normalized = norm_text(text)
    return {
        "title": normalized.split(" ", 1)[0] if normalized else "",
        "articles": {norm_text(x) for x in ARTICLE_RE.findall(text)},
        "law_refs": {norm_text(x) for x in LAW_REF_RE.findall(text)},
        "tokens": tokens(text),
        "content_tokens": tokens(text, remove_stopwords=True),
    }


def audit_rows(
    questions: Mapping[str, Mapping[str, Any]],
    benchmark: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    max_rows: int,
) -> list[dict[str, Any]]:
    scores = {(str(row["qid"]), str(row["hard_doc"])): row for row in score_rows}
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in benchmark:
        qid, hard_doc = str(row["qid"]), str(row["hard_doc"])
        if qid not in questions:
            raise ValueError(f"benchmark qid outside metadata: {qid}")
        key = (qid, hard_doc)
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
    selected.sort(key=lambda row: (numeric_key(str(row["qid"])), int(row.get("hard_source_rank", 999)), numeric_key(str(row["hard_doc"]))))
    selected = selected[:max_rows]
    output = []
    for row in selected:
        qid = str(row["qid"])
        question = questions[qid]
        refs = set(question["references"])
        score = scores.get((qid, str(row["hard_doc"])), {})
        gold_struct = document_structure(str(row["gold_text"]))
        hard_struct = document_structure(str(row["hard_text"]))
        query_tokens = tokens(question["query"], remove_stopwords=True)
        coverage = len(query_tokens & hard_struct["content_tokens"]) / max(len(query_tokens), 1)
        text_jaccard = jaccard(gold_struct["content_tokens"], hard_struct["content_tokens"])
        sequence_ratio = difflib.SequenceMatcher(None, norm_text(row["gold_text"]), norm_text(row["hard_text"])).ratio()
        shared_laws = sorted(gold_struct["law_refs"] & hard_struct["law_refs"])
        shared_articles = sorted(gold_struct["articles"] & hard_struct["articles"])
        exact_duplicate = norm_text(row["gold_text"]) == norm_text(row["hard_text"])
        near_duplicate = (text_jaccard >= 0.85 and sequence_ratio >= 0.90) or sequence_ratio >= 0.97
        same_article_version = bool(shared_laws and shared_articles)
        citation_equivalence_signal = len(shared_laws) >= 2
        qwen_gold_wins = score.get("gold_wins")
        qwen_margin = score.get("margin")
        qwen_supports_hard = bool(qwen_gold_wins is False)
        # This is a triage label, not an authoritative relabel.  The qrel
        # exclusion is explicit and all non-definite buckets are review queues.
        if exact_duplicate or near_duplicate:
            triage = "DUPLICATE_OR_EQUIVALENT"
        elif str(row["hard_doc"]) in refs:
            triage = "AMBIGUOUS"
        elif (
            qwen_supports_hard
            and coverage >= 0.75
            and (same_article_version or citation_equivalence_signal or float(qwen_margin or 0.0) >= -0.50)
        ):
            triage = "LIKELY_RELEVANT_BUT_UNLABELED"
        elif coverage >= 0.70 or same_article_version or citation_equivalence_signal or (qwen_supports_hard and abs(float(qwen_margin or 0.0)) < 1.0):
            triage = "AMBIGUOUS"
        elif qwen_gold_wins is True and float(qwen_margin or 0.0) >= 1.0 and coverage < 0.55:
            triage = "DEFINITE_NEGATIVE"
        else:
            triage = "AMBIGUOUS"
        source = "old_slot5" if question.get("anchor", [None] * 5)[4:5] == [str(row["hard_doc"])] else "dense_top500_false_candidate"
        output.append({
            "qid": qid,
            "query": question["query"],
            "fold": question["fold"],
            "gold_doc": str(row["gold_doc"]),
            "hard_doc": str(row["hard_doc"]),
            "hard_source_rank": int(row.get("hard_source_rank", 0)),
            "source_category": source,
            "qrel_excludes_hard": str(row["hard_doc"]) not in refs,
            "qwen_gold_wins_second_opinion": qwen_gold_wins,
            "qwen_margin_second_opinion": qwen_margin,
            "query_token_coverage": coverage,
            "gold_hard_token_jaccard": text_jaccard,
            "gold_hard_sequence_ratio": sequence_ratio,
            "shared_law_references": shared_laws,
            "shared_articles": shared_articles,
            "exact_duplicate_signal": exact_duplicate,
            "near_duplicate_signal": near_duplicate,
            "same_article_version_signal": same_article_version,
            "citation_equivalence_signal": citation_equivalence_signal,
            "triage_label": triage,
            "needs_manual_review": triage != "DEFINITE_NEGATIVE",
            "gold_rank_best_view": row.get("gold_rank_best_view"),
            "hard_rank_best_view": row.get("hard_rank_best_view"),
            "gold_text": row["gold_text"],
            "hard_text": row["hard_text"],
        })
    return output


def open_readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def hydrate_gold_texts(index_path: Path, doc_ids: list[str]) -> dict[str, str]:
    connection = open_readonly(index_path)
    try:
        placeholders = ",".join("?" for _ in doc_ids)
        rows = connection.execute(
            f"SELECT doc_id, chunk_id, title, passage FROM chunks WHERE doc_id IN ({placeholders}) ORDER BY doc_id, chunk_id",
            doc_ids,
        ).fetchall()
    finally:
        connection.close()
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in rows:
        text = "\n".join(x for x in (row["title"], row["passage"]) if x)
        if text:
            grouped[str(row["doc_id"])].append((str(row["chunk_id"]), text))
    return {doc: texts[0][1] for doc, texts in grouped.items()}


def replace_once(query: str, old: str, new: str) -> str | None:
    if old.casefold() not in query.casefold():
        return None
    pattern = re.compile(re.escape(old), re.IGNORECASE)
    return pattern.sub(new, query, count=1)


def number_mutation(query: str) -> str | None:
    match = NUMBER_RE.search(query)
    if not match:
        return None
    context = query[max(0, match.start() - 32): min(len(query), match.end() + 32)].casefold()
    if not re.search(r"dưới|trên|quá|ít nhất|nhiều hơn|chỉ có|trong vòng|sau bao lâu|từ ngày|đến ngày|bao nhiêu", context):
        return None
    raw = match.group("num").replace(",", ".")
    value = float(raw) if "." in raw else int(raw)
    replacement_value = str(int(value) + 1) if isinstance(value, int) else f"{value + 1:g}"
    return query[:match.start("num")] + replacement_value + query[match.end("num"):]


def generate_mutations(query: str) -> list[tuple[str, str, str]]:
    candidates: list[tuple[str, str, str]] = []

    modal_pairs = (
        ("được phép", "không được phép"),
        ("có thể", "không thể"),
        ("bắt buộc", "không bắt buộc"),
        ("có được", "không được"),
        ("phải", "không phải"),
    )
    for left, right in modal_pairs:
        if re.search(re.escape(left), query, re.IGNORECASE):
            mutated = replace_once(query, left, right)
            if mutated:
                candidates.append(("modality", mutated, f"{left}->{right}"))
                break
        if re.search(re.escape(right), query, re.IGNORECASE):
            mutated = replace_once(query, right, left)
            if mutated:
                candidates.append(("modality", mutated, f"{right}->{left}"))
                break

    if re.search(r"\bkhông\b", query, re.IGNORECASE):
        mutated = re.sub(r"\bkhông\s+", "", query, count=1, flags=re.IGNORECASE)
        candidates.append(("negation", mutated, "remove_first_negation"))
    else:
        for phrase in ("được phép", "có thể", "bắt buộc", "phải"):
            mutated = replace_once(query, phrase, "không " + phrase)
            if mutated:
                candidates.append(("negation", mutated, f"insert_before:{phrase}"))
                break

    subject_pairs = (
        ("người sử dụng lao động", "người lao động"),
        ("người lao động", "người sử dụng lao động"),
        ("cá nhân", "tổ chức"),
        ("tổ chức", "cá nhân"),
        ("doanh nghiệp", "cá nhân"),
        ("cá nhân", "doanh nghiệp"),
        ("người nộp thuế", "tổ chức nộp thuế"),
    )
    for left, right in subject_pairs:
        mutated = replace_once(query, left, right)
        if mutated and right.casefold() not in query.casefold():
            candidates.append(("subject", mutated, f"{left}->{right}"))
            break

    for left, right in (("trừ trường hợp", "kể cả trường hợp"), ("ngoài trường hợp", "trong trường hợp"), ("nếu", "nếu không")):
        mutated = replace_once(query, left, right)
        if mutated:
            candidates.append(("exception" if left != "nếu" else "condition", mutated, f"{left}->{right}"))
            break

    condition = replace_once(query, " khi ", " khi không ")
    if condition:
        candidates.append(("condition", condition, "insert_condition_negation"))

    numeric = number_mutation(query)
    if numeric:
        candidates.append(("numeric_time", numeric, "increment_threshold_number"))

    authority_pairs = (
        ("Bộ trưởng", "Chủ tịch Ủy ban nhân dân"),
        ("Chủ tịch Ủy ban nhân dân", "Bộ trưởng"),
        ("Thủ tướng Chính phủ", "Bộ trưởng"),
        ("Bộ trưởng", "Thủ tướng Chính phủ"),
    )
    for left, right in authority_pairs:
        mutated = replace_once(query, left, right)
        if mutated and right.casefold() not in query.casefold():
            candidates.append(("authority", mutated, f"{left}->{right}"))
            break
    return candidates


def validate_mutation(original: str, mutated: str) -> tuple[bool, dict[str, Any]]:
    original_norm, mutated_norm = norm_text(original), norm_text(mutated)
    original_tokens, mutated_tokens = tokens(original), tokens(mutated)
    changed = sorted((original_tokens ^ mutated_tokens))
    ratio = difflib.SequenceMatcher(None, original_norm, mutated_norm).ratio()
    valid = bool(mutated_norm and original_norm != mutated_norm and ratio >= 0.88 and len(changed) <= 8)
    if re.search(r"\b(không không|được được|có có|khi khi)\b", mutated_norm):
        valid = False
    return valid, {"sequence_ratio": ratio, "symmetric_token_changes": changed, "changed_token_count": len(changed)}


def build_counterfactuals(questions: Mapping[str, Mapping[str, Any]], gold_texts: Mapping[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for qid in sorted(questions, key=numeric_key):
        question = questions[qid]
        gold_doc = sorted(question["references"], key=numeric_key)[0]
        premise = gold_texts.get(gold_doc, "")
        if not premise:
            continue
        for factor, mutated, rule in generate_mutations(question["query"]):
            valid, details = validate_mutation(question["query"], mutated)
            key = (qid, factor, norm_text(mutated))
            if not valid or key in seen:
                continue
            seen.add(key)
            rows.append({
                "counterfactual_id": f"{qid}:{factor}:{len(rows)}",
                "qid": qid,
                "fold": question["fold"],
                "factor": factor,
                "mutation_rule": rule,
                "original_query": question["query"],
                "counterfactual_query": mutated,
                "gold_doc": gold_doc,
                "premise": premise,
                "provisional_label": "NON_ENTAIL_CANDIDATE",
                "validator_status": "structural_pass_pending_legal_or_teacher_validation",
                "teacher_validation_required": factor in {"condition", "exception"},
                **details,
            })
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    questions = load_metadata(args.metadata)
    benchmark = load_jsonl(args.hard_benchmark)
    score_rows = load_jsonl(args.hard_scores) if args.hard_scores.exists() else []
    audit = audit_rows(questions, benchmark, score_rows, args.max_audit_negatives)
    gold_docs = sorted({sorted(questions[qid]["references"], key=numeric_key)[0] for qid in questions}, key=numeric_key)
    gold_texts = hydrate_gold_texts(args.index, gold_docs)
    counterfactuals = build_counterfactuals(questions, gold_texts)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "v36_0_negative_audit.jsonl"
    cf_path = args.output_dir / "v36_b_counterfactual_candidates.jsonl"
    atomic_jsonl(audit_path, audit)
    atomic_jsonl(cf_path, counterfactuals)
    label_counts = Counter(row["triage_label"] for row in audit)
    factor_counts = Counter(row["factor"] for row in counterfactuals)
    teacher_report_path = args.output_dir / "v36_b_qwen_teacher_validation_report.json"
    teacher_report = json.loads(teacher_report_path.read_text(encoding="utf-8")) if teacher_report_path.exists() else None
    report = {
        "kind": "legalir_v36_supervision_repair_cpu_preparation",
        "schema_version": 1,
        "stage": "V36-0_and_V36-B_rules_only_no_training",
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "training_performed": False,
        "qrels_authoritative": True,
        "qwen_role": "second_opinion_only_no_auto_relabel",
        "v36_0": {
            "audited_pairs": len(audit),
            "audited_qids": len({row["qid"] for row in audit}),
            "target_range": [500, 1000],
            "within_target": 500 <= len(audit) <= 1000,
            "triage_counts": dict(sorted(label_counts.items())),
            "manual_review_count": sum(row["needs_manual_review"] for row in audit),
            "review_rate": sum(row["needs_manual_review"] for row in audit) / max(len(audit), 1),
            "qwen_score_rows_joined": sum((str(row["qid"]), str(row["hard_doc"])) in {(str(x["qid"]), str(x["hard_doc"])) for x in score_rows} for row in audit),
            "gate_note": "Triage buckets are not label replacement; non-definite rows require adjudication before cleaning.",
        },
        "v36_b": {
            "candidate_rows": len(counterfactuals),
            "source_qids": len({row["qid"] for row in counterfactuals}),
            "factor_counts": dict(sorted(factor_counts.items())),
            "provisional_non_entail_only": True,
            "structural_validator_passed": True,
            "teacher_validation_pending": sum(row["teacher_validation_required"] for row in counterfactuals),
            "target_range": [4000, 6000],
            "target_reached": 4000 <= len(counterfactuals) <= 6000,
            "training_ready": False,
            "teacher_validation_report": teacher_report,
        },
        "inputs": {
            "metadata": {"path": str(args.metadata.resolve()), "sha256": sha256_file(args.metadata)},
            "hard_benchmark": {"path": str(args.hard_benchmark.resolve()), "sha256": sha256_file(args.hard_benchmark)},
            "hard_scores": {"path": str(args.hard_scores.resolve()), "sha256": sha256_file(args.hard_scores) if args.hard_scores.exists() else None},
            "index": {"path": str(args.index.resolve()), "sha256": sha256_file(args.index)},
        },
        "outputs": {
            "negative_audit": {"path": str(audit_path.resolve()), "sha256": sha256_file(audit_path)},
            "counterfactual_candidates": {"path": str(cf_path.resolve()), "sha256": sha256_file(cf_path)},
            "teacher_validation": {"path": str(teacher_report_path.resolve()), "sha256": sha256_file(teacher_report_path)} if teacher_report_path.exists() else None,
        },
        "decision": {
            "v36_0_cleaning_required_if_manual_review_ge_0_05": (sum(row["needs_manual_review"] for row in audit) / max(len(audit), 1)) >= 0.05,
            "next_safe_step": "adjudicate V36-0 review queue; run V36-A0 only after a permitted NLI checkpoint is available",
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(args.output_dir / "v36_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--hard-benchmark", type=Path, required=True)
    parser.add_argument("--hard-scores", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-audit-negatives", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({
        "output": result["outputs"],
        "v36_0": result["v36_0"],
        "v36_b": result["v36_b"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, ensure_ascii=False))
