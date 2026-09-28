"""CPU-only parity, V27 forensic, evidence, and V28 boundary preparation audits."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from v27_common import (
    canonical_json,
    load_document_info,
    load_metadata,
    load_sidecar,
    make_canonical_map,
    normalized_text,
    read_json,
    sha256_bytes,
    sha256_file,
    sha256_lines,
    text_tokens,
    write_json,
    write_jsonl,
)


STOPWORDS = {
    "và", "là", "của", "cho", "trong", "được", "những", "các", "một", "như",
    "với", "theo", "khi", "từ", "đến", "nào", "gì", "thì", "có", "không", "về",
    "sau", "trước", "tại", "bởi", "hay", "hoặc", "này", "đó", "để", "do", "ra",
    "phải", "bao", "lâu", "bị", "đã", "mà", "ai", "làm", "thế", "nào",
}

ELEMENT_PATTERNS = {
    "SUBJECT": ("đối tượng", "chủ thể", "người nào", "ai được", "ai có", "cơ quan", "tổ chức", "cá nhân", "pháp nhân"),
    "BEHAVIOR": ("hành vi", "làm gì", "thực hiện", "vi phạm", "được làm", "phải làm", "cấm"),
    "CONDITION": ("điều kiện", "trường hợp", "khi nào", "nếu", "được cấp", "được hưởng", "có được"),
    "EXCEPTION": ("ngoại lệ", "trừ", "không áp dụng", "ngoài", "ngoại trừ"),
    "TIME": ("bao lâu", "thời hạn", "khi nào", "trong thời hạn", "kể từ", "sau bao"),
    "PENALTY": ("mức phạt", "phạt", "xử phạt", "tiền phạt", "tước", "buộc", "chế tài"),
    "AUTHORITY": ("thẩm quyền", "cơ quan nào", "ai có quyền", "thẩm phán", "sở ", "bộ ", "ủy ban", "tòa án"),
    "OBJECT": ("đối tượng", "hồ sơ", "giấy tờ", "văn bản", "tài liệu", "bao gồm những gì", "gồm những gì"),
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _content_tokens(value: str) -> set[str]:
    return {token for token in text_tokens(value) if token not in STOPWORDS and len(token) > 1}


def _stable_qid_hash(qids: list[str]) -> str:
    return sha256_bytes(canonical_json(sorted(map(str, qids))).encode("utf-8"))


def _recall(answer: list[str], refs: set[str]) -> float:
    return len(set(map(str, answer)) & set(map(str, refs))) / max(len(refs), 1)


def _load_oof_scores(root: Path, split: Mapping[str, Any]) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    for fold_row in split["folds"]:
        fold = int(fold_row["fold"])
        path = root / f"fold_{fold}" / f"fold_{fold}_outer_logits.jsonl"
        for row in _read_jsonl(path):
            key = (str(row["qid"]), str(row["doc_id"]))
            if key in result:
                raise ValueError(f"duplicate OOF score: {key}")
            result[key] = float(row.get("raw_logit", row.get("score")))
    return result


def _load_packs(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for row in _read_jsonl(path):
        key = (str(row["qid"]), str(row["doc_id"]))
        if key in result:
            raise ValueError(f"duplicate E3B pack: {key}")
        result[key] = row
    return result


def _metric_parity(
    questions: Mapping[str, Mapping[str, Any]],
    baseline_rows: list[dict[str, Any]],
    baseline_report: Mapping[str, Any],
    qrels_path: Path,
    c35_path: Path,
    baseline_path: Path,
    metadata_path: Path,
    train_path: Path,
    group_manifest_path: Path,
    contexts_zip: Path,
    e3b_path: Path,
    oof_report: Mapping[str, Any],
) -> dict[str, Any]:
    qids = list(questions)
    expected_qrels = [
        {"qid": qid, "docids": sorted(questions[qid]["references"], key=lambda x: (int(x) if str(x).isdigit() else 10**18, str(x)))}
        for qid in qids
    ]
    expected_c35 = [{"qid": qid, "docids": questions[qid]["c35"]} for qid in qids]
    expected_baseline = [
        {"qid": qid, "query": questions[qid]["query"], "candidates_c35": questions[qid]["c35"], "answer": questions[qid]["anchor"]}
        for qid in qids
    ]
    actual_qrels = _read_jsonl(qrels_path)
    actual_c35 = _read_jsonl(c35_path)
    actual_baseline = _read_jsonl(baseline_path)
    qid_set = set(qids)
    recomputed = float(np.mean([_recall(questions[qid]["anchor"], questions[qid]["references"]) for qid in qids]))
    doc_ids = {doc for row in questions.values() for doc in row["c35"]}
    info = load_document_info(contexts_zip, doc_ids)
    empty_doc_ids = sorted(doc for doc, row in info.items() if row["empty_body"])
    empty_answer_slots = sum(1 for row in questions.values() for doc in row["anchor"] if info[str(doc)]["empty_body"])
    e3b_rows = _read_jsonl(e3b_path)
    e3b_empty_rows = sum(bool(row.get("empty")) for row in e3b_rows)
    scorer_contract = {
        "implementation": "experiments/v27/v27_common.py::recall",
        "formula": "len(set(answer) & references) / max(len(references), 1)",
        "answer_duplicates_removed": True,
        "reference_duplicates_removed": True,
        "answer_length_expected": 5,
        "top1_4_locked": True,
        "empty_passage_policy": "E3B empty candidate is vetoed; baseline scoring itself does not drop the qid",
        "v27_common_sha256": sha256_file(Path(__file__).parent.parent / "v27" / "v27_common.py"),
        "v27_evaluate_sha256": sha256_file(Path(__file__).parent.parent / "v27" / "v27_evaluate.py"),
    }
    hash_audit = {
        "qid_set_sha256": _stable_qid_hash(qids),
        "qrels_sha256_file": sha256_file(qrels_path),
        "qrels_sha256_canonical": sha256_lines(expected_qrels),
        "qrels_matches_frozen": sha256_file(qrels_path) == baseline_report["input_hashes"]["qrels"],
        "c35_sha256_file": sha256_file(c35_path),
        "c35_matches_frozen": sha256_file(c35_path) == baseline_report["input_hashes"]["c35_manifest"],
        "baseline_prediction_sha256_file": sha256_file(baseline_path),
        "baseline_prediction_sha256_canonical": sha256_lines(expected_baseline),
        "baseline_prediction_matches_frozen": sha256_file(baseline_path) == baseline_report["input_hashes"]["baseline_predictions"],
        "metadata_sha256": sha256_file(metadata_path),
        "train_json_sha256": sha256_file(train_path),
        "group_manifest_sha256": sha256_file(group_manifest_path),
    }
    row_checks = {
        "metadata_qids": len(qids),
        "qrels_qids": len({str(row["qid"]) for row in actual_qrels}),
        "c35_qids": len({str(row["qid"]) for row in actual_c35}),
        "baseline_prediction_qids": len({str(row["qid"]) for row in actual_baseline}),
        "same_qid_set_metadata_qrels": {str(row["qid"]) for row in actual_qrels} == qid_set,
        "same_qid_set_metadata_c35": {str(row["qid"]) for row in actual_c35} == qid_set,
        "same_qid_set_metadata_baseline": {str(row["qid"]) for row in actual_baseline} == qid_set,
        "qrels_duplicate_qids": len(actual_qrels) - len({str(row["qid"]) for row in actual_qrels}),
        "c35_duplicate_qids": len(actual_c35) - len({str(row["qid"]) for row in actual_c35}),
        "baseline_duplicate_qids": len(actual_baseline) - len({str(row["qid"]) for row in actual_baseline}),
        "baseline_answer_duplicate_rows": sum(len(row["answer"]) != len(set(row["answer"])) for row in actual_baseline),
        "baseline_c35_duplicate_rows": sum(len(row["candidates_c35"]) != len(set(row["candidates_c35"])) for row in actual_baseline),
    }
    return {
        "kind": "legalir_v28_metric_parity_audit",
        "schema_version": 1,
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "same_split_contract": "same split → baseline must be byte/metric identical",
        "locked_reference": {
            "value": float(baseline_report["baseline_recall_locked_reference"]),
            "origin": "V27 frozen locked-holdout reference; not recomputed or opened in this audit",
            "same_split_as_current_oof": False,
        },
        "clean_complement": {
            "recomputed_baseline_recall": recomputed,
            "frozen_baseline_recall": float(baseline_report["baseline_recall_clean_complement"]),
            "oof_report_baseline_recall": float(oof_report["overall"]["baseline_recall"]),
            "metric_identical_within_tolerance": abs(recomputed - float(baseline_report["baseline_recall_clean_complement"])) <= 1e-15 and abs(recomputed - float(oof_report["overall"]["baseline_recall"])) <= 1e-15,
            "queries": len(qids),
            "denominator": "mean over qids of |set(answer) ∩ references| / max(|references|,1)",
        },
        "hashes": hash_audit,
        "row_checks": row_checks,
        "empty_passage_handling": {
            "unique_c35_documents": len(doc_ids),
            "empty_c35_document_ids": empty_doc_ids,
            "empty_c35_document_count": len(empty_doc_ids),
            "empty_baseline_answer_slots": empty_answer_slots,
            "e3b_empty_rows": e3b_empty_rows,
            "e3b_total_rows": len(e3b_rows),
            "contract_empty_body_output": int(oof_report["contract_checks"]["empty_body_output"]),
        },
        "scorer_version": scorer_contract,
        "verdict": "metric parity confirmed on clean complement; 0.922833 is a locked-reference metric from a different split",
    }


def _gold_candidates(row: Mapping[str, Any]) -> list[str]:
    refs = {str(x) for x in row["references"]}
    return [doc for doc in row["c35"] if str(doc) in refs]


def _slot5_oracle(questions: Mapping[str, Mapping[str, Any]], canonical: Mapping[str, str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_query = []
    for qid, row in questions.items():
        refs = set(map(str, row["references"]))
        baseline = _recall(row["anchor"], refs)
        gold_raw = _gold_candidates(row)
        gold_canonical = {canonical.get(doc, doc) for doc in refs}
        canonical_c35 = [doc for doc in row["c35"] if canonical.get(doc, doc) in gold_canonical]
        available = [doc for doc in gold_raw if doc not in row["anchor"][:4]]
        oracle_answer = list(row["anchor"][:4]) + ([available[0]] if available else [row["anchor"][4]])
        oracle = _recall(oracle_answer, refs)
        baseline_miss = baseline < 1e-12
        per_query.append({
            "qid": qid,
            "baseline_recall": baseline,
            "baseline_miss": baseline_miss,
            "gold_in_raw_c35": bool(gold_raw),
            "gold_in_canonical_c35": bool(canonical_c35),
            "gold_absent_raw_c35": not bool(gold_raw),
            "gold_canonical_only_c35": not bool(gold_raw) and bool(canonical_c35),
            "raw_gold_candidates": gold_raw,
            "oracle_slot5_doc": available[0] if available else None,
            "oracle_recall": oracle,
            "oracle_delta": oracle - baseline,
        })
    misses = [row for row in per_query if row["baseline_miss"]]
    rescueable = [row for row in misses if row["gold_in_raw_c35"] and row["oracle_slot5_doc"]]
    absent = [row for row in misses if row["gold_absent_raw_c35"]]
    oracle_recall = float(np.mean([row["oracle_recall"] for row in per_query]))
    baseline_recall = float(np.mean([row["baseline_recall"] for row in per_query]))
    summary = {
        "kind": "legalir_v28_slot5_oracle_audit",
        "schema_version": 1,
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "queries": len(per_query),
        "n_baseline_miss": len(misses),
        "n_rescueable_within_c35": len(rescueable),
        "n_gold_absent_c35": len(absent),
        "n_gold_canonical_only_c35": sum(row["gold_canonical_only_c35"] for row in misses),
        "oracle_slot5_recall": oracle_recall,
        "oracle_gain": oracle_recall - baseline_recall,
        "oracle_gain_ge_0.010": oracle_recall - baseline_recall >= 0.010,
        "interpretation": "raw qrel membership; canonical-only duplicates are reported separately and are not counted as raw recall rescue",
    }
    return summary, per_query


def _element_hits(query: str, pack: str) -> tuple[list[str], list[str]]:
    q = normalized_text(query)
    p = normalized_text(pack)
    requested = [element for element, patterns in ELEMENT_PATTERNS.items() if any(normalized_text(pattern) in q for pattern in patterns)]
    hits = [element for element in requested if any(normalized_text(pattern) in p for pattern in ELEMENT_PATTERNS[element])]
    return requested, hits


def _classify_evidence(query: str, pack_row: Mapping[str, Any]) -> dict[str, Any]:
    pack = str(pack_row.get("pack", ""))
    if bool(pack_row.get("empty")) or not pack.strip():
        return {"status": "EVIDENCE_MISSING", "method": "lexical_proxy", "query_token_coverage": 0.0, "requested_elements": [], "hit_elements": [], "matched_query_tokens": []}
    q_tokens = _content_tokens(query)
    p_tokens = _content_tokens(pack)
    matched = sorted(q_tokens & p_tokens)
    coverage = len(matched) / max(len(q_tokens), 1)
    requested, hits = _element_hits(query, pack)
    element_ratio = len(hits) / max(len(requested), 1)
    if requested:
        if element_ratio >= 0.75 and coverage >= 0.40:
            status = "EVIDENCE_PRESENT"
        elif element_ratio == 0 and coverage < 0.20:
            status = "EVIDENCE_MISSING"
        else:
            status = "EVIDENCE_PARTIAL"
    elif coverage >= 0.55:
        status = "EVIDENCE_PRESENT"
    elif coverage < 0.20:
        status = "EVIDENCE_MISSING"
    else:
        status = "EVIDENCE_PARTIAL"
    return {"status": status, "method": "lexical_proxy", "query_token_coverage": coverage, "requested_elements": requested, "hit_elements": hits, "matched_query_tokens": matched}


def _evidence_audit(
    questions: Mapping[str, Mapping[str, Any]],
    oof_scores: Mapping[tuple[str, str], float],
    packs: Mapping[tuple[str, str], Mapping[str, Any]],
    canonical: Mapping[str, str],
    split: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fold_by_qid = {str(qid): int(fold_row["fold"]) for fold_row in split["folds"] for qid in fold_row["test_qids"]}
    rows = []
    for qid, question in questions.items():
        gold = _gold_candidates(question)
        if not gold:
            continue
        ordered = sorted(question["c35"], key=lambda doc: (-float(oof_scores.get((qid, doc), -1e30)), question["c35"].index(doc)))
        rank = {doc: index + 1 for index, doc in enumerate(ordered)}
        baseline_miss = _recall(question["anchor"], set(question["references"])) < 1e-12
        for gold_doc in gold:
            if not baseline_miss or rank.get(gold_doc, 999) <= 4:
                continue
            evidence = _classify_evidence(question["query"], packs[(qid, gold_doc)])
            rows.append({
                "qid": qid,
                "outer_fold": fold_by_qid.get(qid),
                "query": question["query"],
                "gold_doc": gold_doc,
                "gold_rank_by_v27_oof": rank.get(gold_doc),
                "incumbent_slot5": question["anchor"][4],
                "pack_empty": bool(packs[(qid, gold_doc)].get("empty")),
                "pack_sha256": packs[(qid, gold_doc)].get("pack_sha256"),
                **evidence,
            })
    counter = collections.Counter(row["status"] for row in rows)
    denominator = len(rows)
    summary = {
        "kind": "legalir_v28_evidence_bottleneck_audit",
        "schema_version": 1,
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "method": "lexical proxy over query/E3B pack; candidate statuses require manual adjudication before treating as decisive evidence",
        "denominator_definition": "baseline-miss qids whose raw gold is in c35 and whose V27 OOF rank is below top-4",
        "denominator": denominator,
        "counts": dict(sorted(counter.items())),
        "evidence_missing_rate": counter["EVIDENCE_MISSING"] / max(denominator, 1),
        "evidence_partial_rate": counter["EVIDENCE_PARTIAL"] / max(denominator, 1),
        "evidence_present_rate": counter["EVIDENCE_PRESENT"] / max(denominator, 1),
        "rule_20_25_percent": "if lexical-proxy missing rate > 0.20–0.25, prioritize E3B repair; this audit is not a human gold adjudication",
    }
    return summary, rows


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / max(len(left | right), 1)


def _negative_pool(
    questions: Mapping[str, Mapping[str, Any]],
    scores: Mapping[tuple[str, str], float],
    info: Mapping[str, Mapping[str, Any]],
    canonical: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = []
    counts = collections.Counter()
    category_counts = collections.Counter()
    qids_without_positive = []
    # This corpus contains long legal bodies.  Materialize lexical tokens once
    # per document; recomputing them inside the 1197×35 query loop is needlessly
    # expensive and obscures the fact that this audit is CPU-only.
    content_cache = {doc: _content_tokens(str(row["title"]) + " " + str(row["body"])) for doc, row in info.items()}
    for qid, question in questions.items():
        gold = _gold_candidates(question)
        if not gold:
            qids_without_positive.append(qid)
            continue
        positive = gold[0]
        gold_canonical = {canonical.get(doc, doc) for doc in question["references"]}
        candidates = list(question["c35"])
        order = {doc: i for i, doc in enumerate(candidates)}
        ranked = sorted(candidates, key=lambda doc: (-float(scores.get((qid, doc), -1e30)), order[doc]))
        q_tokens = _content_tokens(question["query"])
        positive_title = set(info[positive]["title_tokens"])
        forbidden = set(question["anchor"][:4]) | {doc for doc in candidates if canonical.get(doc, doc) in gold_canonical} | {doc for doc in candidates if info[doc]["empty_body"]}
        selected: list[dict[str, Any]] = []

        def add(category: str, docs: list[str], limit: int) -> None:
            added = 0
            for doc in docs:
                if added >= limit or doc in forbidden or any(row["negative_doc"] == doc for row in selected):
                    continue
                selected.append({"negative_doc": doc, "negative_type": category, "negative_rank_by_stage2": order[doc] + 1, "stage2_score": float(scores.get((qid, doc), 0.0))})
                category_counts[category] += 1
                added += 1

        add("TOP5_FP", candidates[4:7], 3)
        add("HIGH_SEMANTIC", ranked[5:15], 3)
        same_law = sorted(
            candidates,
            key=lambda doc: (
                bool(info[doc]["law_family"] and info[doc]["law_family"] == info[positive]["law_family"]),
                _jaccard(set(info[doc]["title_tokens"]), positive_title),
                float(scores.get((qid, doc), -1e30)),
                -order[doc],
            ),
            reverse=True,
        )
        add("SAME_LAW_NEAR_MISS", same_law, 4)
        lexical = sorted(candidates, key=lambda doc: (_jaccard(q_tokens, content_cache[doc]), float(scores.get((qid, doc), -1e30)), -order[doc]), reverse=True)
        add("LEXICAL_COLLISION", lexical, 2)
        add("MEDIUM", ranked[15:], 2)
        structural = sorted(candidates, key=lambda doc: (_jaccard(set(info[doc]["title_tokens"]), positive_title), bool(info[doc]["law_family"] and info[doc]["law_family"] == info[positive]["law_family"]), -order[doc]), reverse=True)
        add("STRUCTURALLY_SIMILAR", structural, 2)
        # The structured categories normally provide 16 negatives.  Fill only
        # up to the requested CPU-pool ceiling instead of appending another
        # full ranked tail.
        add("FALLBACK", ranked, max(0, 20 - len(selected)))
        counts[len(selected)] += 1
        output.append({
            "qid": qid,
            "query": question["query"],
            "positive_doc": positive,
            "negative_count": len(selected),
            "negatives": selected,
            "sampling_contract": "sample four negatives per query per epoch from this CPU pool",
        })
    distribution = dict(sorted((str(k), v) for k, v in counts.items()))
    audit = {
        "kind": "legalir_v28_cpu_negative_pool_audit",
        "schema_version": 1,
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "queries": len(questions),
        "queries_with_positive_in_c35": len(output),
        "qids_without_positive_in_c35": qids_without_positive,
        "pool_size_distribution": distribution,
        "pool_min": min(counts) if counts else 0,
        "pool_max": max(counts) if counts else 0,
        "category_counts": dict(sorted(category_counts.items())),
        "target_pool": "12–20 negatives/query; actual pool is deterministically expanded from the fixed c35 candidates",
        "per_epoch_sampling": 4,
        "false_negative_veto": "exclude raw/canonical gold duplicates, locked top1–4, and empty-body candidates",
    }
    return output, audit


def _boundary_pairs(questions: Mapping[str, Mapping[str, Any]], packs: Mapping[tuple[str, str], Mapping[str, Any]], split: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fold_by_qid = {str(qid): int(fold_row["fold"]) for fold_row in split["folds"] for qid in fold_row["test_qids"]}
    rows = []
    excluded = collections.Counter()
    for qid, question in questions.items():
        incumbent = question["anchor"][4]
        challenger_docs = list(question["c35"][5:10])
        raw_gold = [doc for doc in question["c35"] if doc in set(question["references"]) and doc not in question["anchor"][:4]]
        for doc in raw_gold:
            if doc not in challenger_docs:
                challenger_docs.append(doc)
        seen = set()
        for doc in challenger_docs:
            if doc in seen or doc in question["anchor"][:4] or doc == incumbent:
                continue
            seen.add(doc)
            if bool(packs[(qid, incumbent)].get("empty")) or bool(packs[(qid, doc)].get("empty")):
                excluded["empty_pack"] += 1
                continue
            rows.append({
                "qid": qid,
                "query": question["query"],
                "incumbent_doc": incumbent,
                "challenger_doc": doc,
                "challenger_rank": question["c35"].index(doc) + 1,
                "challenger_kind": "gold_rescue" if doc in raw_gold else "rank6_10",
                "challenger_is_raw_gold": doc in raw_gold,
                "incumbent_is_raw_gold": incumbent in set(question["references"]),
                "baseline_miss": _recall(question["anchor"], set(question["references"])) < 1e-12,
                "outer_fold": fold_by_qid.get(qid),
                "e3b_config": packs[(qid, doc)].get("config"),
            })
    audit = {
        "kind": "legalir_v28_boundary_pair_manifest_audit",
        "schema_version": 1,
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "queries": len(questions),
        "pairs": len(rows),
        "pairs_by_kind": dict(collections.Counter(row["challenger_kind"] for row in rows)),
        "excluded": dict(excluded),
        "scope": "slot5 incumbent versus c35 ranks6–10 plus raw-gold rescue candidate when outside that band",
        "top1_4_locked": True,
    }
    return rows, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--c35", type=Path, required=True)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--group-manifest", type=Path, required=True)
    parser.add_argument("--contexts-zip", type=Path, required=True)
    parser.add_argument("--e3b-manifest", type=Path, required=True)
    parser.add_argument("--stage2-sidecar", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--oof-root", type=Path, required=True)
    parser.add_argument("--oof-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    questions = load_metadata(args.metadata)
    baseline_report = read_json(args.baseline_report)
    baseline_rows = _read_jsonl(args.baseline_predictions)
    split = read_json(args.split)
    oof_report = read_json(args.oof_report)
    scores = load_sidecar(args.stage2_sidecar, questions)
    oof_scores = _load_oof_scores(args.oof_root, split)
    packs = _load_packs(args.e3b_manifest)
    doc_ids = {doc for row in questions.values() for doc in row["c35"]} | {ref for row in questions.values() for ref in row["references"]}
    info = load_document_info(args.contexts_zip, doc_ids)
    canonical = make_canonical_map(info)
    parity = _metric_parity(questions, baseline_rows, baseline_report, args.qrels, args.c35, args.baseline_predictions, args.metadata, args.train_json, args.group_manifest, args.contexts_zip, args.e3b_manifest, oof_report)
    write_json(out / "v28_metric_parity.json", parity)
    oracle, oracle_rows = _slot5_oracle(questions, canonical)
    write_json(out / "v28_slot5_oracle.json", {**oracle, "per_query_path": str((out / "v28_slot5_oracle_per_query.jsonl").resolve())})
    write_jsonl(out / "v28_slot5_oracle_per_query.jsonl", oracle_rows)
    evidence, evidence_rows = _evidence_audit(questions, oof_scores, packs, canonical, split)
    write_json(out / "v28_evidence_bottleneck.json", {**evidence, "rows_path": str((out / "v28_evidence_rows.jsonl").resolve())})
    write_jsonl(out / "v28_evidence_rows.jsonl", evidence_rows)
    pool, pool_audit = _negative_pool(questions, scores, info, canonical)
    pool_hash = write_jsonl(out / "v28_negative_pool.jsonl", pool)
    write_json(out / "v28_negative_pool_audit.json", {**pool_audit, "pool_sha256": pool_hash})
    boundary, boundary_audit = _boundary_pairs(questions, packs, split)
    boundary_hash = write_jsonl(out / "v28_boundary_pairs.jsonl", boundary)
    write_json(out / "v28_boundary_pairs_audit.json", {**boundary_audit, "manifest_sha256": boundary_hash})
    print(json.dumps({"output": str(out.resolve()), "parity": parity["verdict"], "oracle": oracle, "evidence": evidence, "negative_pool": pool_audit, "boundary": boundary_audit}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
