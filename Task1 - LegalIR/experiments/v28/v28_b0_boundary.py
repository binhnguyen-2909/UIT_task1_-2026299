"""CPU legal-element boundary diagnostic for V28-B0."""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from v27_common import load_metadata, normalized_text, read_json, text_tokens, write_json, write_jsonl


ELEMENT_PATTERNS = {
    "subject": ("cá nhân", "tổ chức", "người lao động", "người sử dụng lao động", "doanh nghiệp", "cơ quan nhà nước", "người chưa thành niên", "người điều khiển phương tiện", "pháp nhân", "cơ quan", "cơ sở"),
    "action": ("thực hiện", "không thực hiện", "từ chối", "chậm thực hiện", "sử dụng", "chiếm hữu", "kinh doanh", "vận chuyển", "cung cấp", "nộp", "đăng ký", "cấp", "xử lý", "ra quyết định", "yêu cầu", "vi phạm", "hành nghề"),
    "object": ("hồ sơ", "giấy tờ", "tài liệu", "chứng chỉ", "giấy phép", "văn bản", "quyền", "nghĩa vụ", "khoản tiền", "tài sản", "phương tiện", "hợp đồng"),
    "condition": ("khi ", "trong trường hợp", "nếu ", "đáp ứng", "với điều kiện", "đủ điều kiện", "trường hợp", "được phép", "không được"),
    "exception": ("trừ trường hợp", "ngoại trừ", "không áp dụng", "trừ khi", "không bao gồm", "trừ "),
    "modality": ("phải", "được", "có thể", "không được", "bị cấm", "có quyền", "có nghĩa vụ", "được quyền", "không phải"),
    "authority": ("chủ tịch", "tòa án", "toà án", "bộ trưởng", "bộ ", "sở ", "thanh tra", "công an", "ủy ban", "uỷ ban", "thẩm phán", "cơ quan có thẩm quyền", "cơ quan nhà nước"),
}

NUMBER_RE = re.compile(r"(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>ngày|tháng|năm|giờ|tuần|tuổi|%|triệu|tỷ|đồng|mét|km)", re.IGNORECASE)
RANGE_RE = re.compile(r"(?P<lo>\d+(?:[.,]\d+)?)\s*(?:đến|tới|-|–)\s*(?P<hi>\d+(?:[.,]\d+)?)\s*(?P<unit>ngày|tháng|năm|tuổi|%|triệu|tỷ|đồng)", re.IGNORECASE)
CITATION_RES = (
    re.compile(r"\b(?:luật|nghị định|thông tư|quyết định|pháp lệnh|bộ luật)\b[^\n,;.]{0,100}", re.IGNORECASE),
    re.compile(r"\bđiều\s+\d+[a-zđ]?", re.IGNORECASE),
    re.compile(r"\bkhoản\s+\d+", re.IGNORECASE),
    re.compile(r"\bđiểm\s+[a-zđ]", re.IGNORECASE),
)

WEIGHTS = {
    "subject": 0.8,
    "action": 0.8,
    "condition": 1.0,
    "exception": 1.2,
    "numeric": 1.0,
    "modality": 1.5,
    "authority": 0.6,
    "citation": 0.6,
}
PENALTIES = {"subject": 1.5, "exception": 2.0, "numeric": 1.5, "modality": 1.5, "authority": 0.8, "citation": 0.8}


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _snip(value: str, term: str) -> str:
    text = " ".join(str(value).split())
    pos = text.casefold().find(term.casefold())
    if pos < 0:
        return text[:180]
    return text[max(0, pos - 60) : pos + 160]


def extract_signature(text: str) -> dict[str, Any]:
    raw = str(text)
    norm = normalized_text(raw)
    signature: dict[str, Any] = {key: [] for key in (*ELEMENT_PATTERNS, "numeric", "citation", "article", "clause")}
    for key, patterns in ELEMENT_PATTERNS.items():
        signature[key] = [pattern.strip() for pattern in patterns if normalized_text(pattern) in norm]
    numeric = []
    for match in RANGE_RE.finditer(raw):
        numeric.append({"kind": "range", "low": float(match.group("lo").replace(",", ".")), "high": float(match.group("hi").replace(",", ".")), "unit": match.group("unit").casefold(), "text": match.group(0)})
    for match in NUMBER_RE.finditer(raw):
        numeric.append({"kind": "value", "value": float(match.group("value").replace(",", ".")), "unit": match.group("unit").casefold(), "text": match.group(0)})
    signature["numeric"] = _unique([json.dumps(row, ensure_ascii=False, sort_keys=True) for row in numeric])
    signature["numeric"] = [json.loads(row) for row in signature["numeric"]]
    citations = []
    for regex in CITATION_RES:
        citations.extend(match.group(0).strip() for match in regex.finditer(raw))
    signature["citation"] = _unique(citations)
    signature["article"] = _unique([value for value in signature["citation"] if value.casefold().startswith("điều")])
    signature["clause"] = _unique([value for value in signature["citation"] if value.casefold().startswith(("khoản", "điểm"))])
    signature["token_count"] = len(text_tokens(raw))
    return signature


def _numeric_match(query: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> bool:
    if not query:
        return False
    qvals = {(row.get("unit"), row.get("value"), row.get("low"), row.get("high")) for row in query}
    cvals = {(row.get("unit"), row.get("value"), row.get("low"), row.get("high")) for row in candidate}
    return bool(qvals & cvals)


def _numeric_contradiction(query: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> bool:
    if not query or not candidate:
        return False
    qunits = {row.get("unit") for row in query}
    same_unit = [row for row in candidate if row.get("unit") in qunits]
    return bool(same_unit) and not _numeric_match(query, same_unit)


def _signature_features(query_sig: dict[str, Any], candidate_sig: dict[str, Any]) -> tuple[dict[str, int], dict[str, int]]:
    matches = {}
    for key in ELEMENT_PATTERNS:
        matches[key] = int(bool(query_sig[key]) and bool(set(query_sig[key]) & set(candidate_sig[key])))
    matches["numeric"] = int(_numeric_match(query_sig["numeric"], candidate_sig["numeric"]))
    matches["citation"] = int(bool(query_sig["citation"]) and bool(set(query_sig["citation"]) & set(candidate_sig["citation"])))
    contradictions = {
        "subject": int(bool(query_sig["subject"]) and bool(candidate_sig["subject"]) and not bool(set(query_sig["subject"]) & set(candidate_sig["subject"]))),
        "exception": int(bool(query_sig["exception"]) and bool(candidate_sig["exception"]) and not bool(set(query_sig["exception"]) & set(candidate_sig["exception"]))),
        "modality": int(bool(query_sig["modality"]) and bool(candidate_sig["modality"]) and not bool(set(query_sig["modality"]) & set(candidate_sig["modality"]))),
        "numeric": int(_numeric_contradiction(query_sig["numeric"], candidate_sig["numeric"])),
        "authority": int(bool(query_sig["authority"]) and bool(candidate_sig["authority"]) and not bool(set(query_sig["authority"]) & set(candidate_sig["authority"]))),
        "citation": int(bool(query_sig["citation"]) and bool(candidate_sig["citation"]) and not bool(set(query_sig["citation"]) & set(candidate_sig["citation"]))),
    }
    return matches, contradictions


def _rule_margin(bge_margin: float, gold_matches: dict[str, int], incumbent_matches: dict[str, int], gold_contra: dict[str, int], incumbent_contra: dict[str, int]) -> tuple[float, dict[str, float]]:
    components = {"existing_semantic": float(bge_margin)}
    value = float(bge_margin)
    for key, weight in WEIGHTS.items():
        advantage = gold_matches.get(key, 0) - incumbent_matches.get(key, 0)
        contribution = weight * advantage
        components[f"{key}_match"] = contribution
        value += contribution
    for key, penalty in PENALTIES.items():
        advantage = incumbent_contra.get(key, 0) - gold_contra.get(key, 0)
        contribution = penalty * advantage
        components[f"{key}_contradiction"] = contribution
        value += contribution
    return value, components


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--slot5-oracle", type=Path, required=True)
    parser.add_argument("--e3b-manifest", type=Path, required=True)
    parser.add_argument("--a0-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    questions = load_metadata(args.metadata)
    oracle_rows = [json.loads(line) for line in args.slot5_oracle.read_text(encoding="utf-8").splitlines() if line.strip()]
    packs = {(str(row["qid"]), str(row["doc_id"])): row for row in (json.loads(line) for line in args.e3b_manifest.read_text(encoding="utf-8").splitlines() if line.strip())}
    a0_rows = [json.loads(line) for line in args.a0_scores.read_text(encoding="utf-8").splitlines() if line.strip()]
    a0_margin = {(str(row["qid"]), str(row["challenger_doc"])): float(row["margin"]) for row in a0_rows}
    details = []
    element_outcomes = collections.defaultdict(collections.Counter)
    contradiction_counts = collections.Counter()
    for oracle in oracle_rows:
        if not oracle.get("baseline_miss") or not oracle.get("gold_in_raw_c35") or not oracle.get("oracle_slot5_doc"):
            continue
        qid = str(oracle["qid"])
        question = questions[qid]
        gold_doc = str(oracle["oracle_slot5_doc"])
        incumbent_doc = str(question["anchor"][4])
        query_sig = extract_signature(question["query"])
        gold_sig = extract_signature(packs[(qid, gold_doc)].get("pack", ""))
        incumbent_sig = extract_signature(packs[(qid, incumbent_doc)].get("pack", ""))
        gold_match, gold_contra = _signature_features(query_sig, gold_sig)
        incumbent_match, incumbent_contra = _signature_features(query_sig, incumbent_sig)
        margin = float(a0_margin.get((qid, gold_doc), 0.0))
        rule_margin, components = _rule_margin(margin, gold_match, incumbent_match, gold_contra, incumbent_contra)
        row = {
            "qid": qid,
            "query": question["query"],
            "gold_doc": gold_doc,
            "incumbent_doc": incumbent_doc,
            "bge_margin": margin,
            "rule_margin": rule_margin,
            "rule_winner": "gold" if rule_margin > 1e-12 else "incumbent" if rule_margin < -1e-12 else "tie",
            "query_signature": query_sig,
            "gold_signature": gold_sig,
            "incumbent_signature": incumbent_sig,
            "gold_match": gold_match,
            "incumbent_match": incumbent_match,
            "gold_contradiction": gold_contra,
            "incumbent_contradiction": incumbent_contra,
            "rule_components": components,
        }
        details.append(row)
        for key in (*ELEMENT_PATTERNS, "numeric", "citation"):
            g, i = gold_match.get(key, 0), incumbent_match.get(key, 0)
            element_outcomes[key]["gold_wins" if g > i else "incumbent_wins" if i > g else "tie"] += 1
        for key, value in incumbent_contra.items():
            if value and not gold_contra.get(key, 0):
                contradiction_counts[key] += 1
    rule_wins = sum(row["rule_winner"] == "gold" for row in details)
    rule_losses = sum(row["rule_winner"] == "incumbent" for row in details)
    ties = sum(row["rule_winner"] == "tie" for row in details)
    by_element = {}
    for key, counter in sorted(element_outcomes.items()):
        total = sum(counter.values())
        by_element[key] = {**dict(counter), "evaluated": total, "gold_win_rate": counter["gold_wins"] / max(total - counter["tie"], 1)}
    report = {
        "kind": "legalir_v28_b0_legal_boundary_diagnostic",
        "schema_version": 1,
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "method": "CPU structured signature extraction with explicit match/contradiction features; fixed diagnostic weights, no holdout optimization",
        "cases": len(details),
        "overall": {
            "rule_boundary_win_rate": rule_wins / max(rule_wins + rule_losses, 1),
            "gold_wins": rule_wins,
            "incumbent_wins": rule_losses,
            "ties": ties,
            "bge_reference_win_rate": sum(float(row["bge_margin"]) > 0 for row in details) / max(len(details), 1),
        },
        "by_element": by_element,
        "contradiction_cases": dict(sorted(contradiction_counts.items())),
        "gate": {
            "boundary_win_rate_ge_0.70": rule_wins / max(rule_wins + rule_losses, 1) >= 0.70,
            "better_than_bge_0.588": rule_wins / max(rule_wins + rule_losses, 1) > 0.5882352941176471,
            "decision": "continue_to_v28_b1_e4b" if rule_wins / max(rule_wins + rule_losses, 1) >= 0.70 or rule_wins / max(rule_wins + rule_losses, 1) > 0.5882352941176471 else "close_legal_element_lane_and_prioritize_candidate_recovery",
        },
        "weights": {"match": WEIGHTS, "contradiction_penalty": PENALTIES},
        "diagnostic_caveat": "automatic extractor; inspect v28_b0_boundary_cases.jsonl before treating a feature as legally decisive",
    }
    write_json(out / "v28_b0_legal_boundary_report.json", report)
    write_jsonl(out / "v28_b0_boundary_cases.jsonl", details)
    print(json.dumps({"report": str((out / "v28_b0_legal_boundary_report.json").resolve()), "overall": report["overall"], "gate": report["gate"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
