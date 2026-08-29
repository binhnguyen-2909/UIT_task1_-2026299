"""Apply v57's exact-heading idea to v11 candidates only.

The script consumes a v11 Stage-1 candidate audit and a v11 anchor submission.
It never reads v57 rank caches, frozen predictions, or model outputs.  A
candidate can replace only the fifth answer when its exact normalized query
phrase is longer than the current fifth answer's phrase.  Threshold selection
is fit-only; holdout is report-only.
"""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

try:
    from .evaluate import official_metrics
except ImportError:
    from evaluate import official_metrics


VIETNAMESE = "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
ASCII = "aaaaaaaaaaaaaaaaaeeeeeeeeeeeiiiiiooooooooooooooooouuuuuuuuuuuyyyyyd"
FOLD_TABLE = str.maketrans(VIETNAMESE + VIETNAMESE.upper(), ASCII + ASCII.upper())


def normalize(value: str) -> str:
    import re

    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold().translate(FOLD_TABLE)))


def load_submission(path: Path) -> Dict[str, Dict[str, list[str]]]:
    if path.suffix.casefold() == ".zip":
        with zipfile.ZipFile(path) as archive:
            payload = json.loads(archive.read("submission.json"))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(qid): {"answer": [str(value) for value in item["answer"]]}
        for qid, item in payload.items()
    }


def load_references(path: Path, qids: Iterable[str]) -> Dict[str, Dict[str, list[str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(qid): {"answer": [str(value) for value in payload[str(qid)]["answer"]]}
        for qid in qids
    }


def load_contexts(path: Path) -> tuple[Dict[str, str], Dict[str, str]]:
    documents: Dict[str, str] = {}
    titles: Dict[str, str] = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if not info.filename.endswith(".json"):
                continue
            item = json.loads(archive.read(info))
            doc_id = str(item["id"])
            documents[doc_id] = normalize(item.get("passage", ""))
            titles[doc_id] = str(item.get("name", ""))
    return documents, titles


def load_stage1(path: Path) -> tuple[list[Dict[str, Any]], list[list[str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    queries = payload.get("queries")
    candidates = payload.get("stage1_candidates")
    if not isinstance(queries, list) or not isinstance(candidates, list):
        raise ValueError("v11 Stage-1 audit must contain queries and stage1_candidates")
    if len(queries) != len(candidates):
        raise ValueError("Stage-1 queries/candidates lengths differ")
    return queries, [[str(value) for value in row] for row in candidates]


def longest_match_tokens(tokens: Sequence[str], document: str, maximum: int = 12) -> tuple[int, str]:
    for size in range(min(maximum, len(tokens)), 1, -1):
        for start in range(len(tokens) - size + 1):
            phrase = " ".join(tokens[start:start + size])
            if phrase in document:
                return size, phrase
    return 0, ""


def longest_match(question: str, document: str, maximum: int = 12) -> tuple[int, str]:
    return longest_match_tokens(normalize(question).split(), document, maximum)


def build_proposals(
    queries: Sequence[Mapping[str, Any]],
    candidates: Sequence[Sequence[str]],
    anchor: Mapping[str, Mapping[str, Sequence[str]]],
    documents: Mapping[str, str],
    titles: Mapping[str, str],
    candidate_cutoff: int,
) -> list[Dict[str, Any]]:
    proposals: list[Dict[str, Any]] = []
    if set(anchor) != {str(item["query_id"]) for item in queries}:
        raise ValueError("Anchor and Stage-1 audit question IDs differ")
    for position, item in enumerate(queries, 1):
        qid = str(item["query_id"])
        question = str(item.get("query", item.get("question", "")))
        question_tokens = normalize(question).split()
        answers = [str(value) for value in anchor[qid]["answer"]]
        if len(answers) != 5 or len(answers) != len(set(answers)):
            raise ValueError(f"{qid}: anchor must contain exactly five unique IDs")
        tail = answers[-1]
        old_size, old_phrase = longest_match_tokens(question_tokens, documents.get(tail, ""))
        ranked = list(dict.fromkeys(candidates[position - 1]))[:candidate_cutoff]
        best: tuple[int, int, str, str] | None = None
        for rank, doc_id in enumerate(ranked, 1):
            if doc_id in answers:
                continue
            size, phrase = longest_match_tokens(question_tokens, documents.get(doc_id, ""))
            current = (size, -rank, doc_id, phrase)
            if size and (best is None or current > best):
                best = current
        if best and best[0] > 0:
            size, negative_rank, doc_id, phrase = best
            proposals.append({
                "question_id": qid,
                "question": question,
                "old_id": tail,
                "new_id": doc_id,
                "old_match_tokens": old_size,
                "old_match_phrase": old_phrase,
                "new_match_tokens": size,
                "new_match_phrase": phrase,
                "best_candidate_rank": -negative_rank,
                "old_title": titles.get(tail, ""),
                "new_title": titles.get(doc_id, ""),
            })
        if position % 500 == 0 or position == len(queries):
            print(f"v11 exact-heading proposals: {position}/{len(queries)}", flush=True)
    return proposals


def apply_policy(anchor: Mapping[str, Mapping[str, Sequence[str]]], proposals: Sequence[Mapping[str, Any]], threshold: int):
    output = {str(qid): {"answer": [str(value) for value in item["answer"]]} for qid, item in anchor.items()}
    accepted = []
    for item in proposals:
        if int(item["new_match_tokens"]) < threshold or int(item["new_match_tokens"]) <= int(item["old_match_tokens"]):
            continue
        qid = str(item["question_id"])
        if output[qid]["answer"][-1] != str(item["old_id"]):
            raise ValueError(f"{qid}: anchor tail changed before policy application")
        values = output[qid]["answer"][:-1] + [str(item["new_id"])]
        if len(values) != 5 or len(values) != len(set(values)):
            raise ValueError(f"{qid}: invalid one-slot replacement")
        output[qid] = {"answer": values}
        accepted.append(dict(item))
    return output, accepted


def evaluate_split(audit_path: Path, anchor_path: Path, reference_path: Path, contexts, cutoff: int):
    queries, candidates = load_stage1(audit_path)
    anchor = load_submission(anchor_path)
    documents, titles = contexts
    proposals = build_proposals(queries, candidates, anchor, documents, titles, cutoff)
    references = load_references(reference_path, anchor)
    baseline = official_metrics(anchor, references)
    policies = {}
    for threshold in range(10, 5, -1):
        submission, accepted = apply_policy(anchor, proposals, threshold)
        metrics = official_metrics(submission, references)
        policies[threshold] = {
            "changes": len(accepted),
            "metrics": metrics,
            "delta": {key: metrics[key] - baseline[key] for key in ("recall", "precision")},
        }
    chosen = max(range(10, 5, -1), key=lambda value: (policies[value]["delta"]["recall"], -policies[value]["changes"], value))
    return {
        "questions": len(queries),
        "baseline": baseline,
        "proposals": len(proposals),
        "policies": policies,
        "chosen_threshold": chosen,
    }, (queries, candidates, anchor, proposals, titles)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts-zip", type=Path, required=True)
    parser.add_argument("--fit-audit", type=Path, required=True)
    parser.add_argument("--fit-anchor", type=Path, required=True)
    parser.add_argument("--fit-reference", type=Path, required=True)
    parser.add_argument("--holdout-audit", type=Path, required=True)
    parser.add_argument("--holdout-anchor", type=Path, required=True)
    parser.add_argument("--holdout-reference", type=Path, required=True)
    parser.add_argument("--public-audit", type=Path, required=True)
    parser.add_argument("--public-anchor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-cutoff", type=int, default=200)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    contexts = load_contexts(args.contexts_zip)
    fit_report, _ = evaluate_split(args.fit_audit, args.fit_anchor, args.fit_reference, contexts, args.candidate_cutoff)
    holdout_report, _ = evaluate_split(args.holdout_audit, args.holdout_anchor, args.holdout_reference, contexts, args.candidate_cutoff)
    chosen = int(fit_report["chosen_threshold"])
    public_queries, public_candidates = load_stage1(args.public_audit)
    public_anchor = load_submission(args.public_anchor)
    documents, titles = contexts
    public_proposals = build_proposals(public_queries, public_candidates, public_anchor, documents, titles, args.candidate_cutoff)
    public_submission, accepted = apply_policy(public_anchor, public_proposals, chosen)
    output_submission = args.output_dir / f"v11_exact_heading_t{chosen}.json"
    output_submission.write_text(json.dumps(public_submission, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "kind": "legalir_v11_exact_heading_rescue",
        "candidate_cutoff": args.candidate_cutoff,
        "policy_selected_on": "v11 fit only",
        "chosen_threshold": chosen,
        "fit": fit_report,
        "holdout": holdout_report,
        "public": {"questions": len(public_queries), "proposals": len(public_proposals), "changes": len(accepted), "submission": str(output_submission.resolve()), "v57_prediction_used": False},
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "v11_exact_heading_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
