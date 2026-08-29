"""Benchmark and apply anchor-consistent exact-heading tail rescues.

The public v49/legal2 gains showed that long verbatim legal headings are a
high-precision signal which the fused ranker can leave just outside the top
five.  This script measures that signal on labelled fit/holdout caches and
then applies frozen policies directly to the *actual* public anchor.  It never
assumes that ``production_fusion[:5]`` equals the public submission.

Candidate documents are limited to the top ``candidate_cutoff`` of existing
leak-free lexical sources.  A fifth-slot replacement is accepted only when:

* the candidate contains a contiguous query phrase of at least N tokens; and
* that phrase is strictly longer than the best phrase in the anchor tail.

The output includes a fit-selected policy plus explicit conservative and
balanced policies.  Holdout metrics are reported, never used to select N.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from evaluate import official_metrics


VIETNAMESE = "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
ASCII = "aaaaaaaaaaaaaaaaaeeeeeeeeeeeiiiiiooooooooooooooooouuuuuuuuuuuyyyyyd"
FOLD_TABLE = str.maketrans(
    VIETNAMESE + VIETNAMESE.upper(), ASCII + ASCII.upper()
)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
DEFAULT_SOURCES = (
    "focused_chunk",
    "chunk_phrase",
    "or_bm25",
    "lexical_fusion",
    "production_fusion",
)
POLICY_THRESHOLDS = tuple(range(10, 5, -1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-cache", type=Path, required=True)
    parser.add_argument("--holdout-cache", type=Path, required=True)
    parser.add_argument("--public-cache", type=Path, required=True)
    parser.add_argument("--public-anchor", type=Path, required=True)
    parser.add_argument("--contexts-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-cutoff", type=int, default=12)
    parser.add_argument("--conservative-ngram", type=int, default=10)
    parser.add_argument("--balanced-ngram", type=int, default=7)
    return parser.parse_args()


def normalize(value: str) -> str:
    return " ".join(TOKEN_PATTERN.findall(str(value).casefold().translate(FOLD_TABLE)))


def load_cache(path: Path, require_gold: bool) -> list[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        header = json.loads(next(stream))
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"Empty rank cache: {path}")
    qids = [str(row.get("question_id")) for row in rows]
    if len(qids) != len(set(qids)):
        raise ValueError(f"Duplicate question IDs in {path}")
    if require_gold and any(not row.get("gold") for row in rows):
        raise ValueError(f"Labelled cache contains empty gold answers: {path}")
    return rows


def load_submission(path: Path) -> Dict[str, Dict[str, list[str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    output: Dict[str, Dict[str, list[str]]] = {}
    for qid, item in payload.items():
        answers = [str(value) for value in item.get("answer", [])]
        if not 0 < len(answers) <= 5 or len(answers) != len(set(answers)):
            raise ValueError(f"Invalid anchor answer list for {qid}: {answers}")
        output[str(qid)] = {"answer": answers}
    return output


def load_corpus(path: Path) -> tuple[Dict[str, str], Dict[str, str]]:
    """Load one normalized full text and title per official context ID."""
    documents: Dict[str, str] = {}
    titles: Dict[str, str] = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if not info.filename.endswith(".json"):
                continue
            item = json.loads(archive.read(info))
            doc_id = str(item["id"])
            if doc_id in documents:
                raise ValueError(f"Duplicate context ID in archive: {doc_id}")
            documents[doc_id] = normalize(item.get("passage", ""))
            titles[doc_id] = str(item.get("name", ""))
    if not documents:
        raise ValueError(f"No context JSON files in {path}")
    return documents, titles


def longest_match(question: str, document: str, maximum: int = 10) -> tuple[int, str]:
    tokens = normalize(question).split()
    for size in range(min(maximum, len(tokens)), 1, -1):
        for start in range(len(tokens) - size + 1):
            phrase = " ".join(tokens[start:start + size])
            if phrase in document:
                return size, phrase
    return 0, ""


def source_candidates(
    row: Mapping[str, Any], cutoff: int
) -> tuple[Dict[str, int], Dict[str, list[str]]]:
    rankings = row.get("source_rankings", {})
    missing = [source for source in DEFAULT_SOURCES if source not in rankings]
    if missing:
        raise ValueError(
            f"{row.get('question_id')}: cache lacks required sources {missing}"
        )
    best_rank: Dict[str, int] = {}
    selected: Dict[str, list[str]] = {}
    for source in DEFAULT_SOURCES:
        values = [str(value) for value in rankings[source][:cutoff]]
        selected[source] = values
        for rank, doc_id in enumerate(values, 1):
            best_rank[doc_id] = min(best_rank.get(doc_id, cutoff + 1), rank)
    return best_rank, selected


def baseline_from_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, list[str]]]:
    baseline: Dict[str, Dict[str, list[str]]] = {}
    for row in rows:
        qid = str(row["question_id"])
        answers = [
            str(value)
            for value in row["source_rankings"]["production_fusion"][:5]
        ]
        if len(answers) != 5 or len(answers) != len(set(answers)):
            raise ValueError(f"{qid}: invalid production baseline {answers}")
        baseline[qid] = {"answer": answers}
    return baseline


def proposal(
    row: Mapping[str, Any],
    anchor_answers: Sequence[str],
    documents: Mapping[str, str],
    titles: Mapping[str, str],
    cutoff: int,
) -> Dict[str, Any] | None:
    if len(anchor_answers) != 5:
        return None
    anchor = [str(value) for value in anchor_answers]
    tail = anchor[-1]
    ranks, source_lists = source_candidates(row, cutoff)
    old_size, old_phrase = longest_match(
        str(row["question"]), documents.get(tail, "")
    )

    candidates: list[tuple[int, int, str, str]] = []
    for doc_id, rank in ranks.items():
        if doc_id in anchor:
            continue
        size, phrase = longest_match(
            str(row["question"]), documents.get(doc_id, "")
        )
        if size:
            candidates.append((size, -rank, doc_id, phrase))
    if not candidates:
        return None
    size, negative_rank, doc_id, phrase = max(candidates)
    return {
        "question_id": str(row["question_id"]),
        "question": str(row["question"]),
        "old_id": tail,
        "new_id": doc_id,
        "old_match_tokens": old_size,
        "old_match_phrase": old_phrase,
        "new_match_tokens": size,
        "new_match_phrase": phrase,
        "best_source_rank": -negative_rank,
        "old_title": titles.get(tail, ""),
        "new_title": titles.get(doc_id, ""),
        "candidate_cutoff": cutoff,
        "source_rankings": {
            source: values for source, values in source_lists.items()
            if doc_id in values
        },
    }


def build_proposals(
    rows: Sequence[Mapping[str, Any]],
    anchor: Mapping[str, Mapping[str, Sequence[str]]],
    documents: Mapping[str, str],
    titles: Mapping[str, str],
    cutoff: int,
) -> list[Dict[str, Any]]:
    row_ids = {str(row["question_id"]) for row in rows}
    if set(anchor) != row_ids:
        missing = sorted(row_ids - set(anchor))
        extra = sorted(set(anchor) - row_ids)
        raise ValueError(
            f"Anchor/cache question mismatch; missing={missing[:5]} extra={extra[:5]}"
        )
    output: list[Dict[str, Any]] = []
    for position, row in enumerate(rows, 1):
        qid = str(row["question_id"])
        item = proposal(
            row,
            [str(value) for value in anchor[qid]["answer"]],
            documents,
            titles,
            cutoff,
        )
        if item is not None:
            output.append(item)
        if position % 200 == 0 or position == len(rows):
            print(f"Exact-heading proposals: {position}/{len(rows)}", flush=True)
    return output


def apply_policy(
    anchor: Mapping[str, Mapping[str, Sequence[str]]],
    proposals: Iterable[Mapping[str, Any]],
    minimum_ngram: int,
) -> tuple[Dict[str, Dict[str, list[str]]], list[Dict[str, Any]]]:
    output = {
        str(qid): {"answer": [str(value) for value in item["answer"]]}
        for qid, item in anchor.items()
    }
    accepted: list[Dict[str, Any]] = []
    for proposal_row in proposals:
        if int(proposal_row["new_match_tokens"]) < minimum_ngram:
            continue
        if int(proposal_row["new_match_tokens"]) <= int(
            proposal_row["old_match_tokens"]
        ):
            continue
        qid = str(proposal_row["question_id"])
        expected_old = str(proposal_row["old_id"])
        if output[qid]["answer"][-1] != expected_old:
            raise ValueError(f"{qid}: proposal no longer matches anchor tail")
        values = output[qid]["answer"][:-1] + [str(proposal_row["new_id"])]
        if len(values) != 5 or len(values) != len(set(values)):
            raise ValueError(f"{qid}: policy produced invalid answer list {values}")
        output[qid] = {"answer": values}
        accepted.append(dict(proposal_row))
    return output, accepted


def references(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, list[str]]]:
    return {
        str(row["question_id"]): {
            "answer": [str(value) for value in row["gold"]]
        }
        for row in rows
    }


def benchmark_policies(
    rows: Sequence[Mapping[str, Any]],
    anchor: Mapping[str, Mapping[str, Sequence[str]]],
    proposals: Sequence[Mapping[str, Any]],
) -> tuple[Dict[str, Any], Dict[int, Dict[str, Any]]]:
    refs = references(rows)
    baseline_metrics = official_metrics(dict(anchor), refs)
    reports: Dict[int, Dict[str, Any]] = {}
    for threshold in POLICY_THRESHOLDS:
        submission, accepted = apply_policy(anchor, proposals, threshold)
        metrics = official_metrics(submission, refs)
        reports[threshold] = {
            "minimum_ngram": threshold,
            "changes": len(accepted),
            "metrics": metrics,
            "delta": {
                key: metrics[key] - baseline_metrics[key]
                for key in ("recall", "precision")
            },
        }
    return baseline_metrics, reports


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    if args.candidate_cutoff <= 0:
        raise ValueError("--candidate-cutoff must be positive")
    for value in (args.conservative_ngram, args.balanced_ngram):
        if value not in POLICY_THRESHOLDS:
            raise ValueError(f"N-gram threshold must be one of {POLICY_THRESHOLDS}")

    started = time.perf_counter()
    fit_rows = load_cache(args.fit_cache, require_gold=True)
    holdout_rows = load_cache(args.holdout_cache, require_gold=True)
    public_rows = load_cache(args.public_cache, require_gold=False)
    public_anchor = load_submission(args.public_anchor)
    print("Loading and normalizing official contexts...", flush=True)
    documents, titles = load_corpus(args.contexts_zip)

    fit_anchor = baseline_from_rows(fit_rows)
    holdout_anchor = baseline_from_rows(holdout_rows)
    fit_proposals = build_proposals(
        fit_rows, fit_anchor, documents, titles, args.candidate_cutoff
    )
    fit_baseline, fit_reports = benchmark_policies(
        fit_rows, fit_anchor, fit_proposals
    )
    # Policy selection is fit-only.  Prefer fewer changes only when recall
    # deltas tie, making the rule deterministic and independent of holdout.
    chosen_threshold = max(
        POLICY_THRESHOLDS,
        key=lambda threshold: (
            fit_reports[threshold]["delta"]["recall"],
            -fit_reports[threshold]["changes"],
            threshold,
        ),
    )

    holdout_proposals = build_proposals(
        holdout_rows, holdout_anchor, documents, titles, args.candidate_cutoff
    )
    holdout_baseline, holdout_reports = benchmark_policies(
        holdout_rows, holdout_anchor, holdout_proposals
    )
    public_proposals = build_proposals(
        public_rows, public_anchor, documents, titles, args.candidate_cutoff
    )

    policy_names = {
        "fit_selected": chosen_threshold,
        "conservative": args.conservative_ngram,
        "balanced": args.balanced_ngram,
    }
    public_outputs: Dict[str, Any] = {}
    for name, threshold in policy_names.items():
        submission, accepted = apply_policy(
            public_anchor, public_proposals, threshold
        )
        submission_path = args.output_dir / f"sub_v49_legal2_exact_{name}.json"
        audit_path = args.output_dir / f"sub_v49_legal2_exact_{name}.audit.json"
        write_json(submission_path, submission)
        write_json(
            audit_path,
            {
                "kind": "legalir_exact_heading_public_apply",
                "anchor": str(args.public_anchor.resolve()),
                "public_cache": str(args.public_cache.resolve()),
                "minimum_ngram": threshold,
                "candidate_cutoff": args.candidate_cutoff,
                "changed": len(accepted),
                "changes": accepted,
                "public_gold_used": False,
                "anchor_consistent": True,
            },
        )
        public_outputs[name] = {
            "minimum_ngram": threshold,
            "changes": len(accepted),
            "submission": str(submission_path.resolve()),
            "audit": str(audit_path.resolve()),
        }

    report = {
        "kind": "legalir_exact_heading_rescue_benchmark",
        "candidate_cutoff": args.candidate_cutoff,
        "sources": list(DEFAULT_SOURCES),
        "policy_selected_on": "fit only",
        "chosen_threshold": chosen_threshold,
        "fit": {
            "questions": len(fit_rows),
            "baseline": fit_baseline,
            "policies": fit_reports,
        },
        "holdout": {
            "questions": len(holdout_rows),
            "baseline": holdout_baseline,
            "policies": holdout_reports,
        },
        "public": public_outputs,
        "elapsed_seconds": time.perf_counter() - started,
        "contract": (
            "fit-only policy selection; holdout report-only; public labels absent; "
            "replacement compared against actual anchor tail"
        ),
    }
    write_json(args.output_dir / "exact_heading_benchmark_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
