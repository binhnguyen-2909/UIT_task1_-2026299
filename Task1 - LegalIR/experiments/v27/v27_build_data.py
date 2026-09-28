"""Create V27 grouped query-positive/hard-negative training examples."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from v27_common import (
    load_document_info,
    load_metadata,
    load_sidecar,
    make_canonical_map,
    sha256_file,
    text_tokens,
    write_json,
    write_jsonl,
)


def load_packs(path: Path, questions: dict[str, dict]) -> dict[tuple[str, str], dict]:
    packs: dict[tuple[str, str], dict] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["qid"]), str(row["doc_id"]))
            if key in packs:
                raise ValueError(f"duplicate E3B pack at line {line_number}: {key}")
            if str(row.get("config")) != "B2304-D60":
                raise ValueError(f"invalid/empty canonical E3B pack at line {line_number}")
            if not bool(row.get("empty")) and not str(row.get("pack", "")).strip():
                raise ValueError(f"invalid empty=false canonical E3B pack at line {line_number}")
            packs[key] = row
    expected = {(qid, doc) for qid, question in questions.items() for doc in question["c35"]}
    if set(packs) != expected:
        raise ValueError(f"E3B pack coverage mismatch: got={len(packs)} expected={len(expected)}")
    return packs


def pick_top5(candidates: list[str], scores: dict[str, float], forbidden: set[str]) -> tuple[str | None, bool]:
    for index, doc in enumerate(candidates[:5]):
        if doc not in forbidden:
            return doc, True
    for doc in sorted(candidates, key=lambda value: (-scores[value], candidates.index(value))):
        if doc not in forbidden:
            return doc, False
    return None, False


def pick_neural(candidates: list[str], scores: dict[str, float], forbidden: set[str]) -> tuple[str | None, bool]:
    ordered = sorted(candidates, key=lambda value: (-scores[value], candidates.index(value)))
    band = [doc for doc in ordered[5:15] if doc not in forbidden]
    if band:
        return band[0], True
    return next((doc for doc in ordered if doc not in forbidden), None), False


def pick_same_law(candidates: list[str], scores: dict[str, float], doc_info: dict[str, dict], positives: list[str], forbidden: set[str]) -> tuple[str | None, bool]:
    positive_rows = [doc_info[doc] for doc in positives if doc in doc_info]
    positive_families = {str(row["law_family"]) for row in positive_rows if str(row["law_family"])}
    positive_titles = [set(row["title_tokens"]) for row in positive_rows]
    ranked = sorted(candidates, key=lambda value: (-scores[value], candidates.index(value)))
    scored: list[tuple[bool, float, float, int, str]] = []
    for doc in ranked:
        if doc in forbidden:
            continue
        row = doc_info[doc]
        family = str(row["law_family"])
        overlap = max((len(set(row["title_tokens"]) & tokens) / max(len(set(row["title_tokens"]) | tokens), 1) for tokens in positive_titles), default=0.0)
        scored.append((bool(family and family in positive_families), overlap, float(scores[doc]), -candidates.index(doc), doc))
    if not scored:
        return None, False
    best = max(scored)
    return best[-1], bool(best[0] or best[1] >= 0.15)


def pick_medium(candidates: list[str], scores: dict[str, float], forbidden: set[str]) -> tuple[str | None, bool]:
    ordered = sorted(candidates, key=lambda value: (-scores[value], candidates.index(value)))
    band = [doc for doc in ordered[15:] if doc not in forbidden]
    if band:
        return band[min(len(band) // 2, len(band) - 1)], True
    return next((doc for doc in reversed(ordered) if doc not in forbidden), None), False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--contexts-zip", type=Path, required=True)
    parser.add_argument("--stage2-sidecar", type=Path, required=True)
    parser.add_argument("--e3b-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    questions = load_metadata(args.metadata)
    scores = load_sidecar(args.stage2_sidecar, questions)
    packs = load_packs(args.e3b_manifest, questions)
    candidate_doc_ids = {doc for question in questions.values() for doc in question["c35"]}
    doc_ids = set(candidate_doc_ids)
    doc_ids.update(ref for question in questions.values() for ref in question["references"])
    doc_info = load_document_info(args.contexts_zip, doc_ids)
    canonical = make_canonical_map(doc_info)
    out = args.output_dir
    (out / "data").mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "data" / "canonical_doc_map.jsonl", [{"doc_id": doc, "canonical_doc_id": canonical[doc], "body_hash": doc_info[doc]["body_hash"]} for doc in sorted(doc_info)])

    rows = []
    type_counts = Counter()
    fallback_counts = Counter()
    missing_positive = []
    false_negative_checks = 0
    for qid, question in questions.items():
        candidates = list(question["c35"])
        q_scores = {doc: float(scores[(qid, doc)]) for doc in candidates}
        candidate_order = {doc: index for index, doc in enumerate(candidates)}
        gold_canonical = {canonical[doc] for doc in question["references"] if doc in canonical}
        gold = sorted((doc for doc in candidates if canonical[doc] in gold_canonical), key=lambda doc: candidate_order[doc])
        if not gold:
            missing_positive.append(qid)
            continue
        for positive in gold:
            if not packs[(qid, positive)].get("pack", "").strip():
                continue
            forbidden = {doc for doc in candidates if canonical[doc] in gold_canonical}
            forbidden.add(positive)
            forbidden.update(doc for doc in candidates if not packs[(qid, doc)].get("pack", "").strip())
            selectors = (
                ("TOP5_FP", pick_top5),
                ("NEURAL", pick_neural),
                ("SAME_LAW", lambda c, s, f: pick_same_law(c, s, doc_info, [positive], f)),
                ("MEDIUM", pick_medium),
            )
            selected: set[str] = set()
            for negative_type, selector in selectors:
                negative, strict = selector(candidates, q_scores, forbidden | selected)
                if negative is None:
                    raise ValueError(f"{qid}/{positive}: cannot find {negative_type} negative")
                negative_canonical = canonical[negative]
                if negative_canonical in gold_canonical:
                    raise AssertionError(f"false negative {qid}/{negative}: canonical gold collision")
                selected.add(negative)
                if not strict:
                    fallback_counts[negative_type] += 1
                negative_title_tokens = doc_info[negative]["title_tokens"]
                positive_title_tokens = doc_info[positive]["title_tokens"]
                title_overlap = len(negative_title_tokens & positive_title_tokens) / max(len(negative_title_tokens | positive_title_tokens), 1)
                row = {
                    "qid": qid,
                    "query": question["query"],
                    "positive_doc": positive,
                    "negative_doc": negative,
                    "negative_type": negative_type,
                    "positive_pack": packs[(qid, positive)]["pack"],
                    "negative_pack": packs[(qid, negative)]["pack"],
                    "positive_pack_sha256": packs[(qid, positive)]["pack_sha256"],
                    "negative_pack_sha256": packs[(qid, negative)]["pack_sha256"],
                    "canonical_positive_doc": canonical[positive],
                    "canonical_negative_doc": negative_canonical,
                    "negative_stage2_score": q_scores[negative],
                    "negative_rank_by_stage2": sorted(candidates, key=lambda value: (-q_scores[value], candidates.index(value))).index(negative) + 1,
                    "same_law_positive_title_overlap": title_overlap,
                }
                rows.append(row)
                type_counts[negative_type] += 1
                false_negative_checks += 1
    if missing_positive:
        # Keep these qids in the frozen evaluation universe, but never invent
        # a positive training label outside the fixed c35 candidate pool.
        pass
    pair_hash = write_jsonl(out / "data" / "v27_pairs.jsonl", rows)
    audit = {
        "kind": "legalir_v27_hard_negative_audit",
        "schema_version": 1,
        "split": "clean_complement_only",
        "holdout_opened": False,
        "public_labels_used": False,
        "queries": len(questions),
        "unique_candidate_documents": len(candidate_doc_ids),
        "positive_documents_in_c35": sum(sum(canonical[doc] in {canonical[ref] for ref in question["references"] if ref in canonical} for doc in question["c35"]) for question in questions.values()),
        "qids_without_positive_in_c35": missing_positive,
        "training_rows": len(rows),
        "rows_by_negative_type": dict(type_counts),
        "fallback_rows_by_type": dict(fallback_counts),
        "false_negative_assertions": false_negative_checks,
        "canonical_duplicate_groups": len(set(canonical.values())),
        "inputs": {"metadata": sha256_file(args.metadata), "contexts_zip": sha256_file(args.contexts_zip), "stage2_sidecar": sha256_file(args.stage2_sidecar), "e3b_manifest": sha256_file(args.e3b_manifest)},
        "pairs_sha256": pair_hash,
        "negative_mix_target": {"TOP5_FP": 0.25, "NEURAL": 0.25, "SAME_LAW": 0.25, "MEDIUM": 0.25},
        "contract": {"all_gold_in_c35_are_positive": True, "qids_without_positive_are_not_trained": True, "canonical_duplicate_false_negatives_forbidden": True, "query_relation_preserved": True},
    }
    write_json(out / "reports" / "hard_negative_audit.json", audit)
    print(json.dumps({"pairs": len(rows), "queries": len(questions), "types": dict(type_counts), "fallbacks": dict(fallback_counts), "sha256": pair_hash}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
