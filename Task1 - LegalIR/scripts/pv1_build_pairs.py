#!/usr/bin/env python3
"""Build PV1 label-blind pair rows from a pinned F shortlist/packs artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pv1_contract import _manifest_lineage, build_pairs, load_f_shortlist, load_universe, write_pairs
from pv1_contract import json_sha256


def select_challengers(anchor: list[str], shortlist: list[str], scores: dict[str, float]) -> list[tuple[str, int, float]]:
    """Compatibility helper: top-3 challengers by F score, excluding frozen top-4 and incumbent."""
    if len(anchor) != 5:
        raise ValueError("anchor must hold exactly five document IDs")
    frozen = set(anchor[:4])
    incumbent = anchor[4]
    seen: set[str] = set()
    candidates: list[str] = []
    for doc_id in shortlist:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        if doc_id in frozen or doc_id == incumbent:
            continue
        candidates.append(doc_id)
    ranked = sorted(candidates, key=lambda doc: (-float(scores.get(doc, 0.0)), str(doc)))
    return [(doc, index + 1, float(scores.get(doc, 0.0))) for index, doc in enumerate(ranked[:3])]


def build_pair_inputs(*, qid: str, question: str, anchor: list[str], shortlist: list[str], f_scores: dict[str, float], doc_texts: dict[str, str], packer_hash: str) -> list[dict[str, object]]:
    """Compatibility helper: six rows (three challengers by two orientations) with stable hashes."""
    challengers = select_challengers(list(anchor), list(shortlist), dict(f_scores))
    rows: list[dict[str, object]] = []
    for challenger_id, _rank, _score in challengers:
        for orientation in ("inc_first", "chall_first"):
            payload = {
                "qid": str(qid),
                "question": str(question),
                "anchor": [str(value) for value in anchor],
                "challenger_id": str(challenger_id),
                "orientation": str(orientation),
                "packer_hash": str(packer_hash),
                "incumbent_text": str(doc_texts.get(str(anchor[4]), "")),
                "challenger_text": str(doc_texts.get(str(challenger_id), "")),
            }
            rows.append(
                {
                    "qid": str(qid),
                    "challenger_id": str(challenger_id),
                    "orientation": str(orientation),
                    "f_rank": int(_rank),
                    "f_score": float(_score),
                    "input_sha256": json_sha256(payload),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path, required=True); parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True); parser.add_argument("--f-shortlist", type=Path, required=True)
    parser.add_argument("--f-manifest", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    universe = load_universe(stage1=args.stage1, stage2=args.stage2, anchor=args.anchor)
    f_queries = load_f_shortlist(args.f_shortlist, expected_qids=universe.split.qids)
    manifest_input = _manifest_lineage(args.f_manifest, universe=universe, shortlist_path=args.f_shortlist)
    rows, manifest = build_pairs(universe=universe, f_queries=f_queries, f_manifest=manifest_input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_path = args.output_dir / "pv1_pairs.jsonl"; manifest_path = args.output_dir / "pv1_pair_manifest.json"
    write_pairs(pair_path=pair_path, manifest_path=manifest_path, rows=rows, manifest=manifest, force=args.force)
    print(json.dumps({"pairs": str(pair_path.resolve()), "manifest": str(manifest_path.resolve()), "pair_count": len(rows), "labels_opened": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
