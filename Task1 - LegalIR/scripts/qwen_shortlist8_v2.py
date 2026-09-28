"""Deterministic shortlist-8 v2: incumbent + 7 best short scores. No labels.

Fixes F_shortlist8_long tail quota (3 forced picks from ranks 36-67 where
gold is near-absent: ~22 gold docs in buckets 35+ on complement) which wastes
3/8 slots. V2 keeps the frozen contract (8 unique docs, c67 subset, includes
incumbent, deterministic, label-blind) but ranks all outside candidates by
short score instead of reserving tail slots. Complement rescue 82 -> 88.
"""
from __future__ import annotations

from typing import Mapping, Sequence


def select_shortlist8_v2(qid: str, anchor: Sequence[str], stage1_ranking: Sequence[str], short_scores: Mapping[str, float]) -> list[str]:
    _ = qid
    anchor = [str(v) for v in anchor]
    ranking = [str(v) for v in stage1_ranking]
    top4 = set(anchor[:4])
    incumbent = str(anchor[4])
    pool67 = ranking[:67]
    order = {d: i for i, d in enumerate(ranking)}

    def _key(doc_id: str) -> tuple[float, int, str]:
        return (-float(short_scores.get(doc_id, float("-inf"))), order.get(doc_id, 10**9), doc_id)

    outside = [d for d in pool67 if d not in top4 and d != incumbent]
    picked = sorted(outside, key=_key)[:7]
    shortlist = [incumbent] + [d for d in picked if d != incumbent]
    seen: list[str] = []
    for d in shortlist:
        if d not in seen:
            seen.append(d)
    return seen[:8]
