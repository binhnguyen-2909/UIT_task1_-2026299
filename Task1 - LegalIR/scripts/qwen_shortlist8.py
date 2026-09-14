"""Deterministic shortlist-8 for long scoring. No labels consulted."""
from __future__ import annotations
from typing import Mapping, Sequence

def _short_rank(doc_id: str, scores: Mapping[str, float], stage1_ranking: Sequence[str]) -> tuple[float, int, str]:
    return (-float(scores.get(doc_id, float("-inf"))), list(stage1_ranking).index(doc_id), doc_id)

def select_shortlist8(qid: str, anchor: Sequence[str], stage1_ranking: Sequence[str], short_scores: Mapping[str, float]) -> list[str]:
    anchor = [str(v) for v in anchor]
    ranking = [str(v) for v in stage1_ranking]
    top4 = set(anchor[:4])
    incumbent = str(anchor[4])
    pool67 = ranking[:67]
    outside = [d for d in pool67 if d not in top4 and d != incumbent]
    top4_outside = sorted(outside, key=lambda d: _short_rank(d, short_scores, ranking))[:4]
    picked = set(top4_outside) | {incumbent}
    tail = [d for d in ranking[35:67] if d not in top4 and d not in picked]
    tail3 = sorted(tail, key=lambda d: _short_rank(d, short_scores, ranking))[:3]
    shortlist = [incumbent] + [d for d in top4_outside if d != incumbent] + [d for d in tail3 if d not in picked and d != incumbent]
    if len(shortlist) < 8:
        rest = sorted((d for d in outside if d not in set(shortlist)), key=lambda d: _short_rank(d, short_scores, ranking))
        shortlist += rest[:8 - len(shortlist)]
    seen: list[str] = []
    for d in shortlist:
        if d not in seen:
            seen.append(d)
    return seen[:8]

def oracle_slot5_shortlist8(split: object, shortlists: Mapping[str, Sequence[str]]) -> dict:
    import numpy as np
    rows: list[float] = []
    absent = 0
    for index, qid in enumerate(split.qids):
        gold = set(split.references[index])
        anchor = list(split.anchor[index])
        pool = [str(v) for v in shortlists[str(qid)]]
        cands = [d for d in pool if d not in set(anchor[:4])]
        best = max((float(d in gold) - float(anchor[4] in gold)) / len(gold) for d in cands)
        rows.append(len(set(anchor) & gold) / len(gold) + best)
        absent += int(not (set(pool) & gold))
    return {"slot5_oracle_recall": float(np.mean(rows)), "queries_without_gold_in_pool": absent, "queries": len(rows)}
