"""CPU-only audit lock for new_gold definition. No GPU, no labels beyond complement."""
from __future__ import annotations


def classify_swap(inc_is_gold: bool, cand_is_gold: bool) -> str:
    if (not inc_is_gold) and cand_is_gold:
        return "rescue"
    if inc_is_gold and (not cand_is_gold):
        return "destructive"
    return "neutral"
