#!/usr/bin/env python3
"""F_shortlist8_long 4096 cache: max 8 docs/query from deterministic shortlist.

Complement-only. Shortlist comes from qwen_shortlist8.select_shortlist8
(incumbent + 4 top-short + 3 from 36-67, dedup, fill by short score).
Long scoring reuses the Qwen checkpoint/instruction with 4096 packs.
Own manifest; never presents D_c200_long lineage as F.
"""
from __future__ import annotations

from pathlib import Path

from v27_a_common import V27ContractError

VARIANT = "F_shortlist8_long"
LONG_MAX_TOKENS = 4096
MAX_DOCS_PER_QUERY = 8


def preflight(*, output_dir: Path, short_cache_dir: Path) -> dict:
    for path, label in ((short_cache_dir, "short_cache_dir"), (output_dir, "output_dir")):
        value = str(path).replace("\\", "/").casefold()
        if any(term in value for term in ("holdout", "public", "submission", "locked")):
            raise V27ContractError(f"F cache is complement-only; {label} may not refer to holdout/public: {path}")
    return {
        "variant": VARIANT,
        "max_tokens": LONG_MAX_TOKENS,
        "max_docs_per_query": MAX_DOCS_PER_QUERY,
        "holdout_opened": False,
        "public_labels_used": False,
    }
