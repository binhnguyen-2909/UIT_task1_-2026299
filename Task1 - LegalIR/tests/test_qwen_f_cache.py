"""F_shortlist8_long contract: own lineage, deterministic shortlist, 4096 packs."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

TASK_ROOT = Path("Task1 - LegalIR")
sys.path.insert(0, str(Path("Task1 - LegalIR/scripts")))

import qwen_shortlist8_long_cache as F  # noqa: E402


def test_lineage_is_own_never_d():
    assert F.VARIANT == "F_shortlist8_long"
    assert F.VARIANT != "D_c200_long"
    assert F.LONG_MAX_TOKENS == 4096
    assert F.MAX_DOCS_PER_QUERY == 8
    assert "D_c200" not in F.SHORTLIST_POLICY


def _universe_scores():
    from qwen_c200_common import load_candidate_universe, pool_for, read_score_cache
    U = load_candidate_universe(
        stage1=TASK_ROOT / "output/v10_9_prf/complement_safe_tail_stage1.audit.json",
        stage2=TASK_ROOT / "output/v10_9_prf/complement_safe_tail_stage2_c35_p3.audit.json",
        anchor=TASK_ROOT / "output/v16_rgm/complement_v16b_memory_slot5.json",
        reference=None,
    )
    rows = read_score_cache(sorted((TASK_ROOT / "output/qwen_c200_research/qwen_cache/E_c67_short/shards").glob("*.jsonl")))
    scores = {q: {d: float(r["yes_no_logit"]) for d, r in m.items()} for q, m in rows.items()}
    return U, scores


def test_shortlist_deterministic_incumbent_kept():
    U, scores = _universe_scores()
    qid = str(U.split.qids[3])
    a, _, _ = F._shortlist_for_qid(universe=U, scores=scores, qid=qid)
    b, anchor, _ = F._shortlist_for_qid(universe=U, scores=scores, qid=qid)
    assert a == b
    assert 1 <= len(a) <= 8
    assert str(anchor[4]) in a


def test_records_respect_4096_budget():
    from transformers import AutoTokenizer
    from qwen_c200_common import load_raw_documents
    U, scores = _universe_scores()
    qid = str(U.split.qids[3])
    short, _, _ = F._shortlist_for_qid(universe=U, scores=scores, qid=qid)
    docs = load_raw_documents(TASK_ROOT / "data/raw/public_test/selected-contexts.zip", set(short))
    tok = AutoTokenizer.from_pretrained("models/Qwen3-Reranker-0.6B", local_files_only=True, padding_side="left", use_fast=True)
    recs = F._records_for_qid(tokenizer=tok, universe=U, docs=docs, scores=scores, qid=qid)
    assert 1 <= len(recs) <= 8
    for r in recs:
        assert len(r["input_ids"]) <= 4096
        assert r["prompt"]["document_token_budget"] > 3000
        assert r["pack"]["packed_tokens"] <= r["prompt"]["document_token_budget"]


def test_cli_rejects_holdout_short_cache():
    import subprocess
    p = subprocess.run(
        [sys.executable, "Task1 - LegalIR/scripts/qwen_shortlist8_long_cache.py", "--mode", "preflight",
         "--stage1", "x", "--stage2", "x", "--anchor", "x", "--contexts-zip", "x",
         "--qwen-model", "x", "--short-cache-dir", "/tmp/holdout_evil", "--output-dir", "y"],
        capture_output=True, text=True,
    )
    assert p.returncode != 0
