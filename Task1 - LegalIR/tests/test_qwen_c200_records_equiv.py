"""Equivalence: optimized _records_for_qid must be bit-identical to legacy.

Legacy logic (pre-2026-09-15) called _prompt_components twice per doc:
once with "" for budget, once with packed text. Budget depends only on
(question, max_length), so the optimized version encodes the template
once per QID. Output (input_ids, pack, prompt) must match exactly.
"""
from __future__ import annotations

import sys
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT / "scripts"))

from qwen_c200_cache import _prompt_components, _records_for_qid, pool_for  # noqa: E402
from qwen_c200_common import load_candidate_universe, load_raw_documents, needed_document_ids, pack_document  # noqa: E402


def _legacy_records(tokenizer, universe, docs, qid, query_index, variant, max_length):
    from qwen_c200_common import pack_document as _pack
    question = str(universe.split.questions[query_index])
    records = []
    for rank, doc_id in enumerate(pool_for(universe, qid, variant), 1):
        _, reserve = _prompt_components(tokenizer, question, "", max_length)
        packed, pack_meta = _pack(
            tokenizer, question=question, document=docs[doc_id],
            max_tokens=int(reserve["document_token_budget"]),
        )
        input_ids, prompt_meta = _prompt_components(tokenizer, question, packed, max_length)
        records.append({
            "qid": qid, "doc_id": doc_id, "candidate_rank": rank,
            "input_ids": input_ids, "pack": pack_meta, "prompt": prompt_meta,
        })
    return records


def test_records_for_qid_bit_identical():
    from transformers import AutoTokenizer
    model_dir = TASK_ROOT.parent / "models" / "Qwen3-Reranker-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, padding_side="left", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    universe = load_candidate_universe(
        stage1=TASK_ROOT / "output/v10_9_prf/complement_safe_tail_stage1.audit.json",
        stage2=TASK_ROOT / "output/v10_9_prf/complement_safe_tail_stage2_c35_p3.audit.json",
        anchor=TASK_ROOT / "output/v16_rgm/complement_v16b_memory_slot5.json",
        reference=None,
    )
    # First 3 QIDs of the shard hash space + one long-doc QID (probe heuristic).
    qids = [str(q) for q in universe.split.qids[:3]]
    docs = load_raw_documents(
        TASK_ROOT / "data/raw/public_test/selected-contexts.zip",
        {d for q in qids for d in list(pool_for(universe, q, "E_c67_short"))},
    )
    for qid in qids:
        idx = list(universe.split.qids).index(qid) if not hasattr(universe.split.qids, "index") else universe.split.qids.index(qid)
        new = _records_for_qid(tokenizer=tokenizer, universe=universe, docs=docs, qid=qid, query_index=idx, variant="E_c67_short", max_length=768)
        old = _legacy_records(tokenizer, universe, docs, qid, idx, "E_c67_short", 768)
        assert len(new) == len(old) == 67
        for n, o in zip(new, old):
            assert n["qid"] == o["qid"] and n["doc_id"] == o["doc_id"] and n["candidate_rank"] == o["candidate_rank"]
            assert n["input_ids"] == o["input_ids"], f"input_ids drift {qid}/{n['doc_id']}"
            assert n["pack"] == o["pack"], f"pack drift {qid}/{n['doc_id']}"
            assert n["prompt"] == o["prompt"], f"prompt drift {qid}/{n['doc_id']}"


def test_records_for_qid_short_variant_identical():
    from transformers import AutoTokenizer
    model_dir = TASK_ROOT.parent / "models" / "Qwen3-Reranker-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, padding_side="left", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    universe = load_candidate_universe(
        stage1=TASK_ROOT / "output/v10_9_prf/complement_safe_tail_stage1.audit.json",
        stage2=TASK_ROOT / "output/v10_9_prf/complement_safe_tail_stage2_c35_p3.audit.json",
        anchor=TASK_ROOT / "output/v16_rgm/complement_v16b_memory_slot5.json",
        reference=None,
    )
    qid = str(universe.split.qids[0])
    idx = 0
    docs = load_raw_documents(
        TASK_ROOT / "data/raw/public_test/selected-contexts.zip",
        set(pool_for(universe, qid, "A_c35_short")),
    )
    new = _records_for_qid(tokenizer=tokenizer, universe=universe, docs=docs, qid=qid, query_index=idx, variant="A_c35_short", max_length=768)
    old = _legacy_records(tokenizer, universe, docs, qid, idx, "A_c35_short", 768)
    assert len(new) == len(old) == 35
    for n, o in zip(new, old):
        assert n["input_ids"] == o["input_ids"]
        assert n["pack"] == o["pack"]
        assert n["prompt"] == o["prompt"]
