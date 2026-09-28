"""Equivalence: batched segment encoding in pack_document == legacy per-segment calls."""
from __future__ import annotations

import sys
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT / "scripts"))

from qwen_c200_common import RawDocument, _segments, pack_document  # noqa: E402


def _legacy_pack(tokenizer, question, document, max_tokens):
    heading = f"[TITLE] {document.title.strip()}\n" if document.title.strip() else ""
    full = f"{heading}[DOCUMENT]\n{document.body.strip()}".strip()
    full_ids = tokenizer(full, add_special_tokens=False).get("input_ids", [])
    if len(full_ids) <= max_tokens:
        return full, {"kind": "full_document", "source_tokens": len(full_ids), "packed_tokens": len(full_ids), "segments": 1}
    import re
    from qwen_c200_common import _NUMBER, _terms
    q_terms, q_numbers = _terms(question), set(_NUMBER.findall(question))
    scored = []
    for index, segment in enumerate(_segments(document.body)):
        ids = tokenizer(segment, add_special_tokens=False).get("input_ids", [])
        if not ids:
            continue
        terms = _terms(segment)
        score = float(len(q_terms & terms)) / max(len(q_terms), 1) + 0.25 * len(q_numbers & set(_NUMBER.findall(segment)))
        scored.append((index, score, segment, len(ids)))
    selected = []
    used = len(tokenizer(heading + "[DOCUMENT]\n", add_special_tokens=False).get("input_ids", []))
    for index, _, segment, tokens in sorted(scored, key=lambda row: (-row[1], row[0])):
        if used >= max_tokens:
            break
        take = min(tokens, max_tokens - used)
        if take <= 0:
            continue
        if take < tokens:
            ids = tokenizer(segment, add_special_tokens=False).get("input_ids", [])[:take]
            segment = tokenizer.decode(ids, skip_special_tokens=True)
        selected.append((index, segment, take)); used += take
    selected.sort(key=lambda row: row[0])
    packed = f"{heading}[DOCUMENT]\n" + "\n".join(value for _, value, _ in selected)
    return packed.strip(), {"kind": "legal_boundary_lexical_order", "source_tokens": len(full_ids), "packed_tokens": used, "segments": len(selected)}


def test_pack_batched_identical():
    from transformers import AutoTokenizer
    from qwen_c200_common import load_candidate_universe, load_raw_documents, pool_for
    model_dir = TASK_ROOT.parent / "models" / "Qwen3-Reranker-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, padding_side="left", use_fast=True)
    universe = load_candidate_universe(
        stage1=TASK_ROOT / "output/v10_9_prf/complement_safe_tail_stage1.audit.json",
        stage2=TASK_ROOT / "output/v10_9_prf/complement_safe_tail_stage2_c35_p3.audit.json",
        anchor=TASK_ROOT / "output/v16_rgm/complement_v16b_memory_slot5.json",
        reference=None,
    )
    qid = str(universe.split.qids[5])
    idx = list(universe.split.qids).index(qid)
    question = str(universe.split.questions[idx])
    docs = load_raw_documents(
        TASK_ROOT / "data/raw/public_test/selected-contexts.zip",
        set(pool_for(universe, qid, "E_c67_short")),
    )
    for doc_id in list(pool_for(universe, qid, "E_c67_short"))[:20]:
        for budget in (768, 4096):
            new_packed, new_meta = pack_document(tokenizer, question=question, document=docs[doc_id], max_tokens=budget)
            old_packed, old_meta = _legacy_pack(tokenizer, question, docs[doc_id], budget)
            assert new_packed == old_packed, f"text drift {doc_id}/{budget}"
            assert new_meta == old_meta, f"meta drift {doc_id}/{budget}"
