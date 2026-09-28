"""E/F evaluator wiring: pools, challengers, S2 features, manifest."""
from __future__ import annotations

import sys
from pathlib import Path

TASK_ROOT = Path("Task1 - LegalIR")
sys.path.insert(0, str(Path("Task1 - LegalIR/scripts")))

import qwen_c200_nested_eval as ev  # noqa: E402


def _split_with_scores():
    from qwen_c200_common import load_candidate_universe, read_score_cache
    U = load_candidate_universe(
        stage1=TASK_ROOT / "output/v10_9_prf/complement_safe_tail_stage1.audit.json",
        stage2=TASK_ROOT / "output/v10_9_prf/complement_safe_tail_stage2_c35_p3.audit.json",
        anchor=TASK_ROOT / "output/v16_rgm/complement_v16b_memory_slot5.json",
        reference=None,
    )
    sp = U.split
    setattr(sp, "_qwen_universe", U)
    e = read_score_cache(sorted((TASK_ROOT / "output/qwen_c200_research/qwen_cache/E_c67_short/shards").glob("*.jsonl")))
    f = read_score_cache(sorted((TASK_ROOT / "output/qwen_c200_research/qwen_cache/F_shortlist8_long/shards").glob("*.jsonl")))
    setattr(sp, "_ef_e_scores", e)
    return sp, {"E_c67_short": e, "F_shortlist8_long": f}


def test_eval_set_dispatch():
    import argparse
    ns_ad = argparse.Namespace(eval_set="AD")
    ns_aef = argparse.Namespace(eval_set="AEF")
    assert ev._active_variants(ns_ad) == ev.VARIANTS
    assert ev._active_variants(ns_aef) == ev.EVAL_VARIANTS_EF
    assert ev._report_prefix(ns_ad) == "qwen_c200_"
    assert ev._report_prefix(ns_aef) == "qwen_ef_"


def test_f_pool_matches_shortlist_recompute():
    from qwen_c200_common import pool_for
    from qwen_shortlist8 import select_shortlist8
    sp, caches = _split_with_scores()
    for raw_qid in list(sp.qids)[:5]:
        qid = str(raw_qid)
        pool = ev._variant_pool(sp, qid, "F_shortlist8_long")
        idx = list(sp.qids).index(raw_qid)
        anchor = [str(v) for v in sp.anchor[idx]]
        ranking = [str(v) for v in pool_for(sp._qwen_universe, qid, "E_c67_short")]
        expect = select_shortlist8(qid, anchor, ranking, {d: float(caches["E_c67_short"][qid][d]["yes_no_logit"]) for d in ranking})
        assert pool == expect
        assert 1 <= len(pool) <= 8 and anchor[4] in pool
        assert not (set(pool) & set(anchor[:4]))


def test_f_challengers_and_s2_features():
    sp, caches = _split_with_scores()
    idx = 7
    challengers, incumbent, anchor = ev._ranked_challengers(sp, idx, caches["F_shortlist8_long"], "F_shortlist8_long")
    assert incumbent == anchor[4] and incumbent not in challengers
    assert len(challengers) <= 7
    rows = ev._s2_features(sp, idx, caches["F_shortlist8_long"], "F_shortlist8_long", with_target=False)
    assert 0 < len(rows) <= 3
    for r in rows:
        assert len(r["features"]) == len(ev.S2_FEATURE_SCHEMA) == 5
        assert r["target"] is None


def test_ef_preflight_manifest_on_disk():
    import json
    m = json.load(open(TASK_ROOT / "output/qwen_c200_research/qwen_ef_preflight_report.json"))
    assert m["queries"] == 1197
    assert m["pairs"] == {"E_c67_short": 80199, "F_shortlist8_long": 9576}
    assert m["lineage"]["variants"] == ["E_c67_short", "F_shortlist8_long"]
    assert len(m["planned_configurations"]) == 4
    assert m["holdout_opened"] is False and m["public_labels_used"] is False
