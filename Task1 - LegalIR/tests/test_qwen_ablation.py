"""Ablation flags: defaults preserve frozen behavior; branches change only their axis."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path("Task1 - LegalIR/scripts")))

import qwen_c200_nested_eval as ev  # noqa: E402


def _ns(**kw):
    base = dict(s1_thresholds=[0.0, 0.25, 0.5, 1.0], s1_gap_norm="absolute", skip_f_s2=False, s2_lite=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_defaults_preserve_frozen_behavior():
    args = _ns()
    assert ev._s1_grid(args) == [0.0, 0.25, 0.5, 1.0]
    assert ev._s2_schema(args) == list(ev.S2_FEATURE_SCHEMA) and len(ev._s2_schema(args)) == 5


def test_a1_grid_drops_zero():
    assert ev._s1_grid(_ns(s1_thresholds=[0.25, 0.5, 1.0])) == [0.25, 0.5, 1.0]
    try:
        ev._s1_grid(_ns(s1_thresholds=[]))
        raise AssertionError("empty grid must fail")
    except Exception:
        pass


def test_a2_zscore_gap_math():
    scores = {"q": {"a": {"yes_no_logit": 3.0}, "b": {"yes_no_logit": 1.0}, "c": {"yes_no_logit": 2.0}}}
    pool = ["a", "b", "c"]
    absolute = ev._s1_gap(scores=scores, qid="q", pool=pool, winner="a", incumbent="b", args=_ns())
    assert absolute == 2.0
    z = ev._s1_gap(scores=scores, qid="q", pool=pool, winner="a", incumbent="b", args=_ns(s1_gap_norm="zscore"))
    import numpy as np
    expect = 2.0 / max(float(np.asarray([3.0, 1.0, 2.0], dtype=np.float32).std()), 1e-6)
    assert abs(z - expect) < 1e-9 and z != absolute


def test_b_lite_schema_and_skip_flag():
    assert ev._s2_schema(_ns(s2_lite=True)) == [
        "qwen_candidate_minus_incumbent_logit", "qwen_candidate_within_pool_zscore"]
    assert _ns(skip_f_s2=True).skip_f_s2 is True
    assert _ns().skip_f_s2 is False
