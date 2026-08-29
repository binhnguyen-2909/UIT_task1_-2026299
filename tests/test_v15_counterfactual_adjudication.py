from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "Task1 - LegalIR" / "scripts" / "v15_counterfactual_adjudication.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("v15_counterfactual_adjudication", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

(
    DocView,
    QuestionState,
    _rewrite_views,
    _strong_cycle,
    pairwise_margin,
    tournament,
)= (
    MODULE.DocView,
    MODULE.QuestionState,
    MODULE._rewrite_views,
    MODULE._strong_cycle,
    MODULE.pairwise_margin,
    MODULE.tournament,
)


def _legal(level: float, *, contradiction: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        element_mean=level,
        constraint_scores={"condition": level},
        critical_min=level,
        evidence=SimpleNamespace(score=level, critical_coverage=level),
        contradiction=contradiction,
        number_exact=level,
        structure=level,
        lexical=level,
    )


def _doc(doc_id: str, rank: int, level: float, *, neural: float = 0.0) -> DocView:
    constraints = SimpleNamespace(
        critical=("condition",),
        elements={"condition": "đang nghỉ thai sản"},
    )
    return DocView(
        doc_id=doc_id,
        rank=rank,
        neural=neural,
        v11=level,
        base_features={
            "stage1_dense": level,
            "stage1_chunk_phrase": level,
            "longest_title_phrase_tokens": rank,
        },
        legal=_legal(level),
        constraints=constraints,
        rewrite_scores={name: level for name in ("base", "critical_focus", "exception_split")},
        content_tokens=frozenset({"nghi", "thai", "san"}) if level > 0.5 else frozenset({"nghi", "om"}),
    )


def test_rewrite_views_preserve_negation_and_numeric_cues() -> None:
    constraints = SimpleNamespace(
        elements={"condition": "trong năm 2024"},
        critical=("condition",),
    )
    views = _rewrite_views("Người lao động không được nghỉ trong năm 2024?", constraints)
    assert "2024" in views["numeric_focus"]
    assert "khong" in views["negation_focus"]
    assert "2024" in views["critical_focus"]


def test_strong_cycle_detects_only_cycle_through_index() -> None:
    no_cycle = np.asarray([[0.0, 0.3, -0.3], [-0.3, 0.0, 0.3], [0.3, -0.3, 0.0]])
    assert _strong_cycle(no_cycle, 0) is True
    assert _strong_cycle(no_cycle, 1) is True


def test_pairwise_margin_prefers_exclusive_critical_support() -> None:
    better = _doc("better", 6, 0.9, neural=0.2)
    weaker = _doc("weaker", 5, 0.2, neural=0.1)
    assert pairwise_margin(better, weaker, ("base", "critical_focus")) > 0


def test_tournament_requires_tail_and_two_neighbours() -> None:
    docs = (
        _doc("top1", 1, 0.1),
        _doc("top2", 2, 0.1),
        _doc("top3", 3, 0.1),
        _doc("top4", 4, 0.1),
        _doc("tail", 5, 0.1),
        _doc("candidate", 6, 0.9, neural=0.2),
    )
    state = QuestionState(0, "q", "question", ("top1", "top2", "top3", "top4", "tail"), docs)
    result = tournament(state, ("base", "critical_focus"), 0.0, 8)
    assert result["accepted"] is True
    assert result["candidate"] == "candidate"
