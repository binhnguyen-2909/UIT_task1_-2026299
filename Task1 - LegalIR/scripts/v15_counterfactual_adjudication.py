"""V15 difference-first counterfactual legal adjudication.

This is the algorithmic phase of the post-V14 research direction.  It does not
load ColBERT, add neural parameters, or globally rerank the frozen v11h top
five.  For each question it builds a small tournament over the Stage-1
shortlist, compares documents directly, and may replace only slot five when a
candidate beats the v11h tail, two neighbouring candidates, and the cycle gate.

The implementation keeps the full reused V11/V14 logic explicit:

* frozen v11h Stage-1/Stage-2/v11h score features remain inputs;
* ``v14_safe_rescue.parse_constraints`` and legal evidence selection provide
  typed constraint, contradiction, structure, and weakest-constraint signals;
* the new signals are pairwise differences and exclusive evidence, never an
  independent candidate relevance score;
* query rewriting is a deterministic BTC-only sidecar.  It preserves
  negation/numbers and emits focused views; it never uses an external corpus or
  a learned parameter;
* references are read only for complement selection and the single locked
  holdout audit after all complement gates pass.

The hard V15 gate is stricter than V14: standard delta >= .004, law-family and
template OOD delta >= .003, intervention precision >= .90, destructive rate
<= .01, and worst fold >= -.001.  A failed gate returns a report and never
opens the holdout or public data.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from .evaluate import official_metrics
    from .v14_replacement_pipeline import (
        AuditSplit,
        _load_split,
        law_family_groups,
        load_document_titles,
        prefilter_candidates,
        standard_leak_groups,
        template_groups,
        write_json,
    )
    from .v14_safe_rescue import (
        CONTENT_STOPWORDS,
        Document,
        _content_tokens,
        _normal_tokens,
        _weighted_recall,
        build_candidate_features,
        intervention_report,
        load_context_documents,
        parse_constraints,
        validate_model_budget,
    )
except ImportError:  # pragma: no cover - direct script execution
    from evaluate import official_metrics
    from v14_replacement_pipeline import (
        AuditSplit,
        _load_split,
        law_family_groups,
        load_document_titles,
        prefilter_candidates,
        standard_leak_groups,
        template_groups,
        write_json,
    )
    from v14_safe_rescue import (
        CONTENT_STOPWORDS,
        Document,
        _content_tokens,
        _normal_tokens,
        _weighted_recall,
        build_candidate_features,
        intervention_report,
        load_context_documents,
        parse_constraints,
        validate_model_budget,
    )


SEED = 20260829
SHORTLISTS = (8, 10)
REWRITE_RULES = (
    "base",
    "remove_boilerplate",
    "critical_focus",
    "exception_split",
    "structure_focus",
    "actor_action",
    "negation_focus",
    "numeric_focus",
)
RULE_WEIGHTS = {
    "base": 0.16,
    "remove_boilerplate": 0.10,
    "critical_focus": 0.28,
    "exception_split": 0.34,
    "structure_focus": 0.14,
    "actor_action": 0.10,
    "negation_focus": 0.22,
    "numeric_focus": 0.20,
}
PAIR_WEIGHTS = {
    "ce": 0.34,
    "v11": 0.18,
    "stage1_dense": 0.08,
    "stage1_lexical": 0.08,
    "stage1_heading": 0.10,
    "rank": 0.05,
    "element_mean": 0.12,
    "critical_mean": 0.20,
    "weakest_constraint": 0.34,
    "exclusive_critical": 0.38,
    "evidence": 0.24,
    "evidence_critical": 0.34,
    "contradiction": 0.38,
    "number": 0.20,
    "structure": 0.14,
    "lexical": 0.10,
}
CYCLE_MARGIN = 0.22
MIN_SELECTOR_SCORE = 0.49
MIN_CONSTRAINT_MIN = 0.42
MIN_EVIDENCE_CRITICAL = 0.52
MIN_EVIDENCE_SCORE = 0.40
MAX_CONTRADICTION = 0.25
MIN_INDEPENDENT_SIGNALS = 2


@dataclass(frozen=True)
class DocView:
    doc_id: str
    rank: int
    neural: float
    v11: float
    base_features: Mapping[str, float]
    legal: Any
    constraints: Any
    rewrite_scores: Mapping[str, float]
    content_tokens: frozenset[str]


@dataclass(frozen=True)
class QuestionState:
    index: int
    qid: str
    query: str
    anchor: tuple[str, ...]
    docs: tuple[DocView, ...]


def _bounded(value: float, scale: float = 1.0) -> float:
    return math.tanh(float(value) / max(scale, 1e-9))


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def _rewrite_views(query: str, constraints: Any) -> dict[str, tuple[str, ...]]:
    """Create fixed, label-free query views while preserving legal cues."""

    normalized = " ".join(str(query).split())
    tokens = tuple(_content_tokens(normalized))
    # ``_content_tokens`` intentionally removes function words.  Negation is
    # a legal condition, however, so keep a second normalized stream for the
    # negation/numeric sidecars instead of silently dropping ``không``.
    all_tokens = tuple(_normal_tokens(normalized))
    boilerplate = re.compile(
        r"\b(?:theo quy dinh|duoc quy dinh|nhu the nao|ra sao|la gi|bao gom nhung gi|"
        r"co duoc|co phai|duoc khong|hay khong|trong truong hop nao)\b",
        re.IGNORECASE,
    )
    no_boilerplate = tuple(_content_tokens(boilerplate.sub(" ", normalized)))
    critical = tuple(_content_tokens(" ".join(constraints.elements.get(key, "") for key in constraints.critical)))
    exception = tuple(_content_tokens(" ".join(
        constraints.elements.get(key, "") for key in ("condition", "exception", "temporal")
    )))
    structure = tuple(_content_tokens(" ".join(
        constraints.elements.get(key, "") for key in ("article", "clause", "point", "number")
    )))
    actor_action = tuple(_content_tokens(" ".join(
        constraints.elements.get(key, "") for key in ("actor", "action", "object")
    )))
    negation = tuple(token for token in all_tokens if token in {"khong", "ngoai", "tru"})
    numeric = tuple(token for token in all_tokens if token.isdigit())
    return {
        "base": tokens,
        "remove_boilerplate": no_boilerplate or tokens,
        "critical_focus": _unique((*critical, *critical, *tokens)),
        "exception_split": _unique((*exception, *exception, *critical, *tokens)),
        "structure_focus": _unique((*structure, *structure, *tokens)),
        "actor_action": _unique((*actor_action, *tokens)),
        "negation_focus": _unique((*negation, *negation, *critical, *tokens)),
        "numeric_focus": _unique((*numeric, *numeric, *critical, *tokens)),
    }


def _normal_feature_map(row: np.ndarray, names: Sequence[str]) -> dict[str, float]:
    return {str(name): float(row[index]) for index, name in enumerate(names)}


def _doc_rewrite_scores(
    doc: Document,
    views: Mapping[str, Sequence[str]],
    document_frequency: Mapping[str, int],
    corpus_size: int,
) -> dict[str, float]:
    return {
        name: float(_weighted_recall(tokens, set(doc.content_tokens), document_frequency, corpus_size))
        for name, tokens in views.items()
    }


def build_question_states(
    split: AuditSplit,
    documents: Mapping[str, Document],
    shortlist: int,
) -> tuple[QuestionState, ...]:
    if split.references is None:
        raise ValueError("V15 state construction requires complement references")
    document_frequency: Counter[str] = Counter()
    for doc in documents.values():
        document_frequency.update(doc.content_tokens)
    corpus_size = max(len(documents), 1)
    states: list[QuestionState] = []
    for question_index, qid in enumerate(split.qids):
        query = split.questions[question_index]
        constraints = parse_constraints(query)
        anchor = tuple(split.anchor[qid]["answer"])
        # The V14 high-recall, label-free union is the c35 -> shortlist bridge.
        # ``shortlist`` counts the complete tournament (the frozen five plus
        # three or five challenger slots), so no hidden c50/candidate branch
        # is introduced here.
        challenger_count = max(shortlist - len(anchor), 0)
        challengers, _ = prefilter_candidates(
            split, question_index, 35, challenger_count, top_per_signal=3
        )
        candidate_ids = _unique((*anchor, *challengers))
        neural_values = split.neural_scores[question_index]
        v11_values = split.v11_scores[question_index]
        neural_order = sorted(neural_values, key=lambda doc_id: (-float(neural_values[doc_id]), str(doc_id)))
        views = _rewrite_views(query, constraints)
        docs: list[DocView] = []
        block = split.feature_blocks[question_index]
        by_name = {str(name): index for index, name in enumerate(split.feature_names)}
        selected = list(dict.fromkeys(split.candidates[question_index][:35]))
        rank_by_id = {doc_id: rank for rank, doc_id in enumerate(split.candidates[question_index], 1)}
        for doc_id in candidate_ids:
            doc = documents.get(doc_id)
            if doc is None:
                continue
            rank = int(rank_by_id.get(doc_id, shortlist + 1))
            if doc_id in selected:
                row = block[selected.index(doc_id)]
                base = _normal_feature_map(row, split.feature_names)
            else:
                base = {name: 0.0 for name in split.feature_names}
            scores = _doc_rewrite_scores(doc, views, document_frequency, corpus_size)
            docs.append(DocView(
                doc_id=doc_id,
                rank=rank,
                neural=float(neural_values.get(doc_id, 0.0)),
                v11=float(v11_values.get(doc_id, 0.0)),
                base_features=base,
                legal=build_candidate_features(
                    query,
                    rank,
                    doc,
                    constraints,
                    float(neural_values.get(doc_id, 0.0)),
                    neural_order.index(doc_id) + 1 if doc_id in neural_order else len(neural_order) + 1,
                    list(neural_values.values()),
                    document_frequency,
                    corpus_size,
                    with_evidence=True,
                ),
                constraints=constraints,
                rewrite_scores=scores,
                content_tokens=doc.content_tokens,
            ))
        states.append(QuestionState(question_index, qid, query, anchor, tuple(docs)))
    return tuple(states)


def _feature(view: DocView, name: str) -> float:
    return float(view.base_features.get(name, 0.0))


def _legal_value(view: DocView, name: str) -> float:
    legal = view.legal
    if name == "element_mean":
        return float(legal.element_mean)
    if name == "critical_mean":
        return float(statistics.fmean(legal.constraint_scores.get(key, 0.0) for key in legal.constraint_scores) if legal.constraint_scores else 0.0)
    if name == "weakest_constraint":
        return float(legal.critical_min)
    if name == "evidence":
        return float(legal.evidence.score)
    if name == "evidence_critical":
        return float(legal.evidence.critical_coverage)
    if name == "contradiction":
        return float(legal.contradiction)
    if name == "number":
        return float(legal.number_exact)
    if name == "structure":
        return float(legal.structure)
    if name == "lexical":
        return float(legal.lexical)
    raise KeyError(name)


def _exclusive_critical(a: DocView, b: DocView) -> float:
    a_only = 0.0
    b_only = 0.0
    for key, value in a.legal.constraint_scores.items():
        av = float(value)
        bv = float(b.legal.constraint_scores.get(key, 0.0))
        a_only += max(0.0, av - bv)
        b_only += max(0.0, bv - av)
    denom = max(len(a.legal.constraint_scores), 1)
    token_exclusive = 0.0
    for key in a.constraints.critical:
        a_tokens = set(_content_tokens(a.constraints.elements.get(key, "")))
        b_tokens = set(_content_tokens(b.constraints.elements.get(key, "")))
        if not a_tokens and not b_tokens:
            continue
        a_only_tokens = len(a_tokens - b.content_tokens) / max(len(a_tokens), 1)
        b_only_tokens = len(b_tokens - a.content_tokens) / max(len(b_tokens), 1)
        token_exclusive += a_only_tokens - b_only_tokens
    critical_keys = tuple(getattr(a.constraints, "critical", ()))
    return (a_only - b_only) / denom + token_exclusive / max(len(critical_keys), 1)


def _passes_legal_safety_gate(view: DocView) -> bool:
    """Apply the frozen V14 evidence safety floor before V15 comparison.

    V15 changes the comparison objective, not the meaning of a legally
    admissible evidence set.  Keeping this floor prevents a high pairwise
    margin produced by lexical or neural noise from replacing a valid tail.
    The attribute fallback is only for the tiny structural unit-test doubles;
    production ``CandidateFeatures`` always exposes the complete gate.
    """

    legal = view.legal
    if not hasattr(legal, "independent_signals"):
        return True
    return bool(
        legal.selector_score >= MIN_SELECTOR_SCORE
        and legal.element_min >= MIN_CONSTRAINT_MIN
        and legal.critical_min >= MIN_CONSTRAINT_MIN
        and legal.evidence.critical_coverage >= MIN_EVIDENCE_CRITICAL
        and legal.evidence.score >= MIN_EVIDENCE_SCORE
        and legal.contradiction <= MAX_CONTRADICTION
        and legal.independent_signals >= MIN_INDEPENDENT_SIGNALS
    )


def pairwise_features(a: DocView, b: DocView, active_rules: Sequence[str]) -> dict[str, float]:
    """Return A-minus-B features; positive values prefer A."""

    result = {
        "ce": _bounded(a.neural - b.neural, 0.8),
        "v11": _bounded(a.v11 - b.v11, 0.8),
        "stage1_dense": _bounded(_feature(a, "stage1_dense") - _feature(b, "stage1_dense")),
        "stage1_lexical": _bounded(_feature(a, "stage1_chunk_phrase") - _feature(b, "stage1_chunk_phrase")),
        "stage1_heading": _bounded(_feature(a, "longest_title_phrase_tokens") - _feature(b, "longest_title_phrase_tokens"), 4.0),
        "rank": _bounded((b.rank - a.rank) / 10.0),
        "element_mean": _legal_value(a, "element_mean") - _legal_value(b, "element_mean"),
        "critical_mean": _legal_value(a, "critical_mean") - _legal_value(b, "critical_mean"),
        "weakest_constraint": _legal_value(a, "weakest_constraint") - _legal_value(b, "weakest_constraint"),
        "exclusive_critical": _exclusive_critical(a, b),
        "evidence": _legal_value(a, "evidence") - _legal_value(b, "evidence"),
        "evidence_critical": _legal_value(a, "evidence_critical") - _legal_value(b, "evidence_critical"),
        # Lower contradiction is safer, so reverse this difference.
        "contradiction": _legal_value(b, "contradiction") - _legal_value(a, "contradiction"),
        "number": _legal_value(a, "number") - _legal_value(b, "number"),
        "structure": _legal_value(a, "structure") - _legal_value(b, "structure"),
        "lexical": _legal_value(a, "lexical") - _legal_value(b, "lexical"),
    }
    rewrite_margin = 0.0
    for rule in active_rules:
        rewrite_margin += RULE_WEIGHTS[rule] * (a.rewrite_scores.get(rule, 0.0) - b.rewrite_scores.get(rule, 0.0))
    result["rewrite"] = rewrite_margin
    return result


def pairwise_margin(a: DocView, b: DocView, active_rules: Sequence[str]) -> float:
    values = pairwise_features(a, b, active_rules)
    return float(sum(PAIR_WEIGHTS.get(key, 0.0) * value for key, value in values.items() if key != "rewrite") + values["rewrite"])


def _strong_cycle(matrix: np.ndarray, index: int) -> bool:
    for left in range(matrix.shape[0]):
        if left == index or matrix[index, left] < CYCLE_MARGIN:
            continue
        for right in range(matrix.shape[0]):
            if right in {index, left}:
                continue
            if matrix[left, right] >= CYCLE_MARGIN and matrix[right, index] >= CYCLE_MARGIN:
                return True
    return False


def tournament(
    state: QuestionState,
    active_rules: Sequence[str],
    critical_threshold: float,
    shortlist: int,
) -> dict[str, Any]:
    # ``state.docs`` is already the frozen-anchor plus c35-prefiltered
    # shortlist.  Filtering again by Stage-1 rank would discard the very
    # rank-11..35 challengers the V14 audit proved worth comparing.
    docs = list({doc.doc_id: doc for doc in state.docs}.values())
    docs.sort(key=lambda doc: (doc.rank, doc.doc_id))
    if len(docs) < 2:
        return {"candidate": None, "accepted": False, "reason": "shortlist_too_small", "docs": len(docs)}
    matrix = np.zeros((len(docs), len(docs)), dtype=np.float64)
    for i, j in itertools.permutations(range(len(docs)), 2):
        matrix[i, j] = pairwise_margin(docs[i], docs[j], active_rules)
    aggregate = matrix.sum(axis=1)
    by_id = {doc.doc_id: i for i, doc in enumerate(docs)}
    tail_id = state.anchor[-1] if state.anchor else docs[min(4, len(docs) - 1)].doc_id
    tail_index = by_id.get(tail_id)
    if tail_index is None:
        return {"candidate": None, "accepted": False, "reason": "tail_not_in_shortlist", "docs": len(docs)}
    candidates: list[dict[str, Any]] = []
    for index, doc in enumerate(docs):
        if doc.doc_id in state.anchor or doc.rank <= 5:
            continue
        safety_gate = _passes_legal_safety_gate(doc)
        preceding = [
            other_index for other_index, other in enumerate(docs)
            if other_index != index and other.rank < doc.rank and other.doc_id != tail_id
        ]
        preceding.sort(key=lambda item: (docs[item].rank, docs[item].doc_id), reverse=True)
        neighbors = preceding[:3]
        wins_against_neighbors = sum(matrix[index, other] > 0.0 for other in neighbors)
        margin_tail = float(matrix[index, tail_index])
        critical_margin = float(
            _legal_value(doc, "weakest_constraint") - _legal_value(docs[tail_index], "weakest_constraint")
        )
        accepted = bool(
            safety_gate
            and margin_tail > 0.0
            and wins_against_neighbors >= min(2, len(neighbors))
            and float(aggregate[index] - aggregate[tail_index]) > 0.0
            and critical_margin >= critical_threshold
            and not _strong_cycle(matrix, index)
        )
        candidates.append({
            "candidate": doc.doc_id,
            "rank": doc.rank,
            "aggregate": float(aggregate[index]),
            "margin_vs_tail": margin_tail,
            "critical_margin": critical_margin,
            "neighbors": [docs[item].doc_id for item in neighbors],
            "wins_against_neighbors": int(wins_against_neighbors),
            "cycle": bool(_strong_cycle(matrix, index)),
            "legal_safety_gate": safety_gate,
            "accepted": accepted,
        })
    accepted_rows = [row for row in candidates if row["accepted"]]
    winner = max(accepted_rows, key=lambda row: (row["aggregate"], row["margin_vs_tail"], -row["rank"], row["candidate"])) if accepted_rows else None
    return {
        "candidate": winner["candidate"] if winner else None,
        "accepted": bool(winner),
        "winner": winner,
        "candidates": candidates,
        "tail": tail_id,
        "docs": len(docs),
    }


def predictions_for_states(
    split: AuditSplit,
    states: Sequence[QuestionState],
    active_rules: Sequence[str],
    critical_threshold: float,
    shortlist: int,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, Any]]:
    state_by_index = {state.index: state for state in states}
    output: dict[str, dict[str, list[str]]] = {}
    trace: dict[str, Any] = {}
    for index, qid in enumerate(split.qids):
        state = state_by_index[index]
        row = tournament(state, active_rules, critical_threshold, shortlist)
        answers = list(state.anchor)
        if row.get("accepted") and row.get("candidate") not in answers:
            answers[-1] = str(row["candidate"])
        output[qid] = {"answer": answers}
        trace[qid] = row
    return output, trace


def _subset_metrics(split: AuditSplit, predictions: Mapping[str, Any], indices: Sequence[int]) -> dict[str, Any]:
    qids = [split.qids[index] for index in indices]
    anchor = {qid: split.anchor[qid] for qid in qids}
    refs = {qid: split.references[qid] for qid in qids} if split.references is not None else None
    if refs is None:
        raise ValueError("subset metrics require references")
    baseline = official_metrics(anchor, refs)
    rescued = official_metrics({qid: predictions[qid] for qid in qids}, refs)
    return {
        "baseline": baseline,
        "rescued": rescued,
        "delta": {key: float(rescued[key] - baseline[key]) for key in ("recall", "precision")},
        "intervention": intervention_report(anchor, {qid: predictions[qid] for qid in qids}, refs),
    }


def _folds(split: AuditSplit, groups: np.ndarray) -> np.ndarray:
    from sklearn.model_selection import StratifiedGroupKFold

    labels = np.asarray([
        int(any(
            candidate in set(split.references[qid]["answer"]) and split.anchor[qid]["answer"][-1] not in set(split.references[qid]["answer"])
            for candidate in split.candidates[index][5:]
        ))
        for index, qid in enumerate(split.qids)
    ], dtype=np.int8)
    count = min(5, len(set(map(int, groups))), int(labels.sum()))
    if count < 3:
        return np.arange(len(split.qids), dtype=np.int8) % 3
    result = np.full(len(split.qids), -1, dtype=np.int8)
    questions = np.arange(len(split.qids), dtype=np.int32)
    for fold, (_, valid) in enumerate(StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=SEED).split(questions, labels, groups=groups)):
        result[valid] = fold
    return result


def _fold_report(split: AuditSplit, predictions: Mapping[str, Any], groups: np.ndarray) -> dict[str, Any]:
    fold_ids = _folds(split, groups)
    rows = []
    for fold in sorted(set(map(int, fold_ids))):
        indices = np.flatnonzero(fold_ids == fold).tolist()
        metrics = _subset_metrics(split, predictions, indices)
        rows.append({
            "fold": fold,
            "questions": len(indices),
            "baseline_recall": metrics["baseline"]["recall"],
            "rescued_recall": metrics["rescued"]["recall"],
            "delta_recall": metrics["delta"]["recall"],
        })
    return {
        "folds": rows,
        "worst_fold_delta": min((row["delta_recall"] for row in rows), default=0.0),
        "delta_std": statistics.pstdev([row["delta_recall"] for row in rows]) if len(rows) > 1 else 0.0,
    }


def evaluate_configuration(
    split: AuditSplit,
    states: Sequence[QuestionState],
    active_rules: Sequence[str],
    critical_threshold: float,
    shortlist: int,
    titles: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, dict[str, list[str]]], dict[str, Any]]:
    predictions, trace = predictions_for_states(split, states, active_rules, critical_threshold, shortlist)
    groups = {
        "standard": standard_leak_groups(split),
        "law_family_ood": law_family_groups(split, titles),
        "query_template_ood": template_groups(split),
    }
    evaluations = {
        name: {
            "evaluation": _subset_metrics(split, predictions, list(range(len(split.qids)))),
            "folds": _fold_report(split, predictions, group),
        }
        for name, group in groups.items()
    }
    standard = evaluations["standard"]["evaluation"]
    law = evaluations["law_family_ood"]["evaluation"]
    template = evaluations["query_template_ood"]["evaluation"]
    gates = {
        "standard_delta_at_least_0_004": standard["delta"]["recall"] >= 0.004,
        "law_family_delta_at_least_0_003": law["delta"]["recall"] >= 0.003,
        "query_template_delta_at_least_0_003": template["delta"]["recall"] >= 0.003,
        "intervention_precision_at_least_0_90": all(
            evaluations[name]["evaluation"]["intervention"]["intervention_precision"] >= 0.90
            for name in evaluations
        ),
        "destructive_rate_at_most_0_01": all(
            evaluations[name]["evaluation"]["intervention"]["destructive_rate"] <= 0.01
            for name in evaluations
        ),
        "worst_fold_at_least_minus_0_001": min(
            evaluations[name]["folds"]["worst_fold_delta"] for name in evaluations
        ) >= -0.001,
    }
    return {
        "shortlist": shortlist,
        "rules": list(active_rules),
        "critical_threshold": critical_threshold,
        **evaluations,
        "gates": gates,
        "eligible": all(gates.values()),
    }, predictions, trace


def oracle_report(split: AuditSplit) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for limit in SHORTLISTS:
        eligible = captured = 0
        for index, qid in enumerate(split.qids):
            if split.references is None:
                continue
            gold = set(split.references[qid]["answer"])
            tail = split.anchor[qid]["answer"][-1]
            if tail in gold:
                continue
            rescuable = any(
                6 <= rank <= 35
                and doc_id in gold
                and doc_id not in split.anchor[qid]["answer"]
                for rank, doc_id in enumerate(split.candidates[index][:35], 1)
            )
            if not rescuable:
                continue
            eligible += 1
            challengers, _ = prefilter_candidates(
                split, index, 35, limit - len(split.anchor[qid]["answer"]), top_per_signal=3
            )
            if any(doc_id in gold and doc_id not in split.anchor[qid]["answer"] for doc_id in challengers):
                captured += 1
        output[str(limit)] = {
            "queries_with_tail_miss": eligible,
            "captured_queries": captured,
            "capture_rate": captured / eligible if eligible else 1.0,
            "prefilter": "v14 frozen multi-signal union over c35; top shortlist-5 challengers",
        }
    return output


def residual_candidate_audit(split: AuditSplit) -> dict[str, Any]:
    """Separate c35 headroom from genuinely candidate-starved misses."""

    counts = Counter()
    for index, qid in enumerate(split.qids):
        gold = set(split.references[qid]["answer"]) if split.references else set()
        anchor = set(split.anchor[qid]["answer"])
        if not split.references or not split.anchor[qid]["answer"]:
            continue
        if split.anchor[qid]["answer"][-1] in gold:
            counts["tail_already_gold"] += 1
            continue
        outside_anchor = [
            (rank, doc_id)
            for rank, doc_id in enumerate(split.candidates[index][:35], 1)
            if doc_id in gold and doc_id not in anchor
        ]
        rank6_35 = [row for row in outside_anchor if 6 <= row[0] <= 35]
        rank1_5 = [row for row in outside_anchor if row[0] <= 5]
        if rank6_35:
            counts["slot5_rescuable_inside_c35"] += 1
        elif rank1_5:
            counts["locked_top5_miss_inside_c35"] += 1
        else:
            counts["candidate_starved_at_c35"] += 1
    total = len(split.qids)
    return {
        "questions": total,
        "counts": dict(counts),
        "candidate_starved_rate": counts["candidate_starved_at_c35"] / max(total, 1),
        "policy": "Do not open retrieval outside c35 unless candidate_starved remains the dominant residual after a passing V15/LoRA policy.",
    }


def rewrite_evolution_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Record which BTC-only rewrite sets survive complement selection."""

    by_rules: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        key = "+".join(row["rules"])
        by_rules.setdefault(key, []).append(row)
    summaries = []
    for key, candidates in by_rules.items():
        winner = max(
            candidates,
            key=lambda row: (
                bool(row["eligible"]),
                float(row["standard"]["evaluation"]["delta"]["recall"]),
                float(row["standard"]["evaluation"]["intervention"]["intervention_precision"]),
            ),
        )
        summaries.append({
            "rules": key.split("+") if key else [],
            "best_standard_delta": winner["standard"]["evaluation"]["delta"]["recall"],
            "best_intervention_precision": winner["standard"]["evaluation"]["intervention"]["intervention_precision"],
            "best_destructive_rate": winner["standard"]["evaluation"]["intervention"]["destructive_rate"],
            "eligible": bool(winner["eligible"]),
        })
    selected = max(
        summaries,
        key=lambda row: (row["eligible"], row["best_standard_delta"], row["best_intervention_precision"]),
    ) if summaries else None
    return {
        "method": "complement-only greedy/drop audit over deterministic BTC-only sidecars",
        "selected": selected,
        "candidates": sorted(summaries, key=lambda row: (row["eligible"], row["best_standard_delta"]), reverse=True),
        "holdout_used": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts-zip", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--score-cache", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--complement-stage1", type=Path, required=True)
    parser.add_argument("--complement-stage2", type=Path, required=True)
    parser.add_argument("--complement-reference", type=Path, required=True)
    parser.add_argument("--holdout-stage1", type=Path, required=True)
    parser.add_argument("--holdout-stage2", type=Path, required=True)
    parser.add_argument("--holdout-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def _load_v15_split(args: argparse.Namespace, name: str) -> AuditSplit:
    if name == "complement":
        stage1, stage2, reference = args.complement_stage1, args.complement_stage2, args.complement_reference
    else:
        stage1, stage2, reference = args.holdout_stage1, args.holdout_stage2, args.holdout_reference
    return _load_split(
        name,
        stage1,
        stage2,
        reference,
        args.score_cache,
        args.feature_cache,
        35,
        args.limit,
    )


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    budget = validate_model_budget()
    complement = _load_v15_split(args, "complement")
    titles = load_document_titles(args.index)
    needed: set[str] = set()
    for index, qid in enumerate(complement.qids):
        needed.update(complement.anchor[qid]["answer"])
        for limit in SHORTLISTS:
            challengers, _ = prefilter_candidates(
                complement, index, 35, limit - len(complement.anchor[qid]["answer"]), top_per_signal=3
            )
            needed.update(challengers)
    print(f"V15 loading algorithmic evidence documents: {len(needed)}", flush=True)
    documents = load_context_documents(args.contexts_zip, needed, args.index, 12, "legal_chunking")
    states_by_shortlist = {
        shortlist: build_question_states(complement, documents, shortlist)
        for shortlist in SHORTLISTS
    }
    configs: list[tuple[int, tuple[str, ...], float]] = []
    rule_sets = [
        ("base",),
        ("base", "remove_boilerplate", "critical_focus"),
        ("base", "critical_focus", "exception_split", "negation_focus"),
        tuple(REWRITE_RULES),
    ]
    for shortlist in SHORTLISTS:
        for rules in rule_sets:
            for critical_threshold in (0.0, 0.02, 0.05):
                configs.append((shortlist, tuple(rules), critical_threshold))
    rows: list[dict[str, Any]] = []
    best_payload: tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None = None
    for index, (shortlist, rules, threshold) in enumerate(configs, 1):
        report, predictions, trace = evaluate_configuration(
            complement, states_by_shortlist[shortlist], rules, threshold, shortlist, titles
        )
        rows.append(report)
        key = (
            bool(report["eligible"]),
            float(report["standard"]["evaluation"]["delta"]["recall"]),
            float(report["law_family_ood"]["evaluation"]["delta"]["recall"]),
            float(report["query_template_ood"]["evaluation"]["delta"]["recall"]),
            float(report["standard"]["evaluation"]["intervention"]["intervention_precision"]),
            -int(report["standard"]["evaluation"]["intervention"]["changes"]),
        )
        if best_payload is None or key > best_payload[0]["_key"]:
            report_with_key = dict(report)
            report_with_key["_key"] = key
            best_payload = (report_with_key, predictions, trace)
        print(
            f"V15 config {index}/{len(configs)} shortlist={shortlist} rules={len(rules)} "
            f"critical={threshold:.2f} delta={report['standard']['evaluation']['delta']['recall']:+.6f} "
            f"eligible={report['eligible']}", flush=True
        )
    assert best_payload is not None
    best, complement_predictions, complement_trace = best_payload
    best.pop("_key", None)
    report: dict[str, Any] = {
        "kind": "legalir_v15_counterfactual_pairwise_adjudication",
        "status": "v15_complement_gate_passed" if best["eligible"] else "v15_complement_gate_failed",
        "leakage_policy": {
            "complement_used_for_rule_and_configuration_selection": True,
            "holdout_read": False,
            "holdout_used_for_selection": False,
            "public_read": False,
            "external_corpus": False,
        },
        "model_budget": {
            **budget,
            "v15_added_parameters": 0,
            "deployed_total_parameters": budget["total_parameters"],
            "deployed_total_billions": budget["total_billions"],
            "strictly_under_4b": True,
        },
        "architecture": {
            "frozen_prior": "v11h anchor; preserve answers 1-4; replace slot 5 at most once",
            "candidate_pool": "v11h Stage-1 c35",
            "pairwise_shortlists": list(SHORTLISTS),
            "pairwise_features": list(dict.fromkeys((*PAIR_WEIGHTS, "rewrite"))),
            "aggregation": "Bradley-Terry log-odds equivalent weighted Copeland tournament",
            "consistency_gate": "candidate beats tail, at least two preceding neighbours, aggregate tail margin positive, no strong cycle, critical margin threshold",
            "query_rewrites": list(REWRITE_RULES),
        },
        "reused_logic": {
            "v11h": "Stage-1 c35 candidates, Stage-2 neural scores, v11h selector score and 56-feature cache",
            "parser": "v14_safe_rescue.parse_constraints, including actor/action/object/condition/exception/temporal/modality/number/article/clause/point/topic",
            "evidence": "v14_safe_rescue legal chunking and exhaustive one-to-three passage evidence features",
            "new_logic": "direct A-minus-B features, exclusive typed support, fixed tournament and cycle consistency gate",
            "rewrite": "BTC-only deterministic views; boilerplate removal, critical duplication, exception split, structure/actor-action/negation/numeric focus",
        },
        "oracle": oracle_report(complement),
        "residual_candidate_audit": residual_candidate_audit(complement),
        "rewrite_evolution": rewrite_evolution_report(rows),
        "best": best,
        "leaderboard": sorted(rows, key=lambda row: (
            bool(row["eligible"]),
            row["standard"]["evaluation"]["delta"]["recall"],
            row["law_family_ood"]["evaluation"]["delta"]["recall"],
            row["query_template_ood"]["evaluation"]["delta"]["recall"],
        ), reverse=True)[:20],
        "artifacts": {
            "report": str((args.output_dir / "v15_counterfactual_report.json").resolve()),
        },
        "holdout": {"opened": False},
        "public": {"applied": False},
    }
    if not best["eligible"]:
        report["next_step"] = "Do not open locked holdout, LoRA, listwise specialist, or retrieval expansion; diagnose V15 signal first."
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(args.output_dir / "v15_counterfactual_report.json", report)
        print(json.dumps({"status": report["status"], "report": report["artifacts"]["report"]}, ensure_ascii=False, indent=2), flush=True)
        return 2

    holdout = _load_v15_split(args, "holdout")
    holdout_needed: set[str] = set()
    for index, qid in enumerate(holdout.qids):
        holdout_needed.update(holdout.anchor[qid]["answer"])
        challengers, _ = prefilter_candidates(
            holdout, index, 35, int(best["shortlist"]) - len(holdout.anchor[qid]["answer"]), top_per_signal=3
        )
        holdout_needed.update(challengers)
    holdout_documents = load_context_documents(args.contexts_zip, holdout_needed, args.index, 12, "legal_chunking")
    holdout_states = build_question_states(holdout, holdout_documents, int(best["shortlist"]))
    holdout_predictions, holdout_trace = predictions_for_states(
        holdout,
        holdout_states,
        best["rules"],
        float(best["critical_threshold"]),
        int(best["shortlist"]),
    )
    holdout_evaluation = _subset_metrics(holdout, holdout_predictions, list(range(len(holdout.qids))))
    observed_baseline = float(holdout_evaluation["baseline"]["recall"])
    if abs(observed_baseline - 0.9188333333333334) > 1e-12:
        raise RuntimeError(f"locked v11h baseline mismatch: {observed_baseline:.15f}")
    report["status"] = "v15_holdout_passed" if holdout_evaluation["delta"]["recall"] > 0 else "v15_holdout_rejected"
    report["holdout"] = {
        "opened": True,
        "evaluation": holdout_evaluation,
        "trace": holdout_trace,
        "promoted": bool(
            holdout_evaluation["rescued"]["recall"] >= 0.925
            and holdout_evaluation["delta"]["recall"] > 0
            and holdout_evaluation["intervention"]["intervention_precision"] >= 0.90
            and holdout_evaluation["intervention"]["destructive_rate"] <= 0.01
        ),
    }
    report["elapsed_seconds"] = time.perf_counter() - started
    write_json(args.output_dir / "v15_counterfactual_report.json", report)
    print(json.dumps({"status": report["status"], "report": report["artifacts"]["report"], "holdout": report["holdout"]}, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["holdout"]["promoted"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
