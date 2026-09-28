from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import hsr_contract  # noqa: E402
import hsr_nested_head  # noqa: E402


def _literal_grouped_objective(
    scores: torch.Tensor,
    groups: list[tuple[list[int], list[int], float]],
    weight: torch.Tensor,
    bias: torch.Tensor,
    regularisation: float,
) -> torch.Tensor:
    """The pre-vectorisation grouped loss, kept as a test-only oracle."""
    denominator = float(sum(group_weight for _, _, group_weight in groups))
    terms: list[torch.Tensor] = []
    for positive, negative, group_weight in groups:
        if positive and negative:
            pairwise = torch.nn.functional.softplus(
                scores[torch.as_tensor(negative, dtype=torch.long)][:, None]
                - scores[torch.as_tensor(positive, dtype=torch.long)][None, :]
            ).mean()
        else:
            pairwise = torch.zeros((), dtype=scores.dtype)
        indices = positive + negative
        targets = torch.as_tensor(
            [1.0] * len(positive) + [0.0] * len(negative), dtype=scores.dtype
        )
        pointwise = torch.nn.functional.binary_cross_entropy_with_logits(
            scores[torch.as_tensor(indices, dtype=torch.long)], targets
        )
        terms.append(float(group_weight) * (0.5 * pairwise + 0.5 * pointwise))
    return sum(terms) / denominator + float(regularisation) * (
        weight.square().sum() + bias.square()
    )


class _Split:
    qids = ["q1"]
    questions = ["question"]
    anchor = [["d1", "d2", "d3", "d4", "d5"]]
    references = [{"d6"}]


def _cache() -> hsr_contract.HiddenCache:
    rows = {}
    for rank in range(1, 68):
        doc = f"d{rank}"
        rows[doc] = {
            "qid": "q1", "doc_id": doc, "candidate_rank": rank,
            "evidence_valid": True, "native_yes_minus_no": float(68 - rank),
            "hidden": np.full(1024, float(rank), dtype=np.float16),
        }
    return hsr_contract.HiddenCache(
        scores={"q1": rows}, lineage_sha256="lineage", metadata_paths=(),
        feature_paths=(), hidden_size=1024, hidden_dtype="float16", pairs=67,
        lineage={}, parameter_ledger={"under_4b": True},
    )


class HiddenStateBranchTests(unittest.TestCase):
    def test_plan_is_under_four_b(self) -> None:
        budget = hsr_contract.planned_parameter_budget()
        self.assertTrue(budget["under_4b_plan"])
        self.assertEqual(budget["head_parameters"], 1025)

    def test_selector_keeps_top_four_and_changes_only_slot_five(self) -> None:
        split = _Split()
        c67 = {"q1": [f"d{rank}" for rank in range(1, 68)]}
        base = {"q1": {"answer": ["d1", "d2", "d3", "d4", "d5"]}}
        scores = {"q1": {doc: 0.0 for doc in c67["q1"]}}
        scores["q1"].update({"d5": 0.1, "d6": 2.0, "d7": 1.0})
        predictions, decisions = hsr_nested_head._select_slot5(split, ["q1"], base, scores, c67, 0.5)
        self.assertEqual(predictions["q1"]["answer"][:4], base["q1"]["answer"][:4])
        self.assertEqual(predictions["q1"]["answer"][4], "d6")
        self.assertTrue(decisions["q1"]["changed"])

    def test_training_examples_use_gold_and_native_negative_sampling(self) -> None:
        split = _Split()
        c67 = {"q1": [f"d{rank}" for rank in range(1, 68)]}
        h, z0, y, groups, audit = hsr_nested_head._training_examples(split, _cache(), ["q1"], c67)
        self.assertEqual(h.shape[1], 1024)
        self.assertEqual(int((y == 1).sum()), 1)
        self.assertEqual(int((y == 0).sum()), 8)
        self.assertEqual(len(groups), 1)
        self.assertEqual(audit["qids_with_positive"], 1)

    def test_frozen_e3b_validation_decision_is_the_baseline_source_of_truth(self) -> None:
        fold = SimpleNamespace(
            validation_qids=("q1",),
            reported_validation_decisions={
                "q1": {"changed": True, "replacement": "d6"}
            },
        )

        baseline = hsr_nested_head._base_from_frozen_e3b_decisions(_Split(), fold)

        self.assertEqual(baseline["q1"]["answer"], ["d1", "d2", "d3", "d4", "d6"])

    def test_vectorized_grouped_loss_matches_literal_value_and_gradients(self) -> None:
        groups = [([0, 2], [1, 4], 0.5), ([], [3, 5], 1.0), ([6], [], 0.25)]
        regularisation = 1.0e-3
        vector_scores = torch.tensor([0.4, -0.2, 1.1, 0.7, -1.2, 0.3, 0.9], dtype=torch.float64, requires_grad=True)
        vector_weight = torch.tensor([0.2, -0.4], dtype=torch.float64, requires_grad=True)
        vector_bias = torch.tensor(0.15, dtype=torch.float64, requires_grad=True)
        literal_scores = vector_scores.detach().clone().requires_grad_()
        literal_weight = vector_weight.detach().clone().requires_grad_()
        literal_bias = vector_bias.detach().clone().requires_grad_()

        layout = hsr_nested_head._build_grouped_loss_layout(groups, device="cpu")
        vector_loss = hsr_nested_head._grouped_loss(
            vector_scores, layout, vector_weight, vector_bias, regularisation
        )
        literal_loss = _literal_grouped_objective(
            literal_scores, groups, literal_weight, literal_bias, regularisation
        )
        vector_gradients = torch.autograd.grad(vector_loss, (vector_scores, vector_weight, vector_bias))
        literal_gradients = torch.autograd.grad(literal_loss, (literal_scores, literal_weight, literal_bias))

        self.assertTrue(torch.allclose(vector_loss, literal_loss, atol=1.0e-12, rtol=1.0e-12))
        for actual, expected in zip(vector_gradients, literal_gradients, strict=True):
            self.assertTrue(torch.allclose(actual, expected, atol=1.0e-12, rtol=1.0e-12))

    def test_runner_uses_distinct_eval_preflight_report_and_cache_root(self) -> None:
        runner = SCRIPTS.parent / "experiments" / "track_b_v11_research" / "run_hsr_c67_hidden_head.sh"
        source = runner.read_text(encoding="utf-8")

        self.assertIn('HIDDEN_CACHE_ROOT="${HIDDEN_CACHE_ROOT:-$OUTPUT_DIR}"', source)
        self.assertIn('--hidden-cache-root "$HIDDEN_CACHE_ROOT"', source)
        self.assertIn('hsr_eval_preflight_report.json', source)


if __name__ == "__main__":
    unittest.main()
