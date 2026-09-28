"""PV1 nested-eval tests: exact delta-Recall, C-selection, V report."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path("Task1 - LegalIR/scripts")))


class Pv1NestedEvalTests(unittest.TestCase):
    def test_delta_recall_action_labels(self) -> None:
        from pv1_nested_eval import delta_recall

        # Rescue: +1/n_q.
        d, a = delta_recall(["a", "b", "c", "d", "i1"], "g1", ["g1", "g2"])
        self.assertAlmostEqual(d, 0.5)
        self.assertEqual(a, "rescue")
        # Destructive: -1/n_q.
        d, a = delta_recall(["a", "b", "c", "d", "g1"], "x1", ["g1"])
        self.assertAlmostEqual(d, -1.0)
        self.assertEqual(a, "destructive")
        # Neutral gold->gold and nongold->nongold both delta 0.
        d, a = delta_recall(["a", "b", "c", "d", "g1"], "g1", ["g1", "g2"])
        self.assertAlmostEqual(d, 0.0)
        self.assertEqual(a, "neutral")
        d, a = delta_recall(["a", "b", "c", "d", "x0"], "x1", ["g1"])
        self.assertAlmostEqual(d, 0.0)
        self.assertEqual(a, "neutral")

    def test_recall_uses_gold_list_length(self) -> None:
        from pv1_nested_eval import recall_of

        self.assertAlmostEqual(recall_of(["g1", "a", "b", "c", "d"], ["g1", "g2"]), 0.5)

    def test_select_policy_prefers_gain_then_safety_then_abstain(self) -> None:
        from pv1_nested_eval import select_policy

        cands = {
            "tau_0": {"gain": 0.002, "destructive": 2, "changes": 10},
            "tau_ln3": {"gain": 0.002, "destructive": 1, "changes": 8},
            "abstain": {"gain": 0.0, "destructive": 0, "changes": 0},
        }
        self.assertEqual(select_policy(cands), "tau_ln3")
        flat = {
            "tau_0": {"gain": -0.001, "destructive": 3, "changes": 5},
            "tau_ln3": {"gain": 0.0, "destructive": 0, "changes": 2},
            "abstain": {"gain": 0.0, "destructive": 0, "changes": 0},
        }
        self.assertEqual(select_policy(flat), "abstain")

    def test_evaluate_counts_rescue_destructive_neutral_and_abstain(self) -> None:
        from pv1_nested_eval import evaluate_predictions

        anchor = {
            "q1": ["a", "b", "c", "d", "i1"],
            "q2": ["a", "b", "c", "d", "g9"],
            "q3": ["a", "b", "c", "d", "x0"],
        }
        pred = {
            "q1": ["a", "b", "c", "d", "g1"],  # rescue
            "q2": ["a", "b", "c", "d", "x1"],  # destructive
            "q3": ["a", "b", "c", "d", "x0"],  # abstain (unchanged)
        }
        gold = {"q1": ["g1"], "q2": ["g9"], "q3": ["g1"]}
        rep = evaluate_predictions(pred, gold, anchor)
        self.assertEqual(rep["rescue"], 1)
        self.assertEqual(rep["destructive"], 1)
        self.assertEqual(rep["neutral_swaps"], 0)
        self.assertEqual(rep["abstained"], 1)
        self.assertEqual(rep["queries"], 3)
        self.assertAlmostEqual(rep["gain"], 0.0)


if __name__ == "__main__":
    unittest.main()
