"""PV1 scoring tests: pair margin, two-order utility, slot-5 decision."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path("Task1 - LegalIR/scripts")))


class Pv1ScorePairsTests(unittest.TestCase):
    def test_pair_margin_is_challenger_minus_logsumexp_others(self) -> None:
        from pv1_score_pairs import pair_margin

        logits = {"A": 0.0, "B": 2.0, "C": 0.0, "D": 0.0}
        self.assertAlmostEqual(pair_margin(logits, "B"), 2.0 - math.log(3.0))

    def test_utility_requires_challenger_only_both_orders(self) -> None:
        from pv1_score_pairs import challenger_utility

        fwd = {"A": 0.0, "B": 3.0, "C": 0.0, "D": 0.0}  # inc_first: challenger is B
        rev = {"A": 3.0, "B": 0.0, "C": 0.0, "D": 0.0}  # chall_first: challenger is A
        u = challenger_utility(fwd, rev)
        self.assertTrue(math.isfinite(u))
        bad = {"A": 3.0, "B": 0.0, "C": 0.0, "D": 0.0}  # forward argmax is incumbent
        self.assertEqual(challenger_utility(bad, rev), float("-inf"))
        both = {"A": 0.0, "B": 0.0, "C": 3.0, "D": 0.0}  # C wins one order
        self.assertEqual(challenger_utility(both, rev), float("-inf"))

    def test_utility_is_min_of_two_margins(self) -> None:
        from pv1_score_pairs import challenger_utility, pair_margin

        fwd = {"A": 0.0, "B": 3.0, "C": 1.0, "D": 0.0}
        rev = {"A": 2.0, "B": 0.0, "C": 0.0, "D": 0.0}
        self.assertAlmostEqual(
            challenger_utility(fwd, rev),
            min(pair_margin(fwd, "B"), pair_margin(rev, "A")),
        )

    def test_decide_uses_strictly_greater_and_abstain(self) -> None:
        from pv1_score_pairs import decide_slot5

        anchor = ["t1", "t2", "t3", "t4", "inc"]
        scored = [("c1", 0, 0.5), ("c2", 1, 1.5)]
        self.assertEqual(
            decide_slot5(anchor, scored, {"mode": "threshold", "tau": 0.0}),
            ["t1", "t2", "t3", "t4", "c2"],
        )
        # Strictly greater: utility == tau keeps the incumbent.
        self.assertEqual(
            decide_slot5(anchor, [("c1", 0, 0.0)], {"mode": "threshold", "tau": 0.0}),
            anchor,
        )
        self.assertEqual(
            decide_slot5(anchor, scored, {"mode": "abstain", "tau": None}), anchor
        )
        self.assertEqual(
            decide_slot5(anchor, [], {"mode": "threshold", "tau": 0.0}), anchor
        )

    def test_decide_preserves_top4_and_rejects_anchor_challenger(self) -> None:
        from pv1_score_pairs import decide_slot5

        anchor = ["t1", "t2", "t3", "t4", "inc"]
        got = decide_slot5(anchor, [("c9", 0, 5.0)], {"mode": "threshold", "tau": 0.0})
        self.assertEqual(got[:4], ["t1", "t2", "t3", "t4"])
        self.assertEqual(len(set(got)), 5)
        with self.assertRaises(Exception):
            decide_slot5(anchor, [("inc", 0, 5.0)], {"mode": "threshold", "tau": 0.0})


if __name__ == "__main__":
    unittest.main()
