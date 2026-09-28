"""PV1 debias tests: position-bias-cancelled challenger score, eval-only."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path("Task1 - LegalIR/scripts")))


class Pv1DebiasTests(unittest.TestCase):
    def test_pure_position_bias_cancels_to_zero(self) -> None:
        from pv1_debias import debiased_challenger_score

        # Model always votes Doc1 (+10) regardless of content.
        fwd = {"A": 10.0, "B": 0.0, "C": 0.0, "D": 0.0}  # inc_first: Doc1=inc
        rev = {"A": 10.0, "B": 0.0, "C": 0.0, "D": 0.0}  # chal_first: Doc1=chal
        self.assertAlmostEqual(debiased_challenger_score(fwd, rev), 0.0)

    def test_consistent_challenger_win_is_positive(self) -> None:
        from pv1_debias import debiased_challenger_score

        fwd = {"A": 0.0, "B": 5.0, "C": 0.0, "D": 0.0}
        rev = {"A": 5.0, "B": 0.0, "C": 0.0, "D": 0.0}
        self.assertGreater(debiased_challenger_score(fwd, rev), 0.0)

    def test_consistent_incumbent_win_is_negative(self) -> None:
        from pv1_debias import debiased_challenger_score

        fwd = {"A": 5.0, "B": 0.0, "C": 0.0, "D": 0.0}
        rev = {"A": 0.0, "B": 5.0, "C": 0.0, "D": 0.0}
        self.assertLess(debiased_challenger_score(fwd, rev), 0.0)

    def test_split_orders_abstains_near_zero(self) -> None:
        from pv1_debias import debiased_challenger_score

        fwd = {"A": 0.0, "B": 5.0, "C": 0.0, "D": 0.0}  # challenger wins forward
        rev = {"A": 0.0, "B": 5.0, "C": 0.0, "D": 0.0}  # incumbent wins reverse
        self.assertAlmostEqual(debiased_challenger_score(fwd, rev), 0.0)

    def test_neither_dominant_abstains_near_zero(self) -> None:
        from pv1_debias import debiased_challenger_score

        fwd = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 10.0}
        rev = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 10.0}
        self.assertAlmostEqual(debiased_challenger_score(fwd, rev), 0.0)


if __name__ == "__main__":
    unittest.main()
