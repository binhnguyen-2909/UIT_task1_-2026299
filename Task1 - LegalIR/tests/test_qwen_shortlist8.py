from __future__ import annotations
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path("Task1 - LegalIR/scripts")))
from qwen_shortlist8 import select_shortlist8

class Shortlist8Tests(unittest.TestCase):
    def test_selects_incumbent_plus_4_plus_3_no_top4(self) -> None:
        anchor = ["a1", "a2", "a3", "a4", "inc"]
        stage1 = anchor + ["c5", "c6", "c7", "c8"] + [f"x{i}" for i in range(58)]
        scores = {"c8": 9.0, "c7": 8.0, "c6": 7.0, "c5": 6.0, **{f"x{i}": 5.0 - i * 0.01 for i in range(58)}}
        got = select_shortlist8("q", anchor, stage1, scores)
        self.assertEqual(got[0], "inc")
        self.assertEqual(len(got), 8)
        for t in anchor[:4]:
            self.assertNotIn(t, got)
        self.assertEqual(len(set(got)), 8)

    def test_dedups_and_fills_by_short_score(self) -> None:
        anchor = ["a1", "a2", "a3", "a4", "inc"]
        stage1 = anchor + ["inc", "c5"] + [f"x{i}" for i in range(60)]
        scores = {"c5": 3.0, **{f"x{i}": 2.0 - i * 0.001 for i in range(60)}, "inc": 99.0}
        got = select_shortlist8("q", anchor, stage1, scores)
        self.assertEqual(len(got), len(set(got)))
        self.assertIn("inc", got)
