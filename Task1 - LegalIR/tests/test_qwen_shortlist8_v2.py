"""F-shortlist v2 tests: drop tail quota, keep 8 docs label-blind."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path("Task1 - LegalIR/scripts")))


def _universe():
    ranking = ["inc"] + [f"d{i}" for i in range(66)]
    anchor = ["a1", "a2", "a3", "a4", "inc"]
    # short scores: gold-ish docs d5,d6 score high but live in top, tail docs low
    scores = {d: float(60 - i) for i, d in enumerate(ranking)}
    return anchor, ranking, scores


class FShortlistV2Tests(unittest.TestCase):
    def test_keeps_eight_unique_subset_c67_with_incumbent(self) -> None:
        from qwen_shortlist8_v2 import select_shortlist8_v2

        anchor, ranking, scores = _universe()
        got = select_shortlist8_v2("q", anchor, ranking, scores)
        self.assertEqual(len(got), 8)
        self.assertEqual(len(set(got)), 8)
        self.assertIn("inc", got)
        self.assertTrue(set(got) <= set(ranking[:67]))

    def test_prefers_short_score_over_tail_quota(self) -> None:
        from qwen_shortlist8_v2 import select_shortlist8_v2

        anchor, ranking, scores = _universe()
        # tail docs rank 36..66 get artificially boosted under old policy;
        # v2 must pick top short scores outside top4/inc instead.
        scores = dict(scores)
        for d in ranking[36:67]:
            scores[d] = -1000.0
        got = select_shortlist8_v2("q", anchor, ranking, scores)
        # top outside by score are d5..d11 (d0..d3 are top4? no: anchor a1..a4 not in ranking,
        # so outside = all ranking; top7 = d0..d6) — assert no tail doc picked
        self.assertTrue(all(d not in ranking[36:67] for d in got if d != "inc"))

    def test_deterministic(self) -> None:
        from qwen_shortlist8_v2 import select_shortlist8_v2

        anchor, ranking, scores = _universe()
        self.assertEqual(
            select_shortlist8_v2("q", anchor, ranking, scores),
            select_shortlist8_v2("q", anchor, ranking, scores),
        )


if __name__ == "__main__":
    unittest.main()
