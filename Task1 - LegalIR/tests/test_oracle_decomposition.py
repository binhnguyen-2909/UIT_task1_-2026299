from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import e3b_oracle_decomposition as dec  # noqa: E402


def _q(anchor, nominated, top3, eligible, gold, gate_choice):
    return {
        "anchor": list(anchor),
        "nominated": nominated,
        "ranked": list(top3),
        "ranked_gaps": [1.0] * len(top3),
        "eligible": list(eligible),
        "c35": list(dict.fromkeys(list(anchor) + list(eligible) + ["p", "q", "r"])),
        "unscoreable": [],
        "gold": set(gold),
        "gate_choice": list(gate_choice),
    }


class OracleLevelTests(unittest.TestCase):
    def test_oracle_never_below_anchor_and_gate(self) -> None:
        queries = [
            _q(["a", "b", "c", "d", "e"], "x", ["x", "y"], ["x", "y", "z"], {"x"}, ["a", "b", "c", "d", "e"]),
            _q(["a", "b", "c", "d", "e"], "y", ["y"], ["y"], {"q"}, ["a", "b", "c", "d", "y"]),
            _q(["a", "b", "c", "d", "g"], None, [], ["h"], {"g"}, ["a", "b", "c", "d", "g"]),
        ]
        levels = dec.decompose_queries(queries)
        for row in levels:
            self.assertGreaterEqual(row["r1"], row["anchor_r"] - 1e-12)
            self.assertGreaterEqual(row["r3"], row["r1"] - 1e-12)
            self.assertGreaterEqual(row["r35"], row["r3"] - 1e-12)
            self.assertGreaterEqual(row["r1"], row["r0"] - 1e-12)

    def test_oracle_picks_rescue_when_present(self) -> None:
        queries = [
            _q(["a", "b", "c", "d", "e"], "x", ["x", "y"], ["x", "y", "z"], {"x"}, ["a", "b", "c", "d", "e"]),
        ]
        (row,) = dec.decompose_queries(queries)
        self.assertAlmostEqual(row["anchor_r"], 0.0)
        self.assertAlmostEqual(row["r1"], 1.0)
        self.assertEqual(row["bucket"], "gate_reject")

    def test_destructive_gate_choice_is_caught_by_r1(self) -> None:
        queries = [
            _q(["a", "b", "c", "d", "g"], "y", ["y"], ["y"], {"g"}, ["a", "b", "c", "d", "y"]),
        ]
        (row,) = dec.decompose_queries(queries)
        self.assertAlmostEqual(row["anchor_r"], 1.0)
        self.assertAlmostEqual(row["r0"], 0.0)
        self.assertAlmostEqual(row["r1"], 1.0)
        self.assertEqual(row["bucket"], "destructive_blockable")

    def test_no_gold_in_pool_reports_coverage_bucket(self) -> None:
        queries = [
            _q(["a", "b", "c", "d", "e"], "x", ["x"], ["x"], {"zzz"}, ["a", "b", "c", "d", "e"]),
        ]
        (row,) = dec.decompose_queries(queries)
        self.assertAlmostEqual(row["r35"], row["anchor_r"])
        self.assertEqual(row["bucket"], "no_gain_possible")

    def test_empty_gold_never_crashes(self) -> None:
        queries = [
            _q(["a", "b", "c", "d", "e"], "x", ["x"], ["x"], set(), ["a", "b", "c", "d", "x"]),
        ]
        (row,) = dec.decompose_queries(queries)
        self.assertAlmostEqual(row["r35"], 0.0)

    def test_multi_answer_fractional_recall(self) -> None:
        queries = [
            _q(["a", "b", "c", "d", "e"], "x", ["x"], ["x"], {"d", "x"}, ["a", "b", "c", "d", "e"]),
        ]
        (row,) = dec.decompose_queries(queries)
        self.assertAlmostEqual(row["anchor_r"], 0.5)
        self.assertAlmostEqual(row["r1"], 1.0)
        self.assertAlmostEqual(row["r35"], 1.0)
        self.assertEqual(row["bucket"], "gate_reject")

    def test_deep_split_signal_miss_vs_deep_ranked(self) -> None:
        deep = _q(
            ["a", "b", "c", "d", "e"], "w", ["w", "v", "u", "t", "x"], ["w", "v", "u", "t", "x"],
            {"x"}, ["a", "b", "c", "d", "e"],
        )
        miss = _q(
            ["a", "b", "c", "d", "e"], "w", ["w"], ["w", "x"], {"x"}, ["a", "b", "c", "d", "e"],
        )
        rows = dec.decompose_queries([deep, miss])
        self.assertEqual(
            [(r["bucket"], r["sub"]) for r in rows],
            [("deep_or_filtered", "deep_ranked"), ("deep_or_filtered", "signal_miss")],
        )

    def test_filter_blocked_flag_for_unscoreable_gold(self) -> None:
        row = _q(
            ["a", "b", "c", "d", "e"], None, [], [], {"zz"}, ["a", "b", "c", "d", "e"],
        )
        row["c35"] = ["a", "b", "c", "d", "e", "zz"]
        row["unscoreable"] = ["zz"]
        (out,) = dec.decompose_queries([row])
        self.assertTrue(out["filter_blocked"])
        self.assertFalse(out["gold_absent_c35"])


if __name__ == "__main__":
    unittest.main()
