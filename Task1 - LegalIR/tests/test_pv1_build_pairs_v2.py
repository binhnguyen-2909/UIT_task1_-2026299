"""PV1 contract v2 tests: challengers nominated by F score, not shortlist rank."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path("Task1 - LegalIR/scripts")))

from pv1_contract import FQuery, FShortlistItem, PV1Universe, build_pairs  # noqa: E402


def _universe_score_vs_rank() -> tuple[PV1Universe, dict[str, FQuery]]:
    # Rank order: c1(2) c2(3) c3(4) c4(5) c5(6) c6(7) c7(8); scores reversed so
    # score-first top3 = c7,c6,c5 while rank-first top3 = c1,c2,c3.
    split = SimpleNamespace(
        qids=["q1"],
        questions=["question one"],
        anchor=[["a1", "a2", "a3", "a4", "inc1"]],
        candidates=[["a1"] * 35],
        references=[{"c7"}],
    )
    c67 = {"q1": tuple(["a1", "a2", "a3", "a4", "inc1", "c1", "c2", "c3", "c4", "c5", "c6", "c7"] + [f"x{i}" for i in range(55)])}
    items = tuple(
        [FShortlistItem("inc1", 1, 9.0)]
        + [FShortlistItem(f"c{i}", i + 1, float(i)) for i in range(1, 8)]
    )
    texts = {d: f"evidence {d}" for d in ["inc1"] + [f"c{i}" for i in range(1, 8)]}
    return PV1Universe(split=split, c67=c67, stage1_sha256="1" * 64, stage2_sha256="2" * 64, anchor_sha256="3" * 64), {"q1": FQuery(items, texts)}


class Pv1BuildPairsV2Tests(unittest.TestCase):
    def test_challengers_nominated_by_f_score_first(self) -> None:
        universe, f_queries = _universe_score_vs_rank()
        rows, manifest = build_pairs(universe=universe, f_queries=f_queries, f_manifest={"sha": "x"})
        challengers = manifest["shortlist_checks"]["q1"]["challengers"]
        self.assertEqual(challengers, ["c7", "c6", "c5"])

    def test_lineage_pins_challenger_order(self) -> None:
        universe, f_queries = _universe_score_vs_rank()
        _, manifest = build_pairs(universe=universe, f_queries=f_queries, f_manifest={"sha": "x"})
        self.assertEqual(manifest["lineage"].get("challenger_order"), "f_score_first")


if __name__ == "__main__":
    unittest.main()
