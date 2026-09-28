from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(ROOT))

from pv1_contract import FQuery, FShortlistItem, PV1Universe, audit_opportunities, build_pairs  # noqa: E402
from pv1_nested_eval import _metrics, _predict, _utility  # noqa: E402


def _universe() -> tuple[PV1Universe, dict[str, FQuery]]:
    split = SimpleNamespace(
        qids=["q1", "q2"],
        questions=["question one", "question two"],
        anchor=[["a1", "a2", "a3", "a4", "inc1"], ["b1", "b2", "b3", "b4", "inc2"]],
        candidates=[["a1"] * 35, ["b1"] * 35],
        references=[{"c1"}, {"inc2"}],
    )
    c67 = {
        "q1": tuple(["a1", "a2", "a3", "a4", "inc1", "c1", "c2", "c3"] + [f"x{index}" for index in range(59)]),
        "q2": tuple(["b1", "b2", "b3", "b4", "inc2", "d1", "d2", "d3"] + [f"y{index}" for index in range(59)]),
    }
    f = {
        "q1": FQuery(tuple(FShortlistItem(doc_id, rank, float(10 - rank)) for rank, doc_id in enumerate(["c1", "c2", "c3", "a1", "a2", "a3", "a4", "inc1"], 1)), {doc_id: f"evidence {doc_id}" for doc_id in c67["q1"][:8]}),
        "q2": FQuery(tuple(FShortlistItem(doc_id, rank, float(10 - rank)) for rank, doc_id in enumerate(["d1", "d2", "d3", "b1", "b2", "b3", "b4", "inc2"], 1)), {doc_id: f"evidence {doc_id}" for doc_id in c67["q2"][:8]}),
    }
    return PV1Universe(split=split, c67=c67, stage1_sha256="1" * 64, stage2_sha256="2" * 64, anchor_sha256="3" * 64), f


class PV1PairwiseTests(unittest.TestCase):
    def test_pair_build_audit_and_slot5_metric(self) -> None:
        universe, f_queries = _universe()
        rows, manifest = build_pairs(universe=universe, f_queries=f_queries, f_manifest={"sha": "x"})
        self.assertEqual(len(rows), 12)  # two QIDs × three challengers × two orientations
        self.assertEqual(manifest["expected_rows_per_qid"], {"q1": 6, "q2": 6})
        audit = audit_opportunities(universe_with_labels=universe, f_queries=f_queries)
        self.assertEqual(audit["counts"]["has_rescue_shortlist"], 1)
        self.assertEqual(audit["counts"]["incumbent_gold"], 1)

        scores = {}
        for row in rows:
            qid, challenger, orientation = row["qid"], row["challenger_id"], row["orientation"]
            if qid == "q1" and challenger == "c1":
                logits = [0.0, 4.0, 1.0, 1.0] if orientation == "incumbent_first" else [4.0, 0.0, 1.0, 1.0]
            else:
                logits = [4.0, 0.0, 1.0, 1.0] if orientation == "incumbent_first" else [0.0, 4.0, 1.0, 1.0]
            scores[(qid, row["incumbent_id"], challenger, orientation)] = {"logits_abcd": logits, "f_rank": row["f_rank"], "f_score": row["f_score"]}
        predictions, decisions = _predict(universe.split, indices=[0, 1], pair_manifest=manifest, scores=scores, policy={"mode": "threshold", "tau": 0.0})
        self.assertEqual(predictions["q1"]["answer"][-1], "c1")
        self.assertEqual(predictions["q2"]["answer"][-1], "inc2")
        self.assertTrue(decisions["q1"]["changed"])
        metrics = _metrics(universe.split, predictions)
        self.assertEqual(metrics["successful_rescues"], 1)
        self.assertEqual(metrics["destructive_changes"], 0)
        self.assertAlmostEqual(metrics["gain"], 0.5)

    def test_both_orientation_challenger_only_is_required(self) -> None:
        forward = {"logits_abcd": [0.0, 3.0, 1.0, 1.0]}
        reverse = {"logits_abcd": [3.0, 0.0, 1.0, 1.0]}
        self.assertGreater(_utility(forward, reverse), 0.0)
        self.assertEqual(_utility(forward, {"logits_abcd": [0.0, 3.0, 1.0, 1.0]}), float("-inf"))


if __name__ == "__main__":
    unittest.main()
