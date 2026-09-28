from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import e3b_residual_score_gate as gate  # noqa: E402
from v27_a_common import V27ContractError  # noqa: E402


class E3BResidualScoreGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.qid = "q"
        self.c35 = [f"d{number}" for number in range(35)]
        self.anchor = self.c35[:5]

    def _scores(self, *, incumbent: float = 0.0, challenger: float = 2.0, empty: set[str] | None = None) -> dict[str, dict[str, dict[str, object]]]:
        empty = empty or set()
        rows: dict[str, dict[str, object]] = {}
        for rank, doc_id in enumerate(self.c35, 1):
            if doc_id in empty:
                rows[doc_id] = {
                    "qid": self.qid, "doc_id": doc_id, "candidate_rank": rank,
                    "empty_body": True, "document_mil_score": None, "passage_logits": [],
                }
            else:
                score = challenger if doc_id == "d5" else incumbent if doc_id == "d4" else -1.0
                rows[doc_id] = {
                    "qid": self.qid, "doc_id": doc_id, "candidate_rank": rank,
                    "empty_body": False, "document_mil_score": score, "passage_logits": [score],
                }
        return {self.qid: rows}

    def test_neural_winner_replaces_only_slot_five(self) -> None:
        policy = gate.E3BPolicy(memory_weight=0.0, threshold=0.0)
        proposal = gate.propose_slot5(
            qid=self.qid, anchor=self.anchor, c35=self.c35,
            score_by_seed=[self._scores(challenger=2.0), self._scores(challenger=3.0)],
            memory_by_qid={}, policy=policy,
        )
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal["doc_id"], "d5")
        split = SimpleNamespace(
            qids=[self.qid], anchor=[self.anchor], candidates=[self.c35],
        )
        predictions, decisions = gate.apply_policy(
            split, [0], [self._scores(challenger=2.0), self._scores(challenger=3.0)], {}, policy,
        )
        self.assertEqual(predictions[self.qid]["answer"][:4], self.anchor[:4])
        self.assertEqual(predictions[self.qid]["answer"][4], "d5")
        self.assertTrue(decisions[self.qid]["changed"])

    def test_memory_cannot_promote_neural_loser(self) -> None:
        policy = gate.E3BPolicy(memory_weight=0.10, threshold=0.0)
        proposal = gate.propose_slot5(
            qid=self.qid, anchor=self.anchor, c35=self.c35,
            score_by_seed=[self._scores(incumbent=2.0, challenger=-3.0), self._scores(incumbent=1.0, challenger=-2.0)],
            memory_by_qid={self.qid: {"d5": 1, "d4": 35}}, policy=policy,
        )
        self.assertIsNone(proposal)

    def test_empty_incumbent_and_threshold_tie_abstain(self) -> None:
        policy = gate.E3BPolicy(memory_weight=0.0, threshold=0.0)
        empty_incumbent = gate.propose_slot5(
            qid=self.qid, anchor=self.anchor, c35=self.c35,
            score_by_seed=[self._scores(empty={"d4"}), self._scores(empty={"d4"})],
            memory_by_qid={}, policy=policy,
        )
        self.assertIsNone(empty_incumbent)
        tied = gate.propose_slot5(
            qid=self.qid, anchor=self.anchor, c35=self.c35,
            score_by_seed=[self._scores(challenger=0.0), self._scores(challenger=0.0)],
            memory_by_qid={}, policy=policy,
        )
        self.assertIsNone(tied)

    def test_score_export_requires_every_c35_document_once(self) -> None:
        rows = list(self._scores()[self.qid].values())[:-1]
        with self.assertRaises(V27ContractError):
            gate.validate_full_c35_scores(rows, c35_by_qid={self.qid: self.c35}, expected_qids=[self.qid])

    def test_paired_interval_resamples_groups_not_rows(self) -> None:
        split = SimpleNamespace(
            qids=["q1", "q2"],
            anchor=[["a", "b", "c", "d", "e"], ["f", "g", "h", "i", "j"]],
            references=[{"x"}, {"k"}],
        )
        predictions = {
            "q1": {"answer": ["a", "b", "c", "d", "x"]},
            "q2": {"answer": ["f", "g", "h", "i", "k"]},
        }
        interval = gate.paired_group_confidence_interval(split, predictions, ["g1", "g2"], seed=1, repetitions=50)
        self.assertEqual(interval["groups"], 2)
        self.assertEqual(interval["point_estimate"], 1.0)


if __name__ == "__main__":
    unittest.main()
