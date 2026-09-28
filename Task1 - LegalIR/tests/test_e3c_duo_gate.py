from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import e3c_duo_gate as gate  # noqa: E402
from v27_a_common import V27ContractError  # noqa: E402


class E3CDuoGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.qid = "q"
        self.c35 = [f"d{value}" for value in range(35)]
        self.anchor = self.c35[:5]
        self.split = SimpleNamespace(
            qids=[self.qid], anchor=[self.anchor], candidates=[self.c35], references=[["d5"]],
        )

    def test_target_is_exact_recall_delta(self) -> None:
        self.assertEqual(gate.soft_delta_target(gold=["d5", "d9"], incumbent="d4", candidate="d5"), (0.5, 0.75))
        self.assertEqual(gate.soft_delta_target(gold=["d5", "d9"], incumbent="d5", candidate="d9"), (0.0, 0.5))
        self.assertEqual(gate.soft_delta_target(gold=["d5", "d9"], incumbent="d5", candidate="d8"), (-0.5, 0.25))

    def test_replaces_only_slot_five_at_strict_margin(self) -> None:
        prediction, decision = gate.apply_policy(
            self.split, [0], {self.qid: {"d5": 0.05, "d6": 0.01}}, gate.E3CPolicy(0.02),
        )
        self.assertEqual(prediction[self.qid]["answer"], self.anchor[:4] + ["d5"])
        self.assertTrue(decision[self.qid]["changed"])
        tie, tied = gate.apply_policy(self.split, [0], {self.qid: {"d5": 0.02}}, gate.E3CPolicy(0.02))
        self.assertEqual(tie[self.qid]["answer"], self.anchor)
        self.assertFalse(tied[self.qid]["changed"])

    def test_validation_requires_one_row_per_competitor(self) -> None:
        rows = [
            {"qid": self.qid, "candidate_doc_id": doc_id, "incumbent_doc_id": "d4", "eligible": doc_id != "d5", "predicted_delta": 0.1 if doc_id != "d5" else None}
            for doc_id in self.c35[5:]
        ]
        parsed = gate.validate_duo_scores(rows, split=self.split, expected_indices=[0])
        self.assertNotIn("d5", parsed[self.qid])
        self.assertEqual(len(parsed[self.qid]), 29)
        with self.assertRaises(V27ContractError):
            gate.validate_duo_scores(rows[:-1], split=self.split, expected_indices=[0])

    def test_top_four_cannot_be_replaced(self) -> None:
        prediction, _ = gate.apply_policy(
            self.split, [0], {self.qid: {"d0": 0.99, "d5": 0.1}}, gate.E3CPolicy(0.0),
        )
        self.assertEqual(prediction[self.qid]["answer"][:4], self.anchor[:4])
        self.assertEqual(prediction[self.qid]["answer"][4], "d5")


if __name__ == "__main__":
    unittest.main()
