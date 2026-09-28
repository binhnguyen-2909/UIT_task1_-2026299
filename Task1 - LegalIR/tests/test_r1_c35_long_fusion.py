from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(ROOT))

import e3b_residual_score_gate as e3b_gate  # noqa: E402
from r1_contract import R1_LAMBDAS, apply_r1_policy, normalise_qwen, utility_rows  # noqa: E402


def _score(value: float) -> dict[str, object]:
    return {"document_mil_score": value, "passage_logits": [value], "empty_body": False}


class R1FusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.docs = [f"d{index}" for index in range(35)]
        self.anchor = ["d0", "d1", "d2", "d3", "d4"]
        self.split = SimpleNamespace(qids=["q1"], questions=["query"], candidates=[self.docs], anchor=[self.anchor], references=[{"d10"}])
        seed1 = {"q1": {doc: _score(float(index)) for index, doc in enumerate(self.docs)}}
        seed2 = {"q1": {doc: _score(float(index) + (0.1 if index == 10 else 0.0)) for index, doc in enumerate(self.docs)}}
        self.e2 = [seed1, seed2]
        self.qwen = {"q1": {doc: {"yes_no_logit": float(index) / 3.0} for index, doc in enumerate(self.docs)}}

    def test_qwen_degenerate_guard_is_finite(self) -> None:
        scores, degenerate = normalise_qwen({"a": {"yes_no_logit": 1.0}, "b": {"yes_no_logit": 1.0}}, ["a", "b"])
        self.assertTrue(degenerate)
        self.assertTrue(all(value == value and abs(value) < 10 for value in scores.values()))

    def test_lambda_zero_delegates_exact_e3b_policy(self) -> None:
        expected, expected_decisions = e3b_gate.apply_policy(self.split, [0], self.e2, {}, e3b_gate.E3BPolicy(memory_weight=0.0, threshold=0.50))
        actual, decisions = apply_r1_policy(self.split, [0], self.e2, self.qwen, 0.0)
        self.assertEqual(actual, expected)
        self.assertEqual(decisions["q1"]["original"], expected_decisions["q1"])

    def test_fusion_preserves_top4_and_only_locked_lambdas(self) -> None:
        proposals, diagnostic = utility_rows(qid="q1", anchor=self.anchor, c35=self.docs, e2_by_seed=self.e2, qwen_by_doc=self.qwen["q1"])
        self.assertTrue(diagnostic["incumbent_scoreable"])
        self.assertEqual(set(R1_LAMBDAS), {0.0, 0.25, 0.5})
        predictions, _ = apply_r1_policy(self.split, [0], self.e2, self.qwen, 0.25)
        self.assertEqual(predictions["q1"]["answer"][:4], self.anchor[:4])
        self.assertEqual(len(set(predictions["q1"]["answer"])), 5)
        self.assertTrue(any(row["doc_id"] == "d10" for row in proposals))


if __name__ == "__main__":
    unittest.main()
