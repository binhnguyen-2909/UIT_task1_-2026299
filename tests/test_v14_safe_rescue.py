from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "Task1 - LegalIR" / "scripts" / "v14_safe_rescue.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("v14_safe_rescue", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def fake_features(doc_id: str, rank: int, score: float, strong: bool) -> object:
    evidence = MODULE.EvidenceSet(
        passages=(), score=0.85 if strong else 0.15,
        coverage=0.8 if strong else 0.1,
        critical_coverage=0.8 if strong else 0.1,
        hierarchy=0.8 if strong else 0.0,
        modality=0.8 if strong else 0.0,
        number=0.0,
        redundancy=0.0,
        contradiction=0.0,
    )
    return MODULE.CandidateFeatures(
        doc_id=doc_id, candidate_rank=rank, neural_score=score, neural_rank=rank,
        constraint_scores={"action": 0.85 if strong else 0.1, "condition": 0.8 if strong else 0.1},
        element_mean=0.8 if strong else 0.1,
        element_min=0.8 if strong else 0.1,
        element_std=0.0,
        critical_min=0.8 if strong else 0.1,
        coverage_ratio=0.8 if strong else 0.1,
        rare_coverage=0.8 if strong else 0.1,
        modality=0.8 if strong else 0.0,
        contradiction=0.0,
        number_exact=0.0,
        structure=0.8 if strong else 0.0,
        lexical=0.8 if strong else 0.1,
        evidence=evidence,
    )


class V14SafeRescueTests(unittest.TestCase):
    def test_budget_is_strictly_below_four_billion(self):
        budget = MODULE.validate_model_budget()
        self.assertTrue(budget["under_4b"])
        self.assertEqual(budget["total_parameters"], 1_090_000_000)

    def test_parser_types_permission_question_and_condition(self):
        parsed = MODULE.parse_constraints(
            "Người sử dụng lao động có được đơn phương chấm dứt hợp đồng lao động "
            "với lao động nữ trong thời gian nghỉ thai sản hay không?"
        )
        self.assertEqual(parsed.modality, "permission_question")
        self.assertIn("actor", parsed.elements)
        self.assertIn("action", parsed.elements)
        self.assertIn("condition", parsed.elements)
        self.assertIn("nghi thai san", parsed.elements["condition"])
        self.assertIn("condition", parsed.critical)

    def test_exhaustive_evidence_can_cover_complementary_passages(self):
        query = "nguoi su dung lao dong don phuong cham dut hop dong khi nghi thai san"
        constraints = MODULE.parse_constraints(query)
        passages = [
            MODULE.Passage("d", "d::0", "Dieu 1", "Nguoi su dung lao dong don phuong cham dut hop dong"),
            MODULE.Passage("d", "d::1", "Dieu 1", "Trong thoi gian nghi thai san, lao dong nu duoc bao ve"),
            MODULE.Passage("d", "d::2", "Dieu 2", "Quy dinh khac khong lien quan"),
        ]
        evidence = MODULE.exhaustive_evidence_selection(
            passages, constraints, query, {"nguoi": 1, "su": 1, "dung": 1, "lao": 1, "dong": 1, "thai": 1, "san": 1}, 3
        )
        self.assertGreaterEqual(len(evidence.passages), 2)
        self.assertGreater(evidence.critical_coverage, 0.25)

    def test_safe_rescue_replaces_only_the_fifth_slot(self):
        features = {
            "a": fake_features("a", 1, 1.0, True),
            "b": fake_features("b", 2, 0.9, True),
            "c": fake_features("c", 3, 0.8, True),
            "d": fake_features("d", 4, 0.7, True),
            "e": fake_features("e", 5, 0.1, False),
            "f": fake_features("f", 6, 0.6, True),
        }
        answers, trace = MODULE.apply_safe_rescue(["a", "b", "c", "d", "e"], features)
        self.assertEqual(answers, ["a", "b", "c", "d", "f"])
        self.assertIsNotNone(trace)
        self.assertTrue(trace["accepted"])

    def test_safe_rescue_keeps_anchor_when_candidate_is_not_strong(self):
        features = {str(value): fake_features(str(value), value, 0.1, False) for value in range(1, 8)}
        answers, trace = MODULE.apply_safe_rescue(["1", "2", "3", "4", "5"], features)
        self.assertEqual(answers, ["1", "2", "3", "4", "5"])
        self.assertFalse(trace["accepted"])


if __name__ == "__main__":
    unittest.main()
