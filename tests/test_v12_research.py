from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "Task1 - LegalIR" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location("v12_research", SCRIPT_DIR / "v12_research.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class V12ResearchTests(unittest.TestCase):
    def test_aspect_parser_is_deterministic_and_keeps_legal_predicates(self):
        question = (
            "Người sử dụng lao động có được đơn phương chấm dứt hợp đồng lao động "
            "với lao động nữ trong thời gian nghỉ thai sản hay không?"
        )
        first = MODULE.parse_aspects(question)
        second = MODULE.parse_aspects(question)
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 6)
        joined = " | ".join(first)
        self.assertIn("nghi thai san", joined)
        self.assertIn("don phuong cham dut", joined)
        self.assertIn("co duoc", joined)

    def test_aspect_parser_falls_back_for_plain_question(self):
        aspects = MODULE.parse_aspects("Điều kiện cấp giấy phép là gì")
        self.assertGreaterEqual(len(aspects), 1)
        self.assertTrue(all(aspects))

    def test_greedy_evidence_prefers_complementary_passage(self):
        passages = [
            MODULE.ChunkEvidence("a", "Điều 1", "người lao động nghỉ thai sản sáu tháng"),
            MODULE.ChunkEvidence("b", "Điều 2", "người sử dụng lao động không được sa thải"),
            MODULE.ChunkEvidence("c", "Điều 3", "người lao động nghỉ thai sản sáu tháng"),
        ]
        query = "người sử dụng lao động có được sa thải lao động nữ nghỉ thai sản không"
        aspects = ["người sử dụng lao động không được sa thải", "lao động nữ nghỉ thai sản"]
        selected = MODULE.greedy_evidence_selection(
            passages, query, aspects,
            {"người": 3, "lao động": 3, "nghỉ": 2, "thai": 2, "sản": 2,
             "sử": 1, "dụng": 1, "sa": 1, "thải": 1},
            corpus_size=3,
            limit=2,
        )
        self.assertEqual(len(selected), 2)
        self.assertEqual({item.chunk_id for item in selected}, {"a", "b"})


if __name__ == "__main__":
    unittest.main()
