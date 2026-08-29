from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "Task1 - LegalIR" / "scripts" / "corpus_prf_stage1.py"
SPEC = importlib.util.spec_from_file_location("corpus_prf_stage1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CorpusPrfTests(unittest.TestCase):
    def test_expansion_terms_require_shared_discriminative_corpus_terms(self):
        frequencies = {"hiếm": 3, "chung": 900, "mới": 4, "hỏi": 5}
        terms = MODULE.expansion_terms(
            "câu hỏi",
            [["từ hiếm và chung"], ["hiếm mới"], ["chung"]],
            frequencies,
            total_chunks=1000,
            limit=4,
            min_support=2,
            max_frequency_ratio=0.08,
        )
        self.assertEqual(terms, ["hiếm"])

    def test_rrf_zero_weights_preserve_base_order(self):
        base = ["a", "b", "c"]
        self.assertEqual(
            MODULE.fuse_rankings(base, ["x", "a"], ["y"], 0.0, 0.0, 20),
            base,
        )

    def test_complementary_rank_can_enter_without_duplicates(self):
        result = MODULE.fuse_rankings(
            ["a", "b", "c"], ["x", "a", "b"], [], 1.0, 0.0, 0
        )
        self.assertEqual(len(result), len(set(result)))
        self.assertIn("x", result[:3])

    def test_locked_prefix_is_never_reordered(self):
        result = MODULE.fuse_rankings(
            ["a", "b", "c", "d"], ["x", "a", "b"], ["y"], 1.0, 1.0, 0, 2
        )
        self.assertEqual(result[:2], ["a", "b"])
        self.assertEqual(len(result), len(set(result)))


if __name__ == "__main__":
    unittest.main()
