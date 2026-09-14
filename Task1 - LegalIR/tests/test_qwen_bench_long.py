from __future__ import annotations
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path("Task1 - LegalIR/scripts")))

class BenchContractTests(unittest.TestCase):
    def test_bench_splits_timing_and_checks_eval_nograd(self) -> None:
        import qwen_bench_long as mod
        rep = mod.bench_one_batch(dry_run=True)
        self.assertIn("pack_s", rep)
        self.assertIn("tokenize_s", rep)
        self.assertIn("forward_s", rep)
        self.assertTrue(rep["eval_mode"])
        self.assertTrue(rep["no_grad"])
        self.assertEqual(rep["dtype"], "bfloat16")
