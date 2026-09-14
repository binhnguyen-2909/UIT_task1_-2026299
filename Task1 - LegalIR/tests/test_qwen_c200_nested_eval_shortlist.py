from __future__ import annotations
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path("Task1 - LegalIR/scripts")))

class EvalShortlistTests(unittest.TestCase):
    def test_eval_knows_e_and_f_variants(self) -> None:
        import qwen_c200_nested_eval as ev
        # A/B/C/D path stays untouched; new E/F report path uses its own tuple.
        self.assertIn("E_c67_short", ev.EVAL_VARIANTS_EF)
        self.assertIn("F_shortlist8_long", ev.EVAL_VARIANTS_EF)
        self.assertNotIn("D_c200_long", ev.DEFAULT_EVAL_VARIANTS)
        self.assertIn("A_c35_short", ev.DEFAULT_EVAL_VARIANTS)
