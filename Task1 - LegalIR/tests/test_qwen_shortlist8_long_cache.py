from __future__ import annotations
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path("Task1 - LegalIR/scripts")))

class FLongCacheContractTests(unittest.TestCase):
    def test_module_pins_long_tokens_and_max8(self) -> None:
        import qwen_shortlist8_long_cache as mod
        self.assertEqual(mod.LONG_MAX_TOKENS, 4096)
        self.assertEqual(mod.MAX_DOCS_PER_QUERY, 8)
        self.assertEqual(mod.VARIANT, "F_shortlist8_long")

    def test_preflight_rejects_holdout_paths(self) -> None:
        from pathlib import Path as P
        import qwen_shortlist8_long_cache as mod
        with self.assertRaises(Exception):
            mod.preflight(output_dir=P("/tmp/x"), short_cache_dir=P("/tmp/holdout_evil"))
