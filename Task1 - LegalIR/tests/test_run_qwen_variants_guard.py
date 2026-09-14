from __future__ import annotations
from pathlib import Path
import unittest

RUNNER = Path("Task1 - LegalIR/experiments/track_b_v11_research/run_qwen_c200_research.sh")

class VariantsGuardTests(unittest.TestCase):
    def test_runner_accepts_variants_env_and_passes_to_cache(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn('VARIANTS="${VARIANTS:-A_c35_short}"', text)
        self.assertIn("--variants $VARIANTS", text)

    def test_cache_mode_does_not_hardcode_d_c200_long(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        cache_block = text.split("cache)")[1].split(";;")[0]
        self.assertNotIn("D_c200_long", cache_block)

    def test_tokenizer_parallelism_is_configurable_default_on(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn('TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"', text)
