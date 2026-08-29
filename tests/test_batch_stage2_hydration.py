from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "Task1 - LegalIR" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "batch_stage2_rerank", SCRIPT_DIR / "batch_stage2_rerank.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BatchStage2HydrationTests(unittest.TestCase):
    def test_cached_passage_tokens_preserve_hydration_order(self):
        rows = [
            {"chunk_id": "d::2", "text": "thuế thu nhập cá nhân"},
            {"chunk_id": "d::1", "text": "đăng ký tạm trú"},
            {"chunk_id": "d::3", "text": "xóa đăng ký tạm trú"},
        ]
        uncached = {"d": rows}
        cached = {
            "d": [
                {**row, "tokens": frozenset(MODULE.normalize_tokens(row["text"]))}
                for row in rows
            ]
        }
        left = MODULE.hydrate("xóa tạm trú", ["d"], 2, uncached)
        right = MODULE.hydrate("xóa tạm trú", ["d"], 2, cached)
        self.assertEqual(left, right)
        self.assertEqual(
            right[0]["passages"],
            ["xóa đăng ký tạm trú", "đăng ký tạm trú"],
        )


if __name__ == "__main__":
    unittest.main()
