from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "Task1 - LegalIR" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "dual_dense_stage1", SCRIPT_DIR / "dual_dense_stage1.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DualDenseStage1Tests(unittest.TestCase):
    def test_zero_dense_weights_preserve_base_order(self):
        base = ["a", "b", "c", "d"]
        result = MODULE.fuse_rankings(
            base, [["x", "a"], ["y", "b"]], [0.0, 0.0], 20, 2
        )
        self.assertEqual(result, base)

    def test_locked_prefix_and_deduplication(self):
        result = MODULE.fuse_rankings(
            ["a", "b", "c", "d"],
            [["x", "a", "c"], ["x", "y", "b"]],
            [1.0, 1.0],
            0,
            2,
        )
        self.assertEqual(result[:2], ["a", "b"])
        self.assertEqual(len(result), len(set(result)))
        self.assertEqual(result[2], "x")

    def test_weight_count_must_match_channels(self):
        with self.assertRaises(ValueError):
            MODULE.fuse_rankings(["a"], [["b"]], [], 20, 0)


if __name__ == "__main__":
    unittest.main()
