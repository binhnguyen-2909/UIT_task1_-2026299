from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "Task1 - LegalIR" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location("v13_router", SCRIPT_DIR / "v13_router.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class V13RouterTests(unittest.TestCase):
    def test_diagnostics_identify_v12_only_and_union_upside(self):
        split = types.SimpleNamespace(
            qids=["q1", "q2", "q3", "q4"],
            references=[{"a"}, {"b"}, {"c"}, {"d"}],
        )
        v11h = {qid: {"answer": [answer]} for qid, answer in zip(split.qids, ["a", "x", "y", "d"])}
        v12 = {qid: {"answer": [answer]} for qid, answer in zip(split.qids, ["x", "b", "c", "d"])}
        result = MODULE.branch_diagnostics(split, v11h, v12, range(4))
        self.assertEqual(result["v12_only"], 2)
        self.assertEqual(result["both_correct"], 1)
        self.assertGreater(result["union_upside_vs_v11h"], 0.0)

    def test_router_uses_v12_only_above_locked_threshold(self):
        split = types.SimpleNamespace(qids=["q1", "q2"])
        v11h = {"q1": {"answer": ["a"]}, "q2": {"answer": ["b"]}}
        v12 = {"q1": {"answer": ["x"]}, "q2": {"answer": ["y"]}}
        result = MODULE.routed_predictions(
            split, v11h, v12, [0.2, 0.8], 0.5, [0, 1]
        )
        self.assertEqual(result["q1"]["answer"], ["a"])
        self.assertEqual(result["q2"]["answer"], ["y"])


if __name__ == "__main__":
    unittest.main()
