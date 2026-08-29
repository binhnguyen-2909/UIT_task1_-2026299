from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "Task1 - LegalIR" / "scripts"
SCRIPT = SCRIPTS / "v14_colbert_late_interaction.py"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("v14_colbert_late_interaction", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class V14ColbertLateInteractionTests(unittest.TestCase):
    def test_exhaustive_union_maxsim_can_combine_complementary_passages(self):
        first = torch.tensor([0.9, 0.2, 0.1])
        second = torch.tensor([0.1, 0.8, 0.2])
        third = torch.tensor([0.2, 0.1, 0.7])
        row = MODULE.late_interaction_feature_vector(
            [first, second, third],
            torch.tensor([0.3, 0.3, 0.3]),
            [
                {"action": 0.9, "condition": 0.1},
                {"action": 0.2, "condition": 0.8},
                {"action": 0.1, "condition": 0.4},
            ],
            {"action": 0.1, "condition": 0.1},
            ("action", "condition"),
            ("action", "condition"),
        )
        columns = {name: index for index, name in enumerate(MODULE.COLBERT_DOCUMENT_FEATURES)}
        self.assertGreater(row[columns["full_best_combo2"]], row[columns["full_best_single"]])
        self.assertGreaterEqual(row[columns["full_best_combo3"]], row[columns["full_best_combo2"]])
        self.assertAlmostEqual(row[columns["critical_min"]], 0.8, places=6)
        self.assertEqual(len(row), len(MODULE.COLBERT_DOCUMENT_FEATURES))

    def test_gate_modes_are_label_free_and_keep_calibrated_all(self):
        base = MODULE.PairDataset(
            "complement",
            np.asarray([[0.01, 17.0], [-0.01, 20.0], [0.01, 28.0]], dtype=np.float32),
            np.asarray([1, 0, 0], dtype=np.int8),
            np.asarray(["success", "neutral", "neutral"]),
            np.asarray([0, 0, 0], dtype=np.int32),
            np.asarray(["a", "b", "c"]),
            np.asarray([False, True, False]),
            ("delta__cross_encoder_score", "candidate__stage1_rank"),
            {},
        )
        width = len(MODULE.COLBERT_DOCUMENT_FEATURES)
        delta = np.zeros((3, width), dtype=np.float32)
        columns = {name: index for index, name in enumerate(MODULE.COLBERT_DOCUMENT_FEATURES)}
        delta[0, columns["full_best_combo3"]] = 0.02
        delta[0, columns["critical_mean"]] = 0.01
        masks = MODULE.gate_masks(
            base,
            {
                "candidate": np.zeros_like(delta),
                "tail": np.zeros_like(delta),
                "delta": delta,
                "normalized": np.zeros((3, 1), dtype=np.float32),
            },
        )
        self.assertFalse(masks["phase_a_strict"][0])
        self.assertTrue(masks["phase_a_or_colbert_dominance"][0])
        self.assertTrue(masks["colbert_ce_rank25"][0])
        self.assertFalse(masks["colbert_ce_rank25"][1])
        self.assertFalse(masks["colbert_ce_rank25"][2])
        self.assertTrue(np.all(masks["calibrated_all"]))

    def test_parameter_budget_is_strictly_below_four_billion(self):
        budget = MODULE.parameter_budget(600_000_000, 100_000)
        self.assertTrue(budget["strictly_under_4b"])
        self.assertLess(budget["deployed_total_parameters"], 4_000_000_000)


if __name__ == "__main__":
    unittest.main()
