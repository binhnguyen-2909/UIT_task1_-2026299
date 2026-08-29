from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "Task1 - LegalIR" / "scripts" / "v11_tiny_ltr.py"
SPEC = importlib.util.spec_from_file_location("v11_tiny_ltr", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class V11ListwiseTests(unittest.TestCase):
    def test_listwise_ranker_trains_and_scores_complete_matrix(self) -> None:
        matrix = np.asarray(
            [
                [3.0, 1.0], [1.0, 0.0], [0.0, 1.0],
                [0.0, 2.0], [2.0, 0.0], [0.0, 0.0],
            ],
            dtype=np.float32,
        )
        target = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.int8)
        model = MODULE.train_torch_listwise_ranker(
            matrix,
            target,
            questions=[0, 1],
            candidate_limit=3,
            config={
                "hidden": 0,
                "learning_rate": 0.01,
                "epochs": 2,
                "training_pool": 3,
                "weight_decay": 0.0,
                "query_batch_size": 2,
                "seed": 7,
            },
        )
        scores = MODULE.model_scores(model, matrix)
        self.assertEqual(scores.shape, (6,))
        self.assertTrue(np.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main()
