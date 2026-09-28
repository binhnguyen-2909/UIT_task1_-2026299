from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gpu_transition_metrics import transition_metrics_device  # noqa: E402
from v27_a_common import transition_metrics  # noqa: E402


class GpuTransitionMetricsTests(unittest.TestCase):
    def test_cpu_device_matches_legacy_contract_exactly(self) -> None:
        split = SimpleNamespace(
            qids=["q1", "q2", "q3"],
            anchor=[["a", "b", "c", "d", "e"], ["f", "g", "h", "i", "j"], ["k", "l", "m", "n", "o"]],
            references=[{"z", "p"}, {"u", "j"}, {"k", "w"}],
        )
        prediction = {
            "q1": {"answer": ["a", "b", "c", "d", "z"]},
            "q2": {"answer": ["f", "g", "h", "i", "u"]},
            "q3": {"answer": ["k", "l", "m", "n", "o"]},
        }
        legacy = transition_metrics(split, prediction)
        device = transition_metrics_device(split, prediction, device="cpu")
        self.assertEqual(legacy, device)

    def test_pairwise_baseline_uses_supplied_predictions(self) -> None:
        split = SimpleNamespace(qids=["q"], anchor=[["a", "b", "c", "d", "e"]], references=[{"z"}])
        before = {"q": {"answer": ["a", "b", "c", "d", "e"]}}
        after = {"q": {"answer": ["a", "b", "c", "d", "z"]}}
        metrics = transition_metrics_device(split, after, baseline_predictions=before, device="cpu")
        self.assertEqual(metrics["gain"], 1.0)
        self.assertEqual(metrics["successful_rescues"], 1)


if __name__ == "__main__":
    unittest.main()
