"""PV1 contract tests: preflight binds manifests/hashes, complement-only."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path("Task1 - LegalIR/scripts")))

TASK = Path("Task1 - LegalIR")


class Pv1ContractTests(unittest.TestCase):
    def test_preflight_binds_complement_hashes(self) -> None:
        from pv1_contract import bind_inputs

        rep = bind_inputs(
            stage1=TASK / "output/v10_9_prf/complement_safe_tail_stage1.audit.json",
            stage2=TASK / "output/v10_9_prf/complement_safe_tail_stage2_c35_p3.audit.json",
            anchor=TASK / "output/v16_rgm/complement_v16b_memory_slot5.json",
            reference=TASK
            / "experiments/track_b_v11_research/artifacts/complement_split/complement_queries_1200.json",
            e_manifest=TASK
            / "output/qwen_c200_research/qwen_ef_preflight_label_blind_manifest.json",
            f_report=TASK / "output/qwen_c200_research/qwen_f_cache_report.json",
            gate_report=TASK / "output/e3b_nested_mil/e3b_standard_report.json",
        )
        self.assertTrue(rep["anchor_sha256"].startswith("dd2c754"))
        self.assertTrue(rep["stage1_sha256"].startswith("4b16c371"))
        self.assertEqual(rep["queries"], 1197)
        self.assertFalse(rep["holdout_opened"])
        self.assertFalse(rep["public_labels_used"])

    def test_preflight_rejects_holdout_filename(self) -> None:
        from pv1_contract import bind_inputs

        with self.assertRaises(Exception):
            bind_inputs(
                stage1=TASK / "output/v10_9_prf/complement_safe_tail_stage1.audit.json",
                stage2=TASK / "output/v10_9_prf/complement_safe_tail_stage2_c35_p3.audit.json",
                anchor=Path("Task1 - LegalIR/output/holdout_anchor.json"),
                reference=TASK
                / "experiments/track_b_v11_research/artifacts/complement_split/complement_queries_1200.json",
                e_manifest=TASK
                / "output/qwen_c200_research/qwen_ef_preflight_label_blind_manifest.json",
                f_report=TASK / "output/qwen_c200_research/qwen_f_cache_report.json",
                gate_report=TASK / "output/e3b_nested_mil/e3b_standard_report.json",
            )


if __name__ == "__main__":
    unittest.main()
