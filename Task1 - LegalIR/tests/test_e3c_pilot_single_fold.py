from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import e3c_pilot_single_fold as pilot  # noqa: E402
from v27_a_common import V27ContractError  # noqa: E402


class E3CPilotSingleFoldTests(unittest.TestCase):
    def test_pilot_pins_single_fold_single_seed_single_epoch(self) -> None:
        self.assertEqual(pilot.PILOT_REGIME, "standard")
        self.assertEqual(pilot.PILOT_INNER_SEED, 20260910)
        self.assertEqual(pilot.PILOT_OUTER_FOLD, 0)
        self.assertEqual(pilot.PILOT_EPOCHS, 1)
        self.assertEqual(pilot.PILOT_KIND, "legalir_e3c_pilot_single_fold")

    def test_pilot_requires_exact_single_fold_weight(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with self.assertRaises(FileNotFoundError):
                pilot.pilot_preflight(
                    e3b_root=root,
                    output_dir=root / "out",
                )

    def test_pilot_refuses_holdout_paths(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "holdout_evil"
            root.mkdir()
            with self.assertRaises(V27ContractError):
                pilot.pilot_preflight(
                    e3b_root=root,
                    output_dir=Path(raw) / "out",
                )

    def test_pilot_lineage_marks_exploratory_no_e4(self) -> None:
        lineage = pilot.pilot_lineage(
            e3b_weight_sha256="a" * 64,
            e3b_state_sha256="b" * 64,
        )
        self.assertEqual(lineage["kind"], "legalir_e3c_pilot_single_fold")
        self.assertFalse(lineage["promotion_to_e4_allowed"])
        self.assertFalse(lineage["holdout_opened"])
        self.assertTrue(lineage["exploratory_only"])
        self.assertEqual(lineage["epochs"], 1)


if __name__ == "__main__":
    unittest.main()
