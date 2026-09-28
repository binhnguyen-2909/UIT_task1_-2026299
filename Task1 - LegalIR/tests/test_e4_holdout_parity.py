from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import e4_holdout_parity as e4  # noqa: E402
from v27_a_common import V27ContractError  # noqa: E402


def _e3_report(mode: str, lineage_sha: str) -> dict[str, object]:
    report: dict[str, object] = {
        "kind": "legalir_e3b_nested_mil_full_c35_oof",
        "mode": mode,
        "holdout_opened": False,
        "public_labels_used": False,
        "lineage_contract_sha256": lineage_sha,
        "lineage_contract": {"e3b": {"neural_seeds": [20260908, 20260909]}},
    }
    if mode == "standard":
        report["promotion_to_stress_allowed"] = True
    else:
        report["promotion_to_e4_allowed"] = True
    return report


class E4ParityTests(unittest.TestCase):
    def test_matching_e3_reports_authorize_e4(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            standard = root / "standard.json"
            stress = root / "stress.json"
            standard.write_text(json.dumps(_e3_report("standard", "lineage")), encoding="utf-8")
            stress.write_text(json.dumps(_e3_report("stress", "lineage")), encoding="utf-8")
            authorization = e4.validate_e3b_authorization(standard, stress)
        self.assertEqual(authorization["e3b_lineage_contract_sha256"], "lineage")

    def test_mismatched_e3_lineage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            standard = root / "standard.json"
            stress = root / "stress.json"
            standard.write_text(json.dumps(_e3_report("standard", "a")), encoding="utf-8")
            stress.write_text(json.dumps(_e3_report("stress", "b")), encoding="utf-8")
            with self.assertRaises(V27ContractError):
                e4.validate_e3b_authorization(standard, stress)

    def test_prediction_validator_enforces_frozen_top4_and_c35(self) -> None:
        split = SimpleNamespace(
            qids=["q1"],
            anchor=[["a", "b", "c", "d", "e"]],
            candidates=[["a", "b", "c", "d", "e", "f", *[f"x{i}" for i in range(29)]]],
        )
        e4._validate_e4_predictions(split, {"q1": {"answer": ["a", "b", "c", "d", "f"]}})
        with self.assertRaises(V27ContractError):
            e4._validate_e4_predictions(split, {"q1": {"answer": ["a", "b", "x0", "d", "f"]}})
        with self.assertRaises(V27ContractError):
            e4._validate_e4_predictions(split, {"q1": {"answer": ["a", "b", "c", "d", "outside"]}})

    def test_promotion_requires_both_gain_and_safety(self) -> None:
        passed = e4._promotion_gate({"gain": 0.001, "successful_rescues": 4, "destructive_changes": 2})
        self.assertTrue(passed["pass"])
        harmed = e4._promotion_gate({"gain": 0.001, "successful_rescues": 2, "destructive_changes": 2})
        self.assertFalse(harmed["pass"])
        neutral = e4._promotion_gate({"gain": 0.0, "successful_rescues": 4, "destructive_changes": 0})
        self.assertFalse(neutral["pass"])

    def test_opened_ledger_prevents_second_parity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output_dir=Path(directory))
            preflight = {"lineage_contract_sha256": "lineage"}
            prepared = {"predictions_sha256": "predictions"}
            path = e4._open_holdout_ledger(args, preflight, prepared)
            self.assertTrue(path.is_file())
            with self.assertRaises(V27ContractError):
                e4._open_holdout_ledger(args, preflight, prepared)

    def test_existing_runtime_contract_is_reused_for_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_path = root / "e4_runtime_autoprobe.json"
            runtime = {
                "micro_batch": 4,
                "score_batch_size": 64,
                "gradient_accumulation": 16,
                "gradient_checkpointing": False,
                "max_length": 768,
                "precision": "fp32_master_weights_fp16_autocast",
                "optimizer": "AdamW",
            }
            runtime_path.write_text(json.dumps({
                "kind": "legalir_e3b_gpu_runtime_contract",
                "runtime": runtime,
                "runtime_sha256": e4.json_sha256(runtime),
            }), encoding="utf-8")
            args = SimpleNamespace(output_dir=root, runtime_contract=None, auto_probe=False, vram_cap_gib=None)
            selected, selected_path = e4._load_or_probe_runtime(args, SimpleNamespace())
        self.assertEqual(selected, runtime)
        self.assertEqual(selected_path.name, "e4_runtime_autoprobe.json")


if __name__ == "__main__":
    unittest.main()
