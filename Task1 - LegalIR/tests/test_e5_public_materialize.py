from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import e5_public_materialize as e5  # noqa: E402
from v27_a_common import V27ContractError  # noqa: E402


def _e4_report() -> dict[str, object]:
    return {
        "kind": "legalir_e4_locked_holdout_parity",
        "stage": "locked_holdout_parity_complete",
        "lineage_contract_sha256": "lineage",
        "locked_policy": {
            "kind": "e3b_full_c35_mil_gap",
            "candidate_limit": 35,
            "editable_slot": 5,
            "frozen_top_k": 4,
            "disagreement_weight": 0.25,
            "memory_weight": 0.0,
            "threshold": 0.50,
            "default": "v16b_anchor_abstain",
            "locked": True,
        },
        "promotion": {"pass": True, "promotion_to_e5_allowed": True},
        "holdout_access_count": 1,
        "public_predictions_materialized": False,
    }


class E5PublicTests(unittest.TestCase):
    def test_e5_requires_e4_pass(self) -> None:
        auth = e5.validate_e4_authorization(_e4_report())
        self.assertTrue(auth["promotion_to_e5_allowed"])
        bad = dict(_e4_report())
        bad["promotion"] = {"pass": False, "promotion_to_e5_allowed": False}
        with self.assertRaises(V27ContractError):
            e5.validate_e4_authorization(bad)

    def test_e5_rejects_policy_drift(self) -> None:
        bad = _e4_report()
        bad["locked_policy"] = dict(_e4_report()["locked_policy"])
        bad["locked_policy"]["threshold"] = 0.25
        with self.assertRaises(V27ContractError):
            e5.validate_e4_authorization(bad)

    def test_e5_predictions_hold_c35_top4_contract(self) -> None:
        split = SimpleNamespace(
            qids=["q1"],
            anchor=[["a", "b", "c", "d", "e"]],
            candidates=[["a", "b", "c", "d", "e", "f", *[f"x{i}" for i in range(29)]]],
        )
        e5.validate_e5_predictions(split, {"q1": {"answer": ["a", "b", "c", "d", "f"]}})
        with self.assertRaises(V27ContractError):
            e5.validate_e5_predictions(split, {"q1": {"answer": ["a", "b", "x0", "d", "f"]}})
        with self.assertRaises(V27ContractError):
            e5.validate_e5_predictions(split, {"q1": {"answer": ["a", "b", "c", "d", "outside"]}})

    def test_e5_package_format(self) -> None:
        payload = {"38096": {"answer": ["1", "2", "3", "4", "5"]}}
        self.assertTrue(e5.check_package_payload(payload, ["38096"]))
        with self.assertRaises(V27ContractError):
            e5.check_package_payload({"38096": {"answer": ["1", "2", "3", "4", "4"]}}, ["38096"])

    def test_e5_report_provenance(self) -> None:
        rep = e5.build_e5_report_skeleton(
            e4_report=_e4_report(),
            public_qids=["1"],
            prediction_sha="deadbeef",
        )
        self.assertEqual(rep["locked_policy"]["threshold"], 0.5)
        self.assertFalse(rep["public_labels_used"])
        self.assertFalse(rep["holdout_labels_opened"])
        self.assertTrue(rep["model_budget"]["under_4b"])


if __name__ == "__main__":
    unittest.main()
