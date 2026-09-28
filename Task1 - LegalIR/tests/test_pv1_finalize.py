"""PV1 finalize tests: policy lock, prediction validation, promotion refusal."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path("Task1 - LegalIR/scripts")))


def _args(**over: object) -> dict:
    base: dict = {
        "policy": {"mode": "threshold", "tau": 0.0},
        "model_revision": "rev1",
        "tokenizer_hash": "tok1",
        "prompt_hash": "pv1_v1",
        "packer_hash": "packer0",
        "input_manifest_sha": "manifest0",
        "predictions": {"q1": ["t1", "t2", "t3", "t4", "c1"]},
        "anchor_map": {"q1": ["t1", "t2", "t3", "t4", "inc"]},
        "promotion_pass": True,
        "output_dir": Path(tempfile.mkdtemp(prefix="pv1fin_")),
    }
    base.update(over)
    return base


class Pv1FinalizeTests(unittest.TestCase):
    def test_finalize_writes_policy_with_hashes(self) -> None:
        from pv1_finalize import finalize_policy

        out = finalize_policy(**_args())
        doc = json.loads(Path(out).read_text(encoding="utf-8"))
        self.assertEqual(doc["policy"], {"mode": "threshold", "tau": 0.0})
        self.assertEqual(doc["model_revision"], "rev1")
        self.assertEqual(doc["input_manifest_sha"], "manifest0")

    def test_finalize_refuses_without_promotion(self) -> None:
        from pv1_finalize import finalize_policy

        with self.assertRaises(Exception):
            finalize_policy(**_args(promotion_pass=False))

    def test_finalize_validates_predictions(self) -> None:
        from pv1_finalize import finalize_policy

        with self.assertRaises(Exception):  # duplicate IDs
            finalize_policy(**_args(predictions={"q1": ["t1", "t1", "t3", "t4", "c1"]}))
        with self.assertRaises(Exception):  # top-4 changed
            finalize_policy(**_args(predictions={"q1": ["x1", "t2", "t3", "t4", "c1"]}))
        with self.assertRaises(Exception):  # QID set mismatch
            finalize_policy(
                **_args(predictions={"q2": ["t1", "t2", "t3", "t4", "c1"]})
            )


if __name__ == "__main__":
    unittest.main()
