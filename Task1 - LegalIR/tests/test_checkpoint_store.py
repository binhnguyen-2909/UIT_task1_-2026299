from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import v27_a_common as common  # noqa: E402


class CheckpointStoreTests(unittest.TestCase):
    def test_atomic_save_and_verify_state_dict(self) -> None:
        torch = __import__("torch")
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "w" / "best.pt"
            payload = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor(3)}
            sha = common.atomic_torch_save(payload, path)
            self.assertEqual(len(sha), 64)
            self.assertTrue(path.is_file())
            leftovers = list(path.parent.glob(".best.pt.tmp*")) + list(path.parent.glob("best.pt.tmp"))
            self.assertEqual(leftovers, [])
            loaded = common.verify_torch_state_dict(path)
            self.assertEqual(set(loaded), {"a", "b"})
            self.assertTrue(bool((loaded["a"] == payload["a"]).all()))

    def test_verify_rejects_corrupted_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "bad.pt"
            path.write_bytes(b"not-a-checkpoint")
            with self.assertRaises(Exception):
                common.verify_torch_state_dict(path)

    def test_training_checkpoint_roundtrip_with_required_keys(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "ckpt.pt"
            payload = {
                "kind": "legalir_e3c_duo_training_checkpoint",
                "next_epoch": 1,
                "position": 0,
                "optimizer_state": {},
                "adapter_state": {"x": 1},
                "rng_state": {"python": 0},
                "history": [],
            }
            common.atomic_torch_save(payload, path)
            loaded = common.verify_torch_checkpoint(
                path,
                required_keys=(
                    "kind", "next_epoch", "optimizer_state",
                    "adapter_state", "rng_state", "history",
                ),
            )
            self.assertEqual(loaded["next_epoch"], 1)
            self.assertEqual(loaded["kind"], "legalir_e3c_duo_training_checkpoint")

    def test_weight_manifest_carries_lineage_fields(self) -> None:
        manifest = common.build_weight_manifest(
            regime="standard",
            outer_fold=0,
            inner_seed=20260910,
            neural_seed=20260908,
            fit_seed=20260908,
            split_qids_sha256="f" * 64,
            state_sha256="a" * 64,
            retained_sha256="b" * 64,
            weight_path="/tmp/best.pt",
        )
        for key in (
            "kind", "regime", "outer_fold", "inner_seed", "neural_seed",
            "fit_seed", "split_qids_sha256", "state_sha256",
            "retained_sha256", "weight_path", "created_at",
        ):
            self.assertIn(key, manifest)
        self.assertEqual(manifest["neural_seed"], 20260908)


if __name__ == "__main__":
    unittest.main()
