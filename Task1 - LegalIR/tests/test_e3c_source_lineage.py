from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import e3c_nested_duo_lora as e3c  # noqa: E402


class E3CSourceLineageTests(unittest.TestCase):
    def _fold(self, path: Path, digest: str) -> dict[str, object]:
        return {
            "outer_fold": 0,
            "neural_models": [{
                "neural_seed": e3c.E3C_SOURCE_NEURAL_SEED,
                "weights": {"state_sha256": "a" * 64, "retained_sha256": digest, "retained_path": str(path)},
            }],
        }

    def test_requires_exact_retained_weight_hash(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path = root / "e3b_weights" / "standard" / "inner_20260910_fold_1_neural_20260908.pt"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"retained-e3b-state")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            # Materialize all required fold paths with their matching fold record.
            folds = []
            for fold in range(5):
                target = root / "e3b_weights" / "standard" / f"inner_20260910_fold_{fold + 1}_neural_20260908.pt"
                target.write_bytes(b"retained-e3b-state")
                row = self._fold(target, digest)
                row["outer_fold"] = fold
                folds.append(row)
            report = {"regimes": {"standard": [
                {"inner_seed": 20260910, "folds": folds},
            ]}}
            records = e3c._validate_e3b_sources(
                root=root, reports={"standard": report, "stress": report}, inner_seeds=[20260910], regimes=["standard"],
            )
            self.assertEqual(len(records), 5)
            (root / "e3b_weights" / "standard" / "inner_20260910_fold_3_neural_20260908.pt").write_bytes(b"mutated")
            with self.assertRaises(Exception):
                e3c._validate_e3b_sources(
                    root=root, reports={"standard": report, "stress": report}, inner_seeds=[20260910], regimes=["standard"],
                )


if __name__ == "__main__":
    unittest.main()
