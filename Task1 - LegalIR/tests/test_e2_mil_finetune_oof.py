"""Small CPU contracts for E2 losses and evidence training rows."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from e2_mil_finetune_oof import (  # noqa: E402
    V27ContractError,
    _checkpoint_key,
    _load_frozen_scores,
    _report_path,
    _restore_completed_fold,
    score_full_c35,
    _training_examples,
    e2_loss,
)


class E2Contracts(unittest.TestCase):
    def test_examples_use_c35_positives_and_capped_negatives(self) -> None:
        candidates = [f"d{index}" for index in range(35)]
        split = SimpleNamespace(
            qids=["q1"],
            candidates=[candidates],
            references=[{"d2", "d3"}],
            anchor=[["d0", "d1", "d2", "d4", "d5"]],
            neural_scores=[{doc_id: float(35 - index) for index, doc_id in enumerate(candidates)}],
            source_candidates=[list(reversed(candidates))],
        )
        manifest = {
            ("q1", doc_id): {"qid": "q1", "doc_id": doc_id, "query": "q", "blocks": [{"text": "evidence"}], "empty_body": False}
            for doc_id in candidates
        }
        rows = _training_examples(split, manifest, {("q1", doc_id): float(index) for index, doc_id in enumerate(candidates)}, max_negatives=3)["q1"]
        self.assertEqual({row["doc_id"] for row in rows if row["label"] == 1}, {"d2", "d3"})
        self.assertEqual(sum(row["label"] == 0 for row in rows), 3)
        self.assertTrue(all("frozen_ce_score" in row for row in rows))

    def test_loss_contains_all_registered_terms(self) -> None:
        import torch

        rows = [
            {"label": 1, "candidate_rank": 5, "anchor_member": True, "frozen_ce_score": 1.0},
            {"label": 0, "candidate_rank": 4, "anchor_member": False, "frozen_ce_score": -1.0},
        ]
        total, pieces = e2_loss(torch.tensor([0.4, -0.2], requires_grad=True), rows)
        self.assertGreater(float(total), 0.0)
        self.assertEqual(set(pieces), {"relevance", "boundary", "anchor"})

    def test_empty_body_null_score_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.jsonl"
            path.write_text(
                '{"qid":"q1","doc_id":"empty","frozen_ce_score":null}\n'
                '{"qid":"q1","doc_id":"body","frozen_ce_score":0.0}\n',
                encoding="utf-8",
            )
            scores = _load_frozen_scores(path)
        self.assertIsNone(scores[("q1", "empty")])
        self.assertEqual(scores[("q1", "body")], 0.0)

    def test_fold_checkpoint_requires_the_exact_validation_partition(self) -> None:
        saved = {
            "validation_qids": ["q1"],
            "predictions": {"q1": {"answer": ["d1", "d2", "d3", "d4", "d5"]}},
            "fold_report": {"fold": 0},
        }
        predictions, report = _restore_completed_fold(
            saved, ["q1"], checkpoint_key=_checkpoint_key("standard", 7, 0),
        )
        self.assertEqual(predictions["q1"]["answer"][0], "d1")
        self.assertEqual(report["fold"], 0)
        with self.assertRaises(V27ContractError):
            _restore_completed_fold(saved, ["q2"], checkpoint_key="wrong-fold")

    def test_stress_report_can_be_kept_separate_from_standard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "output"
            explicit = output_dir / "e2_mil_oof_stress_report.json"
            self.assertEqual(_report_path(SimpleNamespace(output_dir=output_dir, report_path=explicit)), explicit)
            self.assertEqual(
                _report_path(SimpleNamespace(output_dir=output_dir, report_path=None)),
                output_dir / "e2_mil_oof_report.json",
            )

    def test_full_c35_export_keeps_empty_body_as_null(self) -> None:
        candidates = [f"d{index}" for index in range(35)]
        split = SimpleNamespace(qids=["q1"], candidates=[candidates])
        manifest = {
            ("q1", doc_id): {
                "qid": "q1", "doc_id": doc_id, "query": "q", "blocks": [{"text": "evidence"}],
                "empty_body": doc_id == "d3",
            }
            for doc_id in candidates
        }
        scores = {
            ("q1", doc_id): {"document_mil_score": float(index), "passage_logits": [float(index)]}
            for index, doc_id in enumerate(candidates)
            if doc_id != "d3"
        }
        with patch("e2_mil_finetune_oof.score_e2_rows_detailed", return_value=scores):
            rows = score_full_c35(split, [0], manifest, object(), object(), device="cpu", batch_size=1)
        self.assertEqual(len(rows), 35)
        empty = next(row for row in rows if row["doc_id"] == "d3")
        self.assertIsNone(empty["document_mil_score"])
        self.assertEqual(empty["passage_logits"], [])


if __name__ == "__main__":
    unittest.main()
