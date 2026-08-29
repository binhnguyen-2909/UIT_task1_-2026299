from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "Task1 - LegalIR" / "scripts"
SCRIPT = SCRIPTS / "v14_replacement_pipeline.py"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("v14_replacement_pipeline", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class V14ReplacementPipelineTests(unittest.TestCase):
    def fake_split(self):
        candidates = [str(index) for index in range(1, 36)]
        names = tuple(MODULE.BASE_FEATURES)
        block = np.zeros((35, len(names)), dtype=np.float32)
        columns = {name: index for index, name in enumerate(names)}
        block[9, columns["idf_recall_document"]] = 1.0
        block[11, columns["longest_phrase_tokens"]] = 12.0
        block[13, columns["rank_consensus_rr"]] = 2.0
        neural = {doc_id: 0.01 for doc_id in candidates}
        neural["10"] = 0.99
        v11 = {doc_id: 0.01 for doc_id in candidates}
        v11["12"] = 0.98
        return MODULE.AuditSplit(
            name="complement",
            qids=["q"],
            questions=["Co duoc thuc hien trong truong hop nay hay khong?"],
            candidates=[candidates],
            neural_scores=[neural],
            v11_scores=[v11],
            anchor={"q": {"answer": ["1", "2", "3", "4", "5"]}},
            references={"q": {"answer": ["12"]}},
            feature_blocks=[block],
            feature_names=names,
        )

    def test_prefilter_is_rank_6_to_35_and_unions_signals(self):
        split = self.fake_split()
        selected, trace = MODULE.prefilter_candidates(split, 0, 35, maximum=12)
        self.assertNotIn("1", selected)
        self.assertNotIn("5", selected)
        self.assertIn("10", selected)
        self.assertIn("12", selected)
        self.assertIn("14", selected)
        self.assertTrue(trace["signal_top"])

    def test_template_signature_records_question_type_and_modality(self):
        signature = MODULE.query_template_signature("Người lao động có được nghỉ trong trường hợp này hay không?")
        self.assertTrue(signature.startswith("permission|permission_question|"))
        self.assertIn("condition", signature)

    def test_probability_policy_changes_only_slot_five(self):
        split = self.fake_split()
        dataset = MODULE.PairDataset(
            "complement",
            np.zeros((2, 1), dtype=np.float32),
            np.asarray([1, 0], dtype=np.int8),
            np.asarray(["success", "neutral"]),
            np.asarray([0, 0], dtype=np.int32),
            np.asarray(["12", "14"]),
            np.asarray([True, True]),
            ("x",),
            {},
        )
        predictions, trace = MODULE.predictions_from_probabilities(
            split, dataset, np.asarray([0.92, 0.10]), threshold=0.90,
        )
        self.assertEqual(predictions["q"]["answer"], ["1", "2", "3", "4", "12"])
        self.assertTrue(trace["q"]["accepted"])


if __name__ == "__main__":
    unittest.main()
