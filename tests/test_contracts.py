from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from model_policy import validate_model_plan
from submission_utils import load_submission, validate_submission


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TASK1_SCRIPTS = ROOT / "Task1 - LegalIR" / "scripts"
sys.path.insert(0, str(TASK1_SCRIPTS))
TASK1_EVAL = load_module("task1_evaluate_test", TASK1_SCRIPTS / "evaluate.py")
sys.path.remove(str(TASK1_SCRIPTS))
TASK2_EVAL = load_module("task2_evaluate_test", ROOT / "Task2 - LegalQA" / "scripts" / "evaluate.py")


class ContractTests(unittest.TestCase):
    def test_task1_metric_matches_duplicate_penalty(self):
        metrics = TASK1_EVAL.official_metrics(
            {"q": {"answer": ["doc", "doc"]}}, {"q": ["doc"]}
        )
        self.assertEqual(metrics, {"precision": 0.5, "recall": 1.0})

    def test_task2_rouge_ascii_tokenizer(self):
        self.assertEqual(TASK2_EVAL.btc_rouge_tokens("Căn cứ Điều 76"), ["c", "n", "c", "i", "u", "76"])
        self.assertEqual(TASK2_EVAL.rouge_l_fmeasure("Căn cứ Điều 76", "Căn cứ Điều 76"), 1.0)

    def test_strict_zip_and_task_contract(self):
        submission = {"q": {"answer": ["7"]}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("submission.json", json.dumps(submission))
            self.assertEqual(load_submission(path), submission)
        self.assertEqual(validate_submission(submission, 1, ["q"]), [])

    def test_neural_model_requires_registry(self):
        with self.assertRaises(ValueError):
            validate_model_plan(None, {"embedding": "D:/models/example"})
        self.assertEqual(validate_model_plan(None, {"embedding": None})["total_parameters"], 0)

    def test_approved_registry_and_parameter_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            registry = root / "registry.json"
            record = {
                "id": "approved-test-model",
                "component": "embedding",
                "local_path": "model",
                "parameters": 100,
                "license": "MIT",
                "registered": True,
                "btc_approved": True,
                "approval_evidence": "test-fixture",
            }
            registry.write_text(json.dumps({"models": [record]}), encoding="utf-8")
            result = validate_model_plan(registry, {"embedding": str(model)})
            self.assertEqual(result["total_parameters"], 100)
            record["parameters"] = 4_000_000_000
            registry.write_text(json.dumps({"models": [record]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_model_plan(registry, {"embedding": str(model)})


if __name__ == "__main__":
    unittest.main()
