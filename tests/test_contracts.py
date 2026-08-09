from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace


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
TASK1_STRUCTURE = load_module(
    "task1_legal_structure_test", TASK1_SCRIPTS / "legal_structure.py"
)
TASK1_RERANKER = load_module(
    "task1_reranker_test", TASK1_SCRIPTS / "reranker.py"
)
TASK1_RERANK_EVAL = load_module(
    "task1_evaluate_reranker_test", TASK1_SCRIPTS / "evaluate_reranker.py"
)
TASK1_MAIN = load_module("task1_main_test", TASK1_SCRIPTS / "main.py")
sys.path.remove(str(TASK1_SCRIPTS))
TASK2_EVAL = load_module(
    "task2_evaluate_test", ROOT / "Task2 - LegalQA" / "scripts" / "evaluate.py"
)


class ContractTests(unittest.TestCase):
    def test_legal_chunking_preserves_article_context(self):
        text = """
        CHƯƠNG I
        QUY ĐỊNH CHUNG

        Điều 1. Phạm vi điều chỉnh
        1. Văn bản này quy định về hồ sơ.
        a) Hồ sơ phải đầy đủ;
        b) Hồ sơ phải hợp lệ.

        CHƯƠNG II
        TRÁCH NHIỆM
        Điều 2. Cơ quan có trách nhiệm trả lời.
        """
        chunks = list(TASK1_STRUCTURE.iter_legal_chunks(text, 180, 20))
        article_one = [chunk for chunk in chunks if chunk.article_number == "1"]
        article_two = [chunk for chunk in chunks if chunk.article_number == "2"]
        self.assertTrue(article_one)
        self.assertTrue(article_two)
        self.assertTrue(all("Điều 1" in chunk.text for chunk in article_one))
        self.assertTrue(all("CHƯƠNG I" in chunk.text for chunk in article_one))
        self.assertTrue(all("CHƯƠNG II" in chunk.text for chunk in article_two))
        self.assertFalse(any("CHƯƠNG II" in chunk.text for chunk in article_one))

    def test_task1_metric_matches_duplicate_penalty(self):
        metrics = TASK1_EVAL.official_metrics(
            {"q": {"answer": ["doc", "doc"]}}, {"q": ["doc"]}
        )
        self.assertEqual(metrics, {"precision": 0.5, "recall": 1.0})

    def test_fallback_reranker_uses_best_passage_and_records_trace(self):
        reranker = TASK1_RERANKER.Reranker()
        ranked = reranker.rerank(
            "thuế thu nhập cá nhân",
            [{
                "doc_id": "1",
                "score": 0.0,
                "text": "đoạn không liên quan",
                "passages": [
                    "đoạn không liên quan",
                    "quy định về thuế thu nhập cá nhân",
                ],
            }],
        )
        self.assertEqual(ranked[0]["rerank_passage_index"], 1)
        self.assertIn("thuế thu nhập", ranked[0]["text"])
        self.assertEqual(reranker.last_trace["input_ids"], ["1"])
        self.assertEqual(reranker.last_trace["neural_ids"], ["1"])
        self.assertIn("1", reranker.last_trace["neural_scores"])

    def test_reranker_rrf_is_deterministic(self):
        lexical = [{"doc_id": value} for value in "abcdef"]
        neural = [{"doc_id": value} for value in "fedcba"]
        fused = TASK1_RERANKER.reciprocal_rank_fuse(
            lexical, neural, neural_weight=0.1, rrf_k=0, top_k=5
        )
        self.assertEqual([item["doc_id"] for item in fused], list("abcdf"))
        self.assertTrue(all("rerank_fusion_score" in item for item in fused))

    def test_exemplar_alignment_preserves_legal_conditions(self):
        query = "Diện tích tối thiểu tách thửa đất nông nghiệp tại Nam Định?"
        same = "Diện tích tối thiểu để tách thửa đất nông nghiệp ở Nam Định?"
        wrong = "Diện tích tối thiểu để tách thửa đất ở Bắc Ninh?"
        self.assertGreater(
            TASK1_RERANK_EVAL.exemplar_alignment(query, same),
            TASK1_RERANK_EVAL.exemplar_alignment(query, wrong),
        )

    def test_reranker_cache_meta_tracks_source_ranking_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.json"
            index = root / "index.sqlite3"
            data.write_text("{}", encoding="utf-8")
            index.write_bytes(b"sqlite")
            common = dict(
                data=data,
                index=index,
                limit=1,
                candidate_count=100,
                top_k_initial=100,
                passages_per_doc=2,
                embedding_model=None,
                dense_weight=0.05,
                question_dense_weight=0.02,
                question_dense_candidates=20,
                query_prefix="query: ",
                passage_prefix="passage: ",
            )
            without_sources = TASK1_RERANK_EVAL.expected_cache_meta(
                SimpleNamespace(**common, store_source_rankings=False)
            )
            with_sources = TASK1_RERANK_EVAL.expected_cache_meta(
                SimpleNamespace(**common, store_source_rankings=True)
            )
            self.assertFalse(without_sources["store_source_rankings"])
            self.assertTrue(with_sources["store_source_rankings"])

    def test_v10_cli_defaults(self):
        args = TASK1_MAIN.build_parser().parse_args([
            "infer", "--input", "public-official.json"
        ])
        self.assertEqual(args.backend, "sqlite")
        self.assertEqual(
            args.index_dir,
            ROOT / "Task1 - LegalIR" / "data" / "processed" / "v10_index",
        )
        self.assertEqual(
            args.output,
            ROOT
            / "Task1 - LegalIR"
            / "output"
            / "public_submission_v10_ce_rrf_top35.zip",
        )
        self.assertEqual(args.chunk_strategy, "fixed")
        self.assertEqual(args.rerank_candidates, 35)
        self.assertEqual(args.reranker_batch_size, 16)
        self.assertEqual(args.reranker_rrf_weight, 0.8)
        self.assertEqual(args.reranker_rrf_k, 5)
        self.assertFalse(hasattr(args, "ltr_model"))
        self.assertFalse(hasattr(args, "llm_reranker_model"))

    def test_task2_rouge_ascii_tokenizer(self):
        self.assertEqual(
            TASK2_EVAL.btc_rouge_tokens("Căn cứ Điều 76"),
            ["c", "n", "c", "i", "u", "76"],
        )
        self.assertEqual(
            TASK2_EVAL.rouge_l_fmeasure("Căn cứ Điều 76", "Căn cứ Điều 76"),
            1.0,
        )

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
        result = validate_model_plan(None, {"embedding": None})
        self.assertEqual(result["total_parameters"], 0)

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
