from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "Task1 - LegalIR" / "scripts" / "v16_memory_oracle.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("v16_memory_oracle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class V16MemoryOracleTests(unittest.TestCase):
    def test_split_isolation_rejects_exact_qid_and_question_overlap(self):
        memory = {"m1": {"question": "Điều kiện cấp phép là gì?"}}
        query = {"q1": {"question": "Điều kiện cấp phép là gì?"}}
        with self.assertRaises(ValueError):
            MODULE.validate_split_isolation(memory, query)

        report = MODULE.validate_split_isolation(
            {"m1": {"question": "Một câu hỏi"}},
            {"q1": {"question": "Câu khác"}},
            allow_exact_overlap=True,
        )
        self.assertEqual(report["qid_overlap"], [])

        with self.assertRaises(ValueError):
            MODULE.validate_split_isolation(
                {"q1": {"question": "Một câu hỏi"}},
                {"q1": {"question": "Câu khác"}},
                allow_exact_overlap=True,
            )

        kept, removed = MODULE.drop_normalized_question_overlap(
            {
                "m1": {"question": "Điều kiện cấp phép là gì?"},
                "m2": {"question": "Câu còn lại"},
            },
            {"q1": {"question": "điều kiện cấp phép là gì?"}},
        )
        self.assertEqual(removed, ["m1"])
        self.assertEqual(set(kept), {"m2"})

    def test_memory_features_apply_document_degree_correction(self):
        memory = MODULE.QRELMemory(
            ["q1", "q2", "q3"],
            np.asarray([[1.0, 0.0], [0.99, 0.1], [0.0, 1.0]], dtype=np.float32),
            {
                "q1": ["popular"],
                "q2": ["popular"],
                "q3": ["rare"],
            },
            memory_k=3,
            temperature=0.2,
            degree_exponent=0.5,
        )
        result = memory.retrieve("target", np.asarray([1.0, 0.0], dtype=np.float32))
        by_id = {item["doc_id"]: item for item in result.documents}
        self.assertEqual(by_id["popular"]["document_qrel_degree"], 2)
        self.assertEqual(by_id["rare"]["document_qrel_degree"], 1)
        self.assertAlmostEqual(
            by_id["popular"]["degree_corrected_memory_score"],
            by_id["popular"]["memory_score"] / np.sqrt(2.0),
            places=6,
        )
        self.assertEqual(len(result.neighbors), 3)
        self.assertTrue({
            "memory_score", "prototype_max", "prototype_top3_mean",
            "neighbor_vote_count", "unique_neighbor_count",
            "neighbor_similarity_gap_1_2", "neighbor_entropy",
            "reciprocal_neighbor_score", "dense_lexical_neighbor_agreement",
            "document_qrel_degree", "degree_corrected_memory_score",
            "memory_rank", "is_memory_outside_c35", "memory_vs_anchor_margin",
        }.issubset(by_id["popular"]))

    def test_secondary_view_contributes_agreement_without_replacing_dense_score(self):
        memory = MODULE.QRELMemory(
            ["q1", "q2", "q3"],
            np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32),
            {"q1": ["d1"], "q2": ["d2"], "q3": ["d3"]},
            memory_k=2,
            temperature=0.2,
            secondary_embeddings={
                "lexical": np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
            },
        )
        result = memory.retrieve(
            "target",
            np.asarray([1.0, 0.0], dtype=np.float32),
            query_views={"lexical": np.asarray([1.0, 0.0, 0.0], dtype=np.float32)},
        )
        by_id = {item.question_id: item for item in result.neighbors}
        self.assertEqual(set(by_id), {"q1", "q2"})
        self.assertAlmostEqual(by_id["q1"].view_agreement, 0.5)
        self.assertAlmostEqual(by_id["q2"].view_agreement, 1.0)

    def test_fusion_preserves_anchor_and_appends_only_new_memory_documents(self):
        result = MODULE.MemoryResult(
            query_id="q",
            neighbors=(),
            query_entropy=0.0,
            documents=[
                {"doc_id": "anchor", "degree_corrected_memory_score": 0.9},
                {"doc_id": "memory-1", "degree_corrected_memory_score": 0.8},
                {"doc_id": "memory-2", "degree_corrected_memory_score": 0.7},
            ],
        )
        fused = MODULE.fuse_memory_candidates(
            ["anchor", "second", "third"], result, memory_top_k=3
        )
        self.assertEqual(fused, ["anchor", "second", "third", "memory-1", "memory-2"])

    def test_oracle_reports_gold_recovered_outside_anchor(self):
        result = MODULE.MemoryResult(
            query_id="q",
            neighbors=(),
            query_entropy=0.0,
            documents=[
                {
                    "doc_id": "gold-tail",
                    "degree_corrected_memory_score": 1.0,
                    "is_memory_outside_c35": True,
                }
            ],
        )
        metrics = MODULE.evaluate_oracle(
            [{"question_id": "q", "answer": ["gold-tail"]}],
            {"q": ["anchor"]},
            {"q": result},
            anchor_cutoff=1,
            memory_top_k=1,
        )
        self.assertEqual(metrics["anchor_oracle_recall"], 0.0)
        self.assertEqual(metrics["memory_union_oracle_recall"], 1.0)
        self.assertEqual(metrics["delta_oracle"], 1.0)
        self.assertEqual(metrics["gold_recovered_outside_anchor"], 1)


if __name__ == "__main__":
    unittest.main()
