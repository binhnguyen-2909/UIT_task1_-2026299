from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import qwen_c200_common as common  # noqa: E402
import qwen_c200_cache as cache  # noqa: E402
import qwen_c200_bundle as bundle  # noqa: E402
import qwen_c200_nested_eval as evaluator  # noqa: E402
from v27_a_common import V27ContractError  # noqa: E402


class _Tokenizer:
    """Small token-boundary tokenizer sufficient to test packing semantics."""

    def __call__(self, value, add_special_tokens=False):  # type: ignore[no-untyped-def]
        return {"input_ids": self.encode(value, add_special_tokens=add_special_tokens)}

    def encode(self, value, add_special_tokens=False):  # type: ignore[no-untyped-def]
        return [index + 1 for index, _ in enumerate(str(value).split())]

    def decode(self, ids, skip_special_tokens=True):  # type: ignore[no-untyped-def]
        return " ".join(f"t{value}" for value in ids)


class QwenC200CommonTests(unittest.TestCase):
    def test_legal_pack_is_label_blind_and_bounded(self) -> None:
        document = common.RawDocument("d", "Nghị định 1", "Điều 1 chung chung.\n\nĐiều 2 điều kiện cấp phép 123/2026.\n\nĐiều 3 ngoại lệ.", "")
        packed, metadata = common.pack_document(_Tokenizer(), question="Điều kiện cấp phép 123/2026 là gì", document=document, max_tokens=11)
        self.assertLessEqual(metadata["packed_tokens"], 11)
        self.assertEqual(metadata["kind"], "legal_boundary_lexical_order")
        self.assertIn("[DOCUMENT]", packed)

    def test_slot5_oracle_is_a_diagnostic_not_prediction(self) -> None:
        split = SimpleNamespace(qids=["q"], anchor=[["a", "b", "c", "d", "e"]], references=[{"z"}])
        result = common.oracle_slot5(split, {"q": ["a", "b", "c", "d", "e", "z"]})
        self.assertEqual(result["slot5_oracle_recall"], 1.0)
        self.assertEqual(result["queries_without_gold_in_pool"], 0)

    def test_resume_discards_partial_qid_and_keeps_only_complete_qids(self) -> None:
        pool = [f"d{index}" for index in range(35)]
        universe = SimpleNamespace(split=SimpleNamespace(qids=["q"]), candidates={"q": tuple(pool)})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard.jsonl"
            common.write_jsonl(path, [{"qid": "q", "doc_id": "d0", "candidate_rank": 1}])
            self.assertEqual(cache._recover_completed_qids(path, universe, "A_c35_short"), set())
            self.assertEqual(path.read_text(encoding="utf-8"), "")
            common.write_jsonl(path, [{"qid": "q", "doc_id": doc_id, "candidate_rank": rank} for rank, doc_id in enumerate(pool, 1)])
            self.assertEqual(cache._recover_completed_qids(path, universe, "A_c35_short"), {"q"})
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 35)


class QwenSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = [f"d{index}" for index in range(35)]
        self.anchor = self.pool[:5]
        self.universe = SimpleNamespace(candidates={"q": tuple(self.pool)})
        self.split = SimpleNamespace(qids=["q"], anchor=[self.anchor], references=[{"d5"}])
        setattr(self.split, "_qwen_universe", self.universe)
        self.scores = {"q": {doc_id: {"yes_no_logit": float(-index)} for index, doc_id in enumerate(self.pool)}}
        self.scores["q"]["d5"]["yes_no_logit"] = 3.0
        self.scores["q"]["d4"]["yes_no_logit"] = 0.5

    def test_s1_replaces_only_slot5_when_gap_is_strict(self) -> None:
        predictions, decisions = evaluator._s1_predictions(self.split, [0], self.scores, "A_c35_short", 1.0)
        self.assertEqual(predictions["q"]["answer"], self.anchor[:4] + ["d5"])
        self.assertTrue(decisions["q"]["changed"])
        abstain, _ = evaluator._s1_predictions(self.split, [0], self.scores, "A_c35_short", 3.0)
        self.assertEqual(abstain["q"]["answer"], self.anchor)

    def test_s1_rejects_anchor_outside_pool(self) -> None:
        self.split.anchor[0][4] = "outside"
        with self.assertRaises(V27ContractError):
            evaluator._s1_predictions(self.split, [0], self.scores, "A_c35_short", 0.0)

    def test_configuration_selection_defaults_to_abstain(self) -> None:
        name, selected, trace = evaluator._select_configuration([
            {"name": "A/S1", "calibration_metrics": {"gain": 0.0, "successful_rescues": 0, "destructive_changes": 0}},
        ])
        self.assertIsNone(name)
        self.assertEqual(selected, {})
        self.assertFalse(trace[0]["eligible"])

    def test_s2_fit_retains_verified_last_selected_and_backup(self) -> None:
        pool = [f"d{index}" for index in range(35)]
        universe = SimpleNamespace(candidates={"q0": tuple(pool), "q1": tuple(pool)})
        split = SimpleNamespace(
            qids=["q0", "q1"],
            anchor=[pool[:5], pool[:5]],
            references=[{"d5"}, {"d6"}],
        )
        setattr(split, "_qwen_universe", universe)
        scores = {
            qid: {doc_id: {"yes_no_logit": float(-index)} for index, doc_id in enumerate(pool)}
            for qid in split.qids
        }
        scores["q0"]["d5"]["yes_no_logit"] = 2.0
        scores["q1"]["d6"]["yes_no_logit"] = 2.0
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output_dir=Path(directory), device="cpu", s2_hidden=4, s2_epochs=4, s2_patience=2, s2_learning_rate=0.01)
            _, report = evaluator._fit_s2(
                split, [0], [1], scores, "A_c35_short", args=args, seed=11,
                regime="standard", inner_seed=22, outer_fold=0, lineage_sha256="lineage",
                partition_metadata={"a_fit": {"qids": ["q0"]}, "a_stop": {"qids": ["q1"]}},
            )
            artifacts = report["artifacts"]
            self.assertTrue(report["checkpoint_verified"])
            self.assertTrue(Path(artifacts["last"]["checkpoint"]).is_file())
            self.assertTrue(Path(artifacts["selected"]["checkpoint"]).is_file())
            self.assertEqual(artifacts["selected"]["sha256"], artifacts["selected_backup"]["sha256"])
            self.assertTrue(Path(artifacts["complete_marker"]).is_file())
            _, resumed = evaluator._fit_s2(
                split, [0], [1], scores, "A_c35_short", args=args, seed=11,
                regime="standard", inner_seed=22, outer_fold=0, lineage_sha256="lineage",
                partition_metadata={"a_fit": {"qids": ["q0"]}, "a_stop": {"qids": ["q1"]}},
            )
            self.assertTrue(resumed["checkpoint_verified"])
            self.assertEqual(resumed["artifacts"]["selected"]["sha256"], artifacts["selected"]["sha256"])
            model_dir = Path(directory) / "qwen_model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "model.safetensors").write_bytes(b"weights")
            fingerprint = common.checkpoint_fingerprint(model_dir)
            result = {
                "kind": evaluator.KIND,
                "holdout_opened": False,
                "public_labels_used": False,
                "checkpoint": {"resumable": True},
                "lineage_sha256": "lineage",
                "lineage": {"qwen_model_fingerprint": fingerprint},
                "regimes": {"standard": [{"folds": [{"calibration_configuration_trace": [{"fit": report}]}]}]},
            }
            result_path = Path(directory) / "qwen_result.json"
            result_path.write_text(__import__("json").dumps(result), encoding="utf-8")
            frozen = []
            for name in ("stage1.json", "stage2.json", "anchor.json", "reference.json", "e1.jsonl", "e3b_standard.json", "e3b_stress.json"):
                path = Path(directory) / name
                path.write_text("{}", encoding="utf-8")
                frozen.append(path)
            bundle_report = bundle.run(SimpleNamespace(
                output_dir=Path(directory), result_report=result_path,
                stage1=frozen[0], stage2=frozen[1], anchor=frozen[2], reference=frozen[3],
                e1_manifest=frozen[4], e3b_standard_report=frozen[5], e3b_stress_report=frozen[6],
                qwen_model=model_dir, output_path=Path(directory) / "reproduction.tar.gz",
            ))
            self.assertTrue(Path(bundle_report["bundle"]).is_file())
            self.assertGreater(bundle_report["retained_s2_artifacts"], 0)


if __name__ == "__main__":
    unittest.main()
