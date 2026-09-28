from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import e3b_nested_mil_scores as e3b  # noqa: E402
from gpu_autoprobe import _choose_fastest_fit  # noqa: E402
from v27_a_common import V27ContractError, group_folds  # noqa: E402


class E3BNestedContracts(unittest.TestCase):
    def test_partitions_are_group_disjoint_and_cover_every_query(self) -> None:
        split = SimpleNamespace(qids=[f"q{index}" for index in range(100)])
        groups = np.asarray([f"g{index // 5}" for index in range(100)])
        outer_train, validation = group_folds(groups)[0]
        partitions, metadata = e3b.make_partitions(
            split=split,
            groups=groups,
            outer_train=outer_train,
            validation=validation,
            inner_seed=20260910,
            outer_fold=0,
        )
        all_members = set()
        all_groups = set()
        for name, values in partitions.items():
            members = set(values.tolist())
            self.assertFalse(all_members & members, name)
            all_members |= members
            values_groups = {groups[index] for index in members}
            self.assertFalse(all_groups & values_groups, name)
            all_groups |= values_groups
            self.assertEqual(metadata[name]["queries"], len(values))
        self.assertEqual(all_members, set(range(100)))

    def test_memory_prior_without_separation_disables_lambda(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            path.write_text(
                '{"kind":"legalir_v16_qrel_memory_features","queries":[{"query_id":"q","memory":{"documents":[{"doc_id":"d","memory_rank":1}]}}],"leakage_policy":{"qid_overlap":["q"],"labels_used_for_query_retrieval":false,"query_answers_used_only_for_oracle_metrics":true}}',
                encoding="utf-8",
            )
            ranks, contract = e3b._load_memory_prior(path, {"q"})
        self.assertEqual(ranks["q"]["d"], 1)
        self.assertFalse(contract["memory_prior_allowed"])

    def test_runtime_selection_uses_fastest_measured_safe_trial(self) -> None:
        selected = _choose_fastest_fit(
            [
                {"ok": True, "fits_budget": True, "elapsed_seconds": 4.0, "batch_size": 4},
                {"ok": True, "fits_budget": True, "elapsed_seconds": 2.0, "batch_size": 2},
                {"ok": True, "fits_budget": False, "elapsed_seconds": 1.0, "batch_size": 8},
            ], phase="training",
        )
        self.assertEqual(selected["batch_size"], 2)
        with self.assertRaises(RuntimeError):
            _choose_fastest_fit([{"ok": False, "reason": "cuda_oom"}], phase="training")

    def test_runtime_contract_cannot_change_registered_accumulation(self) -> None:
        args = SimpleNamespace(micro_batch=1, score_batch_size=1, gradient_accumulation=16, gradient_checkpointing=False)
        runtime = {
            "micro_batch": 8, "score_batch_size": 64, "gradient_accumulation": 16,
            "gradient_checkpointing": False, "max_length": 768,
            "precision": "fp32_master_weights_fp16_autocast", "optimizer": "AdamW",
        }
        e3b._apply_runtime(args, runtime)
        self.assertEqual((args.micro_batch, args.score_batch_size), (8, 64))
        runtime["gradient_accumulation"] = 8
        with self.assertRaises(V27ContractError):
            e3b._apply_runtime(args, runtime)


if __name__ == "__main__":
    unittest.main()
