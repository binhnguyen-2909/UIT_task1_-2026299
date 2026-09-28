from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from e3b_residual_score_gate import E3BPolicy, ranked_challengers  # noqa: E402
import e3b_gate_recalibrate as rec  # noqa: E402


def _row(score: float) -> dict[str, object]:
    return {"document_mil_score": score, "passage_logits": [score], "empty_body": False}


def _scores(spec0: dict[str, float], spec1: dict[str, float], qid: str = "q"):
    return [{qid: {doc: _row(v) for doc, v in spec.items()}} for spec in (spec0, spec1)]


def _universe() -> tuple[list[str], list[str]]:
    top4 = ["t1", "t2", "t3", "t4"]
    others = ["inc", "A", "B"] + [f"f{i}" for i in range(28)]
    return top4, top4 + others


class RecalibrationParamTests(unittest.TestCase):
    def test_higher_disagreement_weight_prefers_agreement(self) -> None:
        top4, c35 = _universe()
        anchor = top4 + ["inc"]
        base = {d: 0.0 for d in c35}
        spec0 = dict(base, inc=1.0, A=6.0, B=3.0)
        spec1 = dict(base, inc=1.0, A=0.5, B=3.0)
        by_seed = _scores(spec0, spec1)
        policy = E3BPolicy(memory_weight=0.0, threshold=0.0)
        loose = ranked_challengers(
            qid="q", anchor=anchor, c35=c35, score_by_seed=by_seed,
            memory_by_qid={}, policy=policy, disagreement_weight=0.0,
        )
        strict = ranked_challengers(
            qid="q", anchor=anchor, c35=c35, score_by_seed=by_seed,
            memory_by_qid={}, policy=policy, disagreement_weight=10.0,
        )
        self.assertTrue(loose and strict)
        self.assertLessEqual(strict[0]["disagreement"], loose[0]["disagreement"])

    def test_consensus_drops_split_seed_challenger(self) -> None:
        top4, c35 = _universe()
        anchor = top4 + ["inc"]
        base = {d: 0.0 for d in c35}
        spec0 = dict(base, inc=1.0, A=4.0, B=2.0)
        spec1 = dict(base, inc=1.0, A=0.0, B=2.0)
        by_seed = _scores(spec0, spec1)
        policy = E3BPolicy(memory_weight=0.0, threshold=0.0)
        default = ranked_challengers(
            qid="q", anchor=anchor, c35=c35, score_by_seed=by_seed,
            memory_by_qid={}, policy=policy,
        )
        consensus = ranked_challengers(
            qid="q", anchor=anchor, c35=c35, score_by_seed=by_seed,
            memory_by_qid={}, policy=policy, require_consensus=True,
        )
        self.assertIn("A", [p["doc_id"] for p in default])
        self.assertNotIn("A", [p["doc_id"] for p in consensus])
        self.assertIn("B", [p["doc_id"] for p in consensus])

    def test_threshold_tie_abstains(self) -> None:
        ranked = [{"doc_id": "x", "utility": 0.25}]
        self.assertIsNone(rec.apply_threshold(ranked, 0.25))
        self.assertEqual(rec.apply_threshold(ranked, 0.249), "x")


class SelectionRuleTests(unittest.TestCase):
    def _trial(self, gain, rescues, destructive, threshold=0.25, mem=0.0):
        return (
            {"threshold": threshold, "memory_weight": mem, "consensus": False, "disagreement_weight": 0.25},
            {"gain": gain, "successful_rescues": rescues, "destructive_changes": destructive},
        )

    def test_picks_max_gain_then_safety(self) -> None:
        trials = [
            self._trial(0.01, 5, 2, threshold=0.1),
            self._trial(0.02, 6, 4, threshold=0.1),
            self._trial(0.02, 6, 1, threshold=0.1),
        ]
        best, _ = rec.select_best(trials, {"gain": 0.0, "successful_rescues": 0, "destructive_changes": 0})
        self.assertEqual(best["threshold"], 0.1)
        self.assertEqual(best["memory_weight"], 0.0)

    def test_tie_prefers_stronger_abstention_then_less_memory(self) -> None:
        trials = [
            self._trial(0.02, 6, 1, threshold=0.1, mem=0.1),
            self._trial(0.02, 6, 1, threshold=0.5, mem=0.1),
            self._trial(0.02, 6, 1, threshold=0.5, mem=0.0),
        ]
        best, _ = rec.select_best(trials, {"gain": 0.0, "successful_rescues": 0, "destructive_changes": 0})
        self.assertEqual((best["threshold"], best["memory_weight"]), (0.5, 0.0))

    def test_falls_back_to_abstain_without_safe_gain(self) -> None:
        trials = [self._trial(0.01, 2, 2), self._trial(-0.01, 0, 3)]
        best, metrics = rec.select_best(trials, {"gain": 0.0, "successful_rescues": 0, "destructive_changes": 0})
        self.assertIsNone(best)
        self.assertEqual(metrics["gain"], 0.0)


if __name__ == "__main__":
    unittest.main()
