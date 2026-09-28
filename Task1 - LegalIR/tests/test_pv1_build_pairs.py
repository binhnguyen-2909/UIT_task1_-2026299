"""PV1 build-pairs tests: top-3 nomination, both orientations, input hash."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path("Task1 - LegalIR/scripts")))


class Pv1BuildPairsTests(unittest.TestCase):
    def _fixture(self) -> dict:
        anchor = ["t1", "t2", "t3", "t4", "inc"]
        shortlist = ["inc", "c1", "c2", "c3", "c4", "c5", "c6", "c7"]
        scores = {"inc": 9.0, "c1": 5.0, "c2": 7.0, "c3": 6.0, "c4": 1.0,
                  "c5": 0.5, "c6": 0.2, "c7": 0.1}
        texts = {d: f"TEXT-{d}" for d in shortlist}
        return {"anchor": anchor, "shortlist": shortlist, "scores": scores,
                "texts": texts}

    def test_selects_top3_by_f_score(self) -> None:
        from pv1_build_pairs import select_challengers

        fx = self._fixture()
        got = select_challengers(fx["anchor"], fx["shortlist"], fx["scores"])
        self.assertEqual([c for c, _, _ in got], ["c2", "c3", "c1"])

    def test_excludes_top4_incumbent_and_duplicates(self) -> None:
        from pv1_build_pairs import select_challengers

        fx = self._fixture()
        shortlist = ["inc", "t1", "c2", "c2", "t3", "c3", "c1", "c4"]
        got = select_challengers(fx["anchor"], shortlist, fx["scores"])
        ids = [c for c, _, _ in got]
        self.assertNotIn("inc", ids)
        self.assertFalse(set(ids) & {"t1", "t2", "t3", "t4"})
        self.assertEqual(len(ids), len(set(ids)))
        self.assertLessEqual(len(ids), 3)

    def test_pair_inputs_have_both_orientations_and_stable_hash(self) -> None:
        from pv1_build_pairs import build_pair_inputs

        fx = self._fixture()
        rows = build_pair_inputs(
            qid="q1", question="Q?", anchor=fx["anchor"],
            shortlist=fx["shortlist"], f_scores=fx["scores"],
            doc_texts=fx["texts"], packer_hash="packer0",
        )
        self.assertEqual(len(rows), 6)  # 3 challengers x 2 orientations
        by_chall: dict[str, set[str]] = {}
        for row in rows:
            by_chall.setdefault(row["challenger_id"], set()).add(row["orientation"])
        for chall, orients in by_chall.items():
            self.assertEqual(orients, {"inc_first", "chall_first"})
        again = build_pair_inputs(
            qid="q1", question="Q?", anchor=fx["anchor"],
            shortlist=fx["shortlist"], f_scores=fx["scores"],
            doc_texts=fx["texts"], packer_hash="packer0",
        )
        self.assertEqual([r["input_sha256"] for r in rows],
                         [r["input_sha256"] for r in again])
        # Orientation changes the input hash.
        first = [r for r in rows if r["challenger_id"] == "c2"]
        self.assertNotEqual(first[0]["input_sha256"], first[1]["input_sha256"])

    def test_no_challenger_keeps_anchor(self) -> None:
        from pv1_build_pairs import build_pair_inputs

        rows = build_pair_inputs(
            qid="q9", question="Q?", anchor=["t1", "t2", "t3", "t4", "inc"],
            shortlist=["inc"], f_scores={"inc": 1.0},
            doc_texts={"inc": "TEXT-inc"}, packer_hash="packer0",
        )
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
