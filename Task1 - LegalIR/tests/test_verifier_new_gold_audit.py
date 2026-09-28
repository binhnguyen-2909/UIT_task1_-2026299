from __future__ import annotations
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path("Task1 - LegalIR/scripts")))

from verifier_audit_lock import classify_swap


class NewGoldAuditTests(unittest.TestCase):
    def test_rescue(self) -> None:
        self.assertEqual(classify_swap(False, True), "rescue")

    def test_destructive(self) -> None:
        self.assertEqual(classify_swap(True, False), "destructive")

    def test_neutral_both_gold(self) -> None:
        self.assertEqual(classify_swap(True, True), "neutral")

    def test_neutral_both_nongold(self) -> None:
        self.assertEqual(classify_swap(False, False), "neutral")

    def test_qid_31410_is_neutral_gold_swap(self) -> None:
        # QID 31410: anchor already holds 3/4 golds, incumbent is gold,
        # outside gold exists but swap is gold->gold => delta 0 => neutral.
        self.assertEqual(classify_swap(True, True), "neutral")


class VerifierPromptTests(unittest.TestCase):
    def test_prompt_is_blinded(self) -> None:
        from qwen_pairwise_verifier import build_pair_prompt

        p = build_pair_prompt("q?", "EV_A", "EV_B")
        self.assertNotIn("incumbent", p.lower())
        self.assertNotIn("gold", p.lower())
        self.assertIn("EV_A", p)
        self.assertIn("EV_B", p)

    def test_order_swap_requires_both_orders(self) -> None:
        from qwen_pairwise_verifier import order_swap_consistent

        ab = {"challenger_margin": 0.5}
        ba = {"challenger_margin": 0.5}
        self.assertTrue(order_swap_consistent(ab, ba, 0.0))
        ab_low = {"challenger_margin": -0.1}
        self.assertFalse(order_swap_consistent(ab_low, ba, 0.0))


class ProbeManifestTests(unittest.TestCase):
    def test_probe_manifest_is_fixed(self) -> None:
        from verifier_probe_manifest import build_probe_manifest, budget_check

        m1 = build_probe_manifest()
        m2 = build_probe_manifest()
        self.assertEqual(m1["qids"], m2["qids"])
        self.assertEqual(len(m1["qids"]), 60)
        b = budget_check()
        self.assertEqual(b["complement_pairs_max"], 7182)
        self.assertEqual(b["public_pairs_max"], 6000)


if __name__ == "__main__":
    unittest.main()
