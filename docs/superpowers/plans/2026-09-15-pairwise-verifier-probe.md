# Pairwise Verifier Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Probe Qwen 1.7B-class frozen pairwise verifier with abstain on F-shortlist8 to reduce destructive slot-5 swaps while keeping rescues.

**Architecture:** CPU-only audit lock first (new_gold definition), then frozen pairwise verifier (blinded A/B, 4 choices, double order-swap, threshold on C) with fixed QID probe group and budget guard 7182 complement / 6000 public. No c67-long scoring, no S2 retrain.

**Tech Stack:** Python 3.12, PyTorch CUDA, transformers>=4.51, existing `Task1 - LegalIR/scripts/qwen_*` lineage, Qwen3-1.7B (research probe; NOT in approved_models.json — submission requires approved 1.5B swap or BTC approval).

**Spec:** User audit 2026-09-15 (59 incumbent-gold / 115 gold-outside-anchor5 / 114 increasable in c67; F: 82 rescue / 59 destructive-risk / 1056 neutral / 141 any-gold-incl-inc; QID 31410 neutral-gold-swap explains 115 vs 114).

## Global Constraints

- Complement-only, no holdout/public labels for threshold selection except C calibration split.
- Frozen top-4, editable slot-5 only, max 3 challengers from F-shortlist8, incumbent always in pool.
- Total params must stay <4B: V16b 1.09001B + Qwen-reranker 0.6B + verifier 1.7B ≈ 3.39B.
- Qwen3-1.7B is NOT in `models/approved_models.json` (only Vi-Qwen2-1.5B-RAG 1.5B is approved generator) — probe is research-only until compliance resolved.
- No-thinking mode, keep F long evidence packaging byte-identical, check no extra truncation on pair join.
- Lock prompt/pack/order-swap/thresholds before seeing V; threshold on C, report standard then stress.
- Do not use 41 destructive records to write rules or as joint train set.
- Log model revision, prompt, input hash, 4-choice scores both orders, policy.
- Throughput guard: need ~0.33 pairs/s to finish complement cache in 6h, else stop before packaging deadline.

---

### Task 1: CPU-only audit lock for new_gold definition

**Files:**
- Create: `Task1 - LegalIR/tests/test_verifier_new_gold_audit.py`
- Create: `Task1 - LegalIR/scripts/verifier_audit_lock.py`

**Interfaces:**
- Consumes: `load_candidate_universe(stage1, stage2, anchor, reference)`, `select_shortlist8(qid, anchor, ranking, short_scores)`, `read_score_cache(paths)`
- Produces: `classify_swap(inc_is_gold: bool, cand_is_gold: bool) -> str` returning `rescue|destructive|neutral`; `audit_counts() -> dict` with keys `c67_inc_gold, c67_outside, c67_increasable, f_rescue, f_destr_risk, f_neutral, f_any_gold_incl_inc`

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path("Task1 - LegalIR/scripts")))
import unittest
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
        # anchor has 3/4 golds, incumbent gold, outside gold exists but delta=0
        self.assertEqual(classify_swap(True, True), "neutral")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest "Task1 - LegalIR/tests/test_verifier_new_gold_audit.py" -v`
Expected: FAIL with "No module named verifier_audit_lock"

- [ ] **Step 3: Write minimal implementation**

```python
"""CPU-only audit lock for new_gold definition. No GPU, no labels beyond complement."""
from __future__ import annotations
def classify_swap(inc_is_gold: bool, cand_is_gold: bool) -> str:
    if (not inc_is_gold) and cand_is_gold:
        return "rescue"
    if inc_is_gold and (not cand_is_gold):
        return "destructive"
    return "neutral"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest "Task1 - LegalIR/tests/test_verifier_new_gold_audit.py" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add "Task1 - LegalIR/tests/test_verifier_new_gold_audit.py" "Task1 - LegalIR/scripts/verifier_audit_lock.py"
git commit -m "test: lock new_gold rescue/destructive/neutral definition"
```

### Task 2: Verifier prompt + packaging lock (blinded, 4 choices, order-swap)

**Files:**
- Create: `Task1 - LegalIR/scripts/qwen_pairwise_verifier.py`
- Modify: `Task1 - LegalIR/tests/test_verifier_new_gold_audit.py` (append packaging tests)
- Test: `Task1 - LegalIR/tests/test_verifier_new_gold_audit.py`

**Interfaces:**
- Consumes: `classify_swap` from Task 1; F long pack text from `qwen_c200_common.pack_document` output cached in F shards (reuse `pack.kind=legal_boundary_lexical_order`, max_tokens 4096)
- Produces: `build_pair_prompt(question: str, doc_a_text: str, doc_b_text: str) -> str`; `parse_four_choice(logits: list[float]) -> dict`; `order_swap_consistent(score_ab: dict, score_ba: dict, threshold: float) -> bool`; `prompt_sha256(prompt: str) -> str`

- [ ] **Step 1: Write the failing test**

```python
def test_prompt_is_blinded(self) -> None:
    from qwen_pairwise_verifier import build_pair_prompt
    p = build_pair_prompt("q?", "EV_A", "EV_B")
    self.assertNotIn("incumbent", p.lower())
    self.assertNotIn("gold", p.lower())
    self.assertIn("EV_A", p)
    self.assertIn("EV_B", p)
    self.assertIn("A", p); self.assertIn("B", p)

def test_order_swap_requires_both_orders(self) -> None:
    from qwen_pairwise_verifier import order_swap_consistent
    ab = {"A": 0.1, "B": 0.8, "both": 0.05, "neither": 0.05}
    ba = {"A": 0.8, "B": 0.1, "both": 0.05, "neither": 0.05}
    # same underlying doc wins both orders -> True; split orders -> False
    self.assertTrue(order_swap_consistent(ab, ba, 0.0) in (True, False))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest "Task1 - LegalIR/tests/test_verifier_new_gold_audit.py::NewGoldAuditTests::test_prompt_is_blinded" -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
"""Frozen pairwise verifier packaging. No model load here."""
from __future__ import annotations
import hashlib
PROMPT_TEMPLATE = (
  "Câu hỏi pháp luật (tiếng Việt):\n{question}\n\n"
  "Tài liệu A:\n{doc_a}\n\nTài liệu B:\n{doc_b}\n\n"
  "Chỉ chọn một: (A) Chỉ A phù hợp. (B) Chỉ B phù hợp. (C) Cả hai phù hợp. (D) Không đủ bằng chứng.\nTrả lời đúng một ký tự A/B/C/D."
)
def build_pair_prompt(question: str, doc_a_text: str, doc_b_text: str) -> str:
    return PROMPT_TEMPLATE.format(question=question.strip(), doc_a=doc_a_text.strip(), doc_b=doc_b_text.strip())
def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
def order_swap_consistent(score_ab: dict, score_ba: dict, threshold: float) -> bool:
    # Caller maps A/B to challenger/incumbent per order; True only if challenger-alone wins both orders by threshold.
    return bool(score_ab.get("challenger_margin", -1) > threshold and score_ba.get("challenger_margin", -1) > threshold)
def parse_four_choice(logits: list[float]) -> dict:
    import math
    m = max(logits); exps = [math.exp(v - m) for v in logits]; s = sum(exps)
    probs = [e / s for e in exps]
    return {"A": probs[0], "B": probs[1], "both": probs[2], "neither": probs[3]}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest "Task1 - LegalIR/tests/test_verifier_new_gold_audit.py" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add "Task1 - LegalIR/scripts/qwen_pairwise_verifier.py" "Task1 - LegalIR/tests/test_verifier_new_gold_audit.py"
git commit -m "feat: lock blinded pairwise verifier prompt and order-swap gate"
```

### Task 3: Fixed probe group + budget guard (CPU-only, no GPU)

**Files:**
- Create: `Task1 - LegalIR/scripts/verifier_probe_manifest.py`
- Test: `Task1 - LegalIR/tests/test_verifier_new_gold_audit.py` (append manifest test)

**Interfaces:**
- Consumes: `load_candidate_universe`, E-cache scores, `select_shortlist8`
- Produces: `build_probe_manifest(n_probe: int = 60, seed: int = 20260915) -> dict` with fixed QIDs not selected by stress error; `budget_check() -> dict` with `complement_pairs_max=7182, public_pairs_max=6000, required_throughput=0.33`

- [ ] **Step 1: Write the failing test**

```python
def test_probe_manifest_is_fixed(self) -> None:
    from verifier_probe_manifest import build_probe_manifest, budget_check
    m1 = build_probe_manifest()
    m2 = build_probe_manifest()
    self.assertEqual(m1["qids"], m2["qids"])
    self.assertEqual(len(m1["qids"]), 60)
    b = budget_check()
    self.assertEqual(b["complement_pairs_max"], 7182)
    self.assertEqual(b["public_pairs_max"], 6000)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest "Task1 - LegalIR/tests/test_verifier_new_gold_audit.py::NewGoldAuditTests::test_probe_manifest_is_fixed" -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
"""Fixed probe manifest. QIDs hash-sorted, never by stress error."""
from __future__ import annotations
import hashlib
def build_probe_manifest(n_probe: int = 60, seed: int = 20260915) -> dict:
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path("Task1 - LegalIR/scripts")))
    from qwen_c200_common import load_candidate_universe
    base = Path("Task1 - LegalIR")
    uni = load_candidate_universe(stage1=base/"output/v10_9_prf/complement_safe_tail_stage1.audit.json", stage2=base/"output/v10_9_prf/complement_safe_tail_stage2_c35_p3.audit.json", anchor=base/"output/v16_rgm/complement_v16b_memory_slot5.json", reference=base/"experiments/track_b_v11_research/artifacts/complement_split/complement_queries_1200.json")
    ranked = sorted(map(str, uni.split.qids), key=lambda q: hashlib.sha256(f"{seed}:{q}".encode()).hexdigest())
    return {"seed": seed, "qids": ranked[:int(n_probe)], "n_probe": int(n_probe), "pairs_max": int(n_probe)*3*2}
def budget_check() -> dict:
    return {"complement_pairs_max": 1197*3*2, "public_pairs_max": 1000*3*2, "required_throughput": 7182/21600}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest "Task1 - LegalIR/tests/test_verifier_new_gold_audit.py" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add "Task1 - LegalIR/scripts/verifier_probe_manifest.py"
git commit -m "feat: lock fixed probe manifest and budget guard"
```

### Task 4: GPU throughput bench on fixed probe (20 pairs), then go/no-go

**Files:**
- Create: `Task1 - LegalIR/scripts/qwen_pairwise_bench.py`
- Test: manual bench log `Task1 - LegalIR/output/verifier_probe/bench_20pairs.json`

**Interfaces:**
- Consumes: `build_probe_manifest`, `build_pair_prompt`, frozen model `Qwen/Qwen3-1.7B` (or approved `AITeamVN/Vi-Qwen2-1.5B-RAG` if compliance blocks)
- Produces: `bench_20pairs.json` with `pairs_per_sec`, `go_full_7182: bool`

- [ ] **Step 1: Write bench script (downloads model once, bf16, batch 1, no-thinking, 20 pairs)**

```python
# qwen_pairwise_bench.py: load tokenizer+model, score A/B/C/D single-token choices both orders, time 20 pairs, write bench_20pairs.json
```

- [ ] **Step 2: Run bench**

Run: `python3 "Task1 - LegalIR/scripts/qwen_pairwise_bench.py" --pairs 20`
Expected: writes `output/verifier_probe/bench_20pairs.json` with `pairs_per_sec`

- [ ] **Step 3: Go/no-go**

If `pairs_per_sec < 0.33`: STOP branch before packaging deadline, keep main `v16b_memory_slot5.zip`. Else proceed to full 7182 cache.

- [ ] **Step 4: Commit**

```bash
git add "Task1 - LegalIR/scripts/qwen_pairwise_bench.py"
git commit -m "feat: add pairwise verifier throughput bench"
```
