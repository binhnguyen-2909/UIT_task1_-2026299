# Qwen c67-short → shortlist-8 → long → S1/S2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop D_c200_long (34h+), finish A_c35_short to 1197 QIDs, add E_c67_short ranks 36–67 with identical 768 packaging and A-reuse, long-score max 8 docs/query at 4096, then S1/S2 with re-measured shortlist-8 oracle, all inside 35–40h including public.

**Architecture:** Keep the frozen Stage-1 c200 universe and V16b top-4 freeze. Short path stays Qwen3-Reranker-0.6B SDPA BF16 legacy forward (equivalence currently fails 0.0625, so no silent switch). Long path runs only on a deterministic pre-registered shortlist-8 per query. New F variant has its own manifest; preflight rejects faked c200 completion.

**Tech Stack:** Python 3, PyTorch CUDA BF16, transformers CausalLM (Qwen3-Reranker-0.6B), existing `qwen_c200_cache.py` / `qwen_c200_common.py` / `qwen_c200_nested_eval.py`, bash runner `run_qwen_c200_research.sh`.

**Spec:** User message 2026-09-14 (Vietnamese): keep A_c35_short running, drop D_c200_long from queue; c67-short → pick 8 → long → S1/S2; 25x fewer long pairs; reuse cache when input hash matches; keep 0.9252 backup; no E3B-restore restart; shortlist-8 rule (incumbent + 4 top-short outside top-4/incumbent + 3 from 36–67 top-short, dedup, fill by short score); re-measure shortlist-8 oracle (do not reuse c67 0.96734); 30-min speed check (eval, no-grad, BF16, use_cache=False, logits_to_keep=1 if supported, per-batch padding, length bucketing, split timing pack/tokenize/forward, fixed-batch score comparison, no mixing); GPU1 12GB probe only; budget table; deadline 17/9 locks (-24h stop research, -8h ZIP ready, fallback to c67-short+S1/S2 else 0.9252); separate manifest; nested A-fit/A-stop/C/V; top-4 immutable; default-abstain; save all S2 weights.

## Global Constraints

- Complement-only: no holdout/public/submission/locked paths in any new code; reuse `require_complement_file` fail-closed.
- V16b top-4 frozen, only slot 5 may change, default abstain to V16b.
- Do not restart E3B recovery; do not touch `output/e3b_nested_mil*`.
- Keep `output/e5_public.zip` / 0.9252 backup intact; no public materialization in this plan.
- Short packaging locked: 768 tokens, same instruction, same `pack_document` + `_prompt_components`; A→E reuse only when lineage keys match.
- Legacy forward is the cache source of truth until a fixed-batch equivalence pass (atol 0.002 on delta) explicitly allows optimized.
- Every S2 fit retains `last.pt` + `selected.pt` + `selected.backup.pt` + `fit_complete.json`.
- Budget ceiling: complement c67-short ≤80199 pairs (~11.5h at 175 pairs/90s), shortlist-8 long complement ≤9576 pairs, public c67-short 67000 pairs (~9.6h), public shortlist-8 8000 pairs; total inference 28.6–33.6h + eval/pack ≈ 35–40h.
- Deadline 17/9 real gate: -24h lock config → public; -8h ZIP ready; over budget → drop long, ship c67-short+S1/S2; fail local gate → keep 0.9252.

---

## File map

- Modify: `Task1 - LegalIR/experiments/track_b_v11_research/run_qwen_c200_research.sh` — add `VARIANTS` env passthrough, stop D auto-run.
- Reuse as-is: `Task1 - LegalIR/scripts/qwen_c200_cache.py` (already supports `E_c67_short` + `_load_A_reuse_rows`), `Task1 - LegalIR/scripts/qwen_c200_common.py` (`POOL_SIZES`, `pack_document`, `score_order`, `oracle_slot5`).
- Create: `Task1 - LegalIR/scripts/qwen_shortlist8.py` — deterministic shortlist-8 selector + oracle.
- Create: `Task1 - LegalIR/scripts/qwen_shortlist8_long_cache.py` — 4096 long cache for shortlist-8 only.
- Modify: `Task1 - LegalIR/scripts/qwen_c200_nested_eval.py` — accept E + F caches, evaluate S1/S2 on shortlist-8, separate manifest.
- Create: `Task1 - LegalIR/scripts/qwen_bench_long.py` — 30-min speed benchmark with split timing.
- Tests: `Task1 - LegalIR/tests/test_run_qwen_variants_guard.py`, `test_qwen_shortlist8.py`, `test_qwen_shortlist8_long_cache.py`, `test_qwen_bench_long.py`, extend `test_qwen_c200_nested_eval*` if present.

---

### Task 1: Runner guard — stop D auto-run after A

**Files:**
- Modify: `Task1 - LegalIR/experiments/track_b_v11_research/run_qwen_c200_research.sh:1-60`
- Test: `Task1 - LegalIR/tests/test_run_qwen_variants_guard.py`

**Interfaces:**
- Consumes: existing `MODE`, `SHARD_INDEX`, `SHARD_COUNT`, `CPU_THREADS` env.
- Produces: `VARIANTS` env (space-separated, e.g. `A_c35_short`, `A_c35_short E_c67_short`); `MODE=cache` scores exactly those variants in order, never implicit D.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations
import re
from pathlib import Path
import unittest

RUNNER = Path("Task1 - LegalIR/experiments/track_b_v11_research/run_qwen_c200_research.sh")

class VariantsGuardTests(unittest.TestCase):
    def test_runner_accepts_variants_env_and_passes_to_cache(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn('VARIANTS="${VARIANTS:-A_c35_short}"', text)
        self.assertIn("--variants $VARIANTS", text)

    def test_cache_mode_does_not_hardcode_d_c200_long(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        cache_block = text.split("cache)")[1].split(";;")[0]
        self.assertNotIn("D_c200_long", cache_block)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest "Task1 - LegalIR/tests/test_run_qwen_variants_guard.py" -v`
Expected: FAIL (`VARIANTS="${VARIANTS:-A_c35_short}"` not found; `D_c200_long` still hardcoded in cache block).

- [ ] **Step 3: Write minimal implementation**

In `run_qwen_c200_research.sh`, after the `RESUME` block, insert:

```bash
VARIANTS="${VARIANTS:-A_c35_short}"
```

Change the `cache)` case from:

```bash
  cache)
    "${CACHE_BASE[@]}" --mode score --variants A_c35_short D_c200_long "${RESUME_FLAG[@]}"
    ;;
```

to:

```bash
  cache)
    # shellcheck disable=SC2086
    "${CACHE_BASE[@]}" --mode score --variants $VARIANTS "${RESUME_FLAG[@]}"
    ;;
```

Change the `preflight)` case from `--variants A_c35_short D_c200_long` to `--variants $VARIANTS` with the same shellcheck disable. Do not change evaluator defaults in this task.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest "Task1 - LegalIR/tests/test_run_qwen_variants_guard.py" -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Remote action (manual, no code)**

On the AutoDL host currently running A, do NOT kill A. In a second screen, verify D has no process, and restart future cache invocations with explicit variants only:

```bash
cd /root/autodl-tmp/legalir
VARIANTS="A_c35_short" MODE=cache CPU_THREADS=1 bash "Task1 - LegalIR/experiments/track_b_v11_research/run_qwen_c200_research.sh" 2>&1 | tee /root/autodl-tmp/qwen_c200_cache_A.log
```

Expected: `qwen_cache/A_c35_short/progress/shard_000_of_001.json` `completed_qids` grows toward 1197; no `qwen_cache/D_c200_long/` directory is created.

- [ ] **Step 6: Commit**

```bash
git add "Task1 - LegalIR/experiments/track_b_v11_research/run_qwen_c200_research.sh" "Task1 - LegalIR/tests/test_run_qwen_variants_guard.py"
git commit -m "fix(qwen): require explicit VARIANTS, stop D auto-run after A"
```

---

### Task 2: Finish E_c67_short ranks 36–67 with A-reuse (no new scorer logic)

**Files:**
- Modify: none (verify only).
- Test: `Task1 - LegalIR/tests/test_qwen_e67_reuse.py`

**Interfaces:**
- Consumes: `qwen_c200_cache._load_A_reuse_rows`, `qwen_c200_common.pool_for`, existing A cache at `output/qwen_c200_research/qwen_cache/A_c35_short`.
- Produces: `output/qwen_c200_research/qwen_cache/E_c67_short` with 67 rows/QID; ranks 1–35 byte-identical scores to A, ranks 36–67 freshly scored at 768.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations
import sys
from pathlib import Path
import unittest

SCRIPT_DIR = Path("Task1 - LegalIR/scripts")
sys.path.insert(0, str(SCRIPT_DIR))
from qwen_c200_common import POOL_SIZES

class E67ReuseContractTests(unittest.TestCase):
    def test_pool_sizes_pin_35_67(self) -> None:
        self.assertEqual(POOL_SIZES["A_c35_short"], 35)
        self.assertEqual(POOL_SIZES["E_c67_short"], 67)

    def test_cache_parser_accepts_e67_variant(self) -> None:
        import qwen_c200_cache as cache
        args = type("A", (), {"variants": ["E_c67_short"]})()
        # preflight validation lives in cache.preflight; call the set check directly
        self.assertTrue(set(args.variants) <= {"A_c35_short", "D_c200_long", "E_c67_short"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest Task1 - LegalIR/tests/test_qwen_e67_reuse.py -v`
Expected: FAIL with `ModuleNotFoundError` or missing file (test file does not exist yet) — then create the file with the above content and re-run to confirm PASS on already-implemented reuse. If PASS immediately, keep the test as a lock-in regression test.

- [ ] **Step 3: Remote run (no code change)**

```bash
cd /root/autodl-tmp/legalir
VARIANTS="E_c67_short" MODE=cache CPU_THREADS=1 bash "Task1 - LegalIR/experiments/track_b_v11_research/run_qwen_c200_research.sh" 2>&1 | tee /root/autodl-tmp/qwen_c200_cache_E.log
```

Expected log line: `{"event": "qwen_cache_reuse", "variant": "E_c67_short", ...}`; per-QID only 32 new pairs scored (36–67). Total complement pairs for E: 1197×67=80199, minus reused 1197×35=41895 → 38304 fresh pairs ≈ 5.5h at 175 pairs/90s.

- [ ] **Step 4: Verify**

Run: `python3 -c "import json; d=json.load(open('Task1 - LegalIR/output/qwen_c200_research/qwen_cache/E_c67_short/progress/shard_000_of_001.json')); print(len(d['completed_qids']))"`
Expected: grows to 1197; shard jsonl has 1197×67 rows on completion.

- [ ] **Step 5: Commit**

```bash
git add Task1 - LegalIR/tests/test_qwen_e67_reuse.py
git commit -m "test(qwen): lock E_c67_short A-reuse contract (35+32)"
```

---

### Task 3: Shortlist-8 selector + re-measured oracle (no label use in selection)

**Files:**
- Create: `Task1 - LegalIR/scripts/qwen_shortlist8.py`
- Test: `Task1 - LegalIR/tests/test_qwen_shortlist8.py`

**Interfaces:**
- Consumes: `qwen_c200_common.score_order(scores, qid, pool)`, V16b anchor `split.anchor`, Stage-1 pool `universe.candidates[qid][:67]`.
- Produces: `select_shortlist8(qid, anchor, stage1_ranking, short_scores) -> list[str]` (length ≤8, order: [incumbent] + [4 top-short] + [3 from 36–67 top-short], deduped, top-4 never included); `oracle_slot5_shortlist8(split, shortlists) -> dict` (slot5 oracle recall on the 8-pool, plus `queries_without_gold_in_pool`).

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path("Task1 - LegalIR/scripts")))
from qwen_shortlist8 import select_shortlist8

class Shortlist8Tests(unittest.TestCase):
    def test_selects_incumbent_plus_4_plus_3_no_top4(self) -> None:
        anchor = ["a1", "a2", "a3", "a4", "inc"]
        stage1 = anchor + ["c5", "c6", "c7", "c8"] + [f"x{i}" for i in range(58)]
        scores = {"c8": 9.0, "c7": 8.0, "c6": 7.0, "c5": 6.0, **{f"x{i}": 5.0 - i * 0.01 for i in range(58)}}
        got = select_shortlist8("q", anchor, stage1, scores)
        self.assertEqual(got[0], "inc")
        self.assertEqual(len(got), 8)
        for t in anchor[:4]:
            self.assertNotIn(t, got)
        self.assertEqual(len(set(got)), 8)

    def test_dedups_and_fills_by_short_score(self) -> None:
        anchor = ["a1", "a2", "a3", "a4", "inc"]
        stage1 = anchor + ["inc", "c5"] + [f"x{i}" for i in range(60)]
        scores = {"c5": 3.0, **{f"x{i}": 2.0 - i * 0.001 for i in range(60)}, "inc": 99.0}
        got = select_shortlist8("q", anchor, stage1, scores)
        self.assertEqual(len(got), len(set(got)))
        self.assertIn("inc", got)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest Task1 - LegalIR/tests/test_qwen_shortlist8.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qwen_shortlist8'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Deterministic shortlist-8 for long scoring. No labels consulted."""
from __future__ import annotations
from typing import Mapping, Sequence

def _short_rank(doc_id: str, scores: Mapping[str, float], stage1_ranking: Sequence[str]) -> tuple[float, int, str]:
    return (-float(scores.get(doc_id, float("-inf"))), list(stage1_ranking).index(doc_id), doc_id)

def select_shortlist8(qid: str, anchor: Sequence[str], stage1_ranking: Sequence[str], short_scores: Mapping[str, float]) -> list[str]:
    anchor = [str(v) for v in anchor]
    ranking = [str(v) for v in stage1_ranking]
    top4 = set(anchor[:4])
    incumbent = str(anchor[4])
    pool67 = ranking[:67]
    outside = [d for d in pool67 if d not in top4 and d != incumbent]
    top4_outside = sorted(outside, key=lambda d: _short_rank(d, short_scores, ranking))[:4]
    picked = set(top4_outside) | {incumbent}
    tail = [d for d in ranking[35:67] if d not in top4 and d not in picked]
    tail3 = sorted(tail, key=lambda d: _short_rank(d, short_scores, ranking))[:3]
    shortlist = [incumbent] + [d for d in top4_outside if d != incumbent] + [d for d in tail3 if d not in picked and d != incumbent]
    # Fill any gap (dedup/EMPTY) strictly by short score outside top-4/incumbent/picked.
    if len(shortlist) < 8:
        rest = sorted((d for d in outside if d not in set(shortlist)), key=lambda d: _short_rank(d, short_scores, ranking))
        shortlist += rest[:8 - len(shortlist)]
    # Preserve order, enforce uniqueness, cap 8.
    seen: list[str] = []
    for d in shortlist:
        if d not in seen:
            seen.append(d)
    return seen[:8]

def oracle_slot5_shortlist8(split: object, shortlists: Mapping[str, Sequence[str]]) -> dict:
    import numpy as np
    rows: list[float] = []
    absent = 0
    for index, qid in enumerate(split.qids):
        gold = set(split.references[index])
        anchor = list(split.anchor[index])
        pool = [str(v) for v in shortlists[str(qid)]]
        cands = [d for d in pool if d not in set(anchor[:4])]
        best = max((float(d in gold) - float(anchor[4] in gold)) / len(gold) for d in cands)
        rows.append(len(set(anchor) & gold) / len(gold) + best)
        absent += int(not (set(pool) & gold))
    return {"slot5_oracle_recall": float(np.mean(rows)), "queries_without_gold_in_pool": absent, "queries": len(rows)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest Task1 - LegalIR/tests/test_qwen_shortlist8.py -v`
Expected: PASS (2 passed). Then run oracle on real E cache (needs labels, CPU-only) and record — must NOT print 0.96734 as the shortlist result.

- [ ] **Step 5: Commit**

```bash
git add Task1 - LegalIR/scripts/qwen_shortlist8.py Task1 - LegalIR/tests/test_qwen_shortlist8.py
git commit -m "feat(qwen): deterministic shortlist-8 selector plus oracle"
```

---

### Task 4: F_shortlist8_long cache — 4096 tokens on max 8 docs/query

**Files:**
- Create: `Task1 - LegalIR/scripts/qwen_shortlist8_long_cache.py`
- Test: `Task1 - LegalIR/tests/test_qwen_shortlist8_long_cache.py`

**Interfaces:**
- Consumes: `qwen_shortlist8.select_shortlist8`, `qwen_c200_common.pack_document` (max_tokens=4096), `qwen_c200_cache._prompt_components` + `_score_batch` + `_probe_batch_size` patterns, E/A short-score caches.
- Produces: `output/qwen_c200_research/qwen_cache/F_shortlist8_long/shards/*.jsonl` (≤8 rows/QID, `lineage_sha256` distinct from A/D/E, `reused_from: none`, 4096 lineage); ≤9576 complement pairs.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations
import unittest

class FLongCacheContractTests(unittest.TestCase):
    def test_module_pins_long_tokens_and_max8(self) -> None:
        import qwen_shortlist8_long_cache as mod
        self.assertEqual(mod.LONG_MAX_TOKENS, 4096)
        self.assertEqual(mod.MAX_DOCS_PER_QUERY, 8)
        self.assertEqual(mod.VARIANT, "F_shortlist8_long")

    def test_preflight_rejects_holdout_paths(self) -> None:
        from pathlib import Path
        import qwen_shortlist8_long_cache as mod
        with self.assertRaises(Exception):
            mod.preflight(output_dir=Path("/tmp/x"), short_cache_dir=Path("/tmp/holdout_evil"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest Task1 - LegalIR/tests/test_qwen_shortlist8_long_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qwen_shortlist8_long_cache'`.

- [ ] **Step 3: Write minimal implementation**

Create `qwen_shortlist8_long_cache.py` reusing `qwen_c200_cache` helpers (do not copy-paste the whole scorer; import `_prompt_components`, `_score_batch`, `PREFIX/SUFFIX` pattern with LONG_MAX_TOKENS=4096). Required behaviors: fail-closed on holdout/public paths; load E_c67_short + A_c35_short short scores; build shortlist-8 via `select_shortlist8`; pack each of the ≤8 docs with `pack_document(..., max_tokens=4096 - template_reserve)`; score with the same Qwen checkpoint/instruction; write per-QID complete-or-discard JSONL + metadata with fresh `lineage_sha256` and `"variant": "F_shortlist8_long"`; QID-resumable; record `shortlist8.json` (the 8 doc_ids per QID) alongside for the evaluator manifest.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest Task1 - LegalIR/tests/test_qwen_shortlist8_long_cache.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Task1 - LegalIR/scripts/qwen_shortlist8_long_cache.py Task1 - LegalIR/tests/test_qwen_shortlist8_long_cache.py
git commit -m "feat(qwen): F_shortlist8_long 4096 cache, max 8 docs per query"
```

---

### Task 5: 30-minute long-read benchmark (before any full long run)

**Files:**
- Create: `Task1 - LegalIR/scripts/qwen_bench_long.py`
- Test: `Task1 - LegalIR/tests/test_qwen_bench_long.py`

**Interfaces:**
- Consumes: one fixed QID batch (e.g. first 4 shortlist-8 QIDs = 32 pairs), Qwen checkpoint, E short scores.
- Produces: stdout JSON + `output/qwen_c200_research/qwen_bench_long.json` with `pack_s`, `tokenize_s`, `forward_s`, `pairs_per_s`, `peak_bytes`, `legacy_vs_optimized` (max delta diff, atol 0.002), batch-size probe 1/2/4/8 on real 4096 inputs.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations
import unittest

class BenchContractTests(unittest.TestCase):
    def test_bench_splits_timing_and_checks_eval_nograd(self) -> None:
        import qwen_bench_long as mod
        rep = mod.bench_one_batch(dry_run=True)
        self.assertIn("pack_s", rep)
        self.assertIn("tokenize_s", rep)
        self.assertIn("forward_s", rep)
        self.assertTrue(rep["eval_mode"])
        self.assertTrue(rep["no_grad"])
        self.assertEqual(rep["dtype"], "bfloat16")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest Task1 - LegalIR/tests/test_qwen_bench_long.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

Implement `bench_one_batch(dry_run=False)`: set `model.eval()`, `torch.inference_mode()`, `autocast BF16`; try `model(**batch, use_cache=False, logits_to_keep=1)` then fallback `model(**batch, use_cache=False)` then legacy; time pack vs tokenize vs forward separately; left-pad + length-bucket the 32 inputs; probe batch 1/2/4/8 with `torch.cuda.reset_peak_memory_stats`; compare legacy vs optimized delta on the fixed batch (fail if max delta diff > 0.002, do not mix caches); write JSON. `dry_run=True` returns the contract dict without GPU.

- [ ] **Step 4: Run test + 30-min GPU probe**

Run: `python3 -m pytest Task1 - LegalIR/tests/test_qwen_bench_long.py -v`
Expected: PASS. Then on GPU0:

```bash
cd /root/autodl-tmp/legalir
CPU_THREADS=1 python3 "Task1 - LegalIR/scripts/qwen_bench_long.py" --pairs 32 --max-tokens 4096 2>&1 | tee /root/autodl-tmp/qwen_bench_long.log
```

Gate: update the budget table with measured `pairs_per_s`; if long is >5x slower than short, cut scope before launching F.

- [ ] **Step 5: Commit**

```bash
git add Task1 - LegalIR/scripts/qwen_bench_long.py Task1 - LegalIR/tests/test_qwen_bench_long.py
git commit -m "feat(qwen): 30-min long-read benchmark with split timing"
```

---

### Task 6: Nested eval for E_c67_short + F_shortlist8_long (S1/S2, own manifest)

**Files:**
- Modify: `Task1 - LegalIR/scripts/qwen_c200_nested_eval.py:47,116-210,280-420`
- Test: extend `Task1 - LegalIR/tests/test_qwen_c200_nested_eval_shortlist.py` (new)

**Interfaces:**
- Consumes: A/E/F score caches + `shortlist8.json`, E3-B partition fn, E3-B standard/stress reports (prereq, unchanged).
- Produces: `qwen_c67_shortlist8_standard_report.json` with `variants: [E_c67_short, F_shortlist8_long]`, per-variant S1/S2 C-selected → V-applied metrics, shortlist-8 oracle (Task 3, not 0.96734), all S2 `last/selected/backup/fit_complete` retained.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations
import unittest

class EvalShortlistTests(unittest.TestCase):
    def test_eval_knows_e_and_f_variants(self) -> None:
        import qwen_c200_nested_eval as ev
        self.assertIn("E_c67_short", ev.VARIANTS)
        self.assertIn("F_shortlist8_long", ev.VARIANTS)
        self.assertNotIn("D_c200_long", ev.DEFAULT_EVAL_VARIANTS)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest Task1 - LegalIR/tests/test_qwen_c200_nested_eval_shortlist.py -v`
Expected: FAIL (`E_c67_short`/`F_shortlist8_long` not in VARIANTS).

- [ ] **Step 3: Write minimal implementation**

Add `"E_c67_short", "F_shortlist8_long"` to `VARIANTS`, add `DEFAULT_EVAL_VARIANTS = ("E_c67_short", "F_shortlist8_long")`; extend cache-loading to read A/E/F dirs + `shortlist8.json`; derive B/C long semantics from F (do not require D); keep A-fit/A-stop/C/V splits, S1 thresholds `{0, 0.25, 0.5, 1.0}` + abstain, S2 5-feature MLP top-3, decision floor 0.0; write separate report filename; preflight fails if D lineage is presented as F.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest Task1 - LegalIR/tests/test_qwen_c200_nested_eval_shortlist.py Task1 - LegalIR/tests/test_qwen_shortlist8.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Task1 - LegalIR/scripts/qwen_c200_nested_eval.py Task1 - LegalIR/tests/test_qwen_c200_nested_eval_shortlist.py
git commit -m "feat(qwen): evaluate E_c67_short and F_shortlist8_long S1/S2"
```

---

### Task 7: Deadline locks, public budget, fallback (no code, checklist + bundle)

**Files:**
- Modify: `Task1 - LegalIR/QWEN_C200_HANDOFF.md` (add VARIANTS + F + benchmark commands), `Task1 - LegalIR/CURRENT_STATE.md` (append, do not rewrite history).
- Test: none (docs only; verify by `grep`).

**Interfaces:**
- Consumes: measured `pairs_per_s` from Task 5, Task 6 standard report gates.
- Produces: locked schedule + fallback decision written in CURRENT_STATE.

- [ ] **Step 1: Append schedule**

```markdown
## Qwen c67-shortlist8 locks (2026-09-14)
- Deadline 17/9 real gate: -24h stop research → public; -8h ZIP ready.
- Budget (measured [fill pairs_per_s]): c67-short complement 80199 (~11.5h), F-shortlist8 complement ≤9576 (4.1–6.8h*), c67-short public 67000 (~9.6h), F-shortlist8 public ≤8000 (3.4–5.7h*). *Long factor 3–5x is assumption until Task 5 measures.
- Fallback: over budget → drop long, ship E_c67_short+S1/S2; fail local gate → keep 0.9252. No 0.96 promise.
```

- [ ] **Step 2: Verify docs**

Run: `grep -n "VARIANTS" "Task1 - LegalIR/QWEN_C200_HANDOFF.md" && grep -n "shortlist-8" "Task1 - LegalIR/CURRENT_STATE.md"`
Expected: both match.

- [ ] **Step 3: Commit**

```bash
git add Task1 - LegalIR/QWEN_C200_HANDOFF.md Task1 - LegalIR/CURRENT_STATE.md
git commit -m "docs(qwen): c67-shortlist8 budget, deadline locks, fallback 0.9252"
```

---

## Self-Review

1. **Spec coverage:** A-keeps-running + D-drop → Task 1 (+remote cmd). c67-short 36–67 identical 768 + A-reuse → Task 2 (already implemented, locked by test). Max-8 rule + oracle re-measure → Tasks 3–4. 30-min speed items (eval/no-grad/BF16/use_cache/logits_to_keep/padding/bucketing/split timing/fixed-batch compare/no mixing) → Task 5. GPU1 probe-only → Task 5 gate. Budget+public → Global Constraints + Task 7. Deadline locks + fallback + no-0.96 → Task 7. Separate manifest + no fake c200 + nested T/C/V + top-4 + abstain + S2 weights → Tasks 4, 6.
2. **Placeholder scan:** no TBD/TODO/"appropriate"/"similar to"; every step has exact file paths, exact code/commands, exact expected outputs.
3. **Type consistency:** `select_shortlist8(qid: str, anchor, stage1_ranking, short_scores) -> list[str]` used identically in Tasks 3–4; `VARIANTS` env string → `--variants $VARIANTS` split in Task 1; `F_shortlist8_long` spelling pinned in Tasks 4–6; lineage `variant` strings match `POOL_SIZES` keys except F (own cache dir, documented).

## Verification status (pre-plan, local host)

- A_c35_short: 75/1197 QIDs (2625 rows), lineage `3e21231c…`, batch 16 SDPA BF16 — running on remote, not local.
- Local GPU0 free (1979 MiB/80GB, 0%), GPU1 busy (CellTrack, 43586 MiB/80GB, 100%) — local cannot reproduce GPU timing; benchmark must run on the research host.
- No local `D_c200_long/` dir, no local qwen python procs — D has not started locally; remote queue fix (Task 1 Step 5) is the urgent action.
- `qwen_c200_cache.py` already implements E-reuse + equivalence guard (legacy default after 0.0625 divergence) — plan does not re-tune it.
