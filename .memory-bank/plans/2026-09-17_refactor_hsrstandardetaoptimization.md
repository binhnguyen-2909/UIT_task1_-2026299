---
type: refactor
topic: hsrstandardetaoptimization
status: completed
depends_on: []
parallel_safe: false
linked_specs: []
created: 2026-09-17
---
# Plan: refactor — hsrstandardetaoptimization

**Baseline commit:** unknown

## Context

**Problem:** The official HSR standard run reached only iteration 10 of the first
`lambda=1e-4` head after the evaluator/start-up period on a shared A100. The
current L-BFGS closure launches per-query GPU operations and also computes two
unused backward passes per logged iteration. The runner also reuses the contract
preflight filename for evaluator preflight, which blocks the documented sequence.

**Expected result:** Preserve the frozen cache, loss, L-BFGS update sequence,
T/C/V partitions, regularisations, taus, gates and output semantics while
vectorising the grouped loss, removing only logging-only backward work, and
making evaluator preflight write a distinct report. The new official run must
use a new output directory and the immutable existing cache so no old checkpoint
or artifact is overwritten.

**Observed prerequisite correction:** The E3B report's frozen validation
decision ledger differs from replaying three decisions in two standard folds.
HSR therefore treats that immutable ledger as the validation baseline source of
truth; it remains hash-validated and does not relax the metric or promotion
gates.

**Related files:**
- `Task1 - LegalIR/scripts/hsr_nested_head.py`
- `Task1 - LegalIR/tests/test_hsr_hidden_state.py`
- `Task1 - LegalIR/experiments/track_b_v11_research/run_hsr_c67_hidden_head.sh`
- `Task1 - LegalIR/HIDDEN_STATE_RERANKER_PROTOCOL.md`
- `HSR_GPU_SERVER_RUNBOOK.md`

---

## Stages

<!-- mb-stage:1 -->
### Stage 1: Lock the numerical and runbook contracts

**What to do:**
- Record the baseline checkpoint and elapsed time without using it as an official
  result; preserve the current cache/output directory unchanged.
- Add failing CPU tests for vectorised grouped pointwise/pairwise loss against a
  literal reference implementation, including empty-positive and
  empty-negative groups, plus gradient equivalence within float32 tolerance.
- Add a runner contract test that evaluator preflight has a distinct report name
  and can read a cache root different from the output root.

**Testing (TDD — tests BEFORE implementation):**
- Red tests cover scalar loss and gradients for mixed group cardinalities and
  query weights; expected values come from the literal existing formula.
- A shell-source contract test verifies `HIDDEN_CACHE_ROOT` and the
  `hsr_eval_preflight_report.json` default are wired into the runner.

**DoD (Definition of Done):**
- [x] The new tests fail against the pre-vectorisation implementation for the
  intended missing helper/runner contract, with no production code changed.
- [x] The current run's durable checkpoint and source output paths are recorded.
- [x] No holdout/public file is read or written.
- [x] tests pass after implementation in Stage 2
- [x] lint clean

**Code rules:** SOLID, DRY, KISS, YAGNI, Clean Architecture

---

<!-- mb-stage:2 -->
### Stage 2: Vectorise the fit path without changing the optimisation contract

**What to do:**
- Build fixed padded index/mask tensors once per head fit and calculate grouped
  pairwise softplus and pointwise BCE in batched GPU operations.
- Keep L-BFGS, `strong_wolfe`, 200-iteration maximum, initial weights/bias,
  regularisations, checkpoint schema and selector unchanged.
- Replace pre/post history logging calls with no-grad loss evaluation; leave the
  optimiser's closure untouched so updates retain the same mathematical path.
- Reconstruct E3B validation baseline from the report's immutable decision
  ledger rather than a mutable replay of its score artifacts.
- Update the runner to write `hsr_eval_preflight_report.json` and accept
  `HIDDEN_CACHE_ROOT` (defaulting to `OUTPUT_DIR`) for safe cache reuse.

**Testing (TDD):**
- Run the new loss/gradient tests and existing HSR selector/training tests.
- Run syntax checks for the runner and a label-blind evaluator preflight against
  the immutable c67 cache.

**DoD:**
- [x] Vectorised and literal losses and gradients agree within `1e-6` for the
  deterministic CPU fixtures.
- [x] Existing HSR tests pass, and no protocol constant, split, candidate pool,
  regularisation or gate changes.
- [x] Evaluator preflight succeeds without overwriting contract preflight.

---

<!-- mb-stage:3 -->
### Stage 3: Benchmark and restart the official standard run safely

**What to do:**
- Terminate the pre-optimisation standard process only after retaining its
  resumable iteration-10 checkpoint; do not delete any artifact.
- Create a fresh output root for the optimised run, pointing at the immutable
  completed c67 cache through `HIDDEN_CACHE_ROOT`.
- Run label-blind evaluator preflight, then standard on one specified GPU;
  measure time to the first 10-iteration checkpoint and report the speedup
  versus the baseline.

**Testing (TDD):**
- Verify the new run writes a distinct evaluator preflight report and first head
  checkpoint with the new output root and exact cache lineage.
- Do not run stress unless the completed standard report says
  `promotion_to_stress_allowed:true`.

**DoD:**
- [x] The old output directory remains intact and resumable.
- [x] The new standard job has a durable first checkpoint and an evidence-backed
  ETA based on measured iteration time.
- [x] The report records GPU selection, cache lineage, elapsed time and the
  exact gate status; no holdout/public label is opened.

---

## Risks and mitigation

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Numerical drift changes selected policies | M | Compare literal/vectorised CPU loss and gradients before GPU execution; retain the literal optimiser and selector. |
| New run accidentally reuses an old checkpoint | M | Use a new output root and keep cache root explicit; checkpoint lineage remains checked. |
| Shared GPU makes ETA unrepresentative | H | Report measured checkpoint rate, never infer from GPU model name; preserve resume checkpoints. |
| Evaluation preflight overwrites contract evidence | M | Use a dedicated report filename and test the runner contract. |

## Gate (plan success criterion)

The plan succeeds only when the vectorised path is numerically equivalent on
tests, the documented preflight sequence works without collision, and a fresh
standard job has a durable checkpoint with a measured faster ETA. A standard
pass is not assumed; stress remains gated by its report.
