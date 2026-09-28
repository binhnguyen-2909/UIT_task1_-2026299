# E4 — one-time locked-holdout parity contract

E4 is authorized only because the matching E3-B standard and stress reports
passed their preregistered gates.  This document locks deployment before the
holdout reference is opened.  It is not an instruction to tune further using
holdout results.

## Locked inputs and decision rule

- The E3-B standard and stress reports must have one identical lineage hash;
  standard must allow stress and stress must allow E4.
- V16b is the holdout anchor:
  `output/v16_rgm/holdout_v16b_memory_slot5.json`.
- Candidate universe is the frozen holdout c35.  Predictions must retain V16b
  positions 1–4 exactly, contain five distinct IDs, and choose slot 5 only
  from c35.
- The policy is fixed before the holdout reference is read:
  `E3BPolicy(memory_weight=0.0, threshold=0.50)`.  It uses the two E2
  full-c35 MIL scores, their median/IQR per-query normalization, the existing
  disagreement penalty, and default abstention.  No memory feature is used at
  inference because its coefficient is zero.

## Deployment models

Two final students are needed for both E4 and the future E5 materialization.
They are **not** a chosen OOF fold checkpoint.

- Start each from the original Vietnamese reranker checkpoint.
- Use E2 seeds `20260908` and `20260909`.
- Fit all sealed 1,197 complement queries for exactly two epochs, with the
  E2 loss/temperature/max length, LR `1e-5`, weight decay `0.01`, maximum
  negatives `8`, and gradient accumulation `16` fixed from E3-B.
- There is no early-stop set and no holdout input in either fit.  Batch size is
  selected only by the audited GPU auto-probe; it cannot change the optimizer,
  precision, evidence count, token limit or learning configuration.
- Save both states and hashes.  Their combined E0+two-E2 inference pipeline is
  `2.61001B`, below the `4B` limit.

## Two-phase execution boundary

### `MODE=prepare` — GPU, label-blind

1. Validate report/model/data lineage, then write the E4 label-blind manifest.
2. Build holdout evidence from stage1/stage2/anchor and raw contexts **without
   loading `holdout_1000.json`**.
3. Fit the two deployment students from complement labels and score every
   holdout c35 row.  Each score artifact includes document MIL and passage
   logits; empty-body rows remain deliberate nulls.
4. Apply the fixed policy and write the candidate-only holdout prediction
   artifact.  This phase is resumable by final-model boundary and records no
   holdout qrels or metrics.

### `MODE=parity` — CUDA, exactly once

After `prepare` succeeds, this is the only command that may read
`holdout_1000.json`.  It first writes `e4_holdout_access_ledger.json` with an
access count of one, then evaluates the already prepared predictions against
the V16b anchor with CUDA tensor membership/Recall operations, writes the
parity report atomically, and completes the
ledger.  If a ledger or report already exists, the script refuses a second
evaluation.  Preserve a crash state for review rather than bypassing it.

All OpenMP/BLAS/Torch CPU worker pools are capped to one thread. JSON parsing,
filesystem access and fast-tokenizer preparation remain host-side because
these are not CUDA operations; they are intentionally bounded rather than
silently using every CPU core.

## Promotion criterion

E4 permits a later E5 public materialization only when all provenance and
format checks pass and both are true on the one locked-holdout evaluation:

1. macro Recall@5 gain versus V16b is strictly positive; and
2. successful rescues exceed destructive changes.

E4 never creates public predictions and cannot submit anything.  A failed E4
means retain the already eligible V16b release and continue any research only
on complement data; do not adjust alpha, threshold, seed, evidence, or policy
from the holdout result.
