# E3-B server handoff — full-c35 nested MIL

This patch starts a **new E3-B GPU experiment**.  It does not rerun E0, E1 or
E2.  It consumes their already sealed *complement* artifacts, keeps V16b as
the fallback anchor, and never reads holdout or public labels.

## What must already exist beside this patch

Unzip the patch at the root of the validated E0–E2 capsule.  Before starting,
these paths must exist:

```text
Task1 - LegalIR/output/v10_9_prf/complement_safe_tail_stage1.audit.json
Task1 - LegalIR/output/v10_9_prf/complement_safe_tail_stage2_c35_p3.audit.json
Task1 - LegalIR/output/v16_rgm/complement_v16b_memory_slot5.json
Task1 - LegalIR/output/v16_rgm/complement_memory_features.json
Task1 - LegalIR/output/e1_evidence_frozen_ce/e1_complement_evidence.jsonl
Task1 - LegalIR/output/e1_evidence_frozen_ce/e1_complement_evidence.meta.json
Task1 - LegalIR/output/e1_evidence_frozen_ce/e1_complement_frozen_ce_scores.jsonl
Task1 - LegalIR/experiments/track_b_v11_research/artifacts/complement_split/complement_queries_1200.json
models/thanhtantran_Vietnamese_Reranker/
```

The code validates hashes/provenance itself.  Do not substitute a similarly
named artifact from another machine or experiment.  A mismatch is an intended
hard failure, not a reason to change an input manually.

## 1. Install the patch and run the CPU-only checks

```bash
cd /root/autodl-tmp/legalir
unzip -o /path/to/LegalIR_E3B_Patch.zip

python3 -m unittest \
  "Task1 - LegalIR/tests/test_e2_mil_finetune_oof.py" \
  "Task1 - LegalIR/tests/test_e3b_residual_score_gate.py" \
  "Task1 - LegalIR/tests/test_e3b_nested_mil_scores.py"

MODE=preflight bash "Task1 - LegalIR/experiments/track_b_v11_research/run_e3b_nested_mil.sh" \
  2>&1 | tee /root/autodl-tmp/e3b_preflight.log
```

Preflight is label-blind and has no GPU training.  It writes
`Task1 - LegalIR/output/e3b_nested_mil/e3b_preflight.json`; it must report
`holdout_opened: false` and `public_labels_used: false`.

## 2. Standard nested OOF — 20 E2 fits

Run only after preflight succeeds.  Auto-probe measures complete E2 train and
score steps in clean subprocesses, chooses a safe batch using currently free
VRAM, and freezes that runtime choice for this run.

```bash
cd /root/autodl-tmp/legalir
MODE=standard \
bash "Task1 - LegalIR/experiments/track_b_v11_research/run_e3b_nested_mil.sh" \
  2>&1 | tee /root/autodl-tmp/e3b_standard.log
```

Standard is exactly five outer folds × two inner policy-split seeds × two E2
neural seeds = **20 fresh E2 fits**.  It is resumable at a completed outer
fold from:

```text
Task1 - LegalIR/output/e3b_nested_mil/e3b_standard_progress.json
```

If the process is interrupted, run the *same* `MODE=standard` command again.
The checkpoint verifies code, inputs, model fingerprint, runtime contract,
full-c35 score artifact hashes and partition QIDs before resuming.  Do not use
`RESUME=false` unless intentionally discarding the whole E3-B standard run.

## 3. Inspect the standard gate

```bash
python3 - <<'PY'
import json
p = "Task1 - LegalIR/output/e3b_nested_mil/e3b_standard_report.json"
r = json.load(open(p, encoding="utf-8"))
print(json.dumps({
    "gates": r["gates"],
    "promotion_to_stress_allowed": r["promotion_to_stress_allowed"],
    "holdout_opened": r["holdout_opened"],
    "public_labels_used": r["public_labels_used"],
    "checkpoint": r["checkpoint"],
}, ensure_ascii=False, indent=2))
PY
```

Only `promotion_to_stress_allowed: true` permits the next command.  A failed
standard gate means stop and preserve the report; do not tune thresholds or
open holdout.

## 4. Stress transfer checks — 60 E2 fits, only after the standard pass

```bash
cd /root/autodl-tmp/legalir
MODE=stress \
bash "Task1 - LegalIR/experiments/track_b_v11_research/run_e3b_nested_mil.sh" \
  2>&1 | tee /root/autodl-tmp/e3b_stress.log
```

Stress is 20 fits in each of law-source, query-template and near-duplicate
grouping regimes: **60 fresh fits total**.  The script refuses to start if the
matching standard report did not pass.  It is separately resumable through
`e3b_stress_progress.json`.

The final stress report may set `promotion_to_e4_allowed: true` only if every
inner seed passes every stress gate.  This patch still does **not** execute
E4.  Preserve the reports and consult the E4 protocol before any holdout
action.

## Resource and safety notes

- The auto-probe reserves `max(2 GiB, 15% of observed free VRAM)` for a shared
  A100/other shared GPU.  It tests micro-batches 1/2/4/8/16 and score batches
  16/32/64/128/256 using the heaviest E2 examples.  It does not lower the
  token length, evidence count, optimizer or precision to make a fit work.
- To place an explicit upper bound on an allocated GPU, add
  `VRAM_CAP_GIB=72` (example) before the mode command.  Omit it on an
  exclusive GPU.  It is not a request for exclusive access.
- Never manually edit an E3-B output, checkpoint, score JSONL or metadata
  sidecar.  Their hashes are part of the resume contract.
- `RETAIN_BEST_WEIGHTS=true` retains every fitted E2 state and requires large
  disk.  It is `false` by default; the code still records a state hash for the
  exact weights used to score C and V.
