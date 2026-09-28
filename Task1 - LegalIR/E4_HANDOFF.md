# E4 server handoff — run once, then stop for review

This patch is for a server that already contains the validated E0–E2 capsule
and completed E3-B reports/artifacts.  It performs the authorized E4 parity
gate; it does not run E5 or submit a public file.

## Before running

Check that the E3-B server output is present and that these two files report
the E3-B pass:

```text
Task1 - LegalIR/output/e3b_nested_mil/e3b_standard_report.json
Task1 - LegalIR/output/e3b_nested_mil/e3b_stress_report.json
```

Do not rename, edit, or recreate an E3-B report.  The patch checks their
lineage hash and refuses mismatches.  E4 also requires the frozen V16b
holdout anchor already present in the capsule:

```text
Task1 - LegalIR/output/v16_rgm/holdout_v16b_memory_slot5.json
```

## 1. Install and preflight — CUDA runtime profile, no holdout labels

```bash
cd /root/autodl-tmp/legalir
unzip -o /path/to/LegalIR_E4_Parity_Patch.zip

python3 -m unittest \
  "Task1 - LegalIR/tests/test_e1_evidence_frozen_ce.py" \
  "Task1 - LegalIR/tests/test_e2_mil_finetune_oof.py" \
  "Task1 - LegalIR/tests/test_e3b_residual_score_gate.py" \
  "Task1 - LegalIR/tests/test_e4_holdout_parity.py"

MODE=preflight \
bash "Task1 - LegalIR/experiments/track_b_v11_research/run_e4_holdout_parity.sh" \
  2>&1 | tee /root/autodl-tmp/e4_preflight.log
```

Preflight must say `holdout_labels_opened: false`.  It can inspect frozen
holdout candidates/c35, but it never opens `holdout_1000.json`.

## 2. Prepare — GPU, still no holdout labels

```bash
cd /root/autodl-tmp/legalir
MODE=prepare \
bash "Task1 - LegalIR/experiments/track_b_v11_research/run_e4_holdout_parity.sh" \
  2>&1 | tee /root/autodl-tmp/e4_prepare.log
```

This builds the label-blind holdout evidence pack, auto-probes current free
VRAM, fits exactly two deployment E2 models, saves them, scores all holdout
c35 candidates, and writes fixed-policy predictions.  It does not load
holdout qrels.  If interrupted, run the same `MODE=prepare` command again;
`e4_prepare_progress.json` verifies hashes and resumes at the final-model
boundary.

Inspect this before allowing parity:

```bash
python3 - <<'PY'
import json
p = "Task1 - LegalIR/output/e4_holdout_parity/e4_prepare_report.json"
r = json.load(open(p, encoding="utf-8"))
print(json.dumps({
    "stage": r["stage"],
    "policy": r["locked_policy"],
    "models": [{"seed": x["seed"], "weights": x["weights"]} for x in r["deployment_models"]],
    "holdout_labels_opened": r["holdout_labels_opened"],
    "predictions": r["predictions"],
}, ensure_ascii=False, indent=2))
PY
```

## 3. Parity — one irreversible CUDA evaluation

Run this only after the owner approves the prepared artifact.  It opens the
locked holdout reference once and then creates a ledger which blocks reruns.
The membership/Recall computation runs on CUDA; JSON reading remains minimal
host I/O and all OpenMP/BLAS pools are capped at one CPU thread.

```bash
cd /root/autodl-tmp/legalir
MODE=parity \
bash "Task1 - LegalIR/experiments/track_b_v11_research/run_e4_holdout_parity.sh" \
  2>&1 | tee /root/autodl-tmp/e4_parity.log
```

Then report **only** the aggregate result; do not tune from it:

```bash
python3 - <<'PY'
import json
p = "Task1 - LegalIR/output/e4_holdout_parity/e4_holdout_parity_report.json"
r = json.load(open(p, encoding="utf-8"))
print(json.dumps({
    "metrics_vs_v16b": r["metrics_vs_v16b"],
    "promotion": r["promotion"],
    "holdout_access_count": r["holdout_access_count"],
    "public_predictions_materialized": r["public_predictions_materialized"],
}, ensure_ascii=False, indent=2))
PY
```

If `promotion_to_e5_allowed` is false, keep V16b and stop.  If it is true,
send the report back before materializing public predictions: E5 is a separate
implementation/release step.
