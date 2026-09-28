# R1 C35 Long Fusion — GPU server runbook

R1 is a separate complement-only experiment. It keeps V16b top-4 and slot-5
incumbent rules, uses the two fold-safe E3B MIL score exports, and adds the
frozen Qwen3-Reranker-0.6B long score over every c35 document. It does not use
PV1, does not retrain E2, does not open holdout, and does not overwrite E3B/E4/E5.

The ZIP does not contain the large model, corpus, qrels, E3B score cache, or
E3B reports. Copy those inputs to the server and set their paths explicitly.
The F/shortlist-8 cache is not enough for R1: R1 needs all fold-safe E3B c35
scores and the Qwen c35-long cache.

## 1. Upload and install

```bash
cd /root/autodl-tmp
unzip -o LegalIR_R1_C35_Long_Fusion_Patch.zip -d legalir
cd /root/autodl-tmp/legalir
python3 -m pip install -r requirements-gpuhub.txt
```

Set the Qwen snapshot and immutable revision. The revision is metadata, not a
permission to download a moving `main`; the snapshot must be available offline.

```bash
export QWEN_MODEL=/root/autodl-tmp/models/Qwen3-Reranker-0.6B
export QWEN_MODEL_REVISION='<immutable-revision-or-snapshot-hash>'
export VRAM_RESERVE_GIB=12
export CPU_THREADS=1
export OUTPUT_DIR='/root/autodl-tmp/legalir/Task1 - LegalIR/output/r1_c35_long_fusion'
```

## 2. Label-blind contract preflight, then GPU pilot

Run the contract preflight first. It validates the E3B report, all referenced
fold-safe C/V score artifacts, hashes, and the gate source without opening
complement qrels. It must finish successfully before any GPU scoring:

```bash
MODE=preflight bash "Task1 - LegalIR/experiments/track_b_v11_research/run_r1_c35_long_fusion.sh" \
  2>&1 | tee /root/autodl-tmp/r1_preflight.log
```

Then run the GPU-only model/corpus preflight and deterministic pilot:

```bash
MODE=qwen-preflight bash "Task1 - LegalIR/experiments/track_b_v11_research/run_r1_c35_long_fusion.sh" \
  2>&1 | tee /root/autodl-tmp/r1_qwen_preflight.log

MODE=qwen-pilot bash "Task1 - LegalIR/experiments/track_b_v11_research/run_r1_c35_long_fusion.sh"
```

The contract report must show `under_4b_estimate:true`; the pilot report must
show the actual parameter ledger with `under_4b:true`. Pilot uses a
deterministic 64-QID sample selected by QID hash, not by qrels or law-source
errors. Confirm both reports have `holdout_opened:false`, cache lineage for
`R1_C35_LONG`, and a measured batch/ETA that leaves time for the full c35
cache, evaluation, and any permitted release path.

## 3. Full GPU cache

```bash
MODE=qwen-cache bash "Task1 - LegalIR/experiments/track_b_v11_research/run_r1_c35_long_fusion.sh" \
  2>&1 | tee /root/autodl-tmp/r1_qwen_cache.log
```

There are at most `1197 × 35 = 41,895` Qwen query-document scores. The cache
uses the existing validated yes/no reranker prompt and legal-boundary packer,
4096-token maximum, BF16, auto-probed batch, atomic whole-QID resume, and
per-row input/model hashes. A second GPU may use a distinct `SHARD_INDEX` with
the same `SHARD_COUNT`; never share one shard between workers.

## 4. Evaluator and gates

```bash
MODE=eval-preflight bash "Task1 - LegalIR/experiments/track_b_v11_research/run_r1_c35_long_fusion.sh"
MODE=standard bash "Task1 - LegalIR/experiments/track_b_v11_research/run_r1_c35_long_fusion.sh" \
  2>&1 | tee /root/autodl-tmp/r1_standard.log
```

`r1_standard_report.json` must have `promotion_to_stress_allowed:true` before:

```bash
MODE=stress bash "Task1 - LegalIR/experiments/track_b_v11_research/run_r1_c35_long_fusion.sh" \
  2>&1 | tee /root/autodl-tmp/r1_stress.log
```

The evaluator tests lambda `0` by delegating to the original E3B fixed policy
(memory 0, threshold 0.50), then evaluates only lambda `0.25` and `0.50` as
new fusion choices. It also compares against historical E3B OOF, reports
rescue/destructive/neutral swaps, and preserves all top-4/order/distinct-ID
invariants. It will fail on incomplete fold C/V score artifacts, use of final
deployment weights in place of OOF exports, cache lineage drift, or a gate
source/config hash mismatch.

R1 has no new trainable weights. Keep the Qwen snapshot, cache, reports,
normalization/policy code, and E3B artifacts. `new_policy_holdout_validated`
remains false; a R1 report is not permission to overwrite the accepted
E3B/E4/E5 release.
