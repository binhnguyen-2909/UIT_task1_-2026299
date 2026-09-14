# Server handoff — Qwen c200 research branch

Unzip this patch at the root of the E0--E3B capsule.  This patch contains
code and documentation only; it deliberately does not bundle data, model
weights, E1/E2 outputs or previous reports.

## Required existing artifacts

```text
Task1 - LegalIR/output/v10_9_prf/complement_safe_tail_stage1.audit.json
Task1 - LegalIR/output/v10_9_prf/complement_safe_tail_stage2_c35_p3.audit.json
Task1 - LegalIR/output/v16_rgm/complement_v16b_memory_slot5.json
Task1 - LegalIR/output/e1_evidence_frozen_ce/e1_complement_evidence.jsonl
Task1 - LegalIR/output/e3b_nested_mil/e3b_standard_report.json
Task1 - LegalIR/output/e3b_nested_mil/e3b_stress_report.json
Task1 - LegalIR/experiments/track_b_v11_research/artifacts/complement_split/complement_queries_1200.json
Task1 - LegalIR/data/raw/public_test/selected-contexts.zip
```

Download Qwen weights once outside the ZIP.  The runner refuses to download
or substitute a checkpoint automatically:

```bash
cd /root/autodl-tmp/legalir
python3 -m pip install --no-cache-dir -r requirements-gpuhub.txt
huggingface-cli download Qwen/Qwen3-Reranker-0.6B \
  --local-dir models/Qwen3-Reranker-0.6B
```

Use a current CUDA PyTorch image.  `transformers>=4.51` is required.

## Commands

```bash
cd /root/autodl-tmp/legalir
unzip -o /path/to/LegalIR_Qwen_C200_Research_Patch.zip

# Does not use complement labels.  Validates frozen cache inputs only.
MODE=preflight bash "Task1 - LegalIR/experiments/track_b_v11_research/run_qwen_c200_research.sh" \
  2>&1 | tee /root/autodl-tmp/qwen_c200_preflight.log

# Qwen GPU scoring. VARIANTS is explicit — never rely on defaults.
# Keep A running to 1197 first; D_c200_long is dropped from the queue.
VARIANTS="A_c35_short" MODE=cache CPU_THREADS=1 bash "Task1 - LegalIR/experiments/track_b_v11_research/run_qwen_c200_research.sh" \
  2>&1 | tee /root/autodl-tmp/qwen_c200_cache_A.log

# After A completes 1197/1197, score ranks 36-67 only (1-35 reused from A):
VARIANTS="E_c67_short" MODE=cache CPU_THREADS=1 bash "Task1 - LegalIR/experiments/track_b_v11_research/run_qwen_c200_research.sh" \
  2>&1 | tee /root/autodl-tmp/qwen_c200_cache_E.log

# 30-min long-read benchmark BEFORE any 4096 run (see scripts/qwen_bench_long.py):
CPU_THREADS=1 python3 "Task1 - LegalIR/scripts/qwen_bench_long.py" --pairs 32 --max-tokens 4096 2>&1 | tee /root/autodl-tmp/qwen_bench_long.log

# Validates all A+D cache shards and writes the evaluator's label-blind manifest.
MODE=cache-preflight CPU_THREADS=1 bash "Task1 - LegalIR/experiments/track_b_v11_research/run_qwen_c200_research.sh" \
  2>&1 | tee /root/autodl-tmp/qwen_c200_cache_preflight.log

# Nested standard evaluation.  GPU-only metrics/MLP; it requires completed A+D cache.
MODE=standard CPU_THREADS=1 bash "Task1 - LegalIR/experiments/track_b_v11_research/run_qwen_c200_research.sh" \
  2>&1 | tee /root/autodl-tmp/qwen_c200_standard.log
```

Inspect the standard result before allocating stress:

```bash
python3 - <<'PY'
import json
p = "Task1 - LegalIR/output/qwen_c200_research/qwen_c200_standard_report.json"
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

Only if `promotion_to_stress_allowed` is `true`:

```bash
MODE=stress CPU_THREADS=1 bash "Task1 - LegalIR/experiments/track_b_v11_research/run_qwen_c200_research.sh" \
  2>&1 | tee /root/autodl-tmp/qwen_c200_stress.log
```

After standard or stress completes, create and download the reproduction
bundle.  It contains Qwen caches, every S2 fit's `last`/A-stop-selected/backup
checkpoints, scaler, feature schema, policy, reports, code, frozen complement
provenance and one shared Qwen model/tokenizer copy.  It is retained even when
a gate fails.

```bash
MODE=bundle bash "Task1 - LegalIR/experiments/track_b_v11_research/run_qwen_c200_research.sh" \
  2>&1 | tee /root/autodl-tmp/qwen_c200_bundle.log
```

The generated bundle is:

```text
Task1 - LegalIR/output/qwen_c200_research/qwen_c200_reproduction_bundle.tar.gz
```

## Parallel cache shards

For two GPUs, launch the following in separate screens, then run evaluator
only after both finish:

```bash
CUDA_VISIBLE_DEVICES=0 MODE=cache SHARD_INDEX=0 SHARD_COUNT=2 CPU_THREADS=1 \
  bash "Task1 - LegalIR/experiments/track_b_v11_research/run_qwen_c200_research.sh"
CUDA_VISIBLE_DEVICES=1 MODE=cache SHARD_INDEX=1 SHARD_COUNT=2 CPU_THREADS=1 \
  bash "Task1 - LegalIR/experiments/track_b_v11_research/run_qwen_c200_research.sh"
```

The evaluator rejects a partial cache, a mixed model/checkpoint lineage, or a
cache that lacks exactly one record for every expected candidate.  Do not edit
cache JSONL or checkpoints manually.

This branch cannot and must not run E4 parity, E5 or any holdout/public
command.  A research pass is evidence for review, not permission to unseal
those stages.
