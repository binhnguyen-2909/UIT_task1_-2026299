# Hidden-state c67 reranker — GPU server runbook

This branch adds only a 1025-parameter head. Qwen is frozen. The expensive
stage is the c67 hidden-state cache; the head fits and selector are small but
run on CUDA as required by the protocol.

The package does not contain the Qwen model, corpus, qrels, E3B reports or
their large score artifacts. The server must already have the exact E3B
standard/stress reports and all four C/V artifacts per fold.

## Upload/install

```bash
cd /root/autodl-tmp
unzip -o LegalIR_HSR_C67_Hidden_Head_Patch.zip -d legalir
cd /root/autodl-tmp/legalir
python3 -m pip install -r requirements-gpuhub.txt

export QWEN_MODEL=/root/autodl-tmp/models/Qwen3-Reranker-0.6B
export QWEN_MODEL_REVISION='immutable-snapshot-revision-or-hash'
export VRAM_RESERVE_GIB=12
export CPU_THREADS=1
export OUTPUT_DIR='/root/autodl-tmp/legalir/Task1 - LegalIR/output/hsr_c67_hidden_head'
```

## 1. Contract preflight — no labels, no GPU forward

```bash
MODE=preflight bash "Task1 - LegalIR/experiments/track_b_v11_research/run_hsr_c67_hidden_head.sh" \
  2>&1 | tee /root/autodl-tmp/hsr_preflight.log
```

This must validate the c35 E3B artifacts, fold manifests, hashes and the
under-4B plan. Do not proceed if it fails.

## 2. GPU exporter preflight and pilot

```bash
MODE=hidden-preflight bash "Task1 - LegalIR/experiments/track_b_v11_research/run_hsr_c67_hidden_head.sh"

MODE=pilot bash "Task1 - LegalIR/experiments/track_b_v11_research/run_hsr_c67_hidden_head.sh" \
  2>&1 | tee /root/autodl-tmp/hsr_hidden_pilot.log
```

Pilot uses a deterministic QID sample, runs auto-probe, verifies hidden-size
and native-logit parity, and writes resumable feature rows. Check measured
throughput including tokenize, forward and disk write. Do not extrapolate from
the GPU name alone.

## 3. Full complement cache

```bash
MODE=cache bash "Task1 - LegalIR/experiments/track_b_v11_research/run_hsr_c67_hidden_head.sh" \
  2>&1 | tee /root/autodl-tmp/hsr_hidden_cache.log
```

There are at most `1197 * 67 = 80,199` pairs. A second GPU may run the same
command with `SHARD_INDEX=1 SHARD_COUNT=2`; never share a shard. A stopped
process resumes from completed whole-QID blocks with `RESUME=true`.

## 4. Nested head preflight and standard

```bash
MODE=eval-preflight bash "Task1 - LegalIR/experiments/track_b_v11_research/run_hsr_c67_hidden_head.sh"

`MODE=eval-preflight` writes `hsr_eval_preflight_report.json`, keeping the
contract preflight report intact. Set `HIDDEN_CACHE_ROOT` only when the
immutable cache lives in a separate output directory.

MODE=standard bash "Task1 - LegalIR/experiments/track_b_v11_research/run_hsr_c67_hidden_head.sh" \
  2>&1 | tee /root/autodl-tmp/hsr_standard.log
```

`hsr_standard_report.json` must report
`promotion_to_stress_allowed:true`. The head checkpoints are under
`output/hsr_c67_hidden_head/standard/head_weights/` and are retained even if a
gate fails.

## 5. Stress

```bash
MODE=stress bash "Task1 - LegalIR/experiments/track_b_v11_research/run_hsr_c67_hidden_head.sh" \
  2>&1 | tee /root/autodl-tmp/hsr_stress.log
```

Stress is blocked unless standard passes with the exact experiment lineage.
No old E4 parity command, old holdout ledger, or public labels are opened by
this runbook.

## Deadline calculation

Use the measured conservative pilot rate `r` and the real public query count:

```text
1.5 * (1197*67 + N_public*67) / r + 12 hours outside cache <= hours remaining
```

If this inequality fails, stop after the pilot or use the accepted V16b/E3B
fallback. A passing complement gate is not a public-score guarantee.
