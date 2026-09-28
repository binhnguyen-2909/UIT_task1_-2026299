#!/usr/bin/env bash
# E1 complement-only evidence diagnostic.  It never opens holdout/public.
set -euo pipefail

TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="$(cd "$TASK_ROOT/.." && pwd)"
PHASE="${PHASE:-all}" # materialize (BQB GPU), score (reranker GPU), or all
PROPOSAL_DEVICE="${PROPOSAL_DEVICE:-cuda}"
SCORE_DEVICE="${SCORE_DEVICE:-cuda}"
BQB_BATCH_SIZE="${BQB_BATCH_SIZE:-128}"
RERANKER_BATCH_SIZE="${RERANKER_BATCH_SIZE:-16}"
QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-16}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false

python3 "$TASK_ROOT/scripts/e1_evidence_frozen_ce.py" \
  --phase "$PHASE" \
  --contexts-zip "$TASK_ROOT/data/raw/public_test/selected-contexts.zip" \
  --stage1 "$TASK_ROOT/output/v10_9_prf/complement_safe_tail_stage1.audit.json" \
  --stage2 "$TASK_ROOT/output/v10_9_prf/complement_safe_tail_stage2_c35_p3.audit.json" \
  --anchor "$TASK_ROOT/output/v16_rgm/complement_v16b_memory_slot5.json" \
  --reference "$TASK_ROOT/experiments/track_b_v11_research/artifacts/complement_split/complement_queries_1200.json" \
  --bqb-model "$REPO_ROOT/models/bqbbao6_vietnamese-legal-embedding" \
  --reranker-model "$REPO_ROOT/models/thanhtantran_Vietnamese_Reranker" \
  --output-dir "$TASK_ROOT/output/e1_evidence_frozen_ce" \
  --proposal-device "$PROPOSAL_DEVICE" \
  --score-device "$SCORE_DEVICE" \
  --bqb-batch-size "$BQB_BATCH_SIZE" \
  --reranker-batch-size "$RERANKER_BATCH_SIZE" \
  --query-chunk-size "$QUERY_CHUNK_SIZE"
