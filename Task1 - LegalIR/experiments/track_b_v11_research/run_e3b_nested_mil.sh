#!/usr/bin/env bash
# E3-B is a new complement-only GPU experiment.  It never opens holdout/public.
set -euo pipefail

TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="$(cd "$TASK_ROOT/.." && pwd)"
MODE="${MODE:-preflight}" # preflight, standard, or stress
OUTPUT_DIR="${OUTPUT_DIR:-$TASK_ROOT/output/e3b_nested_mil}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-16}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
VRAM_CAP_GIB="${VRAM_CAP_GIB:-}"
RUNTIME_CONTRACT="${RUNTIME_CONTRACT:-}"
RETAIN_BEST_WEIGHTS="${RETAIN_BEST_WEIGHTS:-false}"
RESUME="${RESUME:-true}"
NEURAL_SEEDS="${NEURAL_SEEDS:-}"
INNER_SEEDS="${INNER_SEEDS:-}"

case "$GRADIENT_CHECKPOINTING" in
  true) CHECKPOINT_FLAG=(--gradient-checkpointing) ;;
  false) CHECKPOINT_FLAG=(--no-gradient-checkpointing) ;;
  *) echo "GRADIENT_CHECKPOINTING must be true or false" >&2; exit 2 ;;
esac
case "$RETAIN_BEST_WEIGHTS" in
  true) WEIGHT_FLAG=(--retain-best-weights) ;;
  false) WEIGHT_FLAG=() ;;
  *) echo "RETAIN_BEST_WEIGHTS must be true or false" >&2; exit 2 ;;
esac
case "$RESUME" in
  true) RESUME_FLAG=(--resume) ;;
  false) RESUME_FLAG=(--no-resume) ;;
  *) echo "RESUME must be true or false" >&2; exit 2 ;;
esac

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

BASE=(
  python3 "$TASK_ROOT/scripts/e3b_nested_mil_scores.py"
  --stage1 "$TASK_ROOT/output/v10_9_prf/complement_safe_tail_stage1.audit.json"
  --stage2 "$TASK_ROOT/output/v10_9_prf/complement_safe_tail_stage2_c35_p3.audit.json"
  --anchor "$TASK_ROOT/output/v16_rgm/complement_v16b_memory_slot5.json"
  --reference "$TASK_ROOT/experiments/track_b_v11_research/artifacts/complement_split/complement_queries_1200.json"
  --e1-manifest "$TASK_ROOT/output/e1_evidence_frozen_ce/e1_complement_evidence.jsonl"
  --e1-manifest-meta "$TASK_ROOT/output/e1_evidence_frozen_ce/e1_complement_evidence.meta.json"
  --e1-frozen-scores "$TASK_ROOT/output/e1_evidence_frozen_ce/e1_complement_frozen_ce_scores.jsonl"
  --memory-features "$TASK_ROOT/output/v16_rgm/complement_memory_features.json"
  --reranker-model "$REPO_ROOT/models/thanhtantran_Vietnamese_Reranker"
  --output-dir "$OUTPUT_DIR"
  --epochs 2
  --gradient-accumulation "$GRADIENT_ACCUMULATION"
  "${CHECKPOINT_FLAG[@]}"
)

if [[ -n "$VRAM_CAP_GIB" ]]; then
  BASE+=(--vram-cap-gib "$VRAM_CAP_GIB")
fi
# shellcheck disable=SC2206
if [[ -n "$NEURAL_SEEDS" ]]; then BASE+=(--neural-seeds $NEURAL_SEEDS); fi
# shellcheck disable=SC2206
if [[ -n "$INNER_SEEDS" ]]; then BASE+=(--inner-seeds $INNER_SEEDS); fi
if [[ -n "$RUNTIME_CONTRACT" ]]; then
  BASE+=(--runtime-contract "$RUNTIME_CONTRACT" --no-auto-probe)
fi

case "$MODE" in
  preflight)
    "${BASE[@]}" --mode preflight --force
    ;;
  standard)
    "${BASE[@]}" --mode standard "${WEIGHT_FLAG[@]}" "${RESUME_FLAG[@]}" --force
    ;;
  stress)
    "${BASE[@]}" --mode stress \
      --standard-report "$OUTPUT_DIR/e3b_standard_report.json" \
      "${WEIGHT_FLAG[@]}" "${RESUME_FLAG[@]}" --force
    ;;
  *)
    echo "Unknown MODE=$MODE; choose preflight, standard, or stress" >&2
    exit 2
    ;;
esac
