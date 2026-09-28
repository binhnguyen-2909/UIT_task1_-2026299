#!/usr/bin/env bash
# E3-C is complement-only research.  It does not open, write, or authorize
# holdout/public data, and it never alters the separately locked E4 policy.
set -euo pipefail

TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="$(cd "$TASK_ROOT/.." && pwd)"
MODE="${MODE:-preflight}" # preflight, standard, stress
OUTPUT_DIR="${OUTPUT_DIR:-$TASK_ROOT/output/e3c_nested_duo_lora}"
E3B_OUTPUT_DIR="${E3B_OUTPUT_DIR:-$TASK_ROOT/output/e3b_nested_mil}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-8}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
RUNTIME_CONTRACT="${RUNTIME_CONTRACT:-}"
VRAM_CAP_GIB="${VRAM_CAP_GIB:-}"
RESUME="${RESUME:-true}"
CPU_THREADS="${CPU_THREADS:-1}"
INNER_SEEDS="${INNER_SEEDS:-}"

case "$GRADIENT_CHECKPOINTING" in
  true) CHECKPOINT_FLAG=(--gradient-checkpointing) ;;
  false) CHECKPOINT_FLAG=(--no-gradient-checkpointing) ;;
  *) echo "GRADIENT_CHECKPOINTING must be true or false" >&2; exit 2 ;;
esac
case "$RESUME" in
  true) RESUME_FLAG=(--resume) ;;
  false) RESUME_FLAG=(--no-resume) ;;
  *) echo "RESUME must be true or false" >&2; exit 2 ;;
esac

export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_MAX_THREADS="$CPU_THREADS"
export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false

BASE=(
  python3 "$TASK_ROOT/scripts/e3c_nested_duo_lora.py"
  --stage1 "$TASK_ROOT/output/v10_9_prf/complement_safe_tail_stage1.audit.json"
  --stage2 "$TASK_ROOT/output/v10_9_prf/complement_safe_tail_stage2_c35_p3.audit.json"
  --anchor "$TASK_ROOT/output/v16_rgm/complement_v16b_memory_slot5.json"
  --reference "$TASK_ROOT/experiments/track_b_v11_research/artifacts/complement_split/complement_queries_1200.json"
  --e1-manifest "$TASK_ROOT/output/e1_evidence_frozen_ce/e1_complement_evidence.jsonl"
  --e1-manifest-meta "$TASK_ROOT/output/e1_evidence_frozen_ce/e1_complement_evidence.meta.json"
  --e1-frozen-scores "$TASK_ROOT/output/e1_evidence_frozen_ce/e1_complement_frozen_ce_scores.jsonl"
  --reranker-model "$REPO_ROOT/models/thanhtantran_Vietnamese_Reranker"
  --e3b-output-dir "$E3B_OUTPUT_DIR"
  --e3b-standard-report "$E3B_OUTPUT_DIR/e3b_standard_report.json"
  --e3b-stress-report "$E3B_OUTPUT_DIR/e3b_stress_report.json"
  --output-dir "$OUTPUT_DIR"
  --epochs 2
  --learning-rate 5e-5
  --gradient-accumulation "$GRADIENT_ACCUMULATION"
  --cpu-threads "$CPU_THREADS"
  "${CHECKPOINT_FLAG[@]}"
)

if [[ -n "$VRAM_CAP_GIB" ]]; then BASE+=(--vram-cap-gib "$VRAM_CAP_GIB"); fi
# shellcheck disable=SC2206
if [[ -n "$INNER_SEEDS" ]]; then BASE+=(--inner-seeds $INNER_SEEDS); fi
if [[ -n "$RUNTIME_CONTRACT" ]]; then BASE+=(--runtime-contract "$RUNTIME_CONTRACT" --no-auto-probe); fi

case "$MODE" in
  preflight) "${BASE[@]}" --mode preflight --force ;;
  standard) "${BASE[@]}" --mode standard "${RESUME_FLAG[@]}" --force ;;
  stress) "${BASE[@]}" --mode stress --standard-report "$OUTPUT_DIR/e3c_standard_report.json" "${RESUME_FLAG[@]}" --force ;;
  *) echo "Unknown MODE=$MODE; choose preflight, standard, or stress" >&2; exit 2 ;;
esac
