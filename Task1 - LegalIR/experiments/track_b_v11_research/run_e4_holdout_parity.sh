#!/usr/bin/env bash
# E4 is a one-time locked-holdout parity gate for the fixed E3-B policy.
set -euo pipefail

TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="$(cd "$TASK_ROOT/.." && pwd)"
MODE="${MODE:-preflight}" # preflight, prepare, parity
OUTPUT_DIR="${OUTPUT_DIR:-$TASK_ROOT/output/e4_holdout_parity}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-16}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
VRAM_CAP_GIB="${VRAM_CAP_GIB:-}"
RUNTIME_CONTRACT="${RUNTIME_CONTRACT:-}"
RESUME="${RESUME:-true}"
CPU_THREADS="${CPU_THREADS:-1}"

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
  python3 "$TASK_ROOT/scripts/e4_holdout_parity.py"
  --e3b-standard-report "$TASK_ROOT/output/e3b_nested_mil/e3b_standard_report.json"
  --e3b-stress-report "$TASK_ROOT/output/e3b_nested_mil/e3b_stress_report.json"
  --contexts-zip "$TASK_ROOT/data/raw/public_test/selected-contexts.zip"
  --complement-stage1 "$TASK_ROOT/output/v10_9_prf/complement_safe_tail_stage1.audit.json"
  --complement-stage2 "$TASK_ROOT/output/v10_9_prf/complement_safe_tail_stage2_c35_p3.audit.json"
  --complement-anchor "$TASK_ROOT/output/v16_rgm/complement_v16b_memory_slot5.json"
  --complement-reference "$TASK_ROOT/experiments/track_b_v11_research/artifacts/complement_split/complement_queries_1200.json"
  --complement-e1-manifest "$TASK_ROOT/output/e1_evidence_frozen_ce/e1_complement_evidence.jsonl"
  --complement-e1-manifest-meta "$TASK_ROOT/output/e1_evidence_frozen_ce/e1_complement_evidence.meta.json"
  --complement-e1-frozen-scores "$TASK_ROOT/output/e1_evidence_frozen_ce/e1_complement_frozen_ce_scores.jsonl"
  --holdout-stage1 "$TASK_ROOT/output/v10_9_prf/holdout_safe_tail_locked_stage1.audit.json"
  --holdout-stage2 "$TASK_ROOT/output/v10_9_prf/holdout_safe_tail_stage2_c35_p3.audit.json"
  --holdout-anchor "$TASK_ROOT/output/v16_rgm/holdout_v16b_memory_slot5.json"
  --holdout-reference "$TASK_ROOT/experiments/track_b_v11_research/artifacts/holdout_1000.json"
  --bqb-model "$REPO_ROOT/models/bqbbao6_vietnamese-legal-embedding"
  --reranker-model "$REPO_ROOT/models/thanhtantran_Vietnamese_Reranker"
  --output-dir "$OUTPUT_DIR"
  --epochs 2
  --gradient-accumulation "$GRADIENT_ACCUMULATION"
  --cpu-threads "$CPU_THREADS"
  "${CHECKPOINT_FLAG[@]}"
)

if [[ -n "$VRAM_CAP_GIB" ]]; then
  BASE+=(--vram-cap-gib "$VRAM_CAP_GIB")
fi
if [[ -n "$RUNTIME_CONTRACT" ]]; then
  BASE+=(--runtime-contract "$RUNTIME_CONTRACT" --no-auto-probe)
fi

case "$MODE" in
  preflight)
    "${BASE[@]}" --mode preflight --device cuda
    ;;
  prepare)
    "${BASE[@]}" --mode prepare --device cuda "${RESUME_FLAG[@]}"
    ;;
  parity)
    # The one-time reference read stays intact; its metric gate runs under the
    # same CUDA-only runtime profile as label-blind prepare.
    "${BASE[@]}" --mode parity --device cuda
    ;;
  *)
    echo "Unknown MODE=$MODE; choose preflight, prepare, or parity" >&2
    exit 2
    ;;
esac
