#!/usr/bin/env bash
# E2 runs only on the sealed E1 complement artifacts; it cannot submit.
set -euo pipefail

TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="$(cd "$TASK_ROOT/.." && pwd)"
MODE="${MODE:-preflight}" # preflight, probe, standard, stress-regimes, or all-regimes
OUTPUT_DIR="$TASK_ROOT/output/e2_mil_oof"
MICRO_BATCH="${MICRO_BATCH:-16}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-16}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-64}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
RESUME="${RESUME:-true}"

case "$GRADIENT_CHECKPOINTING" in
  true) CHECKPOINT_FLAG=(--gradient-checkpointing) ;;
  false) CHECKPOINT_FLAG=(--no-gradient-checkpointing) ;;
  *) echo "GRADIENT_CHECKPOINTING must be true or false, got: $GRADIENT_CHECKPOINTING" >&2; exit 2 ;;
esac

case "$RESUME" in
  true) RESUME_FLAG=(--resume) ;;
  false) RESUME_FLAG=(--no-resume) ;;
  *) echo "RESUME must be true or false, got: $RESUME" >&2; exit 2 ;;
esac

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false

BASE=(
  python3 "$TASK_ROOT/scripts/e2_mil_finetune_oof.py"
  --stage1 "$TASK_ROOT/output/v10_9_prf/complement_safe_tail_stage1.audit.json"
  --stage2 "$TASK_ROOT/output/v10_9_prf/complement_safe_tail_stage2_c35_p3.audit.json"
  --anchor "$TASK_ROOT/output/v16_rgm/complement_v16b_memory_slot5.json"
  --reference "$TASK_ROOT/experiments/track_b_v11_research/artifacts/complement_split/complement_queries_1200.json"
  --e1-manifest "$TASK_ROOT/output/e1_evidence_frozen_ce/e1_complement_evidence.jsonl"
  --e1-manifest-meta "$TASK_ROOT/output/e1_evidence_frozen_ce/e1_complement_evidence.meta.json"
  --e1-frozen-scores "$TASK_ROOT/output/e1_evidence_frozen_ce/e1_complement_frozen_ce_scores.jsonl"
  --reranker-model "$REPO_ROOT/models/thanhtantran_Vietnamese_Reranker"
  --output-dir "$OUTPUT_DIR"
  --device cuda
)

if [[ "$MODE" == "preflight" ]]; then
  "${BASE[@]}" --preflight --force
  exit 0
fi

# Run this before a long E2 job on any unfamiliar GPU.  The Python program
# performs one *exact* AdamW training step at the selected batch settings and
# writes e2_mil_oof_memory_probe.json.  It is intentionally not a substitute
# for a completed OOF gate.
if [[ "$MODE" == "probe" ]]; then
  "${BASE[@]}" --memory-probe --micro-batch "$MICRO_BATCH" \
    --gradient-accumulation "$GRADIENT_ACCUMULATION" --score-batch-size "$SCORE_BATCH_SIZE" \
    "${CHECKPOINT_FLAG[@]}" --force
  exit 0
fi

if [[ "$MODE" == "standard" ]]; then
  "${BASE[@]}" --regimes standard --epochs 2 --micro-batch "$MICRO_BATCH" \
    --gradient-accumulation "$GRADIENT_ACCUMULATION" --score-batch-size "$SCORE_BATCH_SIZE" \
    "${CHECKPOINT_FLAG[@]}" "${RESUME_FLAG[@]}" --force
  exit 0
fi

if [[ "$MODE" == "all-regimes" ]]; then
  "${BASE[@]}" --regimes standard law_source_family query_template_family near_duplicate_family \
    --epochs 2 --micro-batch "$MICRO_BATCH" --gradient-accumulation "$GRADIENT_ACCUMULATION" \
    --score-batch-size "$SCORE_BATCH_SIZE" \
    "${CHECKPOINT_FLAG[@]}" "${RESUME_FLAG[@]}" --force
  exit 0
fi

if [[ "$MODE" == "stress-regimes" ]]; then
  # Standard OOF was already sealed by the first gate.  This runs only the
  # three unrun out-of-distribution regimes and preserves its report.
  "${BASE[@]}" --regimes law_source_family query_template_family near_duplicate_family \
    --epochs 2 --micro-batch "$MICRO_BATCH" --gradient-accumulation "$GRADIENT_ACCUMULATION" \
    --score-batch-size "$SCORE_BATCH_SIZE" \
    --checkpoint-path "$OUTPUT_DIR/e2_mil_oof_stress_progress.json" \
    --report-path "$OUTPUT_DIR/e2_mil_oof_stress_report.json" \
    "${CHECKPOINT_FLAG[@]}" "${RESUME_FLAG[@]}" --force
  exit 0
fi

echo "Unknown MODE=$MODE; choose preflight, probe, standard, stress-regimes, or all-regimes" >&2
exit 2
