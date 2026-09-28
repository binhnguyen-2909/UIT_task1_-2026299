#!/usr/bin/env bash
# PV1 Qwen3-1.7B pairwise verifier. Complement-only until a fully passing
# stress report authorizes final policy/public inference. Never opens holdout.
set -euo pipefail

TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="$(cd "$TASK_ROOT/.." && pwd)"
MODE="${MODE:-preflight}" # preflight, build-pairs, pilot, cache, eval-preflight, standard, stress, finalize
PV1_RUN_ID="${PV1_RUN_ID:-pv1_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$TASK_ROOT/output/pv1_pairwise_abstain/$PV1_RUN_ID}"
F_SHORTLIST="${F_SHORTLIST:?Set F_SHORTLIST to the pinned F shortlist/packs JSONL}"
F_MANIFEST="${F_MANIFEST:?Set F_MANIFEST to the pinned F source manifest JSON}"
PV1_MODEL="${PV1_MODEL:?Set PV1_MODEL to a local Qwen3-1.7B snapshot}"
PV1_MODEL_REVISION="${PV1_MODEL_REVISION:?Set PV1_MODEL_REVISION to the immutable model revision/hash}"
VRAM_RESERVE_GIB="${VRAM_RESERVE_GIB:?Set VRAM_RESERVE_GIB for the shared GPU}"
CPU_THREADS="${CPU_THREADS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"; SHARD_COUNT="${SHARD_COUNT:-1}"
RESUME="${RESUME:-true}"
PROMOTION_GATE_SOURCE="${PROMOTION_GATE_SOURCE:-$TASK_ROOT/scripts/e3b_nested_mil_scores.py}"
PROMOTION_GATE_CONFIG="${PROMOTION_GATE_CONFIG:-$TASK_ROOT/V27_A_RESOLVED_CONFIG.json}"
PAIR_ROWS="$OUTPUT_DIR/pv1_pairs.jsonl"; PAIR_MANIFEST="$OUTPUT_DIR/pv1_pair_manifest.json"
SCORE_DIR="$OUTPUT_DIR"

case "$RESUME" in
  true) RESUME_FLAG=(--resume) ;;
  false) RESUME_FLAG=(--no-resume) ;;
  *) echo "RESUME must be true or false" >&2; exit 2 ;;
esac

export OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS" OPENBLAS_NUM_THREADS="$CPU_THREADS" NUMEXPR_MAX_THREADS="$CPU_THREADS"
export PYTHONIOENCODING=utf-8 TOKENIZERS_PARALLELISM=false

CONTRACT=(
  python3 "$TASK_ROOT/scripts/pv1_contract.py"
  --stage1 "$TASK_ROOT/output/v10_9_prf/complement_safe_tail_stage1.audit.json"
  --stage2 "$TASK_ROOT/output/v10_9_prf/complement_safe_tail_stage2_c35_p3.audit.json"
  --anchor "$TASK_ROOT/output/v16_rgm/complement_v16b_memory_slot5.json"
  --reference "$TASK_ROOT/experiments/track_b_v11_research/artifacts/complement_split/complement_queries_1200.json"
  --f-shortlist "$F_SHORTLIST" --f-manifest "$F_MANIFEST" --output-dir "$OUTPUT_DIR"
)
BUILD=(
  python3 "$TASK_ROOT/scripts/pv1_build_pairs.py"
  --stage1 "$TASK_ROOT/output/v10_9_prf/complement_safe_tail_stage1.audit.json"
  --stage2 "$TASK_ROOT/output/v10_9_prf/complement_safe_tail_stage2_c35_p3.audit.json"
  --anchor "$TASK_ROOT/output/v16_rgm/complement_v16b_memory_slot5.json"
  --f-shortlist "$F_SHORTLIST" --f-manifest "$F_MANIFEST" --output-dir "$OUTPUT_DIR"
)
SCORE=(
  python3 "$TASK_ROOT/scripts/pv1_score_pairs.py"
  --pairs "$PAIR_ROWS" --pair-manifest "$PAIR_MANIFEST" --model "$PV1_MODEL" --model-revision "$PV1_MODEL_REVISION"
  --output-dir "$OUTPUT_DIR" --vram-reserve-gib "$VRAM_RESERVE_GIB" --cpu-threads "$CPU_THREADS"
  --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT" "${RESUME_FLAG[@]}"
)
EVAL=(
  python3 "$TASK_ROOT/scripts/pv1_nested_eval.py"
  --stage1 "$TASK_ROOT/output/v10_9_prf/complement_safe_tail_stage1.audit.json"
  --stage2 "$TASK_ROOT/output/v10_9_prf/complement_safe_tail_stage2_c35_p3.audit.json"
  --anchor "$TASK_ROOT/output/v16_rgm/complement_v16b_memory_slot5.json"
  --reference "$TASK_ROOT/experiments/track_b_v11_research/artifacts/complement_split/complement_queries_1200.json"
  --e1-manifest "$TASK_ROOT/output/e1_evidence_frozen_ce/e1_complement_evidence.jsonl"
  --pairs "$PAIR_ROWS" --pair-manifest "$PAIR_MANIFEST" --score-dir "$SCORE_DIR"
  --promotion-gate-source "$PROMOTION_GATE_SOURCE" --promotion-gate-config "$PROMOTION_GATE_CONFIG"
  --output-dir "$OUTPUT_DIR" "${RESUME_FLAG[@]}"
)

case "$MODE" in
  preflight) "${CONTRACT[@]}" ;;
  build-pairs) "${BUILD[@]}" ;;
  pilot) "${SCORE[@]}" --mode pilot ;;
  cache) "${SCORE[@]}" --mode cache ;;
  eval-preflight) "${EVAL[@]}" --mode preflight ;;
  standard) "${EVAL[@]}" --mode standard ;;
  stress) "${EVAL[@]}" --mode stress --standard-report "$OUTPUT_DIR/pv1_standard_report.json" ;;
  finalize)
    : "${DEADLINE:?Set DEADLINE as ISO-8601 with timezone}"
    : "${REMAINING_WORK_SECONDS:?Set measured remaining public/bundle ETA including reserve}"
    python3 "$TASK_ROOT/scripts/pv1_finalize.py" \
      --stage1 "$TASK_ROOT/output/v10_9_prf/complement_safe_tail_stage1.audit.json" \
      --stage2 "$TASK_ROOT/output/v10_9_prf/complement_safe_tail_stage2_c35_p3.audit.json" \
      --anchor "$TASK_ROOT/output/v16_rgm/complement_v16b_memory_slot5.json" \
      --reference "$TASK_ROOT/experiments/track_b_v11_research/artifacts/complement_split/complement_queries_1200.json" \
      --pairs "$PAIR_ROWS" --pair-manifest "$PAIR_MANIFEST" --score-dir "$SCORE_DIR" \
      --stress-report "$OUTPUT_DIR/pv1_stress_report.json" --verifier-model "$PV1_MODEL" --output-dir "$OUTPUT_DIR" \
      --deadline "$DEADLINE" --remaining-work-seconds "$REMAINING_WORK_SECONDS" \
      --bundle-path "$OUTPUT_DIR/pv1_reproducible_bundle.tar.gz"
    ;;
  *) echo "Unknown MODE=$MODE" >&2; exit 2 ;;
esac
