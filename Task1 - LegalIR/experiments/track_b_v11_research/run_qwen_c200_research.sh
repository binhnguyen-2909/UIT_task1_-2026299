#!/usr/bin/env bash
# Qwen c35/c67/c200 research branch.  Complement only; never opens E4/E5.
set -euo pipefail

TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="$(cd "$TASK_ROOT/.." && pwd)"
MODE="${MODE:-preflight}" # preflight, cache, cache-preflight, standard, stress, bundle
OUTPUT_DIR="${OUTPUT_DIR:-$TASK_ROOT/output/qwen_c200_research}"
QWEN_MODEL="${QWEN_MODEL:-$REPO_ROOT/models/Qwen3-Reranker-0.6B}"
CACHE_ROOT="${CACHE_ROOT:-$OUTPUT_DIR/qwen_cache}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
BATCH_CANDIDATES="${BATCH_CANDIDATES:-1 2 4 8 16}"
ATTENTION="${ATTENTION:-sdpa}"
CPU_THREADS="${CPU_THREADS:-1}"
RESUME="${RESUME:-true}"
VARIANTS="${VARIANTS:-A_c35_short}"

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

CACHE_BASE=(
  python3 "$TASK_ROOT/scripts/qwen_c200_cache.py"
  --stage1 "$TASK_ROOT/output/v10_9_prf/complement_safe_tail_stage1.audit.json"
  --stage2 "$TASK_ROOT/output/v10_9_prf/complement_safe_tail_stage2_c35_p3.audit.json"
  --anchor "$TASK_ROOT/output/v16_rgm/complement_v16b_memory_slot5.json"
  --contexts-zip "$TASK_ROOT/data/raw/public_test/selected-contexts.zip"
  --qwen-model "$QWEN_MODEL"
  --output-dir "$OUTPUT_DIR"
  --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT"
  --batch-candidates $BATCH_CANDIDATES
  --attention "$ATTENTION" --cpu-threads "$CPU_THREADS" --device cuda
)

EVAL_BASE=(
  python3 "$TASK_ROOT/scripts/qwen_c200_nested_eval.py"
  --stage1 "$TASK_ROOT/output/v10_9_prf/complement_safe_tail_stage1.audit.json"
  --stage2 "$TASK_ROOT/output/v10_9_prf/complement_safe_tail_stage2_c35_p3.audit.json"
  --anchor "$TASK_ROOT/output/v16_rgm/complement_v16b_memory_slot5.json"
  --reference "$TASK_ROOT/experiments/track_b_v11_research/artifacts/complement_split/complement_queries_1200.json"
  --e1-manifest "$TASK_ROOT/output/e1_evidence_frozen_ce/e1_complement_evidence.jsonl"
  --qwen-model "$QWEN_MODEL"
  --short-cache-dir "$CACHE_ROOT/A_c35_short"
  --long-cache-dir "$CACHE_ROOT/D_c200_long"
  --e3b-standard-report "$TASK_ROOT/output/e3b_nested_mil/e3b_standard_report.json"
  --e3b-stress-report "$TASK_ROOT/output/e3b_nested_mil/e3b_stress_report.json"
  --output-dir "$OUTPUT_DIR" --cpu-threads "$CPU_THREADS" --device cuda
)

case "$MODE" in
  preflight)
    # shellcheck disable=SC2086
    "${CACHE_BASE[@]}" --mode preflight --variants $VARIANTS
    ;;
  cache-preflight)
    "${EVAL_BASE[@]}" --mode preflight --force
    ;;
  cache)
    # shellcheck disable=SC2086
    "${CACHE_BASE[@]}" --mode score --variants $VARIANTS "${RESUME_FLAG[@]}"
    ;;
  standard)
    "${EVAL_BASE[@]}" --mode standard "${RESUME_FLAG[@]}" --force
    ;;
  stress)
    "${EVAL_BASE[@]}" --mode stress --standard-report "$OUTPUT_DIR/qwen_c200_standard_report.json" "${RESUME_FLAG[@]}" --force
    ;;
  bundle)
    RESULT_REPORT="${RESULT_REPORT:-$OUTPUT_DIR/qwen_c200_stress_report.json}"
    if [[ ! -f "$RESULT_REPORT" ]]; then
      RESULT_REPORT="$OUTPUT_DIR/qwen_c200_standard_report.json"
    fi
    python3 "$TASK_ROOT/scripts/qwen_c200_bundle.py" \
      --output-dir "$OUTPUT_DIR" \
      --result-report "$RESULT_REPORT" \
      --stage1 "$TASK_ROOT/output/v10_9_prf/complement_safe_tail_stage1.audit.json" \
      --stage2 "$TASK_ROOT/output/v10_9_prf/complement_safe_tail_stage2_c35_p3.audit.json" \
      --anchor "$TASK_ROOT/output/v16_rgm/complement_v16b_memory_slot5.json" \
      --reference "$TASK_ROOT/experiments/track_b_v11_research/artifacts/complement_split/complement_queries_1200.json" \
      --e1-manifest "$TASK_ROOT/output/e1_evidence_frozen_ce/e1_complement_evidence.jsonl" \
      --e3b-standard-report "$TASK_ROOT/output/e3b_nested_mil/e3b_standard_report.json" \
      --e3b-stress-report "$TASK_ROOT/output/e3b_nested_mil/e3b_stress_report.json" \
      --qwen-model "$QWEN_MODEL"
    ;;
  *)
    echo "Unknown MODE=$MODE; choose preflight, cache, cache-preflight, standard, stress, or bundle" >&2
    exit 2
    ;;
esac
