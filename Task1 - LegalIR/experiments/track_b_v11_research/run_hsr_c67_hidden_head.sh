#!/usr/bin/env bash
# Hidden-state c67 reranker. Complement-only until the branch passes its own
# nested standard/stress gates; it never opens the old holdout ledger.
set -euo pipefail

TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${MODE:-preflight}" # preflight, hidden-preflight, pilot, cache, eval-preflight, standard, stress
OUTPUT_DIR="${OUTPUT_DIR:-$TASK_ROOT/output/hsr_c67_hidden_head}"
HIDDEN_CACHE_ROOT="${HIDDEN_CACHE_ROOT:-$OUTPUT_DIR}"
QWEN_MODEL="${QWEN_MODEL:?Set QWEN_MODEL to an offline Qwen3-Reranker-0.6B snapshot}"
QWEN_MODEL_REVISION="${QWEN_MODEL_REVISION:?Set QWEN_MODEL_REVISION to an immutable snapshot revision/hash}"
VRAM_RESERVE_GIB="${VRAM_RESERVE_GIB:?Set VRAM_RESERVE_GIB for the shared GPU}"
CPU_THREADS="${CPU_THREADS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"; SHARD_COUNT="${SHARD_COUNT:-1}"
RESUME="${RESUME:-true}"
E3B_STANDARD_REPORT="${E3B_STANDARD_REPORT:-$TASK_ROOT/output/e3b_nested_mil/e3b_standard_report.json}"
E3B_STRESS_REPORT="${E3B_STRESS_REPORT:-$TASK_ROOT/output/e3b_nested_mil/e3b_stress_report.json}"
GATE_SOURCE="${GATE_SOURCE:-$TASK_ROOT/scripts/e3b_nested_mil_scores.py}"
GATE_CONFIG="${GATE_CONFIG:-$TASK_ROOT/V27_A_RESOLVED_CONFIG.json}"

case "$RESUME" in
  true) RESUME_FLAG=(--resume) ;;
  false) RESUME_FLAG=(--no-resume) ;;
  *) echo "RESUME must be true or false" >&2; exit 2 ;;
esac

export OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS" OPENBLAS_NUM_THREADS="$CPU_THREADS" NUMEXPR_MAX_THREADS="$CPU_THREADS"
export PYTHONIOENCODING=utf-8 TOKENIZERS_PARALLELISM=false

COMMON=(
  --stage1 "$TASK_ROOT/output/v10_9_prf/complement_safe_tail_stage1.audit.json"
  --stage2 "$TASK_ROOT/output/v10_9_prf/complement_safe_tail_stage2_c35_p3.audit.json"
  --anchor "$TASK_ROOT/output/v16_rgm/complement_v16b_memory_slot5.json"
  --contexts-zip "$TASK_ROOT/data/raw/public_test/selected-contexts.zip"
)

CONTRACT=(
  python3 "$TASK_ROOT/scripts/hsr_contract.py" "${COMMON[@]}"
  --qwen-model "$QWEN_MODEL" --model-revision "$QWEN_MODEL_REVISION"
  --e3b-standard-report "$E3B_STANDARD_REPORT" --e3b-stress-report "$E3B_STRESS_REPORT"
  --e1-manifest "$TASK_ROOT/output/e1_evidence_frozen_ce/e1_complement_evidence.jsonl"
  --gate-source "$GATE_SOURCE" --gate-config "$GATE_CONFIG" --output-dir "$OUTPUT_DIR"
)

EXPORT=(
  python3 "$TASK_ROOT/scripts/hsr_hidden_export.py" "${COMMON[@]}"
  --qwen-model "$QWEN_MODEL" --model-revision "$QWEN_MODEL_REVISION" --output-dir "$OUTPUT_DIR"
  --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT" --vram-reserve-gib "$VRAM_RESERVE_GIB"
  --cpu-threads "$CPU_THREADS" "${RESUME_FLAG[@]}"
)

EVAL=(
  python3 "$TASK_ROOT/scripts/hsr_nested_head.py" "${COMMON[@]}"
  --reference "$TASK_ROOT/experiments/track_b_v11_research/artifacts/complement_split/complement_queries_1200.json"
  --e1-manifest "$TASK_ROOT/output/e1_evidence_frozen_ce/e1_complement_evidence.jsonl"
  --e3b-standard-report "$E3B_STANDARD_REPORT" --e3b-stress-report "$E3B_STRESS_REPORT"
  --hidden-cache-root "$HIDDEN_CACHE_ROOT" --gate-source "$GATE_SOURCE" --gate-config "$GATE_CONFIG"
  --output-dir "$OUTPUT_DIR" "${RESUME_FLAG[@]}"
)

case "$MODE" in
  preflight)
    "${CONTRACT[@]}" ;;
  hidden-preflight)
    "${EXPORT[@]}" --mode preflight ;;
  pilot)
    "${EXPORT[@]}" --mode pilot 2>&1 | tee "${HSR_LOG:-/root/autodl-tmp/hsr_hidden_pilot.log}" ;;
  cache)
    "${EXPORT[@]}" --mode cache 2>&1 | tee "${HSR_LOG:-/root/autodl-tmp/hsr_hidden_cache.log}" ;;
  eval-preflight)
    "${EVAL[@]}" --mode preflight --report-path "$OUTPUT_DIR/hsr_eval_preflight_report.json" ;;
  standard)
    "${EVAL[@]}" --mode standard 2>&1 | tee "${HSR_LOG:-/root/autodl-tmp/hsr_standard.log}" ;;
  stress)
    "${EVAL[@]}" --mode stress --standard-report "$OUTPUT_DIR/hsr_standard_report.json" 2>&1 | tee "${HSR_LOG:-/root/autodl-tmp/hsr_stress.log}" ;;
  *) echo "Unknown MODE=$MODE" >&2; exit 2 ;;
esac
