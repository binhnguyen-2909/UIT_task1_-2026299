#!/usr/bin/env bash
# One-command GPU server launcher for the complete E0--E2 capsule.
# E0 is reproduced; sealed E1 artifacts are verified by default; E2 first
# runs an exact memory probe, then only its unrun stress gates.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ROOT="$ROOT_DIR/Task1 - LegalIR"
E1_RUNNER="$TASK_ROOT/experiments/track_b_v11_research/run_e1_complement.sh"
E2_RUNNER="$TASK_ROOT/experiments/track_b_v11_research/run_e2_mil_oof.sh"

AUTO_INSTALL="${AUTO_INSTALL:-true}"
REBUILD_E0="${REBUILD_E0:-true}"
# E1 was already frozen on the complement-only split and is shipped in this
# capsule.  Set REBUILD_E1=true only when intentionally rerunning that
# diagnostic from scratch; it is not required to run E2 stress validation.
REBUILD_E1="${REBUILD_E1:-false}"
RUN_E2_STRESS="${RUN_E2_STRESS:-true}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -d "$TASK_ROOT" ]] || fail "Task directory is missing: $TASK_ROOT"
command -v python3 >/dev/null 2>&1 || fail "python3 is required. Select a GPUHub PyTorch image."
command -v nvidia-smi >/dev/null 2>&1 || fail "No NVIDIA GPU is visible. Select a CUDA/GPU instance."

GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d '[:space:]')"
VRAM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]')"
[[ "$GPU_COUNT" =~ ^[0-9]+$ && "$GPU_COUNT" -ge 1 ]] || fail "nvidia-smi did not report a usable GPU."
[[ "$VRAM_MIB" =~ ^[0-9]+$ ]] || fail "Could not determine GPU VRAM from nvidia-smi."

# Conservative automatic profiles.  They alter only loader/compute batching,
# never the candidate c35 universe, model, loss, seeds, or E2 gate protocol.
if (( VRAM_MIB >= 40000 )); then
  PROFILE="high-memory"
  BQB_BATCH_SIZE=128
  RERANKER_BATCH_SIZE=16
  MICRO_BATCH=16
  SCORE_BATCH_SIZE=64
  GRADIENT_CHECKPOINTING=false
elif (( VRAM_MIB >= 24000 )); then
  PROFILE="medium-memory"
  BQB_BATCH_SIZE=32
  RERANKER_BATCH_SIZE=4
  MICRO_BATCH=1
  SCORE_BATCH_SIZE=4
  GRADIENT_CHECKPOINTING=true
else
  PROFILE="low-memory"
  BQB_BATCH_SIZE=8
  RERANKER_BATCH_SIZE=1
  MICRO_BATCH=1
  SCORE_BATCH_SIZE=1
  GRADIENT_CHECKPOINTING=true
fi
GRADIENT_ACCUMULATION=16

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "{\"event\":\"e0_e2_auto_start\",\"gpu_count\":$GPU_COUNT,\"vram_mib\":$VRAM_MIB,\"profile\":\"$PROFILE\",\"micro_batch\":$MICRO_BATCH,\"score_batch_size\":$SCORE_BATCH_SIZE,\"gradient_checkpointing\":$GRADIENT_CHECKPOINTING}"
nvidia-smi

if [[ "$AUTO_INSTALL" == "true" ]]; then
  python3 -m pip install --no-cache-dir -r "$ROOT_DIR/requirements-gpuhub.txt"
elif [[ "$AUTO_INSTALL" != "false" ]]; then
  fail "AUTO_INSTALL must be true or false."
fi

if [[ "$REBUILD_E0" == "true" ]]; then
  echo "[E0] reproducing locked V16b baseline"
  bash "$ROOT_DIR/run_v16b_gpu.sh"
elif [[ "$REBUILD_E0" != "false" ]]; then
  fail "REBUILD_E0 must be true or false."
fi

if [[ "$REBUILD_E1" == "true" ]]; then
  echo "[E1] rebuilding frozen complement-only diagnostic"
  PHASE=all BQB_BATCH_SIZE="$BQB_BATCH_SIZE" RERANKER_BATCH_SIZE="$RERANKER_BATCH_SIZE" \
    QUERY_CHUNK_SIZE=16 bash "$E1_RUNNER"
elif [[ "$REBUILD_E1" == "false" ]]; then
  for artifact in \
    "$TASK_ROOT/output/e1_evidence_frozen_ce/e1_complement_evidence.jsonl" \
    "$TASK_ROOT/output/e1_evidence_frozen_ce/e1_complement_evidence.meta.json" \
    "$TASK_ROOT/output/e1_evidence_frozen_ce/e1_complement_frozen_ce_scores.jsonl" \
    "$TASK_ROOT/output/e1_evidence_frozen_ce/e1_complement_frozen_ce_report.json"; do
    [[ -s "$artifact" ]] || fail "Sealed E1 artifact is missing: $artifact. Re-run with REBUILD_E1=true."
  done
  echo "[E1] sealed complement-only artifacts verified; not rebuilt."
else
  fail "REBUILD_E1 must be true or false."
fi

echo "[E2] preflight"
MODE=preflight bash "$E2_RUNNER"

echo "[E2] exact CUDA memory probe"
if ! MODE=probe MICRO_BATCH="$MICRO_BATCH" SCORE_BATCH_SIZE="$SCORE_BATCH_SIZE" \
  GRADIENT_ACCUMULATION="$GRADIENT_ACCUMULATION" GRADIENT_CHECKPOINTING="$GRADIENT_CHECKPOINTING" \
  bash "$E2_RUNNER"; then
  cat >&2 <<'EOF'
E2 exact AdamW memory probe did not fit this GPU. No long E2 run was started.
Keep the included E0/E1/E2 artifacts; continue E2 only on a GPU where this
probe passes (the tested high-memory profile uses at least 40 GiB VRAM).
EOF
  exit 3
fi

if [[ "$RUN_E2_STRESS" == "true" ]]; then
  echo "[E2] running only the three pending stress regimes (never re-runs standard)"
  MODE=stress-regimes MICRO_BATCH="$MICRO_BATCH" SCORE_BATCH_SIZE="$SCORE_BATCH_SIZE" \
    GRADIENT_ACCUMULATION="$GRADIENT_ACCUMULATION" GRADIENT_CHECKPOINTING="$GRADIENT_CHECKPOINTING" \
    RESUME=true bash "$E2_RUNNER"
elif [[ "$RUN_E2_STRESS" == "false" ]]; then
  echo "[E2] probe passed; stress-regimes skipped because RUN_E2_STRESS=false."
else
  fail "RUN_E2_STRESS must be true or false."
fi

echo "Completed. E0 report: $TASK_ROOT/output/v16_rgm/v16b_memory_ltr_report.json"
echo "E2 standard report: $TASK_ROOT/output/e2_mil_oof/e2_mil_oof_report.json"
echo "E2 stress report: $TASK_ROOT/output/e2_mil_oof/e2_mil_oof_stress_report.json"
