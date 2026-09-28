#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ROOT="$ROOT_DIR/Task1 - LegalIR"

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "Python 3 was not found in PATH." >&2
    exit 1
fi

required_paths=(
    "$TASK_ROOT/data/raw/public_test/selected-contexts.zip"
    "$TASK_ROOT/data/processed/v11_research_fit_256/legalir.sqlite3"
    "$TASK_ROOT/output/v10_9_prf/complement_safe_tail_stage1.audit.json"
    "$TASK_ROOT/output/v10_9_prf/complement_safe_tail_stage2_c35_p3.audit.json"
    "$TASK_ROOT/output/v10_9_prf/holdout_safe_tail_locked_stage1.audit.json"
    "$TASK_ROOT/output/v10_9_prf/holdout_safe_tail_stage2_c35_p3.audit.json"
    "$TASK_ROOT/output/v10_9_prf/public_safe_tail_stage1.audit.json"
    "$TASK_ROOT/output/v10_9_prf/public_safe_tail_stage2_c35_p3.audit.json"
    "$TASK_ROOT/experiments/track_b_v11_research/artifacts/holdout_1000.json"
    "$TASK_ROOT/experiments/track_b_v11_research/artifacts/complement_split/complement_queries_1200.json"
    "$TASK_ROOT/output/v11_feature_cache/v2_query_norm_c80.npz"
    "$TASK_ROOT/output/v16_rgm/holdout_v11h_locked_predictions.json"
    "$TASK_ROOT/output/v11h_query_norm_ltr/v11h_query_normalized_ltr.json"
    "$TASK_ROOT/output/v16_rgm/complement_memory_features.json"
    "$TASK_ROOT/output/v16_rgm/holdout_memory_features.json"
    "$TASK_ROOT/output/v16_rgm/public_memory_features.json"
)

for path in "${required_paths[@]}"; do
    if [[ ! -e "$path" ]]; then
        echo "Required V16b input is missing: $path" >&2
        exit 2
    fi
done

export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false

# Prefer the original runner when PowerShell 7 is available. The Python path
# below is equivalent for the frozen-artifact case and works in Linux images
# that do not ship pwsh.
if command -v pwsh >/dev/null 2>&1; then
    exec pwsh -NoProfile -File \
        "$TASK_ROOT/experiments/track_b_v11_research/run_v16b_memory_slot5.ps1" \
        -Device cuda
fi

exec "$PYTHON_BIN" "$TASK_ROOT/scripts/v16_memory_ltr.py" \
    --contexts-zip "$TASK_ROOT/data/raw/public_test/selected-contexts.zip" \
    --prior-index "$TASK_ROOT/data/processed/v11_research_fit_256/legalir.sqlite3" \
    --complement-stage1 "$TASK_ROOT/output/v10_9_prf/complement_safe_tail_stage1.audit.json" \
    --complement-stage2 "$TASK_ROOT/output/v10_9_prf/complement_safe_tail_stage2_c35_p3.audit.json" \
    --complement-reference "$TASK_ROOT/experiments/track_b_v11_research/artifacts/complement_split/complement_queries_1200.json" \
    --complement-memory-features "$TASK_ROOT/output/v16_rgm/complement_memory_features.json" \
    --holdout-stage1 "$TASK_ROOT/output/v10_9_prf/holdout_safe_tail_locked_stage1.audit.json" \
    --holdout-stage2 "$TASK_ROOT/output/v10_9_prf/holdout_safe_tail_stage2_c35_p3.audit.json" \
    --holdout-reference "$TASK_ROOT/experiments/track_b_v11_research/artifacts/holdout_1000.json" \
    --holdout-memory-features "$TASK_ROOT/output/v16_rgm/holdout_memory_features.json" \
    --holdout-anchor-json "$TASK_ROOT/output/v16_rgm/holdout_v11h_locked_predictions.json" \
    --public-stage1 "$TASK_ROOT/output/v10_9_prf/public_safe_tail_stage1.audit.json" \
    --public-stage2 "$TASK_ROOT/output/v10_9_prf/public_safe_tail_stage2_c35_p3.audit.json" \
    --public-memory-features "$TASK_ROOT/output/v16_rgm/public_memory_features.json" \
    --public-anchor-json "$TASK_ROOT/output/v11h_query_norm_ltr/v11h_query_normalized_ltr.json" \
    --output-dir "$TASK_ROOT/output/v16_rgm" \
    --candidate-limit 35 \
    --model-profile pointwise-v2-winner \
    --minimum-holdout-delta 0.0 \
    --submission-name v16b_memory_slot5.json \
    --complement-predictions "$TASK_ROOT/output/v16_rgm/complement_v16b_memory_slot5.json"
