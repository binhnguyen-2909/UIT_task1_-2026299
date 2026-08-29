param(
    [int]$Limit = 0
)

$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$scripts = Join-Path $taskRoot 'scripts'
$raw = Join-Path $taskRoot 'data\raw\public_test'
$processed = Join-Path $taskRoot 'data\processed'
$output = Join-Path $taskRoot 'output'
$prf = Join-Path $output 'v10_9_prf'
$artifacts = Join-Path $PSScriptRoot 'artifacts'
$destination = Join-Path $output 'v15_counterfactual'

$env:PYTHONUTF8 = '1'
$env:PYTHONHASHSEED = '20260829'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:OMP_NUM_THREADS = '8'
$env:MKL_NUM_THREADS = '8'

$arguments = @(
    (Join-Path $scripts 'v15_counterfactual_adjudication.py'),
    '--contexts-zip', (Join-Path $raw 'selected-contexts.zip'),
    '--index', (Join-Path $processed 'v11_research_fit_256\legalir.sqlite3'),
    '--score-cache', (Join-Path $output 'v13_v11h_rebuild\v11h_scores.npz'),
    '--feature-cache', (Join-Path $output 'v11_feature_cache\v2_query_norm_c80.npz'),
    '--complement-stage1', (Join-Path $prf 'complement_safe_tail_stage1.audit.json'),
    '--complement-stage2', (Join-Path $prf 'complement_safe_tail_stage2_c35_p3.audit.json'),
    '--complement-reference', (Join-Path $artifacts 'complement_split\complement_queries_1200.json'),
    '--holdout-stage1', (Join-Path $prf 'holdout_safe_tail_locked_stage1.audit.json'),
    '--holdout-stage2', (Join-Path $prf 'holdout_safe_tail_stage2_c35_p3.audit.json'),
    '--holdout-reference', (Join-Path $raw 'train.json'),
    '--output-dir', $destination
)
if ($Limit -gt 0) {
    $arguments += @('--limit', $Limit)
}

New-Item -ItemType Directory -Force -Path $destination | Out-Null
& python @arguments
exit $LASTEXITCODE
