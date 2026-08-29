param(
    [ValidateSet('auto', 'cuda', 'cpu')]
    [string]$Device = 'auto',
    [int]$QueryBatchSize = 8,
    [int]$DocumentBatchSize = 1,
    [int]$ScoreQueryBatchSize = 8,
    [int]$CheckpointEvery = 25,
    [int]$MaxDocumentsThisRun = 0,
    [switch]$ApplyPublic
)

$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$repoRoot = (Resolve-Path (Join-Path $taskRoot '..')).Path
$scripts = Join-Path $taskRoot 'scripts'
$raw = Join-Path $taskRoot 'data\raw\public_test'
$processed = Join-Path $taskRoot 'data\processed'
$output = Join-Path $taskRoot 'output'
$prf = Join-Path $output 'v10_9_prf'
$artifacts = Join-Path $PSScriptRoot 'artifacts'
$destination = Join-Path $output 'v14_colbert_full'

$env:PYTHONUTF8 = '1'
$env:PYTHONHASHSEED = '20260828'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:OMP_NUM_THREADS = '8'
$env:MKL_NUM_THREADS = '8'

$arguments = @(
    (Join-Path $scripts 'v14_colbert_late_interaction.py'),
    '--contexts-zip', (Join-Path $raw 'selected-contexts.zip'),
    '--index', (Join-Path $processed 'v11_research_fit_256\legalir.sqlite3'),
    '--score-cache', (Join-Path $output 'v13_v11h_rebuild\v11h_scores.npz'),
    '--feature-cache', (Join-Path $output 'v11_feature_cache\v2_query_norm_c80.npz'),
    '--complement-stage1', (Join-Path $prf 'complement_safe_tail_stage1.audit.json'),
    '--complement-stage2', (Join-Path $prf 'complement_safe_tail_stage2_c35_p3.audit.json'),
    '--complement-reference', (Join-Path $artifacts 'complement_split\complement_queries_1200.json'),
    '--complement-pair-cache', (Join-Path $output 'v14_replacement_full\cache\complement_pairs.npz'),
    '--holdout-stage1', (Join-Path $prf 'holdout_safe_tail_locked_stage1.audit.json'),
    '--holdout-stage2', (Join-Path $prf 'holdout_safe_tail_stage2_c35_p3.audit.json'),
    '--holdout-reference', (Join-Path $raw 'train.json'),
    '--public-stage1', (Join-Path $prf 'public_safe_tail_stage1.audit.json'),
    '--public-stage2', (Join-Path $prf 'public_safe_tail_stage2_c35_p3.audit.json'),
    '--output-dir', $destination,
    '--device', $Device,
    '--query-batch-size', $QueryBatchSize,
    '--document-batch-size', $DocumentBatchSize,
    '--score-query-batch-size', $ScoreQueryBatchSize,
    '--checkpoint-every', $CheckpointEvery,
    '--minimum-promote-recall', '0.925'
)
if ($MaxDocumentsThisRun -gt 0) {
    $arguments += @('--max-documents-this-run', $MaxDocumentsThisRun)
}
if ($ApplyPublic) {
    $arguments += '--apply-public'
}

New-Item -ItemType Directory -Force -Path $destination | Out-Null
& python @arguments
exit $LASTEXITCODE
