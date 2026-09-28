param(
    [ValidateSet('holdout', 'complement', 'public')]
    [string]$Split = 'holdout',
    [ValidateSet('cuda', 'cpu')]
    [string]$Device = 'cuda',
    [int]$BatchSize = 64,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$repoRoot = (Resolve-Path (Join-Path $taskRoot '..')).Path
$embeddedPython = Join-Path $repoRoot '.python312-embed\python.exe'
if (Test-Path -LiteralPath $embeddedPython) {
    $python = $embeddedPython
} else {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $pythonCommand) {
        throw 'Python runtime not found. Install Python 3.12+ or provide .python312-embed\python.exe.'
    }
    $python = $pythonCommand.Source
}

$script = Join-Path $taskRoot 'scripts\v16_memory_oracle.py'
$raw = Join-Path $taskRoot 'data\raw\public_test'
$processed = Join-Path $taskRoot 'data\processed'
$artifacts = Join-Path $PSScriptRoot 'artifacts'
$outputDir = Join-Path $taskRoot 'output\v16_rgm'
$model = Join-Path $repoRoot 'models\bqbbao6_vietnamese-legal-embedding'

if ($Split -eq 'holdout') {
    $memoryData = Join-Path $artifacts 'train_fit_excluding_holdout.json'
    $queryData = Join-Path $artifacts 'holdout_1000.json'
    $anchor = Join-Path $taskRoot 'output\v10_9_prf\holdout_safe_tail_locked_stage1.audit.json'
    $memoryEmbeddings = Join-Path $processed 'v11_research_fit_256\question_dense_embeddings.npy'
    $prefix = 'holdout'
} elseif ($Split -eq 'complement') {
    $memoryData = Join-Path $artifacts 'holdout_1000.json'
    $queryData = Join-Path $artifacts 'complement_split\complement_queries_1200.json'
    $anchor = Join-Path $taskRoot 'output\v10_9_prf\complement_safe_tail_stage1.audit.json'
    $memoryEmbeddings = Join-Path $processed 'v11_research_256\question_dense_embeddings.npy'
    $prefix = 'complement'
} else {
    $memoryData = Join-Path $raw 'train.json'
    $queryData = Join-Path $raw 'public-official.json'
    $anchor = Join-Path $taskRoot 'output\v10_9_prf\public_safe_tail_stage1.audit.json'
    $memoryEmbeddings = Join-Path $processed 'v11_research_256\question_dense_embeddings.npy'
    $prefix = 'public'
}

$report = Join-Path $outputDir "${prefix}_memory_oracle_report.json"
$features = Join-Path $outputDir "${prefix}_memory_features.json"
foreach ($path in @($memoryData, $queryData, $anchor, $memoryEmbeddings, $model)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required V16a input is missing: $path"
    }
}
if ((Test-Path -LiteralPath $report) -and (Test-Path -LiteralPath $features) -and -not $Force) {
    Write-Output "SKIP V16a $Split (artifacts exist)"
    exit 0
}

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
$env:PYTHONIOENCODING = 'utf-8'
$env:TOKENIZERS_PARALLELISM = 'false'

& $python $script `
    '--memory-data' $memoryData `
    '--query-data' $queryData `
    '--anchor-audit' $anchor `
    '--memory-embeddings' $memoryEmbeddings `
    '--embedding-model' $model `
    '--device' $Device `
    '--batch-size' ([string]$BatchSize) `
    '--max-seq-length' '256' `
    '--memory-k' '50' `
    '--temperature' '0.07' `
    '--degree-exponent' '0.5' `
    '--reciprocal-floor' '0.25' `
    '--anchor-cutoff' '35' `
    '--memory-top-k' '5' `
    '--drop-exact-overlap' `
    '--features-output' $features `
    '--output' $report
if ($LASTEXITCODE -ne 0) {
    throw "V16a $Split failed with exit code $LASTEXITCODE"
}
Write-Output "V16a $Split report: $report"
