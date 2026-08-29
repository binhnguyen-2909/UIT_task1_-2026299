param(
    [ValidateSet('cuda', 'cpu')]
    [string]$Device = 'cuda',
    [int]$DenseBatchSize = 32,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$repoRoot = (Resolve-Path (Join-Path $taskRoot '..')).Path
$python = Join-Path $repoRoot '.python312-embed\python.exe'
$main = Join-Path $taskRoot 'scripts\main.py'
$evaluate = Join-Path $taskRoot 'scripts\evaluate_dense_v10_3.py'
$raw = Join-Path $taskRoot 'data\raw\public_test'
$processed = Join-Path $taskRoot 'data\processed'
$output = Join-Path $taskRoot 'output\v10_3_ablation'
$artifacts = Join-Path $PSScriptRoot 'artifacts'
$sourceIndex = Join-Path $processed 'v11_research_fit_256'
$fitTraining = Join-Path $artifacts 'train_fit_excluding_holdout.json'
$holdout = Join-Path $artifacts 'holdout_1000.json'
$registry = Join-Path $taskRoot 'model_registry.v10.example.json'
$embedding = Join-Path $repoRoot 'models\bqbbao6_vietnamese-legal-embedding'
$contexts = Join-Path $raw 'selected-contexts.zip'

$env:OMP_NUM_THREADS = '8'
$env:MKL_NUM_THREADS = '8'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:PYTHONHASHSEED = '20260826'
$env:PYTHONIOENCODING = 'utf-8'

if (-not (Test-Path -LiteralPath (Join-Path $sourceIndex 'legalir.sqlite3'))) {
    throw "Missing clean fit index: $sourceIndex"
}
New-Item -ItemType Directory -Force -Path $output | Out-Null

$variants = @(
    [pscustomobject]@{ Name = 'prefix_256'; Length = 256; Query = 'query: '; Passage = 'passage: '; Existing = $sourceIndex },
    [pscustomobject]@{ Name = 'no_prefix_256'; Length = 256; Query = ''; Passage = ''; Existing = $null },
    [pscustomobject]@{ Name = 'prefix_384'; Length = 384; Query = 'query: '; Passage = 'passage: '; Existing = $null },
    [pscustomobject]@{ Name = 'prefix_512'; Length = 512; Query = 'query: '; Passage = 'passage: '; Existing = $null }
)

foreach ($variant in $variants) {
    if ($variant.Existing) {
        continue
    }
    $index = Join-Path $processed ('v10_3_fit_' + $variant.Name)
    $variant.Existing = $index
    $db = Join-Path $index 'legalir.sqlite3'
    $config = Join-Path $index 'legalir.config.json'
    if ($Force -or -not (Test-Path -LiteralPath $db)) {
        New-Item -ItemType Directory -Force -Path $index | Out-Null
        Copy-Item -LiteralPath (Join-Path $sourceIndex 'legalir.sqlite3') -Destination $db -Force
        Copy-Item -LiteralPath (Join-Path $sourceIndex 'legalir.config.json') -Destination $config -Force
    }
    $manifest = Join-Path $index 'dense_embeddings.manifest.json'
    $needsBuild = $Force -or -not (Test-Path -LiteralPath $manifest)
    if (-not $needsBuild) {
        $saved = Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json
        $needsBuild = (
            [int]$saved.runtime.max_seq_length -ne [int]$variant.Length -or
            [string]$saved.runtime.query_prefix -ne [string]$variant.Query -or
            [string]$saved.runtime.passage_prefix -ne [string]$variant.Passage
        )
    }
    if ($needsBuild) {
        Write-Output "BUILD $($variant.Name)"
        $arguments = @(
            $main, 'train',
            '--corpus', $contexts,
            '--train-data', $fitTraining,
            '--index-dir', $index,
            '--reuse-index',
            '--chunk-strategy', 'legal',
            '--chunk-size', '1200',
            '--chunk-overlap', '150',
            '--normalize-corpus',
            '--clean-training-data',
            '--embedding-model', $embedding,
            '--embedding-query-prefix', [string]$variant.Query,
            '--embedding-passage-prefix', [string]$variant.Passage,
            '--candidates', '200',
            '--dense-weight', '0.30',
            '--dense-rrf-k', '20',
            '--dense-max-seq-length', [string]$variant.Length,
            '--question-dense-weight', '0.05',
            '--question-dense-rrf-k', '20',
            '--question-dense-candidates', '20',
            '--model-registry', $registry,
            '--device', $Device,
            '--dense-device', $Device,
            '--dense-batch-size', [string]$DenseBatchSize,
            '--dense-checkpoint-every', '100',
            '--no-reranker'
        )
        & $python @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "$($variant.Name) dense build failed with exit code $LASTEXITCODE"
        }
    } else {
        Write-Output "SKIP $($variant.Name) (compatible dense cache exists)"
    }
}

$report = Join-Path $output 'dense_screen_holdout.json'
$evaluationArgs = @(
    $evaluate,
    '--queries', $holdout,
    '--output', $report,
    '--device', $Device,
    '--query-batch-size', '64',
    '--score-batch-size', '32',
    '--chunk-limit', '200'
)
foreach ($variant in $variants) {
    $evaluationArgs += @('--index', [string]$variant.Existing)
}
& $python @evaluationArgs
if ($LASTEXITCODE -ne 0) {
    throw "Dense screening failed with exit code $LASTEXITCODE"
}

Write-Output "V10.3 dense ablation finished: $report"
