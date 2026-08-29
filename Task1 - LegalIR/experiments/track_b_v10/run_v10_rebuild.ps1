param(
    [switch]$Resume,
    [switch]$NoReranker,
    [int]$DenseBatchSize = 32,
    [int]$RerankerBatchSize = 8,
    [ValidateSet('cuda', 'cpu')]
    [string]$Device = 'cuda'
)

$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$repoRoot = (Resolve-Path (Join-Path $taskRoot '..')).Path
$python = Join-Path $repoRoot '.python312-embed\python.exe'
$data = Join-Path $taskRoot 'data\raw\public_test'
$index = Join-Path $taskRoot 'data\processed\v10_github_rebuild'
$output = Join-Path $taskRoot 'output\track_b_v10_github_rebuild.zip'
$registry = Join-Path $taskRoot 'model_registry.v10.example.json'
$embedding = Join-Path $repoRoot 'models\bqbbao6_vietnamese-legal-embedding'
$reranker = Join-Path $repoRoot 'models\thanhtantran_Vietnamese_Reranker'

# Prefer CUDA when available; CPU remains selectable for machines without a
# usable CUDA-enabled PyTorch runtime. The model, chunking and scoring
# configuration are unchanged.
$env:OMP_NUM_THREADS = '8'
$env:MKL_NUM_THREADS = '8'
$env:TOKENIZERS_PARALLELISM = 'false'

New-Item -ItemType Directory -Force -Path (Split-Path $output) | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot 'artifacts') | Out-Null

$trainArgs = @(
    'scripts\main.py', 'train',
    '--corpus', (Join-Path $data 'selected-contexts.zip'),
    '--train-data', (Join-Path $data 'train.json'),
    '--index-dir', $index,
    '--chunk-strategy', 'fixed',
    '--chunk-size', '1200',
    '--chunk-overlap', '150',
    '--embedding-model', $embedding,
    '--candidates', '100',
    '--dense-weight', '0.05',
    '--dense-rrf-k', '60',
    '--question-dense-weight', '0.02',
    '--question-dense-rrf-k', '60',
    '--question-dense-candidates', '20',
    '--model-registry', $registry,
    '--device', $Device,
    '--dense-device', $Device,
    '--dense-batch-size', [string]$DenseBatchSize,
    '--dense-checkpoint-every', '100',
    '--no-normalize-corpus',
    '--no-clean-training-data',
    '--no-reranker-fp16',
    '--no-reranker'
)
if ($Resume) { $trainArgs += '--reuse-index' }
& $python @trainArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not $NoReranker) {
    $inferArgs = @(
        'scripts\main.py', 'infer',
        '--index-dir', $index,
        '--input', (Join-Path $data 'public-official.json'),
        '--output', $output,
        '--reranker-model', $reranker,
        '--rerank-candidates', '35',
        '--rerank-passages-per-doc', '2',
        '--reranker-batch-size', [string]$RerankerBatchSize,
        '--reranker-max-length', '256',
        '--reranker-rrf-weight', '0.8',
        '--reranker-rrf-k', '5',
        '--no-adaptive-rerank',
        '--top-k', '5',
        '--checkpoint-every', '5',
        '--model-registry', $registry,
        '--device', $Device
    )
    $inferCheckpoint = [System.IO.Path]::ChangeExtension($output, '.checkpoint.json')
    if ($Resume -and (Test-Path -LiteralPath $inferCheckpoint)) {
        $inferArgs += '--resume'
    }
    & $python @inferArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$manifestArgs = @(
    'experiments\track_b_v10\write_v10_manifest.py',
    '--task-root', $taskRoot,
    '--repo-root', $repoRoot,
    '--index', $index,
    '--output', (Join-Path $PSScriptRoot 'artifacts\v10_rebuild_manifest.json')
)
& $python @manifestArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Output "v10 rebuild finished: $output"
