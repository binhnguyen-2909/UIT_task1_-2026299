param(
    [switch]$Resume,
    [switch]$NoReranker,
    [int]$DenseBatchSize = 32,
    [int]$DenseMaxSeqLength = 256,
    [int]$RerankerBatchSize = 8,
    [string]$IndexName = 'v11_research_256',
    [ValidateSet('cuda', 'cpu')]
    [string]$Device = 'cuda'
)

$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$repoRoot = (Resolve-Path (Join-Path $taskRoot '..')).Path
$python = Join-Path $repoRoot '.python312-embed\python.exe'
$main = Join-Path $taskRoot 'scripts\main.py'
$data = Join-Path $taskRoot 'data\raw\public_test'
$index = Join-Path $taskRoot ('data\processed\' + $IndexName)
$output = Join-Path $taskRoot ('output\public_submission_' + $IndexName + '_top5.zip')
$registry = Join-Path $taskRoot 'model_registry.v10.example.json'
$embedding = Join-Path $repoRoot 'models\bqbbao6_vietnamese-legal-embedding'
$reranker = Join-Path $repoRoot 'models\thanhtantran_Vietnamese_Reranker'

$env:OMP_NUM_THREADS = '8'
$env:MKL_NUM_THREADS = '8'
$env:TOKENIZERS_PARALLELISM = 'false'

New-Item -ItemType Directory -Force -Path (Split-Path $output) | Out-Null

# V11 changes are explicit and isolated from v10/v57 artifacts: legal chunking,
# wider candidate reach, longer Stage-2 inputs and adaptive evidence hydration.
# The only neural models are those registered in model_registry.v10.example.json
# (BQB 0.33B + Vietnamese reranker 0.76B); v57 checkpoints/LLM outputs are not
# loaded or used.
$trainArgs = @(
    $main, 'train',
    '--corpus', (Join-Path $data 'selected-contexts.zip'),
    '--train-data', (Join-Path $data 'train.json'),
    '--index-dir', $index,
    '--chunk-strategy', 'legal',
    '--chunk-size', '1200',
    '--chunk-overlap', '150',
    '--normalize-corpus',
    '--clean-training-data',
    '--embedding-model', $embedding,
    '--candidates', '200',
    '--dense-weight', '0.30',
    '--dense-rrf-k', '20',
    '--dense-max-seq-length', [string]$DenseMaxSeqLength,
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
if ($Resume) { $trainArgs += '--reuse-index' }
& $python @trainArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not $NoReranker) {
    $inferArgs = @(
        $main, 'infer',
        '--index-dir', $index,
        '--input', (Join-Path $data 'public-official.json'),
        '--output', $output,
        '--reranker-model', $reranker,
        '--rerank-candidates', '80',
        '--rerank-passages-per-doc', '3',
        '--adaptive-rerank',
        '--reranker-aggregation', 'top3',
        '--reranker-fp16',
        '--reranker-batch-size', [string]$RerankerBatchSize,
        '--reranker-max-length', '768',
        '--reranker-rrf-weight', '1.0',
        '--reranker-rrf-k', '10',
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

Write-Output "v11 research run finished: $output"
