param(
    [ValidateSet('cuda', 'cpu')]
    [string]$Device = 'cuda',
    [int]$DenseBatchSize = 32,
    [int]$Stage1Workers = 8,
    [int]$RerankerBatchSize = 16,
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
$rtkWrapper = Join-Path $repoRoot 'rtk.ps1'
$scripts = Join-Path $taskRoot 'scripts'
$raw = Join-Path $taskRoot 'data\raw\public_test'
$processed = Join-Path $taskRoot 'data\processed'
$output = Join-Path $taskRoot 'output'
$artifacts = Join-Path $PSScriptRoot 'artifacts'
$complementArtifacts = Join-Path $artifacts 'complement_split'
$registry = Join-Path $taskRoot 'model_registry.v10.example.json'
$embedding = Join-Path $repoRoot 'models\bqbbao6_vietnamese-legal-embedding'
$reranker = Join-Path $repoRoot 'models\thanhtantran_Vietnamese_Reranker'
$train = Join-Path $raw 'train.json'
$publicQuestions = Join-Path $raw 'public-official.json'
$contexts = Join-Path $raw 'selected-contexts.zip'
$fitTraining = Join-Path $artifacts 'train_fit_excluding_holdout.json'
$holdout = Join-Path $artifacts 'holdout_1000.json'
$fitIndex = Join-Path $processed 'v11_research_fit_256'
$publicIndex = Join-Path $processed 'v11_research_256'
$complementIndex = Join-Path $processed 'v11_research_complement_256'
$complementQueries = Join-Path $complementArtifacts 'complement_queries_1200.json'

$env:OMP_NUM_THREADS = '8'
$env:MKL_NUM_THREADS = '8'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:PYTHONHASHSEED = '20260825'
$env:PYTHONIOENCODING = 'utf-8'

function Invoke-PythonStep {
    param(
        [string]$Name,
        [string[]]$ExpectedOutputs,
        [object[]]$Arguments
    )
    $complete = $true
    foreach ($path in $ExpectedOutputs) {
        if (-not (Test-Path -LiteralPath $path)) {
            $complete = $false
            break
        }
    }
    if ($complete -and -not $Force) {
        Write-Output "SKIP $Name (artifacts exist)"
        return
    }
    Write-Output "RUN  $Name"
    if (Test-Path -LiteralPath $rtkWrapper) {
        & $rtkWrapper $python @Arguments
    } else {
        & $python @Arguments
    }
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
    foreach ($path in $ExpectedOutputs) {
        if (-not (Test-Path -LiteralPath $path)) {
            throw "$Name did not create expected artifact: $path"
        }
    }
}

function Get-IndexArguments {
    param([string]$IndexPath, [string]$TrainingPath)
    return @(
        (Join-Path $scripts 'main.py'), 'train',
        '--corpus', $contexts,
        '--train-data', $TrainingPath,
        '--index-dir', $IndexPath,
        '--chunk-strategy', 'legal',
        '--chunk-size', '1200',
        '--chunk-overlap', '150',
        '--normalize-corpus',
        '--clean-training-data',
        '--embedding-model', $embedding,
        '--candidates', '200',
        '--dense-weight', '0.30',
        '--dense-rrf-k', '20',
        '--dense-max-seq-length', '256',
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
}

function Get-Stage1Arguments {
    param([string]$IndexPath, [string]$InputPath, [string]$ReferencePath, [string]$OutputPath)
    return @(
        (Join-Path $scripts 'parallel_stage1_infer.py'),
        '--index-dir', $IndexPath,
        '--input', $InputPath,
        '--reference', $ReferencePath,
        '--output', $OutputPath,
        '--model-registry', $registry,
        '--workers', [string]$Stage1Workers,
        '--encode-batch-size', '64',
        '--device', $Device,
        '--profile', 'chunk_only',
        '--checkpoint-every', '100'
    )
}

function Get-Stage2Arguments {
    param([string]$IndexPath, [string]$Stage1Audit, [string]$ReferencePath, [string]$OutputPath)
    return @(
        (Join-Path $scripts 'batch_stage2_rerank.py'),
        '--stage1-audit', $Stage1Audit,
        '--index-dir', $IndexPath,
        '--model', $reranker,
        '--model-registry', $registry,
        '--reference', $ReferencePath,
        '--output', $OutputPath,
        '--candidate-limit', '35',
        '--passages-per-doc', '3',
        '--max-length', '768',
        '--batch-size', [string]$RerankerBatchSize,
        '--device', $Device,
        '--fp16',
        '--neural-weight', '1.0',
        '--rrf-k', '10'
    )
}

New-Item -ItemType Directory -Force -Path $output, $artifacts, $complementArtifacts | Out-Null

Invoke-PythonStep 'materialize holdout' @($holdout) @(
    (Join-Path $scripts 'materialize_v11_holdout.py'),
    '--train', $train,
    '--fit', $fitTraining,
    '--output', $holdout,
    '--expected-questions', '1000'
)

Invoke-PythonStep 'build fit index' @(
    (Join-Path $fitIndex 'legalir.sqlite3'),
    (Join-Path $fitIndex 'dense_embeddings.npy')
) (Get-IndexArguments $fitIndex $fitTraining)

Invoke-PythonStep 'build public index' @(
    (Join-Path $publicIndex 'legalir.sqlite3'),
    (Join-Path $publicIndex 'dense_embeddings.npy')
) (Get-IndexArguments $publicIndex $train)

$holdoutStage1 = Join-Path $output 'v11_holdout_stage1_chunk_only_top5.zip'
$publicStage1 = Join-Path $output 'v11_public_stage1_chunk_only_top5.zip'
Invoke-PythonStep 'holdout Stage-1' @($holdoutStage1, [IO.Path]::ChangeExtension($holdoutStage1, '.audit.json')) `
    (Get-Stage1Arguments $fitIndex $holdout $holdout $holdoutStage1)
Invoke-PythonStep 'public Stage-1' @($publicStage1, [IO.Path]::ChangeExtension($publicStage1, '.audit.json')) `
    (Get-Stage1Arguments $publicIndex $publicQuestions $publicQuestions $publicStage1)

$holdoutStage2 = Join-Path $output 'v11_holdout_stage2_c35_p3_top5.zip'
$publicStage2 = Join-Path $output 'v11_public_stage2_c35_p3_top5.zip'
Invoke-PythonStep 'holdout Stage-2 c35/p3' @($holdoutStage2, [IO.Path]::ChangeExtension($holdoutStage2, '.audit.json')) `
    (Get-Stage2Arguments $fitIndex ([IO.Path]::ChangeExtension($holdoutStage1, '.audit.json')) $holdout $holdoutStage2)
Invoke-PythonStep 'public Stage-2 c35/p3' @($publicStage2, [IO.Path]::ChangeExtension($publicStage2, '.audit.json')) `
    (Get-Stage2Arguments $publicIndex ([IO.Path]::ChangeExtension($publicStage1, '.audit.json')) $publicQuestions $publicStage2)

$complementReport = Join-Path $complementArtifacts 'complement_split_report.json'
if (-not (Test-Path -LiteralPath (Join-Path $complementIndex 'legalir.sqlite3'))) {
    Invoke-PythonStep 'prepare complement index/split' @(
        (Join-Path $complementIndex 'legalir.sqlite3'), $complementQueries, $complementReport
    ) @(
        (Join-Path $scripts 'prepare_v11_complement_split.py'),
        '--source-index', $fitIndex,
        '--output-index', $complementIndex,
        '--train-json', $train,
        '--holdout-audit', ([IO.Path]::ChangeExtension($holdoutStage1, '.audit.json')),
        '--query-count', '1200',
        '--artifact-dir', $complementArtifacts
    )
} else {
    Write-Output 'SKIP prepare complement index/split (artifacts exist)'
}

$complementStage1 = Join-Path $output 'v11_complement_stage1_chunk_only_top5.zip'
Invoke-PythonStep 'complement Stage-1' @(
    $complementStage1, [IO.Path]::ChangeExtension($complementStage1, '.audit.json')
) (Get-Stage1Arguments $complementIndex $complementQueries $complementQueries $complementStage1)

$complementStage2 = Join-Path $output 'v11_complement_stage2_c35_p3_top5.zip'
Invoke-PythonStep 'complement Stage-2 c35/p3' @(
    $complementStage2, [IO.Path]::ChangeExtension($complementStage2, '.audit.json')
) (Get-Stage2Arguments $complementIndex ([IO.Path]::ChangeExtension($complementStage1, '.audit.json')) $complementQueries $complementStage2)

$ltrDir = Join-Path $output 'v11_tiny_ltr_complement_no_prior'
$ltrJson = Join-Path $ltrDir 'v11f_tiny_ltr_complement.json'
$ltrReport = Join-Path $ltrDir 'v11_tiny_ltr_complement_report.json'
Invoke-PythonStep 'grouped complement LTR and locked holdout gate' @($ltrJson, $ltrReport) @(
    (Join-Path $scripts 'v11_tiny_ltr_complement.py'),
    '--contexts-zip', $contexts,
    '--prior-index', (Join-Path $fitIndex 'legalir.sqlite3'),
    '--complement-stage1', ([IO.Path]::ChangeExtension($complementStage1, '.audit.json')),
    '--complement-stage2', ([IO.Path]::ChangeExtension($complementStage2, '.audit.json')),
    '--complement-reference', $complementQueries,
    '--holdout-stage1', ([IO.Path]::ChangeExtension($holdoutStage1, '.audit.json')),
    '--holdout-stage2', ([IO.Path]::ChangeExtension($holdoutStage2, '.audit.json')),
    '--holdout-reference', $train,
    '--public-stage1', ([IO.Path]::ChangeExtension($publicStage1, '.audit.json')),
    '--public-stage2', ([IO.Path]::ChangeExtension($publicStage2, '.audit.json')),
    '--output-dir', $ltrDir,
    '--candidate-limit', '80',
    '--minimum-holdout-delta', '0.0'
)

$submission = Join-Path $output 'v11f.zip'
Invoke-PythonStep 'package v11f' @($submission) @(
    (Join-Path $scripts 'package_submission.py'),
    '--input', $ltrJson,
    '--output', $submission,
    '--expected-questions', '1000',
    '--expected-question-data', $publicQuestions
)

Write-Output "v11f end-to-end run finished: $submission"
