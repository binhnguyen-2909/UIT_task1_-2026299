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
$prfOutput = Join-Path $output 'v10_9_prf'
$artifacts = Join-Path $PSScriptRoot 'artifacts'
$complementArtifacts = Join-Path $artifacts 'complement_split'
$registry = Join-Path $taskRoot 'model_registry.v10.example.json'
$reranker = Join-Path $repoRoot 'models\thanhtantran_Vietnamese_Reranker'
$train = Join-Path $raw 'train.json'
$publicQuestions = Join-Path $raw 'public-official.json'
$contexts = Join-Path $raw 'selected-contexts.zip'
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

function Get-PrfArguments {
    param(
        [string]$IndexPath,
        [string]$Stage1Audit,
        [AllowEmptyString()][string]$ReferencePath,
        [string]$ReportPath,
        [string]$SelectedAuditPath
    )
    $arguments = @(
        (Join-Path $scripts 'corpus_prf_stage1.py'),
        '--stage1-audit', $Stage1Audit,
        '--index-dir', $IndexPath,
        '--report', $ReportPath,
        '--selected-audit', $SelectedAuditPath,
        '--seed-docs', '4',
        '--seed-passages', '1',
        '--terms', '4',
        '--min-support', '2',
        '--anchor-query-term',
        '--prf-weights', '0.5',
        '--citation-weights', '0.5',
        '--rrf-k', '40',
        '--lock-top', '5',
        '--progress-every', '100'
    )
    if ($ReferencePath) {
        $arguments += @('--reference', $ReferencePath)
    }
    return $arguments
}

function Get-IncrementalStage2Arguments {
    param(
        [string]$IndexPath,
        [string]$Stage1Audit,
        [AllowEmptyString()][string]$ReferencePath,
        [string]$OutputPath,
        [string]$ReuseAuditPath
    )
    $arguments = @(
        (Join-Path $scripts 'batch_stage2_rerank.py'),
        '--stage1-audit', $Stage1Audit,
        '--index-dir', $IndexPath,
        '--model', $reranker,
        '--model-registry', $registry,
        '--output', $OutputPath,
        '--candidate-limit', '35',
        '--passages-per-doc', '3',
        '--max-length', '768',
        '--batch-size', [string]$RerankerBatchSize,
        '--device', $Device,
        '--fp16',
        '--neural-weight', '1.0',
        '--rrf-k', '10',
        '--reuse-audit', $ReuseAuditPath
    )
    if ($ReferencePath) {
        $arguments += @('--reference', $ReferencePath)
    }
    return $arguments
}

New-Item -ItemType Directory -Force -Path $output, $prfOutput | Out-Null

# V11g is a strict extension of the reproducible V11f pipeline.  This call is
# cache-aware; on a normal rerun all completed V11f stages are skipped.
$v11fRunner = Join-Path $PSScriptRoot 'run_v11f_end_to_end.ps1'
$v11fParameters = @{
    Device = $Device
    DenseBatchSize = $DenseBatchSize
    Stage1Workers = $Stage1Workers
    RerankerBatchSize = $RerankerBatchSize
}
if ($Force) {
    $v11fParameters['Force'] = $true
}
& $v11fRunner @v11fParameters
if (-not $?) {
    throw 'V11f prerequisite pipeline failed'
}

$baseHoldoutStage1 = Join-Path $output 'v11_holdout_stage1_chunk_only_top5.audit.json'
$basePublicStage1 = Join-Path $output 'v11_public_stage1_chunk_only_top5.audit.json'
$baseComplementStage1 = Join-Path $output 'v11_complement_stage1_chunk_only_top5.audit.json'
$baseHoldoutStage2 = Join-Path $output 'v11_holdout_stage2_c35_p3_top5.audit.json'
$basePublicStage2 = Join-Path $output 'v11_public_stage2_c35_p3_top5.audit.json'
$baseComplementStage2 = Join-Path $output 'v11_complement_stage2_c35_p3_top5.audit.json'

$complementPrfReport = Join-Path $prfOutput 'complement_safe_tail_report.json'
$complementPrfStage1 = Join-Path $prfOutput 'complement_safe_tail_stage1.audit.json'
$holdoutPrfReport = Join-Path $prfOutput 'holdout_safe_tail_locked_report.json'
$holdoutPrfStage1 = Join-Path $prfOutput 'holdout_safe_tail_locked_stage1.audit.json'
$publicPrfReport = Join-Path $prfOutput 'public_safe_tail_report.json'
$publicPrfStage1 = Join-Path $prfOutput 'public_safe_tail_stage1.audit.json'

Invoke-PythonStep 'complement fixed safe-tail PRF' @($complementPrfReport, $complementPrfStage1) `
    (Get-PrfArguments $complementIndex $baseComplementStage1 $complementQueries $complementPrfReport $complementPrfStage1)
Invoke-PythonStep 'locked holdout fixed safe-tail PRF' @($holdoutPrfReport, $holdoutPrfStage1) `
    (Get-PrfArguments $fitIndex $baseHoldoutStage1 $train $holdoutPrfReport $holdoutPrfStage1)
Invoke-PythonStep 'public fixed safe-tail PRF' @($publicPrfReport, $publicPrfStage1) `
    (Get-PrfArguments $publicIndex $basePublicStage1 '' $publicPrfReport $publicPrfStage1)

$complementPrfStage2 = Join-Path $prfOutput 'complement_safe_tail_stage2_c35_p3.zip'
$holdoutPrfStage2 = Join-Path $prfOutput 'holdout_safe_tail_stage2_c35_p3.zip'
$publicPrfStage2 = Join-Path $prfOutput 'public_safe_tail_stage2_c35_p3.zip'

Invoke-PythonStep 'complement incremental Stage-2' @(
    $complementPrfStage2, [IO.Path]::ChangeExtension($complementPrfStage2, '.audit.json')
) (Get-IncrementalStage2Arguments $complementIndex $complementPrfStage1 $complementQueries $complementPrfStage2 $baseComplementStage2)
Invoke-PythonStep 'locked holdout incremental Stage-2' @(
    $holdoutPrfStage2, [IO.Path]::ChangeExtension($holdoutPrfStage2, '.audit.json')
) (Get-IncrementalStage2Arguments $fitIndex $holdoutPrfStage1 $train $holdoutPrfStage2 $baseHoldoutStage2)
Invoke-PythonStep 'public incremental Stage-2' @(
    $publicPrfStage2, [IO.Path]::ChangeExtension($publicPrfStage2, '.audit.json')
) (Get-IncrementalStage2Arguments $publicIndex $publicPrfStage1 '' $publicPrfStage2 $basePublicStage2)

$ltrDir = Join-Path $output 'v11g_prf_ltr'
$ltrJson = Join-Path $ltrDir 'v11f_tiny_ltr_complement.json'
$ltrReport = Join-Path $ltrDir 'v11_tiny_ltr_complement_report.json'
Invoke-PythonStep 'V11g grouped complement LTR and locked holdout gate' @($ltrJson, $ltrReport) @(
    (Join-Path $scripts 'v11_tiny_ltr_complement.py'),
    '--contexts-zip', $contexts,
    '--prior-index', (Join-Path $fitIndex 'legalir.sqlite3'),
    '--complement-stage1', $complementPrfStage1,
    '--complement-stage2', ([IO.Path]::ChangeExtension($complementPrfStage2, '.audit.json')),
    '--complement-reference', $complementQueries,
    '--holdout-stage1', $holdoutPrfStage1,
    '--holdout-stage2', ([IO.Path]::ChangeExtension($holdoutPrfStage2, '.audit.json')),
    '--holdout-reference', $train,
    '--public-stage1', $publicPrfStage1,
    '--public-stage2', ([IO.Path]::ChangeExtension($publicPrfStage2, '.audit.json')),
    '--output-dir', $ltrDir,
    '--candidate-limit', '80',
    '--feature-set', 'v1',
    '--minimum-holdout-delta', '0.0'
)

$submission = Join-Path $output 'v11g.zip'
Invoke-PythonStep 'package V11g' @($submission) @(
    (Join-Path $scripts 'package_submission.py'),
    '--input', $ltrJson,
    '--output', $submission,
    '--expected-questions', '1000',
    '--expected-question-data', $publicQuestions
)

Write-Output "V11g end-to-end run finished: $submission"
