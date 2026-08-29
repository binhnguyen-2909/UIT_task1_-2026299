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
$contexts = Join-Path $raw 'selected-contexts.zip'
$train = Join-Path $raw 'train.json'
$publicQuestions = Join-Path $raw 'public-official.json'
$complementQueries = Join-Path $artifacts 'complement_split\complement_queries_1200.json'
$fitIndex = Join-Path $processed 'v11_research_fit_256'

# The complete pipeline uses one 330M embedding model, one 760M reranker and
# a negligible logistic selector: approximately 1.09B parameters, strictly
# below the BTC 4B limit. The three stages below are execution stages in this
# one file, not three simultaneously counted model lineages.
$modelParameterBudget = 1090000000
$modelParameterLimit = 4000000000
if ($modelParameterBudget -ge $modelParameterLimit) {
    throw 'Configured end-to-end model budget violates the BTC 4B limit.'
}

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
        '--embedding-model', (Join-Path $repoRoot 'models\bqbbao6_vietnamese-legal-embedding'),
        '--candidates', '200',
        '--dense-weight', '0.30',
        '--dense-rrf-k', '20',
        '--dense-max-seq-length', '256',
        '--question-dense-weight', '0.05',
        '--question-dense-rrf-k', '20',
        '--question-dense-candidates', '20',
        '--model-registry', (Join-Path $taskRoot 'model_registry.v10.example.json'),
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
        '--model-registry', (Join-Path $taskRoot 'model_registry.v10.example.json'),
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
        '--model', (Join-Path $repoRoot 'models\thanhtantran_Vietnamese_Reranker'),
        '--model-registry', (Join-Path $taskRoot 'model_registry.v10.example.json'),
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
        '--model', (Join-Path $repoRoot 'models\thanhtantran_Vietnamese_Reranker'),
        '--model-registry', (Join-Path $taskRoot 'model_registry.v10.example.json'),
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

New-Item -ItemType Directory -Force -Path $output, $prfOutput, $artifacts, $complementArtifacts | Out-Null

# This is the single-file v11h runner. F, G and H remain explicit internal
# stages so the research lineage is readable, but no external runner is called.
# ---------------------------------------------------------------------------
# Internal Stage F: compliant baseline index + hybrid retrieval + reranking.
# ---------------------------------------------------------------------------
Write-Output '=== Internal Stage F: compliant baseline ==='
$fFitTraining = Join-Path $artifacts 'train_fit_excluding_holdout.json'
$fHoldout = Join-Path $artifacts 'holdout_1000.json'
Invoke-PythonStep 'materialize holdout' @($fHoldout) @(
    (Join-Path $scripts 'materialize_v11_holdout.py'),
    '--train', $train,
    '--fit', $fFitTraining,
    '--output', $fHoldout,
    '--expected-questions', '1000'
)

Invoke-PythonStep 'build fit index' @(
    (Join-Path $fitIndex 'legalir.sqlite3'),
    (Join-Path $fitIndex 'dense_embeddings.npy')
) (Get-IndexArguments $fitIndex $fFitTraining)

$publicIndex = Join-Path $processed 'v11_research_256'
$complementIndex = Join-Path $processed 'v11_research_complement_256'
Invoke-PythonStep 'build public index' @(
    (Join-Path $publicIndex 'legalir.sqlite3'),
    (Join-Path $publicIndex 'dense_embeddings.npy')
) (Get-IndexArguments $publicIndex $train)

$fHoldoutStage1 = Join-Path $output 'v11_holdout_stage1_chunk_only_top5.zip'
$fPublicStage1 = Join-Path $output 'v11_public_stage1_chunk_only_top5.zip'
Invoke-PythonStep 'holdout Stage-1' @($fHoldoutStage1, [IO.Path]::ChangeExtension($fHoldoutStage1, '.audit.json')) `
    (Get-Stage1Arguments $fitIndex $fHoldout $fHoldout $fHoldoutStage1)
Invoke-PythonStep 'public Stage-1' @($fPublicStage1, [IO.Path]::ChangeExtension($fPublicStage1, '.audit.json')) `
    (Get-Stage1Arguments $publicIndex $publicQuestions $publicQuestions $fPublicStage1)

$fHoldoutStage2 = Join-Path $output 'v11_holdout_stage2_c35_p3_top5.zip'
$fPublicStage2 = Join-Path $output 'v11_public_stage2_c35_p3_top5.zip'
Invoke-PythonStep 'holdout Stage-2 c35/p3' @($fHoldoutStage2, [IO.Path]::ChangeExtension($fHoldoutStage2, '.audit.json')) `
    (Get-Stage2Arguments $fitIndex ([IO.Path]::ChangeExtension($fHoldoutStage1, '.audit.json')) $fHoldout $fHoldoutStage2)
Invoke-PythonStep 'public Stage-2 c35/p3' @($fPublicStage2, [IO.Path]::ChangeExtension($fPublicStage1, '.audit.json')) `
    (Get-Stage2Arguments $publicIndex ([IO.Path]::ChangeExtension($fPublicStage1, '.audit.json')) $publicQuestions $fPublicStage2)

$fComplementReport = Join-Path $complementArtifacts 'complement_split_report.json'
$complementQueries = Join-Path $complementArtifacts 'complement_queries_1200.json'
if (-not (Test-Path -LiteralPath (Join-Path $complementIndex 'legalir.sqlite3'))) {
    Invoke-PythonStep 'prepare complement index/split' @(
        (Join-Path $complementIndex 'legalir.sqlite3'), $complementQueries, $fComplementReport
    ) @(
        (Join-Path $scripts 'prepare_v11_complement_split.py'),
        '--source-index', $fitIndex,
        '--output-index', $complementIndex,
        '--train-json', $train,
        '--holdout-audit', ([IO.Path]::ChangeExtension($fHoldoutStage1, '.audit.json')),
        '--query-count', '1200',
        '--artifact-dir', $complementArtifacts
    )
} else {
    Write-Output 'SKIP prepare complement index/split (artifacts exist)'
}

$fComplementStage1 = Join-Path $output 'v11_complement_stage1_chunk_only_top5.zip'
Invoke-PythonStep 'complement Stage-1' @(
    $fComplementStage1, [IO.Path]::ChangeExtension($fComplementStage1, '.audit.json')
) (Get-Stage1Arguments $complementIndex $complementQueries $complementQueries $fComplementStage1)

$fComplementStage2 = Join-Path $output 'v11_complement_stage2_c35_p3_top5.zip'
Invoke-PythonStep 'complement Stage-2 c35/p3' @(
    $fComplementStage2, [IO.Path]::ChangeExtension($fComplementStage2, '.audit.json')
) (Get-Stage2Arguments $complementIndex ([IO.Path]::ChangeExtension($fComplementStage1, '.audit.json')) $complementQueries $fComplementStage2)

$fLtrDir = Join-Path $output 'v11_tiny_ltr_complement_no_prior'
$fLtrJson = Join-Path $fLtrDir 'v11f_tiny_ltr_complement.json'
$fLtrReport = Join-Path $fLtrDir 'v11_tiny_ltr_complement_report.json'
Invoke-PythonStep 'grouped complement LTR and locked holdout gate' @($fLtrJson, $fLtrReport) @(
    (Join-Path $scripts 'v11_tiny_ltr_complement.py'),
    '--contexts-zip', $contexts,
    '--prior-index', (Join-Path $fitIndex 'legalir.sqlite3'),
    '--complement-stage1', ([IO.Path]::ChangeExtension($fComplementStage1, '.audit.json')),
    '--complement-stage2', ([IO.Path]::ChangeExtension($fComplementStage2, '.audit.json')),
    '--complement-reference', $complementQueries,
    '--holdout-stage1', ([IO.Path]::ChangeExtension($fHoldoutStage1, '.audit.json')),
    '--holdout-stage2', ([IO.Path]::ChangeExtension($fHoldoutStage2, '.audit.json')),
    '--holdout-reference', $train,
    '--public-stage1', ([IO.Path]::ChangeExtension($fPublicStage1, '.audit.json')),
    '--public-stage2', ([IO.Path]::ChangeExtension($fPublicStage2, '.audit.json')),
    '--output-dir', $fLtrDir,
    '--candidate-limit', '80',
    '--minimum-holdout-delta', '0.0'
)

$v11fSubmission = Join-Path $output 'v11f.zip'
Invoke-PythonStep 'package v11f checkpoint' @($v11fSubmission) @(
    (Join-Path $scripts 'package_submission.py'),
    '--input', $fLtrJson,
    '--output', $v11fSubmission,
    '--expected-questions', '1000',
    '--expected-question-data', $publicQuestions
)

# ---------------------------------------------------------------------------
# Internal Stage G: safe-tail PRF/citation completion and grouped LTR v1.
# ---------------------------------------------------------------------------
Write-Output '=== Internal Stage G: safe-tail PRF + LTR v1 ==='
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
New-Item -ItemType Directory -Force -Path $prfOutput | Out-Null

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

$gLtrDir = Join-Path $output 'v11g_prf_ltr'
$gLtrJson = Join-Path $gLtrDir 'v11f_tiny_ltr_complement.json'
$gLtrReport = Join-Path $gLtrDir 'v11_tiny_ltr_complement_report.json'
Invoke-PythonStep 'V11g grouped complement LTR and locked holdout gate' @($gLtrJson, $gLtrReport) @(
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
    '--output-dir', $gLtrDir,
    '--candidate-limit', '80',
    '--feature-set', 'v1',
    '--minimum-holdout-delta', '0.0'
)

$v11gSubmission = Join-Path $output 'v11g.zip'
Invoke-PythonStep 'package v11g checkpoint' @($v11gSubmission) @(
    (Join-Path $scripts 'package_submission.py'),
    '--input', $gLtrJson,
    '--output', $v11gSubmission,
    '--expected-questions', '1000',
    '--expected-question-data', $publicQuestions
)

$complementStage1 = Join-Path $prfOutput 'complement_safe_tail_stage1.audit.json'
$holdoutStage1 = Join-Path $prfOutput 'holdout_safe_tail_locked_stage1.audit.json'
$publicStage1 = Join-Path $prfOutput 'public_safe_tail_stage1.audit.json'
$complementStage2 = Join-Path $prfOutput 'complement_safe_tail_stage2_c35_p3.audit.json'
$holdoutStage2 = Join-Path $prfOutput 'holdout_safe_tail_stage2_c35_p3.audit.json'
$publicStage2 = Join-Path $prfOutput 'public_safe_tail_stage2_c35_p3.audit.json'

$ltrDir = Join-Path $output 'v11h_query_norm_ltr'
$featureCacheDir = Join-Path $output 'v11_feature_cache'
$featureCache = Join-Path $featureCacheDir 'v2_query_norm_c80.npz'
$ltrJson = Join-Path $ltrDir 'v11h_query_normalized_ltr.json'
$ltrReport = Join-Path $ltrDir 'v11_tiny_ltr_complement_report.json'
New-Item -ItemType Directory -Force -Path $ltrDir, $featureCacheDir | Out-Null

Invoke-PythonStep 'V11h query-normalized grouped OOF LTR and locked holdout gate' @(
    $ltrJson, $ltrReport, $featureCache
) @(
    (Join-Path $scripts 'v11_tiny_ltr_complement.py'),
    '--contexts-zip', $contexts,
    '--prior-index', (Join-Path $fitIndex 'legalir.sqlite3'),
    '--complement-stage1', $complementStage1,
    '--complement-stage2', $complementStage2,
    '--complement-reference', $complementQueries,
    '--holdout-stage1', $holdoutStage1,
    '--holdout-stage2', $holdoutStage2,
    '--holdout-reference', $train,
    '--public-stage1', $publicStage1,
    '--public-stage2', $publicStage2,
    '--output-dir', $ltrDir,
    '--candidate-limit', '80',
    '--feature-set', 'v2-query-normalized',
    '--model-profile', 'pointwise-v2-winner',
    '--submission-name', 'v11h_query_normalized_ltr.json',
    '--feature-cache', $featureCache,
    '--minimum-holdout-delta', '0.0'
)

$submission = Join-Path $output 'v11h.zip'
Invoke-PythonStep 'package V11h' @($submission) @(
    (Join-Path $scripts 'package_submission.py'),
    '--input', $ltrJson,
    '--output', $submission,
    '--expected-questions', '1000',
    '--expected-question-data', $publicQuestions
)

Write-Output "V11h single-file end-to-end run finished: $submission"
Write-Output 'Model budget: 1.09B / 4B (compliant)'
