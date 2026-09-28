param(
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

$memoryRunner = Join-Path $PSScriptRoot 'run_v16a_memory_oracle.ps1'
$memoryScript = Join-Path $taskRoot 'scripts\v16_memory_ltr.py'
$anchorScript = Join-Path $taskRoot 'scripts\materialize_v11h_locked_predictions.py'
$raw = Join-Path $taskRoot 'data\raw\public_test'
$artifacts = Join-Path $PSScriptRoot 'artifacts'
$outputDir = Join-Path $taskRoot 'output\v16_rgm'
$prfOutput = Join-Path $taskRoot 'output\v10_9_prf'
$processed = Join-Path $taskRoot 'data\processed'
$contexts = Join-Path $raw 'selected-contexts.zip'
$fitIndex = Join-Path $processed 'v11_research_fit_256'
$v11hFeatureCache = Join-Path $taskRoot 'output\v11_feature_cache\v2_query_norm_c80.npz'
$holdoutAnchor = Join-Path $outputDir 'holdout_v11h_locked_predictions.json'
$publicAnchor = Join-Path $taskRoot 'output\v11h_query_norm_ltr\v11h_query_normalized_ltr.json'
$referenceHoldout = Join-Path $artifacts 'holdout_1000.json'
$referenceComplement = Join-Path $artifacts 'complement_split\complement_queries_1200.json'

$budget = 1090010000
$limit = 4000000000
if ($budget -ge $limit) {
    throw "V16b model budget violates the 4B limit: $budget >= $limit"
}

$featurePaths = @(
    (Join-Path $outputDir 'complement_memory_features.json'),
    (Join-Path $outputDir 'holdout_memory_features.json'),
    (Join-Path $outputDir 'public_memory_features.json')
)
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
if (-not (Test-Path -LiteralPath $featurePaths[2]) -or $Force) {
    Write-Output 'Public V16a memory features are missing; building them without public labels.'
    & $memoryRunner -Split public -Device $Device -BatchSize $BatchSize
    if ($LASTEXITCODE -ne 0) {
        throw "V16a public feature build failed with exit code $LASTEXITCODE"
    }
}
foreach ($path in $featurePaths) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required memory feature artifact is missing: $path"
    }
}

if (-not (Test-Path -LiteralPath $holdoutAnchor) -or $Force) {
    Write-Output 'Materializing the locked v11h holdout anchor.'
    & $python $anchorScript `
        '--complement-stage1' (Join-Path $prfOutput 'complement_safe_tail_stage1.audit.json') `
        '--complement-stage2' (Join-Path $prfOutput 'complement_safe_tail_stage2_c35_p3.audit.json') `
        '--complement-reference' $referenceComplement `
        '--holdout-stage1' (Join-Path $prfOutput 'holdout_safe_tail_locked_stage1.audit.json') `
        '--holdout-stage2' (Join-Path $prfOutput 'holdout_safe_tail_stage2_c35_p3.audit.json') `
        '--holdout-reference' $referenceHoldout `
        '--feature-cache' $v11hFeatureCache `
        '--output' $holdoutAnchor
    if ($LASTEXITCODE -ne 0) {
        throw "v11h holdout anchor materialization failed with exit code $LASTEXITCODE"
    }
}

$arguments = @(
    $memoryScript,
    '--contexts-zip', $contexts,
    '--prior-index', (Join-Path $fitIndex 'legalir.sqlite3'),
    '--complement-stage1', (Join-Path $prfOutput 'complement_safe_tail_stage1.audit.json'),
    '--complement-stage2', (Join-Path $prfOutput 'complement_safe_tail_stage2_c35_p3.audit.json'),
    '--complement-reference', $referenceComplement,
    '--complement-memory-features', $featurePaths[0],
    '--holdout-stage1', (Join-Path $prfOutput 'holdout_safe_tail_locked_stage1.audit.json'),
    '--holdout-stage2', (Join-Path $prfOutput 'holdout_safe_tail_stage2_c35_p3.audit.json'),
    '--holdout-reference', $referenceHoldout,
    '--holdout-memory-features', $featurePaths[1],
    '--holdout-anchor-json', $holdoutAnchor,
    '--public-stage1', (Join-Path $prfOutput 'public_safe_tail_stage1.audit.json'),
    '--public-stage2', (Join-Path $prfOutput 'public_safe_tail_stage2_c35_p3.audit.json'),
    '--public-memory-features', $featurePaths[2],
    '--public-anchor-json', $publicAnchor,
    '--output-dir', $outputDir,
    '--candidate-limit', '35',
    '--model-profile', 'pointwise-v2-winner',
    '--minimum-holdout-delta', '0.0',
    '--submission-name', 'v16b_memory_slot5.json'
)

foreach ($path in @($contexts, (Join-Path $fitIndex 'legalir.sqlite3'), $referenceHoldout, $referenceComplement, $v11hFeatureCache, $publicAnchor, $holdoutAnchor)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required V16b input is missing: $path"
    }
}

$env:PYTHONIOENCODING = 'utf-8'
$env:TOKENIZERS_PARALLELISM = 'false'
& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "V16b memory slot-5 experiment failed with exit code $LASTEXITCODE"
}

Write-Output "V16b report: $(Join-Path $outputDir 'v16b_memory_ltr_report.json')"
Write-Output "V16b candidate contract: c35 only; model budget: $budget / $limit"
