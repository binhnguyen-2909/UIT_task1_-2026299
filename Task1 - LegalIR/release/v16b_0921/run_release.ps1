param(
    [ValidateSet('cuda', 'cpu')]
    [string]$Device = 'cuda',
    [int]$DenseBatchSize = 32,
    [int]$Stage1Workers = 8,
    [int]$RerankerBatchSize = 16,
    [int]$MemoryBatchSize = 64,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$repoRoot = (Resolve-Path (Join-Path $taskRoot '..')).Path
$runnerRoot = Join-Path $taskRoot 'experiments\track_b_v11_research'
$outputRoot = Join-Path $taskRoot 'output'
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

function Invoke-CheckedRunner {
    param([string]$Name, [string]$Path, [object[]]$Arguments)
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing $Name runner: $Path"
    }
    Write-Output "RUN $Name"
    & $Path @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

$v11hArguments = @(
    '-Device', $Device,
    '-DenseBatchSize', $DenseBatchSize,
    '-Stage1Workers', $Stage1Workers,
    '-RerankerBatchSize', $RerankerBatchSize
)
if ($Force) { $v11hArguments += '-Force' }
Invoke-CheckedRunner 'V11h base pipeline' (Join-Path $runnerRoot 'run_v11h_end_to_end.ps1') $v11hArguments

foreach ($split in @('complement', 'holdout', 'public')) {
    $memoryArguments = @('-Split', $split, '-Device', $Device, '-BatchSize', $MemoryBatchSize)
    if ($Force) { $memoryArguments += '-Force' }
    Invoke-CheckedRunner "V16a memory features ($split)" (Join-Path $runnerRoot 'run_v16a_memory_oracle.ps1') $memoryArguments
}

$v16bArguments = @('-Device', $Device, '-BatchSize', $MemoryBatchSize)
if ($Force) { $v16bArguments += '-Force' }
Invoke-CheckedRunner 'V16b slot-5 gate' (Join-Path $runnerRoot 'run_v16b_memory_slot5.ps1') $v16bArguments

$submissionJson = Join-Path $outputRoot 'v16_rgm\v16b_memory_slot5.json'
$submissionZip = Join-Path $outputRoot 'v16b_memory_slot5.zip'
$questions = Join-Path $taskRoot 'data\raw\public_test\public-official.json'
& $python (Join-Path $taskRoot 'scripts\package_submission.py') `
    '--input' $submissionJson `
    '--output' $submissionZip `
    '--expected-questions' '1000' `
    '--expected-question-data' $questions
if ($LASTEXITCODE -ne 0) {
    throw "V16b packaging failed with exit code $LASTEXITCODE"
}

Write-Output "V16b release submission: $submissionZip"
