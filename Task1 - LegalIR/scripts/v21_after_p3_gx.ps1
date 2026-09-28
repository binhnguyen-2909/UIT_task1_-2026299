param(
    [Parameter(Mandatory = $true)]
    [int]$P3ProcessId,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$workspace = Split-Path -Parent $taskRoot
$archiveRoot = Join-Path $taskRoot 'archive\2026-08-31_retired_experiments'
$outputRoot = Join-Path $taskRoot 'output\v21_p3_gx'
$p3ReportPath = Join-Path $outputRoot 'v21_p3_gx_report.json'

Wait-Process -Id $P3ProcessId

if (-not (Test-Path -LiteralPath $p3ReportPath -PathType Leaf)) {
    throw "P3-GX exited without its report: $p3ReportPath"
}
$p3Report = Get-Content -Raw -LiteralPath $p3ReportPath | ConvertFrom-Json
$regimes = @('standard_grouped', 'law_family_ood', 'query_template_ood', 'near_duplicate_ood')
foreach ($regime in $regimes) {
    $scorePath = [string]$p3Report.regimes.$regime.score_path
    if ([string]::IsNullOrWhiteSpace($scorePath) -or -not (Test-Path -LiteralPath $scorePath -PathType Leaf)) {
        throw "P3-GX did not produce a complete OOF score matrix for $regime"
    }
}

& $PythonExe (Join-Path $PSScriptRoot 'v21_winner_policy.py') `
    --metadata (Join-Path $archiveRoot 'Task1-output\v18b_evidence_packs_corrected_r3\v18b_metadata.json') `
    --block-features (Join-Path $archiveRoot 'Task1-output\v19_cater_block_features_r1\v19_block_features.jsonl') `
    --pack-scores (Join-Path $archiveRoot 'Task1-output\v18b_evidence_packs_corrected_r3\v18b_pack_scores.jsonl') `
    --fold-manifest (Join-Path $outputRoot 'fold_manifest.json') `
    --p3-report $p3ReportPath `
    --legacy-cater (Join-Path $archiveRoot 'Task1-scripts\v20_cater_r.py') `
    --output (Join-Path $outputRoot 'v21_winner_policy_report.json') `
    --force
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Set-Location -LiteralPath $workspace
