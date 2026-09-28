param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RtkArguments
)

$ErrorActionPreference = 'Stop'
$binary = Join-Path $PSScriptRoot '.tools\rtk-0.45.0\rtk.exe'
$state = Join-Path $PSScriptRoot '.rtk'
$tee = Join-Path $state 'tee'
$claudeState = Join-Path $state 'claude'

if (-not (Test-Path -LiteralPath $binary)) {
    throw "Project-local RTK binary is missing: $binary"
}

New-Item -ItemType Directory -Force -Path $state, $tee, $claudeState | Out-Null
$env:RTK_DB_PATH = Join-Path $state 'history.db'
$env:RTK_TEE_DIR = $tee
$env:RTK_CLAUDE_DIR = $claudeState
$env:CLAUDE_CONFIG_DIR = $claudeState
$env:RTK_TELEMETRY_DISABLED = '1'

& $binary @RtkArguments
exit $LASTEXITCODE
