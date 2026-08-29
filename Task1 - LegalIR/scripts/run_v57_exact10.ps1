param()

$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$pipeline = Join-Path $project 'output\current\v57_exact10'
$cache = Join-Path $pipeline 'cache'
$rebuild = Join-Path $pipeline 'rebuild'
$release = Join-Path $project 'output\release'

& py -3 (Join-Path $PSScriptRoot 'benchmark_exact_heading_rescue.py') `
    --fit-cache (Join-Path $cache 'fit_rank_cache.jsonl') `
    --holdout-cache (Join-Path $cache 'holdout_rank_cache.jsonl') `
    --public-cache (Join-Path $cache 'public_rank_cache.jsonl') `
    --public-anchor (Join-Path $pipeline 'parent_legal2.json') `
    --contexts-zip (Join-Path $project 'data\raw\public_test\selected-contexts.zip') `
    --output-dir $rebuild `
    --candidate-cutoff 12 `
    --conservative-ngram 10 `
    --balanced-ngram 7

& py -3 (Join-Path $PSScriptRoot 'package_submission.py') `
    --input (Join-Path $rebuild 'sub_v49_legal2_exact_conservative.json') `
    --output (Join-Path $release 'sub_v57_exact10.zip') `
    --expected-questions 1000 `
    --expected-question-data (Join-Path $project 'data\raw\public_test\public-official.json')

Copy-Item -LiteralPath (Join-Path $rebuild 'sub_v49_legal2_exact_conservative.json') `
    -Destination (Join-Path $pipeline 'baseline_v57_exact10.json') -Force
Copy-Item -LiteralPath (Join-Path $rebuild 'sub_v49_legal2_exact_conservative.audit.json') `
    -Destination (Join-Path $pipeline 'baseline_v57_exact10.audit.json') -Force
Copy-Item -LiteralPath (Join-Path $pipeline 'baseline_v57_exact10.json') `
    -Destination (Join-Path $release 'sub_v57_exact10.json') -Force
Copy-Item -LiteralPath (Join-Path $pipeline 'baseline_v57_exact10.audit.json') `
    -Destination (Join-Path $release 'sub_v57_exact10.audit.json') -Force

Write-Output ("v57 exact10 rebuilt: " + (Join-Path $release 'sub_v57_exact10.zip'))
