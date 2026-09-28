[CmdletBinding()]
param(
    [ValidateSet("probe", "stress")]
    [string]$Mode = "probe",
    [string]$Python = "python",
    [switch]$Fresh
)

# RTX 5070 has 12GB VRAM.  This runner keeps the registered E2 objective but
# uses the smallest executable batches.  Run the probe before a stress job;
# it exercises a complete AdamW step and fails closed on CUDA OOM.
$ErrorActionPreference = "Stop"
$TaskRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$RepoRoot = Split-Path -Parent $TaskRoot
$OutputDir = Join-Path $TaskRoot "output\e2_mil_oof"

$env:OMP_NUM_THREADS = if ($env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS } else { "8" }
$env:MKL_NUM_THREADS = if ($env:MKL_NUM_THREADS) { $env:MKL_NUM_THREADS } else { "8" }
$env:OPENBLAS_NUM_THREADS = if ($env:OPENBLAS_NUM_THREADS) { $env:OPENBLAS_NUM_THREADS } else { "8" }
$env:PYTHONIOENCODING = "utf-8"
$env:TOKENIZERS_PARALLELISM = "false"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$Base = @(
    (Join-Path $TaskRoot "scripts\e2_mil_finetune_oof.py"),
    "--stage1", (Join-Path $TaskRoot "output\v10_9_prf\complement_safe_tail_stage1.audit.json"),
    "--stage2", (Join-Path $TaskRoot "output\v10_9_prf\complement_safe_tail_stage2_c35_p3.audit.json"),
    "--anchor", (Join-Path $TaskRoot "output\v16_rgm\complement_v16b_memory_slot5.json"),
    "--reference", (Join-Path $TaskRoot "experiments\track_b_v11_research\artifacts\complement_split\complement_queries_1200.json"),
    "--e1-manifest", (Join-Path $TaskRoot "output\e1_evidence_frozen_ce\e1_complement_evidence.jsonl"),
    "--e1-manifest-meta", (Join-Path $TaskRoot "output\e1_evidence_frozen_ce\e1_complement_evidence.meta.json"),
    "--e1-frozen-scores", (Join-Path $TaskRoot "output\e1_evidence_frozen_ce\e1_complement_frozen_ce_scores.jsonl"),
    "--reranker-model", (Join-Path $RepoRoot "models\thanhtantran_Vietnamese_Reranker"),
    "--output-dir", $OutputDir,
    "--device", "cuda",
    "--micro-batch", "1",
    "--gradient-accumulation", "16",
    "--score-batch-size", "1",
    "--gradient-checkpointing",
    "--force"
)

if ($Mode -eq "probe") {
    & $Python @Base "--memory-probe"
    exit $LASTEXITCODE
}

$ResumeFlag = if ($Fresh) { "--no-resume" } else { "--resume" }
& $Python @Base `
    "--regimes" "law_source_family" "query_template_family" "near_duplicate_family" `
    "--checkpoint-path" (Join-Path $OutputDir "e2_mil_oof_stress_progress.json") `
    "--report-path" (Join-Path $OutputDir "e2_mil_oof_stress_report.json") `
    $ResumeFlag
exit $LASTEXITCODE
