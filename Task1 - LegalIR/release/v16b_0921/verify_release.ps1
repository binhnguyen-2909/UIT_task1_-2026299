param([switch]$VerifyHashes)

$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$repoRoot = (Resolve-Path (Join-Path $taskRoot '..')).Path

$required = @(
    'data\raw\public_test\train.json',
    'data\raw\public_test\public-official.json',
    'data\raw\public_test\selected-contexts.zip',
    'data\processed\v11_research_fit_256\legalir.sqlite3',
    'data\processed\v11_research_fit_256\dense_embeddings.npy',
    'data\processed\v11_research_fit_256\question_dense_embeddings.npy',
    'data\processed\v11_research_256\legalir.sqlite3',
    'data\processed\v11_research_256\dense_embeddings.npy',
    'data\processed\v11_research_256\question_dense_embeddings.npy',
    'data\processed\v11_research_complement_256\legalir.sqlite3',
    'data\processed\v11_research_complement_256\dense_embeddings.npy',
    'experiments\track_b_v11_research\artifacts\holdout_1000.json',
    'experiments\track_b_v11_research\artifacts\train_fit_excluding_holdout.json',
    'experiments\track_b_v11_research\artifacts\complement_split\complement_queries_1200.json',
    'experiments\track_b_v11_research\run_v11h_end_to_end.ps1',
    'experiments\track_b_v11_research\run_v16a_memory_oracle.ps1',
    'experiments\track_b_v11_research\run_v16b_memory_slot5.ps1',
    'model_registry.v10.example.json',
    'output\v10_9_prf\complement_safe_tail_stage2_c35_p3.audit.json',
    'output\v10_9_prf\holdout_safe_tail_stage2_c35_p3.audit.json',
    'output\v10_9_prf\public_safe_tail_stage2_c35_p3.audit.json',
    'output\v11_feature_cache\v2_query_norm_c80.npz',
    'output\v11h_query_norm_ltr\v11h_query_normalized_ltr.json',
    'output\v16_rgm\complement_memory_features.json',
    'output\v16_rgm\holdout_memory_features.json',
    'output\v16_rgm\public_memory_features.json',
    'output\v16_rgm\holdout_v11h_locked_predictions.json',
    'output\v16_rgm\v16b_memory_ltr_report.json',
    'output\v16_rgm\v16b_memory_slot5.json',
    'output\v11h.zip',
    'output\v16b_memory_slot5.zip'
)
$requiredScripts = @(
    'main.py', 'materialize_v11_holdout.py', 'parallel_stage1_infer.py',
    'corpus_prf_stage1.py', 'batch_stage2_rerank.py',
    'prepare_v11_complement_split.py', 'v11_tiny_ltr_complement.py',
    'v11_tiny_ltr.py', 'v11_exact_heading_rescue.py', 'package_submission.py',
    'materialize_v11h_locked_predictions.py', 'v16_memory_oracle.py',
    'v16_memory_ltr.py', 'bm25_retriever.py', 'data_quality.py', 'evaluate.py',
    'legal_structure.py', 'reranker.py', 'retrieval_types.py', 'sqlite_retriever.py'
) | ForEach-Object { Join-Path 'scripts' $_ }
$required += $requiredScripts

$missing = @($required | Where-Object { -not (Test-Path -LiteralPath (Join-Path $taskRoot $_)) })
$rootRequired = @(
    (Join-Path $repoRoot 'models\bqbbao6_vietnamese-legal-embedding'),
    (Join-Path $repoRoot 'models\thanhtantran_Vietnamese_Reranker'),
    (Join-Path $repoRoot 'scripts\model_policy.py'),
    (Join-Path $repoRoot 'requirements-core.txt'),
    (Join-Path $repoRoot 'requirements-neural.txt')
)
$missing += @($rootRequired | Where-Object { -not (Test-Path -LiteralPath $_) })
if ($missing.Count -gt 0) {
    $missing | ForEach-Object { Write-Error "Missing release input: $_" }
    throw "V16b release verification failed: $($missing.Count) required path(s) are absent."
}

if ($VerifyHashes) {
    $expectedHashes = @{
        'output\v16b_memory_slot5.zip' = 'B2816BD73AE23FBDFADD6563DC43728896F078A6AD8032B6743984B3AADADFD9'
        'output\v16_rgm\v16b_memory_slot5.json' = 'C9EFE92A62B11EFEFEA2F91EBD112A336018F4C66C910FE5A8FCFE4234B91ACC'
        'output\v16_rgm\v16b_memory_ltr_report.json' = 'B608A7F441619A0E035D4A5F282CF6F1F8F5667D24945E3B702656D0CB610089'
        'output\v11h.zip' = '57CDEF271C86824B128482AD141E140224E8002B7ACC7536CB79DAE90EB7F175'
        'output\v11h_query_norm_ltr\v11h_query_normalized_ltr.json' = 'BFF7CF7F2FFB9D570CC6DE72A59831189B9D440607147EFB668330ED834FE1E8'
        'output\v11_feature_cache\v2_query_norm_c80.npz' = '2C527D36EABEEDA800232C5E6A93F935E46C6B9175BED25BF4A755ABECE664EB'
    }
    foreach ($relativePath in $expectedHashes.Keys) {
        $actual = (Get-FileHash -LiteralPath (Join-Path $taskRoot $relativePath) -Algorithm SHA256).Hash
        if ($actual -ne $expectedHashes[$relativePath]) {
            throw "Hash mismatch for $relativePath. Expected $($expectedHashes[$relativePath]), got $actual"
        }
    }
}

Write-Output "V16b 0.9210 release verified ($($required.Count) task paths, $($rootRequired.Count) workspace paths)."
