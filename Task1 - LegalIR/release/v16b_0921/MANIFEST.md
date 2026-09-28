# Retained V16b/V11h manifest

The items below must stay available to recreate and verify the 0.9210 submission. All paths are relative to the workspace root.

## Frozen delivery and audit evidence

| Path | SHA-256 | Purpose |
|---|---|---|
| `Task1 - LegalIR/output/v16b_memory_slot5.zip` | `B2816BD73AE23FBDFADD6563DC43728896F078A6AD8032B6743984B3AADADFD9` | Frozen, eligible submission (0.9210 public). |
| `Task1 - LegalIR/output/v16_rgm/v16b_memory_slot5.json` | `C9EFE92A62B11EFEFEA2F91EBD112A336018F4C66C910FE5A8FCFE4234B91ACC` | Unpacked V16b submission. |
| `Task1 - LegalIR/output/v16_rgm/v16b_memory_ltr_report.json` | `B608A7F441619A0E035D4A5F282CF6F1F8F5667D24945E3B702656D0CB610089` | Promotion report and policy. |
| `Task1 - LegalIR/output/v11h.zip` | `57CDEF271C86824B128482AD141E140224E8002B7ACC7536CB79DAE90EB7F175` | V16b anchor submission. |
| `Task1 - LegalIR/output/v11h_query_norm_ltr/v11h_query_normalized_ltr.json` | `BFF7CF7F2FFB9D570CC6DE72A59831189B9D440607147EFB668330ED834FE1E8` | V11h public anchor. |
| `Task1 - LegalIR/output/v11_feature_cache/v2_query_norm_c80.npz` | `2C527D36EABEEDA800232C5E6A93F935E46C6B9175BED25BF4A755ABECE664EB` | V11h query-normalized feature cache. |

Keep every file under these generated-artifact directories, not just the listed files: `output/v10_9_prf`, `output/v11_feature_cache`, `output/v11_tiny_ltr_complement_no_prior`, `output/v11g_prf_ltr`, `output/v11h_query_norm_ltr`, and `output/v16_rgm`.

## Inputs, indexes, and models

- `Task1 - LegalIR/data/raw/public_test/{train.json,public-official.json,selected-contexts.zip}`
- `Task1 - LegalIR/data/processed/v11_research_fit_256/`
- `Task1 - LegalIR/data/processed/v11_research_256/`
- `Task1 - LegalIR/data/processed/v11_research_complement_256/`
- `Task1 - LegalIR/experiments/track_b_v11_research/artifacts/`, including `holdout_1000.json`, `train_fit_excluding_holdout.json`, and `complement_split/`.
- `models/bqbbao6_vietnamese-legal-embedding/`
- `models/thanhtantran_Vietnamese_Reranker/`
- `Task1 - LegalIR/model_registry.v10.example.json`, `scripts/model_policy.py`, `requirements-core.txt`, and `requirements-neural.txt`.

## Source and runners

The execution entrypoints are:

- `Task1 - LegalIR/experiments/track_b_v11_research/run_v11h_end_to_end.ps1`
- `Task1 - LegalIR/experiments/track_b_v11_research/run_v16a_memory_oracle.ps1`
- `Task1 - LegalIR/experiments/track_b_v11_research/run_v16b_memory_slot5.ps1`
- `Task1 - LegalIR/release/v16b_0921/run_release.ps1`

Keep the following source modules alongside them: `main.py`, `materialize_v11_holdout.py`, `parallel_stage1_infer.py`, `corpus_prf_stage1.py`, `batch_stage2_rerank.py`, `prepare_v11_complement_split.py`, `v11_tiny_ltr_complement.py`, `v11_tiny_ltr.py`, `v11_exact_heading_rescue.py`, `package_submission.py`, `materialize_v11h_locked_predictions.py`, `v16_memory_oracle.py`, `v16_memory_ltr.py`, `bm25_retriever.py`, `data_quality.py`, `evaluate.py`, `legal_structure.py`, `reranker.py`, `retrieval_types.py`, and `sqlite_retriever.py` in `Task1 - LegalIR/scripts/`.

`run_release.ps1` invokes the original V11h runner, refreshes all three V16a memory feature splits, runs V16b, then packages the JSON. This gives a single end-to-end command without duplicating the working pipeline.

## Non-negotiable contract

- No manual labels or public-label reads.
- V16b may only intervene in slot 5; top 4 and candidate identity stay locked.
- Memory features are fit-only and exclude normalized-question overlap.
- Do not change candidate pool, parser, or fallback behavior under the release name. Such a change is a new experiment, not a V16b rebuild.
