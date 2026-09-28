# V16-RGM — QREL memory and c35 slot-5 checkpoints

## Scope

This checkpoint implements the first gate from the V16-RGM proposal:

```text
fit-only labelled questions
        -> BQB nearest historical questions
        -> reciprocal/degree-corrected QREL propagation
        -> memory document features
        -> v11h c35 anchor union memory@5
        -> candidate oracle audit
```

The V16a oracle did not change the v11h submission. The promoted V16b gate
adds only fit-only memory features to the existing c35 selector path; no graph
teacher, LoRA, candidate expansion, or holdout-driven tuning is included.

## Implementation

`scripts/v16_memory_oracle.py` provides:

- exact normalized-question and question-ID split isolation checks;
- duplicate-collapsed training memory compatible with v11h;
- BQB top-`K` historical-query lookup;
- reciprocal-neighbor gating with a configurable confidence floor;
- document-degree-corrected QREL propagation;
- `prototype_max`, `prototype_top3_mean`, vote, entropy, agreement, rank, and
  anchor-margin features;
- conservative anchor-preserving candidate fusion;
- JSON feature output for a later selector experiment;
- oracle metrics for the incremental memory channel.

`experiments/track_b_v11_research/run_v16a_memory_oracle.ps1` reproduces the
holdout or complementary run with the existing local BQB model and caches.

## Results

The runs used `memory_k=50`, temperature `0.07`, degree exponent `0.5`,
reciprocal floor `0.25`, anchor cutoff `35`, and memory top-`5`.

| Split | Anchor oracle@35 | Anchor + memory@5 | Delta | Gold recovered outside c35 |
|---|---:|---:|---:|---:|
| Complement, 1,200 queries | 0.950097 | 0.955097 | **+0.005000** | 7 |
| Locked holdout, 1,000 queries | 0.967667 | 0.969167 | **+0.001500** | 4 |

The historical split contained normalized duplicate questions across the two
partitions. The strict run removed 3 duplicate memory rows for the complement
experiment and 5 for the holdout experiment before retrieval; labels from the
evaluated queries were used only for oracle measurement. The default behavior
of the tool remains to reject such overlap.

## V16a gate decision

The proposed promotion gate was `Delta Oracle >= 0.01`. Neither split reaches
it, so graph distillation is **not opened**. The memory channel is a real but
small signal and remains useful as an auditable feature artifact for future
work; it is not promoted directly into v11h.

Generated local artifacts:

- `output/v16_rgm/complement_memory_oracle_report.json`;
- `output/v16_rgm/complement_memory_features.json`;
- `output/v16_rgm/holdout_memory_oracle_report.json`;
- `output/v16_rgm/holdout_memory_features.json`.

To reproduce:

```powershell
Task1 - LegalIR\experiments\track_b_v11_research\run_v16a_memory_oracle.ps1 -Split complement -Device cuda
Task1 - LegalIR\experiments\track_b_v11_research\run_v16a_memory_oracle.ps1 -Split holdout -Device cuda
```

## V16b c35-only memory slot-5 gate

The bottleneck audit showed that the useful intervention is inside the frozen
c35 candidate set. V16b therefore:

- keeps exactly the existing 35 Stage-2 candidates per query;
- appends the QREL-memory/prototype/degree-corrected features only to those
  candidates;
- uses grouped OOF complement selection with a memory floor of `0.005`;
- keeps the locked v11h first four answers and can replace only slot 5;
- uses the locked v11h answer as the holdout/public anchor;
- remains under the BTC parameter limit: `1,090,010,000 < 4,000,000,000`.

| Split | Anchor | V16b gate | Delta |
|---|---:|---:|---:|
| Complement, 1,197 clean queries | 0.878752 | 0.886062 | **+0.007310** |
| Locked holdout, 1,000 queries | 0.918833 | 0.922833 | **+0.004000** |

The gate changed 177 holdout queries, rescued 9 and made 4 destructive changes.
Because the locked holdout improved over v11h, V16b was promoted. The packaged
submission is `output/v16b_memory_slot5.zip`; the full report is
`output/v16_rgm/v16b_memory_ltr_report.json` and the reproducible runner is
`experiments/track_b_v11_research/run_v16b_memory_slot5.ps1`.

## V16d-lite graph/QREL adapter diagnostic

The next roadmap step was tested as a strictly bounded diagnostic. A tiny
residual adapter was trained from fit-only QREL graph supervision, but its
output was applied only to the historical-query representation used by memory
KNN. It did not modify document retrieval, the c35 candidate set, the frozen
cross-encoder, PRF, or BM25. The run used the clean complement only; three
normalized overlaps were removed and neither public labels nor locked holdout
labels were read.

| Clean-complement metric | Original BQB | V16d-lite | Delta |
|---|---:|---:|---:|
| Neighbor QREL precision@5 | 0.089500 | 0.099167 | +0.009667 |
| Memory recall@5 | 0.295069 | 0.310764 | **+0.015694** |
| Prototype precision@5 | 0.063000 | 0.066333 | +0.003333 |

This confirms a real improvement in memory geometry. However, the actual V16b
downstream gate stayed at OOF Recall `0.8860623782` (delta `0.0000000`), while
intervention precision improved from `0.888889` to `0.920000`. Since the roadmap
requires downstream recall gain before opening holdout, V16d-lite is diagnostic
only and is not fused into V16b.

Reproduction:

```powershell
Task1 - LegalIR\experiments\track_b_v11_research\run_v16d_lite_diagnostic.ps1 -Device cuda
Task1 - LegalIR\experiments\track_b_v11_research\run_v16d_lite_oof_gate.ps1 -Device cuda
```

Reports:

- `output/v16d_lite/v16d_lite_complement_report.json`;
- `output/v16d_lite/v16d_lite_oof_gate_report.json`.
