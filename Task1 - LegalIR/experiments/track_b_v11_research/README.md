# Track B — v11 research upgrade

This runner is an isolated A/B branch over the v10 codebase. It does not
overwrite `data/processed/v10_github_rebuild`, the v10 submission, or v57
artifacts. V57 contributes algorithmic ideas only; no v57 checkpoint, cached
prediction, or upstream LLM is loaded.

The end-to-end model budget is approximately **1.09B**:

```text
BQB Vietnamese legal embedding  0.33B
Vietnamese reranker             0.76B
                                -----
                                1.09B < 4B
```

Implemented changes:

1. BTC-only text normalization and duplicate-question label union during index
   build. The original question remains the representative; no external text
   or generated labels are introduced.
2. Structure-aware `Điều`/`Khoản`/`Điểm` chunking with the existing legal
   chunker, keeping the v10 character budget for a fair comparison.
3. Candidate pool 100 → 200, dense weight 0.05 → 0.30, dense RRF k 60 → 20,
   and dense max length 384 → 256 for the RTX 3050 runtime budget.
4. Reranker reach was ablated at c20/c35/c50/c80. c80/p1 regressed on both
   fit and holdout; c50/p3 and c80/p3 exceeded the one-hour GPU budget. The
   selected public candidate keeps the completed c35/p3 reranker and extends
   the deterministic exact-heading lookup to Stage-1 cutoff80.
5. Reranker max length 256 → 768 and document score aggregation
   `best + 0.15*second + 0.05*third`. The default remains `max` for v10.
6. Stage-2 RRF weight/k 0.8/5 → 1.0/10.
7. Tiny supervised fusion/LTR now uses a complementary clean index and grouped
   OOF selection. Supervised document prior is disabled after a locked test
   exposed label leakage. The promoted candidate is a 64-hidden-unit RankNet
   with c35 ranking plus threshold-10 exact-heading rescue.

Run from the repository root:

```powershell
Task1 - LegalIR\experiments\track_b_v11_research\run_v11_research.ps1 -Device cuda
```

To reproduce the promoted complement/LTR candidate through packaging, run:

```powershell
Task1 - LegalIR\experiments\track_b_v11_research\run_v11f_end_to_end.ps1 -Device cuda
```

The end-to-end runner is cache-aware and skips already validated artifacts;
use `-Force` only when intentionally recomputing expensive stages.

The script writes the new index under
`Task1 - LegalIR/data/processed/v11_research_256` by default and the submission
under `Task1 - LegalIR/output`. Use `-NoReranker` to build
only Stage 1, then inspect the audit before spending GPU time on Stage 2.

Compare `Recall@5`, candidate-oracle recall, reranker input coverage, latency,
and the number of unique answers per query against the v10 audit.

Current public fallback: `output/v11d.zip` (leaderboard Recall `0.9037`
rounded; holdout Recall `0.90275`). Next public trial: `output/v11f.zip`
(clean complement grouped-OOF Recall `0.89435`; locked holdout Recall
`0.9079167`).
