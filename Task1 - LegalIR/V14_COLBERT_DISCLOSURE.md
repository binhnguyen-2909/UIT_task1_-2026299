# LegalIR V14 — Phase-B ColBERT late-interaction disclosure

## Scope and frozen baseline

V14 is a selective rescue layer around the frozen v11h system. It does not
replace retrieval, globally rerank an answer list, expand c35 to c50, or alter
the first four v11h answers. The only permitted action is replacing answer slot
five once with a non-anchor candidate whose frozen Stage-1 rank is 6–35.

The exact locked v11h anchor is reconstructed from:

- grouped-OOF pointwise selector scores in
  `output/v13_v11h_rebuild/v11h_scores.npz`;
- the `longest_phrase_tokens >= 10` exact-heading tail rescue in
  `output/v11_feature_cache/v2_query_norm_c80.npz`;
- frozen safe-tail Stage-1 c35 and Stage-2 cross-encoder audit artifacts in
  `output/v10_9_prf`.

The locked-holdout baseline assertion is Recall@5 = `0.9188333333333334`.
Any full run aborts if reconstruction differs by more than `1e-12`.

## Why Phase B exists

The rank-6–35 one-slot replacement oracle shows substantial headroom:

| Split | v11h baseline | Slot-5 oracle | Recall upside |
|---|---:|---:|---:|
| Complement (1,197 queries) | 0.901866 | 0.940922 | +0.039056 |
| Locked holdout (report-only, 1,000 queries) | 0.918833 | 0.953000 | +0.034167 |

Phase A produced a real but insufficient complement signal. Its best standard
OOF delta was `+0.002924` with intervention precision `1.0` and no destructive
changes, just below the required `+0.003`. Query-template OOD remained unstable
(`+0.000835`, precision `0.6`, worst fold `-0.00655`). V14 therefore added the
late-interaction discriminator described below, but did not relax the locked
holdout promotion protocol.

## Reused logic, completely enumerated

1. **Candidate generation:** frozen v11h PRF safe-tail Stage-1 c35. Only ranks
   6–35 not already present in the v11h top five can be proposed.
2. **Frozen neural signals:** the existing 330M embedding retriever and 760M
   Vietnamese cross encoder. Neither is fine-tuned in V14.
3. **Frozen v11 feature cache:** 56 explicitly curated columns covering ranks,
   Stage-1 components, Stage-2 query-normalized scores, exact lexical/numeric
   matching, title/body matching, PRF movement and rank consensus. Candidate,
   current tail and candidate-minus-tail copies are retained.
4. **Deterministic typed parser:** `v14_safe_rescue.parse_constraints` is reused
   verbatim for actor, action, object, condition, exception, temporal phrase,
   modality, number, article, clause, point and fallback topic.
5. **Legal hierarchy chunker:** `legal_structure.iter_legal_chunks` is reused at
   1,800 characters with 180-character overlap. Every chunk is considered; no
   document is prefix-truncated before evidence scoring.
6. **Phase-A evidence:** deterministic completeness, modality, contradiction,
   numeric and legal-structure features; all legal chunks are shortlisted to
   six and every one-, two- and three-passage set is considered (41 sets).
7. **Metric-aligned target:** positive means the proposed document is gold while
   the current slot-five document is not gold. Neutral and destructive outcomes
   remain explicit. Training is candidate-vs-tail, not generic relevance.
8. **One-slot policy:** choose at most one proposal per query above a frozen
   threshold; only `answers[-1]` may change.
9. **Validation groups:** standard leakage-aware groups, law-family OOD groups,
   query-template OOD groups and exact worst-fold diagnostics are reused.
10. **Immutable Phase-A pair cache:** all 255 reused candidate/tail/delta/query
    features come from
    `output/v14_replacement_full/cache/complement_pairs.npz` (32,912 rows).

## New Phase-B logic

The new model is `jinaai/jina-colbert-v2`, pinned to revision
`a9dc5cd7293d4c71dbbba04829923ba4d0e4f6ea`. Its external XLM-RoBERTa remote
code is pinned to revision
`bd55a5ec8e6c0fb1d6c26efb4b6a4a74ce8a88d3`. The license reported by the model
card is CC-BY-NC-4.0, so commercial deployment requires a separate license
review.

The full 0.559497216B network is loaded in FP16. Token embeddings are truncated
to Matryoshka dimension 64 and re-normalized; truncating the output dimension
does not reduce the counted network parameters.

For each query/document pair:

1. Encode the full query once.
2. Encode each non-empty typed legal element once; duplicate strings are cached.
3. Encode the title and every hierarchical legal chunk document-major, so a
   document shared by multiple queries is encoded once.
4. Compute token-level ColBERT MaxSim for the full query and every typed element.
5. Rank all passages by full-query MaxSim, retain six, and exhaustively compute
   union MaxSim for every one-, two- and three-passage combination.
6. Record single-passage, top-k, margin, title, element mean/min/max/std,
   critical-element mean/min/max, each fixed element score, combination gains
   and passage count.
7. Join candidate, frozen tail and candidate-minus-tail copies of these fields.
8. Add per-query inverse-rank, z-score and min-max transforms for the thirteen
   pre-registered aggregate/critical fields.

Four label-free safety modes are evaluated without changing the one-slot rule:

- `phase_a_strict`: retain the original deterministic contradiction/evidence
  hard gate;
- `phase_a_or_colbert_dominance`: accept the old gate or a fixed full-query plus
  critical-element ColBERT dominance test;
- `colbert_ce_rank25`: accept only a proposal with non-negative frozen
  cross-encoder margin over the v11h tail and frozen Stage-1 rank <=25. This
  is a two-signal guard (existing CE agreement plus ColBERT ranking), and the
  rank cap removes the noisy lower tail of the rank-6--35 proposal pool. The
  constants are frozen before any holdout/public evaluation;
- `calibrated_all`: treat deterministic contradiction as a model feature and
  rely on OOF threshold calibration plus intervention-precision gates. This
  avoids permanently discarding the 15/58 complement positive rows rejected by
  the Phase-A hard gate.

The selector leaderboard includes three frozen zero-shot ColBERT formulas and
the pre-registered Phase-A logistic/LightGBM configurations. No Phase-B neural
fine-tuning is performed.

## Leakage and promotion protocol

All model fitting, gate-mode choice and threshold choice use complement labels
only. The locked holdout is not encoded until one configuration passes every
complement requirement:

- standard grouped OOF Recall delta >= `+0.003`;
- law-family OOD Recall delta >= `+0.002`;
- query-template OOD Recall delta >= `+0.002`;
- intervention precision >= `0.85` and destructive rate <= `0.03` in each;
- worst fold across all three validations >= `-0.002`.

If and only if these pass, the threshold, model, feature schema and safety mode
are frozen and evaluated once on locked holdout. Promotion additionally
requires holdout Recall@5 >= `0.925`, positive delta, intervention precision >=
`0.85` and destructive rate <= `0.03`. Public data is encoded and changed only
after holdout promotion.

## End-to-end parameter accounting

| Component | Parameters |
|---|---:|
| Frozen v11h embedding model | 330,000,000 |
| Frozen v11h cross encoder | 760,000,000 |
| Jina ColBERT v2 (runtime exact count) | 559,497,216 |
| Phase-A algorithmic parser/evidence | 0 |
| Replacement selector | counted exactly after fitting (0 for frozen direct policy) |
| Total before tiny selector | **1,649,497,216 (1.6495B)** |
| Hard ceiling | **strictly less than 4,000,000,000** |

The runner asserts the ceiling before scoring and again after the deployed fold
ensemble has been selected. It counts every fold selector's scalar parameters.

## GPU contention and checkpoints

Device policy is CUDA-first. On the RTX 3050 6 GB, the model uses FP16/eager
attention and Matryoshka dimension 64. The completed holdout run loaded the
model on CUDA with 3,319,791,616 bytes free at load time; no unrelated OCR
process was running at the final checkpoint. If another GPU job contends for
VRAM, this runner keeps it intact, retries with smaller CUDA batches, and only
after a repeated CUDA OOM moves this runner to CPU. It never kills another
process to recover VRAM.

Query embeddings and document-pair features are independently cached. Document
scores are atomically checkpointed, currently every 100 documents, and a run
resumes only when model revision, code revision, split, source artifacts,
feature schema and embedding dimension match the cache signature.

## Reproduction

Install only the Phase-B additions:

```powershell
.\rtk.ps1 pip install -r "Task1 - LegalIR\requirements-v14-colbert.txt"
```

Run the complete guarded pipeline (CUDA first, holdout/public gated):

```powershell
& "Task1 - LegalIR\experiments\track_b_v11_research\run_v14_colbert_end_to_end.ps1" `
  -Device auto -QueryBatchSize 8 -DocumentBatchSize 2 `
  -ScoreQueryBatchSize 8 -CheckpointEvery 100 -ApplyPublic
```

For a resumable smoke checkpoint, add `-MaxDocumentsThisRun 25`.

## Artifacts

- Oracle report: `output/v14_replacement_oracle/v14_replacement_oracle_report.json`
- Phase-A report: `output/v14_replacement_full/v14_replacement_report.json`
- Phase-B report: `output/v14_colbert_full/v14_colbert_report.json`
- Query cache: `output/v14_colbert_full/cache/complement_colbert_queries.npz`
- Atomic score cache: `output/v14_colbert_full/cache/complement_colbert_scores.npz`
- Locked-holdout pair cache: `output/v14_colbert_full/cache/holdout_pairs.npz`
- Locked-holdout score cache: `output/v14_colbert_full/cache/holdout_colbert_scores.npz`

## Final guarded-run checkpoint (2026-08-29)

The complement-selected frozen policy was:

- score: `frozen_full_delta` = ColBERT full-query best three-passage union
  MaxSim(candidate) minus the same value for the v11h slot-five tail;
- safety mode: `colbert_ce_rank25` = frozen v11h cross-encoder candidate-minus-tail
  margin `>= 0` and Stage-1 candidate rank `<=25`;
- threshold: `0.02197723388671874`;
- replacement: at most one candidate, only in slot five.

The exact complement result passed every pre-holdout gate: Recall delta
`+0.005012531328320802` in standard, law-family OOD and query-template OOD;
intervention precision `0.8888888888888888`; destructive rate
`0.004651162790697674`; eight successful and one destructive rescue.

The locked holdout was then evaluated once with that frozen policy. It was
rejected and never used to retune the policy:

| Metric | v11h baseline | V14 rescued | Delta |
|---|---:|---:|---:|
| Recall@5 | 0.9188333333333333 | 0.9178333333333333 | -0.0010000000000000009 |
| Precision | 0.1959999999999980 | 0.1957999999999980 | -0.0001999999999999780 |
| Interventions | — | 191 | — |
| Successful / destructive | — | 1 / 2 | — |
| Intervention precision | — | 0.3333333333333333 | — |
| Destructive rate | — | 0.010471204188481676 | — |

Status is `holdout_rejected`; public data was not encoded and no public
submission was changed. The exact machine-readable result is
`output/v14_colbert_full/v14_colbert_report.json`. Because the new signal is
not stable across the locked split, the next research step remains a fresh
holdout-blind robustness experiment; V14 does not open LoRA/listwise fine-tuning
or replace the v11h public baseline.
