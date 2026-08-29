# v11 research experiment log

## 2026-08-25 — GPU fit/holdout validation and exact-heading rescue

All runs below use the compliant v11 model plan (BQB Vietnamese embedding
330M + Vietnamese reranker 760M; total 1.09B). No v57 checkpoint, prediction,
rank cache, or upstream LLM was loaded. v57 exact10 was used only as the
algorithmic inspiration for the one-slot exact-heading rescue.

### Stage 1

Configuration: `v11_research_fit_256`, legal chunks, dense weight `0.30`,
dense RRF `k=20`, query dense weight `0.05`, 200 candidates, `chunk_only`,
CUDA query encoding.

- Fit: 800 questions; Recall@5 `1.0000`; candidate oracle @20/@100/@200
  `1.0000`.
- Holdout: 1,000 questions; Recall@5 `0.8279167`; candidate oracle @20
  `0.9383333`, @100 `0.9821667`, @200 `0.9855`.

Artifacts:

- `output/v11_fit_stage1_chunk_only_top5.audit.json`
- `output/v11_holdout_stage1_chunk_only_top5.audit.json`
- `output/v11_public_stage1_chunk_only_top5.audit.json`

### Stage 2

Configuration: 20 candidates, 3 passages/document, reranker max length 768,
FP16, neural RRF weight `1.0`, RRF `k=10`, top-3 passage aggregation.

- Fit: Recall@5 `0.991875`; Precision@5 `0.21625`.
- Holdout: Recall@5 `0.8889167`; Precision@5 `0.1898`.

Artifacts:

- `output/v11_fit_stage2_c20_p3_top5.audit.json`
- `output/v11_holdout_stage2_c20_p3_top5.audit.json`

### Exact-heading rescue

The policy replaces only slot 5, only when a v11 Stage-1 candidate has a
strictly longer normalized exact phrase than the current Stage-2 tail. The
threshold is selected on fit only; holdout is report-only.

- Fit-selected threshold: `10` tokens.
- Fit baseline → threshold 10: `0.991875` → `0.990625` recall.
- Holdout baseline → threshold 10: `0.8889167` → `0.8959167` recall
  (`+0.0070`), precision `0.1898` → `0.1912`.
- Threshold 9 reaches `0.8974167` on holdout but is not selected because that
  would tune on holdout.
- Public changes: 150 one-slot replacements.

Artifacts:

- `output/v11_exact_heading/v11_exact_heading_report.json`
- `output/v11_exact_heading/v11_exact_heading_t10.json`
- `output/v11e.zip` — short-name submission candidate.

### Decision

Submit `v11e.zip` as the next public trial. Keep `v11.zip` as the v11 anchor
(leaderboard score `0.8962`) and keep v57 exact10 only as an algorithm
reference. Do not promote threshold 9 without a fresh fit-only selection.

## 2026-08-25 — V10.7 dense sweep and V10.4 reranker reach

The Stage-1 runner now records leak-free overrides for dense RRF weight/k and
question-dense settings, allowing ablations without rebuilding the index.

- Dense `.60` on the full holdout: Stage-1 Recall@5 `0.8240833`, oracle@20
  `0.9518333`; the higher candidate ceiling did not improve top-5 ordering.
- Dense `.60` + Stage-2 c20/p3/768: Recall `0.8910833`; better than the
  `.30` c20 baseline `0.8889167`, but below c35.
- `balanced`/`full` probes did not beat the stable `chunk_only` branch at the
  tested `.60` setting, so they were not promoted.

### Reranker reach experiment

Using the same Stage-1 `chunk_only` audit, BQB embedding, Vietnamese
reranker, p3, max length 768 and FP16:

- c20: holdout Recall `0.8889167`.
- c35: holdout Recall `0.89925`, Precision `0.1916`.
- c35 + exact-heading threshold 10 selected on fit: holdout Recall
  `0.90075`, Precision `0.1920`.
- Threshold 8 reaches `0.9015833` on holdout but remains report-only because
  it was not selected on fit.

Public c35 was scored with the same compliant model plan and the c35
threshold-10 artifact was packaged as `output/v11c.zip`. This is the next
submission candidate; c80 remains the next V10.4 experiment only if c35's
public score does not hold.

## Public result — v11c

The submitted `v11c.zip` returned final public scores:

- Recall: `0.903666668`
- Precision: `0.193800001`

This establishes v11c as the new public anchor, improving over v11e recall
`0.899166668` and v11 recall `0.8962`. The result ZIPs were verified locally:
`scoring_result (1).zip` contains `scores.json` and reports the values above;
`prediction_result (1).zip` contains the platform metadata. The next planned
experiment remains V10.4 reranker reach c80, with c35 locked as the comparison
baseline.

## V10.4 — c80/p1 screening

Using the locked Stage-1 `chunk_only` audit and the same reranker/max-length
configuration as c35, increasing the Stage-2 candidate reach to 80 while
keeping one passage per document produced:

- Holdout Recall: `0.864583333`
- Holdout Precision: `0.1838`
- Pairs scored: `80000`

This is a clear regression versus c35 raw Recall `0.89925` and c35 plus the
fit-selected exact-heading rescue Recall `0.90075`. The p1 branch is rejected;
the next V10.4 probe is c80/p3 to test whether the regression is caused by
single-passage evidence loss rather than candidate reach alone.

## V10.4 — c80/p3 timeout

The c80/p3 holdout probe was launched with the locked `chunk_only` Stage-1
audit, Vietnamese reranker, max length 768, FP16 and batch size 24. It built
`239716` query-passage pairs and consumed one hour of GPU time without writing
the final audit/ZIP. It was stopped at the explicit timeout; no metric is
assigned and the branch is not promoted. GPU memory returned to idle after
the stop. The next action is a lower-cost reach/aggregation probe rather than
rerunning the same unbounded workload.

## V10.4 — c50/p3 timeout

The intermediate c50/p3 probe used the same Stage-1 audit, reranker and
max-length 768 configuration with batch size 32. It built `149835`
query-passage pairs and was stopped after the one-hour GPU timeout without
writing a final audit/ZIP. No metric is assigned and the branch is not
promoted. The repeated long-tail behavior indicates that full-reach
CrossEncoder sweeps are not cost-effective on the 6 GB GPU; subsequent work
uses the completed c35 audit for deterministic fusion and rescue ablations.

## Deterministic reach sweep and next candidate — v11d

With c35 Stage-2 audits already complete, the exact-heading algorithm was
swept offline over Stage-1 candidate cutoffs. The threshold was selected on
fit only and then reported on holdout:

- cutoff 35 / threshold 10: holdout Recall `0.90075`.
- cutoff 50 / threshold 10: holdout Recall `0.90075`.
- cutoff 80 / threshold 10: holdout Recall `0.90275`, Precision `0.1924`.

The cutoff-80 policy is the best stable fit-selected reach in this sweep. The
public submission was regenerated from the raw c35 public anchor (not from a
public score) and packaged as `output/v11d.zip`; it differs from v11c in 13
answer slots. It uses the same BQB embedding + Vietnamese reranker plan,
totaling `1.09B` parameters and remaining under the `<4B` budget. ZIP and
question-ID/answer-shape validation passed; external public scoring remains
the next required verification.

### Reach-80 fit-side audit

The matching c80/p1 fit run completed on GPU with `64000` pairs:

- Fit Recall: `0.962291667`
- Fit Precision: `0.2085`
- Holdout Recall: `0.864583333`
- Holdout Precision: `0.1838`

The c80/p1 neural reach is rejected on both fit generalization and holdout;
the fit-side artifact is retained only for audit. The selected v11d artifact
therefore uses the completed c35 reranker plus deterministic exact-heading
reach80/t10, whose holdout Recall is `0.90275`.

## Public result — v11d plateau

The submitted `v11d.zip` returned leaderboard Recall `0.9037` (rounded), with
no material change from v11c Recall `0.903666668`. Deterministic exact-heading
reach is therefore considered saturated; the next roadmap phase is V11 tiny
supervised fusion/LTR.

## V11 tiny LTR — leakage audit and rejected protocol

The original 800-question fit audit cannot supervise LTR because its retrieval
metadata contains `train_exact_memory`. A first protocol split the clean
1,000-question holdout into grouped selection/final partitions. It achieved
selection OOF Recall `0.91718`, but the locked document-disjoint final test
reached only `0.89747`, below its c35+exact anchor `0.90445`. No submission was
created. Artifact:

- `output/v11_tiny_ltr/v11_tiny_ltr_report.json`

## Complement clean split

A complementary index was made by reusing the official corpus and BQB dense
cache, then replacing only the supervised layer with the 1,000 old holdout
questions. A label-blind SHA256 sample of 1,200 questions from the other 6,000
was retrieved by that index. Three normalized duplicate questions activated
exact memory and were excluded, leaving 1,197 clean LTR questions.

Complement Stage-1:

- Recall@5 `0.7548194`.
- Oracle@35 `0.9414861`; oracle@80 `0.96725`.

Complement Stage-2 c35/p3/768/FP16:

- Recall `0.8645503` after excluding the three exact-memory rows.
- Precision `0.1837928`.
- 125,887 query-passage pairs scored on the RTX 3050.

Artifacts:

- `data/processed/v11_research_complement_256/`
- `output/v11_complement_stage1_chunk_only_top5.audit.json`
- `output/v11_complement_stage2_c35_p3_top5.audit.json`
- `experiments/track_b_v11_research/artifacts/complement_split/`

### Rejected supervised-prior run

The first complement LTR run reached OOF Recall `0.92387` but collapsed to
`0.73458` on the locked original holdout. Audit showed that
`document_prior` came from the 6,000-question fit index and therefore directly
represented complement labels. This feature was disabled globally; the
failed report is retained at
`output/v11_tiny_ltr_complement/v11_tiny_ltr_complement_report.json`.

### Promoted no-prior LTR — v11f

With supervised document prior fixed to zero, grouped model/policy selection
chose a tiny CUDA RankNet (one 64-unit hidden layer, under 10,000 parameters),
top-35 model ranking, and the fixed exact-heading threshold-10 rescue.

- Complement grouped OOF: Recall `0.89434698`, Precision `0.19030911`, fold
  Recall std `0.00628008`.
- Complement c35+exact anchor: Recall `0.86851852`; delta `+0.02582846`.
- Locked original holdout: Recall `0.90791667`, Precision `0.1932`.
- Holdout c35+exact anchor: Recall `0.90275`; delta `+0.00516667`.
- Model plan: BQB 0.33B + reranker 0.76B + tiny RankNet <0.00001B = about
  `1.09B`, under 4B.

The public mapping changes 986/1,000 queries versus v11d while retaining
72.4% mean answer-set overlap. It is packaged as `output/v11f.zip`; ZIP member,
1,000-QID identity, five unique IDs/query, and all 17 unit tests pass. Public
leaderboard scoring remains the final external gate; keep v11d as fallback.

### V13 conditional routing and controlled reranker fine-tune — final

V13 used the locked v11h anchor and v12 outputs on a clean complement split.
The diagnostic union top-5 had recall `0.91502367`, with 13 v12-only wins and
21 v11h-only wins. A grouped OOF router reached robust recall `0.90098693`,
slightly above the v11h complement robust baseline `0.90021953`, so the locked
holdout gate was opened.

Locked holdout results:

- v11h anchor: Recall `0.91883333`.
- V13 router: Recall `0.91883333` (no gain).
- Controlled hard-negative LoRA reranker fine-tune: Recall `0.90108333`,
  Precision `0.19240000`, delta `-0.01775000` versus v11h.

The fine-tune trained only on clean complement data and was evaluated on GPU.
The merged checkpoint is retained at
`output/v13_reranker_finetune/model/`, with evaluation at
`output/v13_reranker_finetune/holdout_stage2.json`. It is rejected and v11h
remains the main public submission. This closes the V10→V11 research path as
requested; no further experiments are scheduled.
