# Lessons and current bottlenecks at 0.9210

## What is proven

V11h is the reliable retrieval/reranking base: its query-normalized LTR achieved 0.9183 public and 0.918833 locked-holdout Recall@5. V16b improved it without touching the top four or creating candidates. Its fit-only memory gate produced a locked-holdout gain of +0.004000 to 0.922833 and the submitted ZIP scored 0.9210 publicly. This is the only promoted lineage.

The observed bottleneck is not that a selector has no signal. It is safe cross-regime transfer: a policy must distinguish rescue from destructive rank-5 swaps after it leaves the average OOF distribution.

## Evidence for stopping promotion

| Attempt | Useful signal | Why it is not a release candidate |
|---|---|---|
| Listwise/rank-normalized selector after V11h | Better complement OOF | Locked holdout fell to 0.915000/0.914000: OOF alone overstates transfer. |
| P1 pairwise deferral | Recall +0.002924; bootstrap lower bound +0.000418; precision 1.0 | Missed +0.003 by 0.000076 and failed all four registered OOD/regime checks. |
| V19 block aggregation | Block oracle +0.038499 | Weighted/calibrated policies had negative bootstrap lower bounds, poor calibration, or a negative worst regime. |
| P2 CATER-R | Block oracle +0.039404; cached CE contains some pair signal | Adaptive/weighted interventions had precision 0.594595 and negative bootstrap bounds. The four-head setwise model abstained into zero gain, robust -0.000320, precision 0.5. |
| P3 LoRA + CATER-R exploratory result | OOF +0.003342, robust +0.004091, bootstrap LB +0.000835, precision 1.0 | The adapted CE's OOF scores fail required group isolation (548 standard, 605 law-family, 1,063 template, 19 near-duplicate validation overlaps). It cannot validate a holdout promotion. |

## Why 0.921 is hard to exceed safely

1. **Candidate coverage limits the ceiling.** The later clean-complement c35 oracle is about 0.95039. A c35-only selector can reorder, not recover a missing relevant document, so it cannot by itself substantiate a 0.96 result. The separately audited expanded-pool coverage oracle reached 0.976093, which is the likely eventual source of headroom.

2. **Rank-5 changes have asymmetric cost.** There are few true rescues and a single destructive swap removes one of only five output documents. V16b succeeded because it was deliberately narrow; less conservative gates gain average recall but lose intervention precision or subgroup safety.

3. **Average OOF is insufficient.** The rejected V11 listwise and later P1/CATER results all demonstrate that a good pooled average can hide a bad law-family, template, duplicate, or standard-regime outcome.

4. **Current CE signal is weak rather than inverted.** Frozen CE rescue-vs-destructive pair AUC is 0.652872; rescue median margin is -0.0003002 and destructive is -0.0096724. It can help, but is not reliable enough to justify broad swaps. Fine-tuning is not a shortcut unless its score generation is fully group-cross-fitted.

5. **Inference parity is part of calibration.** The abandoned external r8 path used `skip-bqb`, structural-first blocks, and replacement of un-packable challengers. That changes the meaning/distribution of CATER features relative to OOF. A positive or negative result from it cannot evaluate P3.

6. **Holdout is a confirmation, not a tuning loop.** It remains closed. The exact-parity preparer found five invalid documents affecting ten queries; those must force abstention and keep document identity, never silently substitute the next challenger.

## Rule for a future promotion

Keep V16b as the anchor. Promote a new policy only after it has all of:

- at least +0.003 Recall@5,
- a positive paired-bootstrap lower bound,
- required robust gain and high intervention precision,
- non-negative results in all four independently cross-fitted regimes, and
- identical OOF/holdout preprocessing, candidate identity, evidence packs, fallback, and feature definitions.

Only after a valid selector passes should an expanded candidate pool be tried progressively (K16, then K32, then K64) using the same conservative gate. Before that, re-train any LoRA CE with boundary-specific semi-hard pairs, LoRA, anchor-preservation regularization, and scores cross-fitted by the exact four registered group definitions. Never use manual labels for BTC data.
