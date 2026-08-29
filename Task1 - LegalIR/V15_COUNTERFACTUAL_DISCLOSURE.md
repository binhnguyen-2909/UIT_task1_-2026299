# V15 Counterfactual Pairwise Legal Adjudication

Checkpoint: 2026-08-29.  This document records the post-V14 experiment exactly
as run on the complement split.  It is not a public submission and it does not
change `output/v11h.zip`.

## Decision

V15 algorithmic phase is **rejected before locked holdout**.  The complement
gate requires standard recall delta `>= +0.004`, law-family and query-template
delta `>= +0.003`, intervention precision `>= 0.90`, destructive rate `<=
0.01`, and worst grouped fold delta `>= -0.001`.  No tested configuration met
the gate.

| Item | Result |
|---|---:|
| Complement baseline Recall@5 | `0.9018657755` |
| Best complement Recall delta | `-0.0012531328` |
| Best shortlist | `8` (frozen five + three challengers) |
| Best rewrite set | `base` |
| Best critical margin | `0.00` |
| Standard intervention precision | `0.40` |
| Standard destructive rate | `0.0292682927` |
| Standard changes / success / destructive | `205 / 4 / 6` |
| Oracle capture, shortlist 8 / 10 | `43.86% / 59.65%` |
| Locked holdout opened | **No** |
| Public applied | **No** |

The oracle is restricted to a valid one-slot rescue: a gold document must be a
Stage-1 c35 candidate at rank 6--35, not already in the frozen anchor, and the
current slot five must be wrong.  This corrected accounting identifies 57
slot-five-rescuable complement queries.  It is separate from the larger
unconstrained top-8/top-10 candidate oracle.

## End-to-end architecture

```text
frozen v11h Stage-1 c35
        |
        +-- V14 multi-signal prefilter over ranks 6..35
        |      (neural, rank, lexical, IDF, phrase, numeric, consensus)
        |
        +-- shortlist 8 or 10 = frozen anchor of five + 3 or 5 challengers
        |
        +-- direct A-minus-B legal comparison for every pair
        |      CE/v11/stage-1 deltas
        |      typed constraint and weakest-constraint deltas
        |      exclusive critical evidence
        |      evidence coverage, contradiction, number, structure
        |      BTC-only deterministic query-rewrite sidecar deltas
        |
        +-- tournament / weighted Copeland aggregation
        |      candidate must beat slot five, beat two neighbours,
        |      exceed aggregate tail margin, pass legal safety floor,
        |      pass critical-margin and strong-cycle gates
        |
        +-- replace slot five at most once
```

The tournament is a comparison mechanism, not a new independent relevance
score.  It never changes answers 1--4 and never expands beyond the frozen c35
candidate pool.

## Reused logic, written explicitly

- **v11h prior:** the existing Stage-1 c35 candidate IDs, Stage-2 neural
  scores, v11h scores, and 56-feature cache remain inputs.  The public v11h
  artifact remains the main submission.
- **V14 prefilter:** the label-free multi-signal union is reused to bridge c35
  to the 8/10 tournament.  This is why the second V15 run needed 4,061 raw
  documents rather than silently using only the first Stage-1 ranks.
- **Legal parser:** `v14_safe_rescue.parse_constraints` is reused for typed
  actor/action/object/condition/exception/temporal/modality/number/article/
  clause/point/topic constraints.
- **Evidence:** `load_context_documents`, legal chunking, and
  `build_candidate_features(..., with_evidence=True)` are reused.  Candidate
  evidence still uses the existing one-to-three-passage legal evidence logic,
  contradiction, numeric, structure, weakest-constraint, and independent
  signal calculations.
- **Metric and policy:** `official_metrics` and `intervention_report` are
  reused.  The policy is still one replacement at slot five, with complement
  selection only and no holdout feedback.

## New V15 logic

1. A document is represented as a `DocView` containing frozen scores, legal
   evidence, typed constraints, rewrite scores, and content tokens.
2. For `(q, dA, dB)`, features are strictly A-minus-B.  Lower contradiction is
   reversed so positive always means “prefer A”.
3. Exclusive critical support counts typed constraint support present in A and
   absent or weaker in B.  It is not a second global relevance score.
4. Rewrites are deterministic BTC-only sidecars: boilerplate removal, critical
   duplication, exception split, structure focus, actor-action focus,
   negation preservation, and numeric preservation.  No external corpus and
   no learned rewrite parameter are used.
5. The strong-cycle gate rejects a challenger involved in a confident
   A-beats-B-beats-C-beats-A cycle.
6. The frozen V14 legal safety floor remains mandatory before a pairwise win
   can replace the tail.

## Why the next neural branch is closed

The candidate oracle proves that c35 still contains reachable headroom, but the
algorithmic pairwise signal did not transfer into safe recall gain: the best
configuration lost `0.001253` recall and had only `0.40` intervention
precision.  Therefore the research condition for LoRA pairwise/group-softmax
is not met.  The listwise specialist and retrieval outside c35 remain closed;
they must not be opened based on this failed gate.  A future run should first
improve or replace the legal difference signal on complement and pass all
pre-holdout gates.

The residual audit on complement is also explicit: 57 queries are slot-5
rescuable inside c35, 13 have a gold miss in locked top-5 positions, 36 already
have a gold tail, and 1,091 are candidate-starved at c35.  This does **not**
open retrieval yet: the research order requires a passing V15/LoRA adjudicator
before using that residual to justify a retrieval expansion.  The BTC-only
rewrite sweep selected `base`; adding boilerplate, critical, exception,
negation, structure, actor-action, or numeric sidecars did not improve the
complement gate and reduced intervention precision.

## Parameter budget

V15 adds no neural parameters and uses no new model:

| Component | Parameters |
|---|---:|
| v11h embedding | `330,000,000` |
| v11h cross-encoder | `760,000,000` |
| V15 algorithmic pairwise/evidence/rewrite logic | `0` |
| **End-to-end total** | **`1,090,000,000` (`1.09B`)** |

This is strictly below the requested 4B ceiling.  The earlier V14 ColBERT
artifact remains a separate rejected experiment at `1,649,497,216` total
parameters; it is not part of the V15 deployed path.

## Artifacts

- Runner: `experiments/track_b_v11_research/run_v15_counterfactual.ps1`
- Implementation: `scripts/v15_counterfactual_adjudication.py`
- Tests: `tests/test_v15_counterfactual_adjudication.py`
- Report: `output/v15_counterfactual/v15_counterfactual_report.json`
