# Qwen c35/c67/c200 nested research protocol

## Scope

This is a complement-only research branch after E3-B.  It keeps the sealed
V16b top four IDs unchanged, chooses only slot 5, and starts from the frozen
Stage-1 audit.  It does not rerun E0--E3B, open holdout, generate a public
submission, or execute E4/E5.

The branch is intentionally diagnostic: the c35/c67/c200 oracle recalls are
upper bounds on a candidate pool, not model results or expected private score.

## Frozen inputs

- Stage-1 provides at least 200 unique candidates per complement query.
- Stage-2 must retain the exact Stage-1 c35 prefix.
- V16b supplies the five-ID anchor; ranks 1--4 stay frozen.
- E1 supplies the source-family evidence manifest for nested group splits.
- completed passing E3-B standard and stress reports are a prerequisite.

All cache metadata records SHA-256 fingerprints for Stage-1, Stage-2, V16b,
the contexts ZIP, Qwen checkpoint and scoring code.  Any mismatch is a hard
failure.

## Qwen score cache

The cache has exactly two GPU passes:

| Cache | Pool | Context | Also serves |
| --- | ---: | ---: | --- |
| A | c35 | 768 | A/S1 and A/S2 |
| D | c200 | 4096 | B=c35-long, C=c67-long, D=c200-long |

The scorer uses Qwen3-Reranker-0.6B via the official causal-LM `yes` minus
`no` final-logit score.  Legal text packs are label-blind: preserve a full
document if it fits; otherwise select lexical legal-boundary units and restore
their original order.  Scoring runs on CUDA BF16.  Host-side ZIP/JSON I/O is
restricted to one CPU thread and never performs selector metrics.

Cache is resumable by QID.  A restart discards incomplete QID records and
recomputes only those records, so an OOM/interruption cannot produce a partly
scored query that looks complete.

## Nested evaluation

The evaluator writes and freezes a label-blind cache manifest first.  Only
then does it load complement qrels.  For every grouped outer fold and each
inner seed it forms non-overlapping A-fit, A-stop, C (calibration) and V
(validation) partitions using the E3-B partition function.

For each A/B/C/D pool, it compares:

- **S1:** top Qwen challenger versus V16b incumbent using a gap threshold.
  The threshold is selected solely on C from `{0, 0.25, 0.5, 1.0}` plus the
  abstain baseline.
- **S2:** a tiny 5-feature MLP trained on top-three Qwen challengers in
  A-fit, early stopped on A-stop, to predict candidate `Δ Recall@5` against
  the incumbent.  It chooses a replacement only if predicted delta is above
  zero; otherwise it abstains.

The eight A/B/C/D × S1/S2 configurations are compared only on C.  The chosen
configuration is applied once to V.  Output reports include candidate pool
oracle, Qwen rank Recall@1/@3, calibration traces and exact V16b transition
metrics.  The only research gate is positive gain and more rescues than
destructive changes for both inner seeds.  Passing standard authorizes this
branch's stress test; it never authorizes E4/E5 or holdout.

The Qwen cache checkpoints after each complete QID.  Nested evaluation
checkpoints predictions and the fold report after each complete outer fold.
Every S2 fit—A/B/C/D, regime, inner seed and outer fold—retains all of the
following even if a later gate fails:

- `last.pt`: weights, AdamW optimizer state, cursor/step, RNG state, scaler,
  feature schema, architecture, policy and provenance, written atomically at
  every epoch so that the fit can resume;
- `selected.pt`: the state selected solely by minimum A-stop SmoothL1, never
  by calibration C or validation V;
- `selected.backup.pt`: hash-verified copy of `selected.pt`; and
- `fit_complete.json`: only written after each saved state is reloaded into a
  fresh model, scored on a fixed A-stop batch within tolerance, and produces
  identical slot-5 decisions.

An interrupted incomplete fold resumes its S2 fit from `last.pt`; completed
outer folds remain protected by the outer-fold prediction checkpoint.

## Runs

1. `MODE=preflight` validates frozen cache inputs without loading complement
   qrels.
2. `MODE=cache` creates A+D GPU caches.  Use `SHARD_COUNT`/`SHARD_INDEX` to
   divide independent cache shards over GPUs; evaluator begins only after all
   shards exist.
3. `MODE=cache-preflight` validates all shard metadata/coverage without
   loading complement qrels.
4. `MODE=standard` evaluates nested standard groups after cache completion.
5. `MODE=stress` requires a passing Qwen standard report and evaluates the
   law-source, query-template and near-duplicate group regimes.
6. `MODE=bundle` packages the completed cache, retained S2 states, policy,
   feature schema, reports, code, frozen complement provenance and exactly one
   Qwen model/tokenizer directory into a hash-recorded reproduction bundle.
   The backbone is never duplicated per fit.

No command in this protocol accepts a holdout or public path.
