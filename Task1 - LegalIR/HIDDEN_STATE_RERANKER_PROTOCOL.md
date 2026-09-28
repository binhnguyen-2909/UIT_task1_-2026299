# Hidden-state c67 linear reranker protocol

This is a new, complement-only branch. It freezes the approved
`Qwen/Qwen3-Reranker-0.6B` checkpoint, reuses the validated long legal packer
at 4096 tokens, and trains only one linear head with 1024 weights plus a bias.
It does not modify R1, E3B, E4, E5, or the accepted V16b release.

## Locked data boundary

- Candidate pool: the exact first 67 IDs from the frozen Stage-1/Stage-2
  universe, including the E3B incumbent.
- Backbone cache: label-blind and reusable across folds; at most
  `1197 * 67 = 80,199` query-document vectors.
- Nested split: the E3B report's `a_fit + a_stop` is T, its calibration is C,
  and its outer validation is V. Two inner seeds and the same stress regimes
  are required. No random replacement split is allowed.
- E3B score artifacts remain mandatory for reconstructing the fold-specific
  incumbent and checking provenance; deployment weights are not a substitute.

## Feature and native parity

For every c67 pair the GPU exporter stores the final non-padding decoder hidden
state, the native `yes - no` logit, evidence validity, packing/input hashes and
the model/tokenizer/packer/pool lineage. It uses `last_hidden_state` only and
does not retain all decoder layers. The checkpoint config must report
`hidden_size=1024`.

The exporter checks that the last-state linear readout reproduces the native
logit delta before allowing a pilot/full cache, with locked tolerances recorded
in the GPU probe. The on-disk hidden dtype is float16 by default; round-trip
error is measured before labels are opened.

## Head and loss

For hidden state `h`, use `x = h / max(||h||2, 1e-8)` and
`s = z0 + w·x + b`, initialized with `w=0, b=0`. No QID, document ID,
source-family or rule-family feature is permitted.

Training samples are made only from T: all valid gold documents present in c67
are positives; up to eight deterministic negatives are four high-native-score
non-gold documents plus four middle-ranked non-gold documents. Queries without
a positive in c67 still contribute pointwise negative loss. The pointwise and
pairwise terms are query-weighted by `1 / max(|Gq|, 1)`. The head is fit in
float32 with full-batch L-BFGS on CUDA, at most 200 iterations, for the locked
regularisations `{1e-4, 1e-3, 1e-2}`. Every head checkpoint keeps its weights,
optimizer state, train-QID hash, cache lineage and convergence history.

## Selector and comparisons

The fold-specific E3B output is the default answer. For a challenger in c67,
replace only slot 5 when `s(challenger) > 0` and
`s(challenger) - s(incumbent) > tau`, with `tau` in `{0,1,2}`. Top-4, five
unique IDs, c67 membership and abstention are checked for every query.

The report contains E3B baseline, native Qwen with its C-selected tau, and the
hidden head with its C-selected `(regularisation, tau)`. Selection is made on
C only; V is read once. Ties prefer fewer destructive changes, fewer changes,
stronger regularisation/larger tau, with exact E3B baseline retained on a full
tie.

The new branch passes only if standard gains at least `0.003` Recall over E3B
for both inner seeds, has a positive grouped paired-bootstrap lower bound,
rescues more than it destroys, and remains non-negative with the same checks
for every stress seed/regime. These are research gates, not a promise of 0.95.

## Holdout/release rule

Preflight, pilot, cache, standard and stress are complement-only. Public
features are not generated unless the new stress gate passes. A final head
refit on all complement and public inference require a separate release step;
they do not inherit E4/E5 approval. Keep the accepted V16b/E3B artifact as the
fallback and keep `new_policy_holdout_validated:false`.
