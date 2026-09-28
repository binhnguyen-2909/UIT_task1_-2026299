# E1 — evidence proposal + frozen cross-encoder diagnostic

E1 is the first experiment in the new evidence-first plan.  It is intentionally
limited to the already-sealed **complement** split and does not produce a
submission.

## Invariants

- The candidate universe is exactly the V16b c35 audit; E1 cannot retrieve a
  new document.
- Before any qrel file is opened, raw corpus text is first split at legal
  Điều/Khoản/Điểm boundaries, then windowed to 384 **reranker tokens** with
  title/unit parent context repeated in every block.  Those blocks receive
  candidates from three label-blind sources: BM25-style lexical overlap,
  frozen BQB dense similarity, and legal structure/citation signals.
- At most four blocks per source are unioned (at most 12 proposals), then a
  deterministic MMR pass selects three blocks; it opens a fourth only when
  the next evidence score is within the preregistered uncertainty margin.
- The existing 760M Vietnamese reranker is frozen.  It scores the selected
  passages only; E1 does not train, calibrate, select a threshold, or promote
  a policy.
- The complement reference is loaded solely after the manifest is sealed, to
  calculate Recall@5 and transition diagnostics.  Holdout and public artifacts
  are rejected by the runner.
- The manifest fingerprints the raw corpus, audit artifacts, BQB checkpoint
  and reranker tokenizer.  Frozen-score output separately fingerprints the
  exact reranker weights, so an in-place model replacement cannot silently
  reuse an incompatible evidence/cache artifact.

## Runtime phases

`PHASE=materialize` needs a GPU when BQB dense proposals are used.  It encodes
the unique c35 evidence blocks and all complement queries.  `PHASE=score` is
also GPU work: it runs frozen reranker inference for at most
`1197 × 35 × 4 = 167,580` query-passage pairs.  `PHASE=all` performs both.

On the RTX PRO 6000 96GB server, start with the launcher defaults (`BQB=128`,
`reranker=16`).  Increase only the reranker batch after the first stable
progress output; artifact contents are deterministic regardless of batch size.

## Promotion boundary

E1 is useful only if its complement report shows a genuine gain over the V16b
anchor and the c35 oracle still leaves headroom.  Only then should E2 be
implemented: MIL aggregation and fine-tuning with relevance plus boundary
loss, with a fresh label-separation protocol.  Do not use E1 output directly
as a public submission.
