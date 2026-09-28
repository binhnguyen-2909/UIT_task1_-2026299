# PV1 Pairwise Abstain Protocol

PV1 is a complement-only verifier experiment. It preserves V16b ranks 1–4,
compares the frozen incumbent at rank 5 with up to three F-shortlist
challengers, and defaults to the V16b answer unless both document orders give
the challenger the sole-evidence verdict.

## Input contract

`F_SHORTLIST` is a JSONL file with exactly one row per complement QID:

```json
{"qid":"123","shortlist":[{"doc_id":"d1","f_rank":1,"f_score":2.3}],"packed_documents":[{"doc_id":"d1","packed_text":"[F packed evidence]"}]}
```

Every row has exactly eight unique shortlist entries. `packed_documents` must
include every shortlisted document and the V16b incumbent when it is not in
the shortlist. `F_MANIFEST` contains a `lineage` object with exact
`anchor_sha256`, `stage1_sha256`, `split: "complement"`,
`packer_sha256`, and optionally `shortlist_sha256`.

The preflight refuses a cache that is not a c67 subset, loses an incumbent
pack, changes the c35 prefix, or has a hash mismatch. It writes its pair
manifest before reading complement qrels for the diagnostic audit.

## GPU work only

`MODE=pilot` and `MODE=cache` load Qwen3-1.7B on CUDA. The scorer uses BF16,
Qwen non-thinking chat generation, direct first-token logits for A/B/C/D,
left padding, `use_cache=False`, and `logits_to_keep=1`. It refuses overflow
instead of truncating evidence. Cache rows are atomic/resumable per complete
QID and each row includes both text/prompt/model provenance hashes.

`MODE=standard`, `MODE=stress`, and `MODE=finalize` consume frozen cache and
are deliberately lightweight. They do not train a model. Standard chooses
only `tau=0`, `tau=ln(3)`, or abstain on C; it evaluates one chosen policy on
V. Stress is blocked until standard passes the exact bound gate source.

## Safety and release

PV1 never opens holdout, does not use public labels, does not change E/F's
failed status, and cannot make a submission. A passing stress report merely
permits freezing a complement final policy. A separate label-free public
input/pair/cache contract is required before any public inference.

The actual model parameter count is read on GPU and combined with V16b and
the E/F reranker. The cache run aborts at `>= 4,000,000,000` parameters;
quantization cannot bypass that check.
