# V25-SV Preregistration — BGE-Gemma Semantic Swap Verifier

**Status:** proposed confirmatory protocol; training is forbidden until every preflight assertion below passes and the resolved configuration is sealed.

**Single objective:** produce one frozen candidate capable of improving the locked V16b holdout, with a public-stage target of Recall@5 ≥ 0.93. This protocol does not open K32/K64, does not revive V23-A/V23-B, and does not stack any exploratory V23 gain.

## 1. Decision and falsifiable hypothesis

V23-B0 proved that the c35 edit beam contains enough oracle headroom (+0.05347), but frozen E3B/small-head features cannot separate helpful edits from harmful edits at the required precision: 27 improved / 7 harmed / 174 neutral, precision 0.794, with a negative standard-regime fold.

V25-SV changes the representation family, not the threshold. It uses a 2.51B multilingual LLM cross-encoder to read the Vietnamese legal query and raw evidence from the incumbent and proposed documents. The model predicts the exact utility class of the **whole edit**:

- `IMPROVE`: proposed top-5 has higher Recall@5;
- `TIE`: unchanged Recall@5;
- `HARM`: proposed top-5 has lower Recall@5.

The hypothesis is falsified if a one-shot grouped OOF run cannot produce at least +0.012 Recall gain with intervention precision ≥0.90 and stable regime transfer. A failure closes this semantic-verifier recipe; it is not followed by a learning-rate, threshold, prompt, context-length, or LoRA sweep.

## 2. Frozen scope

- Anchor: V16b, public 0.9210, locked holdout 0.9228333, deployment ledger 1.09001B.
- Development labels: organizer-provided labels only.
- Candidate pool: existing c35 only.
- Output: exactly five unique `document_id` values.
- Immutable ranks: top 1–3 byte-identical to V16b.
- Mutable ranks: slots 4–5 only; zero, one, or two replacements.
- Default action: identity/no-change.
- Holdout, public labels, K32, K64 and submission remain closed until the complete OOF contract passes.
- V23 exploratory weights, thresholds and projections are not reused.
- V23-B0's edit identities and exact qrel-derived ΔRecall labels may be reused as deterministic training records; E3B scores are not model inputs.

## 3. Semantic model and parameter ledger

**Base checkpoint:** `BAAI/bge-reranker-v2-gemma`, resolved to a concrete repository commit SHA before any preprocessing or training. The literal revision `main` is forbidden in a run.

Why this model:

- it is a multilingual LLM cross-encoder designed for reranking;
- official BGE documentation reports 2.51B parameters;
- it supports instruction-style relevance scoring and official LoRA fine-tuning;
- it uses most of the remaining parameter budget while keeping the complete pipeline below 4B.

Deployment ledger:

| Component | Unique loaded parameters |
|---|---:|
| Frozen V16b pipeline | 1.09001B |
| BGE-Gemma semantic verifier | 2.51B |
| LoRA adapters + 3-class head | ≤0.03B |
| **Worst-case total** | **≤3.63001B** |

No second semantic model, teacher model, retriever, ensemble member or sidecar may be loaded at final inference.

## 4. One-source-of-truth configuration

The executable must load one `RESOLVED_CONFIG.json`. The preregistration report, preprocessing, model constructor, trainer and evaluator must read the same object. No duplicated constants are allowed.

Before preprocessing, write and hash:

- base model repository commit SHA;
- tokenizer commit SHA;
- tokenizer vocabulary hash;
- model config hash and exact parameter count;
- LoRA target module names and tensor shapes;
- maximum sequence length and evidence budgets;
- V16b anchor checksum;
- query, document, qrel, c35 and fold-manifest checksums;
- software versions, deterministic flags and seed.

Mandatory hard assertions:

1. Runtime values equal every resolved value.
2. Actual model graph, LoRA tensors and 3-class head shapes equal the recorded shapes.
3. Parameter ledger is below 4B.
4. Every query has an identity edit.
5. Every edit references only organizer documents and c35 candidates.
6. All variants of a query map to one outer fold.
7. Any mismatch aborts before the first forward pass.

## 5. Independent semantic evidence pack

This is not an E3B-score continuation. It builds a new raw-text pack with the frozen base BGE-Gemma checkpoint.

1. Split every candidate document into 192-token blocks with 48-token overlap, preserving deterministic document order and block IDs.
2. Score each `(query, block)` with the frozen base checkpoint using a fixed Vietnamese legal relevance instruction.
3. For each document, retain the two highest-scoring non-overlapping blocks; ties break by earliest block ID.
4. Cache raw block text, IDs, token counts and base logits. Cache completeness is a bijection contract; missing keys abort.
5. Base block selection is computed before LoRA training and remains frozen in every fold and at final inference.

Each document receives exactly 192 evidence tokens in the verifier input: take the first 96 tokens from each retained block; if only one block exists, take up to 192 tokens from it. No dynamic reallocation between documents is allowed.

Verifier input has a fixed 1024-token ceiling:

- instruction and structural tokens: ≤64;
- query: ≤192;
- original slots 4–5 evidence: ≤384 total;
- proposed slots 4–5 evidence: ≤384 total.

Truncation is deterministic from the end of the lowest-scoring retained block. Query text is never truncated before document evidence.

## 6. Edit examples and leakage control

Start from the deterministic V23-B0 beam of identity, one-swap and two-swap edits. Recompute ΔRecall@5 independently from qrels and fail if any stored V23-B0 label disagrees.

For each source query and each training fold, choose at most:

- two `IMPROVE` edits;
- two `HARM` edits;
- two `TIE` edits.

Selection is deterministic by SHA256 of `(seed, qid, edit_id)` after preferring the smallest absolute V16b rank margin within each class. If a class is absent, do not backfill it with another class.

Every example is duplicated with original/proposed order reversed:

- `IMPROVE` becomes `HARM`;
- `HARM` becomes `IMPROVE`;
- `TIE` remains `TIE`.

All examples from a query remain in the same fit, calibration or evaluation partition. The total loss weight of each source query is exactly 1, irrespective of its number of edits. Synthetic variants never increase the effective query count.

## 7. Model and fixed training recipe

Architecture:

- BGE-Gemma base checkpoint;
- LoRA on `q_proj`, `k_proj`, `v_proj`, `o_proj`;
- final hidden state connected to one 3-class linear head;
- no E3B vectors, V23 projections or hand-built regime heads.

Fixed optimization:

| Field | Value |
|---|---:|
| Precision | BF16 |
| Epochs | 1 |
| LoRA rank / alpha / dropout | 32 / 64 / 0.05 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Warm-up ratio | 0.05 |
| Effective batch size | 32 queries |
| Max gradient norm | 1.0 |
| Seed | 20260901 |
| Checkpoint selection | final step only |

Loss is query-normalized weighted cross-entropy with class costs:

- `IMPROVE`: 1.0;
- `TIE`: 0.5;
- `HARM`: 2.0.

No focal-loss variant, auxiliary loss, epoch selection, learning-rate search, prompt search or checkpoint selection is permitted.

## 8. Cross-fitting and policy construction

Use the four existing grouped regimes and five outer folds per regime. For each outer fold:

1. Split outer-train queries deterministically into 80% fit and 20% calibration by group hash.
2. Train one LoRA adapter on fit queries only.
3. Score every candidate edit for calibration and outer-evaluation queries.
4. Define `gain_score = P(IMPROVE) - P(HARM)`.
5. On calibration only, choose the lowest gain threshold whose accepted non-tie edits have empirical precision ≥0.90, at least five improvements, and no more than one harmed query. If no threshold exists, that fold must abstain completely.
6. For each evaluation query, choose the accepted edit with highest gain score; ties prefer fewer swaps and then identity.

The four regimes are independent evaluations of one architecture. They are not four inference heads and do not vote or veto. The final full-fit model uses the standard-grouped OOF calibration threshold fixed before holdout.

## 9. Confirmatory OOF contract

All conditions must pass simultaneously:

| Metric | Required |
|---|---:|
| Recall gain vs V16b | ≥ +0.012 |
| Robust gain | ≥ +0.010 |
| Paired bootstrap lower 95% | ≥ +0.004 |
| Improved / (improved + harmed) | ≥ 0.90 |
| Net query-equivalents | ≥ +15 |
| Neutral / improved accepted edits | ≤ 4.0 |
| Aggregate gain in each of four regimes | ≥ 0 |
| Worst outer-fold gain | ≥ −0.002 |
| Top 1–3 byte-identical | yes |
| Changed slots | 4–5 only |
| Output cardinality | exactly 5 unique IDs |

Bootstrap unit is query, stratified by outer fold. Candidate edits and symmetric examples are not bootstrap units.

Failure of any item closes V25-SV. It does not authorize another threshold, prompt, seed, evidence length, LoRA rank or base checkpoint.

## 10. Full fit and the single holdout opening

Only after the confirmatory OOF contract passes:

1. Seal OOF report, predictions, thresholds, manifests, code commit and environment lockfile.
2. Train one full-fit adapter for exactly one epoch using the identical recipe.
3. Freeze the pooled standard-grouped threshold derived from OOF; do not recalibrate on holdout.
4. Run the locked holdout exactly once.

Holdout success criteria:

- Recall@5 strictly above V16b holdout 0.9228333;
- target Recall@5 ≥0.9300;
- exactly five unique IDs per query;
- top 1–3 unchanged;
- no post-holdout modification or selective submission.

If holdout does not improve, close V25-SV and retain V16b. Public submission is authorized only if holdout improves; the public recipe must be byte-identical to the holdout recipe.

## 11. Run order

1. Implement the resolved-config loader and runtime assertions.
2. Resolve and seal the model/tokenizer commit SHAs.
3. Recompute and verify V23-B0 edit labels.
4. Build the independent BGE-Gemma raw evidence manifest.
5. Run a two-query dry run solely to validate shapes, determinism and parameter count; no metric inspection.
6. Execute all frozen outer folds once.
7. Produce the confirmatory contract report.
8. Stop on failure; otherwise full-fit and open holdout once.

## 12. Why this is the one next step

Candidate coverage is not the limiting factor: V23-B0's edit oracle gain is +0.05347. The limiting factor is semantic discrimination between a helpful and a harmful change. A 2.51B raw-text multilingual cross-encoder is the largest single verifier that fits beside V16b under the 4B ceiling. It directly attacks that missing information and removes the small-head/E3B representation bottleneck.

No protocol can guarantee a holdout increase before opening the holdout. This preregistration maximizes the chance of a real transfer by requiring a substantially larger OOF margin than the desired minimum gain, blocking adaptive retuning, and authorizing exactly one holdout evaluation.

## Primary sources

- BGE documentation, model size and multilingual reranking: https://bge-model.com/tutorial/5_Reranking/5.2.html
- Official BGE-Gemma model card and LoRA fine-tuning recipe: https://huggingface.co/BAAI/bge-reranker-v2-gemma
- BGE/FlagEmbedding official repository: https://github.com/FlagOpen/FlagEmbedding
- Selective classification via a predictor/rejector: https://proceedings.mlr.press/v237/mao24a.html
