# V25-SV server execution specification

This is a concise server-run companion to the authoritative
`V25_BGE_Gemma_Semantic_Swap_Verifier_PREREGISTRATION.md`. It is reconstructed
from the local resolved configuration; `RESOLVED_CONFIG.local-reference.json`
remains the audit reference.

## Frozen scope

- Anchor: V16b; top 1–3 are byte-identical and only slots 4–5 may change.
- Candidate pool: existing c35 only; identity/no-change is the default.
- Development scope: clean complement only. Public labels and locked holdout
  remain closed through confirmatory OOF.
- Base model: `BAAI/bge-reranker-v2-gemma` revision
  `1787044f8b6fb740a9de4557c3a12377f84d9e17`.
- Deployment ledger: V16b 1.09001B + BGE-Gemma 2,506,172,416 + LoRA/head
  7,378,944 = 3,603,561,360 parameters, below 4B.

## Raw evidence

- Split each candidate document into 192-token blocks with 48-token overlap.
- Score each `(query, block)` with frozen BGE-Gemma and the fixed Vietnamese
  legal-relevance instruction in `RESOLVED_CONFIG.json`.
- Retain two highest-scoring non-overlapping blocks; ties use earliest block
  ID. Keep 96 tokens from each retained block, 192 evidence tokens/document.
- Verifier inputs have a 1024-token ceiling: ≤64 control, ≤192 query, 384
  original-evidence and 384 proposed-evidence tokens.

## Verifier and OOF

- BF16 BGE-Gemma with LoRA rank 32, alpha 64, dropout 0.05 on q/k/v/o.
- Three labels: `IMPROVE`, `TIE`, `HARM`; reverse each selected edit and swap
  IMPROVE/HARM.
- One epoch, learning rate 2e-5, weight decay 0.01, warmup ratio 0.05,
  effective batch 32 queries, max gradient norm 1.0, seed 20260901.
- Four grouped regimes and five outer folds. Per outer fold, split groups 80/20
  into fit/calibration. A calibration threshold needs precision ≥0.90, at
  least five improvements and at most one harm; otherwise the fold abstains.

## OOF gates before any holdout action

- Recall gain ≥ +0.012; robust gain ≥ +0.010; bootstrap lower 95% ≥ +0.004.
- Intervention precision ≥0.90; net query-equivalents ≥15; neutral selected
  edits ≤4.
- All four regimes have non-negative aggregate gain; worst outer fold ≥−0.002.

Failure closes V25-SV: do not tune the threshold, prompt, evidence length,
LoRA rank, base checkpoint or open holdout.
