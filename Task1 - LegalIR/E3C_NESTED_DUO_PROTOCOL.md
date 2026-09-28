# E3-C nested duo LoRA protocol

E3-C is a complement-only research branch. It does not replace the E3-B policy
already locked for E4, and cannot authorize an E4 parity run, submission, or
holdout access.

For every `(regime, inner seed, outer fold)`, E3-C uses the retained E3-B
student with neural seed `20260908` from the same partition. It then trains a
rank-8 LoRA comparator on **A-fit** only. Its text input is the query, a c35
challenger's frozen E1 evidence, and the incumbent V16b slot-five evidence.
Every c35 challenger other than frozen top-four and incumbent is scored; there
is no prior nomination stage. EMPTY_BODY sides are explicit, unscoreable rows
that default to V16b abstention.

The per-pair target is `delta=(I[c in G]-I[i in G])/|G|`, transformed to
`(1+delta)/2` for BCE-with-logits. Pair weights sum to one per query. The
reported value is `2*sigmoid(logit)-1`. A challenger must strictly exceed a
policy margin; otherwise slot five remains V16b. Top four, c35 membership, and
five unique IDs are invariant checks.

Nested split: A-fit trains; A-stop early-stops by BCE; C alone chooses from
`{0,.01,.02,.05,.10}` plus abstention; V alone is evaluated. No C/V labels are
used in model fitting. Standard comprises 2 inner seeds × 5 outer folds = 10
new comparator fits. Stress comprises 3 regimes × 2 × 5 = 30 fits and is
blocked until the standard gate passes.

The runtime uses BF16, max 1536 tokens, LoRA `r=8`, `alpha=16`, dropout `.05`,
and at most two epochs. Before a GPU run, auto-probe executes a real BF16
forward/backward/optimizer step on an evidence-heavy duo. The chosen batch is
recorded; gradient accumulation is never silently altered. Training checkpoint
files retain adapter, optimizer, RNG, epoch and position and can resume.

All pair scoring, BCE, calibration-policy metrics, OOF metrics and promotion
gates run as CUDA tensor operations. File parsing and tokenizer construction
are inherently host-side, so E3-C caps OpenMP/BLAS/Torch CPU pools at one
thread (`CPU_THREADS=1` by default) rather than letting them saturate the host.

Preflight fails closed if the exact retained E3-B weight or full-c35 C/V export
is absent or hash-mismatched. Do not substitute the base reranker checkpoint:
that would violate the same-fold source condition.

The complete inference budget remains approximately 3.37B parameters
(`2.61B` E3-B + `0.76B` comparator), below 4B. E3-C reports deltas both versus
V16b and fixed E3-B `tau=.5, mem_w=0`, but is exploratory; no result promotes
to holdout or changes E4 without a separately approved protocol.
