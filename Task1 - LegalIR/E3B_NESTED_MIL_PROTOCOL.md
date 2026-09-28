# E3-B — full-c35 MIL residual gate

E3-B is a new complement-only experiment after E3-A failed its safety gate. It
does not authorize E4, a holdout read, public inference, or a replacement for
the locked V16b submission.

## Fixed decision contract

- E0 V16b remains the anchor. Its first four IDs must be copied exactly.
- E3-B can replace only slot 5, using an ID already in the frozen c35. Outputs
  must contain five distinct IDs. Abstention returns V16b exactly.
- Two freshly fitted E2 MIL students score every eligible c35 candidate. Each
  score export contains the document temperature-logsumexp score and all
  passage logits; an EMPTY_BODY record is a deliberate null score and cannot
  be proposed.
- Per seed, scores are normalised within query by median/IQR over valid c35
  evidence rows. For candidate `c` and V16b incumbent `i`, E3-B uses
  `mean_gap(c) - 0.25 * abs(gap_1(c)-gap_2(c)) + lambda*(m(c)-m(i))`.
  A candidate needs `mean_gap > 0` and utility strictly above `tau`.
- The initial policy grid is lambda in `{0, 0.10}`, tau in
  `{0, 0.25, 0.50, 1.00}`, plus abstain. If V16b memory provenance does not
  demonstrate zero query-ID overlap and no complement query-label retrieval,
  lambda `0.10` is automatically excluded.

## Nested dependency separation

For each regime, outer fold and inner seed:

1. Outer group split produces `T` and unseen `V`.
2. A group-disjoint 20% of `T` becomes E3 calibration `C`; the remainder is
   `A`.
3. A group-disjoint 15% of `A` is E2 early-stop `A-stop`; the rest is
   `A-fit`.
4. Each of two neural seeds starts from the original local reranker checkpoint,
   trains only on `A-fit`, and selects epoch only from `A-stop`.
5. The same frozen two weights score all c35 rows in `C` and `V`. The policy is
   selected using C labels and applied once to V.

Every completed fold records exact QID/group lists and hashes, two C score
artifacts, two V score artifacts, weight-state hashes, selected policy and V
predictions. A resume rejects any changed data/model/code/runtime contract.

## GPU scheduling and auto-probe

E3-B keeps the E2 loss, FP32 master weights, FP16 autocast, GradScaler,
AdamW, max length 768, negative mining and configured gradient accumulation.
Before a logical run, the auto-probe evaluates micro-batches `1,2,4,8,16` and
score batches `16,32,64,128,256` in fresh subprocesses. It uses the heaviest
encoded training queries, two complete optimizer updates per training trial,
and a shared-GPU reserve of `max(2 GiB, 15% of observed free VRAM)`. It never
assumes exclusive use of an A100 80GB. If no candidate fits, the job stops;
it does not silently change optimizer, quantization, precision, passages or
token length.

`MODE=standard` is exactly 5 outer folds x 2 inner seeds x 2 neural seeds =
20 E2 fits. Stress runs are blocked until standard passes, then execute the
same 20-fit schedule for each of three regimes = 60 E2 fits. CPU-only tests
and preflight do not count as neural fits.

## Gates

For each inner seed, standard and every stress regime independently require:

1. macro Recall@5 gain over V16b strictly positive; and
2. successful rescues strictly greater than destructive changes.

Reports also include a paired group-bootstrap interval for the Recall delta.
The eight regime/seed views share complement data and are diagnostic transfer
checks, not eight independent test sets. Only a fully passed standard gate
permits stress allocation. Only every stress gate passing can authorize the
separate, one-time E4 holdout protocol; this script never opens it itself.
