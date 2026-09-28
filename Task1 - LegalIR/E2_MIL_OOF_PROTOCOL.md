# E2 — MIL relevance/boundary CE fine-tuning

E2 is a complement-only, OOF experiment downstream of the sealed E1 evidence
manifest.  It is not a submission run.

- Candidate universe stays fixed at c35.
- A document is represented by the E1-selected three/four evidence passages;
  their score is a temperature-logsumexp MIL aggregate rather than a presumed
  positive passage label.
- Each outer fold reloads the 760M Vietnamese reranker from its original local
  checkpoint.  The fold's validation qrels are never used for its training,
  epoch selection, calibration or thresholding.
- Within the outer-train partition, a group-disjoint inner partition controls
  early stopping.  The first GPU run is `MODE=standard`; only a viable result
  earns the much more expensive four-regime OOF run.
- Each completed outer fold is saved atomically with its validation predictions
  and fold report.  A resumed run accepts those folds only when the complete
  data/model/script/split/optimization contract matches exactly; the current
  in-progress fold is deliberately retrained.
- The training objective is `0.55 relevance + 0.30 boundary + 0.15 anchor`:
  relevance is document qrel BCE, boundary is pairwise positive/negative loss
  weighted by frozen c35 distance to rank five, and anchor is a light score
  regularizer applied only to V16b-anchor documents against their E1 frozen CE
  scores.  It is a stabilizer, not a rule that freezes anchor membership.

The E2 report remains OOF diagnostic output.  A later E3 will be the only
stage permitted to choose a residual top-5 policy that combines V16b memory
and E2 relevance, and it must use another inner/outer separation.
