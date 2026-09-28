# R1 C35 Long Fusion Protocol

R1 combines two fold-safe E3B E2 MIL score exports with a frozen Qwen
reranker score over the canonical c35. It keeps V16b's top four and changes
only slot five. There is no new neural training and no generator/verifier.

For a challenger `d` and V16b incumbent `i`, E3B exports are normalized by
the original adapter. Let `h_E` be the mean of the two E2 normalized gaps and
`p_E = 0.25 * abs(gap_1-gap_2)`. Qwen long scores are median/IQR normalized
within that same c35, with a finite zero/spread guard. R1 evaluates:

```
g_R1(d) = (1 - lambda) * h_E(d) + lambda * (z_F(d)-z_F(i)) - p_E(d)
```

The locked lambda choices are `0`, `0.25`, and `0.50`. Lambda zero delegates
to the original E3B fixed policy (`memory_weight=0`, threshold `0.50`) so
baseline parity is exact. For nonzero lambda the same threshold `0.50` and
canonical c35 tie-break are used. Policy selection happens only on C; V is
read once. Standard must pass for both inner seeds before stress is allowed.

The Qwen cache is label-blind and uses the existing F scorer's long prompt,
yes/no logit difference, legal-boundary packer and 4096-token budget. It has
no c67/shortlist-8 shortcut. Input hash, model fingerprint/revision, packer
lineage, QID shard, token metadata and runtime probe are recorded.

R1 requires the E3B score artifacts named by each standard/stress report:
two neural seeds × calibration/validation for every outer fold. Deployment
checkpoints alone cannot be used as OOF substitutes. All score artifacts are
hash-checked against their E3B report lineage before qrels are loaded.

The report records `new_policy_holdout_validated:false` and
`stress_seen_during_design:true`. A passing research gate does not mutate the
old E3B/E4/E5 release; any public/final release requires a separately bound
release contract and a fresh public label-free input manifest.
