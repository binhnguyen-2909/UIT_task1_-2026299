# V16b scoring and reverse-pipeline forensic audit

**Date:** 2026-09-01  
**Submission:** `v16b_memory_slot5.zip`, SHA-256 `b2816bd73ae23fbdfadd6563dc43728896f078a6ad8032b6743984b3aadadfd9`  
**Exact labels available:** locked holdout 1,000 queries  
**Public labels available:** no

## 1. Bottom line

The official scorer does not contain public gold and cannot reveal whether each public ID is correct. It loads a secret reference from `/app/input/ref/metadata.json`, which is absent from the supplied package, and writes only aggregate Recall and Precision.

The supplied artifact bundle is nevertheless sufficient to produce an exact per-query and per-ID audit on the 1,000-query locked holdout and a structural-risk audit for all 5,000 public prediction slots.

The main conclusion is unambiguous:

- For the next target of 0.93, the dominant bottleneck is **ranking/selection inside c35**, not candidate generation.
- V16b still has `+0.044833` Recall headroom inside its existing c35 but needs only `+0.007167` to reach 0.93.
- Long documents, missing titles, numeric/penalty questions, and semantic-memory collisions explain a material share of the remaining misses.
- The public submission contains three empty-passage IDs. One of them was newly inserted by the V16b memory gate despite having neural rank 20. This is a concrete data/policy failure, not a hypothetical concern.

## 2. What the BTC scorer actually does

For every query, the scorer computes:

- `Recall_q = |set(gold) ∩ set(prediction)| / |gold|`;
- `Precision_q = |set(gold) ∩ set(prediction)| / len(prediction)`;
- both values are zero when the prediction list is empty or has more than five IDs;
- final values are macro averages over queries.

Important details:

- document order does not affect the score;
- duplicate IDs do not increase the numerator but still increase the Precision denominator;
- prediction IDs and gold IDs must have the same Python type;
- the code checks equal query counts, then effectively requires matching QID sets;
- it has no per-query or per-ID diagnostic output.

The V16b ZIP passes every submission-format invariant: exactly one `submission.json`, 1,000 correct public QIDs, five unique string IDs per query, and every ID exists in the organizer corpus.

## 3. Exact locked-holdout waterfall

| Layer | Recall@5 | Precision@5 | Relevant ID hits | Perfect queries | Zero-hit queries | Partial queries |
|---|---:|---:|---:|---:|---:|---:|
| Stage-1 top 5 | 0.827917 | 0.1764 | 882 | 809 | 153 | 38 |
| Stage-2 c35/p3 | 0.900583 | 0.1922 | 961 | 884 | 83 | 33 |
| V11h LTR | 0.918833 | 0.1960 | 980 | 904 | 66 | 30 |
| V16b memory slot 5 | 0.922833 | 0.1970 | 985 inferred from Precision | not persisted | not persisted | not persisted |

The holdout has 1,086 qrel edges: 924 queries have one gold, 67 have two, eight have three, and one has four.

### Transition utility

| Transition | Changed | Improved | Harmed | Neutral | Net query equivalents | Recall gain |
|---|---:|---:|---:|---:|---:|---:|
| Stage 1 → Stage 2 | 997 | 91 | 14 | 895 | +72.667 | +0.072667 |
| Stage 2 → V11h | 961 | 30 | 11 | 959 | +18.250 | +0.018250 |
| V11h → V16b | 177 | 9 rescues | 4 destructive | 164 | +4.000 | +0.004000 |

Only 13 of the 177 V16b changes alter Recall. Among outcome-changing interventions, precision is `9/13 = 69.2%`; among all changes, only `9/177 = 5.1%` are beneficial. V16b succeeds because harm is rare, but the gate spends most interventions on neutral popular documents.

## 4. Where V11h still loses Recall

| Final V11h query state | Queries |
|---|---:|
| Perfect | 904 |
| All gold present in c35 but all missed by top 5 | 42 |
| All gold absent from c35 | 24 |
| Partial/mixed | 30 |

At the gold-edge level:

- 980 gold edges are returned by V11h;
- 63 gold edges are inside c35 but missed by V11h;
- 43 gold edges are absent from c35.

Macro Recall decomposition:

| Error source | Lost Recall | Share of V11h's remaining error |
|---|---:|---:|
| Gold absent from c35 | 0.032333 | 39.8% |
| Gold present in c35 but outside final top 5 | 0.048833 | 60.2% |
| Total remaining error | 0.081167 | 100% |

After V16b, c35 ranking headroom remains about `0.967667 − 0.922833 = 0.044833`. Reaching 0.93 requires converting only about 7.17 query equivalents, roughly 16% of that ranking headroom.

Candidate oracle confirms the same diagnosis:

| Candidate cutoff | Oracle Recall |
|---|---:|
| 5 | 0.827917 |
| 20 | 0.948667 |
| 35 | 0.967667 |
| 50 | 0.971750 |
| 80 | 0.981250 |
| 100 | 0.984000 |
| 200 | 0.986500 |

Opening c35 to a larger pool can matter later, especially for 0.96, but it is not required to cross 0.93.

## 5. Stage-2 and LTR failure signatures

Among the 63 c35 gold edges missed by V11h:

- 18 were already in the original Stage-1 top 5;
- 12 were in the neural top 5;
- 11 were returned by Stage 2 and then removed by V11h;
- 28 belong to documents longer than 20,000 whitespace tokens;
- seven have no title;
- eight are associated with numeric/time questions;
- two are empty-body gold edges.

Gold neural rank within c35 has median 1, p90 8, p95 12, and maximum 29. A bounded reranker focused on the top semantic/legal evidence remains sufficient; exhaustive corpus-wide scoring is unnecessary.

### Document-condition hit rates

These are edge-level hit rates, not macro query Recall.

| Gold document condition | Edges | c35 contains gold | Stage 2 hit | V11h hit |
|---|---:|---:|---:|---:|
| ≤2k tokens | 52 | 0.8846 | 0.7500 | 0.7500 |
| 2k–5k | 137 | 0.9343 | 0.8759 | 0.8905 |
| 5k–20k | 491 | 0.9593 | 0.9063 | 0.9185 |
| >20k | 404 | 0.9802 | 0.8837 | 0.9109 |
| Missing title | 74 | 0.9865 | 0.9189 | 0.8919 |
| Title present | 1,012 | 0.9585 | 0.8824 | 0.9032 |
| ≥10% repeated paragraph text | 200 | 0.9500 | 0.8600 | 0.9000 |
| Empty body | 2 | 1.0000 | 0.0000 | 0.0000 |

The strongest design signals are:

1. c35 already finds long-document golds, but three-passage evidence/reranking fails to promote them;
2. missing-title golds are found extremely well by Stage 1, then V11h LTR reduces their hit rate from 0.9189 to 0.8919;
3. exact empty passages enter c35 but cannot provide semantic evidence;
4. short documents have a separate candidate-generation weakness.

### Heuristic question families

| Family | Queries | Stage 2 | V11h | c35 oracle | Diagnosis |
|---|---:|---:|---:|---:|---|
| `bao_nhieu` | 40 | 0.8417 | 0.8083 | 0.8875 | LTR actively harms numeric questions |
| `muc_phat` | 40 | 0.8125 | 0.8875 | 0.9875 | Large selection headroom remains |
| `la_gi` | 46 | 0.8804 | 0.9022 | 1.0000 | Pure ranking bottleneck |
| `truong_hop` | 41 | 0.9756 | 0.9512 | 0.9878 | LTR harms an already strong Stage 2 |
| `thoi_han` | 59 | 0.9407 | 0.9576 | 0.9831 | Comparatively healthy |
| `ho_so` | 84 | 0.8929 | 0.9464 | 0.9643 | LTR is useful |

These families use a deterministic first-match keyword heuristic and are diagnostic rather than new labels.

## 6. Train-noise effect on the locked holdout

The holdout itself contains three known dirty cases from the official train corpus:

- `42298`: one valid gold plus empty doc `288457`;
- `159914`: a gold in an exact-passage duplicate family;
- `163826`: its only gold is empty doc `131890`.

For a diagnostic clean view, excluding `159914` and `163826` and dropping only empty edge `288457` from `42298` raises V11h from `0.918833` to `0.920675` over 998 queries. This is not a promotion score; it quantifies how train-only corruption depresses and distorts local evaluation.

The exact V16b clean score cannot be reconstructed from this package because its final holdout prediction JSON is absent.

## 7. Public submission structural audit

The public file has no gold, so the following are input-structural facts rather than TP/FP labels.

| Finding | Count |
|---|---:|
| Prediction slots | 5,000 |
| V16b changes from V11h | 165 queries |
| Empty-body outputs | 3 slots / 3 queries |
| Exact-duplicate documents co-returned | 4 slots / 2 queries |
| Canonical source siblings co-returned | 30 slots / 15 queries |
| Missing-title outputs | 471 slots |

Missing title and source siblings are risk flags, not proof of incorrectness. Empty-body outputs are incompatible with the supplied organizer statement that public/private relevant passages are clean.

### Empty-body public outputs

| QID | Rank | Question | Empty doc | Origin |
|---|---:|---|---:|---|
| `30122` | 2 | Khái niệm về điện não số hóa như thế nào? | `288457` | Already in V11h; URL is the Health Insurance Law |
| `91810` | 4 | Có được phép đặt vó đánh bắt cá trên sông hay không? | `263763` | Already in V11h; URL is industrial-explosives safety |
| `58504` | 5 | Chẩn đoán bệnh tụ huyết trùng gia cầm ở gà dùng thiết bị gì? | `131890` | Newly inserted by V16b memory; URL is infectious bronchitis |

The third case reveals the exact memory failure:

- Stage-1 rank: 5;
- frozen neural rank: 20;
- tiny neural score: `0.0002288`;
- memory rank: 1;
- degree-corrected memory score: `0.46165`;
- V16b replaced valid-body V11h slot-5 doc `291854` with empty doc `131890`.

The memory signal overrode absent content evidence because empty-body state was not a hard veto.

### Exact duplicate co-outputs

- QID `1392` returns both `84226` and `121575`, byte-identical copies of the same Population Law page.
- QID `9578` returns both `277743` and `35337`, byte-identical copies of the same minimum-wage decree.

Because the scorer uses exact IDs, automatically dropping a duplicate sibling is not guaranteed safe unless the organizer's clean-test contract specifies canonical identity. They should be handled as a family during training/negative mining, while inference changes require a preregistered identity rule.

### V16b popularity bias

The most frequently inserted V16b documents are broad, high-degree legal documents: `81598` seven times, `23402` six, `21398` five, and `42262`/`46918` four each. Many insertions are semantically plausible, but the holdout transition shows that 164/177 changes are neutral. The current degree correction reduces but does not eliminate generic-document attraction.

## 8. Revised bottleneck after V21–V23

The subsequent experiments rule out two earlier hypotheses:

- E3B is complete; missing cache coverage is not the issue.
- Edit capacity is not the issue: V23-B0 has oracle gain +0.05347.

The repeated failure is **rare-rescue supervision plus insufficient semantic harm rejection**:

| Branch | Signal |
|---|---|
| V16b | 177 edits, 9 rescue, 4 destructive, 164 neutral |
| V22-A | +0.001671 Recall; 2 rescue; bootstrap lower bound 0 |
| V22-B K16 | +0.001672 Recall; 1 rescue; inner folds can contain only one rescue |
| V23-A | +0.00627 Recall, but precision 0.579, bootstrap lower bound −0.00501 and transfer folds negative |
| V23-B0 | net +20 and oracle +0.05347, but precision 0.794 and standard fold −0.00625 |

Training another small head on rescue/non-rescue labels therefore repeats the same statistical failure. The next model must learn dense query–document relevance from all fold-safe qrels and use rescue only as an evaluation outcome.

### Recommended next branch: V27 Evidence-First Boundary Verifier

V27-A adds an approximately 0.1B multilingual MiniLM cross-encoder, trained with relevance positives and within-query hard negatives. It scores at most four evidence blocks per document, freezes top 1–4 and may replace only slot 5 after OOF calibration. EMPTY_BODY is a hard veto.

This directly targets the two measured errors:

1. long documents whose gold is present in c35 but whose evidence is missed by current three-passage scoring;
2. plausible high-degree challengers that memory/E3B cannot distinguish from truly relevant legal evidence.

The local 0.93 target needs only +0.007167 from V16b. V27-A is preregistered for +0.010 OOF, robust gain ≥+0.008, bootstrap 95% lower bound ≥+0.003, all four regimes non-negative, net ≥+12, outcome-changing precision ≥0.90 and at most two destructive changes. Holdout promotion requires +0.008, which would raise 0.922833 to at least 0.930833.

For 0.95, do not expand candidates immediately. First widen the proven verifier to a listwise slots-4–5 policy inside c35 and optionally distill V25 teacher logits into the same small student. Only after this lane is exhausted should candidate-absent queries open c80. c20 oracle is 0.948667 and therefore cannot support a 0.95 target, while c35 oracle 0.967667 can.

The complete execution order and failure contract are in V27_EVIDENCE_BOUNDARY_VERIFIER_ROADMAP.md and V27_PREREGISTRATION.json.

## 9. Exact V16b per-ID attribution still needs one dependency

The newly supplied scripts/v16_memory_ltr.py is the correct V16b runner and confirms:

- c35 is fixed;
- only the locked memory-slot-5 policy family is exposed;
- holdout predictions can be exported through --holdout-predictions;
- public full-fit occurs only after complement and holdout promotion;
- the reported total parameter upper bound is 1.09001B.

However, the runner imports the missing scripts/v11_tiny_ltr.py, including the exact feature construction, model training, policy application and tie-breaking functions. It also requires the exact --prior-index used by load_priors. A reconstruction from cached matrices did not achieve byte-exact public parity and was rejected; approximate parity must not be used to label the 177 holdout interventions.

The fastest artifact that completes exact attribution is:

- output/v16_rgm/holdout_v16b_memory_slot5.json, or the actual file passed to --holdout-predictions.

If that output was never saved, supply both:

- scripts/v11_tiny_ltr.py;
- the exact --prior-index supplied to V16b, or a compatible compact export.

Until then, the per-ID table is exact through V11h, V16b is exact at aggregate holdout level, and the public audit is exact structurally but has no secret-gold TP/FP labels.
