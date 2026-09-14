# LegalIR — trạng thái hiện tại

Cập nhật: 2026-08-29

## Main compliant pipeline

- Candidate chính: `output/v16b_memory_slot5.zip`.
- Public score: **`0.9210`** cho `v16b_memory_slot5.zip` theo leaderboard snapshot; v11h public baseline là **`0.9183`**.
- Locked holdout Recall: **`0.9228333`** (`+0.0040000` so với v11h).
- V16b chỉ dùng c35 và thay tối đa slot 5; public đã sinh 1.000 câu hợp lệ.
- ZIP SHA-256 V16b:
  `B2816BD73AE23FBDFADD6563DC43728896F078A6AD8032B6743984B3AADADFD9`.
- V11h baseline vẫn được giữ tại `output/v11h.zip`.
- Grouped complement OOF Recall: `0.9018658`, fold std `0.0065850`.
- ZIP SHA-256:
  `57CDEF271C86824B128482AD141E140224E8002B7ACC7536CB79DAE90EB7F175`.
- Hai public experimental vừa materialize để A/B:
  - `output/v11l.zip`: c50 candidate pool, public **`0.9181`**.
  - `output/v11m.zip`: prefix-512 safe-tail dense, public **`0.9090`**; đã loại.

V11h dùng BQB Vietnamese legal embedding 330M + Vietnamese reranker 760M +
tiny supervised LTR, tổng xấp xỉ `1.09B < 4B`. Runtime chính chạy CUDA/FP16.

## Các tầng đã khóa

1. Legal FTS/chunk phrase/citation + BQB dense Stage-1.
2. Safe-tail corpus-only PRF/citation completion, khóa nguyên top-5.
3. Stage-2 c35, ba passages/document, max length 768.
4. Query-normalized v2 features bổ sung provenance trước PRF, neural z-score,
   rank agreement, exact phrase/title và length-ratio signals.
5. Pointwise-logistic LTR (`C=1`) được chọn bằng grouped OOF complement.
6. Exact-heading rescue 10 từ và locked-holdout promotion gate.

Safe-tail PRF tăng locked oracle@35 từ `0.960667` lên `0.967667` mà không đổi
Stage-1 Recall@5. LTR chuyển một phần coverage đó thành Recall@5 `0.918833`.
Nút thắt tiếp theo là selector/reranker, không còn chủ yếu là candidate recall.

## Baseline và provenance

- `output/v11g.zip`: public `0.9086`, locked holdout `0.9120000`.
- `output/v11f.zip`: public `0.9106`, locked holdout `0.9079167`.
- `output/v11d.zip`: public fallback đã biết `0.9037` (rounded).
- V10 chỉ giữ làm baseline/fallback.
- V57 exact10 từng đạt public `0.9190`, nhưng lineage khoảng `6.19B+`; không
  được dùng làm end-to-end v11. Chỉ các ý tưởng thuật toán như exact-heading,
  legal chunking và rescue gate được tái triển khai độc lập.

## Trạng thái kỹ thuật

- RTK v0.45.0 project-local là wrapper mặc định cho output shell lớn.
- Public PRF và incremental Stage-2 c35 đã chạy xong trên GPU; 88,9% document
  neural scores được tái dùng.
- `v11h.zip` đã kiểm tra đủ 1.000 QID, đúng 5 ID/câu, không duplicate.
- Listwise direct và score ensemble tăng complement OOF nhưng giảm locked
  holdout, nên đã dừng trước public submission.
- C50 đã bị loại: OOF tăng nhưng locked holdout giảm `0.002000`; không chạy
  public c50.
- Prefix-512 dense safe-tail tăng holdout oracle@35 lên `0.974250` nhưng LTR
  holdout chỉ `0.914333`; đã loại trước public.
- Phase kế tiếp (đang tạm dừng): đo phân phối token thực tế của reranker để quyết định có
  đáng chạy max-length 1024 hay chuyển thẳng sang adaptive legal passages.
- Theo yêu cầu A/B, đã tạm dừng nghiên cứu sau khi package v11m/v11l; không có
  worker Python/GPU còn chạy.
- Kết luận A/B: giữ `v11h.zip` làm main public `0.9183`; giữ `v11l.zip` làm
  fallback gần nhất (`-0.0002`).

## Research-track v12 (2026-08-27) — đã chốt và dừng

Đã triển khai trọn chuỗi selector theo tài liệu nghiên cứu trên candidate pool
c35 cố định của v11h: deterministic multi-aspect parsing, Aspect-MaxSim từ
BQB dense cache (encoder chạy CUDA), greedy complementary evidence-set,
TF-IDF word 1–2 + char_wb 3–5 và grouped LambdaMART objective `lambdarank`
với `ndcg@5`, truncation 8, 15 leaves, depth 5, learning rate 0.03, 400
trees, regularization 2.0.

| Gate | Recall | Ghi chú |
|---|---:|---|
| v12 grouped OOF selection | 0.910290 | robust 0.902168, thấp hơn v11h 0.918833 |
| v12 locked final | 0.906114 | cao hơn baseline cục bộ 0.893013 nhưng không thắng v11h |

Vì grouped OOF chưa vượt locked v11h `0.918833`, điều kiện mở fine-tune
reranker không đạt; không fine-tune và không thay main. Candidate phân tích
(không phải public submission) và report nằm tại `output/v12_research/`.
Pipeline v12 đã dừng theo yêu cầu; main vẫn là `output/v11h.zip`.

## V13 conditional fusion + controlled reranker fine-tune (2026-08-27) — đã chốt và dừng

Đã hoàn tất V13 theo hướng nghiên cứu: khóa lại v11h, đo union/v12-only trên
clean complement, huấn luyện router nhỏ bằng OOF, rồi thử hard-negative LoRA
fine-tune reranker hiện tại chỉ trên complement. Không đọc holdout/public trong
quá trình train hay chọn model.

| Gate | Recall | Kết luận |
|---|---:|---|
| v11h locked holdout | 0.918833 | baseline giữ làm main |
| V13 router locked holdout | 0.918833 | không tăng, không promote |
| V13 fine-tuned reranker locked holdout | 0.901083 | giảm 0.017750, loại |

Checkpoint đã merge, chạy độc lập tại `output/v13_reranker_finetune/model/`;
evaluator locked holdout đã chạy bằng GPU và sinh
`output/v13_reranker_finetune/holdout_stage2.json`.

Kết luận cuối: giữ v11h làm bản chính/public; V13 router và checkpoint
fine-tune chỉ giữ làm tài liệu tham khảo. Dừng nghiên cứu sau gate fine-tune
theo yêu cầu.

## Research-track V14 ColBERT late interaction — checkpoint 2026-08-29

Đã triển khai đúng nhánh Phase-B của nghiên cứu trên candidate pool v11h c35:
Jina-ColBERT-v2 late interaction, legal chunk đầy đủ, typed-element MaxSim,
union evidence 1/2/3 passage, candidate-vs-tail delta, guard CE margin và
rank cap, cùng selector frozen/grouped-OOF. Logic tái sử dụng và provenance đầy
đủ nằm trong `V14_COLBERT_DISCLOSURE.md`.

| Gate | Recall delta | Intervention precision | Destructive rate | Trạng thái |
|---|---:|---:|---:|---|
| Complement standard/law/template OOD | `+0.0050125313` | `0.8888888889` | `0.0046511628` | Pass |
| Locked holdout | `-0.0010000000` | `0.3333333333` | `0.0104712042` | Reject |

Policy thắng trên complement là `frozen_full_delta` + `colbert_ce_rank25`,
threshold `0.02197723388671874`, chỉ thay slot 5 tối đa một lần. Holdout chạy
CUDA đầy đủ, baseline được xác nhận `0.9188333333333333`, rescued chỉ còn
`0.9178333333333333`; vì vậy không public, không thay `v11h.zip`, và không dùng
holdout để retune. V14 đã chứng minh ColBERT có signal cục bộ nhưng chưa đủ
robust để mở LoRA/listwise; bước tiếp theo phải là robustness experiment mới,
holdout-blind.

End-to-end budget V14: v11h embedding `330,000,000` + v11h cross-encoder
`760,000,000` + Jina-ColBERT-v2 exact `559,497,216` + algorithmic logic `0` =
`1,649,497,216` (`1.649497216B`), strictly `<4B`. Report:
`output/v14_colbert_full/v14_colbert_report.json`.

## Research-track V15 Counterfactual Pairwise Legal Adjudication — checkpoint 2026-08-29

Đã triển khai đúng hướng mới sau V14: giữ v11h làm prior, dùng prefilter đa tín
hiệu V14 trên c35, lập tournament shortlist 8/10 với so sánh trực tiếp
`(q, dA, dB)`, exclusive typed evidence, weakest constraint, cycle consistency
và chỉ cho phép thay slot 5 tối đa một lần. Query rewriting là BTC-only
deterministic sidecar, có giữ negation và numeric cue. Toàn bộ parser/evidence/
metric logic dùng lại được ghi trong `V15_COUNTERFACTUAL_DISCLOSURE.md`.

| Gate | Kết quả | Trạng thái |
|---|---:|---|
| Oracle c35 prefilter capture shortlist 8 / 10 | `43.86% / 59.65%` | còn headroom |
| Best complement recall delta | `-0.0012531328` | fail |
| Best standard intervention precision | `0.40` | fail |
| Best standard destructive rate | `0.0292682927` | fail |
| Locked holdout | chưa mở | đúng policy |

V15 algorithmic chưa tạo được robust signal nên không mở LoRA pairwise /
group-softmax, listwise specialist, hay retrieval ngoài c35. `v11h.zip` vẫn là
public main. V15 không thêm tham số; end-to-end dùng lại v11h embedding
`330M` + cross-encoder `760M` = `1.09B`, strictly `<4B`. Report:
`V15_COUNTERFACTUAL_DISCLOSURE.md` (raw report đã được dọn trong cleanup).

## Cleanup checkpoint — 2026-08-29

Đã hoàn tất dọn artifact dư thừa. Workspace hiện giữ đúng chain tái lập v11h:
raw public data, ba index `v11_research_*_256`, các safe-tail c35 và audit cần
thiết, feature/LTR cache của v11f/v11g/v11h, hai model ở workspace root, source,
tests, runner và log/disclosure. `output/v11h.zip` được giữ nguyên; `v11f.zip`
và `v11g.zip` chỉ là checkpoint trung gian để tái lập chain.

Đã xoá 80 file output trung gian/fallback/sweep và các artifact c50 không còn
được runner dùng. Các thư mục index cũ, experiment output V12–V15 và model trùng
đã được dọn ở cùng đợt cleanup. Các file này không còn bản sao cục bộ nhưng có
thể dựng lại từ source + data + model + runner. Hash v11h sau cleanup không đổi:
`57CDEF271C86824B128482AD141E140224E8002B7ACC7536CB79DAE90EB7F175`.

## Single-file runner checkpoint — 2026-08-29

`run_v11h_end_to_end.ps1` hiện đã chứa trực tiếp ba stage F → G → H và không
gọi hai runner chain cũ. Đây là entrypoint dùng khi trình bày/nộp source; các
file `run_v11f_end_to_end.ps1` và `run_v11g_end_to_end.ps1` chỉ còn để đối chiếu
lineage. Budget được khóa ở `1.09B < 4B`, không có model mới.

## V16-RGM QREL-memory checkpoint — 2026-08-29

Đã triển khai Phase A của hướng mới trong `scripts/v16_memory_oracle.py`:
memory fit-only theo BQB, reciprocal-neighbor gating, degree-corrected QREL
propagation, prototype/vote/entropy/agreement features và anchor-preserving
union `c35 + memory@5`. V11h không bị thay đổi.

| Split | Oracle c35 | Oracle c35 + memory@5 | Delta |
|---|---:|---:|---:|
| Complement 1.200 | 0.950097 | 0.955097 | +0.005000 |
| Locked holdout 1.000 | 0.967667 | 0.969167 | +0.001500 |

Split có 3/5 normalized duplicate questions xuyên partition; run đã loại các
memory rows này trước khi đo để giữ leakage gate. Cả hai delta đều thấp hơn
promotion gate `+0.01`, nên chưa mở graph teacher/KD/LoRA. Chi tiết và lệnh dựng
lại nằm trong `V16_RGM_DISCLOSURE.md`. Phần oracle này không mở graph teacher,
nhưng memory features vẫn được dùng cho gate V16b bên dưới.

## V16b c35-only memory slot-5 gate — 2026-08-29

Đã áp dụng đúng bottleneck đã xác định: giữ nguyên 35 candidate của v11h,
không mở rộng Stage-2, thêm các feature QREL-memory fit-only, và chỉ cho phép
thay slot 5 nếu candidate vượt memory floor `0.005` cùng điểm selector không
thấp hơn tail anchor. Anchor holdout/public là output v11h locked hiện hành.

| Gate | Recall | Delta | Trạng thái |
|---|---:|---:|---|
| v11h locked anchor | `0.918833` | — | baseline |
| V16b complement grouped OOF | `0.886062` | `+0.007310` vs exact anchor | pass |
| V16b locked holdout | `0.922833` | **`+0.004000`** | **promoted** |

Holdout transition: 177 queries changed, 9 successful rescues, 4 destructive
changes. Public submission là `output/v16b_memory_slot5.zip`; JSON trước đóng
gói là `output/v16_rgm/v16b_memory_slot5.json`. Report:
`output/v16_rgm/v16b_memory_ltr_report.json`.

End-to-end budget V16b: BQB embedding `330M` + reranker `760M` + tiny selector
`<=10K` = `1,090,010,000 < 4B`. Graph teacher/KD chưa mở.

## V16d-lite graph adapter diagnostic — 2026-08-29

Đã chạy đúng nhánh diagnostic holdout-blind: graph/QREL adapter chỉ được phép
điều chỉnh query representation trong memory KNN; không chạm document retrieval,
c35, cross-encoder, CE/PRF hay BM25. Ba câu bị loại do normalized overlap và
không dùng nhãn public/holdout.

| Metric trên clean complement | BQB gốc | V16d-lite | Delta |
|---|---:|---:|---:|
| Neighbor QREL precision@5 | 0.089500 | 0.099167 | +0.009667 |
| Memory recall@5 | 0.295069 | 0.310764 | **+0.015694** |
| Prototype precision@5 | 0.063000 | 0.066333 | +0.003333 |

Tuy memory quality tăng, khi đưa qua đúng V16b c35 slot-5 policy thì recall OOF
giữ nguyên `0.8860623782`; chỉ intervention precision tăng `0.888889 → 0.920000`.
Vì promotion cần tăng recall OOF, V16d-lite không được promote/fuse và không mở
holdout. Report: `output/v16d_lite/v16d_lite_complement_report.json` và
`output/v16d_lite/v16d_lite_oof_gate_report.json`.

## V17a-lite setwise/boundary head — 2026-08-29

Đã chạy pre-ablation rủi ro thấp trên clean complement: giữ CE/backbone frozen,
khóa top-4, chỉ xét anchor slot 5 với 6 challengers mạnh nhất trong c35, dùng
set head permutation-invariant và asymmetric loss (destructive weight `3x`).
Nhánh này cố ý dùng feature đã audit; CE hidden-state extraction chỉ mở nếu
feature-only setwise head chứng minh có signal.

| Gate | V16b anchor | V17a best (`threshold=0.80`) |
|---|---:|---:|
| Complement OOF Recall@5 | **0.886062** | 0.880284 |
| Robust recall | **0.884042** | 0.878306 |
| Intervention precision | **0.888889** | 0.409091 |
| Rescues / destructive | 24 / 3 | 18 / 26 |

V17a không đạt gate recall lẫn safety, nên bị reject trước locked holdout; không
mở V17b duo comparator, không mở hidden-state extraction và không thay V16b.
Chi tiết: `V17_BOUNDARY_DISCLOSURE.md` và
`output/v17a_boundary/v17a_complement_oof_report.json`.

## Checkpoint quyết định hiện tại — 2026-08-29

- Giữ `output/v16b_memory_slot5.zip` làm main; đây là bản đã có public `0.9210`
  và locked holdout `0.9228333`.
- V16d-lite được giữ làm diagnostic artifact vì tăng memory quality nhưng không
  tăng downstream recall.
- V17a-lite được đóng vì setwise boundary head chưa có signal đủ an toàn.
- Không retune theo holdout, không chạy V17b/V16e và không mở retrieval ngoài
  c35. Budget deployed vẫn `1,090,010,000 < 4B`.

## V20-PDG continuation — 2026-08-31

Theo quyết định nghiên cứu mới, V16b tiếp tục là anchor đã khóa và V18 E3B là
nguồn tín hiệu ứng viên. Đã thêm `scripts/v20_pairwise_deferral_gate.py`:

- loại bỏ exact-document QREL/prototype khỏi feature path mới;
- audit P0 CPU-only trước khi cho phép P1;
- so sánh trực tiếp từng challenger với V16b slot 5;
- chỉ xét c35 ranks 6--12, khóa top-4, tối đa một thay thế;
- logistic pairwise gate có inner-group calibration và ngưỡng mặc định `0.90`;
- abstain mặc định về nguyên V16b anchor;
- fail closed nếu thiếu bootstrap/OOD/promotion evidence.

Đây mới là research continuation, chưa được promote và chưa đọc holdout/public.
Nếu P0/P1 không đạt các gate trong `V20_PDG_PROTOCOL.md`, bước tiếp theo hợp lệ
là block-level CATER với cùng policy slot-5, không phải Direct-5 hay K16/K64.

## V16 boundary-research closure — 2026-08-30

Đã hoàn tất P0 error waterfall, P1 GateCal, P2 Prototype-v2, P3 V16d-Utility,
P4 DuoLegal frozen và kiểm chứng closure holdout-blind. Không có nhánh nào
vượt V16b: P4 tốt nhất chỉ tái hiện chính xác OOF `0.8860623782`, robust
`0.8840418489`, IP `0.8888888889` của V16b. Vì vậy P5 Graph-to-Boundary KD và
P6 joint-text được **cố ý không chạy** theo gate của research report; không mở
holdout mới.

Closure tái lập V16b trên standard/law-family/template/near-duplicate grouped
OOF: recall lần lượt `0.8860624`, `0.8850181`, `0.8852270`, `0.8873155`.
Paired bootstrap 10.000 mẫu so với locked c35 anchor cho delta Recall@5
`+0.0154553`, CI95% `[+0.0083542, +0.0229741]`. Law-family là stress split
khó hơn, dùng để chặn candidate tương lai chứ không retune V16b.

Main không đổi: `output/v16b_memory_slot5.zip`, public `0.9210`, locked
holdout `0.9228333`, deployed `1.09001B < 4B`. Artifact và quyết định đầy đủ:
`V16_BOUNDARY_RESEARCH_DISCLOSURE.md` và
`output/v16_boundary_research/closure/research_closure_report.json`.

## V20-PDG/P1 và V19-CATER execution closure — 2026-08-31

Đã chạy P1 với current-passage CE và giữ kết quả dù recall gain chỉ thiếu
`0.000076` so với ngưỡng cứng: recall OOF `0.8889863552`, gain
`+0.0029239766`, robust gain `+0.0025695132`, paired-bootstrap lower bound
`+0.0004177109`, intervention precision `1.0`, `13` changes / `4` rescues /
`0` destructive. Đây là tín hiệu tốt về mặt chẩn đoán, nhưng promotion vẫn
**fail**: recall gate chưa đủ `+0.003`, và cả `4/4` registered OOD group-isolated
checks đều không đạt non-negative/worst-fold contract. Vì vậy P1 không được
dùng để mở holdout.

Theo protocol, CATER đã được chạy tiếp. Block-aggregation oracle đạt gain
`+0.0384990` so với best V18 audit reference, nên đủ điều kiện đánh giá đúng
`3` biến thể grouped OOF. Không biến thể nào được promote. Best numeric result
là `weighted_blocks`: recall `0.8939988861` (`+0.0079365` so với V16b anchor),
nhưng bootstrap lower bound so với best V18 là `-0.0171261`, ECE `0.0850975`,
và strict standard-group worst fold `-0.5`; `calibrated_setwise` cũng có
bootstrap lower `-0.0199109` và ECE `0.0753029`. `adaptive_budget` chỉ tạo
`1` intervention. Pipeline ledger CATER là `1,090,010,036 < 4B`.

Holdout vẫn đóng, public labels không được dùng, `output/v16b_memory_slot5.zip`
không bị thay thế và không có ZIP mới được nộp. Reports:
`output/v20_pairwise_deferral_gate/v20_pdg_v18b_corrected_r3_currentce.json`,
`output/v19_cater_block_features_r1/audit/v19_cater_feature_audit.json`, và
`output/v19_cater_block_features_r1/audit/v19_cater_grouped_oof_r1.json`.

## V20 CATER-R P2 and frozen-CE representation audit — 2026-08-31

Theo research report mới, đã triển khai riêng `scripts/v20_cater_r.py` với
đúng thứ tự P2-0/P2-1/P2-2/P2-3: kiểm tra provenance V16b, block oracle, đúng
hai deterministic variants và một calibrated setwise two-head utility model
được cross-fit độc lập dưới bốn regime. Model dùng utility
`P(rescue) - 3 * P(destructive)`, nested-OOF thresholds, consensus tối thiểu
3/4, và khóa top-4.

P2-0 tái lập anchor chính xác: Recall@5 `0.8860623782`, không mở holdout.
Block oracle đạt `0.9395293790`, gain `+0.0394041` so với best valid V18
slot-5 boundary policy, nên vượt stop gate `+0.003` và cho phép chạy tiếp.

Không variant nào được promote:

- `adaptive_budget`: Recall `0.8919799500` (`+0.0059176`), nhưng intervention
  precision `0.594595`, bootstrap lower `-0.0028544`;
- `weighted_blocks`: Recall `0.8898217767` (`+0.0037594`), intervention
  precision `0.594595`, bootstrap lower `-0.0048733`;
- `calibrated_setwise`: Recall `0.8860623782` (`+0.0000000`), robust gain
  `-0.0003201`, intervention precision `0.500000`, bootstrap lower
  `-0.0025063`.

Các regime checks strict đều không cho phép promotion; `selected_variant` là
`null`. Ledger upper bound CATER-R là `1.09101B < 4B`. Vì vậy main
`output/v16b_memory_slot5.zip` giữ nguyên và holdout/public vẫn đóng.

Do block oracle pass nhưng CATER-R thất bại, đã chạy audit biểu diễn CE nhỏ
theo điều kiện của report. CE có AUC rescue-vs-destructive `0.652872`, median
margin rescue `-0.0003002` lớn hơn destructive `-0.0096724`; không có systematic
score inversion. Quyết định là `stop_before_ce_finetuning`, không chạy LoRA.
Artifacts: `output/v20_cater_r/v20_cater_r_p2_report.json` và
`output/v20_cater_r/v20_ce_representation_audit.json`.

## V20 conditional P3 CE LoRA and CATER-R recheck — 2026-08-31

Because the registered block oracle passed while CATER-R failed across
regimes, the research report's conditional CE branch was executed. The
existing local 760M CE was adapted with one fixed `r=4` LoRA arm using only
clean-complement boundary pairs: 268 semi-hard pairs, the highest-scoring
negative skipped when possible, one frozen-CE anchor-preservation example per
query, and five query-hash cross-fitted adapters.

The adapted sidecar passed the unchanged CATER-R contract: Recall gain
`+0.0033417`, robust gain `+0.0040910`, paired-bootstrap lower bound
`+0.0008354`, intervention precision `1.0`, 13 consensus interventions, and
all four registered regimes non-negative. The adapted CE remains a sidecar;
the accepted V16b ZIP was not replaced.

The holdout trial was considered but skipped because this checkout has no
holdout CATER-R evidence packs, block features, or adapted score matrix. A
direct opening would require a separate holdout-specific BQB/evidence-pack
materialization. No holdout labels were opened and no new submission ZIP was
produced. Artifacts:
`output/v20_ce_lora_r4_pack/lora_training_report.json`,
`output/v20_ce_lora_r4_pack/adapted_current_passage_scores.json`, and
`output/v20_cater_r_r4_pack/v20_cater_r_r4_report.json`.

## V20 holdout parity audit correction — 2026-08-31

The experimental external r8 branch was stopped and is benchmark-only: it
used `skip-bqb`/structural-first blocks and is not valid for CATER-R,
promotion, holdout evaluation, or submission. The scorer process was stopped
before any r8 result was used.

The adapted CE leakage audit found exact normalized-query isolation, but not
the stricter registered group isolation required by the four-regime contract:
548 validation qids overlap standard groups, 605 overlap law-family groups,
1,063 overlap query-template groups, and 19 overlap near-duplicate groups.
Therefore the existing P3 OOF result remains exploratory and holdout is
locked; the CE was not retuned.

The four deployment-head audit found materially different heads (utility
pairwise correlation about `0.907`–`0.970`, maximum utility delta about
`0.65`–`1.41`), so four-head consensus is retained rather than replacing it
with a single head. A separate exact-parity external preparer was added using
the registered BM25/BQB/MMR proposer and frozen current-passage CE. Its
packability audit found 5 invalid documents affecting 10 qids; those qids are
configured to preserve their original boundary and force-abstain, with no
challenger replacement. Frozen current CE completed for 990 active qids, but
the exact BQB pack manifest was intentionally stopped before completion after
the leakage gate failed. No holdout reference labels were read, no holdout
promotion was performed, and `output/v16b_memory_slot5.zip` remains the only
eligible fallback.

Preflight: `output/v20_p3_external_holdout_exact_r1/holdout_preflight.json`.
Audits: `output/v20_ce_lora_r4_pack/ce_leakage_audit.json` and
`output/v20_cater_r_r4_pack/deployment/head_similarity_audit.json`.

## V16b release and experiment quarantine — 2026-08-31

The reproducible release is now `release/v16b_0921/`. It includes a single
end-to-end runner, a required-path/hash verifier, a manifest of V11h/V16b
inputs, and the current lessons/bottlenecks document. The frozen eligible ZIP
remains `output/v16b_memory_slot5.zip` (public `0.9210`, SHA-256
`B2816BD73AE23FBDFADD6563DC43728896F078A6AD8032B6743984B3AADADFD9`).

All V16-boundary/V16d/V17/V18/V19/V20 outputs, experiment runners, source
scripts, protocols, and accidental workspace-level V20 outputs were moved
recoverably to `archive/2026-08-31_retired_experiments/`. These historical
paths supersede earlier direct `output/v20_*` references in this log; none is
a submission candidate. V11h and V16b code, data, indexes, models, feature
caches, audit artifacts, submission JSONs, and ZIPs remain in place and pass
`release/v16b_0921/verify_release.ps1 -VerifyHashes`.

## Qwen c67-shortlist8 locks (2026-09-14)
- Runner now requires explicit VARIANTS (default A_c35_short); D_c200_long dropped from queue.
- A_c35_short resumed on GPU0 from 75/1197 (QID-resumable); E_c67_short queued after A (36-67 fresh, 1-35 reused).
- Budget (at 175 pairs/90s): c67-short complement 80199 (~11.5h), F-shortlist8 complement <=9576 (4.1-6.8h*), c67-short public 67000 (~9.6h), F-shortlist8 public <=8000 (3.4-5.7h*). *Long 3-5x is assumption until bench measures.
- Deadline 17/9: -24h stop research and lock config to public; -8h ZIP ready. Over budget -> drop long, ship E_c67_short+S1/S2. Fail local gate -> keep 0.9252. No 0.96 promise.
