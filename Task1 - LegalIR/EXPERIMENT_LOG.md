# LegalIR — experiment log cô đọng

## 1. Kiểm tra scorer và validation

Official scorer dùng set intersection, macro Recall/Precision và tối đa 5 ID.
Công thức scorer đúng. Một baseline holdout cũ khớp gold 1000/1000 do leakage
đã được phát hiện và loại khỏi đánh giá.

Lỗi lớn thứ hai là anchor mismatch: top-5 của v49 và `production_fusion` trong
public cache chỉ trùng trên 40/1000 query. Policy học cách thay slot 5 của một
mapping nhưng lại áp lên slot 5 của mapping khác, nên local tăng trong khi
public giảm. Từ đó mọi proposal bắt buộc phải so với đúng anchor submission.

## 2. Các hướng đã thử và bị loại

| Hướng | Kết quả chính | Lý do đóng |
|---|---|---|
| Qwen2.5 answerability | Có rescue cục bộ nhưng full replacement kém production | Không đủ ổn định |
| Qwen3 embedding/reranker | Tăng candidate coverage, selector trực tiếp âm hoặc không thêm gold mới | Coverage không chuyển thành utility |
| BQB/Harrier question-kNN | Holdout tăng mạnh nhưng public 0.9137–0.9162 | Anchor mismatch và legal-topic collision |
| BGE pairwise reranker | Gate local an toàn nhưng chọn sai TNCN/TNDN và agency trên public | Semantic similarity thiếu legal element |
| LambdaRank/nested selector | Public khoảng 0.9150–0.9152 | Không vượt baseline |
| Citation expansion rộng | Dương trên fit, âm trên holdout | Viện dẫn nền không đồng nghĩa văn bản đáp án |

Những artifact và script của các hướng này đã bị xóa sau khi kết luận được ghi
lại tại đây.

## 3. Hai legal rescue tạo legal2

- QID 1392: thay `110993 → 53190`, đúng QĐ 06/QĐ-TTg chứa heading “phục vụ
  phát triển công dân số”. Public tăng từ v49 `0.9160` lên `0.9170`.
- QID 87048: thay `18719 → 203564`, bổ sung Luật Hòa giải, đối thoại tại Tòa
  án cho NĐ 16/2021 đã ở top 1. Public tăng tiếp lên legal2 `0.9175`.

Hai phép A/B này xác nhận exact legal heading và citation completion có thể
hiệu quả khi bằng chứng đủ cụ thể. Nó không xác nhận việc mở rộng hàng loạt.

## 4. Exact-heading v57/v58

Rule mới lấy candidate từ top 12 của `focused_chunk`, `chunk_phrase`,
`or_bm25`, `lexical_fusion` và `production_fusion`. Candidate chỉ thay slot 5
khi cụm liên tiếp khớp câu hỏi dài hơn document cũ.

| Gate | Fit Recall delta | Holdout Recall delta | Public changes | Public score |
|---|---:|---:|---:|---:|
| Exact >=10 từ (v57) | +0.014063 | +0.011500 | 99 | **0.9190** |
| Exact >=7 từ (v58) | +0.022813 | +0.013000 | 374 | 0.9148 |

Kết luận: exact-heading là feature đúng, nhưng độ rộng policy quyết định khả
năng transfer. V57 chặt và tăng public; v58 thay quá nhiều và thất bại dù local
cao hơn. Không được chọn policy tương lai chỉ bằng aggregate holdout delta.

## Baseline cuối

`sub_v57_exact10.zip` là baseline duy nhất còn hiệu lực. Mọi nghiên cứu sau
phải bắt đầu từ chính payload v57 và chứng minh utility theo nhóm query trước
khi tạo một trial public rộng.

Khi công bố phương pháp, v57 phải được mô tả là pipeline có lineage
`Vi-Qwen2-1.5B-RAG-assisted v47 → Qwen2.5-3B-assisted v49 → legal2 rescues →
exact10`. Chỉ riêng bước tăng từ legal2 lên v57 là không dùng model; submission
cuối vẫn kế thừa quyết định của cả hai LLM upstream. Chi tiết flow và policy
được giữ tại `V49_PIPELINE_DISCLOSURE.md`.

## V11 compliant branch (2026-08-24)

Phản hồi BTC xác nhận không được dùng v57 làm end-to-end system vì lineage
v57 vượt `<4B` (manifest tính khoảng `6.19B+`). V11 chỉ kế thừa ý tưởng
thuật toán: legal chunking, hybrid lexical/dense RRF, candidate pool rộng,
adaptive passage hydration, top-3 passage aggregation và exact legal rescue.

Index độc lập `data/processed/v11_research_256` đã build xong với 8.532 văn
bản, 33.846 legal chunks và BQB dense cache. Registry v10 xác nhận BQB
330M + Vietnamese reranker 760M = `1.09B < 4B`. Reranker có tùy chọn FP16
runtime trên CUDA; FP16 không được tính là giảm số tham số.

Exact-heading ablation được chạy riêng trên cache thuật toán cũ để xác nhận
ý tưởng, không làm thành input model cho v11: gate `>=10` từ tăng holdout
Recall `+0.0115` với 128 thay đổi. Đây không phải điểm số của v11 và không
được dùng để gọi v11 đã vượt v57.

## V11 gated stages — full holdout (2026-08-24)

Stage 1 chạy trên `v11_research_fit_256`, dùng 6.000 câu fit để build question
dense và giữ riêng 1.000 câu holdout. Profile `chunk_only` giữ focused+dense+
chunk-phrase, tắt OR/question-dense/character-neighbor để giảm nhiễu.

| Tầng | Recall@5 | Precision@5 | Candidate oracle | Trạng thái |
|---|---:|---:|---|---|
| Stage 1 top-5 | 0.827917 | 0.176400 | @20 0.938333; @100 0.982167 | Qua candidate gate |
| Stage 2 raw audit | 0.827917 | 0.176400 | neural scores đã tính | Loại: batch RRF giữ sai thứ tự |
| Stage 2 repaired, weight1/k10 | **0.888917** | **0.189800** | cùng candidate set | Qua holdout accuracy gate |
| Stage 2 exploratory, weight1.5/k5 | 0.894417 | 0.191000 | cùng candidate set | Chưa chốt; cần fit calibration |

Stage 2 không được phép mở candidate set. Lỗi raw audit đã được sửa trong
`reranker.py` và `batch_stage2_rerank.py`: neural items có `rerank_score` phải
được sort trước khi tạo RRF rank. Vì neural scores của full run đã đúng, audit
được repair offline bằng `repair_stage2_fusion_from_audit.py` thay vì chấm lại
60k cặp GPU. Regression test cho RRF scored-neural đã pass; test cũ về explicit
neural order vẫn giữ nguyên.

Public artifact được tạo bằng đúng cấu hình holdout đã khóa:
`output/public_submission_v11_research_256_chunk_only_c20_p3_top5.zip`.
Hậu kiểm pass: 1.000/1.000 query IDs khớp `public-official.json`, mỗi câu có
đúng 5 ID dạng string và không có duplicate. Public không có gold nên không
gán một public Recall giả; điểm `0.888917` chỉ là holdout có gold.

## V10.3 dense prefix/length ablation (2026-08-26)

Đã chạy GPU dense-only trên clean holdout 1.000 câu cho bốn view. Kết quả quan
trọng:

| View | Recall@35 | Recall@80 | Recall@200 |
|---|---:|---:|---:|
| prefix, 256 | 0.768583 | 0.861583 | 0.906167 |
| no-prefix, 256 | **0.786833** | 0.859917 | 0.904667 |
| prefix, 384 | 0.777750 | 0.869500 | **0.911667** |
| prefix, 512 | 0.778083 | **0.871333** | 0.910000 |

Kết luận: no-prefix 256 tốt nhất ở cutoff vận hành 35, prefix 512 tốt nhất ở
head/tail khác; prefix 384 bị dominated tại các cutoff vận hành và dừng. Hai
view thắng là channel bổ sung tiềm năng, chưa được nhập vào v11g để tránh tăng
chi phí trước khi selector hiện tại được giải quyết.

## V10.9 safe-tail PRF + legal citation QE (2026-08-26)

PRF chỉ đọc corpus: lấy top-4 pseudo-relevant documents, rút tối đa 4 term hiếm
được ít nhất hai seed documents hỗ trợ, cộng legal document-number completion.
Không đọc label trong lúc sinh candidate. Bản đầu tăng oracle nhưng làm giảm
Recall@5; policy cuối khóa nguyên top-5 và chỉ fusion phần đuôi.

| Split | Stage-1 Recall@5 | Oracle@35 trước | Oracle@35 PRF | Oracle@50 PRF |
|---|---:|---:|---:|---:|
| complement 1.200 | 0.754819 (giữ nguyên) | 0.941486 | **0.950097** | 0.959431 |
| locked holdout 1.000 | 0.827917 (giữ nguyên) | 0.960667 | **0.967667** | 0.971750 |

Incremental Stage-2 tái dùng neural cache và chỉ chấm candidate mới. Public đã
tái dùng 31.116/35.000 document scores (88,9%), chấm mới 3.884 documents =
11.648 query-passage pairs trên CUDA/FP16.

## V11g PRF + grouped OOF LTR (2026-08-26)

Model/policy được chọn chỉ từ grouped OOF complement; locked holdout được mở
một lần sau khi khóa cấu hình. Cấu hình thắng là pairwise logistic `C=1.0`, 25
hard negatives, pool 35, exact-heading rescue 10 từ.

| Gate | Recall | Delta |
|---|---:|---:|
| complement c35 + exact | 0.878335 | anchor |
| complement grouped OOF LTR | **0.895809** | +0.017474 |
| locked holdout c35 + exact | 0.903083 | anchor |
| locked holdout LTR | **0.912000** | +0.008917 |

Fold recalls là `0.892361, 0.908472, 0.880056, 0.900976, 0.897141`, std
`0.009475`. Gate được promote và tạo `output/v11g.zip`. Hậu kiểm độc lập: đúng
1 member `submission.json`, 1.000 QID duy nhất, đúng 5 answers/câu; SHA-256
`5C57D6D41AB7733AD059F740343504E1CD1E74508EBAC788713DC033C53918AD`.
Public sau đó đạt `0.9086`; v11f đạt `0.9106`. V11g vì vậy được giữ làm
feature-v1 ablation, không còn là main submission.

## V11h query-normalized LTR (2026-08-26)

Feature set v2 bổ sung 38 tín hiệu theo từng query: provenance/rank trước PRF,
neural log-probability và z-score/min-max/margins, rank agreement, exact phrase
5/7/10, title match và length ratio. Cấu hình được chọn hoàn toàn trên grouped
OOF complement là pointwise logistic `C=1`, pool 35, exact rescue 10 từ.

| Gate | Recall | Ghi chú |
|---|---:|---|
| complement grouped OOF | **0.901866** | fold std `0.006585`; robust `0.900220` |
| locked holdout | **0.918833** | mở sau khi khóa model/policy |
| public | **0.9183** | main compliant, khoảng `1.09B < 4B` |

V11h tăng public `+0.0097` so với v11g và `+0.0077` so với v11f; chênh giữa
locked holdout và public chỉ `0.000534`. Artifact `output/v11h.zip` có SHA-256
`57CDEF271C86824B128482AD141E140224E8002B7ACC7536CB79DAE90EB7F175`.

## Selector ablations sau V11h (2026-08-26)

- Sweep pointwise logistic `C=0.25..20`: `C=1` vẫn thắng, không tạo candidate.
- ListNet/listwise linear tăng complement OOF lên `0.905764` nhưng locked
  holdout chỉ `0.915000`; dừng trước public.
- Rank-normalized ensemble 25% pointwise + 75% listwise đạt complement OOF
  `0.906600` nhưng locked holdout `0.914000`; dừng trước public.

Kết luận: tăng OOF đơn lẻ không đủ. Mọi nhánh tiếp theo phải vượt đồng thời
robust grouped OOF và locked holdout `0.918833` trước khi chấm/package public.

## Candidate pool c50 gate (2026-08-26)

Safe-tail PRF có oracle cao hơn khi mở c35→c50, nên Stage-2 c50 được chấm
incremental trên complement và holdout. Cấu hình pointwise-v2 khóa từ v11h tự
chọn policy pool 50 trên complement.

| Gate | c35 v11h | c50 | Delta |
|---|---:|---:|---:|
| complement grouped OOF | 0.901866 | **0.904372** | +0.002506 |
| complement robust OOF | 0.900220 | **0.901395** | +0.001176 |
| locked holdout | **0.918833** | 0.916833 | -0.002000 |

C50 thất bại strict gate vì holdout giảm; không chạy public c50 và file probe
không phải submission candidate.

Theo yêu cầu kiểm tra leaderboard dù local gate chưa thắng, đã materialize hai
public ZIP experimental sau:

- `output/v11m.zip`: prefix-512 safe-tail, local locked holdout `0.914333`;
  SHA-256 `87226BD42D4C95EFEC1495A6B55A3BBFD7B46149E377472F05D7B46FB71289DA`.
- `output/v11l.zip`: c50 pool, local locked holdout `0.916833`;
  SHA-256 `27448724F7FBFB8D6F892F042D7A3C2CF2983C425A193B137A8BBDEAEB23C69F`.

Leaderboard public xác nhận: v11l đạt `0.9181`, chỉ thấp hơn v11h `0.0002`;
v11m đạt `0.9090` và bị loại. V11h vẫn là main public `0.9183`, v11l là
fallback gần nhất; không tự động thay main theo local probe.

## Prefix-512 safe-tail dense channel (đang chạy, 2026-08-26)

Hai dense view V10.3 được fusion vào tail của Stage-1 PRF với top-5 khóa cứng.
Grid chọn trên complement giữ no-prefix weight 0 và chọn prefix-512 weight
`0.3`, RRF `k=60`.

| Split | Oracle@35 trước | Oracle@35 mới | Oracle@50 trước | Oracle@50 mới |
|---|---:|---:|---:|---:|
| complement | 0.950097 | **0.956347** | 0.959431 | **0.964500** |
| locked holdout (fixed apply) | 0.967667 | **0.974250** | 0.971750 | **0.981750** |

Stage-1 Recall@5 không đổi trên cả hai split. Incremental Stage-2 direct tăng
complement `0.870607 → 0.881903`, nhưng giảm holdout `0.900583 → 0.898083`.
Pointwise-v2 gate cuối đạt OOF `0.900752` (robust `0.898171`) và holdout
`0.914333`, đều thấp hơn v11h. Nhánh bị loại trước public; `v11m_dual_dense_probe`
không phải submission candidate.

## Research-track v12: aspect/evidence/TF-IDF/LambdaMART (2026-08-27)

Theo hướng nghiên cứu đã được rà lại, v12 giữ nguyên Stage-1 và candidate pool
c35 của v11h, chỉ thay selector. Các block được triển khai tuần tự:

1. Parser multi-aspect thuần xác định, giữ full question, legal predicate,
   clause, số và legal noun phrases; có unit test chống thay đổi thứ tự.
2. Aspect-MaxSim `max_j cosine(aspect_i, chunk_j)` trên BQB dense cache hiện có;
   aspect encoder chạy `cuda`.
3. Greedy evidence-set tối đa ba passage, thưởng coverage mới và phạt overlap;
   thêm feature so sánh với top-3 overlap thông thường.
4. TF-IDF word n-gram 1–2 và char_wb 3–5 dùng làm feature selector. Để pipeline
   ổn định RAM, char scan giới hạn 12.000 ký tự/document và 60k feature, không
   đổi candidate pool hay nhãn.
5. Grouped LambdaMART theo đúng gate @5 của nghiên cứu; LightGBM `4.7.0` được
   cài trong embedded runtime dự án.

| Gate | Recall | Precision | Trạng thái |
|---|---:|---:|---|
| selection grouped OOF | `0.910290` | `0.194812` | Không vượt v11h `0.918833` |
| locked final | `0.906114` | `0.191266` | Tăng so baseline cục bộ `0.893013`, nhưng fail promotion-v11h |

OOF fold recalls: `0.853763, 0.925325, 0.944805, 0.931818, 0.896104`, std
`0.032488`, robust recall `0.902168`. Do gate v11h không đạt, reranker
fine-tune không được mở; v12 dừng, không có public upload. Report:
`output/v12_research/v12_research_report.json`. Candidate JSON trong thư mục
này chỉ là artifact phân tích, không thay thế `output/v11h.zip`.

## Research-track V14: Jina-ColBERT late interaction (2026-08-29)

Theo roadmap, V14 giữ nguyên v11h c35 và chỉ thay slot 5 tối đa một lần. Mỗi
document được chunk pháp lý đầy đủ ở 1.800 ký tự với overlap 180; Jina-ColBERT-v2
được dùng cho full-query MaxSim, typed legal elements và exhaustive union
evidence một/hai/ba passage. Candidate/tail/delta và các biến đổi query-relative
được lưu vào cache bất biến; không fine-tune neural model.

Guard frozen được chọn trên complement sau khi so sánh các mode: candidate phải
có frozen cross-encoder margin so với tail `>=0` và Stage-1 rank `<=25`. Selector
thắng là full-query best three-passage union delta với threshold
`0.02197723388671874`, thay slot 5 tối đa một lần.

| Split | Baseline Recall@5 | Rescued Recall@5 | Delta | Successful / destructive | Kết luận |
|---|---:|---:|---:|---:|---|
| Complement grouped/OOD | 0.9018657755 | 0.9068783069 | +0.0050125313 | 8 / 1 | Pass local gates |
| Locked holdout | 0.9188333333 | 0.9178333333 | -0.0010000000 | 1 / 2 | Reject |

Complement pass không chuyển thành holdout gain; 191 interventions có
precision `0.3333333333` trên holdout. Public không được encode và `v11h.zip`
vẫn là main. Jina exact parameter count `559,497,216`; tổng deployed V14 là
`1,649,497,216` (`1.649497216B < 4B`). Report đầy đủ:
`output/v14_colbert_full/v14_colbert_report.json`; disclosure:
`V14_COLBERT_DISCLOSURE.md`.

Kết luận theo nghiên cứu: ColBERT có tín hiệu trên complement nhưng chưa robust
qua locked split, nên chưa mở LoRA/listwise specialist. Nhánh tiếp theo chỉ nên
là một robustness experiment mới, không retune V14 bằng holdout.

## Research-track V15: counterfactual pairwise adjudication (2026-08-29)

V15 đổi objective từ candidate relevance độc lập sang direct A-minus-B legal
adjudication. V11h c35 và V14 label-free prefilter được giữ nguyên; shortlist
8/10 gồm anchor năm tài liệu và 3/5 challenger. Pairwise features gồm delta
CE/v11/Stage-1, typed constraint, exclusive critical support, evidence,
contradiction, number, structure và BTC-only query rewrite sidecars. Tournament
chỉ được thay slot 5 nếu thắng tail, thắng hai neighbour, vượt aggregate tail,
qua cycle gate và qua V14 evidence safety floor.

| Metric | Result |
|---|---:|
| c35 oracle capture shortlist 8 / 10 | `43.86% / 59.65%` |
| Best complement delta | `-0.0012531328` |
| Best standard intervention precision | `0.40` |
| Best standard destructive rate | `0.0292682927` |
| Changes / successes / destructive | `205 / 4 / 6` |
| Holdout | chưa đọc |

V15 fail trước locked-holdout gate; vì vậy không chạy LoRA pairwise/group-
softmax, listwise specialist hay retrieval ngoài c35. Đây là kết quả có chủ ý
theo research gate, không phải thiếu bước. Oracle vẫn cho thấy c35 có headroom,
nhưng tín hiệu pairwise algorithmic hiện chưa đủ precision để khai thác an toàn.
Report và logic đầy đủ nằm ở `output/v15_counterfactual/` và
`V15_COUNTERFACTUAL_DISCLOSURE.md`; public main vẫn là `output/v11h.zip`.

V15 thêm `0` tham số: end-to-end v11h embedding `330M` + cross-encoder `760M`
= `1.09B <4B`.

## Cleanup checkpoint — 2026-08-29

Đã hoàn tất cleanup sau khi chốt v11h. Xoá 80 file output trung gian gồm các
sweep/ablation của V11, fallback v11/v10 cũ, các submission không còn là main,
và các artifact c50 không nằm trong chain v11h. Các thư mục index cũ và thư mục
thí nghiệm V12–V15 đã được dọn ở checkpoint trước; bản model trùng trong
`Task1 - LegalIR/models/` cũng đã được loại bỏ.

Manifest còn giữ cho tái lập end-to-end:

- `data/raw/public_test/`.
- `data/processed/v11_research_fit_256/`, `v11_research_256/`,
  `v11_research_complement_256/`.
- `output/v10_9_prf/` chỉ giữ safe-tail c35/report/stage1 cần cho V11g.
- `output/v11_feature_cache/`, `output/v11_tiny_ltr_complement_no_prior/`,
  `output/v11g_prf_ltr/`, `output/v11h_query_norm_ltr/`.
- Các audit stage-1/stage-2 c35 cần cho chain và `v11f.zip`, `v11g.zip`,
  `v11h.zip`; trong đó chỉ `v11h.zip` là artifact nộp.
- Source, tests, runner `run_v11f_end_to_end.ps1`,
  `run_v11g_end_to_end.ps1`, `run_v11h_end_to_end.ps1`, cùng các disclosure
  vẫn còn nguyên.

Các output đã xoá không còn bản sao cục bộ nhưng có thể rebuild từ manifest trên.
Hash `v11h.zip` sau cleanup vẫn là
`57CDEF271C86824B128482AD141E140224E8002B7ACC7536CB79DAE90EB7F175`.

## Single-file presentation runner — 2026-08-29

Đã hợp nhất logic thực thi của V11f, V11g và V11h vào
`experiments/track_b_v11_research/run_v11h_end_to_end.ps1`. File này chứa ba
stage nội bộ theo đúng thứ tự: F dựng compliant index/retrieval, G chạy
safe-tail PRF/citation + LTR v1, H chạy query-normalized LTR v2 và package
`output/v11h.zip`. Runner không còn gọi `run_v11f_end_to_end.ps1` hay
`run_v11g_end_to_end.ps1`; hai file cũ chỉ là lineage/audit tham khảo.

Việc hợp nhất không thêm model hay tham số: embedding `330M` + reranker `760M`
+ selector logistic rất nhỏ = khoảng `1.09B`, vẫn strictly `<4B`. Các output
F/G chỉ là checkpoint local để đối chiếu; artifact trình bày/nộp duy nhất là
`v11h.zip`.
