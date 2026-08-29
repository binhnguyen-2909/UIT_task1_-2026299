# LegalIR — trạng thái hiện tại

Cập nhật: 2026-08-29

## Main compliant pipeline

- Candidate chính: `output/v11h.zip`.
- Public score: **`0.9183`**.
- Locked holdout Recall: `0.9188333`.
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
