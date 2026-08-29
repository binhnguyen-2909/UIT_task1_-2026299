# LegalIR — Kế hoạch cải tiến Rule-based (Stage A2)

## Bối cảnh

- Stage A đã nộp: `submission_rule_based.zip`, public Recall = **0.8031**.
- Holdout BTC-score hiện tại: 0.7738 (memory), 0.6969 (strict).
- 990/1000 holdout có gold trong candidate union top-100 của các source hiện tại.
  → Selector RRF chưa tận dụng hết pool. Còn ~0.12 recall gap nếu chọn tối ưu.

## Mục tiêu Stage A2

Đẩy public recall rule-based lên mức trần cao nhất có thể (ước lượng 0.83–0.86) bằng
cách mở rộng pool và tune lại selector, **không dùng tham số neural** (vẫn 0B).

## Các cải tiến đề xuất

### A2.1 — Mở rộng candidate pool (thêm 4 source mới)

Tất cả là lexical rule, 0 tham số:

1. **`bigram_doc`**: lấy tất cả bigram từ câu hỏi (>=2 token không chứa stopword
   cả hai bên), truy vấn `documents_fts` bằng phrase `"a b"`. Bổ sung candidate
   chứa n-gram đặc thù mà chunk_phrase (chỉ lấy cặp adjacent trong tokens(keep_stopwords))
   có thể đã lọc mất.

2. **`trigram_doc`**: tương tự với 3-gram, nhưng giới hạn các token "đặc trưng"
   (bỏ stopword và số). Ưu tiên các cụm có chứa tên văn bản, tên cơ quan, năm.

3. **`exact_heading_window`**: lấy các cụm 7-12 token liên tiếp trong câu hỏi
   (khác `exact10` chỉ 10). Trả về doc có chứa cụm đó (trong heading/title hoặc
   trong body). Đây là source riêng để dùng kết hợp với gate cẩn thận ở A2.3.

4. **`rare_and_doc`**: OR của 6-8 token hiếm nhất với title boost cao. Khác
   `focused` ở chỗ dùng OR (không AND), có thể cover câu hỏi không có đủ 3 token
   cùng xuất hiện.

### A2.2 — Tune lại grid selector

- Thử 4-5 weights per source, nhiều preset hơn.
- Thêm `rrf_k ∈ {10, 15, 20, 25, 30, 40, 60}`.
- Thêm `support_cutoff ∈ {1, 2, 3, 4, 5}` (cutoff=1 nghĩa mọi nguồn đều count).
- Thêm `support_bonus ∈ {0, 0.01, 0.02, 0.05, 0.1, 0.2}`.
- Có thể thêm `top_per_source` để cắt mỗi source ở 30 hoặc 50 thay vì 100.

Dùng 5-fold OOF trên fit; khóa trên holdout 1 lần.

### A2.3 — Exact heading gate cẩn thận (cải tiến từ v58 thất bại)

Bài học v58: 374 thay đổi public làm domain-shift. v57: 99 thay đổi, +0.0015.

Đề xuất A2.3:
- Dùng cụm 7-9 token (không chỉ 10 như v57).
- **Gate 3 điều kiện** thay vì 1:
  1. Cụm phải xuất hiện trong **heading** của candidate (ưu tiên `Điều X` hoặc
     `Khoản X` line). Không đủ thì thử trong body.
  2. Candidate chưa có trong top-4 hiện tại (chỉ thay slot 5).
  3. Cụm chứa ≥1 cụm danh t� riêng (số văn bản, năm, tên viết tắt) HOẶC
     cụm chứa ≥2 token thuộc `title_postings` của doc đó.
- Giới hạn thay đổi tối đa ~30-50 câu public (giảm so với 99 v57).

### A2.4 — Citation completion có điều kiện

- Lấy legal reference trong top-4 doc hiện tại (số văn bản dạng `XX/YYYY/NĐ-CP`
  hoặc tương tự).
- Nếu câu hỏi yêu cầu nhiều căn cứ ("căn cứ", "theo quy định", "được dẫn chiếu"):
  → thêm doc chứa cùng legal reference làm candidate bổ sung.
- Gate: chỉ thêm nếu doc mới không trùng top-4, có title overlap với câu hỏi.

## Giao thức đánh giá

Mỗi thử nghiệm phải:
- Chạy 5-fold OOF trên fit.
- Đo holdout recall + precision 1 lần.
- Đếm số câu thay đổi so với baseline A.
- Tạo `submission_*.zip` riêng cho từng thử nghiệm để đối chiếu.
- Lưu `report_*.json` với đầy đủ metric.

## Output Stage A2

- File: `output/stage_submissions/stageA2_rule_based_extended.zip`
- Báo cáo: `experiments/rule_based_ceiling/artifacts/report_stageA2.json`
- Audit: so sánh số câu thay đổi so với Stage A (Public 0.8031).

## Tiêu chí dừng

- Nếu 3 lần thử liên tiếp mà public không tăng ≥+0.005 → dừng stage A, sang B.
- Nếu holdout recall vượt 0.85 → có tín hiệu mạnh để nộp.
- Nếu tổng số câu thay đổi > 50 mà recall không tăng → thu hẹp gate.

## Không làm

- Không thêm neural embedding/reranker (giữ 0B parameters cho stage này).
- Không sửa đổi format submission hay scorer.
- Không dùng leakage (gold của test) để tune.
- Không thay đổi schema SQLite index hiện tại (chỉ thêm hàm retrieval mới).

## V15 addendum — post-V14 research direction (2026-08-29)

Các mục A2 cũ không còn là thứ tự triển khai hiện hành. Theo hướng nghiên cứu
mới, ưu tiên là **difference-first / counterfactual pairwise legal
adjudication** trên candidate pool c35 của v11h:

1. Đo oracle slot-5 trên shortlist 8/10 sau cầu nối prefilter c35.
2. So sánh trực tiếp `(q, dA, dB)` bằng typed legal differences, exclusive
   critical evidence, weakest constraint và consistency/cycle gate; tournament
   chỉ thay slot 5.
3. Chạy BTC-only deterministic query rewrite sidecars và drop rule kém trên
   complement.
4. Chỉ khi standard/law-family/template OOD, intervention precision,
   destructive rate và worst-fold gate cùng đạt mới mở LoRA pairwise/group-
   softmax trên reranker hiện tại.
5. Listwise specialist ~1B và retrieval ngoài c35 giữ **đóng**; chỉ mở theo
   điều kiện transfer/headroom và residual audit được ghi trong V15 disclosure.

Checkpoint V15 algorithmic đã fail gate (`-0.001253` best complement recall
delta), nên các bước neural/listwise/retrieval chưa được phép chạy. End-to-end
budget hiện hành vẫn `1.09B <4B`, v11h public được giữ nguyên.
