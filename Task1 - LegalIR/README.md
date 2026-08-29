# UIT DSC Task 1 — LegalIR v11h main

Pipeline chính hiện tại là **v11h**, được thiết kế để chạy lại end-to-end dưới
giới hạn 4B tham số. V10 chỉ được giữ làm baseline/fallback; v57 exact10 được
giữ để tham khảo thuật toán vì lineage đầy đủ của v57 vượt ngân sách 4B.
Runner nộp/trình bày là một file duy nhất; bên trong có ba stage nội bộ F → G → H
để giữ đúng logic và thứ tự tái lập.

## Trạng thái submission

| File | Điểm/metric đã biết | Trạng thái |
|---|---:|---|
| `output/v11h.zip` | public **`0.9183`**; locked holdout `0.9188333` | **Main compliant** |
| `output/v11g.zip` | public `0.9086`; locked holdout `0.9120000` | Ablation feature v1 |
| `output/v11f.zip` | public `0.9106`; locked holdout `0.9079167` | Fallback compliant trước PRF |
| v57 exact10 | public `0.9190` | Chỉ tham khảo thuật toán; end-to-end >4B |

Model plan v11:

```text
BQB Vietnamese legal embedding  0.33B
Vietnamese reranker             0.76B
Tiny supervised LTR            <0.00001B
                                ---------
                                ~1.09B < 4B
```

## V11h làm gì

1. Stage-1 kết hợp legal FTS/chunk phrase/citation với BQB dense retrieval.
2. Corpus-only PRF và legal-citation completion bổ sung candidate ở phần đuôi;
   top-5 Stage-1 được khóa để không làm hỏng head ordering.
3. Stage-2 chấm 35 document, tối đa 3 legal chunks/document bằng Vietnamese
   reranker, max length 768, FP16.
4. Feature v2 chuẩn hóa neural score theo từng query, giữ provenance trước PRF,
   rank agreement, exact phrase/title và các tín hiệu tỷ lệ độ dài.
5. Tiny pointwise-logistic LTR (`C=1`) học fusion từ 1.197 câu complement sạch
   bằng grouped out-of-fold CV.
6. Supervised `document_prior` bị vô hiệu hóa vì locked test đã chứng minh nó
   gây label leakage.
7. Exact-heading threshold 10 được dùng làm rescue bảo thủ sau LTR.

Safe-tail PRF tăng locked candidate oracle@35 từ `0.960667` lên `0.967667`
trong khi giữ nguyên Stage-1 Recall@5. Trên complement grouped OOF, v11h đạt
Recall `0.901866` với fold std `0.006585`. Cấu hình khóa trước đạt Recall
`0.918833` trên holdout 1.000 câu và public `0.9183`. Public gần locked holdout
trong `0.000534`, cho thấy gate hiện tại transfer tốt hơn các bản trước.

## Chạy lại

Runner v11h end-to-end (tự bỏ qua cache hợp lệ đã có, dùng RTK cho output lớn):

```powershell
Task1 - LegalIR\experiments\track_b_v11_research\run_v11h_end_to_end.ps1 -Device cuda
```

Runner ưu tiên `\.python312-embed\python.exe` nếu runtime local còn tồn tại;
nếu không, nó tự fallback sang `python` trong PATH. Vì vậy runtime nhúng và
output/checkpoint đều bị `.gitignore`, không cần push lên repository. Sau khi
clone, cài dependency bằng:

```powershell
python -m pip install -r requirements-core.txt -r requirements-neural.txt
```

Hai model vẫn cần được đặt ở workspace root:
`models/bqbbao6_vietnamese-legal-embedding/` và
`models/thanhtantran_Vietnamese_Reranker/`.

Các runner `run_v11f_end_to_end.ps1` và `run_v11g_end_to_end.ps1` chỉ giữ làm
tài liệu lineage; runner chính không gọi chúng.

Các bước complement/LTR có script tái lập riêng:

- `scripts/prepare_v11_complement_split.py`
- `scripts/parallel_stage1_infer.py`
- `scripts/corpus_prf_stage1.py`
- `scripts/batch_stage2_rerank.py`
- `scripts/v11_tiny_ltr_complement.py`
- `scripts/package_submission.py`

Chi tiết command, leakage gate, metric và artifact nằm trong
`experiments/track_b_v11_research/EXPERIMENT_LOG.md`.

## Artifact tham khảo

- Các kết quả V10/V57 và toàn bộ ablation được ghi trong
  `EXPERIMENT_LOG.md`, `CURRENT_STATE.md` và các disclosure. Không dùng
  checkpoint/prediction cũ trong v11h; chỉ tái sử dụng ý tưởng đã qua gate.
