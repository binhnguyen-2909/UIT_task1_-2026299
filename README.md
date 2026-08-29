# UIT DSC — LegalIR retained workspace

Workspace này giữ ba mục đích duy nhất:

1. **v11 end-to-end compliant**: legal chunking + BQB embedding + Vietnamese
   reranker, tổng `1.09B < 4B`.
2. **v10 base**: pipeline fixed-chunk để làm baseline tái lập.
3. **v57 algorithm reference**: exact legal-heading rescue và rank-cache để
   nghiên cứu; không dùng v57 model/output làm input cho v11.

## Chạy lại v11 full

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  "D:\projects\UIT DSC\Task1 - LegalIR\experiments\track_b_v11_research\run_v11_research.ps1" `
  -Device cuda -Resume
```

Index mặc định là `Task1 - LegalIR/data/processed/v11_research_256`. Bỏ `-Resume`
nếu muốn build lại index từ đầu. Models cần thiết là:

- `models/bqbbao6_vietnamese-legal-embedding` — 330M;
- `models/thanhtantran_Vietnamese_Reranker` — 760M.

## Chạy lại v10 base

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  "D:\projects\UIT DSC\Task1 - LegalIR\experiments\track_b_v10\run_v10_rebuild.ps1" `
  -Device cuda -Resume
```

## v57 algorithm reference

- `Task1 - LegalIR/scripts/benchmark_exact_heading_rescue.py`
- `Task1 - LegalIR/scripts/run_v57_exact10.ps1`
- `Task1 - LegalIR/output/current/v57_exact10/`
- `Task1 - LegalIR/V49_PIPELINE_DISCLOSURE.md`

Đây chỉ là tài liệu/thuật toán tham khảo. V57 lineage vượt giới hạn BTC khi
tính end-to-end, nên không được dùng model/checkpoint/prediction của v57 cho v11.

## Kết quả đã xác nhận

- Public anchor `v11c.zip`: Recall `0.903666668`, Precision `0.193800001`.
- Candidate kế tiếp `v11d.zip`: c35 reranker + exact-heading cutoff80/t10,
  holdout Recall `0.90275`, Precision `0.1924`.
- `v11d.zip` đã kiểm tra contract: 1000 question IDs khớp public-official,
  mỗi câu có đúng 5 document IDs duy nhất; model plan `1.09B < 4B`.

Runbook chi tiết nằm ở `Task1 - LegalIR/experiments/track_b_v11_research/`.
