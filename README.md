# UIT DSC — LegalIR v11h

Workspace này giữ pipeline **LegalIR v11h end-to-end** làm bản chính. Runner
đơn file chứa ba stage nội bộ F → G → H, dùng tổng khoảng `1.09B < 4B` tham số:

- BQB Vietnamese legal embedding: khoảng `330M`;
- Vietnamese reranker: khoảng `760M`;
- query-normalized logistic selector: rất nhỏ, không đáng kể.

## Chạy lại v11h

Từ repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  "D:\projects\UIT DSC\Task1 - LegalIR\experiments\track_b_v11_research\run_v11h_end_to_end.ps1" `
  -Device cuda
```

Khi GPU không khả dụng:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  "D:\projects\UIT DSC\Task1 - LegalIR\experiments\track_b_v11_research\run_v11h_end_to_end.ps1" `
  -Device cpu
```

Runner tự dùng cache hợp lệ, ưu tiên `.python312-embed\python.exe` nếu có và
fallback sang `python` trong PATH. Cài dependency khi clone repo:

```powershell
python -m pip install -r requirements-core.txt -r requirements-neural.txt
```

Hai model cần được đặt local tại:

- `models/bqbbao6_vietnamese-legal-embedding`;
- `models/thanhtantran_Vietnamese_Reranker`.

Dữ liệu BTC, model, runtime Python, output và artifact trung gian không nằm
trong GitHub; chúng bị `.gitignore` vì có thể rebuild hoặc cung cấp riêng.
Artifact nộp chính là `Task1 - LegalIR/output/v11h.zip`.

## Kết quả đã khóa

- Public Recall: `0.9183`;
- locked holdout Recall@5: `0.9188333333`;
- SHA-256 `v11h.zip`:
  `57CDEF271C86824B128482AD141E140224E8002B7ACC7536CB79DAE90EB7F175`;
- kiểm thử hiện tại: `46 passed`.

## Tài liệu

- `Task1 - LegalIR/FINAL_V11H_END_TO_END_SUMMARY.md`: luồng chạy và model
  budget;
- `Task1 - LegalIR/EXPERIMENT_LOG.md`: toàn bộ thử nghiệm, metric và quyết
  định promote/reject;
- `Task1 - LegalIR/CURRENT_STATE.md`: checkpoint mới nhất;
- `Task1 - LegalIR/V49_PIPELINE_DISCLOSURE.md`,
  `V14_COLBERT_DISCLOSURE.md`, `V15_COUNTERFACTUAL_DISCLOSURE.md`: các hướng
  nghiên cứu đã thử và lý do không thay v11h.
