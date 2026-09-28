# UIT DSC — LegalIR v16b (v11h c35 + QREL-memory slot-5 gate)

Workspace này giữ pipeline **LegalIR v16b** làm candidate chính: v11h
end-to-end với gate thay tối đa slot 5 trên c35. V16b là anchor đã khóa;
không có nhánh nghiên cứu mới nào được promote mặc định.
Tổng khoảng `1.09001B < 4B` tham số:

- BQB Vietnamese legal embedding: khoảng `330M`;
- Vietnamese reranker: khoảng `760M`;
- query-normalized logistic selector: rất nhỏ, không đáng kể.

## Chạy lại V16b

V16b dùng lại toàn bộ index/audit/cache v11h; không mở rộng candidate pool và
không chạy thêm CrossEncoder:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  "D:\projects\UIT DSC\Task1 - LegalIR\experiments\track_b_v11_research\run_v16b_memory_slot5.ps1" `
  -Device cuda
```

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
Artifact nộp chính là `Task1 - LegalIR/output/v16b_memory_slot5.zip`;
`output/v11h.zip` được giữ làm baseline.

## Kết quả đã khóa

- V16b locked holdout Recall@5: `0.9228333333` (`+0.0040000` so với v11h);
- Public leaderboard score: `0.9210` cho `v16b_memory_slot5.zip`;
- V16b policy: `c35`, memory `degree_corrected_memory_score`, chỉ thay slot 5;
- SHA-256 `v16b_memory_slot5.zip`:
  `B2816BD73AE23FBDFADD6563DC43728896F078A6AD8032B6743984B3AADADFD9`;
- v11h baseline public Recall: `0.9183`, locked holdout: `0.9188333333`;
- SHA-256 `v11h.zip`:
  `57CDEF271C86824B128482AD141E140224E8002B7ACC7536CB79DAE90EB7F175`;
- V16b validation: c35 contract, under-4B check và submission package đều pass.

## Tài liệu

- `Task1 - LegalIR/FINAL_V11H_END_TO_END_SUMMARY.md`: luồng chạy và model
  budget;
- `Task1 - LegalIR/EXPERIMENT_LOG.md`: toàn bộ thử nghiệm, metric và quyết
  định promote/reject;
- `Task1 - LegalIR/CURRENT_STATE.md`: checkpoint mới nhất;
- `Task1 - LegalIR/V49_PIPELINE_DISCLOSURE.md`,
  `V14_COLBERT_DISCLOSURE.md`, `V15_COUNTERFACTUAL_DISCLOSURE.md`: các hướng
  nghiên cứu đã thử và lý do không thay v11h.
- `Task1 - LegalIR/V16_RGM_DISCLOSURE.md`: QREL-memory oracle và V16b
  slot-5-gate checkpoint, cùng V16d-lite diagnostic.
- `Task1 - LegalIR/V17_BOUNDARY_DISCLOSURE.md`: V17a-lite setwise boundary
  diagnostic và lý do reject trước holdout.
- `Task1 - LegalIR/V16_BOUNDARY_RESEARCH_DISCLOSURE.md`: closure P0--P4,
  OOD/paired-bootstrap, các SVG diagnostic và lý do P5/P6 không được mở.
- `Task1 - LegalIR/V20_PDG_PROTOCOL.md`: pairwise deferral gate mới, P0/P1
  promotion gates, top-4 freeze và đường chạy CPU-only.
