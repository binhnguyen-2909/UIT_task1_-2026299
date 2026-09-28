# E3 — residual top-5 policy (Task1)

**Anchor:** `output/v16b_memory_slot5.zip` (public 0.9210, holdout 0.9228333). Holdout+public khóa.

## Stages

<!-- mb-stage:1 --> **Stage 1 — khóa inputs.** Chỉ đọc V16b memory output + E2 OOF relevance (cross-fit). Universe giữ c35. DoD: liệt kê SHA inputs + provenance split; không mở holdout/public.
<!-- mb-stage:2 --> **Stage 2 — edit contract.** Đóng băng top-4, tối đa 1 thay slot-5, abstain về V16b, EMPTY_BODY veto, quy tắc duplicate-family preregister. DoD: contract viết xong trước khi chọn ngưỡng.
<!-- mb-stage:3 --> **Stage 3 — gate mọi regime.** Báo cáo standard + law_source + template + near_dup: gain, robust gain, bootstrap lower, precision, destructive. DoD: chỉ tiến E4 khi mọi regime đạt minima preregister; ngược lại dừng.
<!-- mb-stage:4 --> **Stage 4 — bàn giao E4.** Một lần parity holdout, không retrain/retune sau mở. DoD: vượt 0.9228333 + provenance sạch, hoặc giữ anchor.
