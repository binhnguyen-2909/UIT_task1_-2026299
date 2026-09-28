# Lessons

(anti-patterns and recurring mistakes)

- L-001: Đổi passage pack là phải sinh lại downstream scores — cấm tái dùng scores cũ.
- L-002: EMPTY_BODY veto cứng — V16b từng chèn doc rỗng 131890 vào slot-5 public.
- L-003: Probe AdamW exact phải pass trước E2 dài; cấm đổi lén optimizer/quantization.
- L-004: Standard OOF không thay thế 3 stress regime.
- L-005: Cấm dùng nhãn holdout/public để chọn ngưỡng/policy ở E1/E2/E3; E4 là lần mở duy nhất.
- L-006: Head rescue-only nhỏ lặp lại cùng thất bại — học dense relevance, đánh giá rescue như outcome.
