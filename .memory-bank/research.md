# Task1 LegalIR E0–E2 — Research Log

## Hypotheses

| ID | Hypothesis | Status | Metric | Target | Result | EXP |
|----|------------|--------|--------|--------|--------|-----|
| H-001 | Nút thắt là ranking trong c35, không phải candidate generation | ✅ CONFIRMED | oracle − V16b | headroom > need | headroom +0.044833, cần +0.007167 cho 0.93 | forensics 2026-09-01 |
| H-002 | E1 frozen-CE thắng anchor | 🔴 REFUTED | complement gain | > 0 | −0.01128 (0.87478 vs 0.88606); oracle 0.95039 | e1 report 2026-09-08 |
| H-003 | E2 MIL relevance+boundary cải thiện OOF | ✅ CONFIRMED (via E3-B) | OOF gain mọi regime | gain + safety | standard +0.01219/+0.01491; stress 3/3 regime dương, E3-B pass 2026-09-13 | e3b reports 2026-09-11/13 |
| H-004 | Rescue-head nhỏ sửa slot-5 an toàn | 🔴 REFUTED | gain + precision + bootstrap | robust | gain nhỏ, precision thấp, transfer âm (V22/V23) | forensics §8 |
| H-005 | E3-B nested MIL (fresh refit per outer fold) qua gate safety mọi regime | ✅ CONFIRMED | gain + rescue>destructive per inner seed | all pass | standard pass + stress 30/30 pass, promotion_to_e4:true | e3b_standard/stress 2026-09-13 |
| H-006 | Qwen0.6B-long fusion trên c35 (lambda 0/0.25/0.5, giữ top-4, chỉ slot-5) thêm signal vượt E3B | 🔬 TESTING | standard/stress gain + safety + bootstrap | pass gate R1 | preflight/pilot pass (ledger 3.205B <4B); cache+eval đang chạy auto | r1 2026-09-16 |

## Confirmed Findings

- F-001: V16b = BQB 330M + CE 760M + tiny selector = 1.09001B; public 0.9210, holdout 0.9228333; chuyển V11h→V16b: 177 đổi, 9 cứu, 4 phá, 164 trung tính.
- F-002: E1 seal label-blind: 1197 q complement, 41895 rows, 435785 proposals (3 nguồn × ≤4), 142205 selected (MMR 3–4, margin 0.35); CE frozen fp16/768.
- F-003: Lỗi memory cụ thể: doc rỗng 131890 chèn slot-5 public (neural rank 20, memory rank 1) — mọi policy sau cần EMPTY_BODY veto.
- F-004: Doc dài >20k tokens: c35 tìm được gold (0.9802) nhưng rerank 3-passage không đẩy lên; gold mất title bị LTR làm hại (0.9189→0.8919).
- F-005: E2 loss 0.55 relevance BCE + 0.30 boundary pairwise (trọng số theo khoảng cách c35 tới rank-5) + 0.15 anchor tanh-MSE; mỗi outer fold nạp lại CE gốc.
- F-006: E3-B nested separation (A-fit/A-stop/C/V group-disjoint, 2 neural seeds từ checkpoint gốc, policy chọn trên C apply một lần lên V) cho gain dương mọi regime + precision 0.54-1.0; abstain về V16b khi utility ≤ tau.

## Current Experiment

- E3-B standard+stress xong 30/30 folds, `promotion_to_e4_allowed:true`. E4 đã PASS 2026-09-14 (0.928 vs 0.9228333), E5 sealed.
- R1 auto-chain 2026-09-16: cache Qwen c35-long 41.895 pairs (GPU1, resume QID) → eval-preflight → standard (lambda 0/0.25/0.5, parity lambda=0) → stress có điều kiện. Câu hỏi mở: fusion lambda>0 có vượt E3B OOF + safety mọi regime không.
