# Task1 LegalIR E0–E2 — Status

**Current phase:** R1 C35-Long-Fusion auto-chain ĐANG CHẠY (cache → eval-preflight → standard → stress có điều kiện); E5 sealed giữ nguyên
**Focus:** `output/r1_c35_long_fusion/` (tmux `r1_auto`, GPU1, resume QID); chờ cache 41.895 pairs xong → gate standard/stress
**Blockers:** none (E3B-retain OOM 16/9 đã dừng theo lệnh user; R1 không phụ thuộc retain)

## Metrics

- V16b anchor (`output/v16b_memory_slot5.zip`): public 0.9210, holdout Recall@5 0.9228333
- V11h baseline: public 0.9183, holdout 0.918833; budget 1.09001B < 4B (BQB 330M + CE 760M + tiny selector)
- E0 (`output/v16_rgm/`): đủ complement/holdout/public memory features + oracle reports + locked predictions
- E1 (`output/e1_evidence_frozen_ce/`): evidence.jsonl + meta + scores.jsonl + predictions + report — Recall@5 0.8747842 vs anchor 0.8860624 (delta −0.01128); c35 oracle 0.9503899; KHÔNG promote
- E2 (`output/e2_mil_oof/`): standard report (2 seeds × 5 folds, mean Recall 0.8938945, delta +0.00783) + stress report (30/30 folds xong, cả 3 regime gain dương) + memory probe (fit=true); status `oof_only_not_promoted`
- Headroom trong c35: +0.044833; cần +0.007167 để tới 0.93
- E3-B (`output/e3b_nested_mil/`): standard 2026-09-11 xong 10/10 folds (mean gain +0.01219 / +0.01491), gate pass, `promotion_to_stress_allowed:true`; stress 2026-09-13 xong 30/30 folds, cả 3 regime pass (law +0.01114/+0.01086, near_dup +0.01295/+0.01344, template +0.00988/+0.01428), `promotion_to_e4_allowed:true`, `holdout_opened:false`
- E3B-retain (`output/e3b_nested_mil_retain/`): DỪNG 2026-09-16 13:47 sau 5 folds `inner=20260910` (10 weights 2.2G) — CUDA OOM GPU0 (free 90MiB/79G, job khác 30G); checkpoint resumable, đã dừng theo lệnh user để nhường GPU cho R1
- R1 (`output/r1_c35_long_fusion/`): patch v2 SHA `B388861A…3F8340`, 20 files; preflight pass (1197 Q, 41.895 pairs, est 3.21B <4B); qwen-preflight pass; pilot 64/64 QID ~1000s, ledger thực 3.205B <4B (2.61B + 595.7M), `holdout_opened:false`; fix bug `r1_contract` NameError (import trực tiếp, tests 3/3); full cache auto-chain GPU1 đang chạy; TỐI ƯU runtime-only 16/9 (reserve 5G → batch 8, threads 4, expandable_segments; lineage `2388eb57…` giữ nguyên): pace 13.5s/QID, ETA cache ~4.1h

## E đối chiếu (audit 2026-09-10)

- ✅ E0 đầy đủ: complement/holdout/public features + oracle + `v16b_memory_ltr_report.json` + slot5 json
- ✅ E1 đầy đủ 5 artifact + protocol `E1_EVIDENCE_FROZEN_CE_PROTOCOL.md`
- ✅ E2 đầy đủ: protocol + `e2_mil_finetune_oof.py` + test + standard/stress/probe reports
- ⬜ E3 chưa mở (policy + gate); E4 khóa (single parity run); E5 khóa (materialize)
- ✅ E3-B đầy đủ: `e3b_standard_report.json` (2026-09-11) + `e3b_stress_report.json` (2026-09-13) + progress/checkpoint resumable; lineage sạch (stage1 `4b16c37...`, complement-only, 997/1000 memory sau overlap-filter)
- Capsule chạy lại đầy đủ: `run_e0_e2_auto.sh`, `SERVER_AUTO_RUN.md`, GPUHub + RTX5070 runbooks, 2 zip Server_Auto

## Active plans

<!-- mb-active-plans -->
- [2026-09-17] [plans/2026-09-17_refactor_hsrstandardetaoptimization.md](plans/2026-09-17_refactor_hsrstandardetaoptimization.md) — refactor — hsrstandardetaoptimization
<!-- /mb-active-plans -->

- `Task1 - LegalIR/E0_E5_CURRENT_PLAN.md` (protocol gốc) + bank mirror `.memory-bank/plans/2026-09-10_e3_residual_policy.md`

## Recently done

- 2026-09-09 E2 stress xong; 2026-09-08 E1 seal + capsule ship; 2026-09-01 forensics V16b + seal config V27-A
- 2026-09-11 E3-B standard pass (6 attempts, OOM retries); 2026-09-13 E3-B stress pass (7 attempts), E4 ready chưa chạy
- 2026-09-14 E4 PASS (recall 0.928 vs V16b 0.9228333, +0.0051667, 12 rescues > 7 destructive) + E5 sealed (`e5_public.zip` SHA `3c250554…af027b`)
- 2026-09-16 E3B-retain OOM GPU0 → dừng theo lệnh user; R1 patch v2 cài + preflight/pilot pass, auto-chain (`/tmp/r1_auto_chain.sh`, tmux `r1_auto`, GPU1) đang chạy cache → eval-preflight → standard → stress (có điều kiện)
