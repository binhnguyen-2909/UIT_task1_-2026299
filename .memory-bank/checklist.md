# Task1 LegalIR E0–E2 — Checklist

## E3 — residual top-5 policy (XONG E3-B, 2026-09-13)

- ✅ Chốt inputs E3 (V16b memory + E2 relevance, cross-fit, nested inner/outer)
- ✅ Viết edit contract (đóng băng top-4, tối đa 1 thay slot-5, abstain về V16b, EMPTY_BODY veto cứng)
- ✅ Đánh giá standard + 3 stress regime (gain + safety + bootstrap + precision + destructive cap) — standard pass 2026-09-11, stress 30/30 pass 2026-09-13, promotion_to_e4_allowed:true

## E3-C — nested duo LoRA (BLOCKED → ĐANG MỞ ĐƯỜNG, 2026-09-14)

- ✅ Preflight E3C fail-closed đúng: thiếu `e3b_weights/**` (E3B cũ RETAIN=false, retained_path=null)
- ✅ Sửa checkpoint (TDD, 24/24 tests): `atomic_torch_save` + `verify` + manifest lineage trong `v27_a_common`; đấu nối E3B `_retain_weights`, E3C training ckpt, E4 `_save_state_dict`; runner hỗ trợ `NEURAL_SEEDS`/`INNER_SEEDS`
- ⬜ Chốt E3B-retain: full-80 chính thức vs factory-40 (script khóa cứng 2 neural seeds; lọc cần sửa gate + ghi lineage mới) — TẠM DỪNG 2026-09-16: OOM GPU0 sau 5 folds inner=20260910, checkpoint resumable, dừng theo lệnh user nhường GPU cho R1
- ⬜ E3C pilot 1 inner-seed (verify save/load process mới, splits, c35/top-4) → standard 10 → stress 30 theo gate — BLOCKED tới khi retain xong (R1 không phụ thuộc E3C)

## R1 — C35 long fusion Qwen0.6B (AUTO-CHAIN ĐANG CHẠY, 2026-09-16)

- ✅ Patch v2 SHA `B388861A…3F8340`, 20 files tại capsule root; fix bug `r1_contract` NameError trong `r1_qwen_c35_cache.py` (tests R1 3/3 pass)
- ✅ Preflight pass: 1197 Q, 41.895 pairs, est 3.21B <4B, `holdout_opened:false`; qwen-preflight pass (lineage `2388eb57…`)
- ✅ Pilot 64/64 QID ~1000s: ledger thực 3.205B <4B, batch=1 (reserve 12G), `under_4b:true`
- ⬜ Full cache 1197 QID (~5-6h, tmux `r1_auto`, GPU1, resume QID, `/tmp/r1_auto_chain.sh`) → eval-preflight → standard → stress chỉ khi `promotion_to_stress_allowed:true`

## E4 — holdout parity (PASS 2026-09-13, phát hiện 2026-09-14)

- ✅ Parity đã chạy 1 lần duy nhất: recall `0.928` vs V16b `0.9228333`, gain `+0.0051667`, rescues `12 > 7` destructive → `promotion_to_e5_allowed:true`
- ✅ 2 deployment models verify-load OK process mới (393 keys mỗi bản)

## E4 — holdout parity (READY, CHƯA CHẠY)

- ⬜ Một lần parity duy nhất bằng pack/manifest holdout, không retrain/tune sau mở
- ⬜ Promote chỉ khi vượt holdout V16b 0.9228333 + provenance sạch

## E5 — public (XONG 2026-09-14)

- ✅ Materialize 1000 QID × 5 ID unique, QID-match official, labels đóng, budget 2.61B <4B
- ✅ Seal `output/e5_public.zip` (SHA `3c250554832e433c2bdfb744f8ceb8cd8f6509c8a3d2db1ad346744b68af027b`), 455 queries đổi slot-5 vs V16b
- ⬜ Nộp leaderboard lấy public score

## Housekeeping

- ⬜ Chạy `tests/test_e2_mil_finetune_oof.py` trên cây hiện tại
- ✅ Ghi quyết định E3 (mở/bỏ) kèm số gate vào progress.md — E3-B pass 2026-09-13, xem progress.md

<!-- mb-plan:2026-09-17_refactor_hsrstandardetaoptimization.md -->
## Stage 1: Lock the numerical and runbook contracts
- ⬜ Lock the numerical and runbook contracts

<!-- mb-plan:2026-09-17_refactor_hsrstandardetaoptimization.md -->
## Stage 2: Vectorise the fit path without changing the optimisation contract
- ⬜ Vectorise the fit path without changing the optimisation contract

<!-- mb-plan:2026-09-17_refactor_hsrstandardetaoptimization.md -->
## Stage 3: Benchmark and restart the official standard run safely
- ⬜ Benchmark and restart the official standard run safely
