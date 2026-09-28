# Progress

(append-only — never delete old entries)

## 2026-09-10 — Kiểm kê Task1 + dựng bank (UIT DSC/.memory-bank)

- Đối chiếu E: E0 đủ (features+oracle+predictions complement/holdout/public); E1 đủ 5 artifact + protocol; E2 đủ protocol+script+test+standard/stress/probe; E3/E4/E5 đúng trạng thái mở/khóa.
- Ghi status/checklist/roadmap/research/backlog + plan E3 mirror. Tiếp theo: quyết định policy E3, giữ holdout khóa.
- Lưu ý: bản copy `tritruong/LegalIR_E0_E2_RTX5070_Full` (kèm bank nháp cũ) đã biến mất khỏi máy — bank chuẩn giờ nằm tại đây và tại `uit/.memory-bank` (Task2).

## 2026-09-10 — E3-B patch install + preflight pass, standard 20 fits đang chạy

- Vị trí Task1 (từ bank): `/home/hvusynh2/nguyenduong/UIT DSC/Task1 - LegalIR`, root capsule: `/home/hvusynh2/nguyenduong/UIT DSC`.
- Tải file Drive `15UfvbBYGqq7FeWDuNbnYmDqVLhmIE8LK` (thực chất là ZIP patch, 159KB) → lưu `Task1 - LegalIR/E3B_PATCH_2026-09-10.zip`; verify SHA256 15/15 file trong `E3B_PATCH_MANIFEST.json` = ALL_OK rồi unzip tại root capsule.
- Sự cố preflight `E1 manifest Stage-1 provenance drift`: file stage1 trên disk (12.4MB, 2026-08-26) khác bản seal; đối chiếu 8/8 inputs với `LegalIR_E0_E2_Server_Auto_FIXED.zip` → chỉ 1 file drift, 7 OK. Restore đúng 1 file từ capsule FIXED (11.7MB, sha `4b16c37...` khớp meta+audit E3B); bản drift backup tại `/tmp/e3b_backup/...DRIFTED`. Không sửa contract.
- CPU checks: `test_e2_mil_finetune_oof` + `test_e3b_residual_score_gate` + `test_e3b_nested_mil_scores` = 15/15 OK. Preflight pass: `holdout_opened:false`, `public_labels_used:false`.
- Standard (20 fits = 5 outer × 2 inner seed × 2 neural seed) đang chạy background PID 2053646, log `/tmp/e3b_standard.log`, chạy trên GPU1 (`CUDA_VISIBLE_DEVICES=1`, GPU0 bận task khác). Gate: chỉ chạy stress 60 fits khi `promotion_to_stress_allowed:true`; E4 vẫn khóa, không mở holdout.

## 2026-09-13 — E3-B standard+stress pass, E4 ready (actualize từ output)

- Standard xong 2026-09-11T09:59:53+07:00: 10/10 folds, gate `standard.pass:true` (min gain 0.01218, rescue-destructive ≥15, per-seed [true,true]); mean gain +0.01219 (inner 20260910) / +0.01491 (inner 20260911); `promotion_to_stress_allowed:true`, `promotion_to_e4_allowed:false`, `holdout_opened:false`. Supervisor 6 attempts (exit 1/137 OOM/Killed).
- Stress xong 2026-09-13T11:42:59+07:00: 30/30 folds, cả 3 regime pass — law_source minima 0.01086 (mean +0.01114/+0.01086), near_duplicate minima 0.01294 (mean +0.01295/+0.01344), query_template minima 0.00988 (mean +0.00988/+0.01428); per-seed đều true. `promotion_to_e4_allowed:true`, `holdout_opened:false`, `public_labels_used:false`.
- Lineage: stage1 `4b16c37...`, anchor `dd2c754...`, complement-only, memory 997/1000 sau overlap-filter, label-blind manifest trước mở labels. Reports: `Task1 - LegalIR/output/e3b_nested_mil/e3b_standard_report.json` + `e3b_stress_report.json`.
- Quyết định: E3-B pass → E4 READY chưa chạy. Không mở holdout trong lần actualize này. Tiếp theo: verify E4 readiness hoặc chạy E4 parity một phát theo protocol khi có lệnh explicit.

## 2026-09-13 — E4 blocked rồi draft contract (chưa mở holdout)

- Lệnh chạy E4: phát hiện gap — không có E4 runner; script E1/E2/E3B cấm holdout by design (`FORBIDDEN_TERMS`, `_check_complement_only`); protocol E4 chỉ 6 dòng. Dừng theo executing-plans, hỏi hướng đi → chọn Draft E4 contract.
- Evidence khóa policy từ 40 folds complement: plurality + best gain hội tụ `(tau=0.5, mem_w=0.0)` (n=12, mean +0.01766, net +52). Gap thêm: chưa có holdout E1 evidence/frozen-CE (E1 complement-only) → contract gồm H0 (label-blind) + H1 (full-complement refit) + H2 (single read).
- Đã viết `Task1 - LegalIR/E4_PARITY_CONTRACT.md` (policy khóa, procedure, promote gate, forbidden, open items). Holdout mới chỉ `ls` tên file, chưa đọc label. Chờ duyệt contract trước khi viết contract tests.

## 2026-09-13 — E4 patch Phong về, fix winpath, preflight pass, prepare đang chạy

- Nhận `LegalIR_E4_Parity_Patch.zip` qua Drive (link2; link1 là MP4 không liên quan). SHA-256 khớp `d3654416...` Tuyệt đối. Unzip tại capsule root; verify manifest 14/14 file OK. Contrat draft của mình bị patch overwrite (backup tại `/tmp/E4_PARITY_CONTRACT_mydraft.md`).
- Unittest 4 file: 23 tests OK. Preflight lần 1 fail `FileNotFoundError` — root cause: `holdout_safe_tail_locked_stage1.audit.json` (build 26-08) chứa `source_stage1_audit` là path Windows tuyệt đối `D:\projects\...` (chạy được trên máy Phong, hỏng trên Linux vì `Path` coi là relative rồi join). Fix surgical: byte-replace đúng 1 substring thành `../v11_holdout_stage1_chunk_only_top5.audit.json` (cùng convention với complement); backup gốc `...ORIG_WINPATH_BAK`; source resolve ra 1000 queries. An toàn hash: E4 chỉ ghi `sha256_file(holdout_stage1)` vào lineage mới, không so với hash cũ nào.
- Preflight lần 2 pass: `label_blind_preflight`, `holdout_labels_opened:false`, lineage complement khớp E3B (`dd2c754...`, `4b16c37...`).
- `MODE=prepare` đang chạy GPU0 (PID 310618, log `/tmp/e4_prepare.log`). Gate tiếp: `ready_for_parity:true` trong `e4_prepare_report.json` → chạy `MODE=parity`; false → gửi output cho user.

## 2026-09-14 — E3C preflight fail-closed, mở đường bằng E3B-retain rerun

- Lệnh chạy E3C: unit tests E3C 5/5 OK, deps đủ (peft 0.19.1/transformers 5.13.1/accelerate 1.14.0). Preflight fail đúng protocol: thiếu `e3b_weights/standard/inner_20260910_fold_1_neural_20260908.pt` — E3B cũ RETAIN=false (retained_path=null trong report). Không substitute base checkpoint.
- Quyết định user: rerun E3B retain weights. Để giữ lock E4, chạy vào dir riêng `output/e3b_nested_mil_retain/` (E3C hỗ trợ E3B_OUTPUT_DIR override), không ghi đè `output/e3b_nested_mil/` gốc.
- CPU checks E3B 15/15 OK. Preflight retain-dir pass (`holdout_opened:false`). Standard retain 20 fits đang chạy background PID 18956, log `/tmp/e3b_retain_standard.log`. Gate tiếp: standard pass → stress 60 fits → E3C preflight/standard/stress complement-only.

## 2026-09-14 — Sửa checkpoint (TDD) + phát hiện E4 đã PASS + E5 đang chạy

- Thiếu save/load verification: viết `tests/test_checkpoint_store.py` đỏ trước, thêm `atomic_torch_save`/`verify_torch_state_dict`/`verify_torch_checkpoint`/`build_weight_manifest` vào `scripts/v27_a_common.py`, đấu nối `_retain_weights` (E3B, kèm sidecar manifest regime/fold/seeds/split/model/file), training ckpt (E3C, atomic+verify), `_save_state_dict` (E4, atomic+verify). Runner E3B/E3C thêm env `NEURAL_SEEDS`/`INNER_SEEDS` (default giữ nguyên). Tổng 24/24 tests xanh, `py_compile` + `bash -n` OK.
- Job retain-full dừng mềm (0 fold xong, GPU0 trống). Relaunch lọc 1 neural seed bị từ chối: `_preflight` khóa cứng đúng 2 neural seeds — lọc cần sửa gate, chờ chốt full-80 vs factory-40.
- Phát hiện E4 đã chạy 1 lần 2026-09-13 (ledger access=1, immutability guard từ chối prepare lần 2 đúng thiết kế): recall `0.928` vs V16b `0.9228333` (gain `+0.0051667`), rescues `12>7` destructive → `promotion_to_e5_allowed:true`, `e5_executed:false`. Hai deployment `.pt` (2.2GB/bản) verify-load OK process mới (393 keys).
- E5 lần 1 fail-fast đúng file stage2 (`anchor escapes c35`); file chuẩn là `v10_9_prf/public_safe_tail_stage*.audit.json` (khớp sha evidence). Relaunch materialize GPU0 PID 43653, log `/tmp/e5_materialize.log`. GPU1 trống (job Task2 đã xong).

## 2026-09-14 — Phân rã oracle R0/R1/R3/R35 từ cache E3B (CPU-only, khớp phân tích user)

- TDD `tests/test_oracle_decomposition.py` (8 tests) + refactor gate (`ranked_challengers` tách khỏi `propose_slot5`, behavior giữ nguyên, 18 tests cũ xanh). Script `scripts/e3b_oracle_decomposition.py`: R0=gate thực, R1=oracle keep/swap ứng viên đã đề cử, R3=oracle trong top-3 ranked, R35=oracle slot-5 toàn c35; bucket destructive/gate_reject/rank_2_3/no_nomination/deep(signal_miss/deep_ranked) + cờ gold_absent_c35/filter_blocked. Report: `output/e3b_oracle_decomposition/r0_r1_r3_r35_report.json`.
- Khớp hoàn toàn số user: anchor 0.886062, R35 0.949206 mọi seed/regime; seed 20260910 bỏ lỡ 62 (10 gate_reject + 11 rank_2_3 + 16 no_nomination + 25 deep) + 12 destructive; 20260911 bỏ lỡ 63 + 6 destructive; absent_c35=47 mọi split; filter_blocked=0.
- Đóng góp (standard): gate (R1−R0) +0.0150/+0.0165; top-3 (R3−R1) +0.0071/+0.0088; deep (R35−R3) +0.0244/+0.0274 (deep_ranked 15/11, signal_miss 10/11). Thứ tự đề xuất: hiệu chỉnh gate trên C (CPU-only) → selector top-3 nhỏ → comparator full chỉ cho phần deep; nhánh coverage c51/c67 cần contract mới. Full E3C vẫn on-hold theo quyết định user.

## 2026-09-14 — E5 materialize XONG + seal submission ZIP

- E5 chạy ~1h40 (2 seeds × 35.000 rows), report `e5_public_report.json`: 1000 queries, 455 đổi slot-5 vs V16b, labels đóng, budget 2.61B <4B, `promotion_to_e5_allowed:true` từ E4.
- Verify predictions: 1000 QID × 5 ID unique. Đóng `output/e5_public.zip` qua `package_submission.py` với `public-official.json` (QID-match true), SHA-256 `3c250554832e433c2bdfb744f8ceb8cd8f6509c8a3d2db1ad346744b68af027b`. V16b `v16b_memory_slot5.zip` giữ fallback.
- GPU0 hiện do job team khác giữ (`finetune_edge.py`, không phải của mình); GPU1 share nhiều job nhỏ. Hàng chờ nghiên cứu: hiệu chỉnh gate trên C (CPU-only) → selector top-3 → comparator/coverage (chờ chốt).

## 2026-09-14 — Hiệu chỉnh gate trên C: KHÔNG transfer sang V + E3B-retain chạy lại

- TDD `tests/test_gate_recalibration.py` (6 tests) + mở `ranked_challengers` thêm `disagreement_weight`/`require_consensus` (default giữ nguyên, tests cũ xanh). Script `scripts/e3b_gate_recalibrate.py`: grid 252 config (threshold 0–1 step 0.05 × mem {0,0.1} × consensus × dis_w {0.25,0.5,1.0}), chọn đúng luật E3-B trên C, đánh giá trên V; self-check repro policy gốc khớp report tuyệt đối. Report: `output/e3b_oracle_decomposition/gate_recalibration_report.json`.
- Kết quả V−E3B: standard +0.0037/−0.0037; law −0.0033/+0.0000; template +0.0042/−0.0019; neardup +0.0000/+0.0021. Config thắng scatter toàn grid theo fold (overfit C ~200 query/252 config); consensus không bao giờ được chọn. Kết luận: headroom R1−R0 không lấy được bằng tuning threshold — cần family khác (luật top-3 heuristic hoặc comparator).
- E3B full-retain relaunch GPU1 (PID 299702, `VRAM_CAP_GIB=50`, log `/tmp/e3b_retain_full.log`, code checkpoint mới atomic+verify+manifest): autoprobe xong (micro 16), đang fit. Chạy full-80 đúng protocol 2-seed (lọc 1 seed bị `_preflight` từ chối).

## 2026-09-16 — E3B-retain OOM GPU0 → dừng theo lệnh + R1 patch v2 auto-chain

- E3B-retain (`output/e3b_nested_mil_retain/`, RETAIN=true) chạy 16/9 05:12–13:47: xong 5 folds `inner=20260910` (10 weights 2.2G), đang `inner=20260911/fold=1/neural=20260909/epoch=2` thì `torch.OutOfMemoryError` GPU0 (free 90MiB/79G; job `benchmark_table_3.py` 30G + own 30G). Checkpoint resumable (`complete:false`). Đã dừng process + tmux theo lệnh user để nhường GPU cho R1.
- R1 patch v2 Drive `1Zl11cvkJ8jhh90vJpuv86COQF_u7uFrd` → SHA `b388861a…3f8340` khớp, 20 files tại capsule root. `requirements-gpuhub.txt` đã cài. Qwen local `models/Qwen3-Reranker-0.6B` khớp fingerprint cũ; revision `local-safetensors-27cd75a4…`.
- Fix bug patch: `r1_qwen_c35_cache.py:165` dùng `r1_contract.X` nhưng chỉ import names → import trực tiếp `FROZEN_UPSTREAM_PARAMETERS, PARAMETER_LIMIT`, tests R1 3/3 pass.
- Preflight pass (1197 Q, 41.895 pairs, est 3.21B <4B, holdout false); qwen-preflight pass; pilot 64/64 QID ~1000s, ledger thực 3.205B <4B, batch=1 (reserve 12G), `under_4b:true`.
- Auto-chain `/tmp/r1_auto_chain.sh` trong tmux `r1_auto` (GPU1 `CUDA_VISIBLE_DEVICES=1`): qwen-cache resume QID → eval-preflight → standard → stress chỉ khi `promotion_to_stress_allowed:true`. Logs: `/tmp/r1_auto.log`, `/tmp/r1_cache.log`, `/tmp/r1_standard.log`, `/tmp/r1_stress.log`.

## 2026-09-16 — R1 tối ưu ETA (runtime-only, không đổi logic)

- Núm vặn an toàn đã xác minh: batch size bất biến score (pad trái + attention mask), lineage_sha không chứa batch/reserve/threads/shard (`2388eb57…` giữ nguyên sau restart), resume nguyên tử theo QID.
- Đổi `VRAM_RESERVE_GIB` 12→5 + `CPU_THREADS` 1→4 + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` trong `/tmp/r1_auto_chain.sh`; restart tmux `r1_auto` (single owner, resume QID, mất tối đa 1 QID đang bay).
- Probe mới trên GPU1 chọn **batch 8** (batch16 rớt reserve; ledger vẫn 3.205B <4B, `under_4b:true`).
- Pace đo 69→115 QID: **13.5s/QID**, còn 1082 QID → **ETA cache ~4.1h**, sau đó eval-preflight → standard → stress (có điều kiện), toàn bộ trong tmux `r1_auto`.
- KHÔNG chia 2-GPU: shard mới sẽ mồ côi 115 QID đã xong (eval từ chối count lẫn lộn) + GPU0 đang 100% util và từng OOM-kill E3B 2 lần — rủi ro vượt lợi ích (~1h).
