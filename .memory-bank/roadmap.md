# Task1 LegalIR E0–E2 — Roadmap

## Current focus

R1 C35-long-fusion auto-chain đang chạy (cache GPU1 → eval-preflight → standard → stress có điều kiện). E5 sealed giữ nguyên; holdout + public vẫn khóa.

## Active plan

- `Task1 - LegalIR/E0_E5_CURRENT_PLAN.md` + bank mirror `.memory-bank/plans/2026-09-10_e3_residual_policy.md`
- R1 protocol: `Task1 - LegalIR/R1_C35_LONG_FUSION_PROTOCOL.md` + runbook `R1_GPU_SERVER_RUNBOOK.md` (patch v2 SHA `B388861A…3F8340`)
- V16b/E5 là submission đủ điều kiện duy nhất cho tới khi R1 (hoặc E4) pass gate mới.

## Next up

1. R1 standard/stress gate (tự động, chỉ stress khi `promotion_to_stress_allowed:true`). 2. Nộp E5 leaderboard lấy public score. 3. E3B-retain resume + E3C (tạm dừng, không chặn R1).

## Deferred

- V27-A MiniLM verifier (config đã seal, gate OOF +0.010 / holdout +0.008) — chỉ chạy nếu E3 thất bại.
- Mở rộng c80 — chỉ sau khi lane c35 cạn (oracle c35 0.967667 đủ cho 0.93).

## Declined

- E1 frozen CE làm submission (−0.01128). V12–V15/V17/V19/V20/CATER (rớt gate). c50 + prefix-512 (hồi quy holdout).

## Active plans

<!-- mb-active-plans -->
- [2026-09-17] [plans/2026-09-17_refactor_hsrstandardetaoptimization.md](plans/2026-09-17_refactor_hsrstandardetaoptimization.md) — refactor — hsrstandardetaoptimization
<!-- /mb-active-plans -->

<!-- mb-roadmap-auto -->
## Now (in progress)

_None._

## Next (strict order — depends)

_None._

## Parallel-safe (can run now)

_None._

## Paused / Archived

_None._

## Linked Specs (active)

_None._
<!-- /mb-roadmap-auto -->
