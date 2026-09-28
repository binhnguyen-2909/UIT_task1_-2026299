# Task1 LegalIR E0–E2 — Backlog

## Ideas (HIGH / MEDIUM / LOW)

- [HIGH] E3 residual slot-5 policy (mở ngay).
- [MED] V27-A MiniLM verifier nếu E3 thất bại.
- [LOW] c80, distill V25 teacher.

## Architectural Decisions (ADR)

- ADR-001 (2026-08-29): V16b là main khóa; V11h fallback.
- ADR-002 (2026-09-08): E1 chỉ diagnostic, không promote/submission.
- ADR-003 (2026-09-09): E2 giữ `oof_only_not_promoted`; gate E3/E4/E5 không đổi.
- ADR-004 (2026-09-01): Seal config V27-A trước train.
- ADR-005 (2026-08-31): Phạm vi giữ lại = V16b release + E0–E2; experiment cũ archive.

## Rejected Ideas

- E1 submission; listwise-direct/score-ensemble; CATER (bootstrap âm); CE LoRA r=4 promote (leakage group-isolation); nhánh r8 skip-bqb.
