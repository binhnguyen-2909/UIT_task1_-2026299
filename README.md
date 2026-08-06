# UIT Data Science Challenge 2026 — LegalIR & LegalQA

Repo chuẩn bị cho public test gồm hai pipeline offline:

- `Task1 - LegalIR`: truy hồi 1–5 document ID từ corpus BTC.
- `Task2 - LegalQA`: truy hồi context rồi tạo câu trả lời extractive hoặc bằng local model đã được duyệt.

Không pipeline nào gọi API hoặc tự tải model. Baseline chạy không-model; neural model phải có weights local, license, parameter count, trạng thái đăng ký và BTC phê duyệt trong model registry.

Tài liệu bàn giao đầy đủ: [PUBLIC_TEST_HANDOFF.md](PUBLIC_TEST_HANDOFF.md).

## Kiểm tra nhanh

```powershell
python -m pip install -r requirements-core.txt
python -m unittest discover -s tests -v
python scripts\verify_submissions.py
```

## Phân tích corpus khi BTC phát dữ liệu

```powershell
python scripts\corpus_analyzer.py `
  --corpus "D:\data\selected-contexts.zip" `
  --export reports\corpus_summary.json
```

## Cấu trúc chính

```text
Task1 - LegalIR/scripts/       LegalIR loader/retrieval/evaluator/CLI
Task2 - LegalQA/scripts/       LegalQA loader/retrieval/answer/evaluator/CLI
scripts/                       shared validation, model policy, corpus audit
models/                        model-registry template; không chứa weights
tests/                         contract and end-to-end smoke tests
PUBLIC_TEST_HANDOFF.md         algorithm, flow, runbook, experiments
```
