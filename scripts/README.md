# Shared public-test utilities

- `corpus_analyzer.py`: đọc ZIP/directory/JSON/TXT và audit schema, độ dài, duplicate, citation, topic.
- `official_round_analyzer.py`: audit warmup, prediction và scorer parity cho từng task.
- `submission_utils.py`: kiểm tra JSON/ZIP contract của BTC.
- `verify_submissions.py`: validator trước khi upload.
- `model_policy.py`: fail-closed approval, license và giới hạn `<4B` parameters.

Ví dụ:

```powershell
python scripts\official_round_analyzer.py --mode corpus --corpus "D:\data\selected-contexts.zip"
python scripts\verify_submissions.py --task 1 `
  --task1-submission "Task1 - LegalIR\output\public_submission_v10_ce_rrf_top35.zip" `
  --task1-input "D:\data\public-official.json"
python scripts\verify_submissions.py --task 2 `
  --task2-submission "Task2 - LegalQA\output\submission.zip" `
  --task2-input "D:\data\public-official.json"
```

Corpus và public input BTC là local-only; không commit chúng vào repository. Xem [PUBLIC_TEST_HANDOFF.md](../PUBLIC_TEST_HANDOFF.md) để biết flow và checklist đầy đủ.
