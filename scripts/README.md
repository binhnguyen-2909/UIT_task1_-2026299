# Shared public-test utilities

- `corpus_analyzer.py`: reads official ZIP/directory/JSON/TXT contexts and reports schema, length, duplicate, citation and topic statistics.
- `official_round_analyzer.py`: warmup, prediction and scorer-parity audits for either task.
- `submission_utils.py`: strict JSON/ZIP contract helpers.
- `verify_submissions.py`: pre-upload validation for both outputs.
- `model_policy.py`: fail-closed BTC approval, license and `<4B` parameter checks.

Examples:

```powershell
python scripts\official_round_analyzer.py --mode warmup --task 1
python scripts\official_round_analyzer.py --mode corpus --corpus "D:\data\selected-contexts.zip"
python scripts\official_round_analyzer.py --mode predictions --task 2 --input "D:\data\public-official.json"
python scripts\verify_submissions.py --task1-input "D:\data\public-task1.json" --task2-input "D:\data\public-task2.json"
```

See [`../PUBLIC_TEST_HANDOFF.md`](../PUBLIC_TEST_HANDOFF.md) for the full runbook.
