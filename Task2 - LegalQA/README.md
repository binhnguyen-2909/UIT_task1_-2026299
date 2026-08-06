# Task 2 — Legal Question Answering

Pipeline public-test: official corpus loader → passage chunks → BM25 + sparse offline TF-IDF → RRF → optional offline reranker → extractive answer or approved local generator. The zero-model path no longer imports NumPy and does not allocate a dense `chunks × vocabulary` matrix.

```powershell
python scripts\main.py `
  --corpus "D:\data\selected-contexts.zip" `
  --input "D:\data\public-official.json" `
  --output output\submission.zip
```

The default is a zero-model extractive baseline. Neural models are local-only and fail closed unless registered and BTC-approved. See [`../PUBLIC_TEST_HANDOFF.md`](../PUBLIC_TEST_HANDOFF.md) for algorithm, multi-model experiments, scorer parity and the teammate checklist.
