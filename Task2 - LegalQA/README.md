# LegalQA — Task 2

Pipeline: official corpus loader → chunks → BM25 + sparse TF-IDF → RRF → optional local reranker → extractive answer hoặc approved local generator.

## Contract và tiêu chí chấm

- Input là câu hỏi pháp luật tiếng Việt; output là câu trả lời bằng văn xuôi.
- METEOR là metric chính; ROUGE-L là metric phụ.
- Mỗi answer phải là string không rỗng và question ID phải khớp input.
- Neural models chỉ chạy offline từ local path và phải qua model registry/BTC approval.

Dataset overview: `DSC2026_Task2_LegalQA_Data_Overview.docx`. Corpus gồm nhiều `context_*.json` với các field `id`, `name`, `link`, `passage`; các split `train.json`, `warmup.json`, `public-official.json`, `private-official.json` được cung cấp theo từng giai đoạn.

## Chạy baseline

Baseline mặc định không dùng model:

```powershell
python scripts\main.py `
  --corpus "D:\data\selected-contexts.zip" `
  --input "D:\data\public-official.json" `
  --output output\submission.zip
```

Lệnh sinh `output/submission.zip` và `output/inference_audit.json`. ZIP hợp lệ chỉ chứa duy nhất `submission.json` ở root.

Xem [PUBLIC_TEST_HANDOFF.md](../PUBLIC_TEST_HANDOFF.md) để biết cách đăng ký model, scorer parity và checklist trước submit.
