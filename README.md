# UIT Data Science Challenge 2026 — LegalIR & LegalQA

Repository này chứa hai pipeline offline cho public test:

- `Task1 - LegalIR`: truy hồi 1–5 document ID từ corpus pháp luật.
- `Task2 - LegalQA`: truy hồi context rồi tạo câu trả lời extractive hoặc bằng local model đã được duyệt.

## Tiêu chí chấm cần nhớ

- Task 1: Recall là metric chính; Precision là metric phụ/tie-break. Mỗi answer phải có 1–5 document ID dạng string, không trùng.
- Task 2: METEOR là metric chính; ROUGE-L là metric phụ. Mỗi answer phải là một string không rỗng.
- Cả hai task: question ID của submission phải khớp input; ZIP chỉ chứa đúng `submission.json` ở root.
- Neural model chỉ được load từ local path, phải có registry/license/parameter count và BTC approval. Runtime không tự tải model hay gọi API.

Chi tiết thuật toán và runbook nằm trong [PUBLIC_TEST_HANDOFF.md](PUBLIC_TEST_HANDOFF.md).

## Kiểm tra nhanh

```powershell
python -m pip install -r requirements-core.txt
python -m unittest discover -s tests -v
python scripts\verify_submissions.py --task 1
```

Lệnh trên kiểm tra submission Task 1 v4 hiện có. Khi có public input thật, truyền thêm `--task1-input` để validator kiểm tra đúng question ID. Task 2 hiện chỉ là scaffold/code nền và chưa có kết luận từ public test.

## Cấu trúc chính

```text
Task1 - LegalIR/       loader, SQLite FTS5 retrieval, evaluator và submission mới nhất
Task2 - LegalQA/       loader, retrieval/answer pipeline và scorer (chưa public-test)
scripts/               validator, corpus audit, model policy và submission contract
tests/                 contract tests và end-to-end smoke tests
models/                chỉ giữ registry template; weights và registry local không commit
```

## Dataset và artifact local

Corpus `selected-contexts.zip`, các file `train.json`/`public-official.json`/`private-official.json`, SQLite index, model weights và audit sinh trong lúc chạy đều bị `.gitignore` để repo GitHub nhẹ và không phát tán dữ liệu BTC. Đặt chúng ở một thư mục local, ví dụ `D:\data`, rồi truyền path vào CLI theo README của từng task.

File overview của Task 2 được giữ lại tại `Task2 - LegalQA/DSC2026_Task2_LegalQA_Data_Overview.docx`; file này mô tả schema `context_*.json` gồm `id`, `name`, `link`, `passage` và cách chấm METEOR/ROUGE-L.

Submission mới nhất của Task 1 nằm ở `Task1 - LegalIR/output/public_submission_v4_chunkphrase.zip`. Không dùng `Task2/output/submission.zip` để báo cáo public-test score.
