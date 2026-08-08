# Public-test handoff — UIT DSC 2026 LegalIR

## Phạm vi hiện tại

Handoff này tập trung vào **Task 1 — LegalIR**. Submission tốt nhất hiện tại là v4 chunk-phrase với `Recall@5 = 0.8253`:

```text
Task1 - LegalIR/output/public_submission_v4_chunkphrase.zip
```

Task 2 mới dừng ở code/scaffold và chưa được đánh giá trên public test. Không dùng artifact hoặc điểm Task 2 để kết luận chất lượng public hiện tại.

## 1. Contract của BTC

Corpus chính thức gồm nhiều `context_*.json`:

```json
{
  "id": 740,
  "name": "Tên văn bản",
  "link": "...",
  "passage": "Nội dung pháp luật..."
}
```

Public input:

```json
{
  "question_id": {
    "question": "Câu hỏi..."
  }
}
```

Task 1 submission:

```json
{
  "question_id": {
    "answer": ["740"]
  }
}
```

Invariant trước khi upload:

1. Tập question ID phải khớp chính xác input.
2. Mỗi answer có 1–5 document ID dạng string và không duplicate.
3. ZIP chỉ có đúng một file `submission.json` ở root.
4. Chỉ output ID BTC; không thay bằng URL, số hiệu luật hoặc ID tự tạo.

Task 1 dùng Recall làm metric chính và Precision làm metric phụ/tie-break.

## 2. Pipeline v4

```text
Question
  ├─ Exact memory từ train data
  ├─ BM25 chunk FTS5
  │    ├─ focused token query
  │    ├─ OR query để mở rộng recall
  │    └─ chunk-phrase RRF
  ├─ document phrase FTS5
  └─ RRF fusion → collapse theo document → unique top-5 IDs
```

Backend production là SQLite FTS5. Public v4 dùng chunk 1.400 ký tự, overlap 200, candidate pool 40 và phrase search theo bigram 2–4 token. Tất cả chạy offline; không gọi API và không tự tải model.

## 3. Tái tạo

Corpus và input BTC là local-only. Đặt chúng ngoài repository, ví dụ `D:\data`:

```powershell
cd "D:\projects\UIT DSC\Task1 - LegalIR"

python scripts\main.py train `
  --backend sqlite `
  --corpus "D:\data\selected-contexts.zip" `
  --train-data "D:\data\train.json" `
  --index-dir "data\processed\public_index" `
  --chunk-size 1400 `
  --chunk-overlap 200 `
  --candidates 40

python scripts\main.py infer `
  --index-dir "data\processed\public_index" `
  --input "D:\data\public-official.json" `
  --output "output\public_submission_v4_chunkphrase.zip" `
  --top-k 5 `
  --checkpoint-every 10
```

Nếu có reference/warmup tương ứng, chạy local metric:

```powershell
python scripts\main.py eval `
  --predictions "output\public_submission_v4_chunkphrase.zip" `
  --references "D:\data\warmup.json" `
  --all-top-k
```

## 4. Validation trước upload

Kiểm tra cấu trúc submission v4:

```powershell
cd "D:\projects\UIT DSC"
python scripts\verify_submissions.py --task 1 `
  --task1-submission "Task1 - LegalIR\output\public_submission_v4_chunkphrase.zip"
```

Khi có public input thật, thêm:

```powershell
  --task1-input "D:\data\public-official.json"
```

Kiểm tra toàn bộ regression suite:

```powershell
python -m unittest discover -s tests -v
```

Smoke test có cả pipeline Task 2 để bảo vệ contract chung, nhưng không phải public-test score của Task 2.

## 5. Data/model policy

- `selected-contexts.zip`, `train.json`, `public-official.json`, `private-official.json` và SQLite index không commit.
- Model weights và registry local không commit; chỉ giữ `models/approved_models.example.json` làm template.
- Nếu bật neural branch, model phải có local weights, license, parameter count và BTC approval trong registry. Không bật cờ để bỏ qua policy.
- Audit/checkpoint/output JSON là artifact tái tạo được và bị ignore; chỉ giữ ZIP submission cần bàn giao.

## 6. File chính

- `Task1 - LegalIR/scripts/main.py`: CLI train/infer/eval.
- `Task1 - LegalIR/scripts/sqlite_retriever.py`: SQLite FTS5, BM25, phrase search và RRF.
- `Task1 - LegalIR/scripts/corpus_loader.py`: đọc corpus và chunk text.
- `Task1 - LegalIR/scripts/evaluate.py`: metric parity/local evaluation.
- `Task1 - LegalIR/Scoring-Program-Task-LegalIR/scoring.py`: scorer BTC.
- `scripts/submission_utils.py`: strict submission contract.
- `scripts/verify_submissions.py`: validator trước upload.
- `tests/`: contract và end-to-end smoke tests.

Khi thay đổi thuật toán hoặc tham số v4, cập nhật README và file handoff trong cùng commit để teammate có thể tái tạo đúng submission.
