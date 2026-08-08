# LegalIR — Task 1

**Current best:** `Recall@5 = 0.8253` — v4 chunk-phrase, đã submit public test.

Phạm vi bàn giao của README này là **Task 1**. Task 2 chưa được đánh giá trên public test nên không dùng để kết luận chất lượng hiện tại.

## Tiêu chí và data contract

- Recall là metric chính; Precision là metric phụ/tie-break.
- Corpus gồm nhiều `context_*.json` với `id`, `name`, `link`, `passage`.
- Input public có dạng `{question_id: {"question": "..."}}`.
- Output có dạng `{question_id: {"answer": ["document_id", ...]}}`.
- Mỗi answer phải có 1–5 document ID dạng string, không duplicate.
- ZIP submission chỉ chứa đúng một file `submission.json` ở root.
- Task 1 dùng đúng ID trong corpus BTC; không thay bằng URL, số hiệu văn bản hoặc ID tự tạo.

Corpus/input/index là artifact local và đã bị `.gitignore`; khi chạy hãy dùng đường dẫn dữ liệu thật, ví dụ `D:\data`.

## Artifact hiện tại

Submission v4 được giữ ở:

```text
Task1 - LegalIR/output/public_submission_v4_chunkphrase.zip
```

ZIP này đã được kiểm tra cấu trúc: chỉ có `submission.json` ở root, gồm 1.000 question ID và mỗi answer không rỗng.

Kiểm tra format:

```powershell
python scripts\verify_submissions.py --task 1 `
  --task1-submission "Task1 - LegalIR\output\public_submission_v4_chunkphrase.zip"
```

Khi có public input chính thức, thêm:

```powershell
  --task1-input "D:\data\public-official.json"
```

## Pipeline

```text
Question
  ├─ Exact memory từ train data
  ├─ BM25 chunk FTS5
  │    ├─ focused token query
  │    ├─ OR query để mở rộng recall
  │    └─ chunk-phrase RRF
  ├─ document phrase FTS5
  └─ RRF fusion (k=20) → unique top-5 document IDs
```

Backend production là SQLite FTS5. Chunk mặc định cho public run v4 là 1.400 ký tự, overlap 200; phrase search dùng các bigram 2–4 token và collapse về một kết quả tốt nhất cho mỗi document.

## Tái tạo v4

```powershell
cd "Task1 - LegalIR"

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

Nếu chạy trên warmup/reference có nhãn, đánh giá bằng:

```powershell
python scripts\main.py eval `
  --predictions "output\public_submission_v4_chunkphrase.zip" `
  --references "D:\data\warmup.json" `
  --all-top-k
```

## File chính

| File | Vai trò |
|---|---|
| `scripts/main.py` | CLI `train`, `infer`, `eval`, `variants` |
| `scripts/sqlite_retriever.py` | SQLite FTS5, BM25, phrase search và RRF |
| `scripts/corpus_loader.py` | Đọc corpus ZIP/directory và chunk text |
| `scripts/bm25_retriever.py` | Tokenizer tiếng Việt và BM25 baseline |
| `scripts/evaluate.py` | Metric parity và local evaluation |
| `Scoring-Program-Task-LegalIR/scoring.py` | Scorer của BTC |

Dense/neural branch chỉ là hướng thử nghiệm local; không phải cấu hình v4 đang được bàn giao.
