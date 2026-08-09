# Public-test handoff — UIT DSC 2026 LegalIR

## Trạng thái hiện tại

Task 1 đang dùng duy nhất pipeline **v10**. Kết quả public đã xác nhận:

| Recall | Precision |
|---:|---:|
| **0.8871** | **0.1900** |

File nộp đã chấm:

```text
Task1 - LegalIR/output/public_submission_v10_ce_rrf_top35.zip
```

SHA-256:

```text
b7bf4365b6d1486356987c835013435580c702594fb23ed9cab4193fde954000
```

Các pipeline và artifact v4–v9 đã được loại khỏi workspace để tránh chạy nhầm. Task 2 không thuộc phạm vi handoff này.

## Pipeline production

```text
Question
  -> SQLite FTS5/BM25 + phrase/legal signals
  -> dense passage + dense train-question retrieval
  -> RRF candidate fusion (100 documents)
  -> thanhtantran/Vietnamese_Reranker (top 35, 2 passages/document)
  -> lexical/neural RRF
  -> 5 unique document_id
```

Hai model được dùng:

- `bqbbao6/vietnamese-legal-embedding` — 330M parameters;
- `thanhtantran/Vietnamese_Reranker` — 760M parameters.

Tổng khai báo là 1.09B parameters, dưới giới hạn 4B. Model chỉ được load từ local path và phải khớp `Task1 - LegalIR/model_registry.v10.example.json`.

## Chạy và tái tạo

Toàn bộ data layout, lệnh build index, lệnh inference, cấu hình v10, resume và troubleshooting nằm tại:

```text
Task1 - LegalIR/README.md
```

Không dùng tham số khác README nếu mục tiêu là tái tạo đúng bản v10 đã nộp.

## Kiểm tra trước khi nộp

Chạy từ repository root:

```powershell
python scripts\verify_submissions.py --task 1 `
  --task1-submission "Task1 - LegalIR\output\public_submission_v10_ce_rrf_top35.zip" `
  --task1-input "Task1 - LegalIR\data\raw\public_test\public-official.json"

python -m pytest -q tests\test_contracts.py
```

Submission hợp lệ phải đáp ứng đồng thời:

1. Có đúng 1.000 question ID và khớp input.
2. Mỗi câu có 1–5 `document_id` dạng string, không trùng.
3. ZIP chỉ chứa `submission.json` ở root.
4. Không có câu nào vượt quá 5 document ID.

## File cần giữ

- `Task1 - LegalIR/scripts/main.py`: CLI train/infer/eval.
- `Task1 - LegalIR/scripts/sqlite_retriever.py`: candidate retrieval và fusion v10.
- `Task1 - LegalIR/scripts/reranker.py`: cross-encoder và final RRF.
- `Task1 - LegalIR/scripts/legal_structure.py`: chunk strategy tùy chọn để benchmark.
- `Task1 - LegalIR/scripts/evaluate.py`: local metric.
- `Task1 - LegalIR/scripts/analyze_candidate_oracle.py`: đo candidate ceiling.
- `Task1 - LegalIR/scripts/evaluate_reranker.py`: đánh giá reranker offline.
- `scripts/verify_submissions.py`: kiểm tra submission contract.

Data, index, model weights, audit và checkpoint là artifact local; không push lên GitHub. Chỉ giữ ZIP v10 nếu team cần đối chiếu đúng submission đã chấm.
