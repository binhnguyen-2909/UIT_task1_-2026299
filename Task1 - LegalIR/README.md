# UIT DSC 2026 — Task 1 LegalIR (v10)

Pipeline production hiện tại của đội cho Task 1. Bản public v10 đạt:

| Recall | Precision | Số dự đoán |
|---:|---:|---:|
| **0.8871** | **0.1900** | 5 document_id/câu |

Submission đã chấm: `output/public_submission_v10_ce_rrf_top35.zip`

SHA-256: `b7bf4365b6d1486356987c835013435580c702594fb23ed9cab4193fde954000`

## 1. Pipeline v10

### Stage 1 — Candidate retrieval

Mỗi câu hỏi lấy tối đa 100 candidate từ các tín hiệu bổ sung cho nhau:

- SQLite FTS5/BM25 trên fixed-size chunks 1.200 ký tự, overlap 150;
- exact phrase và legal bigram;
- câu hỏi train gần nhất và document prior;
- dense passage retrieval bằng `bqbbao6/vietnamese-legal-embedding`;
- dense train-question retrieval bằng cùng embedding model;
- Reciprocal Rank Fusion (RRF) để tạo candidate ranking ban đầu.

Cấu hình đã dùng:

| Tham số | Giá trị |
|---|---:|
| Candidate pool | 100 |
| Dense passage weight | 0.05 |
| Dense RRF k | 60 |
| Train-question dense weight | 0.02 |
| Train-question dense RRF k | 60 |
| Train-question neighbors | 20 |

### Stage 2 — Cross-encoder reranking

- Model: `thanhtantran/Vietnamese_Reranker`;
- rerank 35 candidate đầu;
- tối đa 2 passage giàu tín hiệu cho mỗi document;
- fusion lexical rank và neural rank bằng RRF;
- lexical weight `1.0`, neural weight `0.8`, RRF `k=5`;
- luôn xuất đúng 5 `document_id` khác nhau.

Tổng tham số neural khai báo: khoảng **1.09B** (`330M + 760M`), dưới giới hạn 4B.

Không dùng LambdaMART v9, LLM listwise hoặc dynamic top-k trong pipeline này.

## 2. Cấu trúc repository

```text
Task1 - LegalIR/
├── README.md
├── model_registry.v10.example.json
├── Scoring-Program-Task-LegalIR/
│   ├── metadata.yaml
│   └── scoring.py
├── scripts/
│   ├── main.py                    # train / infer / eval / variants
│   ├── sqlite_retriever.py        # retrieval và fusion v10
│   ├── reranker.py                # cross-encoder + RRF
│   ├── bm25_retriever.py          # tokenize/query expansion
│   ├── legal_structure.py         # chunk strategy tùy chọn để benchmark
│   ├── retrieval_types.py
│   ├── evaluate.py
│   ├── analyze_candidate_oracle.py
│   └── evaluate_reranker.py
└── output/
    └── public_submission_v10_ce_rrf_top35.zip
```

`data/raw`, `data/processed`, model weights, audit và checkpoint là local artifacts; `.gitignore` không cho push các file lớn này.

## 3. Chuẩn bị môi trường

Khuyến nghị Python 3.10+, Windows PowerShell và NVIDIA GPU 6 GB VRAM.

Chạy từ repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-core.txt -r requirements-neural.txt
```

Hai model phải nằm đúng vị trí:

```text
models/
├── bqbbao6_vietnamese-legal-embedding/
└── thanhtantran_Vietnamese_Reranker/
```

Có thể tải bằng Hugging Face CLI từ repository root:

```powershell
hf download bqbbao6/vietnamese-legal-embedding `
  --local-dir "models\bqbbao6_vietnamese-legal-embedding"

hf download thanhtantran/Vietnamese_Reranker `
  --local-dir "models\thanhtantran_Vietnamese_Reranker"
```

`model_registry.v10.example.json` đã khai báo đúng hai model và tổng tham số. Không đổi model nếu chưa cập nhật registry và kiểm tra giới hạn 4B.

## 4. Chuẩn bị data

Đặt dữ liệu BTC như sau:

```text
Task1 - LegalIR/data/raw/public_test/
├── selected-contexts.zip
├── train.json
└── public-official.json
```

Không cần giải nén `selected-contexts.zip`. Ba file này không được commit lên GitHub.

## 5. Build index v10

Chạy từ thư mục `Task1 - LegalIR`:

```powershell
python scripts\main.py train `
  --corpus "data\raw\public_test\selected-contexts.zip" `
  --train-data "data\raw\public_test\train.json" `
  --index-dir "data\processed\v10_index" `
  --chunk-strategy fixed `
  --chunk-size 1200 `
  --chunk-overlap 150 `
  --embedding-model "..\models\bqbbao6_vietnamese-legal-embedding" `
  --candidates 100 `
  --dense-weight 0.05 `
  --dense-rrf-k 60 `
  --question-dense-weight 0.02 `
  --question-dense-rrf-k 60 `
  --question-dense-candidates 20 `
  --model-registry "model_registry.v10.example.json" `
  --device cuda `
  --dense-device cuda `
  --no-reranker
```

Index sinh ra tại `data/processed/v10_index/` gồm SQLite FTS5, dense passage embeddings và dense train-question embeddings.

Nếu SQLite index đã có và chỉ cần build lại dense caches, thêm `--reuse-index`.

## 6. Chạy public inference

```powershell
python scripts\main.py infer `
  --index-dir "data\processed\v10_index" `
  --input "data\raw\public_test\public-official.json" `
  --output "output\public_submission_v10_ce_rrf_top35.zip" `
  --reranker-model "..\models\thanhtantran_Vietnamese_Reranker" `
  --rerank-candidates 35 `
  --rerank-passages-per-doc 2 `
  --reranker-batch-size 16 `
  --reranker-rrf-weight 0.8 `
  --reranker-rrf-k 5 `
  --top-k 5 `
  --checkpoint-every 5 `
  --model-registry "model_registry.v10.example.json" `
  --device cuda
```

RTX 3050 6 GB đã chạy đủ 1.000 câu trong khoảng 112 phút. Nếu bị ngắt, chạy lại đúng command và thêm `--resume`. Không đổi option khi resume vì checkpoint có run signature.

Pipeline tạo ba local artifacts:

- `*.zip`: file duy nhất cần nộp;
- `*.audit.json`: trace lexical/neural để debug;
- `*.checkpoint.json`: trạng thái resume.

## 7. Verify trước khi nộp

Từ thư mục `Task1 - LegalIR`:

```powershell
python ..\scripts\verify_submissions.py `
  --task 1 `
  --task1-submission "output\public_submission_v10_ce_rrf_top35.zip" `
  --task1-input "data\raw\public_test\public-official.json"
```

Kết quả bắt buộc:

```text
Questions: 1000
Predicted doc IDs: 5000 (avg 5.00)
[VALID] Format, IDs, and archive contract passed
READY TO UPLOAD
```

ZIP phải chỉ có một file `submission.json` ở root. Trả nhiều hơn 5 ID cho bất kỳ câu nào sẽ bị chấm Recall và Precision bằng 0 cho câu đó.

## 8. Tests

Từ repository root:

```powershell
python -m pytest -q tests\test_contracts.py
```

Không chạy `pytest` toàn repository nếu máy chưa cài dependency riêng của Task2.

## 9. Troubleshooting

- OOM: giảm `--reranker-batch-size` từ 16 xuống 8; ranking không đổi.
- Sai model path: kiểm tra hai thư mục trong `../models` và registry.
- Resume bị từ chối: command/input/index/config khác lần chạy trước; xóa checkpoint cũ hoặc dùng output name mới.
- Output thiếu câu: không tự sửa JSON; chạy lại bằng `--resume`, sau đó verify ZIP.
