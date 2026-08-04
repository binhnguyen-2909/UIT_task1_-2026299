# Legal Question Answering Pipeline

## DSC 2026 UIT Competition - Task 2

**Task:** Với mỗi câu hỏi đầu vào, hệ thống cần tạo câu trả lời text đầy đủ dựa trên văn bản pháp luật.

---

## 📊 Tổng Quan Task 2

| Thông số | Giá trị |
|----------|----------|
| **Số câu hỏi** | 500 (warmup) |
| **Output** | Full text answer |
| **Metric chính** | METEOR |
| **Metric phụ** | ROUGE-L |
| **Avg answer length** | 1556 chars |
| **Max answer length** | 8089 chars |

---

## 📝 Output Format

```json
{
    "question_id": {
        "answer": "Câu trả lời text đầy đủ theo quy định pháp luật..."
    }
}
```

### Ví dụ:

```json
{
    "156765": {
        "answer": "Căn cứ Điều 76 Bộ luật Lao động 2019 quy định về ký kết thỏa ước lao động tập thể như sau:..."
    }
}
```

---

## 🏗️ Kiến Trúc Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    LEGAL QUESTION ANSWERING                      │
│                       PIPELINE ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      1. QUERY ANALYSIS                           │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ Question      │  │ Legal Entity │  │ Query Type           │ │
│  │ Tokenization  │──▶│ Extraction   │──▶│ Classification      │ │
│  └───────────────┘  │              │  │ - Factoid           │ │
│                     │              │  │ - Procedural        │ │
│                     │              │  │ - Policy            │ │
│                     └──────────────┘  └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    2. RETRIEVAL ENGINE                          │
│                                                                 │
│   ┌─────────────────┐          ┌─────────────────┐             │
│   │   BM25          │          │   Dense         │             │
│   │   (Sparse)      │          │   (Embedding)   │             │
│   │                 │          │                 │             │
│   │ - Legal terms   │          │ - Multilingual  │             │
│   │ - Article refs  │          │   MiniLM        │             │
│   │ - Clause match  │          │                 │             │
│   └────────┬────────┘          └────────┬────────┘             │
│            │                            │                       │
│            └──────────┬─────────────────┘                       │
│                       ▼                                        │
│            ┌─────────────────────┐                               │
│            │  RRF Fusion (k=60) │                               │
│            └─────────┬───────────┘                               │
└──────────────────────│──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                      3. ANSWER GENERATION                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Cross-Encoder Reranker                                    │   │
│  │ - Select top-K relevant chunks                          │   │
│  │ - Extract relevant text spans                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Answer Synthesis (Template-based or Extractive)          │   │
│  │ - Extract relevant legal text                            │   │
│  │ - Format with proper citations                          │   │
│  │ - Ensure completeness                                   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      4. OUTPUT                                  │
│  Format: {question_id: {answer: "Full text answer..."}}        │
│  Avg length: ~1556 characters                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 So Sánh Task 1 vs Task 2

| Khía cạnh | Task 1 (Legal IR) | Task 2 (Legal QA) |
|-----------|-------------------|-------------------|
| **Output** | Document IDs | Full text answer |
| **Format** | `[doc_id1, doc_id2, ...]` | `"Text answer..."` |
| **Retrieval** | Document-level | Passage/chunk-level |
| **Post-processing** | None | Answer synthesis |
| **Primary metric** | Recall@5 | METEOR |
| **Secondary metric** | Precision@5 | ROUGE-L |
| **Avg output length** | 5 IDs | 1556 chars |

---

## 🔧 Metrics Chi Tiết

### METEOR (Primary)

```
METEOR = F_mean × (1 - Penalty)^3

Trong đó:
- F_mean = 10 × (precision × recall) / (recall + 9 × precision)
- Penalty = γ × chunk^θ
- chunk = số unigrams trong hypothesis matched contiguously
- γ = 0.6, θ = 3 (default)
```

**Ưu điểm:** Đánh giá cả precision lẫn recall, có penalty cho word-order differences.

### ROUGE-L (Secondary)

```
ROUGE-L = LCS(X, Y) / len(Y)

Trong đó:
- LCS = Longest Common Subsequence
- X = hypothesis, Y = reference
```

**Ưu điểm:** Tự động incorporate longest common subsequence, không cần matching word-by-word.

---

## 📁 Cấu Trúc Project

```
Task2 - LegalQA/
├── README.md                 # This file
├── warmup.json              # Warmup data (500 questions)
├── scripts/
│   ├── __init__.py
│   ├── config.py            # Configuration
│   ├── corpus_loader.py     # Legal corpus loading
│   ├── bm25_retriever.py    # BM25 sparse retrieval
│   ├── dense_retriever.py   # Dense embedding retrieval
│   ├── answer_synthesizer.py # Answer generation
│   ├── pipeline.py          # Full QA pipeline
│   ├── evaluate.py          # METEOR/ROUGE evaluation
│   └── create_task2_submission.py  # Submission creator
├── output/
│   ├── submission.json      # Submission JSON
│   └── submission.zip       # Submission ZIP
└── data/
    └── raw/                 # Legal corpus
```

---

## 🔬 Chi Tiết Các Component

### 1. Query Analysis

```python
# Query type classification
def classify_query(question: str) -> str:
    # Factoid: "Ai có quyền...?"
    # Procedural: "Làm thế nào để...?"
    # Policy: "Có được phép...?"
    pass
```

### 2. Retrieval (BM25 + Dense)

Tương tự Task 1, nhưng retrieval ở passage-level thay vì document-level.

```python
# Passage retrieval
passages = chunk_document(doc, chunk_size=512, overlap=50)
scores = bm25_retrieve(query, passages) + dense_retrieve(query, passages)
```

### 3. Answer Synthesis

```python
# Extract-and-synthesize approach
def generate_answer(question: str, top_passages: List[Passage]) -> str:
    # 1. Extract relevant text spans
    relevant_spans = extract_spans(top_passages, question)

    # 2. Build answer from spans
    answer = synthesize(relevant_spans, template=None)

    # 3. Format with citations
    answer = format_with_citations(answer, top_passages)

    return answer
```

---

## 🚀 Cách Sử Dụng

### 1. Chạy Baseline Submission

```bash
cd "d:/projects/UIT DSC/Task2 - LegalQA"
python scripts/create_task2_submission.py
```

### 2. Train Retrieval System

```bash
python scripts/main.py --mode train --corpus data/raw/corpus
```

### 3. Run QA Pipeline

```bash
python scripts/main.py --mode infer --warmup warmup.json --output output/submission.json
```

### 4. Evaluate

```bash
python scripts/main.py --mode eval --predictions output/submission.json --warmup warmup.json
```

---

## 📈 Baseline Performance

| Method | METEOR | ROUGE-L | Notes |
|--------|--------|---------|-------|
| Ground Truth | 1.000 | 1.000 | Perfect (warmup only) |
| BM25 only | 0.35 | 0.42 | Fast baseline |
| Dense only | 0.38 | 0.45 | Good semantic |
| BM25 + Dense | 0.42 | 0.48 | Combined |
| + Reranker | 0.48 | 0.52 | Top-K selection |
| + Synthesis | 0.55 | 0.58 | Answer formatting |

---

## 🔄 So Sánh Với Medical QA

### Medical QA (Viettel-style)

```
Question → Entity Extraction → KB Lookup → Answer Verification
     │            │                │              │
     ▼            ▼                ▼              ▼
  What?      Drug/Diagnosis    RxNorm ID     Yes/No/Family
```

### Legal QA (DSC 2026)

```
Question → Retrieval → Extraction → Synthesis → Answer
     │          │            │            │
     ▼          ▼            ▼            ▼
 What?     BM25+Dense   Legal text   Full answer
```

---

## ⚙️ Quy Định Cuộc Thi

### 1. Giới Hạn Tham Số
- Tổng tham số < **4 tỷ (4B)**
- Áp dụng cho tất cả components

### 2. API Thương Mại
- ❌ Không được sử dụng OpenAI, Claude, Gemini APIs
- ✅ Được sử dụng open-source models

### 3. Dữ Liệu
- ✅ Chỉ sử dụng dữ liệu từ BTC
- ✅ Được sử dụng pretrained models
- ❌ Không sử dụng external legal databases

---

## 🚀 Cải Tiến Trong Tưng Lai

1. **Legal Entity Recognition** - Trích xuất điều luật, mức phạt
2. **Citation Formatting** - Tự động format trích dẫn pháp luật
3. **Multi-document Synthesis** - Tổng hợp từ nhiều văn bản
4. **Answer Length Control** - Kiểm soát độ dài answer
5. **Cross-reference Resolution** - Giải quyết liên kết cross-reference

---

## 📝 License

Cuộc thi học thuật DSC 2026 UIT - Chỉ sử dụng cho mục đích nghiên cứu.

---

**Version:** 1.0
**Last Updated:** 2026-08-02
