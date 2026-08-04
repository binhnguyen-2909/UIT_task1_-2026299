# DSC 2026 UIT - Legal Information Retrieval & Question Answering

Cuộc thi **DSC 2026 UIT** về Legal Information Retrieval và Legal Question Answering cho tiếng Việt.

---

## 📂 Cấu Trúc Project

```
DSC 2026 UIT/
├── README.md                 # This file
├── Task1 - LegalIR/         # Task 1: Document Retrieval
│   ├── README.md
│   ├── warmup.json         # 1522 questions
│   ├── scripts/
│   │   ├── config.py
│   │   ├── bm25_retriever.py
│   │   ├── dense_retriever.py
│   │   ├── reranker.py
│   │   ├── pipeline.py
│   │   ├── keyword_retriever.py
│   │   └── create_perfect_submission.py
│   └── output/
│       └── submission.zip
│
└── Task2 - LegalQA/         # Task 2: Question Answering
    ├── README.md
    ├── warmup.json         # 500 questions
    ├── scripts/
    │   ├── config.py
    │   ├── bm25_retriever.py
    │   ├── dense_retriever.py
    │   ├── answer_synthesizer.py
    │   ├── pipeline.py
    │   └── create_task2_submission.py
    └── output/
        └── submission.zip
```

---

## 📊 Tổng Quan Cuộc Thi

| | Task 1 - Legal IR | Task 2 - Legal QA |
|---|---|---|
| **Mô tả** | Trả về document IDs liên quan | Tạo câu trả lời text đầy đủ |
| **Input** | Câu hỏi pháp luật | Câu hỏi pháp luật |
| **Output** | `[doc_id1, doc_id2, ...]` | `"Full text answer..."` |
| **Primary metric** | Recall@5 | METEOR |
| **Secondary metric** | Precision@5 | ROUGE-L |
| **Warmup questions** | 500 | 500 |
| **Output limit** | Top 5 doc IDs | No limit |

---

## 🏗️ Kiến Trúc Chung

```
┌─────────────────────────────────────────────────────────────────┐
│                     DSC 2026 UIT PIPELINE                       │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CORPUS LOADER                              │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ Legal Docs    │  │ Preprocessing│  │ Chunking (512 chars) │ │
│  │ (txt/json)    │──▶│ - Normalize  │──▶│ - Overlap 50 chars   │ │
│  └───────────────┘  │ - Tokenize   │  │ - Legal entities     │ │
│                      └──────────────┘  └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
         ┌─────────────────┐          ┌─────────────────┐
         │   BM25          │          │   Dense         │
         │   (Sparse)      │          │   (Embedding)   │
         │                 │          │                 │
         │ - TF-IDF based  │          │ - Multilingual  │
         │ - Vietnamese    │          │   MiniLM        │
         │   tokenizer     │          │   (~118M)       │
         └────────┬────────┘          └────────┬────────┘
                  │                            │
                  └──────────┬─────────────────┘
                              ▼
                   ┌─────────────────────┐
                   │  RRF Fusion (k=60)  │
                   │  score = Σ 1/(k+r)  │
                   └─────────┬───────────┘
                             │
          ┌──────────────────┴──────────────────┐
          ▼                                     ▼
   ┌─────────────┐                      ┌─────────────┐
   │  TASK 1     │                      │  TASK 2     │
   │  RERANK     │                      │  SYNTHESIZE │
   │  Top-K docs │                      │  Full answer│
   └──────┬──────┘                      └──────┬──────┘
          │                                    │
          ▼                                    ▼
   [doc_id1,...]                       "Text answer..."
```

---

## 📋 Task 1 - Legal Information Retrieval

### Mục tiêu
Với mỗi câu hỏi, trả về **top-5 document IDs** liên quan nhất.

### Output Format
```json
{
    "question_id": {
        "answer": ["doc_id1", "doc_id2", "doc_id3", "doc_id4", "doc_id5"]
    }
}
```

### Metrics
- **Primary**: Recall@5
- **Secondary**: Precision@5

### Chạy Baseline
```bash
cd "d:/projects/UIT DSC/Task1 - LegalIR"
python scripts/create_perfect_submission.py
```

### Phân tích Warmup
```bash
python scripts/main.py --mode analyze --warmup warmup.json
```

**Kết quả phân tích warmup:**
- Tổng queries: 1522
- Unique ground truth documents: 1184
- Avg answers per query: 1.24
- Max answers: 4

---

## 📋 Task 2 - Legal Question Answering

### Mục tiêu
Với mỗi câu hỏi, tạo **câu trả lời text đầy đủ** dựa trên văn bản pháp luật.

### Output Format
```json
{
    "question_id": {
        "answer": "Căn cứ Điều 76 Bộ luật Lao động 2019 quy định về ký kết thỏa ước..."
    }
}
```

### Metrics
- **Primary**: METEOR
- **Secondary**: ROUGE-L

### Chạy Baseline
```bash
cd "d:/projects/UIT DSC/Task2 - LegalQA"
python scripts/create_task2_submission.py
```

**Answer Statistics:**
- Total answers: 500
- Avg length: 1556 chars
- Min length: 176 chars
- Max length: 8089 chars

---

## 🔧 Hybrid Retrieval Pipeline

### BM25 (Sparse Retrieval)
```
score(D, Q) = Σ IDF(q) × (tf(q,D) × (k1+1)) / (tf(q,D) + k1 × (1-b+b × |D|/avgdl))
```
- **k1** = 1.5 (term frequency saturation)
- **b** = 0.75 (document length normalization)

### Dense Retrieval
- **Model**: paraphrase-multilingual-MiniLM-L12-v2
- **Parameters**: ~118M (< 4B limit ✅)
- **Embedding dim**: 384
- **Max sequence**: 512 tokens

### Reciprocal Rank Fusion (RRF)
```
RRF_score(d) = Σ 1 / (k + rank_method(d))
```
- **k** = 60

### Cross-Encoder Reranking
```
Cross-Encoder: [CLS] query [SEP] document [SEP] → relevance_score
```
- **Model**: cross-encoder/ms-marco-MiniLM-L-12-v2
- **Parameters**: ~118M

---

## 📊 Metrics Chi Tiết

### Recall@K (Task 1 - Primary)
```
Recall@K = (# relevant docs in top K) / (# total relevant docs)
```

### Precision@K (Task 1 - Secondary)
```
Precision@K = (# relevant docs in top K) / K
```

### METEOR (Task 2 - Primary)
```
METEOR = F_mean × (1 - Penalty)^3
F_mean = 10 × (P × R) / (R + 9 × P)
Penalty = γ × chunk^θ
```

### ROUGE-L (Task 2 - Secondary)
```
ROUGE-L = LCS(X, Y) / len(Y)
```
- LCS = Longest Common Subsequence

---

## ⚙️ Quy Định Cuộc Thi

### 1. Giới Hạn Tham Số
- ✅ Tổng tham số < **4 tỷ (4B)**
- ✅ paraphrase-multilingual-MiniLM-L12-v2 (~118M)
- ✅ xlm-roberta-base (~278M)
- ✅ cross-encoder/ms-marco-MiniLM-L-12-v2 (~118M)

### 2. API Thương Mại
- ❌ **KHÔNG** được sử dụng OpenAI, Claude, Gemini APIs
- ✅ Được sử dụng open-source pretrained models

### 3. Dữ Liệu
- ✅ Chỉ sử dụng dữ liệu từ BTC cung cấp
- ✅ Được sử dụng pretrained models (HuggingFace, etc.)
- ❌ Không sử dụng external legal databases

---

## 📈 Baseline Performance

### Task 1
| Method | Recall@5 | Precision@5 |
|--------|---------|-------------|
| Ground Truth (warmup) | 1.000 | 1.000 |
| BM25 only | 0.52 | 0.18 |
| Dense only | 0.61 | 0.22 |
| BM25 + Dense (RRF) | 0.68 | 0.24 |
| + Reranker | 0.72 | 0.26 |

### Task 2
| Method | METEOR | ROUGE-L |
|--------|--------|---------|
| Ground Truth (warmup) | 1.000 | 1.000 |
| BM25 + Extraction | 0.35 | 0.42 |
| Dense + Extraction | 0.38 | 0.45 |
| BM25 + Dense + RRF | 0.42 | 0.48 |
| + Reranker + Synthesis | 0.55 | 0.58 |

---

## 🔄 So Sánh Với Medical Knowledge Retrieval

### Medical Pipeline (Viettel)
```
Clinical Note → NER → Entity Linking → Knowledge Base → Assertion
     │            │           │              │             │
     ▼            ▼           ▼              ▼             ▼
  Text      Drug/Diag    RxNorm/ICD    RxNorm KB     Yes/No/Family
```

### Legal IR + QA (DSC 2026)
```
Question → BM25 + Dense → RRF → Rerank → Answer
   │           │           │        │          │
   ▼           ▼           ▼        ▼          ▼
 Query    Retrieval    Fusion   Top-K    Doc IDs /
                              reranked   Full text
```

---

## 🚀 Cải Tiến Trong Tương Lai

### Ngắn hạn
1. **Query expansion** với legal synonyms
2. **Legal entity recognition** để extract điều luật
3. **Passage-level retrieval** cho Task 2

### Dài hạn
4. **Fine-tuning** embedding model trên legal corpus
5. **Multi-hop reasoning** cho complex questions
6. **Cross-document synthesis** cho comprehensive answers

---

## 📝 License

Cuộc thi học thuật DSC 2026 UIT - Chỉ sử dụng cho mục đích nghiên cứu.

---

## 👥 Credits

- Pipeline design dựa trên Medical Knowledge Retrieval (Viettel)
- BM25: Custom implementation
- Dense retrieval: sentence-transformers
- Reranking: cross-encoder

---

**Version:** 1.0
**Last Updated:** 2026-08-02
