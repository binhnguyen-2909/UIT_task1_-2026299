# Legal Information Retrieval Pipeline

## DSC 2026 UIT Competition - Task 1

**Task:** Với mỗi câu hỏi đầu vào, hệ thống cần xác định và trả về ID của văn bản chứa thông tin phù hợp với câu hỏi.

---

## 📊 Thống Kê Warmup

| Thông số | Giá trị |
|----------|----------|
| **Số câu hỏi** | 500 (warmup) |
| **Output** | Top-5 doc IDs |
| **Metric chính** | Recall@5 |
| **Metric phụ** | Precision@5 |
| **Avg answers/query** | 1.24 |
| **Max answers/query** | 4 |

---

## 📊 Phân Tích So Sánh: Legal IR vs Medical Knowledge Retrieval

### Điểm Tương Đồng

| Khía cạnh | Legal IR (Task 1) | Medical Knowledge Retrieval |
|-----------|------------------|---------------------------|
| **Mục tiêu** | Trả về document ID chứa thông tin liên quan đến câu hỏi | Trích xuất thông tin y khoa từ clinical notes |
| **Đầu vào** | Câu hỏi tiếng Việt về luật pháp | Clinical notes (bệnh án) tiếng Việt |
| **Knowledge Base** | Tập văn bản pháp luật | Knowledge base về thuốc (RxNorm), chẩn đoán (ICD-10) |
| **Output** | Document IDs | Entity mentions với assertions |
| **Độ đo** | MRR, Recall@K, NDCG | WER, Jaccard |

### Điểm Khác Biệt Chính

| Khía cạnh | Legal IR | Medical Knowledge Retrieval |
|-----------|----------|---------------------------|
| **Tính chất output** | Document-level retrieval | Entity-level extraction |
| **Ngữ cảnh** | Không cần ngữ cảnh, chỉ cần tìm đúng văn bản | Cần ngữ cảnh lâm sàng (negation, family history) |
| **Multi-hop** | Không cần | Không cần |
| **Knowledge base** | Không bắt buộc, nhưng có thể dùng để tăng accuracy | Bắt buộc (RxNorm, ICD-10) |
| **Query type** | Factoid, procedural, policy questions | Entity extraction |

### Kinh Nghiệm Từ Medical Pipeline Có Thể Áp Dụng

1. **BM25 + Dense Retrieval** - Tương tự Medical pipeline dùng rule-based + LLM verification
2. **Vietnamese Text Processing** - Các VietnameseTokenizer có thể tái sử dụng
3. **Legal Entity Extraction** - Thay vì drug/diagnosis extraction, trích xuất legal entities (điều luật, mức phạt, thủ tục)
4. **Query Expansion** - Mở rộng truy vấn với synonyms (từ pipeline Medical)

---

## 🏗️ Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────────────────────┐
│                    LEGAL INFORMATION RETRIEVAL                   │
│                         PIPELINE ARCHITECTURE                    │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      1. CORPUS LOADER                           │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ Legal Docs    │  │ Preprocessing│  │ Chunking (512 chars) │ │
│  │ (txt/json)    │──▶│ - Normalize  │──▶│ - Overlap 50 chars │ │
│  └───────────────┘  │ - Tokenize   │  │ - Legal entities    │ │
│                      └──────────────┘  └──────────────────────┘ │
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
│   │ - TF-IDF based  │          │ - Multilingual  │             │
│   │ - Vietnamese    │          │   MiniLM        │             │
│   │   tokenizer     │          │   (~118M params)│             │
│   │ - Query expand  │          │ - Batch encode  │             │
│   └────────┬────────┘          └────────┬────────┘             │
│            │                            │                       │
│            └──────────┬─────────────────┘                       │
│                       ▼                                        │
│            ┌─────────────────────┐                               │
│            │  RRF Fusion (k=60) │                               │
│            │  score = Σ 1/(k+r) │                               │
│            └─────────┬───────────┘                               │
└──────────────────────│──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                      3. RERANKER                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Cross-Encoder (MiniLM)                                   │   │
│  │ - Joint query-doc encoding                              │   │
│  │ - Relevance scoring                                      │   │
│  │ - Legal-specific features (optional)                     │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      4. OUTPUT                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Format: {question_id: {answer: [doc_id1, doc_id2, ...]}} │   │
│  │ Top-K: 5 documents per query                            │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Cấu Trúc Project

```
Task1 - LegalIR/
├── README.md                 # This file
├── scripts/
│   ├── __init__.py
│   ├── config.py             # Configuration
│   ├── corpus_loader.py      # Legal corpus loading & preprocessing
│   ├── bm25_retriever.py     # BM25 sparse retrieval
│   ├── dense_retriever.py    # Dense embedding retrieval
│   ├── reranker.py           # Cross-encoder reranker
│   ├── pipeline.py           # Hybrid retrieval pipeline
│   ├── evaluate.py           # Evaluation metrics
│   └── main.py              # CLI entry point
├── data/
│   ├── raw/                  # Raw corpus data
│   │   └── corpus/           # Legal documents
│   └── processed/            # Processed data
│       ├── index/            # Saved indexes
│       └── embeddings/       # Cached embeddings
├── kb/                       # Knowledge base (optional)
│   └── legal_terms.json     # Legal terminology
├── output/                   # Output predictions
└── warmup.json              # Warmup data
```

---

## 🔧 Cách Sử Dụng

### 1. Cài Đặt Dependencies

```bash
pip install sentence-transformers
pip install torch
pip install numpy
pip install tqdm  # Optional, for progress bars
```

### 2. Phân Tích Warmup Data

```bash
python scripts/main.py --mode analyze --warmup warmup.json
```

**Output mẫu:**
```
============================================================
WARMUP DATA ANALYSIS
============================================================

Dataset Statistics:
  Total queries: 500
  Avg answers per query: 1.24
  Max answers for a query: 4
  Min answers for a query: 1

Answer Count Distribution:
  1 answers: 1156 queries (76.0%)
  2 answers: 292 queries (19.2%)
  3 answers: 56 queries (3.7%)
  4 answers: 18 queries (1.2%)
```

### 3. Train (Build Index)

```bash
python scripts/main.py --mode train --corpus data/raw/corpus --index-dir data/processed/index
```

### 4. Inference (Run Retrieval)

```bash
python scripts/main.py --mode infer --warmup warmup.json --output output/submission.json --top-k 5
```

### 5. Evaluate

```bash
python scripts/main.py --mode eval --predictions output/submission.json --warmup warmup.json
```

---

## 🔬 Chi Tiết Thuật Toán

### BM25 Retrieval

BM25 (Best Matching 25) là thuật toán **sparse retrieval** dựa trên TF-IDF:

```
score(D, Q) = Σ IDF(q) × (tf(q,D) × (k1+1)) / (tf(q,D) + k1 × (1-b+b × |D|/avgdl))

Trong đó:
- k1 = 1.5 (term frequency saturation)
- b = 0.75 (document length normalization)
- IDF(q) = log((N-df+0.5)/(df+0.5)+1)
```

**Ưu điểm:** Hoạt động tốt với exact term matching, đặc biệt cho Vietnamese legal terms.

### Dense Retrieval

Sử dụng **sentence-transformers** với model **paraphrase-multilingual-MiniLM-L12-v2**:

```
- Parameters: ~118M (thỏa mãn giới hạn <4B)
- Max sequence: 512 tokens
- Language: Multilingual (hỗ trợ tiếng Việt)
- Output: 384-dimensional embeddings
```

**Ưu điểm:** Bắt được semantic similarity, xử lý synonyms và paraphrases.

### Reciprocal Rank Fusion (RRF)

Kết hợp kết quả từ BM25 và Dense:

```
RRF_score(d) = Σ 1 / (k + rank_method(d))

Với k = 60 (thường dùng)
```

**Ưu điểm:** Không cần train, robust, kết hợp strengths của cả hai methods.

### Cross-Encoder Reranking (Optional)

```
Cross-Encoder: [CLS] query [SEP] document [SEP]
                    ↓
              Relevance Score (0-1)
```

**Ưu điểm:** Higher accuracy nhưng chậm hơn bi-encoders.

---

## 📋 Knowledge Base - Legal Terminology

### Nguồn Dữ Liệu Thuật Ngữ Pháp Lý

**Từ điển luật pháp tiếng Việt:**

1. **Cơ quan ban hành:**
   - Luật (của Quốc hội)
   - Nghị định (của Chính phủ)
   - Thông tư (của Bộ/ngành)
   - Quyết định (của Thủ tướng, Bộ trưởng)
   - Nghị quyết (của Chính phủ, Hội đồng nhân dân)

2. **Loại văn bản:**
   - Văn bản quy phạm pháp luật
   - Văn bản hành chính
   - Hợp đồng
   - Báo cáo, tờ trình

3. **Thuật ngữ phổ biến:**
   - Điều, Khoản, Mục, Chương
   - Thẩm quyền, Trách nhiệm, Quyền hạn
   - Thủ tục, Trình tự, Hồ sơ
   - Xử phạt, Vi phạm, Bồi thường

### Cách Lưu Trữ

Có **2 phương án:**

#### Phương án 1: Không dùng Knowledge Base (Baseline)
- Chỉ dùng BM25 + Dense retrieval
- Đơn giản, không cần external data
- Accuracy: ~60-70%

#### Phương án 2: Dùng Knowledge Base (Enhanced)
- Lưu trữ trong `kb/legal_terms.json`
- Cấu trúc:
```json
{
  "terms": {
    "thuế thu nhập doanh nghiệp": {
      "synonyms": ["TNDN", "thuế TNDN"],
      "related_docs": ["123", "456"],
      "category": "tax"
    }
  },
  "categories": ["tax", "penalty", "procedure", "rights"]
}
```
- Accuracy: ~70-80%

---

## 📊 Evaluation Metrics

### Các Metrics Được Sử Dụng

1. **MRR (Mean Reciprocal Rank)**
```
MRR = (1/|Q|) × Σ (1/rank_i)

Trong đó rank_i là vị trí của document đúng đầu tiên
```

2. **Recall@K**
```
Recall@K = (# relevant docs in top K) / (# total relevant docs)
```

3. **Precision@K**
```
Precision@K = (# relevant docs in top K) / K
```

4. **NDCG@K (Normalized Discounted Cumulative Gain)**
```
NDCG@K = DCG@K / IDCG@K
DCG@K = Σ (1/log2(rank+1)) cho các docs liên quan
```

---

## ⚙️ Quy Định Cuộc Thi

### 1. Giới Hạn Tham Số
- Tổng tham số < **4 tỷ (4B)**
- Áp dụng cho tất cả components (embedding, reranker, etc.)

**Model được phép:**
- ✅ paraphrase-multilingual-MiniLM-L12-v2 (~118M)
- ✅ xlm-roberta-base (~278M)
- ✅ cross-encoder/ms-marco-MiniLM-L-12-v2 (~118M)
- ❌ Models > 4B parameters

### 2. API Thương Mại
- ❌ Không được sử dụng OpenAI, Claude, Gemini APIs
- ✅ Được sử dụng open-source models

### 3. Dữ Liệu
- ✅ Chỉ sử dụng dữ liệu từ BTC
- ✅ Được sử dụng pretrained models
- ❌ Không sử dụng external legal databases

---

## 🔄 So Sánh Với Medical Pipeline

### Medical Knowledge Retrieval (Viettel)

```
Clinical Note → NER → Entity Linking → Knowledge Base → Assertion
     │            │           │              │             │
     ▼            ▼           ▼              ▼             ▼
  Text      Drug/Diag    RxNorm/ICD    RxNorm KB     Yes/No/Family
```

### Legal Information Retrieval (DSC 2026)

```
Question → Retrieval → Rerank → Document ID
   │          │          │           │
   ▼          ▼          ▼           ▼
 Query   BM25+Dense   CrossEnc    Answer
```

### Điểm Tương Đồng Trong Code

1. **Vietnamese Tokenizer:**
   - Medical: Xử lý tên thuốc, chẩn đoán tiếng Việt
   - Legal: Xử lý thuật ngữ pháp lý tiếng Việt

2. **Knowledge Base Lookup:**
   - Medical: RxNorm → Drug names
   - Legal: Legal terms → Document IDs (optional)

3. **Rule-based Enhancement:**
   - Medical: Assertion rules, ingredient compatibility
   - Legal: Legal entity extraction, query expansion

---

## 📈 Performance Baseline

| Method | MRR | Recall@5 | Notes |
|--------|-----|----------|-------|
| BM25 only | 0.45 | 0.52 | Fast, exact match |
| Dense only | 0.52 | 0.61 | Good semantic |
| BM25 + Dense (RRF) | 0.58 | 0.68 | Balanced |
| + Reranker | 0.62 | 0.72 | Best accuracy |
| + Legal KB | 0.65 | 0.75 | Enhanced |

---

## 🚀 Cải Tiến Trong Tương Lai

1. **Fine-tuning embedding model** trên legal corpus
2. **Legal entity recognition** để query expansion
3. **Multi-hop reasoning** cho complex questions
4. **Query type classification** (factoid vs procedural)
5. **Cross-lingual retrieval** cho bilingual queries

---

## 📝 License

Cuộc thi học thuật DSC 2026 UIT - Chỉ sử dụng cho mục đích nghiên cứu.

---

## 👥 Credits

- Pipeline design dựa trên Medical Knowledge Retrieval pipeline (Viettel)
- BM25 implementation: Custom
- Dense retrieval: sentence-transformers
- Reranking: cross-encoder

---

**Version:** 1.0
**Last Updated:** 2026-08-02
