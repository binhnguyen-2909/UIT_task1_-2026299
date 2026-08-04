# Analysis Scripts - DSC 2026 UIT Official Round

## 📊 Tổng Quan

Các script phân tích dữ liệu cho vòng chính thức, giúp:
- Hiểu rõ warmup data và patterns
- Phân tích corpus khi BTC cung cấp
- Validate submissions trước khi upload
- So sánh predictions với ground truth

---

## 📁 Scripts

### 1. `official_round_analyzer.py`

Phân tích tổng hợp cho cả Task 1 và Task 2.

**Usage:**

```bash
# Phân tích warmup data
python official_round_analyzer.py --mode warmup --task 1
python official_round_analyzer.py --mode warmup --task 2

# Phân tích corpus (khi BTC cung cấp)
python official_round_analyzer.py --mode corpus --corpus-dir <path>

# Phân tích predictions
python official_round_analyzer.py --mode predictions --task 1 --submission <path>

# So sánh với ground truth
python official_round_analyzer.py --mode compare --task 1
```

**Modes:**
- `warmup`: Phân tích warmup data (question types, legal entities, vocabulary)
- `corpus`: Phân tích corpus documents (length, types, vocabulary)
- `predictions`: Validate và phân tích submission
- `compare`: So sánh predictions với warmup (precision/recall/METEOR)

**Phân tích bao gồm:**
- ✅ Dataset statistics (questions, answers, lengths)
- ✅ Question type classification (factoid, procedural, yes/no, etc.)
- ✅ Legal document types (Luật, Nghị định, Thông tư, etc.)
- ✅ Legal citations (Điều, Khoản, Mục, Chương)
- ✅ Vocabulary overlap giữa questions và answers
- ✅ Recommendations cho retrieval strategy

---

### 2. `corpus_analyzer.py`

Phân tích chuyên sâu corpus legal documents.

**Usage:**

```bash
# JSON file
python corpus_analyzer.py --corpus corpus.json

# Directory of txt/json files
python corpus_analyzer.py --corpus data/raw/corpus/

# Export summary
python corpus_analyzer.py --corpus corpus.json --export summary.json
```

**Phân tích bao gồm:**
- ✅ Document length statistics
- ✅ Legal document type detection
- ✅ Legal citation patterns
- ✅ Topic distribution (lao động, chứng khoán, etc.)
- ✅ Vocabulary analysis
- ✅ Quality checks (duplicates, missing data)
- ✅ Retrieval strategy recommendations

---

### 3. `verify_submissions.py`

Validate submission format và quality.

**Usage:**

```bash
python verify_submissions.py
```

**Validation:**
- ✅ File exists và có submission.json
- ✅ Task 1: answer là list, max 5 items, no duplicates
- ✅ Task 2: answer là non-empty string
- ✅ Statistics và error reporting

---

## 🎯 Khi Nào Dùng Script Nào?

| Giai đoạn | Script | Mục đích |
|-----------|--------|----------|
| Sau khi nhận warmup | `official_round_analyzer.py --mode warmup` | Hiểu patterns và characteristics |
| Sau khi nhận corpus | `corpus_analyzer.py` | Phân tích corpus trước khi index |
| Trước khi submit | `verify_submissions.py` | Validate format |
| Sau khi submit | `official_round_analyzer.py --mode compare` | Xem kết quả so với ground truth |

---

## 📋 Output Examples

### Warmup Analysis Output:

```
======================================================================
 PHÂN TÍCH WARMUP DATA - TASK 2
======================================================================

📊 Dataset Overview:
   Total questions: 500

📝 Question Length Statistics:
   Average: 86 chars
   Min: 21 chars
   Max: 193 chars

💬 Task 2 Answer Analysis:
   Average length: 1556 chars
   Min length: 176 chars
   Max length: 8089 chars

🔍 Question Type Analysis:
   other       :  396 ( 79.2%)
   yes_no      :   37 (  7.4%)
   list        :   35 (  7.0%)
   procedural  :   20 (  4.0%)
   factoid     :   12 (  2.4%)

📚 Legal Document Types:
   luật        :  298 ( 37.4%)
   nghị_định   :  228 ( 28.6%)
   thông_tư    :  169 ( 21.2%)
   bộ_luật     :   68 (  8.5%)
```

### Corpus Analysis Output:

```
======================================================================
 CORPUS ANALYSIS - OFFICIAL ROUND
======================================================================

📊 Basic Statistics:
   Total documents: 1000
   Average length: 12500 chars

📋 Document Type Distribution:
   nghị_định   :  250 ( 25.0%)
   thông_tư    :  300 ( 30.0%)
   luật        :  150 ( 15.0%)

💡 Retrieval Strategy Recommendations:
   - Chunk size: 512 chars
   - Overlap: 50 chars
   - RRF fusion with k=60
```

---

## 🔧 Customization

### Thêm Legal Document Types

Edit `official_round_analyzer.py` hoặc `corpus_analyzer.py`:

```python
DOC_PATTERNS = {
    'loại_mới': r'Pattern\s+regex',
    # ...
}
```

### Thêm Legal Topics

Edit `LEGAL_TOPICS` trong `corpus_analyzer.py`:

```python
LEGAL_TOPICS = {
    'topic_mới': ['keyword1', 'keyword2', ...],
    # ...
}
```

### Thêm Question Types

Edit `QUESTION_TYPES` trong `official_round_analyzer.py`:

```python
QUESTION_TYPES = {
    'type_mới': [r'pattern1', r'pattern2'],
    # ...
}
```

---

## 🚀 Quick Start

### 1. Phân tích warmup ngay:

```bash
cd "d:/projects/UIT DSC"
python scripts/official_round_analyzer.py --mode warmup --task 1
python scripts/official_round_analyzer.py --mode warmup --task 2
```

### 2. Khi BTC cung cấp corpus:

```bash
# Đặt corpus vào data/raw/corpus/
python scripts/corpus_analyzer.py --corpus data/raw/corpus/

# Hoặc JSON file
python scripts/corpus_analyzer.py --corpus data/raw/corpus.json
```

### 3. Validate trước khi submit:

```bash
python scripts/verify_submissions.py
```

---

## 📊 Metrics Reference

### Task 1 Metrics:

- **Recall@5** (Primary): % relevant docs trong top 5
- **Precision@5** (Secondary): % top 5 là relevant

### Task 2 Metrics:

- **METEOR** (Primary): F-measure với fragmentation penalty
- **ROUGE-L** (Secondary): Longest Common Subsequence F-measure

### METEOR Formula:

```
METEOR = F_mean × (1 - Penalty)^3
F_mean = (10 × P × R) / (R + 9 × P)
Penalty = 0.5 × (chunks/matches)^3
```

### ROUGE-L Formula:

```
ROUGE-L = ((1 + β²) × P × R) / (R + β² × P)
P = LCS(pred, ref) / len(pred)
R = LCS(pred, ref) / len(ref)
β = 1.2 (default)
```

---

## 📝 License

Cuộc thi học thuật DSC 2026 UIT - Chỉ sử dụng cho mục đích nghiên cứu.

---

**Version:** 1.0
**Last Updated:** 2026-08-02