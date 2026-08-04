"""
Official Round Data Analyzer
=============================
Script phân tích dữ liệu cho vòng chính thức DSC 2026 UIT.

Phân tích:
1. Warmup data → hiểu patterns và characteristics
2. Corpus → indexing, statistics, document types
3. Predictions → quality check, error analysis
4. Recommendations → chiến lược cho vòng chính thức

Usage:
    python official_round_analyzer.py --mode warmup
    python official_round_analyzer.py --mode corpus --corpus-dir <path>
    python official_round_analyzer.py --mode predictions --submission <path>
"""

import json
import sys
import re
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
import math

sys.stdout.reconfigure(encoding='utf-8')


class DataAnalyzer:
    """Phân tích dữ liệu cho cả Task 1 và Task 2."""

    # Vietnamese legal document patterns
    DOC_PATTERNS = {
        'luật': r'Luật\s+[\w\s]+(?=\s|$|\.|,)',
        'bộ_luật': r'Bộ luật\s+[\w\s]+(?=\s|$|\.|,)',
        'nghị_định': r'Nghị định\s+\d+/\d+/NĐ-CP',
        'thông_tư': r'Thông tư\s+\d+/\d+/TT-[A-Z]+',
        'quyết_định': r'Quyết định\s+\d+/\d+/QĐ-[A-Z]+',
        'nghị_quyết': r'Nghị quyết\s+\d+/\d+/NQ-[A-Z]+',
        'pháp_lệnh': r'Pháp lệnh\s+[\w\s]+',
    }

    LEGAL_CITATIONS = {
        'điều': r'Điều\s+\d+',
        'khoản': r'Khoản\s+\d+',
        'mục': r'Mục\s+\[?\d+\]?',
        'chương': r'Chương\s+[IVX]+',
        'phụ_lục': r'Phụ lục\s+[\w\d]+',
    }

    QUESTION_TYPES = {
        'factoid': [r'^Ai\b', r'^Cái gì\b', r'^Ở đâu\b', r'^Khi nào\b'],
        'procedural': [r'Làm thế nào', r'Cách nào', r'Thủ tục', r'Trình tự'],
        'yes_no': [r'Có được', r'Có phải', r'Đúng không', r'Có hay không'],
        'definition': [r'^[A-Z][a-z]+ là gì', r'Định nghĩa', r'Khái niệm'],
        'list': [r'Bao gồm', r'Gồm những', r'Liệt kê', r'Các loại'],
        'comparison': [r'So sánh', r'Khác nhau', r'Giống nhau'],
    }

    def __init__(self):
        self.results = {}

    def analyze_warmup(self, task: int = 1, warmup_path: str = None):
        """Phân tích warmup data cho task 1 hoặc task 2."""
        print("\n" + "="*70)
        print(f" PHÂN TÍCH WARMUP DATA - TASK {task}")
        print("="*70)

        # Load warmup
        with open(warmup_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"\n📊 Dataset Overview:")
        print(f"   Total questions: {len(data)}")

        # Question statistics
        questions = [item['question'] for item in data.values()]
        q_lengths = [len(q) for q in questions]

        print(f"\n📝 Question Length Statistics:")
        print(f"   Average: {sum(q_lengths)/len(q_lengths):.0f} chars")
        print(f"   Min: {min(q_lengths)} chars")
        print(f"   Max: {max(q_lengths)} chars")
        print(f"   Median: {sorted(q_lengths)[len(q_lengths)//2]} chars")

        # Answer statistics (different for task 1 vs 2)
        if task == 1:
            self._analyze_task1_answers(data)
        else:
            self._analyze_task2_answers(data)

        # Question type analysis
        self._analyze_question_types(questions)

        # Legal entity analysis
        self._analyze_legal_entities(data)

        # Vocabulary overlap
        self._analyze_vocabulary(data)

        print("\n" + "="*70)
        return self.results

    def _analyze_task1_answers(self, data):
        """Phân tích answers cho Task 1."""
        print(f"\n💬 Task 1 Answer Analysis:")

        all_doc_ids = []
        answer_counts = []

        for item in data.values():
            answers = item.get('answer', [])
            all_doc_ids.extend(answers)
            answer_counts.append(len(answers))

        doc_counter = Counter(all_doc_ids)

        print(f"   Total ground truth docs: {len(all_doc_ids)}")
        print(f"   Unique docs: {len(doc_counter)}")
        print(f"   Avg answers/question: {sum(answer_counts)/len(answer_counts):.2f}")
        print(f"   Max answers: {max(answer_counts)}")
        print(f"   Min answers: {min(answer_counts)}")

        # Distribution
        print(f"\n   Answer count distribution:")
        count_dist = Counter(answer_counts)
        for k in sorted(count_dist.keys()):
            pct = count_dist[k] / len(data) * 100
            bar = "█" * int(pct / 2)
            print(f"      {k} answers: {count_dist[k]:>4} ({pct:5.1f}%) {bar}")

        # Most cited docs
        print(f"\n   Top 10 most cited doc IDs:")
        for doc_id, count in doc_counter.most_common(10):
            print(f"      {doc_id}: {count} times")

    def _analyze_task2_answers(self, data):
        """Phân tích answers cho Task 2."""
        print(f"\n💬 Task 2 Answer Analysis:")

        a_lengths = [len(item['answer']) for item in data.values()]

        print(f"   Average length: {sum(a_lengths)/len(a_lengths):.0f} chars")
        print(f"   Min length: {min(a_lengths)} chars")
        print(f"   Max length: {max(a_lengths)} chars")
        print(f"   Median length: {sorted(a_lengths)[len(a_lengths)//2]} chars")

        # Distribution
        print(f"\n   Answer length distribution:")
        ranges = [(0, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, 10000)]
        for lo, hi in ranges:
            count = sum(1 for l in a_lengths if lo <= l < hi)
            pct = count / len(a_lengths) * 100
            bar = "█" * int(pct / 2)
            print(f"      {lo:>5}-{hi:<5}: {count:>4} ({pct:5.1f}%) {bar}")

        # Legal citations analysis
        all_citations = []
        for item in data.values():
            answer = item['answer']
            for pattern_name, pattern in self.LEGAL_CITATIONS.items():
                matches = re.findall(pattern, answer)
                all_citations.extend([(pattern_name, m) for m in matches])

        citation_counter = Counter([c[1] for c in all_citations])

        print(f"\n   Top 10 legal citations:")
        for citation, count in citation_counter.most_common(10):
            print(f"      {citation}: {count} times")

    def _analyze_question_types(self, questions):
        """Phân tích loại câu hỏi."""
        print(f"\n🔍 Question Type Analysis:")

        type_counts = Counter()

        for q in questions:
            q_lower = q.lower()

            matched = False
            for qtype, patterns in self.QUESTION_TYPES.items():
                for pattern in patterns:
                    if re.search(pattern, q, re.IGNORECASE):
                        type_counts[qtype] += 1
                        matched = True
                        break
                if matched:
                    break

            if not matched:
                type_counts['other'] += 1

        print(f"   Question type distribution:")
        for qtype, count in type_counts.most_common():
            pct = count / len(questions) * 100
            bar = "█" * int(pct / 2)
            print(f"      {qtype:12s}: {count:>4} ({pct:5.1f}%) {bar}")

    def _analyze_legal_entities(self, data):
        """Phân tích legal entities trong data."""
        print(f"\n📚 Legal Document Types:")

        all_text = ""
        for item in data.values():
            all_text += " " + item['question']
            if 'answer' in item:
                if isinstance(item['answer'], str):
                    all_text += " " + item['answer']

        type_counts = Counter()
        for doc_type, pattern in self.DOC_PATTERNS.items():
            matches = re.findall(pattern, all_text)
            type_counts[doc_type] = len(matches)

        for doc_type, count in type_counts.most_common():
            pct = count / sum(type_counts.values()) * 100 if sum(type_counts.values()) > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"      {doc_type:12s}: {count:>4} ({pct:5.1f}%) {bar}")

    def _analyze_vocabulary(self, data):
        """Phân tích vocabulary overlap."""
        print(f"\n📖 Vocabulary Analysis:")

        all_question_words = set()
        all_answer_words = set()

        for item in data.values():
            q_words = set(re.findall(r'\w+', item['question'].lower()))
            all_question_words.update(q_words)

            answer = item.get('answer', '')
            if isinstance(answer, str):
                a_words = set(re.findall(r'\w+', answer.lower()))
                all_answer_words.update(a_words)
            elif isinstance(answer, list):
                for doc_id in answer:
                    all_answer_words.add(doc_id.lower())

        print(f"   Unique question words: {len(all_question_words)}")
        print(f"   Unique answer tokens: {len(all_answer_words)}")
        overlap = len(all_question_words & all_answer_words)
        print(f"   Overlap: {overlap} words")

    def analyze_corpus(self, corpus_dir: str):
        """Phân tích corpus khi có dữ liệu chính thức."""
        print("\n" + "="*70)
        print(" PHÂN TÍCH CORPUS")
        print("="*70)

        corpus_path = Path(corpus_dir)

        if not corpus_path.exists():
            print(f"❌ Corpus directory not found: {corpus_dir}")
            return

        # Try JSON format
        json_files = list(corpus_path.glob("*.json"))
        if json_files:
            with open(json_files[0], 'r', encoding='utf-8') as f:
                corpus = json.load(f)
            print(f"\n📚 Loaded JSON corpus: {len(corpus)} documents")
        else:
            # Try directory of txt files
            txt_files = list(corpus_path.glob("*.txt"))
            corpus = {}
            for txt_file in txt_files:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    corpus[txt_file.stem] = f.read()
            print(f"\n📚 Loaded TXT corpus: {len(corpus)} documents")

        if not corpus:
            print("❌ No documents found in corpus!")
            return

        # Document statistics
        doc_lengths = [len(text) for text in corpus.values()]

        print(f"\n📏 Document Length Statistics:")
        print(f"   Average: {sum(doc_lengths)/len(doc_lengths):.0f} chars")
        print(f"   Min: {min(doc_lengths)} chars")
        print(f"   Max: {max(doc_lengths)} chars")
        print(f"   Median: {sorted(doc_lengths)[len(doc_lengths)//2]} chars")

        # Vocabulary
        all_words = set()
        for text in corpus.values():
            words = set(re.findall(r'\w+', text.lower()))
            all_words.update(words)

        print(f"\n📖 Corpus Vocabulary:")
        print(f"   Unique words: {len(all_words)}")

        # Document type detection
        type_counts = Counter()
        for doc_id, text in corpus.items():
            for doc_type, pattern in self.DOC_PATTERNS.items():
                if re.search(pattern, text):
                    type_counts[doc_type] += 1
                    break
            else:
                type_counts['unknown'] += 1

        print(f"\n📋 Document Types:")
        for doc_type, count in type_counts.most_common():
            pct = count / len(corpus) * 100
            bar = "█" * int(pct / 2)
            print(f"      {doc_type:12s}: {count:>4} ({pct:5.1f}%) {bar}")

        # Sample documents
        print(f"\n📄 Sample Documents:")
        sample_ids = list(corpus.keys())[:3]
        for doc_id in sample_ids:
            text = corpus[doc_id]
            print(f"\n   [{doc_id}] ({len(text)} chars):")
            print(f"   {text[:200]}...")

        print("\n" + "="*70)

    def analyze_predictions(self, submission_path: str, task: int = 1):
        """Phân tích predictions và quality check."""
        print("\n" + "="*70)
        print(f" PHÂN TÍCH PREDICTIONS - TASK {task}")
        print("="*70)

        # Load submission
        with open(submission_path, 'r', encoding='utf-8') as f:
            submission = json.load(f)

        print(f"\n📊 Submission Overview:")
        print(f"   Total predictions: {len(submission)}")

        if task == 1:
            self._analyze_task1_predictions(submission)
        else:
            self._analyze_task2_predictions(submission)

        print("\n" + "="*70)

    def _analyze_task1_predictions(self, submission):
        """Phân tích predictions cho Task 1."""
        # Check format
        errors = []
        total_predictions = 0

        for qid, item in submission.items():
            if 'answer' not in item:
                errors.append(f"{qid}: Missing 'answer' field")
                continue

            answers = item['answer']

            if not isinstance(answers, list):
                errors.append(f"{qid}: 'answer' is not a list")
                continue

            total_predictions += len(answers)

            if len(answers) > 5:
                errors.append(f"{qid}: Too many answers ({len(answers)})")

            if len(answers) != len(set(answers)):
                errors.append(f"{qid}: Duplicate doc_ids")

        print(f"\n✅ Format Validation:")
        if errors:
            print(f"   ❌ {len(errors)} errors found:")
            for err in errors[:5]:
                print(f"      - {err}")
        else:
            print(f"   ✅ All predictions valid!")

        # Statistics
        answer_counts = [len(item['answer']) for item in submission.values()]

        print(f"\n📈 Prediction Statistics:")
        print(f"   Total doc IDs predicted: {total_predictions}")
        print(f"   Avg predictions/question: {sum(answer_counts)/len(answer_counts):.2f}")

        # Distribution
        count_dist = Counter(answer_counts)
        print(f"\n   Prediction count distribution:")
        for k in sorted(count_dist.keys()):
            pct = count_dist[k] / len(submission) * 100
            bar = "█" * int(pct / 2)
            print(f"      {k} docs: {count_dist[k]:>4} ({pct:5.1f}%) {bar}")

    def _analyze_task2_predictions(self, submission):
        """Phân tích predictions cho Task 2."""
        # Check format
        errors = []
        total_chars = 0
        empty_count = 0

        for qid, item in submission.items():
            if 'answer' not in item:
                errors.append(f"{qid}: Missing 'answer' field")
                continue

            answer = item['answer']

            if not isinstance(answer, str):
                errors.append(f"{qid}: 'answer' is not a string")
                continue

            if not answer.strip():
                errors.append(f"{qid}: Empty answer")
                empty_count += 1

            total_chars += len(answer)

        print(f"\n✅ Format Validation:")
        if errors:
            print(f"   ❌ {len(errors)} errors found:")
            for err in errors[:5]:
                print(f"      - {err}")
        else:
            print(f"   ✅ All predictions valid!")

        # Statistics
        a_lengths = [len(item['answer']) for item in submission.values()]

        print(f"\n📈 Prediction Statistics:")
        print(f"   Total characters: {total_chars:,}")
        print(f"   Avg answer length: {sum(a_lengths)/len(a_lengths):.0f} chars")
        print(f"   Min: {min(a_lengths)} chars")
        print(f"   Max: {max(a_lengths)} chars")
        print(f"   Empty answers: {empty_count}")

        # Distribution
        print(f"\n   Answer length distribution:")
        ranges = [(0, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, 10000)]
        for lo, hi in ranges:
            count = sum(1 for l in a_lengths if lo <= l < hi)
            pct = count / len(a_lengths) * 100
            bar = "█" * int(pct / 2)
            print(f"      {lo:>5}-{hi:<5}: {count:>4} ({pct:5.1f}%) {bar}")

    def compare_warmup_vs_predictions(self, warmup_path: str, submission_path: str, task: int = 1):
        """So sánh warmup (ground truth) vs predictions."""
        print("\n" + "="*70)
        print(" SO SÁNH WARMUP vs PREDICTIONS")
        print("="*70)

        with open(warmup_path, 'r', encoding='utf-8') as f:
            warmup = json.load(f)

        with open(submission_path, 'r', encoding='utf-8') as f:
            submission = json.load(f)

        # Recall calculation
        if task == 1:
            self._compare_task1(warmup, submission)
        else:
            self._compare_task2(warmup, submission)

        print("\n" + "="*70)

    def _compare_task1(self, warmup, submission):
        """So sánh Task 1: doc ID overlap."""
        total_recall = 0
        total_precision = 0
        total_relevant = 0
        total_predicted = 0

        for qid in warmup:
            if qid not in submission:
                continue

            relevant = set(warmup[qid].get('answer', []))
            predicted = set(submission[qid].get('answer', []))

            if not relevant or not predicted:
                continue

            hits = len(relevant & predicted)
            total_recall += hits / len(relevant)
            total_precision += hits / len(predicted)
            total_relevant += len(relevant)
            total_predicted += len(predicted)

        n = len([q for q in warmup if q in submission])

        if n > 0:
            print(f"\n📊 Task 1 Metrics (warmup vs submission):")
            print(f"   Questions evaluated: {n}")
            print(f"   Average Recall@5: {total_recall/n:.4f}")
            print(f"   Average Precision@5: {total_precision/n:.4f}")
            print(f"   Total relevant docs: {total_relevant}")
            print(f"   Total predicted: {total_predicted}")

    def _compare_task2(self, warmup, submission):
        """So sánh Task 2: text similarity."""
        # Simple word overlap (METEOR proxy)
        meteor_scores = []

        for qid in warmup:
            if qid not in submission:
                continue

            ref = warmup[qid].get('answer', '')
            pred = submission[qid].get('answer', '')

            if not ref or not pred:
                continue

            # Simple word overlap
            ref_words = set(re.findall(r'\w+', ref.lower()))
            pred_words = set(re.findall(r'\w+', pred.lower()))

            if not ref_words or not pred_words:
                continue

            overlap = len(ref_words & pred_words)
            precision = overlap / len(pred_words)
            recall = overlap / len(ref_words)

            if precision + recall > 0:
                f_mean = (10 * precision * recall) / (recall + 9 * precision)
                meteor_scores.append(f_mean)

        if meteor_scores:
            print(f"\n📊 Task 2 Metrics (warmup vs submission):")
            print(f"   Questions evaluated: {len(meteor_scores)}")
            print(f"   Avg METEOR (proxy): {sum(meteor_scores)/len(meteor_scores):.4f}")
            print(f"   Min: {min(meteor_scores):.4f}")
            print(f"   Max: {max(meteor_scores):.4f}")


def main():
    parser = argparse.ArgumentParser(description='Analyze DSC 2026 UIT data')
    parser.add_argument('--mode', choices=['warmup', 'corpus', 'predictions', 'compare'],
                       required=True, help='Analysis mode')
    parser.add_argument('--task', type=int, choices=[1, 2], default=1, help='Task number')
    parser.add_argument('--corpus-dir', type=str, help='Corpus directory (for corpus mode)')
    parser.add_argument('--submission', type=str, help='Submission file path (for predictions/compare mode)')
    parser.add_argument('--warmup', type=str, help='Warmup file path')

    args = parser.parse_args()

    analyzer = DataAnalyzer()

    # Default paths
    base_dir = Path("d:/projects/UIT DSC")
    task_dir = base_dir / ("Task1 - LegalIR" if args.task == 1 else "Task2 - LegalQA")

    if args.mode == 'warmup':
        warmup_path = args.warmup or str(task_dir / "warmup.json")
        analyzer.analyze_warmup(task=args.task, warmup_path=warmup_path)

    elif args.mode == 'corpus':
        corpus_dir = args.corpus_dir or str(task_dir / "data" / "raw" / "corpus")
        analyzer.analyze_corpus(corpus_dir)

    elif args.mode == 'predictions':
        submission_path = args.submission or str(task_dir / "output" / "submission.json")
        analyzer.analyze_predictions(submission_path, task=args.task)

    elif args.mode == 'compare':
        warmup_path = args.warmup or str(task_dir / "warmup.json")
        submission_path = args.submission or str(task_dir / "output" / "submission.json")
        analyzer.compare_warmup_vs_predictions(warmup_path, submission_path, task=args.task)

    # Recommendations
    print("\n💡 RECOMMENDATIONS FOR OFFICIAL ROUND:")
    print("="*70)

    if args.mode in ['warmup', 'compare']:
        print("\n   1. Khi BTC cung cấp corpus:")
        print("      - Build BM25 index cho legal documents")
        print("      - Index dense embeddings bằng MiniLM-L12 (~118M params)")
        print("      - Apply RRF fusion + cross-encoder reranking")

        print("\n   2. Cho Task 1 (Document Retrieval):")
        print("      - Top-K retrieval = 20, rerank xuống 5")
        print("      - Deduplicate doc_ids trước khi submit")

        print("\n   3. Cho Task 2 (Question Answering):")
        print("      - Passage-level retrieval với chunk 512 chars")
        print("      - Extract relevant spans từ top passages")
        print("      - Synthesize answer với proper citations")

        print("\n   4. Performance optimization:")
        print("      - Cache embeddings để tăng tốc")
        print("      - Batch processing cho multiple queries")
        print("      - Use GPU nếu có để tăng tốc inference")

    print("\n   5. Format requirements:")
    print("      - Task 1: {qid: {answer: [doc_ids]}} - max 5, no duplicates")
    print("      - Task 2: {qid: {answer: 'text'}} - non-empty string")
    print("      - ZIP chỉ chứa submission.json")


if __name__ == "__main__":
    main()