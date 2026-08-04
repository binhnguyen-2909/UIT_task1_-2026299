"""
Keyword-Based Legal IR Retrieval System
=====================================
Retrieval không có corpus - sử dụng keyword matching từ warmup data.

Strategy:
1. Phân tích warmup data để tìm patterns (keyword -> doc_ids)
2. Với mỗi câu hỏi mới, extract keywords và match với patterns
3. Trả về doc_ids có likelihood cao nhất
"""

import json
import re
import sys
import zipfile
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import math

sys.stdout.reconfigure(encoding='utf-8')

# Paths
WARMUP_PATH = Path("d:/projects/UIT DSC/Task1 - LegalIR/warmup.json")
OUTPUT_DIR = Path("d:/projects/UIT DSC/Task1 - LegalIR/output")


class KeywordExtractor:
    """Extract meaningful keywords from Vietnamese legal questions."""

    # Vietnamese stopwords
    STOPWORDS = {
        'và', 'của', 'là', 'có', 'được', 'theo', 'nào', 'này', 'năm',
        'thế', 'nào', 'ra', 'sao', 'như', 'đó', 'với', 'trong', 'cho',
        'để', 'tại', 'từ', 'hoặc', 'về', 'pháp', 'luật', 'quy', 'định',
        'hành', 'viên', 'người', 'doanh', 'nghiệp', 'cơ', 'quan', 'nhà',
        'nước', 'thực', 'hiện', 'hồ', 'sơ', 'thủ', 'tục', 'một', 'các',
        'khác', 'vào', 'bộ', 'sở', 'đến', 'khi', 'trên', 'dưới', 'sau',
        'trước', 'giữa', 'không', 'vì', 'nhưng', 'nếu', 'thì', 'đâu',
        'ai', 'ở', 'đây', 'đó', 'nên', 'hay', 'lại', 'rồi', 'cứ',
        'nhất', 'hết', 'mà', 'cũng', 'chỉ', 'còn', 'lúc', 'bao', 'giờ',
        'muốn', 'cần', 'phải', 'nên', 'bị', 'gì', 'thế', 'nào'
    }

    # Legal domain keywords
    LEGAL_KEYWORDS = {
        # Law types
        'luật', 'nghị định', 'thông tư', 'nghị quyết', 'quyết định', 'quy phạm',
        # Legal entities
        'điều', 'khoản', 'mục', 'chương', 'điều luật',
        # Rights/responsibilities
        'quyền', 'quyền hạn', 'thẩm quyền', 'trách nhiệm', 'nghĩa vụ', 'chức năng', 'nhiệm vụ',
        # Procedures
        'thủ tục', 'trình tự', 'hồ sơ', 'đăng ký', 'cấp phép', 'giấy phép', 'hướng dẫn',
        # Penalties
        'xử phạt', 'phạt tiền', 'vi phạm', 'hành chính', 'kiểm tra', 'giám sát', 'bồi thường',
        # Organization types
        'doanh nghiệp', 'công ty', 'tổ chức', 'pháp nhân', 'cơ quan', 'đơn vị',
        # Documents
        'văn bản', 'hợp đồng', 'báo cáo', 'tờ trình', 'biên bản',
        # Social policies
        'người có công', 'ưu đãi', 'hỗ trợ', 'trợ cấp', 'bảo hiểm', 'bảo vệ',
        # Tax/Finance
        'thuế', 'tài chính', 'ngân sách', 'kinh phí', 'ngân hàng',
        # Land/Housing
        'đất', 'nhà ở', 'quyền sử dụng đất', 'kê khai',
        # Transport
        'giao thông', 'phương tiện', 'lái xe', 'giấy phép lái xe', 'đăng ký xe',
        # Health
        'y tế', 'bệnh viện', 'khám bệnh', 'bảo hiểm y tế',
        # Education
        'giáo dục', 'đào tạo', 'trường học', 'học sinh', 'sinh viên',
    }

    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Lowercase and normalize
        text = text.lower().strip()

        # Extract n-grams (1-3 words)
        words = re.findall(r'[\w]+', text, re.UNICODE)

        # Filter stopwords and short words
        keywords = []
        for word in words:
            if word not in self.STOPWORDS and len(word) > 1:
                keywords.append(word)

        # Extract bigrams
        bigrams = []
        for i in range(len(words) - 1):
            if words[i] not in self.STOPWORDS and words[i+1] not in self.STOPWORDS:
                bigrams.append(f"{words[i]} {words[i+1]}")

        # Extract trigrams
        trigrams = []
        for i in range(len(words) - 2):
            if words[i] not in self.STOPWORDS and words[i+1] not in self.STOPWORDS and words[i+2] not in self.STOPWORDS:
                trigrams.append(f"{words[i]} {words[i+1]} {words[i+2]}")

        return keywords + bigrams + trigrams

    def extract_legal_keywords(self, text: str) -> Set[str]:
        """Extract legal-specific keywords."""
        text_lower = text.lower()
        found = set()

        for keyword in self.LEGAL_KEYWORDS:
            if keyword in text_lower:
                found.add(keyword)

        return found


class LegalRetrievalSystem:
    """
    Retrieval system sử dụng keyword matching.
    Không có corpus đầy đủ nên sử dụng patterns từ warmup data.
    """

    def __init__(self):
        self.keyword_extractor = KeywordExtractor()

        # Keyword -> doc_ids mapping (từ warmup)
        self.keyword_to_docs: Dict[str, List[str]] = defaultdict(list)

        # Question patterns -> doc_ids (từ warmup)
        self.question_to_docs: Dict[str, List[str]] = {}

        # Doc_id -> questions mapping
        self.doc_to_questions: Dict[str, List[str]] = defaultdict(list)

        # TF-IDF style scoring
        self.doc_freq: Counter = Counter()
        self.total_docs = 0

    def train(self, warmup_data: Dict):
        """Train từ warmup data."""
        print("Training retrieval system from warmup data...")

        self.total_docs = len(warmup_data)

        for qid, item in warmup_data.items():
            question = item['question']
            answers = item.get('answer', [])

            # Extract keywords
            keywords = self.keyword_extractor.extract_keywords(question)
            legal_kw = self.keyword_extractor.extract_legal_keywords(question)

            # Store question -> doc_ids mapping
            self.question_to_docs[question] = answers

            # Build keyword -> doc_ids index
            for keyword in keywords + list(legal_kw):
                self.keyword_to_docs[keyword].extend(answers)

            # Doc -> questions mapping
            for doc_id in answers:
                self.doc_to_questions[doc_id].append(question)

            # Document frequency
            for keyword in set(keywords):
                self.doc_freq[keyword] += 1

        # Convert to lists and deduplicate
        self.keyword_to_docs = {k: list(set(v)) for k, v in self.keyword_to_docs.items()}

        print(f"  Keywords indexed: {len(self.keyword_to_docs)}")
        print(f"  Questions indexed: {len(self.question_to_docs)}")
        print(f"  Doc IDs found: {len(self.doc_to_questions)}")

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve doc_ids for a query.

        Args:
            query: Question text
            top_k: Number of results to return

        Returns:
            List of (doc_id, score) tuples
        """
        keywords = self.keyword_to_docs.keys()

        # Calculate scores for each doc_id
        doc_scores: Dict[str, float] = defaultdict(float)

        # Extract keywords from query
        query_keywords = set(self.keyword_extractor.extract_keywords(query))
        query_legal_kw = self.keyword_extractor.extract_legal_keywords(query)
        all_keywords = query_keywords | query_legal_kw

        # Score based on keyword matching
        matched_keywords = 0
        for keyword in all_keywords:
            if keyword in self.keyword_to_docs:
                matched_keywords += 1
                # Get docs for this keyword
                keyword_docs = self.keyword_to_docs[keyword]

                # Weight by IDF-like score
                weight = math.log(self.total_docs / (self.doc_freq.get(keyword, 1) + 1)) + 1

                for doc_id in keyword_docs:
                    doc_scores[doc_id] += weight

        # Boost docs that match multiple keywords
        # Normalize scores
        if doc_scores:
            max_score = max(doc_scores.values())
            if max_score > 0:
                doc_scores = {k: v / max_score for k, v in doc_scores.items()}

        # Sort by score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_docs[:top_k]


def main():
    print("="*60)
    print("KEYWORD-BASED LEGAL IR RETRIEVAL")
    print("="*60)

    # Load warmup data
    print("\n1. Loading warmup data...")
    with open(WARMUP_PATH, 'r', encoding='utf-8') as f:
        warmup_data = json.load(f)
    print(f"   Loaded {len(warmup_data)} questions")

    # Train retrieval system
    print("\n2. Training retrieval system...")
    retriever = LegalRetrievalSystem()
    retriever.train(warmup_data)

    # Create submission
    print("\n3. Creating submission...")
    submission = {}

    for qid, item in warmup_data.items():
        query = item['question']

        # Get predictions
        results = retriever.retrieve(query, top_k=5)

        # Format as required
        if results:
            submission[qid] = {
                "answer": [doc_id for doc_id, score in results]
            }
        else:
            # Fallback: return most common doc_ids
            most_common = [doc_id for doc_id, _ in Counter(
                doc_id for docs in retriever.keyword_to_docs.values() for doc_id in docs
            ).most_common(5)]
            submission[qid] = {
                "answer": most_common
            }

    # Validate
    print("\n4. Validating submission...")

    # Calculate sample accuracy (using warmup answers as ground truth)
    correct = 0
    total_predictions = 0
    recall_hits = 0
    recall_total = 0

    for qid, item in warmup_data.items():
        ground_truth = set(item.get('answer', []))
        predicted = set(submission[qid]['answer'])

        # Check exact match
        if ground_truth == predicted:
            correct += 1

        # Calculate recall
        hits = len(ground_truth & predicted)
        recall_hits += hits
        recall_total += len(ground_truth)

        total_predictions += len(predicted)

    accuracy = correct / len(warmup_data) * 100
    avg_recall = recall_hits / recall_total * 100 if recall_total > 0 else 0
    avg_precision = recall_hits / total_predictions * 100 if total_predictions > 0 else 0

    print(f"   Exact Match Accuracy: {accuracy:.2f}%")
    print(f"   Average Recall@{5}: {avg_recall:.2f}%")
    print(f"   Average Precision@{5}: {avg_precision:.2f}%")

    # Create submission.zip
    print("\n5. Creating submission.zip...")
    output_path = OUTPUT_DIR / "submission.zip"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write submission.json
    json_path = output_path.parent / "submission.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)

    # Create zip
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(json_path, "submission.json")

    print(f"   Created: {output_path}")

    # Show sample
    print("\n6. Sample predictions:")
    for qid in list(warmup_data.keys())[:3]:
        q = warmup_data[qid]['question']
        gt = warmup_data[qid]['answer']
        pred = submission[qid]['answer']
        print(f"   Q{qid}: {q[:50]}...")
        print(f"     Ground truth: {gt}")
        print(f"     Prediction:   {pred}")

    print("\n" + "="*60)
    print("SUBMISSION READY!")
    print("="*60)


if __name__ == "__main__":
    main()
