"""
Task 2 - LegalQA Full Pipeline
================================
Pipeline thực sự cho Task 2:
1. BM25 + Dense retrieval để tìm relevant passages
2. Extract relevant text spans
3. Synthesize full answer

Sử dụng khi BTC cung cấp corpus chính thức.
"""

import json
import zipfile
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
import math

sys.stdout.reconfigure(encoding='utf-8')

WARMUP_PATH = Path("d:/projects/UIT DSC/Task2 - LegalQA/warmup.json")
OUTPUT_DIR = Path("d:/projects/UIT DSC/Task2 - LegalQA/output")


class VietnameseTokenizer:
    """Vietnamese text tokenizer cho legal domain."""

    STOPWORDS = {
        'và', 'của', 'là', 'có', 'được', 'theo', 'nào', 'này', 'năm',
        'thế', 'ra', 'sao', 'như', 'đó', 'với', 'trong', 'cho',
        'để', 'tại', 'từ', 'hoặc', 'về', 'pháp', 'luật', 'quy', 'định',
        'hành', 'viên', 'người', 'doanh', 'nghiệp', 'cơ', 'quan', 'nhà',
        'nước', 'thực', 'hiện', 'hồ', 'sơ', 'thủ', 'tục', 'một', 'các',
        'khác', 'vào', 'bộ', 'sở', 'đến', 'khi', 'trên', 'dưới', 'sau',
        'trước', 'giữa', 'không', 'vì', 'nhưng', 'nếu', 'thì', 'đâu',
        'ai', 'ở', 'đây', 'nên', 'hay', 'lại', 'rồi', 'cứ',
        'nhất', 'hết', 'mà', 'cũng', 'chỉ', 'còn', 'lúc', 'bao', 'giờ',
        'muốn', 'cần', 'phải', 'bị', 'gì', 'thế', 'nào'
    }

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text thành list các từ."""
        text = text.lower().strip()
        words = re.findall(r'[\w]+', text, re.UNICODE)

        # Filter stopwords
        return [w for w in words if w not in self.STOPWORDS and len(w) > 1]


class BM25:
    """BM25 sparse retrieval."""

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = Counter()
        self.doc_lens = {}
        self.avg_doc_len = 0
        self.docs = {}
        self.tokenizer = VietnameseTokenizer()

    def fit(self, docs: Dict[str, str]):
        """Index documents."""
        self.docs = docs
        total_len = 0

        for doc_id, text in docs.items():
            tokens = self.tokenizer.tokenize(text)
            self.doc_lens[doc_id] = len(tokens)

            # Document frequency
            for token in set(tokens):
                self.doc_freqs[token] += 1

            total_len += len(tokens)

        self.avg_doc_len = total_len / len(docs) if docs else 0

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search top-k documents."""
        query_tokens = self.tokenizer.tokenize(query)
        scores = {}

        N = len(self.docs)

        for doc_id, text in self.docs.items():
            doc_tokens = self.tokenizer.tokenize(text)
            doc_len = self.doc_lens[doc_id]

            score = 0
            for qt in query_tokens:
                if qt not in self.doc_freqs:
                    continue

                # TF
                tf = doc_tokens.count(qt)

                # IDF
                df = self.doc_freqs[qt]
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

                # BM25 score
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)

                score += idf * (numerator / denominator)

            if score > 0:
                scores[doc_id] = score

        # Sort by score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]


class AnswerSynthesizer:
    """Synthesize answer từ retrieved passages."""

    def __init__(self):
        self.tokenizer = VietnameseTokenizer()

    def extract_relevant_spans(self, query: str, passage: str, max_spans: int = 3) -> List[str]:
        """Extract relevant text spans từ passage."""
        query_tokens = set(self.tokenizer.tokenize(query))
        passage_lower = passage.lower()

        # Split passage into sentences
        sentences = re.split(r'[.!?]\s+', passage)

        # Score sentences by overlap với query
        scored = []
        for sent in sentences:
            sent_tokens = set(self.tokenizer.tokenize(sent))
            overlap = len(query_tokens & sent_tokens)
            if overlap > 0:
                scored.append((sent, overlap))

        # Sort by overlap
        scored.sort(key=lambda x: x[1], reverse=True)

        return [s for s, _ in scored[:max_spans]]

    def synthesize(self, query: str, passages: List[Tuple[str, str, float]]) -> str:
        """
        Synthesize answer từ top passages.

        Args:
            query: Question
            passages: List of (doc_id, text, score)

        Returns:
            Synthesized answer
        """
        if not passages:
            return "Không tìm thấy thông tin liên quan."

        # Extract relevant spans from top passages
        all_spans = []
        for doc_id, text, score in passages[:3]:  # Top 3 passages
            spans = self.extract_relevant_spans(query, text, max_spans=2)
            all_spans.extend(spans)

        if all_spans:
            # Combine spans
            answer = " ".join(all_spans)

            # Truncate to reasonable length
            if len(answer) > 2000:
                answer = answer[:2000] + "..."

            return answer

        # Fallback: return first passage
        return passages[0][1][:1500]


class LegalQAPipeline:
    """Full pipeline cho Task 2."""

    def __init__(self, corpus_dir: str = None):
        self.bm25 = BM25()
        self.synthesizer = AnswerSynthesizer()
        self.corpus = {}

    def load_corpus(self, corpus_dir: str):
        """
        Load corpus từ directory.

        Expected format:
        - JSON file với {"doc_id": "text", ...}
        - Hoặc nhiều .txt files
        """
        corpus_path = Path(corpus_dir)

        if not corpus_path.exists():
            print(f"⚠️ Corpus directory not found: {corpus_dir}")
            return False

        # Try JSON first
        json_files = list(corpus_path.glob("*.json"))
        if json_files:
            with open(json_files[0], 'r', encoding='utf-8') as f:
                self.corpus = json.load(f)
            print(f"   Loaded {len(self.corpus)} docs from {json_files[0]}")
            return True

        # Try txt files
        txt_files = list(corpus_path.glob("*.txt"))
        for txt_file in txt_files:
            doc_id = txt_file.stem
            with open(txt_file, 'r', encoding='utf-8') as f:
                self.corpus[doc_id] = f.read()

        if self.corpus:
            print(f"   Loaded {len(self.corpus)} docs from txt files")
            return True

        return False

    def build_index(self):
        """Build BM25 index."""
        if not self.corpus:
            print("⚠️ No corpus loaded!")
            return False

        print("   Building BM25 index...")
        self.bm25.fit(self.corpus)
        print(f"   Indexed {len(self.corpus)} docs")
        return True

    def answer_question(self, query: str, top_k: int = 5) -> str:
        """Generate answer for question."""
        # Retrieve
        results = self.bm25.search(query, top_k=top_k)

        if not results:
            return "Không tìm thấy thông tin liên quan đến câu hỏi."

        # Get passages
        passages = [(doc_id, self.corpus[doc_id], score) for doc_id, score in results]

        # Synthesize answer
        answer = self.synthesizer.synthesize(query, passages)
        return answer

    def run(self, warmup_path: str, output_path: str):
        """Run pipeline trên warmup data."""
        # Load warmup
        with open(warmup_path, 'r', encoding='utf-8') as f:
            warmup_data = json.load(f)

        print(f"\n📊 Processing {len(warmup_data)} questions...")

        submission = {}
        for i, (qid, item) in enumerate(warmup_data.items(), 1):
            query = item['question']
            answer = self.answer_question(query)
            submission[qid] = {"answer": answer}

            if i % 50 == 0:
                print(f"   Processed {i}/{len(warmup_data)}")

        # Save submission
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        json_path = output_path.parent / "submission.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(submission, f, ensure_ascii=False, indent=2)

        # Create zip
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(json_path, "submission.json")

        print(f"\n✅ Submission saved: {output_path}")
        return submission


def main():
    print("="*60)
    print("TASK 2 - LEGAL QA PIPELINE")
    print("="*60)

    # Initialize pipeline
    pipeline = LegalQAPipeline()

    # Try to load corpus (if available)
    corpus_dir = "d:/projects/UIT DSC/Task2 - LegalQA/data/raw/corpus"
    if pipeline.load_corpus(corpus_dir):
        pipeline.build_index()
    else:
        print("\n⚠️ No corpus available - using fallback mode")
        print("   BTC sẽ cung cấp corpus trong giai đoạn chính thức")
        return

    # Run pipeline
    pipeline.run(
        warmup_path=str(WARMUP_PATH),
        output_path=str(OUTPUT_DIR / "submission.zip")
    )

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()