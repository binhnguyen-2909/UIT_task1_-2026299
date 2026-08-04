"""
Corpus Analyzer for Official Round
===================================
Phân tích chi tiết corpus khi BTC cung cấp.

Phân tích:
- Document distribution và statistics
- Legal document type detection
- Vocabulary và keyword analysis
- Question-document matching potential
- Retrieval strategy recommendations
"""

import json
import sys
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
import math

sys.stdout.reconfigure(encoding='utf-8')


class CorpusAnalyzer:
    """Phân tích corpus cho vòng chính thức."""

    LEGAL_DOC_PATTERNS = {
        'luật': r'Luật\s+[\w\s]+?(?=\s+năm|\s+số|\s+\d{4}|\.|,|$)',
        'bộ_luật': r'Bộ luật\s+[\w\s]+?(?=\s+năm|\s+số|\s+\d{4}|\.|,|$)',
        'nghị_định': r'Nghị định\s+\d+/\d+/NĐ-CP',
        'thông_tư': r'Thông tư\s+\d+/\d+/TT-[A-Z]+',
        'quyết_định': r'Quyết định\s+\d+/\d+/QĐ-[A-Z]+',
        'nghị_quyết': r'Nghị quyết\s+\d+/\d+/NQ-[A-Z]+',
        'pháp_lệnh': r'Pháp lệnh\s+[\w\s]+',
        'hiến_pháp': r'Hiến pháp\s+[\w\s]*',
        'sắc_lệnh': r'Sắc lệnh\s+[\w\s]+',
        'chỉ_thị': r'Chỉ thị\s+\d+/\d+/CT-[A-Z]+',
    }

    LEGAL_CITATION_PATTERNS = {
        'điều': r'Điều\s+\d+',
        'khoản': r'Khoản\s+\d+',
        'mục': r'Mục\s+\d+',
        'chương': r'Chương\s+[IVX]+',
        'phụ_lục': r'Phụ lục\s+[\w\d]+',
        'điểm': r'Điểm\s+[a-zđ]\s+Khoản',
    }

    LEGAL_TOPICS = {
        'lao_động': ['lao động', 'bộ luật lao động', 'thỏa ước', 'hợp đồng lao động'],
        'chứng_khoán': ['chứng khoán', 'cổ phiếu', 'trái phiếu', 'đấu thầu'],
        'giáo_dục': ['giáo dục', 'đào tạo', 'học sinh', 'sinh viên', 'trường học'],
        'y_tế': ['y tế', 'bệnh viện', 'khám bệnh', 'bảo hiểm y tế'],
        'đất_đai': ['đất đai', 'quyền sử dụng đất', 'giấy chứng nhận'],
        'thuế': ['thuế', 'thuế thu nhập', 'thuế giá trị gia tăng'],
        'giao_thông': ['giao thông', 'phương tiện', 'lái xe', 'đăng ký xe'],
        'xây_dựng': ['xây dựng', 'công trình', 'giấy phép xây dựng'],
        'bảo_hiểm': ['bảo hiểm', 'bảo hiểm xã hội', 'bảo hiểm y tế'],
        'hành_chính': ['hành chính', 'thủ tục hành chính', 'công vụ'],
    }

    def __init__(self):
        self.corpus = {}
        self.doc_metadata = {}

    def load_corpus(self, corpus_path: str) -> bool:
        """Load corpus từ JSON file hoặc directory."""
        path = Path(corpus_path)

        if not path.exists():
            print(f"❌ Corpus path not found: {corpus_path}")
            return False

        # JSON file
        if path.is_file() and path.suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                self.corpus = json.load(f)
            print(f"✅ Loaded {len(self.corpus)} documents from {path}")
            return True

        # Directory of txt/json files
        if path.is_dir():
            json_files = list(path.glob("*.json"))
            txt_files = list(path.glob("*.txt"))

            for json_file in json_files:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.corpus.update(data)

            for txt_file in txt_files:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    self.corpus[txt_file.stem] = f.read()

            if self.corpus:
                print(f"✅ Loaded {len(self.corpus)} documents from {path}")
                return True

        print(f"❌ No documents found at {corpus_path}")
        return False

    def analyze(self):
        """Phân tích toàn diện corpus."""
        if not self.corpus:
            print("❌ No corpus loaded!")
            return

        print("\n" + "="*70)
        print(" CORPUS ANALYSIS - OFFICIAL ROUND")
        print("="*70)

        # 1. Basic statistics
        self._basic_statistics()

        # 2. Document type distribution
        self._document_types()

        # 3. Legal citations analysis
        self._legal_citations()

        # 4. Topic distribution
        self._topic_distribution()

        # 5. Vocabulary analysis
        self._vocabulary_analysis()

        # 6. Document quality
        self._quality_analysis()

        # 7. Retrieval recommendations
        self._retrieval_recommendations()

        print("\n" + "="*70)

    def _basic_statistics(self):
        """Thống kê cơ bản."""
        print(f"\n📊 Basic Statistics:")

        n = len(self.corpus)
        lengths = [len(text) for text in self.corpus.values()]

        print(f"   Total documents: {n}")
        print(f"   Total characters: {sum(lengths):,}")
        print(f"   Average length: {sum(lengths)/n:.0f} chars")
        print(f"   Min length: {min(lengths)} chars")
        print(f"   Max length: {max(lengths)} chars")
        print(f"   Median length: {sorted(lengths)[n//2]} chars")

        # Length distribution
        ranges = [(0, 1000), (1000, 5000), (5000, 10000), (10000, 50000), (50000, 1000000)]
        print(f"\n   Length distribution:")
        for lo, hi in ranges:
            count = sum(1 for l in lengths if lo <= l < hi)
            pct = count / n * 100
            bar = "█" * int(pct / 2)
            print(f"      {lo:>6}-{hi:<6}: {count:>4} ({pct:5.1f}%) {bar}")

    def _document_types(self):
        """Phân tích document types."""
        print(f"\n📋 Document Type Distribution:")

        type_counts = Counter()
        type_docs = defaultdict(list)

        for doc_id, text in self.corpus.items():
            detected = False
            for doc_type, pattern in self.LEGAL_DOC_PATTERNS.items():
                if re.search(pattern, text):
                    type_counts[doc_type] += 1
                    type_docs[doc_type].append(doc_id)
                    detected = True
                    break

            if not detected:
                type_counts['unknown'] += 1
                type_docs['unknown'].append(doc_id)

        n = len(self.corpus)
        for doc_type, count in type_counts.most_common():
            pct = count / n * 100
            bar = "█" * int(pct / 2)
            print(f"   {doc_type:15s}: {count:>4} ({pct:5.1f}%) {bar}")

        # Store for later use
        self.doc_metadata['types'] = dict(type_counts)

    def _legal_citations(self):
        """Phân tích legal citations trong corpus."""
        print(f"\n📚 Legal Citation Analysis:")

        all_citations = []
        for doc_id, text in self.corpus.items():
            for cit_type, pattern in self.LEGAL_CITATION_PATTERNS.items():
                matches = re.findall(pattern, text)
                all_citations.extend([(cit_type, m) for m in matches])

        type_counter = Counter([c[0] for c in all_citations])
        value_counter = Counter([c[1] for c in all_citations])

        print(f"   Total citations: {len(all_citations)}")
        print(f"\n   Citation types:")
        for cit_type, count in type_counter.most_common():
            pct = count / len(all_citations) * 100
            bar = "█" * int(pct / 2)
            print(f"      {cit_type:15s}: {count:>5} ({pct:5.1f}%) {bar}")

        print(f"\n   Top 10 most cited:")
        for citation, count in value_counter.most_common(10):
            print(f"      {citation}: {count} times")

    def _topic_distribution(self):
        """Phân tích legal topics."""
        print(f"\n🎯 Legal Topic Distribution:")

        topic_counts = Counter()

        for doc_id, text in self.corpus.items():
            text_lower = text.lower()

            for topic, keywords in self.LEGAL_TOPICS.items():
                matches = sum(1 for kw in keywords if kw in text_lower)
                if matches > 0:
                    topic_counts[topic] += matches

        total = sum(topic_counts.values())

        for topic, count in topic_counts.most_common():
            pct = count / total * 100 if total > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"   {topic:15s}: {count:>4} ({pct:5.1f}%) {bar}")

    def _vocabulary_analysis(self):
        """Phân tích vocabulary."""
        print(f"\n📖 Vocabulary Analysis:")

        all_words = set()
        word_counter = Counter()

        for text in self.corpus.values():
            words = re.findall(r'\w+', text.lower())
            for word in words:
                if len(word) > 2:  # Skip very short words
                    all_words.add(word)
                    word_counter[word] += 1

        print(f"   Total unique words: {len(all_words):,}")
        print(f"   Total word occurrences: {sum(word_counter.values()):,}")

        # Top words
        print(f"\n   Top 20 most frequent words:")
        for word, count in word_counter.most_common(20):
            print(f"      {word:15s}: {count:>5} times")

    def _quality_analysis(self):
        """Phân tích chất lượng documents."""
        print(f"\n✅ Quality Analysis:")

        # Empty/very short documents
        very_short = sum(1 for text in self.corpus.values() if len(text) < 100)
        print(f"   Very short docs (<100 chars): {very_short}")

        # Documents without legal markers
        no_markers = sum(1 for text in self.corpus.values()
                        if not any(re.search(p, text) for p in self.LEGAL_DOC_PATTERNS.values()))
        print(f"   Docs without legal markers: {no_markers}")

        # Duplicate detection (simple)
        text_hashes = {}
        duplicates = 0
        for doc_id, text in self.corpus.items():
            h = hash(text[:1000])
            if h in text_hashes:
                duplicates += 1
            text_hashes[h] = doc_id

        print(f"   Potential duplicates: {duplicates}")

    def _retrieval_recommendations(self):
        """Đề xuất retrieval strategy."""
        print(f"\n💡 Retrieval Strategy Recommendations:")

        n = len(self.corpus)
        avg_len = sum(len(t) for t in self.corpus.values()) / n if n > 0 else 0

        print(f"\n   1. BM25 Configuration:")
        print(f"      - k1 = 1.5 (term frequency saturation)")
        print(f"      - b = 0.75 (length normalization)")
        print(f"      - Vietnamese tokenizer (custom stopwords)")

        if avg_len > 5000:
            print(f"\n   2. Chunking Strategy:")
            print(f"      - Average length {avg_len:.0f} chars (large docs)")
            print(f"      - Chunk size: 512 chars")
            print(f"      - Overlap: 50 chars")
            print(f"      - ~{int(avg_len/462)} chunks per document")

        print(f"\n   3. Dense Retrieval:")
        print(f"      - Model: paraphrase-multilingual-MiniLM-L12-v2")
        print(f"      - Embedding dim: 384")
        print(f"      - Max sequence: 512 tokens")

        print(f"\n   4. Hybrid Fusion:")
        print(f"      - RRF with k=60")
        print(f"      - Top-K = 20 (initial) → 5 (final)")

        print(f"\n   5. Performance Estimation:")
        print(f"      - {n} documents → ~{int(n * avg_len / 512):,} chunks")
        print(f"      - Embedding time: ~{int(n * 0.1):.1f}s (CPU) / ~{int(n * 0.01):.1f}s (GPU)")
        print(f"      - Query time: <1s per query")

    def export_summary(self, output_path: str):
        """Export summary ra JSON."""
        summary = {
            'total_docs': len(self.corpus),
            'doc_metadata': self.doc_metadata,
            'analysis_date': str(Path(__file__).parent),
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n📄 Summary exported: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Analyze corpus for official round')
    parser.add_argument('--corpus', type=str, required=True, help='Corpus path (file or directory)')
    parser.add_argument('--export', type=str, help='Export summary to file')

    args = parser.parse_args()

    analyzer = CorpusAnalyzer()

    if analyzer.load_corpus(args.corpus):
        analyzer.analyze()

        if args.export:
            analyzer.export_summary(args.export)


if __name__ == "__main__":
    # Default usage
    base_dir = Path("d:/projects/UIT DSC")
    default_corpus = base_dir / "data" / "corpus.json"

    if default_corpus.exists():
        analyzer = CorpusAnalyzer()
        if analyzer.load_corpus(str(default_corpus)):
            analyzer.analyze()
    else:
        print(f"ℹ️  No corpus found at {default_corpus}")
        print(f"   Usage: python corpus_analyzer.py --corpus <path>")
        print(f"   Waiting for BTC to provide official corpus...")