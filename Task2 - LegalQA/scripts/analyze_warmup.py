"""
Analyze Task 2 Warmup Data
============================
Phân tích chi tiết warmup.json cho Task 2 Legal QA.
"""

import json
import sys
import re
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

WARMUP_PATH = Path("d:/projects/UIT DSC/Task2 - LegalQA/warmup.json")


def analyze():
    """Phân tích chi tiết Task 2 warmup data."""
    print("="*60)
    print("TASK 2 WARMUP DATA ANALYSIS")
    print("="*60)

    # Load
    with open(WARMUP_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"\n📊 Dataset Statistics:")
    print(f"   Total questions: {len(data)}")

    # Question length analysis
    q_lengths = [len(item['question']) for item in data.values()]
    print(f"\n📝 Question Length:")
    print(f"   Avg: {sum(q_lengths)/len(q_lengths):.0f} chars")
    print(f"   Min: {min(q_lengths)} chars")
    print(f"   Max: {max(q_lengths)} chars")

    # Answer length analysis
    a_lengths = [len(item['answer']) for item in data.values()]
    print(f"\n💬 Answer Length:")
    print(f"   Avg: {sum(a_lengths)/len(a_lengths):.0f} chars")
    print(f"   Min: {min(a_lengths)} chars")
    print(f"   Max: {max(a_lengths)} chars")

    # Distribution
    print(f"\n📈 Answer Length Distribution:")
    ranges = [(0, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, 10000)]
    for lo, hi in ranges:
        count = sum(1 for l in a_lengths if lo <= l < hi)
        pct = count/len(a_lengths)*100
        bar = "█" * int(pct/2)
        print(f"   {lo:>5}-{hi:<5} chars: {count:>4} ({pct:5.1f}%) {bar}")

    # Legal document references
    print(f"\n📚 Legal Document References:")
    all_refs = []
    for item in data.values():
        answer = item['answer']

        # Extract Điều X references
        dieu_refs = re.findall(r'Điều\s+\d+', answer)
        all_refs.extend(dieu_refs)

        # Extract Nghị định X
        nd_refs = re.findall(r'Nghị định\s+\d+/\d+/NĐ-CP', answer)
        all_refs.extend(nd_refs)

        # Extract Luật references
        luat_refs = re.findall(r'Luật\s+[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẬẪÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỘỖƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ][a-zàáảãạăắằẳẵặâấầẩậẫèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổộỗơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]+', answer)
        all_refs.extend(luat_refs)

    ref_counter = Counter(all_refs)
    print(f"   Total legal references: {len(all_refs)}")
    print(f"   Unique references: {len(ref_counter)}")
    print(f"\n   Top 10 most cited:")
    for ref, count in ref_counter.most_common(10):
        print(f"      {ref}: {count} times")

    # Sample analysis
    print(f"\n📋 Sample Questions & Answers:")
    print("="*60)

    for i, (qid, item) in enumerate(data.items()):
        if i >= 3:
            break

        print(f"\n[Q{qid}]")
        print(f"Question ({len(item['question'])} chars):")
        print(f"  {item['question']}")
        print(f"\nAnswer ({len(item['answer'])} chars):")
        print(f"  {item['answer'][:300]}...")

    print("\n" + "="*60)


if __name__ == "__main__":
    analyze()