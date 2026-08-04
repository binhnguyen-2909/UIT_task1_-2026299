"""
Task 2 Evaluation - METEOR & ROUGE-L
======================================
Tính METEOR và ROUGE-L scores cho Task 2 predictions.

Note: Đây là simplified implementation. Để có kết quả chính xác,
nên dùng official packages: nltk.translate.meteor_score và rouge_score.
"""

import json
import sys
from pathlib import Path
from collections import Counter
import re
from typing import List, Set

sys.stdout.reconfigure(encoding='utf-8')


def tokenize(text: str) -> List[str]:
    """Simple Vietnamese tokenizer."""
    text = text.lower().strip()
    # Remove punctuation nhưng giữ lại words
    words = re.findall(r'[\w]+', text, re.UNICODE)
    return words


def lcs_length(x: List, y: List) -> int:
    """Longest Common Subsequence length."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i-1] == y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]


def rouge_l(prediction: str, reference: str, beta: float = 1.2) -> float:
    """
    Compute ROUGE-L F-measure.

    ROUGE-L = ((1 + beta^2) * P * R) / (R + beta^2 * P)
    P = LCS(pred, ref) / len(pred)
    R = LCS(pred, ref) / len(ref)
    """
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = lcs_length(pred_tokens, ref_tokens)

    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    f_score = ((1 + beta**2) * precision * recall) / (recall + beta**2 * precision)
    return f_score


def meteor_score(prediction: str, reference: str) -> float:
    """
    Simplified METEOR score.

    METEOR = F_mean * (1 - Penalty)
    F_mean = (10 * P * R) / (R + 9 * P)
    """
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    # Count matches
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)

    matches = sum((pred_counter & ref_counter).values())

    if matches == 0:
        return 0.0

    # Precision and Recall
    precision = matches / len(pred_tokens)
    recall = matches / len(ref_tokens)

    # F-mean (weighted harmonic mean)
    f_mean = (10 * precision * recall) / (recall + 9 * precision)

    # Penalty for fragmentation
    # Simplified: count contiguous matches
    chunks = 0
    i = 0
    while i < len(pred_tokens):
        if pred_tokens[i] in ref_tokens:
            j = i
            while j < len(pred_tokens) and pred_tokens[j] in ref_tokens:
                j += 1
            chunks += 1
            i = j
        else:
            i += 1

    penalty = 0.5 * (chunks / matches) ** 3 if matches > 0 else 0

    score = f_mean * (1 - penalty)
    return score


def evaluate(predictions_path: str, references_path: str):
    """Evaluate predictions against references."""
    print("="*60)
    print("TASK 2 EVALUATION - METEOR & ROUGE-L")
    print("="*60)

    # Load
    with open(predictions_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    with open(references_path, 'r', encoding='utf-8') as f:
        references = json.load(f)

    print(f"\n📊 Evaluating {len(predictions)} predictions...")

    meteor_scores = []
    rouge_l_scores = []

    for qid in predictions:
        if qid not in references:
            continue

        pred_answer = predictions[qid].get('answer', '')
        ref_answer = references[qid].get('answer', '')

        if not pred_answer or not ref_answer:
            continue

        # Compute metrics
        meteor = meteor_score(pred_answer, ref_answer)
        rouge = rouge_l(pred_answer, ref_answer)

        meteor_scores.append(meteor)
        rouge_l_scores.append(rouge)

    # Aggregate (sample first 50 để demo nhanh)
    sample_size = min(50, len(meteor_scores))
    meteor_sample = meteor_scores[:sample_size]
    rouge_sample = rouge_l_scores[:sample_size]

    if meteor_scores:
        avg_meteor = sum(meteor_scores) / len(meteor_scores)
        avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)

        print(f"\n📈 Results:")
        print(f"   Evaluated: {len(meteor_scores)} questions")
        print(f"\n   METEOR (Primary):")
        print(f"      Average: {avg_meteor:.4f}")
        print(f"      Min:     {min(meteor_scores):.4f}")
        print(f"      Max:     {max(meteor_scores):.4f}")

        print(f"\n   ROUGE-L (Secondary):")
        print(f"      Average: {avg_rouge_l:.4f}")
        print(f"      Min:     {min(rouge_l_scores):.4f}")
        print(f"      Max:     {max(rouge_l_scores):.4f}")

        # Distribution (sample only)
        print(f"\n   METEOR Distribution (sample of {sample_size}):")
        ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        for lo, hi in ranges:
            count = sum(1 for s in meteor_sample if lo <= s < hi)
            pct = count / sample_size * 100 if sample_size > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"      {lo:.1f}-{hi:.1f}: {count:>4} ({pct:5.1f}%) {bar}")
    else:
        print("❌ No valid predictions to evaluate!")

    print("\n" + "="*60)


def main():
    pred_path = Path("d:/projects/UIT DSC/Task2 - LegalQA/output/submission.json")
    ref_path = Path("d:/projects/UIT DSC/Task2 - LegalQA/warmup.json")

    evaluate(str(pred_path), str(ref_path))


if __name__ == "__main__":
    main()