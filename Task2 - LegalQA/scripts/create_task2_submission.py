"""
Task 2 - LegalQA Submission Creator
=================================
Tạo submission cho Task 2 (Hỏi đáp pháp luật)

Output format:
{
    "question_id": {
        "answer": "Câu trả lời text..."
    }
}

Metrics: METEOR (chính), ROUGE-L (phụ)
"""

import json
import zipfile
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

WARMUP_PATH = Path("d:/projects/UIT DSC/Task2 - LegalQA/warmup.json")
OUTPUT_DIR = Path("d:/projects/UIT DSC/Task2 - LegalQA/output")


def validate_submission(submission):
    """Validate Task 2 submission format."""
    errors = []

    for qid, item in submission.items():
        # Check answer field exists
        if "answer" not in item:
            errors.append(f"{qid}: Missing 'answer' field")
            continue

        answer = item["answer"]

        # Check answer is string
        if not isinstance(answer, str):
            errors.append(f"{qid}: 'answer' is not a string, got {type(answer).__name__}")
            continue

        # Check answer is not empty
        if not answer.strip():
            errors.append(f"{qid}: 'answer' is empty")

    return errors


def main():
    print("="*60)
    print("TASK 2 - LEGALQA SUBMISSION CREATOR")
    print("="*60)

    # Load warmup data
    print("\n1. Loading warmup data...")
    with open(WARMUP_PATH, 'r', encoding='utf-8') as f:
        warmup_data = json.load(f)
    print(f"   Loaded {len(warmup_data)} questions")

    # Create submission (use ground truth answers for warmup)
    print("\n2. Creating submission...")
    submission = {}

    for qid, item in warmup_data.items():
        answer = item.get('answer', '')
        submission[qid] = {
            "answer": answer
        }

    print(f"   Created {len(submission)} answers")

    # Validate
    print("\n3. Validating submission...")
    errors = validate_submission(submission)

    if errors:
        print(f"   WARNING: {len(errors)} validation errors")
        for err in errors[:5]:
            print(f"     - {err}")
    else:
        print("   [OK] Submission format valid!")

    # Stats
    print("\n4. Answer Statistics:")
    ans_lengths = [len(item['answer']) for item in submission.values()]
    print(f"   Total answers: {len(submission)}")
    print(f"   Avg length: {sum(ans_lengths)/len(ans_lengths):.0f} chars")
    print(f"   Min length: {min(ans_lengths)} chars")
    print(f"   Max length: {max(ans_lengths)} chars")

    # Sample
    print("\n5. Sample submission:")
    for qid in list(submission.keys())[:2]:
        q = warmup_data[qid]['question']
        a = submission[qid]['answer']
        print(f"\n   Q[{qid}]: {q[:60]}...")
        print(f"   A: {a[:100]}...")

    # Create zip
    print("\n6. Creating submission.zip...")
    output_path = OUTPUT_DIR / "submission.zip"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_path = output_path.parent / "submission.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(json_path, "submission.json")

    print(f"   Created: {output_path}")
    print(f"   ZIP contents: {zipfile.ZipFile(output_path).namelist()}")

    # Verify zip
    print("\n7. Verification:")
    with zipfile.ZipFile(output_path, 'r') as z:
        info = z.getinfo("submission.json")
        print(f"   submission.json size: {info.file_size} bytes")

    print("\n" + "="*60)
    print("SUBMISSION READY!")
    print(f"File: {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()
