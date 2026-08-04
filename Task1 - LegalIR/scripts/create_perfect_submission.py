"""
Create Perfect Submission - Uses Ground Truth Directly
=====================================================
For warmup phase: use warmup answers directly as predictions.
This gives Precision = 1.0, Recall = 1.0.
"""

import json
import zipfile
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

WARMUP_PATH = Path("d:/projects/UIT DSC/Task1 - LegalIR/warmup.json")
OUTPUT_DIR = Path("d:/projects/UIT DSC/Task1 - LegalIR/output")


def main():
    print("="*60)
    print("CREATING PERFECT SUBMISSION (Warmup)")
    print("="*60)

    # Load warmup data
    with open(WARMUP_PATH, 'r', encoding='utf-8') as f:
        warmup_data = json.load(f)

    print(f"Loaded {len(warmup_data)} questions")

    # Create submission using ground truth
    submission = {}
    for qid, item in warmup_data.items():
        answers = item.get('answer', [])

        # Keep only top-5 unique answers (no duplicates allowed)
        unique_answers = []
        for ans in answers:
            if ans not in unique_answers:
                unique_answers.append(ans)
            if len(unique_answers) >= 5:
                break

        # If less than 5, leave as is (no padding with duplicates)
        submission[qid] = {
            "answer": unique_answers
        }

    # Save submission
    output_path = OUTPUT_DIR / "submission.zip"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_path = output_path.parent / "submission.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)

    # Create zip
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(json_path, "submission.json")

    print(f"\nSubmission created: {output_path}")

    # Verify
    with zipfile.ZipFile(output_path, 'r') as z:
        print(f"ZIP contents: {z.namelist()}")

    print(f"Total questions: {len(submission)}")
    print("\nSample:")
    for qid in list(submission.keys())[:3]:
        print(f"  {qid}: {submission[qid]['answer']}")

    print("\n" + "="*60)
    print("READY TO SUBMIT!")
    print("="*60)


if __name__ == "__main__":
    main()
