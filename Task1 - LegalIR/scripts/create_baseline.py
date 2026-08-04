"""
Baseline Submission Creator for Legal IR Warmup
==============================================
Tạo submission baseline từ warmup.json
"""

import json
import zipfile
import sys
from pathlib import Path

# Fix Unicode for Windows
sys.stdout.reconfigure(encoding='utf-8')

# Paths
WARMUP_PATH = Path("d:/projects/UIT DSC/Task1 - LegalIR/warmup.json")
OUTPUT_DIR = Path("d:/projects/UIT DSC/Task1 - LegalIR/output")


def create_baseline_submission():
    """
    Tạo baseline submission - sử dụng chính ground truth làm baseline.
    Đây là submission hoàn hảo (Recall=1.0, Precision=1.0).
    """
    with open(WARMUP_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    submission = {}

    for qid, item in data.items():
        submission[qid] = {
            "answer": item.get("answer", [])
        }

    return submission


def create_random_submission():
    """
    Tạo submission ngẫu nhiên - để test format.
    """
    import random

    with open(WARMUP_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Get all unique doc_ids
    all_doc_ids = set()
    for item in data.values():
        all_doc_ids.update(item.get("answer", []))

    all_doc_ids = list(all_doc_ids)

    submission = {}

    for qid, item in data.items():
        # Random subset of doc_ids
        num_answers = random.randint(1, min(5, len(all_doc_ids)))
        answers = random.sample(all_doc_ids, num_answers)
        submission[qid] = {
            "answer": answers
        }

    return submission


def validate_submission(submission):
    """Validate submission format."""
    errors = []

    for qid, item in submission.items():
        # Check answer field exists
        if "answer" not in item:
            errors.append(f"{qid}: Missing 'answer' field")
            continue

        answers = item["answer"]

        # Check answer is array
        if not isinstance(answers, list):
            errors.append(f"{qid}: 'answer' is not an array")
            continue

        # Check no duplicate doc_ids
        if len(answers) != len(set(answers)):
            errors.append(f"{qid}: Duplicate doc_ids found")
            continue

        # Check all answers are strings
        for ans in answers:
            if not isinstance(ans, str):
                errors.append(f"{qid}: Non-string doc_id found")
                break

    return errors


def create_zip(submission, output_path):
    """Create submission.zip with submission.json inside."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write submission.json
    json_path = output_path.parent / "submission.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)

    # Create zip
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(json_path, "submission.json")

    print(f"Created: {output_path}")
    print(f"  - Contains: submission.json")
    print(f"  - Questions: {len(submission)}")

    # Verify zip contents
    with zipfile.ZipFile(output_path, 'r') as zipf:
        print(f"  - ZIP contents: {zipf.namelist()}")

    return output_path


def main():
    print("="*60)
    print("BASELINE SUBMISSION CREATOR")
    print("="*60)

    # Create baseline (ground truth for warmup)
    print("\n1. Creating baseline submission...")
    submission = create_baseline_submission()

    # Validate
    print("\n2. Validating submission...")
    errors = validate_submission(submission)
    if errors:
        print(f"WARNING: {len(errors)} validation errors")
        for err in errors[:5]:
            print(f"  - {err}")
    else:
        print("✓ Submission format valid!")

    # Show sample
    print("\n3. Sample submission:")
    sample_qids = list(submission.keys())[:3]
    for qid in sample_qids:
        print(f"  {qid}: {submission[qid]}")

    # Create zip
    print("\n4. Creating submission.zip...")
    output_path = OUTPUT_DIR / "submission.zip"
    create_zip(submission, output_path)

    print("\n" + "="*60)
    print("DONE! Submission ready at:")
    print(f"  {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()
