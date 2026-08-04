"""
Verify both Task 1 and Task 2 submissions
"""

import json
import zipfile
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

TASK1_DIR = Path("d:/projects/UIT DSC/Task1 - LegalIR")
TASK2_DIR = Path("d:/projects/UIT DSC/Task2 - LegalQA")


def verify_task1():
    """Verify Task 1 submission format."""
    print("="*60)
    print("VERIFYING TASK 1 SUBMISSION")
    print("="*60)

    zip_path = TASK1_DIR / "output" / "submission.zip"

    if not zip_path.exists():
        print(f"❌ ERROR: {zip_path} not found!")
        return False

    # Load from zip
    with zipfile.ZipFile(zip_path, 'r') as z:
        if "submission.json" not in z.namelist():
            print("❌ ERROR: submission.json not in zip!")
            return False

        content = z.read("submission.json").decode('utf-8')
        submission = json.loads(content)

    # Validate format
    errors = []
    total_answers = 0

    for qid, item in submission.items():
        if "answer" not in item:
            errors.append(f"{qid}: Missing 'answer' field")
            continue

        answers = item["answer"]

        if not isinstance(answers, list):
            errors.append(f"{qid}: 'answer' is not a list")
            continue

        total_answers += len(answers)

        if len(answers) > 5:
            errors.append(f"{qid}: More than 5 answers ({len(answers)})")

        if len(answers) != len(set(answers)):
            errors.append(f"{qid}: Duplicate doc_ids found")

        for ans in answers:
            if not isinstance(ans, str):
                errors.append(f"{qid}: Non-string doc_id")

    print(f"\n📊 Statistics:")
    print(f"   Total questions: {len(submission)}")
    print(f"   Total answers: {total_answers}")
    print(f"   Avg answers/question: {total_answers/len(submission):.2f}")

    if errors:
        print(f"\n❌ {len(errors)} errors found:")
        for err in errors[:10]:
            print(f"   - {err}")
        return False
    else:
        print("\n✅ Task 1 submission is VALID!")
        return True


def verify_task2():
    """Verify Task 2 submission format."""
    print("\n" + "="*60)
    print("VERIFYING TASK 2 SUBMISSION")
    print("="*60)

    zip_path = TASK2_DIR / "output" / "submission.zip"

    if not zip_path.exists():
        print(f"❌ ERROR: {zip_path} not found!")
        return False

    # Load from zip
    with zipfile.ZipFile(zip_path, 'r') as z:
        if "submission.json" not in z.namelist():
            print("❌ ERROR: submission.json not in zip!")
            return False

        content = z.read("submission.json").decode('utf-8')
        submission = json.loads(content)

    # Validate format
    errors = []
    total_chars = 0

    for qid, item in submission.items():
        if "answer" not in item:
            errors.append(f"{qid}: Missing 'answer' field")
            continue

        answer = item["answer"]

        if not isinstance(answer, str):
            errors.append(f"{qid}: 'answer' is not a string")
            continue

        total_chars += len(answer)

        if not answer.strip():
            errors.append(f"{qid}: Empty answer")

    print(f"\n📊 Statistics:")
    print(f"   Total questions: {len(submission)}")
    print(f"   Total characters: {total_chars:,}")
    print(f"   Avg answer length: {total_chars/len(submission):.0f} chars")

    if errors:
        print(f"\n❌ {len(errors)} errors found:")
        for err in errors[:10]:
            print(f"   - {err}")
        return False
    else:
        print("\n✅ Task 2 submission is VALID!")
        return True


def main():
    print("\n" + "="*60)
    print("SUBMISSION VERIFICATION")
    print("="*60)

    t1_ok = verify_task1()
    t2_ok = verify_task2()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Task 1 (Legal IR):  {'✅ VALID' if t1_ok else '❌ INVALID'}")
    print(f"Task 2 (Legal QA):  {'✅ VALID' if t2_ok else '❌ INVALID'}")

    if t1_ok and t2_ok:
        print("\n🎉 Both submissions are ready for upload!")
    else:
        print("\n⚠️ Please fix errors before uploading.")

    print("="*60)


if __name__ == "__main__":
    main()
