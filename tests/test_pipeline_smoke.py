from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command):
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, encoding="utf-8")
    if completed.returncode:
        raise AssertionError(f"Command failed:\n{completed.stdout}\n{completed.stderr}")


class PipelineSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.work = Path(self.temporary.name)
        self.corpus = self.work / "selected-contexts.zip"
        contexts = [
            {"id": 740, "name": "Luật mẫu", "link": "local", "passage": "Điều 5. Người lao động có quyền nghỉ hằng năm."},
            {"id": 741, "name": "Nghị định mẫu", "link": "local", "passage": "Điều 9. Mức phạt đối với hành vi vi phạm là 2 triệu đồng."},
        ]
        with zipfile.ZipFile(self.corpus, "w") as archive:
            for index, context in enumerate(contexts):
                archive.writestr(f"context_{index}.json", json.dumps(context, ensure_ascii=False))
        self.input = self.work / "public-official.json"
        self.input.write_text(json.dumps({
            "q1": {"question": "Người lao động có quyền nghỉ hằng năm không?"},
            "q2": {"question": "Mức phạt hành vi vi phạm là bao nhiêu?"},
        }, ensure_ascii=False), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def test_task1_end_to_end(self):
        main = ROOT / "Task1 - LegalIR" / "scripts" / "main.py"
        index = self.work / "task1-index"
        output = self.work / "task1" / "submission.zip"
        run([sys.executable, str(main), "train", "--corpus", str(self.corpus), "--index-dir", str(index), "--no-reranker"])
        run([sys.executable, str(main), "infer", "--input", str(self.input), "--index-dir", str(index), "--output", str(output)])
        checkpoint = output.with_suffix(".checkpoint.json")
        self.assertEqual(json.loads(checkpoint.read_text(encoding="utf-8"))["status"], "complete")
        first_payload = output.read_bytes()
        run([
            sys.executable, str(main), "infer", "--input", str(self.input),
            "--index-dir", str(index), "--output", str(output), "--resume",
        ])
        self.assertEqual(output.read_bytes(), first_payload)
        with zipfile.ZipFile(output) as archive:
            data = json.loads(archive.read("submission.json"))
        self.assertEqual(set(data), {"q1", "q2"})
        self.assertEqual(data["q1"]["answer"][0], "740")
        self.assertEqual(data["q2"]["answer"][0], "741")

    def test_task1_optional_document_phrase_fts(self):
        main = ROOT / "Task1 - LegalIR" / "scripts" / "main.py"
        index = self.work / "task1-v3-index"
        output = self.work / "task1-v3" / "submission.zip"
        document_input = self.work / "document-public.json"
        document_input.write_text(
            json.dumps({"q3": {"question": "annual leave employee rights"}}),
            encoding="utf-8",
        )

        def create_document_database(path):
            connection = sqlite3.connect(path)
            connection.executescript("""
                CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE VIRTUAL TABLE documents USING fts5(
                    doc_id UNINDEXED,
                    title,
                    passage,
                    tokenize='unicode61 remove_diacritics 0'
                );
                INSERT INTO metadata(key, value) VALUES ('complete', '1');
            """)
            connection.executemany(
                "INSERT INTO documents(doc_id, title, passage) VALUES (?, ?, ?)",
                [
                    ("740", "leave law", "annual leave employee rights"),
                    ("741", "penalty law", "administrative monetary penalty"),
                ],
            )
            connection.commit()
            connection.close()

        document_db = self.work / "document-fts.sqlite3"
        other_document_db = self.work / "other-document-fts.sqlite3"
        create_document_database(document_db)
        create_document_database(other_document_db)
        run([
            sys.executable, str(main), "train", "--corpus", str(self.corpus),
            "--index-dir", str(index), "--no-reranker",
        ])
        run([
            sys.executable, str(main), "infer", "--input", str(document_input),
            "--index-dir", str(index), "--output", str(output),
            "--document-fts-db", str(document_db),
        ])
        audit = json.loads(output.with_suffix(".audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["backend"], "sqlite_chunk_fts5_v5_document_phrase_rrf")
        self.assertEqual(audit["document_fts"]["documents"], 2)
        self.assertEqual(audit["document_fts"]["phrase_weight"], 0.25)
        self.assertTrue(all(
            item["metadata"]["document_phrase"]["enabled"]
            for item in audit["queries"]
        ))
        self.assertTrue(any(
            item["metadata"]["document_phrase"]["candidates"] > 0
            for item in audit["queries"]
        ))

        incompatible = subprocess.run(
            [
                sys.executable, str(main), "infer", "--input", str(document_input),
                "--index-dir", str(index), "--output", str(output), "--resume",
                "--document-fts-db", str(other_document_db),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            encoding="utf-8",
        )
        self.assertNotEqual(incompatible.returncode, 0)
        self.assertIn("checkpoint", (incompatible.stdout + incompatible.stderr).lower())

    def test_task2_end_to_end(self):
        main = ROOT / "Task2 - LegalQA" / "scripts" / "main.py"
        output = self.work / "task2" / "submission.zip"
        run([sys.executable, str(main), "--corpus", str(self.corpus), "--input", str(self.input), "--output", str(output)])
        with zipfile.ZipFile(output) as archive:
            data = json.loads(archive.read("submission.json"))
        self.assertIn("nghỉ hằng năm", data["q1"]["answer"])
        self.assertIn("2 triệu đồng", data["q2"]["answer"])

    def test_official_corpus_analyzer_reads_zip_schema(self):
        analyzer = ROOT / "scripts" / "corpus_analyzer.py"
        report = self.work / "corpus-report.json"
        run([sys.executable, str(analyzer), "--corpus", str(self.corpus), "--export", str(report)])
        data = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(data["documents"], 2)


if __name__ == "__main__":
    unittest.main()
