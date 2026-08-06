"""One CLI for warmup, corpus, prediction and scorer-parity audits."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

from corpus_analyzer import CorpusAnalyzer
from submission_utils import load_submission, validate_submission


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


ROOT = Path(__file__).resolve().parent.parent


def task_dir(task: int) -> Path:
    return ROOT / ("Task1 - LegalIR" if task == 1 else "Task2 - LegalQA")


def load_object(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def analyze_warmup(path: Path, task: int) -> Dict[str, Any]:
    data = load_object(path)
    question_lengths = [len(item.get("question", "")) for item in data.values()]
    summary: Dict[str, Any] = {
        "task": task,
        "questions": len(data),
        "question_length": {
            "min": min(question_lengths),
            "max": max(question_lengths),
            "mean": statistics.fmean(question_lengths),
            "median": statistics.median(question_lengths),
        },
    }
    if task == 1:
        counts = [len(item.get("answer", [])) for item in data.values()]
        document_frequency: Counter[str] = Counter(
            str(doc_id) for item in data.values() for doc_id in item.get("answer", [])
        )
        summary.update({
            "answer_count_distribution": dict(sorted(Counter(counts).items())),
            "answer_documents": sum(counts),
            "unique_documents_in_answers": len(document_frequency),
            "top_documents": document_frequency.most_common(20),
        })
    else:
        lengths = [len(item.get("answer", "")) for item in data.values()]
        summary["answer_length"] = {
            "min": min(lengths),
            "max": max(lengths),
            "mean": statistics.fmean(lengths),
            "median": statistics.median(lengths),
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def analyze_predictions(path: Path, input_path: Path, task: int) -> Dict[str, Any]:
    predictions = load_submission(path)
    expected = load_object(input_path)
    errors = validate_submission(predictions, task, expected.keys())
    summary: Dict[str, Any] = {
        "task": task,
        "questions": len(predictions),
        "valid": not errors,
        "errors": errors,
    }
    if task == 1:
        lengths = [len(item.get("answer", [])) for item in predictions.values() if isinstance(item, dict)]
    else:
        lengths = [len(item.get("answer", "")) for item in predictions.values() if isinstance(item, dict)]
    summary["answer_length_or_count"] = dict(Counter(lengths).most_common(20))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def compare_task1(predictions: Dict[str, Any], references: Dict[str, Any]) -> Dict[str, float]:
    if set(predictions) != set(references):
        raise ValueError("Prediction and reference question ID sets differ")
    precision, recall = [], []
    for qid, reference in references.items():
        truth = reference.get("answer", [])
        predicted = predictions[qid].get("answer", [])
        valid = isinstance(predicted, list) and 0 < len(predicted) <= 5
        hits = len(set(truth) & set(predicted)) if valid else 0
        precision.append(hits / len(predicted) if valid else 0.0)
        recall.append(hits / len(truth) if valid and truth else 0.0)
    return {"precision": statistics.fmean(precision), "recall": statistics.fmean(recall)}


def compare_task2(predictions: Dict[str, Any], references: Dict[str, Any]) -> Dict[str, float]:
    evaluator_path = task_dir(2) / "scripts" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("task2_official_evaluator", evaluator_path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Cannot load {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate_data(predictions, references)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("warmup", "corpus", "predictions", "compare"), required=True)
    parser.add_argument("--task", type=int, choices=(1, 2), default=1)
    parser.add_argument("--warmup", type=Path)
    parser.add_argument("--input", type=Path, help="Question input for prediction ID validation")
    parser.add_argument("--corpus", "--corpus-dir", dest="corpus", type=Path)
    parser.add_argument("--submission", type=Path)
    parser.add_argument("--export", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    warmup = args.warmup or task_dir(args.task) / "warmup.json"
    if args.mode == "warmup":
        result = analyze_warmup(warmup, args.task)
    elif args.mode == "corpus":
        if not args.corpus:
            raise ValueError("--corpus is required in corpus mode")
        analyzer = CorpusAnalyzer()
        analyzer.load_corpus(args.corpus)
        result = analyzer.analyze()
    elif args.mode == "predictions":
        submission = args.submission or task_dir(args.task) / "output" / "submission.zip"
        result = analyze_predictions(submission, args.input or warmup, args.task)
    else:
        submission = args.submission or task_dir(args.task) / "output" / "submission.zip"
        predictions, references = load_submission(submission), load_object(warmup)
        result = compare_task1(predictions, references) if args.task == 1 else compare_task2(predictions, references)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
