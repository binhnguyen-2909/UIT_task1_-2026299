"""Task 1 inference helpers and metrics matching the BTC scorer."""

from __future__ import annotations

import json
import logging
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from .retrieval_types import QueryResult, RetrievalResult
except ImportError:
    from retrieval_types import QueryResult, RetrievalResult


LOGGER = logging.getLogger(__name__)


def load_questions(path: Path) -> Dict[str, str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Input must be a JSON object keyed by question ID")
    questions: Dict[str, str] = {}
    for qid, item in data.items():
        if not isinstance(item, dict) or not isinstance(item.get("question"), str):
            raise ValueError(f"{qid}: missing string question")
        questions[str(qid)] = item["question"]
    return questions


def official_metrics(predictions: Dict[str, Any], references: Dict[str, Any]) -> Dict[str, float]:
    """Reproduce ``Scoring-Program-Task-LegalIR/scoring.py`` exactly.

    The additional key-set check turns the scorer's eventual KeyError into an
    actionable message. Duplicates remain in the precision denominator, as in
    the BTC implementation.
    """
    if set(predictions) != set(references):
        missing = sorted(set(references) - set(predictions))
        extra = sorted(set(predictions) - set(references))
        raise ValueError(f"Question ID mismatch; missing={missing[:5]}, extra={extra[:5]}")
    precision_values: List[float] = []
    recall_values: List[float] = []
    for qid in references:
        truth_value = references[qid]
        truth = truth_value.get("answer", []) if isinstance(truth_value, dict) else truth_value
        prediction_value = predictions[qid]
        predicted = prediction_value.get("answer") if isinstance(prediction_value, dict) else None
        if not isinstance(truth, list) or not truth:
            raise ValueError(f"{qid}: reference answer must be a non-empty list")
        if not isinstance(predicted, list):
            raise ValueError(f"{qid}: prediction answer must be a list")
        valid = 0 < len(predicted) <= 5
        hits = len(set(map(str, truth)) & set(map(str, predicted))) if valid else 0
        recall_values.append(hits / len(truth) if valid else 0.0)
        precision_values.append(hits / len(predicted) if valid else 0.0)
    count = len(references)
    return {
        "precision": sum(precision_values) / count if count else 0.0,
        "recall": sum(recall_values) / count if count else 0.0,
    }


class LegalIREvaluator:
    def __init__(self, retriever: Any):
        self.retriever = retriever
        self.ground_truth: Dict[str, Dict[str, Any]] = {}
        self.predictions: Dict[str, Dict[str, Any]] = {}

    def load_ground_truth(self, path: Path) -> Dict[str, Dict[str, Any]]:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self.ground_truth = {
            str(qid): {
                "question": item["question"],
                "answer": [str(doc_id) for doc_id in item.get("answer", [])],
            }
            for qid, item in raw.items()
        }
        return self.ground_truth

    @staticmethod
    def _restore_result(payload: Dict[str, Any]) -> QueryResult:
        return QueryResult(
            query_id=str(payload["query_id"]),
            query=str(payload["query"]),
            results=[RetrievalResult(**item) for item in payload.get("results", [])],
            processing_time=float(payload.get("processing_time", 0.0)),
            metadata=dict(payload.get("metadata", {})),
        )

    @staticmethod
    def _write_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(path)

    def run_inference(
        self,
        input_path: Optional[Path] = None,
        top_k: int = 5,
        checkpoint_path: Optional[Path] = None,
        checkpoint_every: int = 10,
        resume: bool = False,
        run_signature: Optional[str] = None,
    ) -> List[QueryResult]:
        if checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive")
        questions = load_questions(input_path) if input_path else {
            qid: item["question"] for qid, item in self.ground_truth.items()
        }
        if input_path and run_signature is None:
            run_signature = hashlib.sha256(Path(input_path).read_bytes()).hexdigest()
        completed: Dict[str, Dict[str, Any]] = {}
        checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        if resume:
            if checkpoint_path is None or not checkpoint_path.exists():
                raise FileNotFoundError("--resume requested but checkpoint does not exist")
            saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if saved.get("signature") != run_signature or saved.get("top_k") != top_k:
                raise ValueError("Checkpoint is incompatible with this input/index/top-k")
            completed = {
                str(qid): item for qid, item in saved.get("completed", {}).items()
                if str(qid) in questions
            }
            LOGGER.info("Resumed %d/%d completed questions", len(completed), len(questions))

        result_by_id: Dict[str, QueryResult] = {
            qid: self._restore_result(payload) for qid, payload in completed.items()
        }
        self.predictions = {}
        total = len(questions)
        for position, (qid, question) in enumerate(questions.items(), 1):
            if qid in result_by_id:
                continue
            result = self.retriever.retrieve(question, qid, top_k)
            result_by_id[qid] = result
            completed[qid] = result.to_dict()
            completed_count = len(result_by_id)
            if checkpoint_path and (
                completed_count % checkpoint_every == 0 or completed_count == total
            ):
                self._write_checkpoint(checkpoint_path, {
                    "version": 1,
                    "status": "complete" if completed_count == total else "running",
                    "signature": run_signature,
                    "top_k": top_k,
                    "total": total,
                    "completed": completed,
                })
            if completed_count % checkpoint_every == 0 or completed_count == total:
                LOGGER.info("Inference progress: %d/%d questions", completed_count, total)
        results = [result_by_id[qid] for qid in questions]
        self.predictions = {
            qid: {"answer": [item.doc_id for item in result_by_id[qid].results]}
            for qid in questions
        }
        return results

    def evaluate(self, predictions: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        refs = {qid: item["answer"] for qid, item in self.ground_truth.items()}
        return official_metrics(predictions or self.predictions, refs)

    def save_submission(self, path: Path, top_k: int = 5) -> Dict[str, Any]:
        submission = {
            qid: {"answer": list(dict.fromkeys(item["answer"]))[:top_k]}
            for qid, item in self.predictions.items()
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")
        return submission


class WarmupProcessor:
    @staticmethod
    def load_warmup_queries(path: Path) -> List[Tuple[str, str]]:
        return list(load_questions(path).items())


def analyze_warmup_data(path: Path) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    lengths = [len(item.get("answer", [])) for item in data.values()]
    distribution = Counter(lengths)
    doc_frequency: Counter[str] = Counter(
        str(doc_id) for item in data.values() for doc_id in item.get("answer", [])
    )
    return {
        "num_queries": len(data),
        "total_ground_truth_docs": sum(lengths),
        "avg_answers_per_query": sum(lengths) / len(lengths) if lengths else 0.0,
        "min_answers": min(lengths) if lengths else 0,
        "max_answers": max(lengths) if lengths else 0,
        "answer_count_distribution": dict(sorted(distribution.items())),
        "num_unique_ground_truth_docs": len(doc_frequency),
        "most_frequent_documents": doc_frequency.most_common(20),
    }


def print_analysis(analysis: Dict[str, Any]) -> None:
    print(json.dumps(analysis, ensure_ascii=False, indent=2))
