"""CLI for DSC 2026 Task 2 LegalQA (fully offline, no API server)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict

try:
    from .task2_pipeline import build_pipeline_from_corpus
except ImportError:
    from task2_pipeline import build_pipeline_from_corpus


TASK_ROOT = Path(__file__).resolve().parent.parent
COMMON_SCRIPTS = TASK_ROOT.parent / "scripts"
if str(COMMON_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMMON_SCRIPTS))
from model_policy import validate_model_plan  # noqa: E402


DEFAULT_OUTPUT = TASK_ROOT / "output" / "submission.zip"
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def load_questions(path: Path) -> Dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Input must be a JSON object keyed by question ID")
    output: Dict[str, str] = {}
    for qid, value in data.items():
        if not isinstance(value, dict) or not isinstance(value.get("question"), str):
            raise ValueError(f"{qid}: missing string question")
        output[str(qid)] = value["question"]
    return output


def write_bundle(submission: Dict[str, Any], output: Path) -> None:
    zip_path = output if output.suffix.lower() == ".zip" else output.with_suffix(".zip")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = zip_path.with_name("submission.json")
    payload = json.dumps(submission, ensure_ascii=False, indent=2).encode("utf-8")
    json_path.write_bytes(payload)
    info = zipfile.ZipInfo("submission.json", (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(info, payload)


def run(args: argparse.Namespace) -> int:
    compliance = validate_model_plan(
        args.model_registry,
        {
            "embedding": args.embedding_model,
            "reranker": args.reranker_model,
            "generator": args.generator_model,
        },
    )
    pipeline, loader = build_pipeline_from_corpus(
        args.corpus,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        generator_model=args.generator_model,
        device=args.device,
        candidates=args.candidates,
        answer_chunks=args.answer_chunks,
        max_answer_chars=args.max_answer_chars,
        sparse_weight=args.sparse_weight,
        semantic_weight=args.semantic_weight,
    )
    questions = load_questions(args.input)
    submission: Dict[str, Any] = {}
    audits: Dict[str, Any] = {}
    for index, (qid, question) in enumerate(questions.items(), 1):
        answer, audit = pipeline.answer(question)
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError(f"{qid}: empty answer")
        submission[qid] = {"answer": answer.strip()}
        audits[qid] = audit
        if index % 50 == 0:
            LOGGER.info("Processed %d/%d questions", index, len(questions))
    if set(submission) != set(questions):
        raise RuntimeError("Internal error: output IDs differ from input IDs")
    write_bundle(submission, args.output)
    audit_path = args.output.parent / "inference_audit.json"
    audit_path.write_text(json.dumps({
        "compliance": compliance,
        "corpus_documents": len(loader.documents),
        "questions": len(questions),
        "configuration": {
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "candidates": args.candidates,
            "answer_chunks": args.answer_chunks,
            "max_answer_chars": args.max_answer_chars,
            "sparse_weight": args.sparse_weight,
            "semantic_weight": args.semantic_weight,
        },
        "queries": audits,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %d answers to %s", len(submission), args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True, help="selected-contexts.zip or extracted directory")
    parser.add_argument("--input", type=Path, required=True, help="public-official.json or warmup.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=1800)
    parser.add_argument("--chunk-overlap", type=int, default=250)
    parser.add_argument("--candidates", type=int, default=60)
    parser.add_argument("--answer-chunks", type=int, default=5)
    parser.add_argument("--max-answer-chars", type=int, default=8000)
    parser.add_argument("--sparse-weight", type=float, default=1.0)
    parser.add_argument("--semantic-weight", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--embedding-model", help="BTC-approved downloaded local directory")
    parser.add_argument("--reranker-model", help="BTC-approved downloaded local directory")
    parser.add_argument("--generator-model", help="BTC-approved downloaded local directory")
    parser.add_argument("--model-registry", type=Path, help="Completed approved-model registry JSON")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
