"""CLI for DSC 2026 Task 1 LegalIR.

Examples:
  python scripts/main.py train --corpus selected-contexts.zip
  python scripts/main.py infer --input public-official.json --output output/submission.zip
  python scripts/main.py eval --predictions output/submission.zip --references warmup.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import statistics
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict

try:
    from .evaluate import LegalIREvaluator, analyze_warmup_data, official_metrics, print_analysis
    from .sqlite_retriever import SQLiteFTSRetriever
except ImportError:
    from evaluate import LegalIREvaluator, analyze_warmup_data, official_metrics, print_analysis
    from sqlite_retriever import SQLiteFTSRetriever


TASK_ROOT = Path(__file__).resolve().parent.parent
COMMON_SCRIPTS = TASK_ROOT.parent / "scripts"
if str(COMMON_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMMON_SCRIPTS))
from model_policy import validate_model_plan  # noqa: E402

DEFAULT_INDEX = TASK_ROOT / "data" / "processed" / "public_index"
DEFAULT_OUTPUT = TASK_ROOT / "output" / "submission.zip"
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def _load_submission(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as archive:
            names = [item.filename for item in archive.infolist() if not item.is_dir()]
            if names != ["submission.json"]:
                raise ValueError(f"ZIP must contain exactly submission.json, found {names}")
            data = json.loads(archive.read("submission.json").decode("utf-8"))
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Submission must be a JSON object")
    return data


def _write_bundle(submission: Dict[str, Any], output: Path) -> None:
    zip_path = output if output.suffix.lower() == ".zip" else output.with_suffix(".zip")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep an inspectable sidecar without overwriting another run in the same
    # directory.  The ZIP member remains exactly ``submission.json`` per BTC.
    json_path = zip_path.with_suffix(".json")
    payload = json.dumps(submission, ensure_ascii=False, indent=2).encode("utf-8")
    json_path.write_bytes(payload)
    info = zipfile.ZipInfo("submission.json", (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(info, payload)


def _build_retriever(args: argparse.Namespace) -> Any:
    if args.backend == "sqlite" and args.embedding_model:
        raise ValueError("--embedding-model is only supported by --backend hybrid")
    validate_model_plan(
        args.model_registry,
        {
            "embedding": args.embedding_model,
            "reranker": None if args.no_reranker else args.reranker_model,
        },
    )
    if args.backend == "sqlite":
        db_path = Path(args.index_dir) / "legalir.sqlite3"
        use_reranker = bool(args.reranker_model) and not args.no_reranker
        retriever = SQLiteFTSRetriever(
            db_path=db_path,
            use_reranker=use_reranker,
            reranker_model=None if args.no_reranker else args.reranker_model,
            device=args.device,
            top_k_initial=args.candidates,
            use_query_expansion=args.query_expansion,
            title_weight=args.title_weight,
            alias_weight=args.alias_weight,
            passage_weight=args.passage_weight,
            question_weight=args.question_weight,
            max_query_terms=args.max_query_terms,
            or_query_terms=args.or_query_terms,
            or_chunk_limit=args.or_chunk_limit,
            base_fusion_weight=args.base_fusion_weight,
            or_fusion_weight=args.or_fusion_weight,
            retrieval_rrf_k=args.retrieval_rrf_k,
            document_fts_db=getattr(args, "document_fts_db", None),
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        manifest = retriever.build(Path(args.corpus), train_data=args.train_data)
        LOGGER.info(
            "Built SQLite FTS5 index with %d documents in %.1fs",
            manifest["documents"], manifest["build_seconds"],
        )
        return retriever

    try:
        from .corpus_loader import LegalCorpusLoader
        from .pipeline import HybridRetriever
    except ImportError:
        from corpus_loader import LegalCorpusLoader
        from pipeline import HybridRetriever
    loader = LegalCorpusLoader(Path(args.corpus))
    documents = loader.load_corpus()
    chunks = loader.chunk_documents(documents, args.chunk_size, args.chunk_overlap)
    retriever = HybridRetriever(
        bm25_weight=args.bm25_weight,
        dense_weight=args.dense_weight,
        use_reranker=not args.no_reranker,
        use_query_expansion=args.query_expansion,
        top_k_initial=args.candidates,
        top_k_final=args.top_k,
        embedding_model=args.embedding_model,
        reranker_model=None if args.no_reranker else args.reranker_model,
        device=args.device,
    )
    retriever.index_documents(chunks, Path(args.index_dir))
    LOGGER.info("Indexed %d contexts as %d chunks", len(documents), len(chunks))
    return retriever


def command_train(args: argparse.Namespace) -> int:
    _build_retriever(args)
    return 0


def command_infer(args: argparse.Namespace) -> int:
    index = Path(args.index_dir)
    if args.corpus:
        retriever = _build_retriever(args)
    elif index.joinpath("legalir.sqlite3").exists():
        db_path = index / "legalir.sqlite3"
        saved = json.loads(db_path.with_suffix(".config.json").read_text(encoding="utf-8"))
        validate_model_plan(
            args.model_registry,
            {"embedding": None, "reranker": saved.get("reranker_model")},
        )
        retriever = SQLiteFTSRetriever.load(
            db_path,
            device=args.device,
            document_fts_db=args.document_fts_db,
        )
    elif index.joinpath("config.json").exists():
        try:
            from .pipeline import HybridRetriever
        except ImportError:
            from pipeline import HybridRetriever
        saved = json.loads(index.joinpath("config.json").read_text(encoding="utf-8"))
        validate_model_plan(
            args.model_registry,
            {"embedding": saved.get("embedding_model"), "reranker": saved.get("reranker_model")},
        )
        retriever = HybridRetriever.load(index, device=args.device)
    else:
        raise FileNotFoundError("No index found; provide --corpus to build one")
    if args.document_fts_db and not isinstance(retriever, SQLiteFTSRetriever):
        raise ValueError("--document-fts-db requires the SQLite backend")
    evaluator = LegalIREvaluator(retriever)
    input_path = Path(args.input)
    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    index_signature = None
    if isinstance(retriever, SQLiteFTSRetriever):
        stat = retriever.db_path.stat()
        index_signature = {
            "path": str(retriever.db_path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "backend": retriever.BACKEND,
        }
        if retriever.document_fts_db is not None:
            document_stat = retriever.document_fts_db.stat()
            index_signature["document_fts"] = {
                "path": str(retriever.document_fts_db),
                "size": document_stat.st_size,
                "mtime_ns": document_stat.st_mtime_ns,
            }
    signature = hashlib.sha256(json.dumps({
        "input_sha256": input_hash,
        "top_k": args.top_k,
        "index": index_signature,
    }, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    checkpoint_path = (
        Path(args.checkpoint_output)
        if args.checkpoint_output
        else Path(args.output).with_suffix(".checkpoint.json")
    )
    results = evaluator.run_inference(
        input_path,
        top_k=args.top_k,
        checkpoint_path=checkpoint_path,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        run_signature=signature,
    )
    submission = {
        qid: {"answer": list(dict.fromkeys(item["answer"]))[: args.top_k]}
        for qid, item in evaluator.predictions.items()
    }
    expected_ids = set(json.loads(Path(args.input).read_text(encoding="utf-8")))
    if set(submission) != expected_ids:
        raise RuntimeError("Internal error: output question IDs do not match input")
    if any(not 1 <= len(item["answer"]) <= 5 for item in submission.values()):
        raise RuntimeError("At least one query has zero or more than five results")
    _write_bundle(submission, Path(args.output))
    methods: Dict[str, int] = {}
    for result in results:
        method = result.results[0].method if result.results else "no-result"
        methods[method] = methods.get(method, 0) + 1
    latencies = [result.processing_time for result in results]
    audit = {
        "input": str(Path(args.input).resolve()),
        "output": str(Path(args.output).resolve()),
        "backend": getattr(retriever, "BACKEND", "hybrid"),
        "document_fts": (
            retriever.document_fts_info()
            if isinstance(retriever, SQLiteFTSRetriever)
            else {"enabled": False}
        ),
        "questions": len(results),
        "top_k": args.top_k,
        "answer_count_distribution": {
            str(size): sum(len(item["answer"]) == size for item in submission.values())
            for size in range(1, 6)
        },
        "first_result_methods": methods,
        "exact_memory_queries": sum(bool(result.metadata.get("exact_memory")) for result in results),
        "zero_score_fallback_queries": sum(
            bool(result.metadata.get("zero_score_fallback")) for result in results
        ),
        "latency_seconds": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "max": max(latencies) if latencies else 0.0,
            "total": sum(latencies),
        },
        "queries": [{
            "question_id": result.query_id,
            "answers": [item.doc_id for item in result.results],
            "scores": [item.score for item in result.results],
            "rerank_scores": [item.rerank_score for item in result.results],
            "metadata": result.metadata,
            "processing_time": result.processing_time,
        } for result in results],
    }
    audit_path = (
        Path(args.audit_output)
        if args.audit_output
        else Path(args.output).with_suffix(".audit.json")
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Wrote valid submission for %d questions to %s", len(submission), args.output)
    LOGGER.info("Wrote inference audit to %s", audit_path)
    if hasattr(retriever, "close"):
        retriever.close()
    return 0


def command_eval(args: argparse.Namespace) -> int:
    predictions = _load_submission(Path(args.predictions))
    references = json.loads(Path(args.references).read_text(encoding="utf-8"))
    if args.all_top_k:
        report = {
            f"top_{top_k}": official_metrics({
                qid: {"answer": item["answer"][:top_k]}
                for qid, item in predictions.items()
            }, references)
            for top_k in range(1, 6)
        }
    else:
        report = official_metrics(predictions, references)
    print(json.dumps(report, indent=2))
    return 0


def command_analyze(args: argparse.Namespace) -> int:
    print_analysis(analyze_warmup_data(Path(args.input)))
    return 0


def command_variants(args: argparse.Namespace) -> int:
    predictions = _load_submission(Path(args.predictions))
    output_dir = Path(args.output_dir)
    for top_k in range(1, 6):
        variant = {
            qid: {"answer": list(dict.fromkeys(item["answer"]))[:top_k]}
            for qid, item in predictions.items()
        }
        if any(not item["answer"] for item in variant.values()):
            raise ValueError("Cannot create variants from an empty prediction list")
        _write_bundle(variant, output_dir / f"{args.prefix}_top{top_k}.zip")
    LOGGER.info("Wrote top-1 through top-5 variants to %s", output_dir)
    return 0


def _retrieval_options(parser: argparse.ArgumentParser, corpus_required: bool = False) -> None:
    parser.add_argument(
        "--backend", choices=("sqlite", "hybrid"), default="sqlite",
        help="sqlite streams the official corpus to disk; hybrid is for small experiments",
    )
    parser.add_argument("--corpus", type=Path, required=corpus_required)
    parser.add_argument("--train-data", type=Path, help="Optional labelled data for aliases/exact memory")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--candidates", type=int, default=40)
    parser.add_argument("--top-k", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--title-weight", type=float, default=3.0)
    parser.add_argument("--alias-weight", type=float, default=2.0)
    parser.add_argument("--passage-weight", type=float, default=1.0)
    parser.add_argument("--question-weight", type=float, default=4.0)
    parser.add_argument(
        "--max-query-terms", type=int, default=12,
        help="Use the rarest indexed terms to keep official-corpus FTS queries bounded",
    )
    parser.add_argument("--or-query-terms", type=int, default=12)
    parser.add_argument("--or-chunk-limit", type=int, default=400)
    parser.add_argument("--base-fusion-weight", type=float, default=3.0)
    parser.add_argument("--or-fusion-weight", type=float, default=7.0)
    parser.add_argument("--retrieval-rrf-k", type=int, default=20)
    parser.add_argument("--embedding-model", help="Downloaded local model directory only")
    parser.add_argument("--reranker-model", help="Downloaded local model directory only")
    parser.add_argument("--model-registry", type=Path, help="Completed BTC-approved model registry JSON")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--query-expansion", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    _retrieval_options(train, corpus_required=True)
    train.set_defaults(handler=command_train)
    infer = commands.add_parser("infer")
    _retrieval_options(infer)
    infer.add_argument(
        "--document-fts-db",
        type=Path,
        help=(
            "Optional complete one-row-per-document FTS5 database used by "
            "the validated v3 exact-phrase fusion"
        ),
    )
    infer.add_argument("--input", type=Path, required=True)
    infer.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    infer.add_argument(
        "--audit-output", type=Path,
        help="Defaults next to --output with an .audit.json suffix",
    )
    infer.add_argument(
        "--checkpoint-output", type=Path,
        help="Persistent atomic progress file; defaults next to --output",
    )
    infer.add_argument("--checkpoint-every", type=int, default=10)
    infer.add_argument(
        "--resume", action="store_true",
        help="Resume a compatible interrupted inference checkpoint",
    )
    infer.set_defaults(handler=command_infer)
    evaluate = commands.add_parser("eval")
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--references", type=Path, required=True)
    evaluate.add_argument(
        "--all-top-k", action="store_true",
        help="Report official precision/recall after truncating predictions to k=1..5",
    )
    evaluate.set_defaults(handler=command_eval)
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--input", type=Path, required=True)
    analyze.set_defaults(handler=command_analyze)
    variants = commands.add_parser("variants")
    variants.add_argument("--predictions", type=Path, required=True)
    variants.add_argument("--output-dir", type=Path, default=TASK_ROOT / "output")
    variants.add_argument("--prefix", default="public_submission")
    variants.set_defaults(handler=command_variants)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
