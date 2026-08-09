"""Evaluate approved cross-encoders on one cached, leak-free candidate pool."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
TASK_SCRIPTS = Path(__file__).resolve().parent
for path in (ROOT / "scripts", TASK_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from analyze_candidate_oracle import ids, recall, source_rankings  # noqa: E402
from model_policy import validate_model_plan  # noqa: E402
from reranker import Reranker, reciprocal_rank_fuse  # noqa: E402
from sqlite_retriever import SQLiteFTSRetriever  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--details-output",
        type=Path,
        help=(
            "Optional JSONL audit with lexical order, neural order, and neural "
            "scores for each query. This enables shortlist/fusion diagnostics "
            "without scoring the cross-encoder again."
        ),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--candidate-count", type=int, default=100)
    parser.add_argument(
        "--score-candidates",
        type=int,
        default=35,
        help=(
            "Score only the first N candidates from a compatible larger cache. "
            "This does not alter cache identity and mirrors the runtime rerank shortlist."
        ),
    )
    parser.add_argument("--top-k-initial", type=int, default=100)
    parser.add_argument("--reranker-model", action="append", default=[])
    parser.add_argument("--embedding-model")
    parser.add_argument("--model-registry", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dense-weight", type=float, default=0.05)
    parser.add_argument("--question-dense-weight", type=float, default=0.02)
    parser.add_argument("--question-dense-candidates", type=int, default=20)
    parser.add_argument("--query-prefix", default="query: ")
    parser.add_argument("--passage-prefix", default="passage: ")
    parser.add_argument("--passages-per-doc", type=int, default=2)
    parser.add_argument(
        "--train-exemplars",
        type=Path,
        help=(
            "Leak-free labelled train JSON. When supplied, score the closest "
            "training questions attached to each candidate document as a "
            "separate supervised-memory branch."
        ),
    )
    parser.add_argument("--exemplars-per-doc", type=int, default=2)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--store-source-rankings",
        action="store_true",
        help="Persist branch document-ID rankings for offline RRF weight tuning",
    )
    return parser.parse_args()


def file_signature(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "sha256": digest,
    }


def expected_cache_meta(args: argparse.Namespace) -> Dict[str, Any]:
    index_stat = args.index.stat()
    return {
        "schema": 1,
        "data": file_signature(args.data),
        "index": {
            "path": str(args.index.resolve()),
            "size": index_stat.st_size,
            "mtime_ns": index_stat.st_mtime_ns,
        },
        "limit": args.limit,
        "candidate_count": args.candidate_count,
        "top_k_initial": args.top_k_initial,
        "passages_per_doc": args.passages_per_doc,
        "embedding_model": (
            str(Path(args.embedding_model).resolve()) if args.embedding_model else None
        ),
        "dense_weight": args.dense_weight,
        "question_dense_weight": args.question_dense_weight,
        "question_dense_candidates": args.question_dense_candidates,
        "query_prefix": args.query_prefix,
        "passage_prefix": args.passage_prefix,
        "store_source_rankings": bool(getattr(args, "store_source_rankings", False)),
    }


def build_cache(args: argparse.Namespace, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = json.loads(args.data.read_text(encoding="utf-8"))
    items = list(data.items())[: args.limit]
    retriever = SQLiteFTSRetriever(
        args.index,
        use_reranker=False,
        top_k_initial=args.top_k_initial,
        rerank_passages_per_doc=args.passages_per_doc,
        enable_dense=bool(args.embedding_model),
        dense_model_path=(
            str(Path(args.embedding_model).resolve()) if args.embedding_model else None
        ),
        dense_device=args.device,
        dense_batch_size=args.batch_size,
        dense_max_seq_length=384,
        dense_query_prefix=args.query_prefix,
        dense_passage_prefix=args.passage_prefix,
        dense_weight=args.dense_weight,
        question_dense_weight=args.question_dense_weight,
        question_dense_candidates=args.question_dense_candidates,
    )
    records: List[Dict[str, Any]] = []
    try:
        for position, (question_id, item) in enumerate(items, 1):
            rankings = source_rankings(retriever, str(item["question"]))
            ranking = rankings["production_fusion"][: args.candidate_count]
            record = {
                "question_id": str(question_id),
                "question": str(item["question"]),
                "gold": [str(value) for value in item.get("answer", [])],
                "candidates": ranking,
            }
            if getattr(args, "store_source_rankings", False):
                record["source_rankings"] = {
                    name: ids(values)
                    for name, values in rankings.items()
                }
            records.append(record)
            if position % 10 == 0:
                print(f"Candidate cache: {position}/{len(items)}", flush=True)
    finally:
        retriever.close()
    args.candidate_cache.parent.mkdir(parents=True, exist_ok=True)
    with args.candidate_cache.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"_meta": meta}, ensure_ascii=False) + "\n")
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def load_or_build_cache(args: argparse.Namespace) -> List[Dict[str, Any]]:
    meta = expected_cache_meta(args)
    if args.candidate_cache.is_file() and not args.rebuild_cache:
        with args.candidate_cache.open("r", encoding="utf-8") as stream:
            header = json.loads(next(stream))
            if header.get("_meta") != meta:
                raise ValueError(
                    "Candidate cache does not match data/index/options; use --rebuild-cache"
                )
            return [json.loads(line) for line in stream if line.strip()]
    return build_cache(args, meta)


def metrics(
    records: List[Dict[str, Any]], key: str, top_k: int | None = 5
) -> Dict[str, Any]:
    recalls: List[float] = []
    precisions: List[float] = []
    for record in records:
        predicted = record[key][:top_k] if top_k is not None else record[key]
        gold = [str(value) for value in record["gold"]]
        relevant = set(gold)
        matched = len(relevant & set(map(str, predicted)))
        recalls.append(matched / len(relevant) if relevant else 0.0)
        precisions.append(matched / len(predicted) if predicted else 0.0)
    return {
        "recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "precision": sum(precisions) / len(precisions) if precisions else 0.0,
        "per_question_recall": recalls,
    }


def rrf_predictions(
    lexical: List[str], neural: List[str], weight: float, rrf_k: int, top_k: int = 5
) -> List[str]:
    """Fuse ranks conservatively; lexical order wins deterministic ties."""

    fused = reciprocal_rank_fuse(
        [{"doc_id": str(doc_id)} for doc_id in lexical],
        [{"doc_id": str(doc_id)} for doc_id in neural],
        neural_weight=weight,
        rrf_k=rrf_k,
        top_k=top_k,
    )
    return ids(fused)


EXEMPLAR_TOKEN_RE = re.compile(
    r"[0-9]+(?:[./-][0-9a-z]+)*|[^\W_]+", re.UNICODE
)


def _exemplar_tokens(text: str) -> List[str]:
    return EXEMPLAR_TOKEN_RE.findall(
        unicodedata.normalize("NFC", str(text)).casefold()
    )


def exemplar_alignment(query: str, exemplar: str) -> float:
    """Cheap condition-preserving shortlist score before neural scoring."""

    query_tokens = _exemplar_tokens(query)
    exemplar_tokens = _exemplar_tokens(exemplar)
    if not query_tokens or not exemplar_tokens:
        return 0.0
    exemplar_set = set(exemplar_tokens)
    unigram_recall = sum(token in exemplar_set for token in query_tokens) / len(
        query_tokens
    )
    query_bigrams = list(zip(query_tokens, query_tokens[1:]))
    exemplar_bigrams = set(zip(exemplar_tokens, exemplar_tokens[1:]))
    bigram_recall = (
        sum(pair in exemplar_bigrams for pair in query_bigrams) / len(query_bigrams)
        if query_bigrams
        else 0.0
    )
    # Jaccard penalizes generic long exemplars while recalls preserve exact
    # legal conditions such as province names, dates, and procedure variants.
    query_set = set(query_tokens)
    jaccard = len(query_set & exemplar_set) / max(1, len(query_set | exemplar_set))
    return 0.45 * unigram_recall + 0.35 * bigram_recall + 0.20 * jaccard


def load_exemplar_bank(path: Path) -> Dict[str, List[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("--train-exemplars must be keyed by question ID")
    bank: Dict[str, List[str]] = defaultdict(list)
    seen: Dict[str, set[str]] = defaultdict(set)
    for item in data.values():
        if not isinstance(item, dict):
            continue
        question = item.get("question")
        answers = item.get("answer")
        if not isinstance(question, str) or not isinstance(answers, list):
            continue
        question = " ".join(question.split())
        if not question:
            continue
        for doc_id_value in answers:
            doc_id = str(doc_id_value)
            if question not in seen[doc_id]:
                seen[doc_id].add(question)
                bank[doc_id].append(question)
    return dict(bank)


def select_exemplars(
    query: str, exemplars: Sequence[str], limit: int
) -> List[str]:
    ranked = sorted(
        enumerate(exemplars),
        key=lambda pair: (-exemplar_alignment(query, pair[1]), pair[0]),
    )
    return [text for _, text in ranked[:limit]]


def exemplar_ranking(
    reranker: Reranker,
    query: str,
    candidates: Sequence[Dict[str, Any]],
    bank: Dict[str, List[str]],
    exemplars_per_doc: int,
) -> List[Dict[str, Any]]:
    """Rank seen documents by query-to-training-question relevance.

    Documents without a labelled exemplar are retained after scored documents
    in lexical order, so this branch can be fused without losing open-world
    corpus candidates.
    """

    scored: List[Dict[str, Any]] = []
    unscored: List[Dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)
        examples = select_exemplars(
            query, bank.get(str(item["doc_id"]), ()), exemplars_per_doc
        )
        if examples:
            item["passages"] = examples
            item["exemplar_count"] = len(examples)
            scored.append(item)
        else:
            unscored.append(item)
    ranked = reranker.rerank(query, scored, top_k=None) if scored else []
    return [*ranked, *unscored]


def multi_rrf_predictions(
    rankings: Sequence[Tuple[float, Sequence[str]]],
    rrf_k: int,
    top_k: int = 5,
) -> List[str]:
    """Fuse independent ranks once; use the first ranking for stable ties."""

    if rrf_k < 0 or any(weight < 0 for weight, _ in rankings):
        raise ValueError("RRF weights and k must be non-negative")
    scores: Dict[str, float] = defaultdict(float)
    first_rank: Dict[str, int] = {}
    universe: List[str] = []
    seen_universe: set[str] = set()
    for source_index, (weight, ranking) in enumerate(rankings):
        seen: set[str] = set()
        rank = 0
        for doc_id_value in ranking:
            doc_id = str(doc_id_value)
            if doc_id in seen:
                continue
            seen.add(doc_id)
            rank += 1
            if doc_id not in seen_universe:
                seen_universe.add(doc_id)
                universe.append(doc_id)
            if source_index == 0:
                first_rank[doc_id] = rank
            scores[doc_id] += float(weight) / (rrf_k + rank)
    missing_rank = len(universe) + 1
    return sorted(
        universe,
        key=lambda doc_id: (
            -scores[doc_id], first_rank.get(doc_id, missing_rank), doc_id
        ),
    )[:top_k]


def main() -> int:
    args = parse_args()
    if min(
        args.candidate_count,
        args.top_k_initial,
        args.batch_size,
        args.exemplars_per_doc,
    ) <= 0:
        raise ValueError("Candidate and batch limits must be positive")
    if args.embedding_model:
        validate_model_plan(
            args.model_registry,
            {"embedding": str(Path(args.embedding_model).resolve())},
        )
    records = load_or_build_cache(args)
    scored_candidate_count = args.score_candidates or args.candidate_count
    if not 1 <= scored_candidate_count <= args.candidate_count:
        raise ValueError(
            "--score-candidates must be between 1 and --candidate-count"
        )
    if scored_candidate_count < args.candidate_count:
        for record in records:
            record["candidates"] = record["candidates"][:scored_candidate_count]
    exemplar_bank = (
        load_exemplar_bank(args.train_exemplars) if args.train_exemplars else None
    )
    for record in records:
        record["baseline"] = ids(record["candidates"][:5])
        record["oracle_candidates"] = ids(record["candidates"])
    baseline = metrics(records, "baseline")
    oracle = metrics(records, "oracle_candidates", top_k=None)
    report: Dict[str, Any] = {
        "questions": len(records),
        "candidate_count": scored_candidate_count,
        "cache_candidate_count": args.candidate_count,
        "passages_per_doc": args.passages_per_doc,
        "baseline": {
            key: value for key, value in baseline.items() if key != "per_question_recall"
        },
        "candidate_oracle": {
            key: value for key, value in oracle.items() if key != "per_question_recall"
        },
        "train_exemplars": (
            {
                "path": str(args.train_exemplars.resolve()),
                "documents": len(exemplar_bank),
                "questions": sum(len(values) for values in exemplar_bank.values()),
                "per_document": args.exemplars_per_doc,
            }
            if exemplar_bank is not None
            else None
        ),
        "models": [],
    }
    model_details: Dict[str, List[Dict[str, Any]]] = {}
    for model_path_value in args.reranker_model:
        model_path = str(Path(model_path_value).resolve())
        validate_model_plan(args.model_registry, {"reranker": model_path})
        reranker = Reranker(
            model_path,
            device=args.device,
            batch_size=args.batch_size,
        )
        current_details: List[Dict[str, Any]] = []
        fusion_keys: List[tuple[str, float, int]] = []
        for rrf_k in (0, 5, 20, 60):
            for weight in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
                fusion_keys.append((f"rrf_k{rrf_k}_w{weight:g}", weight, rrf_k))
        for position, record in enumerate(records, 1):
            ranked = reranker.rerank(
                str(record["question"]),
                record["candidates"],
                top_k=None,
            )
            neural_ids = ids(ranked)
            lexical_ids = ids(record["candidates"])
            current_details.append({
                "question_id": str(record["question_id"]),
                "gold": [str(value) for value in record["gold"]],
                "lexical": lexical_ids,
                "neural": neural_ids,
                "neural_scores": {
                    str(item["doc_id"]): float(item["rerank_score"])
                    for item in ranked
                },
            })
            record["model_prediction"] = neural_ids[:5]
            record["model_prediction_full"] = neural_ids
            for key, weight, rrf_k in fusion_keys:
                record[key] = rrf_predictions(
                    lexical_ids, neural_ids, weight=weight, rrf_k=rrf_k
                )
            if position % 25 == 0:
                print(
                    f"Reranker {Path(model_path).name}: {position}/{len(records)}",
                    flush=True,
                )
        result = metrics(records, "model_prediction")
        deltas = [
            model_value - base_value
            for model_value, base_value in zip(
                result["per_question_recall"], baseline["per_question_recall"]
            )
        ]
        fusion_results = []
        for key, weight, rrf_k in fusion_keys:
            fused = metrics(records, key)
            fusion_results.append({
                "key": key,
                "weight": weight,
                "rrf_k": rrf_k,
                "recall": fused["recall"],
                "precision": fused["precision"],
            })
        fusion_results.sort(
            key=lambda item: (-item["recall"], -item["precision"], item["weight"], item["rrf_k"])
        )
        model_report: Dict[str, Any] = {
            "path": model_path,
            "recall": result["recall"],
            "precision": result["precision"],
            "rescued_questions": sum(delta > 0 for delta in deltas),
            "hurt_questions": sum(delta < 0 for delta in deltas),
            "unchanged_questions": sum(delta == 0 for delta in deltas),
            "best_fusions": fusion_results[:8],
        }
        if exemplar_bank is not None:
            for position, record in enumerate(records, 1):
                lexical_ids = ids(record["candidates"])
                passage_ids = list(map(str, record["model_prediction_full"]))
                exemplar_ranked = exemplar_ranking(
                    reranker,
                    str(record["question"]),
                    record["candidates"],
                    exemplar_bank,
                    args.exemplars_per_doc,
                )
                exemplar_ids = ids(exemplar_ranked)
                current_details[position - 1]["exemplar"] = exemplar_ids
                current_details[position - 1]["exemplar_scores"] = {
                    str(item["doc_id"]): float(item["rerank_score"])
                    for item in exemplar_ranked
                    if item.get("rerank_score") is not None
                }
                record["exemplar_prediction"] = exemplar_ids[:5]
                record["exemplar_prediction_full"] = exemplar_ids
                for rrf_k in (0, 5, 20, 60):
                    for passage_weight in (0.1, 0.25, 0.5, 1.0):
                        for exemplar_weight in (0.1, 0.25, 0.5, 1.0, 2.0):
                            key = (
                                f"hybrid_k{rrf_k}_p{passage_weight:g}_"
                                f"e{exemplar_weight:g}"
                            )
                            record[key] = multi_rrf_predictions(
                                (
                                    (1.0, lexical_ids),
                                    (passage_weight, passage_ids),
                                    (exemplar_weight, exemplar_ids),
                                ),
                                rrf_k=rrf_k,
                            )
                if position % 25 == 0:
                    print(
                        f"Exemplar {Path(model_path).name}: "
                        f"{position}/{len(records)}",
                        flush=True,
                    )
            exemplar_metrics = metrics(records, "exemplar_prediction")
            hybrid_results: List[Dict[str, Any]] = []
            for rrf_k in (0, 5, 20, 60):
                for passage_weight in (0.1, 0.25, 0.5, 1.0):
                    for exemplar_weight in (0.1, 0.25, 0.5, 1.0, 2.0):
                        key = (
                            f"hybrid_k{rrf_k}_p{passage_weight:g}_"
                            f"e{exemplar_weight:g}"
                        )
                        value = metrics(records, key)
                        hybrid_results.append({
                            "key": key,
                            "rrf_k": rrf_k,
                            "passage_weight": passage_weight,
                            "exemplar_weight": exemplar_weight,
                            "recall": value["recall"],
                            "precision": value["precision"],
                        })
            hybrid_results.sort(
                key=lambda item: (
                    -item["recall"],
                    -item["precision"],
                    item["passage_weight"] + item["exemplar_weight"],
                    item["rrf_k"],
                )
            )
            model_report["exemplar_only"] = {
                "recall": exemplar_metrics["recall"],
                "precision": exemplar_metrics["precision"],
            }
            model_report["best_hybrid_fusions"] = hybrid_results[:12]
        report["models"].append(model_report)
        model_details[model_path] = current_details
        del reranker
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.details_output:
        args.details_output.parent.mkdir(parents=True, exist_ok=True)
        with args.details_output.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({
                "_meta": {
                    "questions": len(records),
                    "candidate_count": scored_candidate_count,
                    "cache_candidate_count": args.candidate_count,
                    "models": list(model_details),
                }
            }, ensure_ascii=False) + "\n")
            for model_path, details in model_details.items():
                for detail in details:
                    stream.write(json.dumps({
                        "model": model_path,
                        **detail,
                    }, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
