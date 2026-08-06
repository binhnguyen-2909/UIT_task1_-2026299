"""Leak-free LegalIR document-family and co-label expansion benchmark.

The experiment deliberately does not modify the production retriever.  All
label-derived structures are rebuilt inside each training fold, while title
families use only corpus metadata (no relevance labels).
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Sequence, Tuple

try:
    from .bm25_retriever import VietnameseTokenizer
    from .evaluate import official_metrics
except ImportError:
    from bm25_retriever import VietnameseTokenizer
    from evaluate import official_metrics


def fts_expression(tokens: Sequence[str]) -> str:
    return " OR ".join(
        f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens
    )


def fold_index(
    records: Sequence[Tuple[str, Mapping[str, Any]]],
) -> Tuple[sqlite3.Connection, Dict[str, Counter[str]], Counter[str]]:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript("""
        CREATE VIRTUAL TABLE questions USING fts5(
            question_id UNINDEXED,
            answers UNINDEXED,
            question,
            tokenize='unicode61 remove_diacritics 0'
        );
        CREATE VIRTUAL TABLE aliases USING fts5(
            doc_id UNINDEXED,
            alias_text,
            tokenize='unicode61 remove_diacritics 0'
        );
    """)
    aliases: DefaultDict[str, List[str]] = defaultdict(list)
    label_count: Counter[str] = Counter()
    graph: DefaultDict[str, Counter[str]] = defaultdict(Counter)
    rows = []
    for qid, item in records:
        answers = list(dict.fromkeys(map(str, item["answer"])))
        question = str(item["question"])
        rows.append((qid, json.dumps(answers), question))
        label_count.update(answers)
        for doc_id in answers:
            aliases[doc_id].append(question)
            for other in answers:
                if other != doc_id:
                    graph[doc_id][other] += 1
    connection.executemany(
        "INSERT INTO questions(question_id, answers, question) VALUES (?, ?, ?)",
        rows,
    )
    connection.executemany(
        "INSERT INTO aliases(doc_id, alias_text) VALUES (?, ?)",
        ((doc_id, "\n".join(questions)) for doc_id, questions in aliases.items()),
    )
    connection.execute("INSERT INTO questions(questions) VALUES ('optimize')")
    connection.execute("INSERT INTO aliases(aliases) VALUES ('optimize')")
    return connection, dict(graph), label_count


def supervised_scores(
    connection: sqlite3.Connection,
    query_expression: str,
    neighbor_limit: int,
    rrf_k: int = 20,
) -> Tuple[Dict[str, float], List[str], Dict[str, float]]:
    if not query_expression:
        return {}, [], {
            "nearest_multi": 0.0,
            "top3_any_multi": 0.0,
            "top5_multi_fraction": 0.0,
            "weighted_multi_fraction": 0.0,
        }
    question_rows = list(connection.execute(
        """
        SELECT answers FROM questions WHERE questions MATCH ?
        ORDER BY bm25(questions, 0.0, 0.0, 1.0), question_id LIMIT ?
        """,
        (query_expression, neighbor_limit),
    ))
    question_doc_scores: DefaultDict[str, float] = defaultdict(float)
    decoded_answers = [list(map(str, json.loads(row["answers"]))) for row in question_rows]
    multi_weights = [1.0 / (rrf_k + rank) for rank, answers in enumerate(decoded_answers, 1) if len(answers) > 1]
    all_weights = [1.0 / (rrf_k + rank) for rank in range(1, len(decoded_answers) + 1)]
    ambiguity = {
        "nearest_multi": float(bool(decoded_answers and len(decoded_answers[0]) > 1)),
        "top3_any_multi": float(any(len(answers) > 1 for answers in decoded_answers[:3])),
        "top5_multi_fraction": (
            sum(len(answers) > 1 for answers in decoded_answers[:5])
            / max(1, min(5, len(decoded_answers)))
        ),
        "weighted_multi_fraction": sum(multi_weights) / max(1e-12, sum(all_weights)),
    }
    for rank, answers in enumerate(decoded_answers, 1):
        for doc_id in answers:
            question_doc_scores[doc_id] += 1.0 / (rrf_k + rank)
    question_ranking = sorted(
        question_doc_scores,
        key=lambda doc_id: (-question_doc_scores[doc_id], doc_id),
    )
    alias_ranking = [str(row["doc_id"]) for row in connection.execute(
        """
        SELECT doc_id FROM aliases WHERE aliases MATCH ?
        ORDER BY bm25(aliases, 0.0, 1.0), doc_id LIMIT ?
        """,
        (query_expression, neighbor_limit),
    )]
    scores: DefaultDict[str, float] = defaultdict(float)
    for ranking in (question_ranking, alias_ranking):
        for rank, doc_id in enumerate(ranking, 1):
            scores[doc_id] += 1.0 / (rrf_k + rank)
    ranking = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
    return dict(scores), ranking, ambiguity


def accent_fold(text: str) -> str:
    value = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in value if unicodedata.category(ch) != "Mn")


def family_key(title: str) -> str:
    """Make a deliberately conservative series key from a legal title."""
    value = accent_fold(title)
    # Remove website article ids and replace legal document/part numbers.  Year
    # is also removed so amendments in one named series may be linked.
    value = re.sub(r"\b\d{5,}\b", " ", value)
    value = re.sub(r"\b\d{1,4}\b", " # ", value)
    value = re.sub(r"[^a-z#]+", " ", value)
    tokens = value.split()
    # document_meta often stores "slug | page title"; collapse exact repeated
    # halves without trying to infer domain-specific legal equivalence.
    compact: List[str] = []
    for token in tokens:
        if not compact or token != compact[-1] or token == "#":
            compact.append(token)
    return " ".join(compact)


def build_families(
    database: Path,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, str]]:
    connection = sqlite3.connect(database)
    try:
        titles = {
            str(doc_id): str(title)
            for doc_id, title in connection.execute(
                "SELECT doc_id, title FROM document_meta"
            )
        }
    finally:
        connection.close()
    exact_groups: DefaultDict[str, List[str]] = defaultdict(list)
    series_groups: DefaultDict[str, List[str]] = defaultdict(list)
    for doc_id, title in titles.items():
        exact_groups[accent_fold(title).strip()].append(doc_id)
        series_groups[family_key(title)].append(doc_id)
    exact = {
        doc_id: sorted(group)
        for group in exact_groups.values() if 1 < len(group) <= 5
        for doc_id in group
    }
    # Large generic families are unsafe at top-5 and intentionally excluded.
    series = {
        doc_id: sorted(group)
        for group in series_groups.values() if 1 < len(group) <= 5
        for doc_id in group
    }
    return exact, series, titles


def add_expansion(
    base_scores: Mapping[str, float],
    graph: Mapping[str, Counter[str]],
    label_count: Mapping[str, int],
    family: Mapping[str, Sequence[str]],
    graph_weight: float,
    family_weight: float,
    source_limit: int = 10,
    minimum_pair_count: int = 1,
) -> List[str]:
    if not base_scores:
        return []
    maximum = max(base_scores.values()) or 1.0
    scores = {doc_id: score / maximum for doc_id, score in base_scores.items()}
    sources = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))[:source_limit]
    for source in sources:
        source_score = scores[source]
        denominator = max(1, int(label_count.get(source, 0)))
        for target, pair_count in graph.get(source, {}).items():
            if pair_count < minimum_pair_count:
                continue
            conditional = pair_count / denominator
            scores[target] = scores.get(target, 0.0) + (
                graph_weight * source_score * conditional
            )
        siblings = family.get(source, ())
        if len(siblings) > 1:
            share = family_weight * source_score / (len(siblings) - 1)
            for target in siblings:
                if target != source:
                    scores[target] = scores.get(target, 0.0) + share
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))


def graph_reserved_slot(
    base_ranking: Sequence[str],
    graph: Mapping[str, Counter[str]],
    label_count: Mapping[str, int],
) -> List[str]:
    """Reserve rank five for the strongest co-label reached from top four."""
    kept = list(dict.fromkeys(base_ranking[:4]))
    votes: DefaultDict[str, float] = defaultdict(float)
    for rank, source in enumerate(kept, 1):
        denominator = max(1, int(label_count.get(source, 0)))
        for target, count in graph.get(source, {}).items():
            if target not in kept:
                votes[target] += count / denominator / rank
    if votes:
        kept.append(min(votes, key=lambda doc_id: (-votes[doc_id], doc_id)))
    for doc_id in base_ranking:
        if len(kept) >= 5:
            break
        if doc_id not in kept:
            kept.append(doc_id)
    return kept


def subset_metrics(
    predictions: Mapping[str, Sequence[str]],
    references: Mapping[str, Sequence[str]],
    multi_only: bool,
) -> Dict[str, float]:
    selected = {
        qid: {"answer": list(predictions[qid])[:5]}
        for qid, answer in references.items()
        if (len(answer) > 1) == multi_only
    }
    refs = {
        qid: {"answer": list(answer)}
        for qid, answer in references.items()
        if qid in selected
    }
    if not selected:
        return {"precision": 0.0, "recall": 0.0, "queries": 0}
    result = official_metrics(selected, refs)
    return {**result, "queries": len(selected)}


def evaluate_fold(
    items: Sequence[Tuple[str, Mapping[str, Any]]],
    fold: int,
    neighbor_limit: int,
    exact_family: Mapping[str, Sequence[str]],
    series_family: Mapping[str, Sequence[str]],
) -> Dict[str, Any]:
    training = [item for index, item in enumerate(items) if index % 5 != fold]
    holdout = [item for index, item in enumerate(items) if index % 5 == fold]
    connection, graph, label_count = fold_index(training)
    tokenizer = VietnameseTokenizer(bigrams=False)
    predictions: DefaultDict[str, Dict[str, List[str]]] = defaultdict(dict)
    references: Dict[str, List[str]] = {}
    try:
        for qid, item in holdout:
            references[qid] = list(dict.fromkeys(map(str, item["answer"])))
            tokens = list(dict.fromkeys(tokenizer.tokenize(str(item["question"]))))
            scores, ranking, ambiguity = supervised_scores(
                connection, fts_expression(tokens), neighbor_limit
            )
            predictions["baseline_sum_alias"][qid] = ranking[:5]
            predictions["graph_reserved_rank5"][qid] = graph_reserved_slot(
                ranking, graph, label_count
            )
            for weight in (0.05, 0.10, 0.20, 0.40, 0.80):
                name = f"graph_w{weight:.2f}"
                predictions[name][qid] = add_expansion(
                    scores, graph, label_count, {}, weight, 0.0
                )[:5]
            for source_limit in (1, 3, 5):
                for weight in (0.05, 0.10, 0.20):
                    name = f"graph_top{source_limit}_w{weight:.2f}"
                    predictions[name][qid] = add_expansion(
                        scores,
                        graph,
                        label_count,
                        {},
                        weight,
                        0.0,
                        source_limit=source_limit,
                    )[:5]
            for minimum_count in (2, 3):
                for weight in (0.10, 0.20, 0.40, 0.80):
                    name = f"graph_count{minimum_count}_w{weight:.2f}"
                    predictions[name][qid] = add_expansion(
                        scores,
                        graph,
                        label_count,
                        {},
                        weight,
                        0.0,
                        source_limit=5,
                        minimum_pair_count=minimum_count,
                    )[:5]
            gated_graph = add_expansion(
                scores,
                graph,
                label_count,
                {},
                0.20,
                0.0,
                source_limit=5,
            )[:5]
            for gate_name, enabled in (
                ("nearest_multi", ambiguity["nearest_multi"] >= 1.0),
                ("top3_any_multi", ambiguity["top3_any_multi"] >= 1.0),
                ("top5_multi40", ambiguity["top5_multi_fraction"] >= 0.40),
                ("weighted_multi10", ambiguity["weighted_multi_fraction"] >= 0.10),
                ("weighted_multi20", ambiguity["weighted_multi_fraction"] >= 0.20),
            ):
                predictions[f"graph_gate_{gate_name}"][qid] = (
                    gated_graph if enabled else ranking[:5]
                )
            for name, families in (
                ("exact_title", exact_family), ("series_title", series_family)
            ):
                for weight in (0.02, 0.05, 0.10, 0.20):
                    variant = f"{name}_w{weight:.2f}"
                    predictions[variant][qid] = add_expansion(
                        scores, {}, {}, families, 0.0, weight
                    )[:5]
            for maximum_size in (2, 3):
                smaller_family = {
                    doc_id: group
                    for doc_id, group in series_family.items()
                    if len(group) <= maximum_size
                }
                for weight in (0.10, 0.20, 0.40):
                    variant = f"series_max{maximum_size}_w{weight:.2f}"
                    predictions[variant][qid] = add_expansion(
                        scores, {}, {}, smaller_family, 0.0, weight
                    )[:5]
            # Test the best-prior graph range together with conservative family
            # expansion; the report decides whether any variant is safe.
            for graph_weight in (0.10, 0.20, 0.40):
                variant = f"graph{graph_weight:.2f}_series0.05"
                predictions[variant][qid] = add_expansion(
                    scores,
                    graph,
                    label_count,
                    series_family,
                    graph_weight,
                    0.05,
                )[:5]
    finally:
        connection.close()
    report: Dict[str, Any] = {}
    for name, values in predictions.items():
        overall = official_metrics(
            {qid: {"answer": docs[:5]} for qid, docs in values.items()},
            {qid: {"answer": answer} for qid, answer in references.items()},
        )
        report[name] = {
            "overall": overall,
            "single_answer": subset_metrics(values, references, False),
            "multi_answer": subset_metrics(values, references, True),
        }
    return report


def average_reports(folds: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    names = next(iter(folds.values())).keys()
    output: Dict[str, Any] = {}
    for name in names:
        output[name] = {}
        for subset in ("overall", "single_answer", "multi_answer"):
            output[name][subset] = {
                metric: sum(report[name][subset][metric] for report in folds.values())
                / len(folds)
                for metric in ("precision", "recall")
            }
            if subset != "overall":
                output[name][subset]["queries_per_fold"] = sum(
                    report[name][subset]["queries"] for report in folds.values()
                ) / len(folds)
    return output


def corpus_stats(
    items: Sequence[Tuple[str, Mapping[str, Any]]],
    exact: Mapping[str, Sequence[str]],
    series: Mapping[str, Sequence[str]],
) -> Dict[str, Any]:
    label_count: Counter[str] = Counter()
    pairs: Counter[Tuple[str, str]] = Counter()
    answer_sizes: Counter[int] = Counter()
    for _, item in items:
        answers = list(dict.fromkeys(map(str, item["answer"])))
        answer_sizes[len(answers)] += 1
        label_count.update(answers)
        for index, first in enumerate(answers):
            for second in answers[index + 1:]:
                pairs[tuple(sorted((first, second)))] += 1
    return {
        "queries": len(items),
        "answer_size_distribution": dict(sorted(answer_sizes.items())),
        "unique_label_documents": len(label_count),
        "co_label_pairs": len(pairs),
        "co_label_pairs_seen_more_than_once": sum(value > 1 for value in pairs.values()),
        "exact_title_clustered_documents": len(exact),
        "series_title_clustered_documents": len(series),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--neighbor-limit", type=int, default=160)
    parser.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    data = json.loads(args.train.read_text(encoding="utf-8"))
    items = [(str(qid), item) for qid, item in data.items()]
    exact_family, series_family, _ = build_families(args.database)
    folds = {
        str(fold): evaluate_fold(
            items,
            fold,
            args.neighbor_limit,
            exact_family,
            series_family,
        )
        for fold in args.folds
    }
    average = average_reports(folds)
    baseline = average["baseline_sum_alias"]["overall"]["recall"]
    best_name = max(
        average,
        key=lambda name: average[name]["overall"]["recall"],
    )
    best_recall = average[best_name]["overall"]["recall"]
    report = {
        "protocol": {
            "split": "deterministic 5-fold by original_position % 5",
            "leakage_control": (
                "question aliases, label priors, and co-label graph are built "
                "from the four training folds only; title families use corpus "
                "metadata without relevance labels"
            ),
            "neighbor_limit": args.neighbor_limit,
            "top_k": 5,
        },
        "corpus_stats": corpus_stats(items, exact_family, series_family),
        "folds": folds,
        "average": average,
        "baseline": "baseline_sum_alias",
        "best_by_recall": best_name,
        "best_recall": best_recall,
        "delta_recall_vs_baseline": best_recall - baseline,
        "promotion_rule": (
            "delta Recall must exceed 0.005 and be positive on all five folds"
        ),
    }
    deltas = [
        folds[str(fold)][best_name]["overall"]["recall"]
        - folds[str(fold)]["baseline_sum_alias"]["overall"]["recall"]
        for fold in args.folds
    ]
    report["best_fold_deltas"] = deltas
    report["safe_to_promote"] = (
        best_recall - baseline > 0.005 and all(delta > 0 for delta in deltas)
    )
    report["recommendation"] = (
        "PROMOTE" if report["safe_to_promote"] else
        "DO_NOT_PROMOTE: gain is below 0.005 and/or not robust across folds"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "best": best_name,
        "baseline_recall": baseline,
        "best_recall": best_recall,
        "delta": best_recall - baseline,
        "fold_deltas": deltas,
        "safe_to_promote": report["safe_to_promote"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
