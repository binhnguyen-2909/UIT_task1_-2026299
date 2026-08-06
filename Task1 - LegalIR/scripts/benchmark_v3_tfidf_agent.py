"""Leak-free five-fold TF-IDF nearest-question benchmark for LegalIR.

This experiment is deliberately dependency-free: it uses sparse Python
dictionaries and inverted postings instead of sklearn.  It never reads the
public questions or the document corpus, and therefore measures only the
incremental value of labelled-question transfer.

Examples
--------
Quick exploratory fold::

    python scripts/benchmark_v3_tfidf_agent.py --train data/raw/public_test/train.json \
        --output reports/v3_tfidf_agent.json --folds 0

Full deterministic five-fold run::

    python scripts/benchmark_v3_tfidf_agent.py --train data/raw/public_test/train.json \
        --output reports/v3_tfidf_agent.json
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import re
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


WORD_RE = re.compile(r"[^\W_]+|\d+", flags=re.UNICODE)


def accent_fold(text: str) -> str:
    """Lowercase and remove Vietnamese accents deterministically."""

    text = text.lower().replace("đ", "d")
    return "".join(
        character
        for character in unicodedata.normalize("NFD", text)
        if unicodedata.category(character) != "Mn"
    )


def tokens(text: str, *, folded: bool = False) -> List[str]:
    if folded:
        text = accent_fold(text)
    else:
        text = unicodedata.normalize("NFC", text.lower())
    return WORD_RE.findall(text)


def word12_features(text: str, *, folded: bool = False) -> Counter[str]:
    words = tokens(text, folded=folded)
    output: Counter[str] = Counter(f"w:{word}" for word in words)
    output.update(
        f"b:{left}\x1f{right}" for left, right in zip(words, words[1:])
    )
    return output


def char_features(text: str, minimum: int = 3, maximum: int = 4) -> Counter[str]:
    normalized = " ".join(tokens(text, folded=True))
    output: Counter[str] = Counter()
    for size in range(minimum, maximum + 1):
        output.update(
            f"c{size}:{normalized[index:index + size]}"
            for index in range(max(0, len(normalized) - size + 1))
        )
    return output


@dataclass(frozen=True)
class Mode:
    name: str
    use_word: bool
    folded_word: bool = False
    use_char: bool = False
    char_weight: float = 0.35
    max_df_ratio: float = 0.35


MODES: Dict[str, Mode] = {
    "word12": Mode("word12", use_word=True, max_df_ratio=0.65),
    "word12_fold": Mode(
        "word12_fold", use_word=True, folded_word=True, max_df_ratio=0.65
    ),
    "char34_fold": Mode(
        "char34_fold", use_word=False, use_char=True, max_df_ratio=0.25
    ),
    "hybrid_w12_c34": Mode(
        "hybrid_w12_c34",
        use_word=True,
        use_char=True,
        char_weight=0.35,
        max_df_ratio=0.35,
    ),
}


def extract_features(text: str, mode: Mode) -> Counter[str]:
    output: Counter[str] = Counter()
    if mode.use_word:
        output.update(word12_features(text, folded=mode.folded_word))
    if mode.use_char:
        output.update(char_features(text))
    return output


class SparseTfidfQuestions:
    """Cosine TF-IDF index over labelled training questions."""

    def __init__(
        self,
        all_features: Sequence[Counter[str]],
        training_indices: Sequence[int],
        mode: Mode,
    ) -> None:
        self.mode = mode
        self.training_indices = list(training_indices)
        document_frequency: Counter[str] = Counter()
        for index in self.training_indices:
            document_frequency.update(all_features[index].keys())

        count = len(self.training_indices)
        maximum_df = max(2, int(mode.max_df_ratio * count))
        self.idf = {
            feature: math.log((count + 1.0) / (frequency + 1.0)) + 1.0
            for feature, frequency in document_frequency.items()
            if 2 <= frequency <= maximum_df
        }
        postings: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        for index in self.training_indices:
            vector = self._weighted(all_features[index])
            norm = math.sqrt(sum(value * value for value in vector.values()))
            if norm <= 0.0:
                continue
            inverse_norm = 1.0 / norm
            for feature, value in vector.items():
                postings[feature].append((index, value * inverse_norm))
        self.postings = dict(postings)

    def _weighted(self, features: Mapping[str, int]) -> Dict[str, float]:
        weighted: Dict[str, float] = {}
        for feature, frequency in features.items():
            idf = self.idf.get(feature)
            if idf is None or frequency <= 0:
                continue
            family_weight = (
                self.mode.char_weight if feature.startswith("c") else 1.0
            )
            weighted[feature] = (
                (1.0 + math.log(float(frequency))) * idf * family_weight
            )
        return weighted

    def neighbors(
        self,
        query_features: Mapping[str, int],
        limit: int,
    ) -> List[Tuple[int, float]]:
        query = self._weighted(query_features)
        norm = math.sqrt(sum(value * value for value in query.values()))
        if norm <= 0.0:
            return []
        inverse_norm = 1.0 / norm
        scores: Dict[int, float] = defaultdict(float)
        for feature, query_value in query.items():
            normalized_query = query_value * inverse_norm
            for index, training_value in self.postings.get(feature, ()):
                scores[index] += normalized_query * training_value
        return heapq.nlargest(
            limit,
            scores.items(),
            key=lambda item: (item[1], -item[0]),
        )


def document_rankings(
    neighbors: Sequence[Tuple[int, float]],
    answers: Sequence[Sequence[str]],
    popular_docs: Sequence[str],
    candidate_limit: int,
) -> Dict[str, List[str]]:
    maximum: Dict[str, float] = {}
    top_three: Dict[str, List[float]] = defaultdict(list)
    similarity_rrf: Dict[str, float] = defaultdict(float)
    for rank, (index, similarity) in enumerate(neighbors, 1):
        for doc_id in answers[index]:
            maximum[doc_id] = max(maximum.get(doc_id, 0.0), similarity)
            heap = top_three[doc_id]
            if len(heap) < 3:
                heapq.heappush(heap, similarity)
            elif similarity > heap[0]:
                heapq.heapreplace(heap, similarity)
            similarity_rrf[doc_id] += similarity / (20.0 + rank)

    rankings = {
        "max": sorted(maximum, key=lambda doc: (-maximum[doc], doc)),
        "top3_sum": sorted(
            top_three, key=lambda doc: (-sum(top_three[doc]), doc)
        ),
        "sim_rrf": sorted(
            similarity_rrf, key=lambda doc: (-similarity_rrf[doc], doc)
        ),
    }
    for ranking in rankings.values():
        seen = set(ranking)
        ranking.extend(doc for doc in popular_docs if doc not in seen)
        del ranking[candidate_limit:]
    return rankings


def rrf(rankings: Iterable[Tuple[float, Sequence[str]]], k: int = 20) -> List[str]:
    scores: Dict[str, float] = defaultdict(float)
    for weight, ranking in rankings:
        for rank, doc_id in enumerate(ranking, 1):
            scores[doc_id] += weight / (k + rank)
    return sorted(scores, key=lambda doc: (-scores[doc], doc))


def official_metrics(
    predictions: Mapping[str, Sequence[str]],
    references: Mapping[str, Sequence[str]],
) -> Dict[str, float]:
    recall = 0.0
    precision = 0.0
    for qid, relevant_docs in references.items():
        relevant = set(relevant_docs)
        predicted = list(dict.fromkeys(predictions.get(qid, ())))[:5]
        correct = len(relevant.intersection(predicted))
        recall += correct / len(relevant) if relevant else 0.0
        precision += correct / len(predicted) if predicted else 0.0
    count = len(references)
    return {
        "precision": precision / count if count else 0.0,
        "recall": recall / count if count else 0.0,
    }


def candidate_recall(
    predictions: Mapping[str, Sequence[str]],
    references: Mapping[str, Sequence[str]],
    cutoff: int,
) -> float:
    total = 0.0
    for qid, relevant_docs in references.items():
        relevant = set(relevant_docs)
        predicted = set(predictions.get(qid, ())[:cutoff])
        total += len(relevant.intersection(predicted)) / len(relevant)
    return total / len(references)


def availability_recall(
    holdout_indices: Sequence[int],
    answers: Sequence[Sequence[str]],
    training_indices: Sequence[int],
) -> float:
    available = {doc for index in training_indices for doc in answers[index]}
    values = []
    for index in holdout_indices:
        relevant = set(answers[index])
        values.append(len(relevant.intersection(available)) / len(relevant))
    return sum(values) / len(values)


def benchmark_fold(
    items: Sequence[Tuple[str, Dict[str, Any]]],
    feature_banks: Mapping[str, Sequence[Counter[str]]],
    modes: Sequence[Mode],
    fold: int,
    neighbor_limit: int,
    candidate_limit: int,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, List[str]]]]:
    training_indices = [i for i in range(len(items)) if i % 5 != fold]
    holdout_indices = [i for i in range(len(items)) if i % 5 == fold]
    answers = [tuple(map(str, item["answer"])) for _, item in items]
    popularity: Counter[str] = Counter(
        doc for index in training_indices for doc in answers[index]
    )
    popular_docs = sorted(popularity, key=lambda doc: (-popularity[doc], doc))
    references = {items[i][0]: answers[i] for i in holdout_indices}
    predictions: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    mode_timings: Dict[str, float] = {}

    for mode in modes:
        started = time.perf_counter()
        bank = feature_banks[mode.name]
        index = SparseTfidfQuestions(bank, training_indices, mode)
        for position, item_index in enumerate(holdout_indices, 1):
            qid = items[item_index][0]
            nearest = index.neighbors(bank[item_index], neighbor_limit)
            rankings = document_rankings(
                nearest, answers, popular_docs, candidate_limit
            )
            for strategy, ranking in rankings.items():
                predictions[f"{mode.name}:{strategy}"][qid] = ranking
            if position % 350 == 0:
                print(
                    f"fold={fold} mode={mode.name} "
                    f"questions={position}/{len(holdout_indices)}",
                    flush=True,
                )
        mode_timings[mode.name] = time.perf_counter() - started

    # Cross-family RRF is the intended production-facing signal.  Use max and
    # top3 separately so the benchmark can expose popularity overfitting.
    if len(modes) > 1:
        for qid in references:
            predictions["fusion:max_equal"][qid] = rrf(
                (1.0, predictions[f"{mode.name}:max"][qid]) for mode in modes
            )[:candidate_limit]
            predictions["fusion:top3_equal"][qid] = rrf(
                (1.0, predictions[f"{mode.name}:top3_sum"][qid])
                for mode in modes
            )[:candidate_limit]
            predictions["fusion:sim_rrf_equal"][qid] = rrf(
                (1.0, predictions[f"{mode.name}:sim_rrf"][qid])
                for mode in modes
            )[:candidate_limit]

    report: Dict[str, Any] = {
        "question_count": len(holdout_indices),
        "label_availability_recall_ceiling": availability_recall(
            holdout_indices, answers, training_indices
        ),
        "timing_seconds": mode_timings,
        "metrics": {},
    }
    for name, values in sorted(predictions.items()):
        report["metrics"][name] = {
            **official_metrics(values, references),
            "candidate_recall_at_20": candidate_recall(values, references, 20),
            "candidate_recall_at_40": candidate_recall(values, references, 40),
        }
    return report, predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=sorted(MODES),
        default=["word12", "word12_fold", "char34_fold", "hybrid_w12_c34"],
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--neighbor-limit", type=int, default=160)
    parser.add_argument("--candidate-limit", type=int, default=40)
    args = parser.parse_args()
    if any(fold not in range(5) for fold in args.folds):
        parser.error("folds must be between 0 and 4")

    raw = json.loads(args.train.read_text(encoding="utf-8"))
    items = [(str(qid), item) for qid, item in raw.items()]
    modes = [MODES[name] for name in args.modes]
    feature_started = time.perf_counter()
    feature_banks = {
        mode.name: [extract_features(item["question"], mode) for _, item in items]
        for mode in modes
    }
    feature_seconds = time.perf_counter() - feature_started

    folds: Dict[str, Any] = {}
    for fold in args.folds:
        fold_report, _ = benchmark_fold(
            items,
            feature_banks,
            modes,
            fold,
            args.neighbor_limit,
            args.candidate_limit,
        )
        folds[str(fold)] = fold_report

    names = sorted(next(iter(folds.values()))["metrics"])
    average = {
        name: {
            metric: sum(fold["metrics"][name][metric] for fold in folds.values())
            / len(folds)
            for metric in (
                "precision",
                "recall",
                "candidate_recall_at_20",
                "candidate_recall_at_40",
            )
        }
        for name in names
    }
    report = {
        "experiment": "dependency-free sparse cosine TF-IDF nearest labelled question",
        "split": "leak-free deterministic five-fold by original_position % 5",
        "data": {"questions": len(items), "source": str(args.train)},
        "configuration": {
            "modes": args.modes,
            "neighbor_limit": args.neighbor_limit,
            "candidate_limit": args.candidate_limit,
            "folds": args.folds,
            "feature_extraction_seconds": feature_seconds,
        },
        "folds": folds,
        "average": average,
        "best_by_recall_at_5": max(
            average,
            key=lambda name: (average[name]["recall"], average[name]["precision"]),
        ),
        "best_by_candidate_recall_at_40": max(
            average,
            key=lambda name: average[name]["candidate_recall_at_40"],
        ),
        "selection_guidance": (
            "Treat supervised TF-IDF as an optional candidate source only. "
            "Do not promote it into the final top five unless a separate "
            "fusion holdout shows a recall gain over the passage baseline."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"average": average, "best": report["best_by_recall_at_5"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
