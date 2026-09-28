"""Leak-aware tiny supervised fusion/LTR for the v11 LegalIR pipeline.

The existing fit audit is deliberately not used for supervision because its
index contains exact question memory.  Instead, this script splits the clean
1,000-question holdout by connected reference-document groups, selects a tiny
ranker and a conservative fusion policy with grouped out-of-fold predictions,
then opens a document-disjoint final test once.  If the locked policy clears
the promotion gate, it is refit on all clean labelled questions and applied to
the public candidate audit.

No public labels, v57 predictions, or external legal data are consumed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sqlite3
import statistics
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .evaluate import official_metrics
    from .v11_exact_heading_rescue import normalize
except ImportError:
    from evaluate import official_metrics
    from v11_exact_heading_rescue import normalize


NUMBER_RE = re.compile(r"\d+")
QUESTION_TYPES = (
    "bao lau", "bao nhieu", "la gi", "gom nhung", "dieu kien",
    "ho so", "muc phat", "thoi han", "co duoc", "do ai",
    "nhu the nao", "truong hop nao",
)


@dataclass
class AuditSplit:
    name: str
    qids: list[str]
    questions: list[str]
    candidates: list[list[str]]
    baseline: list[list[str]]
    neural_scores: list[dict[str, float]]
    result_rows: list[dict[str, dict[str, Any]]]
    references: list[set[str]] | None = None
    stage1_path: Path | None = None


@dataclass
class Document:
    text: str
    title: str
    tokens: frozenset[str]
    title_tokens: frozenset[str]
    numbers: frozenset[str]
    token_count: int
    prior: int


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left = self.find(left)
        right = self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_audit_split(
    name: str,
    stage1_path: Path,
    stage2_path: Path,
    reference_path: Path | None,
) -> AuditSplit:
    stage1 = read_json(stage1_path)
    stage2 = read_json(stage2_path)
    queries = stage1.get("queries")
    candidate_rows = stage1.get("stage1_candidates")
    score_rows = stage2.get("queries")
    if not isinstance(queries, list) or not isinstance(candidate_rows, list):
        raise ValueError(f"{name}: invalid Stage-1 audit")
    if not isinstance(score_rows, list) or len(queries) != len(score_rows):
        raise ValueError(f"{name}: Stage-1/Stage-2 row counts differ")
    by_qid = {str(row["question_id"]): row for row in score_rows}
    raw_references = read_json(reference_path) if reference_path else None
    qids: list[str] = []
    questions: list[str] = []
    candidates: list[list[str]] = []
    baseline: list[list[str]] = []
    neural_scores: list[dict[str, float]] = []
    result_rows: list[dict[str, dict[str, Any]]] = []
    references: list[set[str]] | None = [] if raw_references is not None else None
    for query, raw_candidates in zip(queries, candidate_rows):
        qid = str(query["query_id"])
        score_row = by_qid.get(qid)
        if score_row is None:
            raise ValueError(f"{name}: Stage-2 scores missing {qid}")
        unique_candidates = list(dict.fromkeys(str(value) for value in raw_candidates))
        answers = list(dict.fromkeys(str(value) for value in score_row["answers"]))
        if len(answers) != 5:
            raise ValueError(f"{name}/{qid}: baseline must have five answers")
        qids.append(qid)
        questions.append(str(query.get("query", query.get("question", ""))))
        candidates.append(unique_candidates)
        baseline.append(answers)
        neural_scores.append({str(key): float(value) for key, value in score_row["neural_scores"].items()})
        result_rows.append({str(row["doc_id"]): row for row in query.get("results", [])})
        if references is not None:
            references.append(set(str(value) for value in raw_references[qid]["answer"]))
    return AuditSplit(
        name=name,
        qids=qids,
        questions=questions,
        candidates=candidates,
        baseline=baseline,
        neural_scores=neural_scores,
        result_rows=result_rows,
        references=references,
        stage1_path=stage1_path.resolve(),
    )


def load_priors(index_path: Path) -> dict[str, int]:
    connection = sqlite3.connect(index_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        return {str(doc_id): int(count) for doc_id, count in connection.execute(
            "SELECT doc_id, label_count FROM document_prior"
        )}
    finally:
        connection.close()


def load_documents(
    contexts_zip: Path,
    needed_ids: set[str],
    priors: Mapping[str, int],
) -> tuple[dict[str, Document], Counter[str]]:
    documents: dict[str, Document] = {}
    document_frequency: Counter[str] = Counter()
    with zipfile.ZipFile(contexts_zip) as archive:
        infos = [info for info in archive.infolist() if info.filename.endswith(".json")]
        for position, info in enumerate(infos, 1):
            item = json.loads(archive.read(info))
            doc_id = str(item["id"])
            if doc_id not in needed_ids:
                continue
            body = normalize(str(item.get("passage", "")))
            title = normalize(str(item.get("name", "")))
            tokens = frozenset(body.split())
            title_tokens = frozenset(title.split())
            document_frequency.update(tokens)
            documents[doc_id] = Document(
                text=body,
                title=title,
                tokens=tokens,
                title_tokens=title_tokens,
                numbers=frozenset(NUMBER_RE.findall(body)),
                token_count=len(body.split()),
                prior=int(priors.get(doc_id, 0)),
            )
            if position % 1000 == 0:
                print(f"contexts: {position}/{len(infos)} ({len(documents)} needed)", flush=True)
    missing = needed_ids - set(documents)
    if missing:
        print(f"warning: {len(missing)} candidate document IDs absent from contexts", flush=True)
    return documents, document_frequency


def longest_match(
    tokens: Sequence[str],
    document: str,
    maximum: int = 12,
    minimum: int = 10,
) -> int:
    """Return the longest conservative exact-heading match.

    v11d's accepted rescue threshold is ten tokens.  Searching two-through-nine
    token windows adds millions of full-document scans without affecting that
    policy, so the LTR feature intentionally measures only the actionable
    ten-through-twelve range.
    """

    for size in range(min(maximum, len(tokens)), minimum - 1, -1):
        for start in range(len(tokens) - size + 1):
            if " ".join(tokens[start:start + size]) in document:
                return size
    return 0


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def build_features(
    split: AuditSplit,
    documents: Mapping[str, Document],
    document_frequency: Mapping[str, int],
    candidate_limit: int,
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    feature_names = [
        "rank", "inv_rank", "log_rank", "rank_fraction",
        "top5", "top10", "top20", "top35",
        "baseline_present", "baseline_inv_rank",
        "neural_present", "neural_signed_log", "neural_inv_rank", "neural_rank_fraction",
        "stage1_score_known", "stage1_score", "stage1_dense", "stage1_chunk_phrase", "stage1_citation",
        "query_log_tokens", "document_log_tokens", "title_log_tokens",
        "query_recall_document", "document_precision", "jaccard_document", "idf_recall_document",
        "query_recall_title", "title_precision", "jaccard_title", "idf_recall_title",
        "longest_phrase_tokens", "longest_phrase_fraction", "query_pair_token_coverage",
        "number_recall", "number_precision", "number_exact", "query_has_number",
        "exact_query_in_document", "document_prior_disabled",
        *(f"question_type__{value.replace(' ', '_')}" for value in QUESTION_TYPES),
    ]
    rows: list[list[float]] = []
    labels: list[int] | None = [] if split.references is not None else None
    corpus_size = max(len(documents), 1)
    empty = Document("", "", frozenset(), frozenset(), frozenset(), 0, 0)
    for query_index, (question, candidates) in enumerate(zip(split.questions, split.candidates)):
        qnorm = normalize(question)
        qtokens_list = qnorm.split()
        qtokens = frozenset(qtokens_list)
        qnumbers = frozenset(NUMBER_RE.findall(qnorm))
        token_pairs = [
            (qtokens_list[i], qtokens_list[i + 1])
            for i in range(max(len(qtokens_list) - 1, 0))
        ]
        idf = {
            token: math.log((corpus_size + 1.0) / (float(document_frequency.get(token, 0)) + 1.0)) + 1.0
            for token in qtokens
        }
        total_idf = sum(idf.values())
        base_rank = {doc_id: rank for rank, doc_id in enumerate(split.baseline[query_index], 1)}
        neural = split.neural_scores[query_index]
        neural_rank = {
            doc_id: rank
            for rank, (doc_id, _) in enumerate(sorted(neural.items(), key=lambda item: (-item[1], item[0])), 1)
        }
        result_map = split.result_rows[query_index]
        selected = list(candidates[:candidate_limit])
        if len(selected) < candidate_limit:
            selected.extend([""] * (candidate_limit - len(selected)))
        for rank, doc_id in enumerate(selected, 1):
            doc = documents.get(doc_id, empty)
            overlap = qtokens & doc.tokens
            title_overlap = qtokens & doc.title_tokens
            union = qtokens | doc.tokens
            title_union = qtokens | doc.title_tokens
            phrase = longest_match(qtokens_list, doc.text)
            pair_hits = sum(left in doc.tokens and right in doc.tokens for left, right in token_pairs)
            number_overlap = qnumbers & doc.numbers
            raw_neural = neural.get(doc_id)
            neural_value = float(raw_neural) if raw_neural is not None else 0.0
            neural_log = math.copysign(math.log1p(abs(neural_value)), neural_value)
            result = result_map.get(doc_id, {})
            method = str(result.get("method", ""))
            raw_stage1_score = result.get("score")
            stage1_score = math.copysign(math.log1p(abs(float(raw_stage1_score))), float(raw_stage1_score)) if raw_stage1_score is not None else 0.0
            weighted_overlap = sum(idf[token] for token in overlap)
            weighted_title_overlap = sum(idf[token] for token in title_overlap)
            values = [
                float(rank), 1.0 / rank, math.log1p(rank), rank / candidate_limit,
                float(rank <= 5), float(rank <= 10), float(rank <= 20), float(rank <= 35),
                float(doc_id in base_rank), 1.0 / base_rank[doc_id] if doc_id in base_rank else 0.0,
                float(raw_neural is not None), neural_log,
                1.0 / neural_rank[doc_id] if doc_id in neural_rank else 0.0,
                neural_rank[doc_id] / max(len(neural_rank), 1) if doc_id in neural_rank else 1.25,
                float(raw_stage1_score is not None), stage1_score,
                float("dense" in method), float("chunk_phrase" in method), float("citation" in method),
                math.log1p(len(qtokens_list)), math.log1p(doc.token_count), math.log1p(len(doc.title.split())),
                ratio(len(overlap), len(qtokens)), ratio(len(overlap), len(doc.tokens)), ratio(len(overlap), len(union)), ratio(weighted_overlap, total_idf),
                ratio(len(title_overlap), len(qtokens)), ratio(len(title_overlap), len(doc.title_tokens)), ratio(len(title_overlap), len(title_union)), ratio(weighted_title_overlap, total_idf),
                float(phrase), ratio(phrase, len(qtokens_list)), ratio(pair_hits, len(token_pairs)),
                ratio(len(number_overlap), len(qnumbers)), ratio(len(number_overlap), len(doc.numbers)),
                float(bool(qnumbers) and qnumbers == doc.numbers), float(bool(qnumbers)),
                # A prior built from labelled questions is unsafe for cross-fit
                # LTR: the complement query's own answer can be represented in
                # that prior even when exact question memory is absent.  Keep a
                # fixed zero column so old feature indices remain auditable,
                # but never expose the supervised count to the ranker.
                float(bool(qnorm) and qnorm in doc.text), 0.0,
                *(float(value in qnorm) for value in QUESTION_TYPES),
            ]
            rows.append(values)
            if labels is not None:
                labels.append(int(doc_id in split.references[query_index]))
        if (query_index + 1) % 250 == 0:
            print(f"features/{split.name}: {query_index + 1}/{len(split.qids)}", flush=True)
    matrix = np.asarray(rows, dtype=np.float32)
    target = np.asarray(labels, dtype=np.int8) if labels is not None else None
    return matrix, target, feature_names


def load_source_stage1_evidence(
    split: AuditSplit,
) -> tuple[list[list[str]], list[dict[str, dict[str, Any]]], str]:
    """Load the pre-PRF Stage-1 rows without exposing any labels."""

    if split.stage1_path is None:
        return split.candidates, split.result_rows, "embedded-current-stage1"
    current = read_json(split.stage1_path)
    raw_source = current.get("source_stage1_audit")
    if raw_source:
        # Audit artifacts were originally produced on Windows.  ``Path`` on
        # Linux intentionally does not regard ``D:\\...`` as absolute, which
        # otherwise produces the invalid joined path
        # ``.../v10_9_prf/D:\\projects\\...``.  Resolve an embedded Task1
        # relative suffix against the checked-out task root, while preserving
        # ordinary absolute and relative artifact paths.
        raw_value = str(raw_source)
        source_path = Path(raw_value)
        if not source_path.is_absolute():
            portable = raw_value.replace("\\", "/")
            marker = "Task1 - LegalIR/"
            if marker in portable:
                suffix = portable.split(marker, 1)[1]
                task_root = next(
                    (
                        parent for parent in split.stage1_path.parents
                        if parent.name == "Task1 - LegalIR"
                    ),
                    None,
                )
                if task_root is None:
                    raise ValueError(
                        f"Cannot resolve Task1-relative Stage-1 source: {raw_value}"
                    )
                source_path = task_root / suffix
            else:
                source_path = split.stage1_path.parent / source_path
    else:
        source_path = split.stage1_path
    source = read_json(source_path)
    queries = source.get("queries")
    candidates = source.get("stage1_candidates")
    if not isinstance(queries, list) or not isinstance(candidates, list):
        raise ValueError(f"{split.name}: source Stage-1 audit is invalid: {source_path}")
    by_qid: dict[str, tuple[list[str], dict[str, dict[str, Any]]]] = {}
    for query, ranking in zip(queries, candidates, strict=True):
        qid = str(query["query_id"])
        by_qid[qid] = (
            [str(value) for value in ranking],
            {str(row["doc_id"]): row for row in query.get("results", [])},
        )
    source_candidates: list[list[str]] = []
    source_results: list[dict[str, dict[str, Any]]] = []
    for qid in split.qids:
        if qid not in by_qid:
            raise ValueError(f"{split.name}: source Stage-1 audit lacks {qid}")
        ranking, rows = by_qid[qid]
        source_candidates.append(ranking)
        source_results.append(rows)
    return source_candidates, source_results, str(source_path.resolve())


def build_query_normalized_features(
    split: AuditSplit,
    documents: Mapping[str, Document],
    document_frequency: Mapping[str, int],
    candidate_limit: int,
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    """Extend v1 with query-relative neural and pre-PRF provenance features.

    Cross-encoder probabilities span several orders of magnitude and are not
    calibrated across queries.  V1 used ``log1p(score)``, which compresses that
    evidence.  This feature set keeps v1 intact and appends log-probability,
    per-query normalization/margins, multi-ranker agreement, and the original
    Stage-1 rank before corpus-only PRF.  Every added signal is inference-time
    evidence; no labels or document priors are read here.
    """

    base_matrix, target, feature_names = build_features(
        split, documents, document_frequency, candidate_limit
    )
    source_candidates, source_results, _ = load_source_stage1_evidence(split)
    extra_names = [
        "source_present", "source_rank", "source_inv_rank", "source_rank_fraction",
        "source_top5", "source_top10", "source_top20", "source_top35",
        "source_score_known", "source_signed_log_score",
        "source_bm25_inv_rank", "source_dense_inv_rank",
        "source_method_dense", "source_method_chunk_phrase", "source_method_citation",
        "prf_new_candidate", "prf_rank_gain",
        "neural_log_probability", "neural_query_zscore", "neural_query_minmax",
        "neural_margin_from_best_log", "neural_margin_from_fifth_log",
        "neural_top1", "neural_top3", "neural_top5",
        "final_neural_rank_gap", "source_neural_rank_gap", "final_source_rank_gap",
        "rank_consensus_rr", "top5_vote_fraction",
        "longest_phrase_ge5", "longest_phrase_ge7", "longest_phrase_ge10",
        "longest_title_phrase_tokens", "longest_title_phrase_fraction",
        "exact_query_in_title", "query_document_log_length_ratio",
        "body_title_recall_gap",
    ]
    columns = {name: index for index, name in enumerate(feature_names)}
    empty = Document("", "", frozenset(), frozenset(), frozenset(), 0, 0)
    extra_rows: list[list[float]] = []
    epsilon = 1e-12
    for query_index, candidates in enumerate(split.candidates):
        selected = list(candidates[:candidate_limit])
        if len(selected) < candidate_limit:
            selected.extend([""] * (candidate_limit - len(selected)))
        source_ranking = source_candidates[query_index]
        source_rank = {doc_id: rank for rank, doc_id in enumerate(source_ranking, 1)}
        source_row_map = source_results[query_index]
        baseline_rank = {
            doc_id: rank for rank, doc_id in enumerate(split.baseline[query_index], 1)
        }
        neural = split.neural_scores[query_index]
        neural_rank = {
            doc_id: rank
            for rank, (doc_id, _) in enumerate(
                sorted(neural.items(), key=lambda item: (-item[1], item[0])), 1
            )
        }
        log_scores = {
            doc_id: math.log(max(float(value), epsilon))
            for doc_id, value in neural.items()
            if math.isfinite(float(value))
        }
        ordered_logs = sorted(log_scores.values(), reverse=True)
        if ordered_logs:
            log_mean = statistics.fmean(ordered_logs)
            log_std = statistics.pstdev(ordered_logs) or 1.0
            log_min = ordered_logs[-1]
            log_max = ordered_logs[0]
            log_span = log_max - log_min or 1.0
            fifth_log = ordered_logs[min(4, len(ordered_logs) - 1)]
            missing_log = log_min - 2.0
        else:
            log_mean, log_std, log_min, log_max = 0.0, 1.0, 0.0, 0.0
            log_span, fifth_log, missing_log = 1.0, 0.0, -2.0
        qtokens_list = normalize(split.questions[query_index]).split()
        for local_index, doc_id in enumerate(selected):
            final_rank = local_index + 1
            source_position = source_rank.get(doc_id)
            neural_position = neural_rank.get(doc_id)
            baseline_position = baseline_rank.get(doc_id)
            source_row = source_row_map.get(doc_id, {})
            raw_source_score = source_row.get("score")
            source_score = float(raw_source_score) if raw_source_score is not None else 0.0
            signed_source_score = math.copysign(
                math.log1p(abs(source_score)), source_score
            )
            bm25_rank = source_row.get("bm25_rank")
            dense_rank = source_row.get("dense_rank")
            method = str(source_row.get("method", ""))
            neural_log = log_scores.get(doc_id, missing_log)
            neural_z = (neural_log - log_mean) / log_std
            neural_minmax = (neural_log - log_min) / log_span
            neural_minmax = min(max(neural_minmax, 0.0), 1.0)
            final_inv = 1.0 / final_rank
            source_inv = 1.0 / source_position if source_position else 0.0
            neural_inv = 1.0 / neural_position if neural_position else 0.0
            baseline_inv = 1.0 / baseline_position if baseline_position else 0.0
            top5_votes = sum(
                position is not None and position <= 5
                for position in (final_rank, source_position, neural_position, baseline_position)
            )
            matrix_row = base_matrix[query_index * candidate_limit + local_index]
            phrase = matrix_row[columns["longest_phrase_tokens"]]
            document = documents.get(doc_id, empty)
            title_phrase = longest_match(qtokens_list, document.title)
            source_fraction = (
                source_position / max(len(source_ranking), 1)
                if source_position else 1.25
            )
            extra_rows.append([
                float(source_position is not None), float(source_position or 0),
                source_inv, source_fraction,
                float(source_position is not None and source_position <= 5),
                float(source_position is not None and source_position <= 10),
                float(source_position is not None and source_position <= 20),
                float(source_position is not None and source_position <= 35),
                float(raw_source_score is not None), signed_source_score,
                1.0 / float(bm25_rank) if bm25_rank else 0.0,
                1.0 / float(dense_rank) if dense_rank else 0.0,
                float("dense" in method), float("chunk_phrase" in method),
                float("citation" in method),
                float(source_position is None),
                (source_position - final_rank) / candidate_limit if source_position else 0.0,
                neural_log, neural_z, neural_minmax,
                neural_log - log_max, neural_log - fifth_log,
                float(neural_position == 1),
                float(neural_position is not None and neural_position <= 3),
                float(neural_position is not None and neural_position <= 5),
                abs(final_rank - neural_position) / candidate_limit if neural_position else 1.25,
                abs(source_position - neural_position) / candidate_limit
                if source_position and neural_position else 1.25,
                abs(final_rank - source_position) / candidate_limit if source_position else 1.25,
                final_inv + source_inv + neural_inv + baseline_inv,
                top5_votes / 4.0,
                float(phrase >= 5), float(phrase >= 7), float(phrase >= 10),
                float(title_phrase), ratio(title_phrase, len(qtokens_list)),
                float(bool(qtokens_list) and " ".join(qtokens_list) in document.title),
                math.log((len(qtokens_list) + 1.0) / (document.token_count + 1.0)),
                matrix_row[columns["query_recall_document"]]
                - matrix_row[columns["query_recall_title"]],
            ])
    extra_matrix = np.asarray(extra_rows, dtype=np.float32)
    if extra_matrix.shape[0] != base_matrix.shape[0]:
        raise RuntimeError("query-normalized feature row count mismatch")
    return np.hstack((base_matrix, extra_matrix)), target, feature_names + extra_names


def build_leak_groups(split: AuditSplit, similarity_threshold: float = 0.75) -> np.ndarray:
    if split.references is None:
        raise ValueError("Leak groups require references")
    union = UnionFind(len(split.qids))
    by_document: dict[str, list[int]] = defaultdict(list)
    for index, answers in enumerate(split.references):
        for doc_id in answers:
            by_document[doc_id].append(index)
    for indices in by_document.values():
        for index in indices[1:]:
            union.union(indices[0], index)
    vectors = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True
    ).fit_transform(split.questions)
    similarities = (vectors @ vectors.T).tocsr()
    similarities.setdiag(0)
    for row in range(similarities.shape[0]):
        start, stop = similarities.indptr[row], similarities.indptr[row + 1]
        for column, value in zip(similarities.indices[start:stop], similarities.data[start:stop]):
            if column > row and float(value) >= similarity_threshold:
                union.union(row, int(column))
    roots = [union.find(index) for index in range(len(split.qids))]
    remap = {root: group for group, root in enumerate(sorted(set(roots)))}
    return np.asarray([remap[root] for root in roots], dtype=np.int32)


def row_indices(question_indices: Sequence[int], candidate_limit: int) -> np.ndarray:
    return np.concatenate([
        np.arange(index * candidate_limit, (index + 1) * candidate_limit, dtype=np.int64)
        for index in question_indices
    ])


def point_weights(target: np.ndarray, questions: Sequence[int], candidate_limit: int) -> np.ndarray:
    weights = np.zeros(len(questions) * candidate_limit, dtype=np.float32)
    for local, question in enumerate(questions):
        values = target[question * candidate_limit:(question + 1) * candidate_limit]
        positives = int(values.sum())
        negatives = candidate_limit - positives
        if not positives:
            continue
        block = weights[local * candidate_limit:(local + 1) * candidate_limit]
        block[values == 1] = 0.5 / positives
        block[values == 0] = 0.5 / max(negatives, 1)
    return weights


def pairwise_rows(
    matrix: np.ndarray,
    target: np.ndarray,
    questions: Sequence[int],
    candidate_limit: int,
    hard_negatives: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    differences: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    for question in questions:
        start = question * candidate_limit
        values = target[start:start + candidate_limit]
        positives = np.flatnonzero(values == 1)
        negatives = np.flatnonzero(values == 0)[:hard_negatives]
        if not len(positives) or not len(negatives):
            continue
        weight = 0.5 / (len(positives) * len(negatives))
        for positive in positives:
            for negative in negatives:
                difference = matrix[start + positive] - matrix[start + negative]
                differences.extend((difference, -difference))
                labels.extend((1, 0))
                weights.extend((weight, weight))
    return (
        np.asarray(differences, dtype=np.float32),
        np.asarray(labels, dtype=np.int8),
        np.asarray(weights, dtype=np.float32),
    )


class TorchRanker:
    def __init__(self, network: Any, mean: np.ndarray, scale: np.ndarray, device: str):
        self.network = network
        self.mean = mean
        self.scale = scale
        self.device = device

    def decision_function(self, matrix: np.ndarray) -> np.ndarray:
        import torch

        self.network.eval()
        output: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(matrix), 8192):
                values = (matrix[start:start + 8192] - self.mean) / self.scale
                tensor = torch.from_numpy(values.astype(np.float32, copy=False)).to(self.device)
                output.append(self.network(tensor).squeeze(1).cpu().numpy())
        return np.concatenate(output)


def train_torch_ranker(
    matrix: np.ndarray,
    target: np.ndarray,
    questions: Sequence[int],
    candidate_limit: int,
    config: Mapping[str, Any],
) -> TorchRanker:
    import torch
    from torch import nn

    seed = int(config.get("seed", 20260825))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    indices = row_indices(questions, candidate_limit)
    training = matrix[indices]
    mean = training.mean(axis=0, keepdims=True).astype(np.float32)
    scale = training.std(axis=0, keepdims=True).astype(np.float32)
    scale[scale < 1e-5] = 1.0
    normalized = ((matrix - mean) / scale).astype(np.float32)
    positive_rows: list[int] = []
    negative_rows: list[int] = []
    hard_negatives = int(config.get("hard_negatives", 25))
    for question in questions:
        start = question * candidate_limit
        values = target[start:start + candidate_limit]
        positives = np.flatnonzero(values == 1)
        negatives = np.flatnonzero(values == 0)[:hard_negatives]
        for positive in positives:
            for negative in negatives:
                positive_rows.append(start + int(positive))
                negative_rows.append(start + int(negative))
    features = torch.from_numpy(normalized).to(device)
    positives = torch.tensor(positive_rows, dtype=torch.long, device=device)
    negatives = torch.tensor(negative_rows, dtype=torch.long, device=device)
    hidden = int(config.get("hidden", 32))
    network = nn.Sequential(
        nn.Linear(matrix.shape[1], hidden),
        nn.ReLU(),
        nn.Dropout(float(config.get("dropout", 0.05))),
        nn.Linear(hidden, 1),
    ).to(device)
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=float(config.get("learning_rate", 0.002)),
        weight_decay=float(config.get("weight_decay", 0.0005)),
    )
    batch_size = int(config.get("batch_size", 2048))
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    network.train()
    for _ in range(int(config.get("epochs", 30))):
        order = torch.randperm(len(positives), generator=generator, device=device)
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            positive_score = network(features[positives[batch]]).squeeze(1)
            negative_score = network(features[negatives[batch]]).squeeze(1)
            loss = torch.nn.functional.softplus(-(positive_score - negative_score)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return TorchRanker(network, mean, scale, device)


def train_torch_listwise_ranker(
    matrix: np.ndarray,
    target: np.ndarray,
    questions: Sequence[int],
    candidate_limit: int,
    config: Mapping[str, Any],
) -> TorchRanker:
    """Train a query-normalized ListNet-style ranker on the candidate slate."""

    import torch
    from torch import nn

    seed = int(config.get("seed", 20260826))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    training_pool = min(int(config.get("training_pool", 35)), candidate_limit)
    indices = row_indices(questions, candidate_limit)
    training = matrix[indices]
    mean = training.mean(axis=0, keepdims=True).astype(np.float32)
    scale = training.std(axis=0, keepdims=True).astype(np.float32)
    scale[scale < 1e-5] = 1.0
    normalized = ((matrix - mean) / scale).astype(np.float32)
    question_count = matrix.shape[0] // candidate_limit
    feature_blocks = normalized.reshape(question_count, candidate_limit, -1)
    target_blocks = target.reshape(question_count, candidate_limit)
    valid_questions = [
        int(question) for question in questions
        if int(target_blocks[int(question), :training_pool].sum()) > 0
    ]
    if not valid_questions:
        raise ValueError("Listwise training has no positive candidate slates")
    features = torch.from_numpy(
        feature_blocks[valid_questions, :training_pool]
    ).to(device)
    labels = torch.from_numpy(
        target_blocks[valid_questions, :training_pool].astype(np.float32)
    ).to(device)
    target_distribution = labels / labels.sum(dim=1, keepdim=True).clamp_min(1.0)
    hidden = int(config.get("hidden", 0))
    if hidden > 0:
        network: nn.Module = nn.Sequential(
            nn.Linear(matrix.shape[1], hidden),
            nn.GELU(),
            nn.Dropout(float(config.get("dropout", 0.05))),
            nn.Linear(hidden, 1),
        )
    else:
        network = nn.Linear(matrix.shape[1], 1)
    network = network.to(device)
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=float(config.get("learning_rate", 0.01)),
        weight_decay=float(config.get("weight_decay", 0.001)),
    )
    batch_size = int(config.get("query_batch_size", 256))
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    network.train()
    for _ in range(int(config.get("epochs", 180))):
        order = torch.randperm(len(valid_questions), generator=generator, device=device)
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            scores = network(features[batch]).squeeze(2)
            log_probabilities = torch.log_softmax(scores, dim=1)
            loss = -(target_distribution[batch] * log_probabilities).sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            optimizer.step()
    return TorchRanker(network, mean, scale, device)


def train_model(
    config: Mapping[str, Any],
    matrix: np.ndarray,
    target: np.ndarray,
    questions: Sequence[int],
    candidate_limit: int,
) -> Any:
    kind = str(config["kind"])
    if kind == "pairwise_logistic":
        differences, labels, weights = pairwise_rows(
            matrix, target, questions, candidate_limit, int(config["hard_negatives"])
        )
        model = Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                C=float(config["C"]), solver="liblinear", fit_intercept=False,
                max_iter=800, random_state=20260825,
            )),
        ])
        model.fit(differences, labels, model__sample_weight=weights)
        return model
    if kind == "pointwise_logistic":
        indices = row_indices(questions, candidate_limit)
        model = Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                C=float(config["C"]), solver="liblinear", max_iter=800,
                random_state=20260825,
            )),
        ])
        model.fit(
            matrix[indices], target[indices],
            model__sample_weight=point_weights(target, questions, candidate_limit),
        )
        return model
    if kind == "hist_gradient_boosting":
        indices = row_indices(questions, candidate_limit)
        model = HistGradientBoostingClassifier(
            learning_rate=float(config["learning_rate"]),
            max_iter=int(config["max_iter"]),
            max_leaf_nodes=int(config["max_leaf_nodes"]),
            min_samples_leaf=int(config.get("min_samples_leaf", 30)),
            l2_regularization=float(config.get("l2_regularization", 1.0)),
            random_state=20260825,
        )
        model.fit(
            matrix[indices], target[indices],
            sample_weight=point_weights(target, questions, candidate_limit),
        )
        return model
    if kind == "torch_ranknet":
        return train_torch_ranker(matrix, target, questions, candidate_limit, config)
    if kind == "torch_listwise":
        return train_torch_listwise_ranker(
            matrix, target, questions, candidate_limit, config
        )
    raise ValueError(f"Unknown model kind: {kind}")


def model_scores(model: Any, matrix: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(matrix), dtype=np.float32)
    probabilities = model.predict_proba(matrix)
    return np.asarray(probabilities[:, 1], dtype=np.float32)


def exact_tail_rescue(
    answers: list[str],
    candidates: Sequence[str],
    feature_block: np.ndarray,
    feature_names: Sequence[str],
    threshold: int = 10,
) -> list[str]:
    phrase_column = feature_names.index("longest_phrase_tokens")
    phrase_by_id = {
        doc_id: int(feature_block[rank, phrase_column])
        for rank, doc_id in enumerate(candidates)
        if rank < len(feature_block)
    }
    tail_score = phrase_by_id.get(answers[-1], 0)
    best: tuple[int, int, str] | None = None
    for rank, doc_id in enumerate(candidates, 1):
        if doc_id in answers:
            continue
        phrase = phrase_by_id.get(doc_id, 0)
        candidate = (phrase, -rank, doc_id)
        if phrase >= threshold and (best is None or candidate > best):
            best = candidate
    if best is not None and best[0] > tail_score:
        return [*answers[:-1], best[2]]
    return answers


def apply_policy(
    split: AuditSplit,
    scores: np.ndarray,
    question_indices: Sequence[int],
    matrix: np.ndarray,
    feature_names: Sequence[str],
    candidate_limit: int,
    policy: Mapping[str, Any],
) -> dict[str, dict[str, list[str]]]:
    output: dict[str, dict[str, list[str]]] = {}
    pool = int(policy["pool"])
    for question in question_indices:
        candidates = split.candidates[question][:pool]
        values = scores[question, :pool]
        model_order = sorted(range(len(candidates)), key=lambda rank: (-float(values[rank]), rank))
        model_rank = {rank: position for position, rank in enumerate(model_order, 1)}
        if policy["kind"] == "prefix":
            keep = int(policy["keep"])
            answers = list(split.baseline[question][:keep])
            for rank in model_order:
                doc_id = candidates[rank]
                if doc_id not in answers:
                    answers.append(doc_id)
                if len(answers) == 5:
                    break
        elif policy["kind"] == "memory_slot5":
            # V16b is intentionally conservative: keep the locked anchor's
            # first four answers and replace only its weakest slot when a
            # non-anchor c35 candidate clears both gates.  The memory signal
            # is an inference-time feature, never a label lookup.
            answers = list(split.baseline[question][:5])
            if len(answers) != 5:
                raise ValueError(
                    f"{split.name}/{split.qids[question]}: baseline must have five answers"
                )
            memory_feature = str(policy.get("memory_feature", "degree_corrected_memory_score"))
            if memory_feature not in feature_names:
                raise ValueError(f"Missing memory feature for policy: {memory_feature}")
            memory_column = feature_names.index(memory_feature)
            feature_block = matrix[question * candidate_limit:(question + 1) * candidate_limit]
            tail_doc = answers[-1]
            try:
                tail_rank = candidates.index(tail_doc)
                tail_model_score = float(values[tail_rank])
                tail_memory_score = float(feature_block[tail_rank, memory_column])
            except ValueError:
                tail_model_score = float("-inf")
                tail_memory_score = 0.0
            memory_threshold = max(
                float(policy.get("memory_floor", 0.0)),
                tail_memory_score + float(policy.get("memory_margin", 0.0)),
            )
            model_threshold = tail_model_score + float(policy.get("model_margin", 0.0))
            replacement: str | None = None
            for rank in model_order:
                doc_id = candidates[rank]
                if doc_id in answers:
                    continue
                model_score = float(values[rank])
                memory_score = float(feature_block[rank, memory_column])
                if model_score < model_threshold or memory_score < memory_threshold:
                    continue
                replacement = doc_id
                break
            if replacement is not None:
                answers = [*answers[:-1], replacement]
        elif policy["kind"] == "rrf":
            base_rank = {doc_id: rank for rank, doc_id in enumerate(split.baseline[question], 1)}
            k = float(policy["k"])
            stage_weight = float(policy["stage_weight"])
            base_weight = float(policy["base_weight"])
            fused: list[tuple[float, int, str]] = []
            for rank, doc_id in enumerate(candidates, 1):
                score = 1.0 / (k + model_rank[rank - 1])
                score += stage_weight / (k + rank)
                if doc_id in base_rank:
                    score += base_weight / (k + base_rank[doc_id])
                fused.append((score, -rank, doc_id))
            answers = [item[2] for item in sorted(fused, reverse=True)[:5]]
        else:
            raise ValueError(f"Unknown policy: {policy['kind']}")
        if len(answers) != 5 or len(set(answers)) != 5:
            raise ValueError(f"{split.name}/{split.qids[question]}: policy produced invalid answers")
        if bool(policy.get("exact_rescue")):
            block = matrix[question * candidate_limit:(question + 1) * candidate_limit]
            answers = exact_tail_rescue(answers, split.candidates[question][:pool], block, feature_names)
        output[split.qids[question]] = {"answer": answers}
    return output


def baseline_predictions(
    split: AuditSplit,
    question_indices: Sequence[int],
    matrix: np.ndarray,
    feature_names: Sequence[str],
    candidate_limit: int,
    exact_rescue: bool,
) -> dict[str, dict[str, list[str]]]:
    output: dict[str, dict[str, list[str]]] = {}
    for question in question_indices:
        answers = list(split.baseline[question])
        if exact_rescue:
            block = matrix[question * candidate_limit:(question + 1) * candidate_limit]
            answers = exact_tail_rescue(
                answers, split.candidates[question][:candidate_limit], block, feature_names
            )
        output[split.qids[question]] = {"answer": answers}
    return output


def reference_map(split: AuditSplit, question_indices: Sequence[int]) -> dict[str, dict[str, list[str]]]:
    if split.references is None:
        raise ValueError("References are unavailable")
    return {
        split.qids[index]: {"answer": sorted(split.references[index])}
        for index in question_indices
    }


def policy_grid() -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    for pool in (35, 50, 80):
        for keep in range(5):
            for exact in (False, True):
                policies.append({"kind": "prefix", "pool": pool, "keep": keep, "exact_rescue": exact})
        for k in (5, 20):
            for stage_weight in (0.0, 0.5, 1.0):
                for base_weight in (0.0, 0.5, 1.0):
                    for exact in (False, True):
                        policies.append({
                            "kind": "rrf", "pool": pool, "k": k,
                            "stage_weight": stage_weight, "base_weight": base_weight,
                            "exact_rescue": exact,
                        })
    return policies


def model_grid(profile: str = "base") -> list[dict[str, Any]]:
    if profile == "pointwise-v2-winner":
        return [{"kind": "pointwise_logistic", "C": 1.0}]
    if profile == "listwise-v2-winner":
        return [{
            "kind": "torch_listwise", "hidden": 0,
            "learning_rate": 0.005, "epochs": 220,
            "training_pool": 35, "weight_decay": 0.005,
            "query_batch_size": 256, "seed": 20260826,
        }]
    if profile == "pointwise-c-sweep":
        return [
            {"kind": "pointwise_logistic", "C": value}
            for value in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0)
        ]
    if profile == "listwise-v2":
        return [
            {
                "kind": "torch_listwise", "hidden": 0,
                "learning_rate": learning_rate, "epochs": 220,
                "training_pool": 35, "weight_decay": weight_decay,
                "query_batch_size": 256, "seed": 20260826,
            }
            for learning_rate, weight_decay in (
                (0.02, 0.0001), (0.01, 0.001), (0.005, 0.005)
            )
        ] + [
            {
                "kind": "torch_listwise", "hidden": hidden,
                "learning_rate": 0.003, "epochs": 160,
                "training_pool": 35, "weight_decay": 0.001,
                "dropout": 0.05, "query_batch_size": 256, "seed": 20260826,
            }
            for hidden in (16, 32)
        ]
    if profile != "base":
        raise ValueError(f"Unknown model-grid profile: {profile}")
    output: list[dict[str, Any]] = []
    for value in (0.05, 0.2, 1.0):
        output.append({"kind": "pairwise_logistic", "C": value, "hard_negatives": 25})
        output.append({"kind": "pointwise_logistic", "C": value})
    for leaves, learning_rate in ((7, 0.08), (15, 0.06), (31, 0.05)):
        output.append({
            "kind": "hist_gradient_boosting", "max_leaf_nodes": leaves,
            "learning_rate": learning_rate, "max_iter": 140,
            "min_samples_leaf": 35, "l2_regularization": 2.0,
        })
    for hidden, learning_rate in ((32, 0.002), (64, 0.0015), (64, 0.003)):
        output.append({
            "kind": "torch_ranknet", "hidden": hidden,
            "learning_rate": learning_rate, "epochs": 30,
            "hard_negatives": 25, "dropout": 0.05,
            "weight_decay": 0.0005, "seed": 20260825,
        })
    return output


def evaluate_policy(
    split: AuditSplit,
    predictions: Mapping[str, Mapping[str, Sequence[str]]],
    question_indices: Sequence[int],
    fold_by_question: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    metrics = official_metrics(dict(predictions), reference_map(split, question_indices))
    result: dict[str, Any] = dict(metrics)
    if fold_by_question is not None:
        recalls: list[float] = []
        for fold in sorted(set(fold_by_question.values())):
            indices = [index for index in question_indices if fold_by_question[index] == fold]
            subset = {split.qids[index]: predictions[split.qids[index]] for index in indices}
            recalls.append(float(official_metrics(subset, reference_map(split, indices))["recall"]))
        result["fold_recalls"] = recalls
        result["fold_recall_mean"] = statistics.mean(recalls)
        result["fold_recall_std"] = statistics.pstdev(recalls)
        result["robust_recall"] = float(metrics["recall"]) - 0.25 * result["fold_recall_std"]
    return result


def select_model_and_policy(
    split: AuditSplit,
    matrix: np.ndarray,
    target: np.ndarray,
    feature_names: Sequence[str],
    selection_indices: np.ndarray,
    groups: np.ndarray,
    candidate_limit: int,
    model_profile: str = "base",
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], np.ndarray, dict[int, int]]:
    group_kfold = GroupKFold(n_splits=5)
    folds = list(group_kfold.split(selection_indices, groups=groups[selection_indices]))
    fold_by_question: dict[int, int] = {}
    configurations = model_grid(model_profile)
    policies = policy_grid()
    leaderboard: list[dict[str, Any]] = []
    best_scores: np.ndarray | None = None
    for config_index, config in enumerate(configurations, 1):
        print(f"CV model {config_index}/{len(configurations)}: {config}", flush=True)
        out_of_fold = np.full((len(split.qids), candidate_limit), np.nan, dtype=np.float32)
        for fold, (train_local, validation_local) in enumerate(folds):
            train_questions = selection_indices[train_local]
            validation_questions = selection_indices[validation_local]
            for question in validation_questions:
                fold_by_question[int(question)] = fold
            model = train_model(config, matrix, target, train_questions, candidate_limit)
            validation_rows = row_indices(validation_questions, candidate_limit)
            values = model_scores(model, matrix[validation_rows]).reshape(-1, candidate_limit)
            out_of_fold[validation_questions] = values
        best_for_model: dict[str, Any] | None = None
        for policy in policies:
            predictions = apply_policy(
                split, out_of_fold, selection_indices, matrix, feature_names,
                candidate_limit, policy,
            )
            evaluation = evaluate_policy(
                split, predictions, selection_indices, fold_by_question
            )
            candidate = {"config": dict(config), "policy": dict(policy), "selection_oof": evaluation}
            key = (
                evaluation["robust_recall"], evaluation["recall"],
                evaluation["precision"], -evaluation["fold_recall_std"],
            )
            if best_for_model is None:
                best_for_model = candidate
            else:
                previous = best_for_model["selection_oof"]
                previous_key = (
                    previous["robust_recall"], previous["recall"],
                    previous["precision"], -previous["fold_recall_std"],
                )
                if key > previous_key:
                    best_for_model = candidate
        assert best_for_model is not None
        leaderboard.append(best_for_model)
        leaderboard.sort(
            key=lambda item: (
                item["selection_oof"]["robust_recall"],
                item["selection_oof"]["recall"],
                item["selection_oof"]["precision"],
            ),
            reverse=True,
        )
        if leaderboard[0] is best_for_model:
            best_scores = out_of_fold.copy()
        print("  best policy", json.dumps(best_for_model, ensure_ascii=False), flush=True)
    winner = leaderboard[0]
    if best_scores is None:
        raise RuntimeError("No winning OOF scores retained")
    return winner["config"], winner["policy"], leaderboard, best_scores, fold_by_question


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts-zip", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--holdout-stage1", type=Path, required=True)
    parser.add_argument("--holdout-stage2", type=Path, required=True)
    parser.add_argument("--holdout-reference", type=Path, required=True)
    parser.add_argument("--public-stage1", type=Path, required=True)
    parser.add_argument("--public-stage2", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-limit", type=int, default=80)
    parser.add_argument("--final-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--minimum-final-delta", type=float, default=0.0)
    args = parser.parse_args()
    if args.candidate_limit < 80:
        raise ValueError("candidate-limit must be at least 80 for the fixed policy grid")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    holdout = load_audit_split(
        "holdout", args.holdout_stage1, args.holdout_stage2, args.holdout_reference
    )
    public = load_audit_split(
        "public", args.public_stage1, args.public_stage2, None
    )
    needed_ids = {
        doc_id
        for split in (holdout, public)
        for row in split.candidates
        for doc_id in row[:args.candidate_limit]
    }
    needed_ids.update(doc_id for split in (holdout, public) for row in split.baseline for doc_id in row)
    priors = load_priors(args.index)
    documents, document_frequency = load_documents(args.contexts_zip, needed_ids, priors)
    holdout_matrix, holdout_target, feature_names = build_features(
        holdout, documents, document_frequency, args.candidate_limit
    )
    public_matrix, _, public_feature_names = build_features(
        public, documents, document_frequency, args.candidate_limit
    )
    if feature_names != public_feature_names or holdout_target is None:
        raise RuntimeError("Feature schema mismatch")
    groups = build_leak_groups(holdout)
    all_indices = np.arange(len(holdout.qids), dtype=np.int32)
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=args.final_fraction, random_state=args.seed
    )
    selection_local, final_local = next(splitter.split(all_indices, groups=groups))
    selection_indices = all_indices[selection_local]
    final_indices = all_indices[final_local]
    selection_documents = set().union(*(holdout.references[index] for index in selection_indices))
    final_documents = set().union(*(holdout.references[index] for index in final_indices))
    shared_reference_documents = selection_documents & final_documents
    if shared_reference_documents:
        raise RuntimeError("Final split is not reference-document-disjoint")
    print(
        f"locked split: selection={len(selection_indices)}, final={len(final_indices)}, "
        f"groups={len(set(groups))}, shared reference docs=0",
        flush=True,
    )
    config, policy, leaderboard, oof_scores, fold_by_question = select_model_and_policy(
        holdout, holdout_matrix, holdout_target, feature_names,
        selection_indices, groups, args.candidate_limit,
    )
    locked_selection = apply_policy(
        holdout, oof_scores, selection_indices, holdout_matrix, feature_names,
        args.candidate_limit, policy,
    )
    selection_evaluation = evaluate_policy(
        holdout, locked_selection, selection_indices, fold_by_question
    )
    selection_baseline = evaluate_policy(
        holdout,
        baseline_predictions(
            holdout, selection_indices, holdout_matrix, feature_names,
            args.candidate_limit, False,
        ),
        selection_indices,
    )
    selection_exact = evaluate_policy(
        holdout,
        baseline_predictions(
            holdout, selection_indices, holdout_matrix, feature_names,
            args.candidate_limit, True,
        ),
        selection_indices,
    )
    print("configuration and policy locked; opening final test once", flush=True)
    locked_model = train_model(
        config, holdout_matrix, holdout_target, selection_indices, args.candidate_limit
    )
    final_rows = row_indices(final_indices, args.candidate_limit)
    final_scores = np.full((len(holdout.qids), args.candidate_limit), np.nan, dtype=np.float32)
    final_scores[final_indices] = model_scores(
        locked_model, holdout_matrix[final_rows]
    ).reshape(-1, args.candidate_limit)
    final_predictions = apply_policy(
        holdout, final_scores, final_indices, holdout_matrix, feature_names,
        args.candidate_limit, policy,
    )
    final_evaluation = evaluate_policy(holdout, final_predictions, final_indices)
    final_baseline = evaluate_policy(
        holdout,
        baseline_predictions(
            holdout, final_indices, holdout_matrix, feature_names,
            args.candidate_limit, False,
        ),
        final_indices,
    )
    final_exact = evaluate_policy(
        holdout,
        baseline_predictions(
            holdout, final_indices, holdout_matrix, feature_names,
            args.candidate_limit, True,
        ),
        final_indices,
    )
    promotion_delta = float(final_evaluation["recall"] - final_exact["recall"])
    promoted = (
        promotion_delta >= args.minimum_final_delta
        and float(selection_evaluation["recall"]) >= float(selection_exact["recall"])
    )
    public_output: Path | None = None
    public_changes = 0
    if promoted:
        final_model = train_model(
            config, holdout_matrix, holdout_target, all_indices, args.candidate_limit
        )
        public_values = model_scores(final_model, public_matrix).reshape(
            len(public.qids), args.candidate_limit
        )
        public_predictions = apply_policy(
            public, public_values, all_indices, public_matrix, feature_names,
            args.candidate_limit, policy,
        )
        public_output = args.output_dir / "v11f_tiny_ltr.json"
        write_json(public_output, public_predictions)
        public_changes = sum(
            public_predictions[qid]["answer"] != public.baseline[index]
            for index, qid in enumerate(public.qids)
        )
    report = {
        "kind": "legalir_v11_tiny_supervised_fusion_ltr",
        "leakage_policy": {
            "fit_audit_used_for_supervision": False,
            "reason": "fit audit contains train_exact_memory",
            "selection_source": "clean holdout questions absent from retrieval index",
            "grouping": "connected shared-reference documents plus char-TFIDF similarity >= 0.75",
            "public_labels_used": False,
            "v57_predictions_used": False,
        },
        "model_compliance": {
            "retrieval_model_billions": 0.33,
            "reranker_model_billions": 0.76,
            "tiny_ltr_parameters_upper_bound": 10000,
            "total_model_plan_billions_approx": 1.09,
            "under_4b": True,
        },
        "candidate_limit": args.candidate_limit,
        "features": list(feature_names),
        "split": {
            "questions": len(holdout.qids),
            "selection_questions": len(selection_indices),
            "final_questions": len(final_indices),
            "groups": len(set(groups)),
            "shared_reference_documents": len(shared_reference_documents),
            "seed": args.seed,
        },
        "locked_config": config,
        "locked_policy": policy,
        "selection": {
            "baseline_c35": selection_baseline,
            "baseline_c35_exact_t10_c80": selection_exact,
            "oof_ltr": selection_evaluation,
        },
        "final_test_opened_once": {
            "baseline_c35": final_baseline,
            "baseline_c35_exact_t10_c80": final_exact,
            "locked_ltr": final_evaluation,
            "delta_vs_exact_recall": promotion_delta,
        },
        "promotion": {
            "minimum_final_delta": args.minimum_final_delta,
            "promoted": promoted,
            "public_submission": str(public_output.resolve()) if public_output else None,
            "public_queries_changed_vs_c35": public_changes,
        },
        "model_leaderboard": leaderboard,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = args.output_dir / "v11_tiny_ltr_report.json"
    write_json(report_path, report)
    print(json.dumps({
        "report": str(report_path.resolve()),
        "locked_config": config,
        "locked_policy": policy,
        "selection": report["selection"],
        "final": report["final_test_opened_once"],
        "promotion": report["promotion"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
