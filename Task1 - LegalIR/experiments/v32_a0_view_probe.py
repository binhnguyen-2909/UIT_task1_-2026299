"""V32-A0/A1 deterministic query-view diversification probe.

This keeps the frozen BGE-M3 document index and only changes the query view.
It is deliberately diagnostic: no Stage2 scoring, no document re-indexing, and
no labels from the locked holdout.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
V27_DIR = SCRIPT_DIR / "v27"
V29_DIR = SCRIPT_DIR / "v29"
for path in (V27_DIR, V29_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from v27_common import load_metadata, sha256_file, write_json, write_jsonl  # noqa: E402
from v29_a0_dense import _sample_manifest  # noqa: E402


TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)
SPACE_RE = re.compile(r"\s+")

# These are question framing only.  The rewrite never removes content words
# or the negation/numeric tokens explicitly protected below.
PREFIX_RE = re.compile(
    r"^\s*(?:(?:hãy\s+)?cho\s+biết|xin\s+cho\s+biết|theo\s+quy\s+định\s+(?:của\s+pháp\s+luật|pháp\s+luật)|theo\s+quy\s+định|trong\s+trường\s+hợp)\s*[,;:]?\s*",
    re.IGNORECASE,
)
SUFFIX_RE = re.compile(
    r"\s*(?:được\s+quy\s+định\s+)?(?:như\s+thế\s+nào|thế\s+nào|ra\s+sao|là\s+gì|có\s+những\s+quyền\s+hạn\s+nào|có\s+những\s+nghĩa\s+vụ\s+nào|có\s+được\s+không)\s*[?。.!！]*\s*$",
    re.IGNORECASE,
)
LEADING_FILLER_RE = re.compile(r"^\s*(?:đối\s+với|về)\s+", re.IGNORECASE)

# Do not let generic question words dominate the anchor.  Negation and
# boundary/modality words are intentionally absent from this set.
ANCHOR_STOP = {
    "ai", "nào", "gì", "sao", "thế", "nào", "như", "được", "phải",
    "hãy", "cho", "biết", "xin", "và", "của", "theo",
}
PROTECTED = {"không", "chưa", "trừ", "quá", "dưới", "trên", "từ", "đến", "nếu", "khi", "trường", "hợp"}


def _clean(text: str) -> str:
    return SPACE_RE.sub(" ", str(text).replace("\n", " ")).strip(" ,;:?")


def _anchor(query: str) -> str:
    text = _clean(query)
    text = PREFIX_RE.sub("", text)
    text = SUFFIX_RE.sub("", text)
    text = LEADING_FILLER_RE.sub("", text)
    # Remove only residual interrogative filler as whole words.  Preserve
    # negation, numeric terms, citations, conditions, and legal modalities.
    words = text.split()
    while words and words[0].casefold() in {"vậy", "thì"}:
        words.pop(0)
    filtered = [word for word in words if word.casefold().strip(".,;:") not in ANCHOR_STOP]
    result = _clean(" ".join(filtered))
    return result or _clean(query)


def _statute_style(query: str, anchor: str) -> str:
    # Keep a short, stable legal register.  The original content remains in
    # the view; only the frame changes.  Avoid duplicate "quy định" tokens.
    body = anchor
    if body.casefold().startswith("quy định"):
        return body
    if re.search(r"\b(xử lý|xử phạt|phạt|trách nhiệm|nghĩa vụ|quyền|điều kiện|thẩm quyền|mức)\b", body, re.IGNORECASE):
        return _clean(f"quy định pháp luật về {body}")
    return _clean(f"quy định pháp luật đối với {body}")


def make_views(query: str) -> dict[str, str]:
    q0 = _clean(query)
    q1 = _anchor(q0)
    q2 = _statute_style(q0, q1)
    return {"Q0": q0, "Q1": q1, "Q2": q2}


def _sample_doc_ids(path: Path, max_chunks_per_doc: int) -> tuple[list[str], list[str]]:
    """Return the exact doc-id sequence corresponding to the frozen vectors."""
    selected_doc_ids: list[str] = []
    all_doc_ids: list[str] = []
    current_doc: str | None = None
    count = 0

    def flush() -> None:
        nonlocal count, current_doc
        if current_doc is None:
            return
        all_doc_ids.append(current_doc)
        if count <= max_chunks_per_doc:
            selected = count
        else:
            head = max(1, max_chunks_per_doc // 4)
            tail = max(1, max_chunks_per_doc // 4)
            interior = max_chunks_per_doc - head - tail
            selected = len(set(range(head)))
            selected += len({min(count - tail - 1, head + int(round((i + 0.5) * ((count - head - tail) / max(interior, 1))))) for i in range(interior)})
            selected += len(set(range(count - tail, count)))
        selected_doc_ids.extend([current_doc] * selected)
        count = 0

    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            doc_id = str(json.loads(line)["doc_id"])
            if current_doc is None:
                current_doc = doc_id
            elif doc_id != current_doc:
                flush()
                current_doc = doc_id
            count += 1
    flush()
    return selected_doc_ids, all_doc_ids


def _load_base_pools(path: Path, qids: set[str]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                qid = str(row["qid"])
                if qid in qids:
                    result[qid] = [{"doc_id": str(x["doc_id"]), "rank": int(x["rank"]), "score": float(x["score"])} for x in row["candidates"]]
    if result.keys() != qids:
        raise ValueError(f"base dense pool coverage mismatch: {len(result)} != {len(qids)}")
    return result


def _retrieve(
    model: SentenceTransformer,
    query_rows: list[tuple[str, str, str]],
    chunk_vectors: np.ndarray,
    chunk_doc_ids: list[str],
    doc_ids: list[str],
    device: str,
    query_batch_size: int,
    top_k: int,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    device_obj = torch.device(device)
    vectors = torch.as_tensor(chunk_vectors, dtype=torch.float16, device=device_obj)
    doc_index = {doc_id: index for index, doc_id in enumerate(doc_ids)}
    chunk_doc_index = torch.tensor([doc_index[doc_id] for doc_id in chunk_doc_ids], dtype=torch.long, device=device_obj)
    output: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for start in range(0, len(query_rows), query_batch_size):
        batch = query_rows[start : start + query_batch_size]
        texts = [row[2] for row in batch]
        qvec = model.encode(texts, batch_size=query_batch_size, show_progress_bar=False, convert_to_tensor=True, normalize_embeddings=True).to(dtype=torch.float16)
        with torch.inference_mode():
            scores = qvec @ vectors.T
            doc_scores = torch.full((len(batch), len(doc_ids)), -torch.inf, dtype=scores.dtype, device=device_obj)
            expanded = chunk_doc_index.unsqueeze(0).expand(len(batch), -1)
            doc_scores.scatter_reduce_(1, expanded, scores, reduce="amax", include_self=True)
            values, indices = torch.topk(doc_scores, k=min(top_k, len(doc_ids)), dim=1)
        values_cpu = values.float().cpu().numpy()
        indices_cpu = indices.cpu().numpy()
        for row_index, (qid, view, _) in enumerate(batch):
            output[(qid, view)] = [{"doc_id": doc_ids[int(index)], "rank": rank + 1, "score": float(score)} for rank, (index, score) in enumerate(zip(indices_cpu[row_index], values_cpu[row_index]))]
        del qvec, scores, doc_scores, expanded
    del vectors, chunk_doc_index
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.empty_cache()
    return output


def _rank(row: list[dict[str, Any]], refs: set[str]) -> int | None:
    for item in row:
        if item["doc_id"] in refs:
            return int(item["rank"])
    return None


def _rank_map(row: list[dict[str, Any]]) -> dict[str, int]:
    return {str(item["doc_id"]): int(item["rank"]) for item in row}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--base-pools", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--max-chunks-per-doc", type=int, default=8)
    parser.add_argument("--probe-fold", type=int, default=-1, help="Use all qids by default; set qid%%5 for a probe subset.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    questions = load_metadata(args.metadata)
    if args.probe_fold >= 0:
        questions = {qid: row for qid, row in questions.items() if int(qid) % 5 == args.probe_fold}
    qids = set(questions)
    base = _load_base_pools(args.base_pools, qids)
    views = {qid: make_views(row["query"]) for qid, row in questions.items()}
    write_jsonl(args.output_dir / "v32_a0_query_views.jsonl", ({"qid": qid, **views[qid]} for qid in sorted(views, key=lambda x: int(x))))

    chunk_doc_ids, doc_ids = _sample_doc_ids(args.manifest, args.max_chunks_per_doc)
    chunk_vectors = np.load(args.embeddings, mmap_mode="r")
    if chunk_vectors.shape[0] != len(chunk_doc_ids):
        raise ValueError(f"embedding/vector alignment mismatch {chunk_vectors.shape[0]} != {len(chunk_doc_ids)}")
    device = args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(str(args.model.resolve()), device=device, local_files_only=True)
    model.max_seq_length = 128
    query_rows = [(qid, view, views[qid][view]) for qid in sorted(questions, key=lambda x: int(x)) for view in ("Q0", "Q1", "Q2")]
    retrieved = _retrieve(model, query_rows, chunk_vectors, chunk_doc_ids, doc_ids, device, args.query_batch_size, 500)
    del model
    write_jsonl(
        args.output_dir / "v32_a1_view_candidates.jsonl",
        (
            {
                "qid": qid,
                "Q0": retrieved[(qid, "Q0")],
                "Q1": retrieved[(qid, "Q1")],
                "Q2": retrieved[(qid, "Q2")],
            }
            for qid in sorted(questions, key=lambda x: int(x))
        ),
    )

    cutoffs = [5, 10, 20, 50, 100, 200, 500]
    summary: dict[str, Any] = {"kind": "legalir_v32_a0_a1_query_view_probe", "schema_version": 1, "queries": len(questions), "probe_fold": args.probe_fold, "views": ["Q0", "Q1", "Q2"], "cutoffs": {}, "failure_set": {}, "elapsed_seconds": None}
    for k in cutoffs:
        q0_gold = 0
        best_view_gold = 0
        q0_rank_values: list[int] = []
        best_rank_values: list[int] = []
        compressed_500_to_50 = 0
        compressed_100_to_20 = 0
        q0_top = 0
        winner_any = 0
        recovered_denominator = 0
        for qid, question in questions.items():
            refs = set(question["references"])
            q0 = retrieved[(qid, "Q0")][:k]
            variant_rows = [retrieved[(qid, view)][:k] for view in ("Q1", "Q2")]
            q0_rank = _rank(q0, refs)
            best_rank = min((rank for row in [q0, *variant_rows] if (rank := _rank(row, refs)) is not None), default=None)
            if q0_rank is not None:
                q0_gold += 1
                q0_rank_values.append(q0_rank)
            if best_rank is not None:
                best_view_gold += 1
                best_rank_values.append(best_rank)
            q0_top += int(q0_rank is not None and q0_rank <= k)
            recovered_denominator += int(best_rank is not None)
            winner_any += int(any(_rank(retrieved[(qid, view)][:k], refs) == 1 for view in ("Q0", "Q1", "Q2")))
            q0_500 = _rank(retrieved[(qid, "Q0")][:500], refs)
            best_50 = min((rank for view in ("Q0", "Q1", "Q2") if (rank := _rank(retrieved[(qid, view)][:50], refs)) is not None), default=None)
            q0_100 = _rank(retrieved[(qid, "Q0")][:100], refs)
            best_20 = min((rank for view in ("Q0", "Q1", "Q2") if (rank := _rank(retrieved[(qid, view)][:20], refs)) is not None), default=None)
            compressed_500_to_50 += int(q0_500 is not None and q0_500 > 100 and best_50 is not None and best_50 <= 50)
            compressed_100_to_20 += int(q0_100 is not None and q0_100 > 20 and best_20 is not None and best_20 <= 20)
        summary["cutoffs"][str(k)] = {
            "q0_gold_at_k": q0_gold,
            "best_view_gold_at_k": best_view_gold,
            "improvement_q0_to_best_view": best_view_gold - q0_gold,
            "q0_rank_median": float(np.median(q0_rank_values)) if q0_rank_values else None,
            "best_view_rank_median": float(np.median(best_rank_values)) if best_rank_values else None,
            "view_winner_rate": winner_any / max(recovered_denominator, 1),
            "view_winner_queries": winner_any,
            "recovered_denominator": recovered_denominator,
            "rank_compressed_500_to_50": compressed_500_to_50,
            "rank_compressed_100_to_20": compressed_100_to_20,
        }

    # Per-gold view ranks on the V29/V31 recovery sets.  This is the direct
    # diagnostic for whether a rewrite changes ordering rather than just recall.
    failure_rows = []
    for qid in sorted(questions, key=lambda x: int(x)):
        question = questions[qid]
        refs = set(question["references"])
        q0_map = _rank_map(retrieved[(qid, "Q0")])
        q1_map = _rank_map(retrieved[(qid, "Q1")])
        q2_map = _rank_map(retrieved[(qid, "Q2")])
        base_new = [str(x["doc_id"]) for x in base[qid] if int(x["rank"]) <= 500 and str(x["doc_id"]) in refs and str(x["doc_id"]) not in set(item["doc_id"] for item in base[qid][:100])]
        ft_rows = []
        for docid in sorted(refs, key=lambda x: int(x) if x.isdigit() else 10**18):
            ranks = {"Q0": q0_map.get(docid), "Q1": q1_map.get(docid), "Q2": q2_map.get(docid)}
            if any(value is not None for value in ranks.values()) and (docid in base_new or min(value for value in ranks.values() if value is not None) <= 100):
                ft_rows.append({"doc_id": docid, "view_ranks": ranks, "best_view_rank": min(value for value in ranks.values() if value is not None)})
        if base_new or ft_rows:
            failure_rows.append({"qid": qid, "base_new_gold": base_new, "gold_view_ranks": ft_rows})
    summary["failure_set"] = {"rows": failure_rows, "queries_with_rows": len(failure_rows)}
    summary["gate"] = {
        "winner_rate_ge_0_30_at_k20": summary["cutoffs"]["20"]["view_winner_rate"] >= 0.30,
        "compressed_500_to_50_at_least_8": summary["cutoffs"]["50"]["rank_compressed_500_to_50"] >= 8,
        "pass_deterministic_probe": summary["cutoffs"]["20"]["view_winner_rate"] >= 0.30 and summary["cutoffs"]["50"]["rank_compressed_500_to_50"] >= 8,
    }
    summary["inputs"] = {"metadata": str(args.metadata.resolve()), "metadata_sha256": sha256_file(args.metadata), "base_pools": str(args.base_pools.resolve()), "base_pools_sha256": sha256_file(args.base_pools), "embeddings": str(args.embeddings.resolve()), "manifest": str(args.manifest.resolve()), "device": device, "max_chunks_per_doc": args.max_chunks_per_doc}
    summary["elapsed_seconds"] = time.perf_counter() - started
    write_json(args.output_dir / "view_rank_report.json", summary)
    print(json.dumps({"report": str((args.output_dir / "view_rank_report.json").resolve()), "queries": len(questions), "device": device, "gate": summary["gate"], "elapsed_seconds": summary["elapsed_seconds"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
