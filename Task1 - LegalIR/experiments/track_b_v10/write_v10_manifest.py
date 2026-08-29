"""Write auditable hashes for the GitHub-v10 rebuild."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def file_entry(path: Path) -> Dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest(path)}


def optional(path: Path) -> Any:
    return file_entry(path) if path.is_file() else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    task = args.task_root
    raw = task / "data" / "raw" / "public_test"
    models = args.repo_root / "models"
    embedding = models / "bqbbao6_vietnamese-legal-embedding" / "model.safetensors"
    reranker = models / "thanhtantran_Vietnamese_Reranker" / "model.safetensors"
    source_files = [
        task / "scripts" / "main.py",
        task / "scripts" / "sqlite_retriever.py",
        task / "scripts" / "bm25_retriever.py",
        task / "scripts" / "reranker.py",
        task / "scripts" / "legal_structure.py",
        task / "model_registry.v10.example.json",
    ]
    source_hashes = {str(path.relative_to(args.repo_root)): file_entry(path) for path in source_files if path.is_file()}
    index_files = [
        args.index / "legalir.sqlite3",
        args.index / "legalir.config.json",
        args.index / "dense_embeddings.npy",
        args.index / "dense_embeddings.ids.json",
        args.index / "dense_embeddings.manifest.json",
        args.index / "question_dense_embeddings.npy",
        args.index / "question_dense_embeddings.ids.json",
        args.index / "question_dense_embeddings.manifest.json",
    ]
    payload = {
        "kind": "track_b_github_v10_rebuild_manifest",
        "github": {"remote": "https://github.com/bphong-cyrus/UIT-DSC.git", "commit": "df086ca2be49cc5d185dfd5a542c1009e1429748"},
        "v10_contract": {"chunk_strategy": "fixed", "chunk_size": 1200, "chunk_overlap": 150, "candidates": 100, "dense_weight": 0.05, "dense_rrf_k": 60, "question_dense_weight": 0.02, "question_dense_rrf_k": 60, "question_dense_candidates": 20, "rerank_candidates": 35, "rerank_passages_per_doc": 2, "reranker_rrf_weight": 0.8, "reranker_rrf_k": 5},
        "data": {name: optional(raw / name) for name in ("selected-contexts.zip", "train.json", "public-official.json")},
        "models": {"embedding_checkpoint": optional(embedding), "reranker_checkpoint": optional(reranker)},
        "source_files": source_hashes,
        "index_files": {str(path.relative_to(args.index)): optional(path) for path in index_files},
        "committed_v10_submission": optional(task / "output" / "public_submission_v10_ce_rrf_top35.zip"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "index_ready": all(value is not None for value in payload["index_files"].values())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
