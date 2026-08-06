"""Portable defaults for Task 1; neural model paths are deliberately unset."""

from pathlib import Path


TASK_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_DIR = TASK_ROOT / "data" / "processed" / "index"
DEFAULT_OUTPUT = TASK_ROOT / "output" / "submission.zip"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
TOP_K_CANDIDATES = 60
TOP_K_FINAL = 5
RRF_K = 60
BM25_K1 = 1.5
BM25_B = 0.75

# Populate these with BTC-approved *local directories* through CLI flags.
EMBEDDING_MODEL = None
RERANKER_MODEL = None
