"""Portable defaults for Task 2; no network/API-capable model is enabled."""

from pathlib import Path


TASK_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = TASK_ROOT / "output" / "submission.zip"

CHUNK_SIZE = 1800
CHUNK_OVERLAP = 250
TOP_K_CANDIDATES = 60
ANSWER_CHUNKS = 5
MAX_ANSWER_CHARS = 8000
RRF_K = 60

# Set only via CLI to downloaded local directories listed in the completed
# BTC-approved model registry.
EMBEDDING_MODEL = None
RERANKER_MODEL = None
GENERATOR_MODEL = None
