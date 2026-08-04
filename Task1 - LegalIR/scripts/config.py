"""
Configuration for Legal Information Retrieval Pipeline
======================================================
Task: Given a question, return document IDs containing relevant legal information.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
KB_DIR = BASE_DIR / "kb"
OUTPUT_DIR = BASE_DIR / "output"

# Legal IR specific paths
CORPUS_DIR = RAW_DIR / "corpus"  # Legal documents collection
INDEX_DIR = PROCESSED_DIR / "index"
EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"

# Model configuration - must be < 4 billion parameters
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # ~118M params
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"  # ~118M params

# Alternative models for Vietnamese legal (if GPU available)
VIETNAMESE_EMBEDDING_MODELS = {
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "xlm_roberta": "sentence-transformers/xlm-roberta-base",  # ~278M params
}

# Retrieval configuration
RETRIEVAL_CONFIG = {
    "top_k_initial": 100,      # Initial retrieval candidates
    "top_k_reranked": 20,       # After reranking
    "top_k_final": 5,           # Final output
    "similarity_threshold": 0.3,  # Minimum similarity score
    "use_bm25": True,
    "use_dense": True,
    "use_hybrid": True,
    "alpha": 0.5,  # Weight for dense (1-alpha for sparse)
}

# Legal domain specific
LEGAL_STOPWORDS = [
    'và', 'của', 'là', 'có', 'được', 'theo', 'nào', 'này', 'năm', 'nào',
    'thế', 'nào', 'ra', 'sao', 'như', 'thế', 'nào', 'đó', 'này', 'với',
    'trong', 'cho', 'để', 'tại', 'từ', 'hoặc', 'về', 'pháp', 'luật',
    'quy', 'định', 'hành', 'viên', 'người', 'doanh', 'nghiệp', 'cơ',
    'quan', 'nhà', 'nước', 'thực', 'hiện', 'hồ', 'sơ', 'thủ', 'tục'
]

# Legal entity patterns for query expansion
LEGAL_ENTITY_PATTERNS = {
    "muc_phat": ["phạt", "tiền", "xử phạt", "vi phạm"],
    "thu_tuc": ["thủ tục", "trình tự", "hồ sơ", "đăng ký", "cấp"],
    "tham_quyen": ["thẩm quyền", "quyết định", "phê duyệt", "ban hành"],
    "quyen_han": ["quyền", "hạn", "trách nhiệm", "nghĩa vụ"],
    "chinh_sach": ["chính sách", "ưu đãi", "hỗ trợ", "trợ cấp"],
}

# BM25 parameters
BM25_CONFIG = {
    "k1": 1.5,
    "b": 0.75,
    "tokenizer": "vi_tokenizer",  # Vietnamese tokenization
}

# Processing configuration
CHUNK_SIZE = 512          # Characters per chunk
CHUNK_OVERLAP = 50         # Overlap between chunks
MAX_SEQ_LENGTH = 512       # Max tokens for embedding

# Output format
OUTPUT_FORMAT = {
    "include_scores": True,
    "include_metadata": True,
    "top_k": 5,
}

# Paths configuration
PATHS = {
    "corpus": RAW_DIR / "corpus",
    "index": INDEX_DIR,
    "embeddings": EMBEDDINGS_DIR,
    "bm25_index": PROCESSED_DIR / "bm25_index.pkl",
    "faiss_index": PROCESSED_DIR / "faiss_index.bin",
    "docstore": PROCESSED_DIR / "docstore.json",
    "metadata": PROCESSED_DIR / "metadata.json",
}

# Create directories
for path in PATHS.values():
    path.parent.mkdir(parents=True, exist_ok=True)
