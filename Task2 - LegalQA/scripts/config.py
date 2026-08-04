"""
Task 2 - LegalQA Configuration
================================
Centralized configuration cho Task 2 pipeline.
"""

import os
from pathlib import Path


class Config:
    """Task 2 Configuration."""

    # Paths
    BASE_DIR = Path(__file__).parent.parent  # Task2 - LegalQA/
    WARMUP_PATH = BASE_DIR / "warmup.json"
    OUTPUT_DIR = BASE_DIR / "output"
    DATA_DIR = BASE_DIR / "data"
    RAW_DIR = DATA_DIR / "raw"
    CORPUS_DIR = RAW_DIR / "corpus"
    PROCESSED_DIR = DATA_DIR / "processed"
    INDEX_DIR = PROCESSED_DIR / "index"
    EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"

    # Submission
    SUBMISSION_JSON = OUTPUT_DIR / "submission.json"
    SUBMISSION_ZIP = OUTPUT_DIR / "submission.zip"

    # Model settings (giới hạn < 4B params)
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DIM = 384
    MAX_SEQ_LENGTH = 512

    # Cross-encoder for reranking
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

    # BM25 parameters
    BM25_K1 = 1.5
    BM25_B = 0.75

    # RRF fusion
    RRF_K = 60

    # Retrieval
    TOP_K_RETRIEVAL = 20  # Initial retrieval
    TOP_K_RERANK = 5     # After reranking

    # Answer synthesis
    MAX_ANSWER_LENGTH = 2000  # chars
    MIN_SPAN_LENGTH = 50      # chars

    # Chunking
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50

    # Create directories
    @classmethod
    def setup(cls):
        """Create necessary directories."""
        for dir_path in [
            cls.OUTPUT_DIR,
            cls.DATA_DIR,
            cls.RAW_DIR,
            cls.CORPUS_DIR,
            cls.PROCESSED_DIR,
            cls.INDEX_DIR,
            cls.EMBEDDINGS_DIR,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Vietnamese stopwords
VIETNAMESE_STOPWORDS = {
    'và', 'của', 'là', 'có', 'được', 'theo', 'nào', 'này', 'năm',
    'thế', 'ra', 'sao', 'như', 'đó', 'với', 'trong', 'cho',
    'để', 'tại', 'từ', 'hoặc', 'về', 'pháp', 'luật', 'quy', 'định',
    'hành', 'viên', 'người', 'doanh', 'nghiệp', 'cơ', 'quan', 'nhà',
    'nước', 'thực', 'hiện', 'hồ', 'sơ', 'thủ', 'tục', 'một', 'các',
    'khác', 'vào', 'bộ', 'sở', 'đến', 'khi', 'trên', 'dưới', 'sau',
    'trước', 'giữa', 'không', 'vì', 'nhưng', 'nếu', 'thì', 'đâu',
    'ai', 'ở', 'đây', 'nên', 'hay', 'lại', 'rồi', 'cứ',
    'nhất', 'hết', 'mà', 'cũng', 'chỉ', 'còn', 'lúc', 'bao', 'giờ',
    'muốn', 'cần', 'phải', 'bị', 'gì', 'thế', 'nào'
}

# Legal document types
LEGAL_DOC_TYPES = {
    'luật': 'Law (Quốc hội)',
    'bộ luật': 'Code (Quốc hội)',
    'nghị định': 'Decree (Chính phủ)',
    'thông tư': 'Circular (Bộ/ngành)',
    'quyết định': 'Decision (Thủ tướng/Bộ trưởng)',
    'nghị quyết': 'Resolution (Chính phủ/HĐND)',
    'pháp lệnh': 'Ordinance (UBTVQH)',
}

# Legal keywords for query analysis
LEGAL_KEYWORDS = {
    'điều', 'khoản', 'mục', 'chương',
    'quyền', 'thẩm quyền', 'trách nhiệm', 'nghĩa vụ',
    'thủ tục', 'trình tự', 'hồ sơ', 'đăng ký', 'cấp phép',
    'xử phạt', 'phạt tiền', 'vi phạm', 'bồi thường',
    'doanh nghiệp', 'công ty', 'tổ chức', 'pháp nhân',
    'thuế', 'tài chính', 'ngân sách',
    'đất', 'nhà ở', 'quyền sử dụng đất',
    'giao thông', 'phương tiện', 'lái xe',
    'y tế', 'bệnh viện', 'bảo hiểm y tế',
    'giáo dục', 'đào tạo', 'học sinh', 'sinh viên',
}


if __name__ == "__main__":
    # Test config
    Config.setup()
    print(f"Base dir: {Config.BASE_DIR}")
    print(f"Warmup: {Config.WARMUP_PATH.exists()}")
    print(f"Output dir: {Config.OUTPUT_DIR.exists()}")
    print(f"Embedding model: {Config.EMBEDDING_MODEL}")
    print(f"Model params: ~118M (< 4B ✅)")