"""
Legal Corpus Loader and Preprocessor
=====================================
Handles loading and preprocessing of Vietnamese legal documents.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VietnameseTextProcessor:
    """Vietnamese text preprocessing utilities."""

    def __init__(self):
        # Common Vietnamese contractions and normalizations
        self.normalizations = [
            (r'\s+', ' '),           # Multiple spaces
            (r'[​-‍﻿]', ''),  # Zero-width chars
            (r'\n\s*\n\s*\n+', '\n\n'),  # Multiple newlines
        ]

        # Legal-specific patterns
        self.legal_patterns = {
            'article_ref': r'(?:Điều|Chương)\s+\d+[a-z]?(?:\s+(?:quy định|khoản|mục))?',
            'number': r'\d+(?:[.,]\d+)*(?:\s*(?:đồng|người|ngày|tháng|năm|vụ))?',
            'law_ref': r'(?:Nghị định|Luật|Thông tư|Nghị quyết|Quyết định)\s+(?:số\s+)?\d+[/]?\d*(?:[/]\d+)?(?:\s*[-\s]\s*[^\n]+)?',
            'date': r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}',
        }

    def normalize(self, text: str) -> str:
        """Normalize Vietnamese text."""
        text = text.strip()
        for pattern, replacement in self.normalizations:
            text = re.sub(pattern, replacement, text)
        return text

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Split by common sentence endings
        sentences = re.split(r'[.!?]+\s*', text)
        return [s.strip() for s in sentences if s.strip()]

    def split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]

    def extract_legal_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract legal entities from text."""
        entities = {}
        for entity_type, pattern in self.legal_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities[entity_type] = list(set(matches))
        return entities


class LegalCorpusLoader:
    """
    Loads and processes Vietnamese legal document corpus.

    Expected corpus structure from warmup.json:
    - question_id: str
    - question: str
    - answer: List[str] (document IDs)
    """

    def __init__(self, corpus_dir: Optional[Path] = None):
        self.corpus_dir = corpus_dir
        self.processor = VietnameseTextProcessor()
        self.documents = {}  # doc_id -> document content

    def load_corpus(self, corpus_path: Optional[Path] = None) -> Dict[str, str]:
        """
        Load corpus from directory or JSON file.

        Args:
            corpus_path: Path to corpus directory or JSON file

        Returns:
            Dictionary mapping document IDs to content
        """
        if corpus_path is None:
            corpus_path = self.corpus_dir

        if corpus_path is None:
            raise ValueError("No corpus path provided")

        corpus_path = Path(corpus_path)

        if corpus_path.is_file():
            return self._load_json_corpus(corpus_path)
        elif corpus_path.is_dir():
            return self._load_directory_corpus(corpus_path)
        else:
            raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    def _load_json_corpus(self, json_path: Path) -> Dict[str, str]:
        """Load corpus from JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        documents = {}
        for doc_id, doc_data in data.items():
            if isinstance(doc_data, str):
                documents[doc_id] = doc_data
            elif isinstance(doc_data, dict) and 'content' in doc_data:
                documents[doc_id] = doc_data['content']

        self.documents = documents
        logger.info(f"Loaded {len(documents)} documents from JSON")
        return documents

    def _load_directory_corpus(self, dir_path: Path) -> Dict[str, str]:
        """Load corpus from directory of text files."""
        documents = {}

        for file_path in dir_path.glob('*.txt'):
            doc_id = file_path.stem  # Use filename without extension as ID
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Normalize content
            content = self.processor.normalize(content)
            documents[doc_id] = content

        self.documents = documents
        logger.info(f"Loaded {len(documents)} documents from directory")
        return documents

    def chunk_documents(self, documents: Dict[str, str],
                        chunk_size: int = 512,
                        overlap: int = 50) -> List[Dict]:
        """
        Split documents into overlapping chunks.

        Args:
            documents: Dictionary of doc_id -> content
            chunk_size: Maximum characters per chunk
            overlap: Character overlap between chunks

        Returns:
            List of chunks with metadata
        """
        chunks = []

        for doc_id, content in documents.items():
            text_chunks = self._create_chunks(content, chunk_size, overlap)

            for idx, chunk_text in enumerate(text_chunks):
                chunks.append({
                    'chunk_id': f"{doc_id}_{idx}",
                    'doc_id': doc_id,
                    'text': chunk_text,
                    'position': idx,
                    'total_chunks': len(text_chunks),
                    'entities': self.processor.extract_legal_entities(chunk_text)
                })

        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks

    def _create_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Create overlapping chunks from text."""
        chunks = []

        # Split by paragraphs first
        paragraphs = self.processor.split_paragraphs(text)

        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para)

            if current_size + para_size <= chunk_size:
                current_chunk.append(para)
                current_size += para_size + 2  # +2 for newline
            else:
                # Save current chunk
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))

                # Start new chunk with overlap
                if overlap > 0 and current_chunk:
                    overlap_text = '\n'.join(current_chunk)
                    # Keep last part for overlap
                    overlap_size = 0
                    overlap_chunks = []
                    for pc in reversed(current_chunk):
                        overlap_chunks.insert(0, pc)
                        overlap_size += len(pc) + 2
                        if overlap_size >= overlap:
                            break
                    current_chunk = overlap_chunks
                    current_size = overlap_size
                else:
                    current_chunk = []
                    current_size = 0

                # Add current paragraph
                if para_size <= chunk_size:
                    current_chunk.append(para)
                    current_size += para_size + 2
                else:
                    # Split large paragraph
                    sub_chunks = self._split_long_paragraph(para, chunk_size)
                    current_chunk = [sub_chunks[-1]]
                    current_size = len(sub_chunks[-1]) + 2
                    chunks.extend(sub_chunks[:-1])

        # Add final chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks

    def _split_long_paragraph(self, text: str, max_size: int) -> List[str]:
        """Split a long paragraph into smaller chunks."""
        chunks = []

        # Try to split at sentence boundaries
        sentences = self.processor.split_sentences(text)

        current_chunk = []
        current_size = 0

        for sentence in sentences:
            if current_size + len(sentence) <= max_size:
                current_chunk.append(sentence)
                current_size += len(sentence)
            else:
                if current_chunk:
                    chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_size = len(sentence)

        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')

        return chunks

    def build_metadata(self, documents: Dict[str, str]) -> Dict:
        """Build metadata for documents."""
        metadata = {
            'total_documents': len(documents),
            'total_characters': sum(len(doc) for doc in documents.values()),
            'avg_doc_length': sum(len(doc) for doc in documents.values()) / len(documents) if documents else 0,
            'document_ids': list(documents.keys()),
        }

        return metadata


def load_warmup_data(warmup_path: Path) -> Dict:
    """Load warmup data to extract ground truth."""
    with open(warmup_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_unique_documents(warmup_data: Dict) -> set:
    """Extract unique document IDs from warmup answers."""
    doc_ids = set()
    for item in warmup_data.values():
        if 'answer' in item:
            doc_ids.update(item['answer'])
    return doc_ids


# Main execution for testing
if __name__ == "__main__":
    import sys

    # Test loading
    corpus_loader = LegalCorpusLoader()

    # Example usage
    test_doc = {
        "123": "Điều 1. Phạm vi điều chỉnh\nLuật này quy định về quản lý, sử dụng tài sản công.",
        "456": "Điều 2. Đối tượng áp dụng\nĐối tượng áp dụng Luật này bao gồm cơ quan nhà nước."
    }

    chunks = corpus_loader.chunk_documents(test_doc, chunk_size=200)
    print(f"Created {len(chunks)} test chunks")

    for chunk in chunks[:3]:
        print(f"Chunk {chunk['chunk_id']}: {chunk['text'][:100]}...")
