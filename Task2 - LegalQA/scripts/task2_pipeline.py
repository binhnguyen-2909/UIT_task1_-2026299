"""Offline hybrid retrieval and answer generation for DSC 2026 LegalQA."""

from __future__ import annotations

import math
import re
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from .corpus_loader import LegalQACorpusLoader
except ImportError:
    from corpus_loader import LegalQACorpusLoader


TOKEN_RE = re.compile(r"[0-9]+(?:[./-][0-9a-z]+)*|[^\W_]+", re.UNICODE)
SENTENCE_RE = re.compile(r"[^\n.!?]+(?:[.!?]+|(?=\n|$))", re.UNICODE)
LEGAL_REFERENCE_RE = re.compile(
    r"\b(?:điều|khoản|điểm|chương|mục)\s+[0-9]+[a-z]?\b", re.IGNORECASE
)
STOPWORDS = {
    "và", "của", "là", "các", "một", "những", "đó", "này", "tại",
    "từ", "với", "cho", "về", "do", "bởi", "thì", "mà", "cũng", "đang",
}


def tokenize(text: str, bigrams: bool = False) -> List[str]:
    normalized = unicodedata.normalize("NFC", str(text)).casefold()
    words = [word for word in TOKEN_RE.findall(normalized) if word not in STOPWORDS]
    return words + [f"{a}__{b}" for a, b in zip(words, words[1:])] if bigrams else words


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = float(k1), float(b)
        self.documents: List[Dict[str, Any]] = []
        self.lengths: List[int] = []
        self.average_length = 1.0
        self.postings: Dict[str, List[Tuple[int, int]]] = {}
        self.idf: Dict[str, float] = {}

    def fit(self, documents: List[Dict[str, Any]]) -> None:
        if not documents:
            raise ValueError("Cannot index an empty corpus")
        self.documents = [dict(document) for document in documents]
        postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.lengths = []
        for index, document in enumerate(self.documents):
            counts = Counter(tokenize(document["text"], bigrams=True))
            self.lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                postings[term].append((index, frequency))
        self.postings = dict(postings)
        count = len(self.documents)
        self.average_length = sum(self.lengths) / count or 1.0
        self.idf = {
            term: math.log(1.0 + (count - len(values) + 0.5) / (len(values) + 0.5))
            for term, values in self.postings.items()
        }

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        scores = [0.0] * len(self.documents)
        for term, query_frequency in Counter(tokenize(query, bigrams=True)).items():
            for index, frequency in self.postings.get(term, []):
                norm = 1.0 - self.b + self.b * self.lengths[index] / self.average_length
                scores[index] += (
                    query_frequency * self.idf[term] * frequency * (self.k1 + 1.0)
                    / (frequency + self.k1 * norm)
                )
        ranking = sorted(enumerate(scores), key=lambda pair: (-pair[1], pair[0]))
        return [
            {**self.documents[index], "score": float(score), "rank": rank, "method": "bm25"}
            for rank, (index, score) in enumerate(ranking[:top_k], 1) if score > 0.0
        ]


class SemanticIndex:
    """Local SentenceTransformer or deterministic TF-IDF semantic view."""

    def __init__(self, model_path: Optional[str] = None, device: str = "auto", batch_size: int = 32):
        self.model_path = model_path
        self.device = None if device == "auto" else device
        self.batch_size = batch_size
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Any = None
        self.sparse_embeddings: List[Dict[int, float]] = []
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.model: Any = None
        self.fallback = model_path is None
        if not self.fallback:
            local_path = Path(model_path)
            if not local_path.exists():
                raise FileNotFoundError(f"Embedding model must be a local directory: {model_path}")
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("Install sentence-transformers for neural retrieval") from exc
            self.model = SentenceTransformer(
                str(local_path), device=self.device, local_files_only=True
            )

    def fit(self, documents: List[Dict[str, Any]]) -> None:
        self.documents = [dict(document) for document in documents]
        texts = [str(document["text"]) for document in documents]
        if not self.fallback:
            try:
                import numpy as np
            except ImportError as exc:
                raise RuntimeError("NumPy is required only for neural embedding mode") from exc
            self.embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
            return
        frequencies: Counter[str] = Counter()
        for text in texts:
            frequencies.update(set(tokenize(text)))
        minimum = 1 if len(texts) < 20 else 2
        terms = sorted(term for term, frequency in frequencies.items() if frequency >= minimum)
        self.vocab = {term: index for index, term in enumerate(terms)}
        self.idf = {
            term: math.log((1.0 + len(texts)) / (1.0 + frequencies[term])) + 1.0
            for term in terms
        }
        self.sparse_embeddings = self._encode_tfidf_sparse(texts)

    def _encode_tfidf_sparse(self, texts: List[str]) -> List[Dict[int, float]]:
        """Sparse normalized TF-IDF; avoids a chunks × vocabulary dense matrix."""
        vectors: List[Dict[int, float]] = []
        for text in texts:
            vector: Dict[int, float] = {}
            for term, frequency in Counter(tokenize(text)).items():
                column = self.vocab.get(term)
                if column is not None:
                    vector[column] = (1.0 + math.log(frequency)) * self.idf[term]
            norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
            vectors.append({column: value / norm for column, value in vector.items()})
        return vectors

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if self.fallback and not self.sparse_embeddings:
            raise RuntimeError("Semantic index has not been built")
        if not self.fallback and self.embeddings is None:
            raise RuntimeError("Semantic index has not been built")
        if self.fallback:
            query_vector = self._encode_tfidf_sparse([query])[0]
            scores = [
                sum(value * document.get(column, 0.0) for column, value in query_vector.items())
                for document in self.sparse_embeddings
            ]
            indices = sorted(range(len(scores)), key=lambda index: (-scores[index], index))[:top_k]
        else:
            import numpy as np
            query_vector = self.model.encode(
                [query], convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True
            )[0]
            scores = self.embeddings @ query_vector
            indices = np.argsort(-scores, kind="stable")[:top_k]
        return [
            {
                **self.documents[int(index)],
                "score": float(scores[index]),
                "rank": rank,
                "method": "tfidf" if self.fallback else "dense",
            }
            for rank, index in enumerate(indices, 1) if scores[index] > 0.0
        ]


class CandidateReranker:
    def __init__(self, model_path: Optional[str] = None, device: str = "auto", batch_size: int = 16):
        self.fallback = model_path is None
        self.model: Any = None
        self.batch_size = batch_size
        if not self.fallback:
            local_path = Path(model_path)
            if not local_path.exists():
                raise FileNotFoundError(f"Reranker must be a local directory: {model_path}")
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError("Install sentence-transformers for neural reranking") from exc
            self.model = CrossEncoder(
                str(local_path), device=None if device == "auto" else device, local_files_only=True
            )

    def rank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        output = [dict(candidate) for candidate in candidates]
        if not output:
            return []
        if self.fallback:
            query_counts = Counter(tokenize(query))
            references = set(LEGAL_REFERENCE_RE.findall(query.casefold()))
            scores = []
            for candidate in output:
                passage_counts = Counter(tokenize(candidate["text"]))
                recall = sum((query_counts & passage_counts).values()) / max(1, sum(query_counts.values()))
                ref_match = float(any(ref in candidate["text"].casefold() for ref in references))
                scores.append(0.75 * recall + 0.15 * ref_match + 0.10 * math.tanh(candidate["score"]))
        else:
            scores = [float(score) for score in self.model.predict(
                [(query, candidate["text"]) for candidate in output],
                batch_size=self.batch_size,
                show_progress_bar=False,
            )]
        for candidate, score in zip(output, scores):
            candidate["rerank_score"] = score
        output.sort(key=lambda item: (-item["rerank_score"], -item["score"], item["chunk_id"]))
        for rank, candidate in enumerate(output, 1):
            candidate["rank"] = rank
        return output


@dataclass(frozen=True)
class SentenceCandidate:
    doc_id: str
    chunk_id: str
    chunk_rank: int
    sentence_index: int
    text: str
    score: float


class ExtractiveAnswerBuilder:
    """Select verbatim legal sentences and retain their source order."""

    def __init__(self, max_answer_chars: int = 8000, max_sentences: int = 12):
        self.max_answer_chars = max_answer_chars
        self.max_sentences = max_sentences

    @staticmethod
    def _sentences(text: str) -> List[str]:
        sentences = [match.group(0).strip() for match in SENTENCE_RE.finditer(text) if match.group(0).strip()]
        return sentences or ([text.strip()] if text.strip() else [])

    def build(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            raise ValueError("Cannot answer without retrieved context")
        query_counts = Counter(tokenize(question))
        query_refs = set(LEGAL_REFERENCE_RE.findall(question.casefold()))
        candidates: List[SentenceCandidate] = []
        for chunk_rank, chunk in enumerate(chunks, 1):
            sentences = self._sentences(chunk["text"])
            for sentence_index, sentence in enumerate(sentences):
                counts = Counter(tokenize(sentence))
                overlap = sum((query_counts & counts).values()) / max(1, sum(query_counts.values()))
                ref_match = float(any(ref in sentence.casefold() for ref in query_refs))
                if overlap <= 0 and ref_match <= 0:
                    continue
                score = overlap + 0.35 * ref_match + 0.10 / chunk_rank
                candidates.append(SentenceCandidate(
                    str(chunk["doc_id"]), str(chunk["chunk_id"]), chunk_rank,
                    sentence_index, sentence, score,
                ))
        if not candidates:
            return str(chunks[0]["text"])[: self.max_answer_chars].rstrip()

        # High precision seeds, then include one neighbour on either side to
        # recover conditions/exceptions that a sentence-only selector misses.
        seeds = sorted(candidates, key=lambda item: (-item.score, item.chunk_rank, item.sentence_index))
        chosen_keys = set()
        source_sentences: Dict[Tuple[str, int], str] = {}
        for chunk_rank, chunk in enumerate(chunks, 1):
            for index, sentence in enumerate(self._sentences(chunk["text"])):
                source_sentences[(str(chunk["chunk_id"]), index)] = sentence
        for seed in seeds[: max(1, self.max_sentences // 3)]:
            for index in (seed.sentence_index - 1, seed.sentence_index, seed.sentence_index + 1):
                key = (seed.chunk_id, index)
                if key in source_sentences:
                    chosen_keys.add((seed.chunk_rank, seed.doc_id, seed.chunk_id, index))

        ordered = sorted(chosen_keys)[: self.max_sentences]
        pieces: List[str] = []
        seen = set()
        current_length = 0
        for _, _, chunk_id, sentence_index in ordered:
            sentence = source_sentences[(chunk_id, sentence_index)].strip()
            canonical = " ".join(sentence.casefold().split())
            if canonical in seen:
                continue
            separator = " " if pieces else ""
            if current_length + len(separator) + len(sentence) > self.max_answer_chars:
                break
            pieces.append(sentence)
            seen.add(canonical)
            current_length += len(separator) + len(sentence)
        return " ".join(pieces) or str(chunks[0]["text"])[: self.max_answer_chars].rstrip()


class LocalGenerator:
    """Optional in-process causal LM; never starts or calls an API server."""

    def __init__(self, model_path: str, device: str = "auto", max_new_tokens: int = 1024):
        local_path = Path(model_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Generator must be a local directory: {model_path}")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install transformers and torch for local generation") from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(local_path), local_files_only=True)
        self.tokenizer.truncation_side = "left"
        kwargs: Dict[str, Any] = {"local_files_only": True, "torch_dtype": "auto"}
        if device == "auto":
            kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(str(local_path), **kwargs)
        if device not in ("auto", None):
            self.model.to(device)
        self.max_new_tokens = max_new_tokens

    def generate(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        context = "\n\n".join(f"[Nguồn {item['doc_id']}]\n{item['text']}" for item in chunks)
        messages = [
            {"role": "system", "content": (
                "Chỉ trả lời bằng thông tin trong ngữ cảnh pháp luật đã cho. "
                "Không bịa, giữ nguyên số điều/khoản, nêu đủ điều kiện và ngoại lệ."
            )},
            {"role": "user", "content": f"Ngữ cảnh:\n{context}\n\nCâu hỏi: {question}\nTrả lời:"},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                # Qwen3: deterministic non-thinking answer is shorter and does
                # not spend the token budget on hidden reasoning text.
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            except (TypeError, ValueError):
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = messages[0]["content"] + "\n\n" + messages[1]["content"]
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not answer:
            raise RuntimeError("Local generator returned an empty answer")
        return answer


class LegalQAPipeline:
    def __init__(
        self,
        embedding_model: Optional[str] = None,
        reranker_model: Optional[str] = None,
        generator_model: Optional[str] = None,
        device: str = "auto",
        candidates: int = 60,
        answer_chunks: int = 5,
        max_answer_chars: int = 8000,
        rrf_k: int = 60,
        sparse_weight: float = 1.0,
        semantic_weight: float = 1.0,
    ):
        if candidates <= 0 or answer_chunks <= 0 or max_answer_chars <= 0:
            raise ValueError("candidates, answer_chunks and max_answer_chars must be positive")
        if sparse_weight < 0 or semantic_weight < 0 or sparse_weight + semantic_weight <= 0:
            raise ValueError("At least one non-negative retrieval weight must be positive")
        self.sparse = BM25Index()
        self.semantic = SemanticIndex(embedding_model, device)
        self.reranker = CandidateReranker(reranker_model, device)
        self.generator = LocalGenerator(generator_model, device) if generator_model else None
        self.extractor = ExtractiveAnswerBuilder(max_answer_chars=max_answer_chars)
        self.candidates = candidates
        self.answer_chunks = answer_chunks
        self.rrf_k = rrf_k
        self.sparse_weight = float(sparse_weight)
        self.semantic_weight = float(semantic_weight)
        self.indexed = False

    def index(self, chunks: List[Dict[str, Any]]) -> None:
        self.sparse.fit(chunks)
        self.semantic.fit(chunks)
        self.indexed = True

    def _fuse(self, sparse: List[Dict[str, Any]], semantic: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        fused: Dict[str, Dict[str, Any]] = {}
        for method, weight, results in (
            ("bm25", self.sparse_weight, sparse),
            ("semantic", self.semantic_weight, semantic),
        ):
            for rank, candidate in enumerate(results, 1):
                key = str(candidate["chunk_id"])
                item = fused.setdefault(key, {
                    **candidate, "score": 0.0, "method": "rrf", "bm25_rank": None, "semantic_rank": None
                })
                item["score"] += weight / (self.rrf_k + rank)
                item[f"{method}_rank"] = rank
        output = list(fused.values())
        output.sort(key=lambda item: (-item["score"], item["chunk_id"]))
        return output

    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        if not self.indexed:
            raise RuntimeError("Pipeline corpus is not indexed")
        sparse = self.sparse.search(question, self.candidates) if self.sparse_weight > 0 else []
        semantic = self.semantic.search(question, self.candidates) if self.semantic_weight > 0 else []
        fused = self._fuse(sparse, semantic)
        if not fused:
            # Out-of-vocabulary safety path; surfaced in the candidate method
            # and inference audit rather than hidden.
            fused = [{
                **document,
                "score": 0.0,
                "rank": rank,
                "method": "zero-score-fallback",
                "bm25_rank": None,
                "semantic_rank": None,
            } for rank, document in enumerate(self.sparse.documents[: self.answer_chunks], 1)]
        return self.reranker.rank(question, fused[: max(20, self.answer_chunks * 4)])

    def answer(self, question: str) -> Tuple[str, Dict[str, Any]]:
        started = time.perf_counter()
        candidates = self.retrieve(question)
        selected: List[Dict[str, Any]] = []
        seen_chunks = set()
        for candidate in candidates:
            key = candidate["chunk_id"]
            if key in seen_chunks:
                continue
            selected.append(candidate)
            seen_chunks.add(key)
            if len(selected) >= self.answer_chunks:
                break
        if not selected:
            raise RuntimeError("No relevant passage was retrieved")
        answer = self.generator.generate(question, selected) if self.generator else self.extractor.build(question, selected)
        audit = {
            "latency_seconds": time.perf_counter() - started,
            "retrieved_chunks": len(candidates),
            "selected_chunks": [item["chunk_id"] for item in selected],
            "selected_doc_ids": [item["doc_id"] for item in selected],
            "semantic_backend": "tfidf" if self.semantic.fallback else "sentence-transformer",
            "reranker_backend": "lexical" if self.reranker.fallback else "cross-encoder",
            "answer_backend": "local-generator" if self.generator else "extractive",
            "zero_score_fallback": bool(selected and selected[0].get("method") == "zero-score-fallback"),
        }
        return answer, audit


def build_pipeline_from_corpus(
    corpus_path: Path,
    chunk_size: int = 1800,
    chunk_overlap: int = 250,
    **pipeline_options: Any,
) -> Tuple[LegalQAPipeline, LegalQACorpusLoader]:
    loader = LegalQACorpusLoader(corpus_path)
    loader.load()
    pipeline = LegalQAPipeline(**pipeline_options)
    pipeline.index(loader.chunks(chunk_size, chunk_overlap))
    return pipeline, loader
