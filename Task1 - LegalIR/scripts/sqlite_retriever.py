"""Scalable SQLite FTS5 retriever for the 350MB+ official LegalIR corpus."""

from __future__ import annotations

import json
import itertools
import re
import sqlite3
import time
import unicodedata
import urllib.parse
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

try:
    from .bm25_retriever import VietnameseTokenizer, expand_query
    from .reranker import Reranker
    from .retrieval_types import QueryResult, RetrievalResult
except ImportError:
    from bm25_retriever import VietnameseTokenizer, expand_query
    from reranker import Reranker
    from retrieval_types import QueryResult, RetrievalResult


def normalize_question(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", str(text)).casefold().split())


def link_text(link: Any) -> str:
    """Turn a legal URL slug into searchable words without making a request."""
    if not isinstance(link, str) or not link:
        return ""
    parsed = urllib.parse.urlparse(link)
    decoded = urllib.parse.unquote(parsed.path)
    decoded = re.sub(r"[-_/]+", " ", decoded)
    decoded = re.sub(r"\b(?:html?|aspx?)\b", " ", decoded, flags=re.IGNORECASE)
    return " ".join(decoded.split())


def display_title(record: Dict[str, Any]) -> str:
    name = record.get("name")
    parts = [name.strip()] if isinstance(name, str) and name.strip() else []
    url_words = link_text(record.get("link"))
    if url_words:
        parts.append(url_words)
    return " | ".join(parts)


def iter_contexts(source: Path) -> Iterator[Dict[str, Any]]:
    """Yield valid official records without retaining the corpus in memory."""
    source = Path(source)
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source, "r") as archive:
            for item in sorted(archive.infolist(), key=lambda info: info.filename):
                if item.is_dir() or item.filename.startswith("__MACOSX/"):
                    continue
                if Path(item.filename).suffix.lower() != ".json":
                    continue
                record = json.loads(archive.read(item).decode("utf-8"))
                if isinstance(record, dict) and "id" in record:
                    yield record
        return
    if source.is_dir():
        for path in sorted(source.rglob("context_*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(record, dict) and "id" in record:
                yield record
        return
    raise ValueError(f"SQLite backend expects selected-contexts.zip or extracted directory: {source}")


def iter_text_chunks(text: str, chunk_size: int, overlap: int) -> Iterator[str]:
    """Yield bounded chunks without materialising a document-sized list."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")
    text = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return
    start = 0
    minimum_boundary = max(1, int(chunk_size * 0.6))
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = max(
                text.rfind("\n", start + minimum_boundary, end),
                text.rfind(" ", start + minimum_boundary, end),
            )
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)


def training_signals_from_data(
    data: Dict[str, Any],
) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, int]]:
    """Return doc aliases, exact question memory, and document label priors."""
    aliases: Dict[str, List[str]] = defaultdict(list)
    memory: Dict[str, List[str]] = defaultdict(list)
    priors: Dict[str, int] = defaultdict(int)
    for item in data.values():
        question = item.get("question")
        answers = item.get("answer")
        if not isinstance(question, str) or not isinstance(answers, list):
            continue
        normalized = normalize_question(question)
        for doc_id in map(str, answers):
            aliases[doc_id].append(question)
            priors[doc_id] += 1
            if doc_id not in memory[normalized]:
                memory[normalized].append(doc_id)
    alias_text = {
        doc_id: "\n".join(dict.fromkeys(questions))
        for doc_id, questions in aliases.items()
    }
    return alias_text, dict(memory), dict(priors)


def load_training_signals(path: Optional[Path]) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, int]]:
    if path is None:
        return {}, {}, {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Training data must be a JSON object keyed by question ID")
    return training_signals_from_data(data)


class SQLiteFTSRetriever:
    """Document-level BM25F index backed by SQLite FTS5.

    FTS5 streams to disk and therefore remains usable for the official corpus,
    unlike a dense ``chunks × vocabulary`` TF-IDF matrix.
    """

    BACKEND = "sqlite_chunk_fts5_v5_document_phrase_rrf"
    DOCUMENT_BASE_WEIGHT = 1.0
    DOCUMENT_PHRASE_WEIGHT = 0.25
    DOCUMENT_RRF_K = 0
    DOCUMENT_PHRASE_PAIR_LIMIT = 16
    DOCUMENT_PHRASE_RESULT_LIMIT = 80

    def __init__(
        self,
        db_path: Path,
        use_reranker: bool = True,
        reranker_model: Optional[str] = None,
        device: Optional[str] = None,
        top_k_initial: int = 100,
        use_query_expansion: bool = False,
        title_weight: float = 3.0,
        alias_weight: float = 2.0,
        passage_weight: float = 1.0,
        question_weight: float = 4.0,
        max_query_terms: int = 12,
        or_query_terms: int = 12,
        or_chunk_limit: int = 400,
        base_fusion_weight: float = 3.0,
        or_fusion_weight: float = 7.0,
        retrieval_rrf_k: int = 20,
        document_fts_db: Optional[Path] = None,
        chunk_size: int = 1400,
        chunk_overlap: int = 200,
    ):
        self.db_path = Path(db_path)
        self.use_reranker = use_reranker
        self.reranker_model = reranker_model
        self.device = device
        self.top_k_initial = int(top_k_initial)
        self.use_query_expansion = use_query_expansion
        self.title_weight = float(title_weight)
        self.alias_weight = float(alias_weight)
        self.passage_weight = float(passage_weight)
        self.question_weight = float(question_weight)
        if max_query_terms <= 0:
            raise ValueError("max_query_terms must be positive")
        self.max_query_terms = int(max_query_terms)
        if or_query_terms <= 0 or or_chunk_limit <= 0 or retrieval_rrf_k < 0:
            raise ValueError("OR retrieval limits must be positive and RRF k non-negative")
        if base_fusion_weight < 0 or or_fusion_weight < 0:
            raise ValueError("Fusion weights must be non-negative")
        self.or_query_terms = int(or_query_terms)
        self.or_chunk_limit = int(or_chunk_limit)
        self.base_fusion_weight = float(base_fusion_weight)
        self.or_fusion_weight = float(or_fusion_weight)
        self.retrieval_rrf_k = int(retrieval_rrf_k)
        self.document_fts_db = (
            Path(document_fts_db).resolve() if document_fts_db is not None else None
        )
        if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("Require chunk_size > chunk_overlap >= 0")
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.tokenizer = VietnameseTokenizer(bigrams=False)
        self.reranker = Reranker(reranker_model, device=device) if use_reranker else None
        self._connection: Optional[sqlite3.Connection] = None
        self._document_connection: Optional[sqlite3.Connection] = None
        self._document_count: Optional[int] = None
        self._question_ngram_cache: Optional[Dict[str, Any]] = None

    def _connect(self, readonly: bool = False) -> sqlite3.Connection:
        if readonly:
            uri = self.db_path.resolve().as_uri() + "?mode=ro"
            connection = sqlite3.connect(uri, uri=True)
        else:
            connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            if not self.db_path.exists():
                raise FileNotFoundError(f"SQLite index not found: {self.db_path}")
            self._connection = self._connect(readonly=True)
        return self._connection

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._document_connection is not None:
            self._document_connection.close()
            self._document_connection = None

    @property
    def document_connection(self) -> sqlite3.Connection:
        """Open the optional one-row-per-document FTS database lazily/read-only."""
        if self.document_fts_db is None:
            raise RuntimeError("Document FTS is not enabled")
        if self._document_connection is None:
            if not self.document_fts_db.is_file():
                raise FileNotFoundError(
                    f"Document FTS database not found: {self.document_fts_db}"
                )
            uri = self.document_fts_db.as_uri() + "?mode=ro"
            connection = sqlite3.connect(uri, uri=True)
            connection.row_factory = sqlite3.Row
            try:
                complete = connection.execute(
                    "SELECT value FROM metadata WHERE key = 'complete'"
                ).fetchone()
                count = int(
                    connection.execute("SELECT count(*) FROM documents").fetchone()[0]
                )
                if complete is None or str(complete[0]) != "1" or count <= 0:
                    raise ValueError(
                        "Document FTS database is incomplete or contains no documents"
                    )
            except (sqlite3.Error, ValueError):
                connection.close()
                raise
            self._document_connection = connection
            self._document_count = count
        return self._document_connection

    def document_fts_info(self) -> Dict[str, Any]:
        """Return auditable optional-database and validated fusion settings."""
        if self.document_fts_db is None:
            return {"enabled": False}
        _ = self.document_connection
        return {
            "enabled": True,
            "path": str(self.document_fts_db),
            "documents": self._document_count,
            "base_weight": self.DOCUMENT_BASE_WEIGHT,
            "phrase_weight": self.DOCUMENT_PHRASE_WEIGHT,
            "rrf_k": self.DOCUMENT_RRF_K,
            "pair_limit": self.DOCUMENT_PHRASE_PAIR_LIMIT,
            "result_limit": self.DOCUMENT_PHRASE_RESULT_LIMIT,
        }

    def build(self, corpus: Path, train_data: Optional[Path] = None) -> Dict[str, Any]:
        self.close()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        building = self.db_path.with_suffix(self.db_path.suffix + ".building")
        if building.exists():
            building.unlink()
        aliases, memory, priors = load_training_signals(train_data)
        connection = sqlite3.connect(building)
        try:
            connection.executescript("""
                PRAGMA journal_mode=OFF;
                PRAGMA synchronous=OFF;
                PRAGMA temp_store=MEMORY;
                PRAGMA cache_size=-262144;
                CREATE VIRTUAL TABLE chunks USING fts5(
                    doc_id UNINDEXED,
                    chunk_id UNINDEXED,
                    title,
                    passage,
                    tokenize='unicode61 remove_diacritics 0'
                );
                CREATE VIRTUAL TABLE aliases USING fts5(
                    doc_id UNINDEXED,
                    alias_text,
                    tokenize='unicode61 remove_diacritics 0'
                );
                CREATE VIRTUAL TABLE question_examples USING fts5(
                    question_id UNINDEXED,
                    answers UNINDEXED,
                    question,
                    tokenize='unicode61 remove_diacritics 0'
                );
                CREATE TABLE document_meta(doc_id TEXT PRIMARY KEY, title TEXT NOT NULL);
                CREATE TABLE document_prior(doc_id TEXT PRIMARY KEY, label_count INTEGER NOT NULL);
                CREATE TABLE question_memory(question TEXT PRIMARY KEY, answers TEXT NOT NULL);
                CREATE TABLE term_stats(term TEXT PRIMARY KEY, doc_freq INTEGER NOT NULL);
                CREATE TABLE manifest(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)
            inserted = 0
            inserted_chunks = 0
            empty_passages = 0
            started = time.perf_counter()
            for record in iter_contexts(Path(corpus)):
                doc_id = str(record["id"])
                passage = record.get("passage")
                passage = passage if isinstance(passage, str) else ""
                title = display_title(record)
                if not passage.strip():
                    empty_passages += 1
                    passage = title or f"văn bản {doc_id}"
                connection.execute(
                    "INSERT INTO document_meta(doc_id, title) VALUES (?, ?)",
                    (doc_id, title),
                )
                alias_text = aliases.get(doc_id)
                if alias_text:
                    connection.execute(
                        "INSERT INTO aliases(doc_id, alias_text) VALUES (?, ?)",
                        (doc_id, alias_text),
                    )
                for position, chunk in enumerate(
                    iter_text_chunks(passage, self.chunk_size, self.chunk_overlap)
                ):
                    connection.execute(
                        "INSERT INTO chunks(doc_id, chunk_id, title, passage) VALUES (?, ?, ?, ?)",
                        (doc_id, f"{doc_id}::chunk::{position}", title, chunk),
                    )
                    inserted_chunks += 1
                inserted += 1
                if inserted % 250 == 0:
                    connection.commit()
            connection.executemany(
                "INSERT INTO document_prior(doc_id, label_count) VALUES (?, ?)",
                sorted(priors.items()),
            )
            connection.executemany(
                "INSERT INTO question_memory(question, answers) VALUES (?, ?)",
                ((question, json.dumps(answers)) for question, answers in sorted(memory.items())),
            )
            training_records: Dict[str, Any] = {}
            if train_data:
                loaded_training = json.loads(Path(train_data).read_text(encoding="utf-8"))
                if isinstance(loaded_training, dict):
                    training_records = loaded_training
            connection.executemany(
                "INSERT INTO question_examples(question_id, answers, question) VALUES (?, ?, ?)",
                (
                    (str(qid), json.dumps(item.get("answer", [])), item["question"])
                    for qid, item in training_records.items()
                    if isinstance(item, dict) and isinstance(item.get("question"), str)
                ),
            )
            manifest = {
                "backend": self.BACKEND,
                "documents": inserted,
                "chunks": inserted_chunks,
                "empty_passages_with_surrogate": empty_passages,
                "training_alias_documents": len(aliases),
                "question_memory_entries": len(memory),
                "training_question_examples": len(training_records),
                "build_seconds": time.perf_counter() - started,
                "title_weight": self.title_weight,
                "alias_weight": self.alias_weight,
                "passage_weight": self.passage_weight,
                "question_weight": self.question_weight,
                "or_query_terms": self.or_query_terms,
                "or_chunk_limit": self.or_chunk_limit,
                "base_fusion_weight": self.base_fusion_weight,
                "or_fusion_weight": self.or_fusion_weight,
                "retrieval_rrf_k": self.retrieval_rrf_k,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "corpus": str(Path(corpus).resolve()),
                "train_data": str(Path(train_data).resolve()) if train_data else None,
            }
            connection.executemany(
                "INSERT INTO manifest(key, value) VALUES (?, ?)",
                ((key, json.dumps(value, ensure_ascii=False)) for key, value in manifest.items()),
            )
            connection.commit()
            connection.execute("INSERT INTO chunks(chunks) VALUES ('optimize')")
            connection.execute("INSERT INTO aliases(aliases) VALUES ('optimize')")
            connection.execute("INSERT INTO question_examples(question_examples) VALUES ('optimize')")
            connection.execute(
                "CREATE VIRTUAL TABLE chunk_vocab USING fts5vocab(chunks, 'row')"
            )
            connection.execute(
                "INSERT INTO term_stats(term, doc_freq) SELECT term, doc FROM chunk_vocab"
            )
            connection.execute("DROP TABLE chunk_vocab")
            connection.commit()
        finally:
            connection.close()
        building.replace(self.db_path)
        self._write_config()
        return manifest

    def _write_config(self) -> None:
        config = {
            "backend": self.BACKEND,
            "db_path": self.db_path.name,
            "use_reranker": self.use_reranker,
            "reranker_model": self.reranker_model,
            "top_k_initial": self.top_k_initial,
            "use_query_expansion": self.use_query_expansion,
            "title_weight": self.title_weight,
            "alias_weight": self.alias_weight,
            "passage_weight": self.passage_weight,
            "question_weight": self.question_weight,
            "max_query_terms": self.max_query_terms,
            "or_query_terms": self.or_query_terms,
            "or_chunk_limit": self.or_chunk_limit,
            "base_fusion_weight": self.base_fusion_weight,
            "or_fusion_weight": self.or_fusion_weight,
            "retrieval_rrf_k": self.retrieval_rrf_k,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        self.db_path.with_suffix(".config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(
        cls,
        db_path: Path,
        device: Optional[str] = None,
        document_fts_db: Optional[Path] = None,
    ) -> "SQLiteFTSRetriever":
        db_path = Path(db_path)
        config = json.loads(db_path.with_suffix(".config.json").read_text(encoding="utf-8"))
        return cls(
            db_path=db_path,
            use_reranker=config["use_reranker"],
            reranker_model=config.get("reranker_model"),
            device=device,
            top_k_initial=config["top_k_initial"],
            use_query_expansion=config["use_query_expansion"],
            title_weight=config["title_weight"],
            alias_weight=config["alias_weight"],
            passage_weight=config["passage_weight"],
            question_weight=config.get("question_weight", 4.0),
            max_query_terms=config.get("max_query_terms", 12),
            or_query_terms=config.get("or_query_terms", 12),
            or_chunk_limit=config.get("or_chunk_limit", 400),
            base_fusion_weight=config.get("base_fusion_weight", 3.0),
            or_fusion_weight=config.get("or_fusion_weight", 7.0),
            retrieval_rrf_k=config.get("retrieval_rrf_k", 20),
            document_fts_db=document_fts_db,
            chunk_size=config.get("chunk_size", 1400),
            chunk_overlap=config.get("chunk_overlap", 200),
        )

    def manifest(self) -> Dict[str, Any]:
        return {
            row["key"]: json.loads(row["value"])
            for row in self.connection.execute("SELECT key, value FROM manifest")
        }

    def replace_training_signals(self, data: Dict[str, Any]) -> Dict[str, int]:
        """Replace the small supervised layer without rebuilding passage chunks."""
        self.close()
        self._question_ngram_cache = None
        aliases, memory, priors = training_signals_from_data(data)
        connection = self._connect(readonly=False)
        try:
            connection.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS question_examples USING fts5(
                    question_id UNINDEXED,
                    answers UNINDEXED,
                    question,
                    tokenize='unicode61 remove_diacritics 0'
                )
            """)
            connection.execute("DELETE FROM aliases")
            connection.execute("DELETE FROM question_examples")
            connection.execute("DELETE FROM document_prior")
            connection.execute("DELETE FROM question_memory")
            connection.executemany(
                "INSERT INTO aliases(doc_id, alias_text) VALUES (?, ?)", sorted(aliases.items())
            )
            connection.executemany(
                "INSERT INTO document_prior(doc_id, label_count) VALUES (?, ?)", sorted(priors.items())
            )
            connection.executemany(
                "INSERT INTO question_memory(question, answers) VALUES (?, ?)",
                ((question, json.dumps(answers)) for question, answers in sorted(memory.items())),
            )
            connection.executemany(
                "INSERT INTO question_examples(question_id, answers, question) VALUES (?, ?, ?)",
                (
                    (str(qid), json.dumps(item.get("answer", [])), item["question"])
                    for qid, item in data.items()
                    if isinstance(item, dict) and isinstance(item.get("question"), str)
                ),
            )
            connection.execute("INSERT INTO aliases(aliases) VALUES ('optimize')")
            connection.execute("INSERT INTO question_examples(question_examples) VALUES ('optimize')")
            for key, value in (
                ("training_alias_documents", len(aliases)),
                ("question_memory_entries", len(memory)),
                ("training_question_examples", len(data)),
            ):
                connection.execute(
                    "INSERT OR REPLACE INTO manifest(key, value) VALUES (?, ?)",
                    (key, json.dumps(value)),
                )
            connection.commit()
        finally:
            connection.close()
        return {
            "training_alias_documents": len(aliases),
            "question_memory_entries": len(memory),
            "prior_documents": len(priors),
            "training_question_examples": len(data),
        }

    def _exact_memory(self, query: str) -> List[str]:
        row = self.connection.execute(
            "SELECT answers FROM question_memory WHERE question = ?", (normalize_question(query),)
        ).fetchone()
        return [str(value) for value in json.loads(row["answers"])] if row else []

    def _match_expression(self, query: str) -> Tuple[str, List[str]]:
        tokens = list(dict.fromkeys(self.tokenizer.tokenize(query)))
        if tokens:
            placeholders = ",".join("?" for _ in tokens)
            try:
                frequencies = {
                    str(row["term"]): int(row["doc_freq"])
                    for row in self.connection.execute(
                        f"SELECT term, doc_freq FROM term_stats WHERE term IN ({placeholders})",
                        tokens,
                    )
                }
            except sqlite3.OperationalError:
                frequencies = {token: 0 for token in tokens}
            # Terms absent from the corpus cannot retrieve anything. Among the
            # rest, rare terms carry more legal-document identity and avoid
            # scanning huge posting lists for generic wording.
            positions = {token: position for position, token in enumerate(tokens)}
            tokens = sorted(
                frequencies,
                key=lambda token: (frequencies[token], positions[token]),
            )[: self.max_query_terms]
        # Escape a token as an FTS phrase. Tokenizer strips quotes/operators,
        # but double escaping keeps this safe for future tokenizer changes.
        expression = " OR ".join(
            f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens
        )
        return expression, tokens

    @staticmethod
    def _focused_expressions(tokens: List[str]) -> List[str]:
        quoted = [f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens]
        if len(quoted) == 1:
            return quoted
        if len(quoted) >= 3:
            # Requiring three of the four rarest terms keeps posting-list
            # intersections small while tolerating one paraphrased word.
            return [
                " AND ".join(parts)
                for parts in itertools.combinations(quoted[:4], 3)
            ]
        return [" AND ".join(quoted)]

    def _chunk_candidates(self, tokens: List[str]) -> List[Dict[str, Any]]:
        expressions = self._focused_expressions(tokens)
        accumulated: Dict[str, Dict[str, Any]] = {}
        per_expression = max(20, self.top_k_initial)
        for expression in expressions:
            rows = list(self.connection.execute(
                """
                SELECT doc_id,
                       chunk_id,
                       title,
                       snippet(chunks, 3, '', '', ' … ', 96) AS evidence,
                       -bm25(chunks, 0.0, 0.0, ?, ?) AS score
                FROM chunks
                WHERE chunks MATCH ?
                ORDER BY bm25(chunks, 0.0, 0.0, ?, ?), doc_id, chunk_id
                LIMIT ?
                """,
                (
                    self.title_weight, self.passage_weight, expression,
                    self.title_weight, self.passage_weight, per_expression,
                ),
            ))
            for rank, row in enumerate(rows, 1):
                doc_id = str(row["doc_id"])
                rrf = 1.0 / (60.0 + rank)
                current = accumulated.get(doc_id)
                if current is None:
                    accumulated[doc_id] = {
                        "doc_id": doc_id,
                        "chunk_id": str(row["chunk_id"]),
                        "text": "\n".join(
                            value for value in (row["title"], row["evidence"]) if value
                        ),
                        "score": rrf,
                        "raw_fts_score": float(row["score"]),
                        "method": "sqlite_chunk_fts5",
                    }
                else:
                    # Best focused-query rank is more reliable than rewarding
                    # generic documents that happen to match every combination.
                    current["score"] = max(float(current["score"]), rrf)
                    if float(row["score"]) > current["raw_fts_score"]:
                        current["raw_fts_score"] = float(row["score"])
                        current["chunk_id"] = str(row["chunk_id"])
                        current["text"] = "\n".join(
                            value for value in (row["title"], row["evidence"]) if value
                        )
        # Very short/OOV-heavy questions can yield too few triple matches.
        # Broaden only then, anchoring each pair on the rarest indexed term.
        if len(accumulated) < 10 and len(tokens) >= 2:
            anchor = f'"{tokens[0].replace(chr(34), chr(34) * 2)}"'
            for token in tokens[1:4]:
                other = f'"{token.replace(chr(34), chr(34) * 2)}"'
                rows = list(self.connection.execute(
                    """
                    SELECT doc_id, chunk_id, title,
                           snippet(chunks, 3, '', '', ' … ', 96) AS evidence,
                           -bm25(chunks, 0.0, 0.0, ?, ?) AS score
                    FROM chunks WHERE chunks MATCH ?
                    ORDER BY bm25(chunks, 0.0, 0.0, ?, ?), doc_id, chunk_id LIMIT ?
                    """,
                    (
                        self.title_weight, self.passage_weight,
                        f"{anchor} AND {other}",
                        self.title_weight, self.passage_weight, per_expression,
                    ),
                ))
                for rank, row in enumerate(rows, 1):
                    doc_id = str(row["doc_id"])
                    if doc_id not in accumulated:
                        accumulated[doc_id] = {
                            "doc_id": doc_id,
                            "chunk_id": str(row["chunk_id"]),
                            "text": "\n".join(
                                value for value in (row["title"], row["evidence"]) if value
                            ),
                            "score": 0.5 / (60.0 + rank),
                            "raw_fts_score": float(row["score"]),
                            "method": "sqlite_chunk_fts5_broad_fallback",
                        }
        ranked = sorted(
            accumulated.values(),
            key=lambda item: (-float(item["score"]), -float(item["raw_fts_score"]), item["doc_id"]),
        )
        return ranked[: self.top_k_initial]

    def _fuse_alias_candidates(
        self, tokens: List[str], candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not tokens or not self.manifest().get("training_alias_documents", 0):
            return candidates
        by_id = {item["doc_id"]: item for item in candidates}
        for expression in self._focused_expressions(tokens):
            rows = list(self.connection.execute(
                """
                SELECT doc_id,
                       snippet(aliases, 1, '', '', ' … ', 64) AS evidence,
                       -bm25(aliases, 0.0, 1.0) AS score
                FROM aliases WHERE aliases MATCH ?
                ORDER BY bm25(aliases, 0.0, 1.0), doc_id LIMIT ?
                """,
                (expression, self.top_k_initial),
            ))
            for rank, row in enumerate(rows, 1):
                doc_id = str(row["doc_id"])
                alias_rrf = self.alias_weight / (60.0 + rank)
                current = by_id.get(doc_id)
                if current is None:
                    by_id[doc_id] = {
                        "doc_id": doc_id,
                        "chunk_id": doc_id,
                        "text": str(row["evidence"] or ""),
                        "score": alias_rrf,
                        "raw_fts_score": float(row["score"]),
                        "method": "sqlite_training_alias",
                    }
                elif alias_rrf > float(current["score"]):
                    current["score"] = alias_rrf
                    current["method"] = "sqlite_chunk_alias_fusion"
        return sorted(
            by_id.values(),
            key=lambda item: (-float(item["score"]), -float(item["raw_fts_score"]), item["doc_id"]),
        )[: self.top_k_initial]

    def _fuse_question_neighbors(
        self, query: str, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Fuse labels from lexically nearest labelled training questions."""
        if not self.manifest().get("training_question_examples", 0):
            return candidates
        tokens = list(dict.fromkeys(self.tokenizer.tokenize(query)))
        if not tokens:
            return candidates
        expression = " OR ".join(
            f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens
        )
        try:
            rows = list(self.connection.execute(
                """
                SELECT question_id, answers, question,
                       -bm25(question_examples, 0.0, 0.0, 1.0) AS score
                FROM question_examples WHERE question_examples MATCH ?
                ORDER BY bm25(question_examples, 0.0, 0.0, 1.0), question_id
                LIMIT ?
                """,
                (expression, self.top_k_initial),
            ))
        except sqlite3.OperationalError:
            return candidates

        by_id = {item["doc_id"]: item for item in candidates}
        for rank, row in enumerate(rows, 1):
            neighbor_rrf = self.question_weight / (20.0 + rank)
            for doc_id in map(str, json.loads(row["answers"])):
                current = by_id.get(doc_id)
                if current is None:
                    by_id[doc_id] = {
                        "doc_id": doc_id,
                        "chunk_id": str(row["question_id"]),
                        "text": str(row["question"]),
                        "score": neighbor_rrf,
                        "raw_fts_score": float(row["score"]),
                        "method": "sqlite_train_question_knn",
                    }
                elif neighbor_rrf > float(current["score"]):
                    current["score"] = neighbor_rrf
                    current["raw_fts_score"] = max(
                        float(current["raw_fts_score"]), float(row["score"])
                    )
                    current["method"] = "sqlite_question_knn_fusion"
        return sorted(
            by_id.values(),
            key=lambda item: (-float(item["score"]), -float(item["raw_fts_score"]), item["doc_id"]),
        )[: self.top_k_initial]

    @staticmethod
    def _accent_fold(text: str) -> str:
        folded = "".join(
            character
            for character in unicodedata.normalize("NFD", str(text).casefold())
            if unicodedata.category(character) != "Mn"
        ).replace("đ", "d")
        return " ".join(re.findall(r"[a-z0-9]+", folded))

    @classmethod
    def _character_features(cls, text: str) -> set[str]:
        normalized = f" {cls._accent_fold(text)} "
        return {
            f"{width}:{normalized[index:index + width]}"
            for width in (3, 4)
            for index in range(max(0, len(normalized) - width + 1))
        }

    def _load_question_ngram_index(self) -> Dict[str, Any]:
        if self._question_ngram_cache is not None:
            return self._question_ngram_cache
        rows: List[Dict[str, Any]] = []
        postings: Dict[str, List[int]] = defaultdict(list)
        try:
            examples = self.connection.execute(
                "SELECT question_id, answers, question FROM question_examples"
            )
            for row in examples:
                features = self._character_features(str(row["question"]))
                index = len(rows)
                rows.append({
                    "question_id": str(row["question_id"]),
                    "answers": [str(value) for value in json.loads(row["answers"])],
                    "question": str(row["question"]),
                    "features": features,
                })
                for feature in features:
                    postings[feature].append(index)
        except sqlite3.OperationalError:
            pass
        self._question_ngram_cache = {"rows": rows, "postings": postings}
        return self._question_ngram_cache

    def _fuse_character_neighbors(
        self, query: str, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Viettel-style accent-folded 3/4-gram Jaccard, validated on LegalIR."""
        index = self._load_question_ngram_index()
        rows: List[Dict[str, Any]] = index["rows"]
        if not rows:
            return candidates
        query_features = self._character_features(query)
        intersections: Counter[int] = Counter()
        for feature in query_features:
            intersections.update(index["postings"].get(feature, ()))
        neighbors: List[Tuple[float, int]] = []
        for row_index, intersection in intersections.items():
            denominator = len(query_features) + len(rows[row_index]["features"]) - intersection
            neighbors.append((intersection / denominator if denominator else 0.0, row_index))
        neighbors.sort(key=lambda item: (-item[0], item[1]))

        ngram_docs: Dict[str, Dict[str, Any]] = {}
        for similarity, row_index in neighbors[:80]:
            row = rows[row_index]
            for doc_id in row["answers"]:
                current = ngram_docs.get(doc_id)
                if current is None or similarity > current["similarity"]:
                    ngram_docs[doc_id] = {
                        "similarity": similarity,
                        "question_id": row["question_id"],
                        "question": row["question"],
                    }
        ngram_ranked = sorted(
            ngram_docs,
            key=lambda doc_id: (-ngram_docs[doc_id]["similarity"], doc_id),
        )[: self.top_k_initial]

        by_id = {item["doc_id"]: dict(item) for item in candidates}
        combined: Dict[str, float] = {
            item["doc_id"]: 3.0 / (5.0 + rank)
            for rank, item in enumerate(candidates, 1)
        }
        for rank, doc_id in enumerate(ngram_ranked, 1):
            combined[doc_id] = combined.get(doc_id, 0.0) + 1.0 / (5.0 + rank)
            if doc_id not in by_id:
                source = ngram_docs[doc_id]
                by_id[doc_id] = {
                    "doc_id": doc_id,
                    "chunk_id": source["question_id"],
                    "text": source["question"],
                    "score": 0.0,
                    "raw_fts_score": source["similarity"],
                    "method": "train_question_char_ngram",
                }
            else:
                by_id[doc_id]["method"] = "sqlite_char_ngram_fusion"
        for doc_id, score in combined.items():
            by_id[doc_id]["score"] = score
        return sorted(
            by_id.values(),
            key=lambda item: (-float(item["score"]), item["doc_id"]),
        )[: self.top_k_initial]

    def _or_bm25_candidates(self, tokens: List[str]) -> List[Dict[str, Any]]:
        """Retrieve a high-recall list with a bounded OR query over passage chunks.

        This deliberately remains separate from the focused-query branch.  The
        two rankings make different errors, which is what made their RRF fusion
        substantially stronger on the labelled holdout.
        """
        selected_tokens = tokens[: self.or_query_terms]
        if not selected_tokens:
            return []
        expression = " OR ".join(
            f'"{token.replace(chr(34), chr(34) * 2)}"' for token in selected_tokens
        )
        try:
            rows = self.connection.execute(
                """
                SELECT doc_id, -bm25(chunks, 0.0, 0.0, 3.0, 1.0) AS score
                FROM chunks WHERE chunks MATCH ?
                ORDER BY bm25(chunks, 0.0, 0.0, 3.0, 1.0), doc_id
                LIMIT ?
                """,
                (expression, self.or_chunk_limit),
            )
            unique: Dict[str, Dict[str, Any]] = {}
            for row in rows:
                doc_id = str(row["doc_id"])
                if doc_id in unique:
                    continue
                unique[doc_id] = {
                    "doc_id": doc_id,
                    "chunk_id": doc_id,
                    "text": "",
                    "score": float(row["score"]),
                    "raw_fts_score": float(row["score"]),
                    "method": "sqlite_or_bm25",
                }
                if len(unique) >= self.top_k_initial:
                    break
            return list(unique.values())
        except sqlite3.OperationalError:
            return []

    def _fuse_or_bm25(
        self,
        base_candidates: List[Dict[str, Any]],
        or_candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Fuse the validated supervised/focused and OR-BM25 rankings by RRF."""
        by_id = {item["doc_id"]: dict(item) for item in base_candidates}
        fused: Dict[str, float] = {}
        for weight, ranking in (
            (self.base_fusion_weight, base_candidates),
            (self.or_fusion_weight, or_candidates),
        ):
            for rank, item in enumerate(ranking, 1):
                doc_id = item["doc_id"]
                fused[doc_id] = fused.get(doc_id, 0.0) + weight / (
                    self.retrieval_rrf_k + rank
                )
                if doc_id not in by_id:
                    by_id[doc_id] = dict(item)
                elif ranking is or_candidates:
                    by_id[doc_id]["method"] = "sqlite_supervised_or_rrf"
                    by_id[doc_id]["raw_fts_score"] = max(
                        float(by_id[doc_id].get("raw_fts_score", 0.0)),
                        float(item.get("raw_fts_score", 0.0)),
                    )
        for doc_id, score in fused.items():
            by_id[doc_id]["score"] = score
        return sorted(
            by_id.values(),
            key=lambda item: (-float(item["score"]), item["doc_id"]),
        )[: self.top_k_initial]

    @staticmethod
    def _quoted_phrase(first: str, second: str) -> str:
        first = first.replace(chr(34), chr(34) * 2)
        second = second.replace(chr(34), chr(34) * 2)
        return f'"{first} {second}"'

    def _document_phrase_candidates(self, query: str) -> List[Dict[str, Any]]:
        """Rank documents containing exact adjacent query-token pairs.

        The expression, BM25 weights and limits intentionally match the
        leak-free v3 holdout benchmark.  This branch is disabled unless an
        explicit ``--document-fts-db`` is supplied.
        """
        if self.document_fts_db is None:
            return []
        tokens = list(dict.fromkeys(self.tokenizer.tokenize(query)))
        if len(tokens) < 2:
            return []
        pairs: Sequence[Tuple[str, str]] = list(
            dict.fromkeys(zip(tokens, tokens[1:]))
        )[: self.DOCUMENT_PHRASE_PAIR_LIMIT]
        expression = " OR ".join(
            self._quoted_phrase(first, second) for first, second in pairs
        )
        rows = self.document_connection.execute(
            """
            SELECT doc_id, -bm25(documents, 0.0, 3.0, 1.0) AS score
            FROM documents
            WHERE documents MATCH ?
            ORDER BY bm25(documents, 0.0, 3.0, 1.0), doc_id
            LIMIT ?
            """,
            (expression, self.DOCUMENT_PHRASE_RESULT_LIMIT),
        )
        return [
            {
                "doc_id": str(row["doc_id"]),
                "chunk_id": str(row["doc_id"]),
                "text": "",
                "score": float(row["score"]),
                "raw_fts_score": float(row["score"]),
                "method": "document_phrase_fts",
            }
            for row in rows
        ]

    def _fuse_document_phrase(
        self,
        base_candidates: List[Dict[str, Any]],
        phrase_candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Fuse the complete v2 ranking with document phrase RRF (1:0.25, k=0)."""
        if not phrase_candidates:
            return base_candidates
        by_id = {item["doc_id"]: dict(item) for item in base_candidates}
        fused: Dict[str, float] = defaultdict(float)
        for weight, ranking in (
            (self.DOCUMENT_BASE_WEIGHT, base_candidates),
            (self.DOCUMENT_PHRASE_WEIGHT, phrase_candidates),
        ):
            for rank, item in enumerate(ranking, 1):
                doc_id = item["doc_id"]
                fused[doc_id] += weight / (self.DOCUMENT_RRF_K + rank)
                if doc_id not in by_id:
                    by_id[doc_id] = dict(item)
        for doc_id, score in fused.items():
            by_id[doc_id]["score"] = score
            by_id[doc_id]["method"] = "sqlite_v2_document_phrase_rrf"
        return sorted(
            by_id.values(),
            key=lambda item: (-float(item["score"]), item["doc_id"]),
        )[: self.top_k_initial]

    def retrieve(self, query: str, query_id: str = "", top_k: int = 5) -> QueryResult:
        if not 1 <= top_k <= 5:
            raise ValueError("BTC Task 1 permits 1 to 5 predictions")
        started = time.perf_counter()
        memory_answers = self._exact_memory(query)

        processed = expand_query(query) if self.use_query_expansion else query
        _, search_terms = self._match_expression(processed)
        candidates = self._chunk_candidates(search_terms) if search_terms else []
        candidates = self._fuse_alias_candidates(search_terms, candidates)
        candidates = self._fuse_question_neighbors(query, candidates)
        candidates = self._fuse_character_neighbors(query, candidates)
        or_candidates = self._or_bm25_candidates(search_terms)
        candidates = self._fuse_or_bm25(candidates, or_candidates)
        document_phrase_candidates = self._document_phrase_candidates(query)
        candidates = self._fuse_document_phrase(
            candidates, document_phrase_candidates
        )
        if memory_answers:
            memory_ids = set(memory_answers)
            exact = [{
                "doc_id": doc_id,
                "chunk_id": doc_id,
                "text": "exact normalized training-question match",
                "score": 1_000_000.0 - rank,
                "raw_fts_score": 1_000_000.0 - rank,
                "method": "train_exact_memory",
            } for rank, doc_id in enumerate(memory_answers)]
            candidates = exact + [
                item for item in candidates if item["doc_id"] not in memory_ids
            ]
        if self.reranker is not None:
            candidates = self.reranker.rerank(query, candidates, top_k=max(top_k, 20))

        used_fallback = False
        if not candidates:
            used_fallback = True
            fallback_rows = list(self.connection.execute(
                "SELECT doc_id, label_count FROM document_prior ORDER BY label_count DESC, doc_id LIMIT ?",
                (top_k,),
            ))
            if not fallback_rows:
                fallback_rows = list(self.connection.execute(
                    "SELECT doc_id, 0 AS label_count FROM document_meta ORDER BY doc_id LIMIT ?",
                    (top_k,),
                ))
            candidates = [{
                "doc_id": str(row["doc_id"]), "chunk_id": str(row["doc_id"]),
                "text": "", "score": 0.0, "rank": rank,
                "method": "zero-score-fallback", "rerank_score": None,
            } for rank, row in enumerate(fallback_rows, 1)]

        selected = candidates[:top_k]
        selected_ids = {item["doc_id"] for item in selected}
        if len(selected) < top_k:
            fill_rows = list(self.connection.execute(
                """
                SELECT m.doc_id, COALESCE(p.label_count, 0) AS label_count
                FROM document_meta m LEFT JOIN document_prior p ON p.doc_id = m.doc_id
                ORDER BY label_count DESC, m.doc_id
                """
            ))
            for row in fill_rows:
                doc_id = str(row["doc_id"])
                if doc_id in selected_ids:
                    continue
                selected.append({
                    "doc_id": doc_id, "chunk_id": doc_id, "text": "",
                    "score": 0.0, "method": "recall_fill", "rerank_score": None,
                })
                selected_ids.add(doc_id)
                if len(selected) == top_k:
                    break
        results = [RetrievalResult(
            doc_id=item["doc_id"],
            text=item.get("text", ""),
            score=float(item.get("score", 0.0)),
            method=item.get("method", "sqlite_fts5"),
            rank=rank,
            chunk_id=item.get("chunk_id", item["doc_id"]),
            rerank_score=(
                float(item["rerank_score"]) if item.get("rerank_score") is not None else None
            ),
        ) for rank, item in enumerate(selected, 1)]
        return QueryResult(
            query_id=str(query_id), query=query, results=results,
            processing_time=time.perf_counter() - started,
            metadata={
                "backend": self.BACKEND,
                "candidates": len(candidates),
                "exact_memory": bool(memory_answers),
                "zero_score_fallback": used_fallback,
                "search_terms": search_terms,
                "or_candidates": len(or_candidates),
                "document_phrase": {
                    "enabled": self.document_fts_db is not None,
                    "candidates": len(document_phrase_candidates),
                    "base_weight": self.DOCUMENT_BASE_WEIGHT,
                    "phrase_weight": self.DOCUMENT_PHRASE_WEIGHT,
                    "rrf_k": self.DOCUMENT_RRF_K,
                },
                "fusion": {
                    "base_weight": self.base_fusion_weight,
                    "or_weight": self.or_fusion_weight,
                    "rrf_k": self.retrieval_rrf_k,
                },
            },
        )
