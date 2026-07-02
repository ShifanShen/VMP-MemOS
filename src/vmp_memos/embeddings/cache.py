"""Persistent SQLite embedding cache and cache-aware embedder wrapper."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Final

from vmp_memos.embeddings.base import (
    BaseEmbedder,
    EmbeddingDimensionError,
    EmbeddingError,
    validate_vector,
)

_CACHE_SCHEMA_VERSION: Final = "1"


class SQLiteEmbeddingCache:
    """Store embeddings by model namespace and exact UTF-8 text content."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @staticmethod
    def cache_key(namespace: str, text: str) -> str:
        payload = f"{namespace}\0{text}".encode()
        return hashlib.sha256(payload).hexdigest()

    def get(self, namespace: str, text: str) -> list[float] | None:
        key = self.cache_key(namespace, text)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT vector_json, dimension FROM embeddings WHERE cache_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            try:
                raw_vector = json.loads(row["vector_json"])
                if not isinstance(raw_vector, list):
                    raise EmbeddingError("Cached vector must be a JSON list")
                return validate_vector(raw_vector, expected_dimension=int(row["dimension"]))
            except (EmbeddingError, TypeError, ValueError):
                connection.execute("DELETE FROM embeddings WHERE cache_key = ?", (key,))
                return None

    def set(self, namespace: str, text: str, vector: Sequence[float]) -> None:
        normalized = validate_vector(vector)
        key = self.cache_key(namespace, text)
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO embeddings (
                    cache_key, namespace, text_hash, dimension, vector_json
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    namespace = excluded.namespace,
                    text_hash = excluded.text_hash,
                    dimension = excluded.dimension,
                    vector_json = excluded.vector_json
                """,
                (
                    key,
                    namespace,
                    text_hash,
                    len(normalized),
                    json.dumps(normalized, separators=(",", ":")),
                ),
            )

    def count(self, namespace: str | None = None) -> int:
        with self._connect() as connection:
            if namespace is None:
                row = connection.execute("SELECT COUNT(*) AS count FROM embeddings").fetchone()
            else:
                row = connection.execute(
                    "SELECT COUNT(*) AS count FROM embeddings WHERE namespace = ?",
                    (namespace,),
                ).fetchone()
        return int(row["count"])

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS cache_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS embeddings (
                    cache_key TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    dimension INTEGER NOT NULL CHECK(dimension > 0),
                    vector_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_embeddings_namespace
                    ON embeddings(namespace);
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO cache_settings(key, value) VALUES ('schema_version', ?)",
                (_CACHE_SCHEMA_VERSION,),
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


class CachedEmbedder(BaseEmbedder):
    """Decorator that batches cache misses and preserves original input order."""

    def __init__(self, embedder: BaseEmbedder, cache: SQLiteEmbeddingCache) -> None:
        self.embedder = embedder
        self.cache = cache
        self._observed_dimension: int | None = embedder.dimension
        self._cache_requests = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._generated = 0

    @property
    def identifier(self) -> str:
        return self.embedder.identifier

    @property
    def dimension(self) -> int | None:
        return self._observed_dimension or self.embedder.dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        normalized_texts = self.validate_texts(texts)
        if not normalized_texts:
            return []

        self._cache_requests += len(normalized_texts)
        results: list[list[float] | None] = [None] * len(normalized_texts)
        missing_texts: list[str] = []
        missing_positions: dict[str, list[int]] = {}
        for index, text in enumerate(normalized_texts):
            cached = self.cache.get(self.identifier, text)
            if cached is not None:
                self._cache_hits += 1
                results[index] = self._observe(cached)
                continue
            self._cache_misses += 1
            if text not in missing_positions:
                missing_texts.append(text)
                missing_positions[text] = []
            missing_positions[text].append(index)

        if missing_texts:
            generated = self.embedder.embed(missing_texts)
            self._generated += len(generated)
            if len(generated) != len(missing_texts):
                raise EmbeddingError(
                    f"Embedder returned {len(generated)} vectors for "
                    f"{len(missing_texts)} cache misses"
                )
            for text, vector in zip(missing_texts, generated, strict=True):
                observed = self._observe(vector)
                self.cache.set(self.identifier, text, observed)
                for index in missing_positions[text]:
                    results[index] = observed.copy()

        if any(vector is None for vector in results):
            raise EmbeddingError("Embedding cache failed to resolve every input")
        return [vector for vector in results if vector is not None]

    def cache_stats(self) -> dict[str, int]:
        """Return cumulative cache request and generation counters."""

        return {
            "requests": self._cache_requests,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "generated": self._generated,
        }

    def release(self) -> None:
        """Release resources held by the wrapped embedder."""

        self.embedder.release()

    def _observe(self, vector: Sequence[float]) -> list[float]:
        normalized = validate_vector(vector, expected_dimension=self._observed_dimension)
        if self._observed_dimension is None:
            self._observed_dimension = len(normalized)
        elif len(normalized) != self._observed_dimension:
            raise EmbeddingDimensionError(
                f"Embedding dimension changed from {self._observed_dimension} "
                f"to {len(normalized)}"
            )
        return normalized
