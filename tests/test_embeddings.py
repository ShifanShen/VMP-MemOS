"""Tests for embedding contracts and the persistent cache wrapper."""

from collections.abc import Sequence

import pytest

from vmp_memos.embeddings import (
    BaseEmbedder,
    CachedEmbedder,
    EmbeddingDimensionError,
    SentenceTransformerEmbedder,
    SQLiteEmbeddingCache,
)


class CountingEmbedder(BaseEmbedder):
    """Small deterministic embedder for cache tests."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def identifier(self) -> str:
        return "test-counting-embedder"

    @property
    def dimension(self) -> int:
        return 3

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        normalized = self.validate_texts(texts)
        self.calls.extend(normalized)
        return [self._vector(text) for text in normalized]

    @staticmethod
    def _vector(text: str) -> list[float]:
        return [
            float(len(text)),
            1.0 if "agent" in text.casefold() else 0.0,
            1.0,
        ]


def test_sqlite_embedding_cache_batches_misses_and_persists_hits(tmp_path) -> None:
    cache = SQLiteEmbeddingCache(tmp_path / "embeddings.sqlite3")
    embedder = CountingEmbedder()
    cached = CachedEmbedder(embedder, cache)

    vectors = cached.embed(["agent memory", "agent memory", "java backend"])

    assert vectors[0] == vectors[1]
    assert embedder.calls == ["agent memory", "java backend"]
    assert cache.count(embedder.identifier) == 2
    assert cached.cache_stats() == {
        "requests": 3,
        "hits": 0,
        "misses": 3,
        "generated": 2,
    }

    second_embedder = CountingEmbedder()
    second_cached = CachedEmbedder(second_embedder, cache)
    assert second_cached.embed(["agent memory"]) == [vectors[0]]
    assert second_embedder.calls == []
    assert second_cached.cache_stats()["hits"] == 1


def test_cached_embedder_rejects_dimension_changes(tmp_path) -> None:
    cache = SQLiteEmbeddingCache(tmp_path / "embeddings.sqlite3")
    cached = CachedEmbedder(CountingEmbedder(), cache)
    cached.embed(["agent memory"])

    cache.set(cached.identifier, "bad vector", [1.0, 2.0, 3.0, 4.0])

    with pytest.raises(EmbeddingDimensionError):
        cached.embed(["bad vector"])


def test_sentence_transformer_embedder_is_lazy_and_identifiable() -> None:
    embedder = SentenceTransformerEmbedder(
        "sentence-transformers/all-MiniLM-L6-v2",
        normalize_embeddings=False,
    )

    assert embedder.dimension is None
    assert embedder.identifier.endswith(":normalize=0")
