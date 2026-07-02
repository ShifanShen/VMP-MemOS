"""Content embedding interfaces, providers, and cache support."""

from vmp_memos.embeddings.base import (
    BaseEmbedder,
    EmbeddingDependencyError,
    EmbeddingDimensionError,
    EmbeddingError,
    validate_vector,
)
from vmp_memos.embeddings.cache import CachedEmbedder, SQLiteEmbeddingCache
from vmp_memos.embeddings.openai_compatible import OpenAICompatibleEmbedder
from vmp_memos.embeddings.sentence_transformer import SentenceTransformerEmbedder

__all__ = [
    "BaseEmbedder",
    "CachedEmbedder",
    "EmbeddingDependencyError",
    "EmbeddingDimensionError",
    "EmbeddingError",
    "OpenAICompatibleEmbedder",
    "SQLiteEmbeddingCache",
    "SentenceTransformerEmbedder",
    "validate_vector",
]
