"""Backend-neutral text embedding contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from math import isfinite


class EmbeddingError(RuntimeError):
    """Base exception for embedding generation and validation failures."""


class EmbeddingDependencyError(EmbeddingError):
    """Raised when an optional embedding provider is not installed."""


class EmbeddingDimensionError(EmbeddingError):
    """Raised when vectors from one embedding space have inconsistent dimensions."""


class BaseEmbedder(ABC):
    """Minimal interface consumed by vector backends."""

    @property
    @abstractmethod
    def identifier(self) -> str:
        """Return a stable identifier for the model and relevant encode settings."""

    @property
    @abstractmethod
    def dimension(self) -> int | None:
        """Return the vector dimension when known, otherwise ``None``."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed non-empty texts in input order."""

    def embed_one(self, text: str) -> list[float]:
        """Embed one text and enforce the provider's batch contract."""

        vectors = self.embed([text])
        if len(vectors) != 1:
            raise EmbeddingError(
                f"Embedder returned {len(vectors)} vectors for one input text"
            )
        return vectors[0]

    def release(self) -> None:
        """Release optional provider resources such as GPU model weights."""

    @staticmethod
    def validate_texts(texts: Sequence[str]) -> list[str]:
        """Normalize the input container while preserving text content and order."""

        normalized = list(texts)
        if not normalized:
            return []
        for index, text in enumerate(normalized):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Embedding input at index {index} must be non-empty text")
        return normalized


def validate_vector(
    vector: Sequence[float],
    *,
    expected_dimension: int | None = None,
) -> list[float]:
    """Return a finite float vector and optionally enforce its dimension."""

    normalized = [float(value) for value in vector]
    if not normalized:
        raise EmbeddingDimensionError("Embedding vector cannot be empty")
    if not all(isfinite(value) for value in normalized):
        raise EmbeddingError("Embedding vector contains a non-finite value")
    if expected_dimension is not None and len(normalized) != expected_dimension:
        raise EmbeddingDimensionError(
            f"Expected embedding dimension {expected_dimension}, got {len(normalized)}"
        )
    return normalized
