"""Memory backend interfaces and implementations."""

from __future__ import annotations

from vmp_memos.backends.base import (
    BaseMemoryBackend,
    InvalidMemoryFileError,
    InvalidMemoryIdError,
    MemoryAlreadyExistsError,
    MemoryBackendError,
    MemoryNotFoundError,
)

__all__ = [
    "BaseMemoryBackend",
    "FileMemoryBackend",
    "HybridBackendError",
    "HybridMemoryBackend",
    "InvalidMemoryFileError",
    "InvalidMemoryIdError",
    "MemoryAlreadyExistsError",
    "MemoryBackendError",
    "MemoryNotFoundError",
    "VectorDimensionError",
    "VectorMemoryBackend",
    "VectorStoreError",
    "cosine_similarity",
]


def __getattr__(name: str) -> object:
    """Import concrete backends lazily so optional dependencies stay optional."""

    if name == "FileMemoryBackend":
        from vmp_memos.backends.file_backend import FileMemoryBackend

        return FileMemoryBackend
    if name in {"HybridBackendError", "HybridMemoryBackend"}:
        from vmp_memos.backends.hybrid_backend import (
            HybridBackendError,
            HybridMemoryBackend,
        )

        exports = {
            "HybridBackendError": HybridBackendError,
            "HybridMemoryBackend": HybridMemoryBackend,
        }
        return exports[name]
    if name in {
        "VectorDimensionError",
        "VectorMemoryBackend",
        "VectorStoreError",
        "cosine_similarity",
    }:
        from vmp_memos.backends.vector_backend import (
            VectorDimensionError,
            VectorMemoryBackend,
            VectorStoreError,
            cosine_similarity,
        )

        exports = {
            "VectorDimensionError": VectorDimensionError,
            "VectorMemoryBackend": VectorMemoryBackend,
            "VectorStoreError": VectorStoreError,
            "cosine_similarity": cosine_similarity,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
