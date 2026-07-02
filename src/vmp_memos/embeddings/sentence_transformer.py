"""Lazy SentenceTransformers implementation of :class:`BaseEmbedder`."""

from __future__ import annotations

import gc
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from vmp_memos.embeddings.base import (
    BaseEmbedder,
    EmbeddingDependencyError,
    EmbeddingDimensionError,
    validate_vector,
)


class SentenceTransformerEmbedder(BaseEmbedder):
    """Generate local embeddings without importing model code until first use."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str = "auto",
        cache_folder: str | Path | None = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.model_name = model_name
        self.device = device
        self.cache_folder = Path(cache_folder).expanduser() if cache_folder else None
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self._model: Any | None = None
        self._dimension: int | None = None

    @property
    def identifier(self) -> str:
        normalized = int(self.normalize_embeddings)
        return f"sentence-transformers:{self.model_name}:normalize={normalized}"

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        normalized_texts = self.validate_texts(texts)
        if not normalized_texts:
            return []
        model = self._load_model()
        try:
            import numpy as np
        except ImportError as exc:
            raise EmbeddingDependencyError(
                "SentenceTransformerEmbedder requires numpy through the optional "
                "embedding dependencies. Install them with: "
                'python -m pip install -e ".[embeddings]"'
            ) from exc

        encoded = model.encode(
            normalized_texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        matrix = np.asarray(encoded, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(normalized_texts):
            raise EmbeddingDimensionError(
                "SentenceTransformer returned an unexpected embedding matrix shape"
            )
        dimension = int(matrix.shape[1])
        if self._dimension is not None and self._dimension != dimension:
            raise EmbeddingDimensionError(
                f"Model dimension changed from {self._dimension} to {dimension}"
            )
        self._dimension = dimension
        return [validate_vector(row, expected_dimension=dimension) for row in matrix.tolist()]

    def release(self) -> None:
        """Unload model weights so an official framework can use the same GPU."""

        if self._model is None:
            return
        self._model = None
        gc.collect()
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingDependencyError(
                "SentenceTransformerEmbedder requires the optional embedding dependencies. "
                'Install them with: python -m pip install -e ".[embeddings]"'
            ) from exc

        kwargs: dict[str, Any] = {}
        if self.device != "auto":
            kwargs["device"] = self.device
        if self.cache_folder is not None:
            self.cache_folder.mkdir(parents=True, exist_ok=True)
            kwargs["cache_folder"] = str(self.cache_folder)
        self._model = SentenceTransformer(self.model_name, **kwargs)
        model_dimension = self._model.get_sentence_embedding_dimension()
        if model_dimension is not None:
            self._dimension = int(model_dimension)
        return self._model
