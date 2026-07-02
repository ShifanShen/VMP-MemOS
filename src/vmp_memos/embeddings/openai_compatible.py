"""OpenAI-compatible HTTP embedding client."""

from __future__ import annotations

import json
from collections.abc import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from vmp_memos.embeddings.base import (
    BaseEmbedder,
    EmbeddingError,
    validate_vector,
)


class OpenAICompatibleEmbedder(BaseEmbedder):
    """Call a local OpenAI-compatible ``/v1/embeddings`` endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        dimension: int,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url cannot be empty")
        if not model.strip():
            raise ValueError("model cannot be empty")
        if dimension < 1:
            raise ValueError("dimension must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._dimension = dimension
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    @property
    def identifier(self) -> str:
        return f"openai-compatible:{self.model}@{self.base_url}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        normalized = self.validate_texts(texts)
        if not normalized:
            return []
        payload = json.dumps(
            {
                "input": normalized,
                "model": self.model,
                "encoding_format": "float",
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            _embeddings_url(self.base_url),
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, OSError, json.JSONDecodeError) as exc:
            raise EmbeddingError(
                f"OpenAI-compatible embedding request failed: {exc}"
            ) from exc
        data = raw.get("data") if isinstance(raw, dict) else None
        if not isinstance(data, list):
            raise EmbeddingError("Embedding response does not contain a data list")
        ordered = sorted(
            data,
            key=lambda item: item.get("index", 0) if isinstance(item, dict) else 0,
        )
        vectors = [
            validate_vector(
                item.get("embedding", []) if isinstance(item, dict) else [],
                expected_dimension=self._dimension,
            )
            for item in ordered
        ]
        if len(vectors) != len(normalized):
            raise EmbeddingError(
                f"Expected {len(normalized)} embeddings, received {len(vectors)}"
            )
        return vectors


def _embeddings_url(base_url: str) -> str:
    return (
        f"{base_url}/embeddings"
        if base_url.endswith("/v1")
        else f"{base_url}/v1/embeddings"
    )
