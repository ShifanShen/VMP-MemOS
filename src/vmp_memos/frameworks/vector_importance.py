"""Vector retrieval reranked with a heuristic importance prior."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import JsonValue

from vmp_memos.frameworks.base import MemoryChunk
from vmp_memos.frameworks.naive_vector import NaiveVectorAdapter
from vmp_memos.frameworks.text import clamp01, heuristic_importance


class VectorImportanceAdapter(NaiveVectorAdapter):
    """Naive vector retrieval plus non-gold importance signal."""

    name = "vector_importance"

    def score_chunk(
        self,
        query: str,
        chunk: MemoryChunk,
        *,
        query_embedding: Sequence[float] | None = None,
        question_date: str | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ) -> float:
        lexical_or_vector = super().score_chunk(
            query,
            chunk,
            query_embedding=query_embedding,
            question_date=question_date,
            metadata=metadata,
        )
        raw_importance = chunk.metadata.get("importance")
        importance = (
            float(raw_importance)
            if isinstance(raw_importance, int | float)
            else heuristic_importance(chunk.content)
        )
        return clamp01(0.75 * lexical_or_vector + 0.25 * importance)
