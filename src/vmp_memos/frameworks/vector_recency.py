"""Vector retrieval reranked with a recency prior."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import JsonValue

from vmp_memos.frameworks.base import MemoryChunk
from vmp_memos.frameworks.naive_vector import NaiveVectorAdapter
from vmp_memos.frameworks.text import clamp01, recency_score


class VectorRecencyAdapter(NaiveVectorAdapter):
    """Naive vector retrieval plus non-gold recency signal."""

    name = "vector_recency"

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
        recency = recency_score(chunk.source_date, question_date)
        return clamp01(0.75 * lexical_or_vector + 0.25 * recency)
