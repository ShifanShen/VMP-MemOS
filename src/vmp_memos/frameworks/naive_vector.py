"""Naive vector/RAG retrieval adapter."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import JsonValue

from vmp_memos.embeddings import BaseEmbedder
from vmp_memos.frameworks.base import InMemorySessionAdapter, MemoryChunk, RetrievedMemory
from vmp_memos.frameworks.text import dense_cosine, sparse_cosine, term_counts
from vmp_memos.schemas import Event


class NaiveVectorAdapter(InMemorySessionAdapter):
    """Vector-style top-k retrieval using a shared embedder or lexical fallback."""

    name = "naive_vector"

    def __init__(self, *, embedder: BaseEmbedder | None = None) -> None:
        super().__init__()
        self.embedder = embedder

    def _ingest_event_impl(self, event: Event) -> None:
        super()._ingest_event_impl(event)

    def _ingest_session_impl(self, events: list[Event]) -> None:
        super()._ingest_session_impl(events)

    def _finalize_ingestion_impl(self) -> None:
        self._embed_new_chunks()

    def score_chunk(
        self,
        query: str,
        chunk: MemoryChunk,
        *,
        query_embedding: Sequence[float] | None = None,
        question_date: str | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ) -> float:
        """Return base similarity score for a query/chunk pair."""

        if query_embedding is not None and chunk.content_embedding:
            return dense_cosine(query_embedding, chunk.content_embedding)
        return sparse_cosine(term_counts(query), term_counts(chunk.content))

    def _retrieve_impl(
        self,
        query: str,
        *,
        top_k: int,
        question_date: str | None,
        metadata: dict[str, JsonValue],
    ) -> list[RetrievedMemory]:
        self._embed_new_chunks()
        query_embedding = self._embed_query(query)
        ranked = [
            (
                self.score_chunk(
                    query,
                    chunk,
                    query_embedding=query_embedding,
                    question_date=question_date,
                    metadata=metadata,
                ),
                chunk,
            )
            for chunk in self.chunks
        ]
        ranked.sort(key=lambda pair: (-pair[0], pair[1].memory_id))
        return [
            chunk.to_retrieved(
                score=score,
                metadata={
                    "retrieval_strategy": self.name,
                    "embedder": self.embedder.identifier if self.embedder else "lexical_fallback",
                },
            )
            for score, chunk in ranked[:top_k]
            if score > 0.0
        ]

    def _embed_new_chunks(self) -> None:
        if self.embedder is None:
            return
        pending = [chunk for chunk in self.chunks if not chunk.content_embedding]
        if not pending:
            return
        vectors = self.embedder.embed([chunk.content for chunk in pending])
        for chunk, vector in zip(pending, vectors, strict=True):
            chunk.content_embedding = list(vector)

    def _embed_query(self, query: str) -> list[float] | None:
        if self.embedder is None:
            return None
        return self.embedder.embed_one(query)
