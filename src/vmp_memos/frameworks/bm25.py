"""Deterministic BM25 retrieval adapter."""

from __future__ import annotations

import math
from collections import Counter

from pydantic import JsonValue

from vmp_memos.frameworks.base import InMemorySessionAdapter, RetrievedMemory
from vmp_memos.frameworks.text import term_counts, terms


class BM25Adapter(InMemorySessionAdapter):
    """Session-level BM25 baseline."""

    name = "bm25"

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        super().__init__()
        self.k1 = k1
        self.b = b

    def _retrieve_impl(
        self,
        query: str,
        *,
        top_k: int,
        question_date: str | None,
        metadata: dict[str, JsonValue],
    ) -> list[RetrievedMemory]:
        query_terms = terms(query)
        if not query_terms or not self.chunks:
            return []
        document_counts = [term_counts(chunk.content) for chunk in self.chunks]
        document_lengths = [sum(counts.values()) for counts in document_counts]
        average_length = sum(document_lengths) / len(document_lengths)
        document_frequency = Counter(
            term for counts in document_counts for term in set(counts)
        )
        total_documents = len(self.chunks)
        ranked: list[tuple[float, int]] = []
        for index, counts in enumerate(document_counts):
            score = 0.0
            doc_length = max(1, document_lengths[index])
            for term in query_terms:
                frequency = counts.get(term, 0)
                if frequency <= 0:
                    continue
                df = document_frequency[term]
                idf = math.log(1.0 + (total_documents - df + 0.5) / (df + 0.5))
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * doc_length / average_length
                )
                score += idf * frequency * (self.k1 + 1.0) / denominator
            ranked.append((score, index))
        ranked.sort(key=lambda pair: (-pair[0], self.chunks[pair[1]].memory_id))
        return [
            self.chunks[index].to_retrieved(
                score=score,
                metadata={"retrieval_strategy": "bm25", "question_date": question_date},
            )
            for score, index in ranked[:top_k]
            if score > 0.0
        ]
