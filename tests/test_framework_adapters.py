"""Tests for built-in memory framework adapters."""

from __future__ import annotations

from vmp_memos.frameworks import (
    BM25Adapter,
    NaiveVectorAdapter,
    VMPRuleAdapter,
    VectorImportanceAdapter,
    VectorRecencyAdapter,
    adapter_for_name,
    default_registry,
)
from vmp_memos.longmemeval import LongMemEvalSample, sample_to_session_events


def test_default_registry_creates_builtin_adapters() -> None:
    registry = default_registry()
    assert "bm25" in registry.names()
    assert adapter_for_name("naive-vector").name == "naive_vector"
    assert adapter_for_name("no_memory").retrieve("anything", top_k=5) == []


def test_builtin_adapters_retrieve_session_evidence(tmp_path) -> None:
    sample = LongMemEvalSample.model_validate(_sample_record())
    query = sample.question
    session_groups = sample_to_session_events(sample)

    for adapter in (
        BM25Adapter(),
        NaiveVectorAdapter(),
        VectorRecencyAdapter(),
        VectorImportanceAdapter(),
        VMPRuleAdapter(),
    ):
        adapter.reset(tmp_path / adapter.name)
        for events in session_groups:
            adapter.ingest_session(events)
        results = adapter.retrieve(
            query,
            top_k=2,
            question_date=sample.question_date,
            metadata={
                "question_id": sample.question_id,
                "question_type": sample.question_type,
            },
        )
        assert results, adapter.name
        assert results[0].source_session_id == "s_new", adapter.name
        assert results[0].token_count > 0
        stats = adapter.stats()
        assert stats["memory_count"] == 2
        assert stats["ingestion_sessions"] == 2


def _sample_record() -> dict:
    return {
        "question_id": "q1",
        "question_type": "knowledge_update",
        "question": "What activity does Alex now prefer?",
        "answer": "swimming",
        "question_date": "2024-02-01",
        "haystack_session_ids": ["s_old", "s_new"],
        "haystack_dates": ["2024-01-01", "2024-01-20"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "Alex said hiking was fun."},
                {"role": "assistant", "content": "I will remember Alex liked hiking."},
            ],
            [
                {"role": "user", "content": "Alex now prefers swimming instead of hiking."},
                {"role": "assistant", "content": "Updated: Alex prefers swimming."},
            ],
        ],
        "answer_session_ids": ["s_new"],
        "has_answer": True,
    }
