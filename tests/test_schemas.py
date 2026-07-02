"""Integration-level tests for all public Phase 1 schemas."""

from vmp_memos.schemas import (
    BenchmarkResult,
    BenchmarkSample,
    Event,
    MemoryCandidate,
    MemoryItem,
    MemorySource,
    OperationType,
    PolicyFeatures,
    RetrievalResult,
)


def test_event_candidate_memory_and_retrieval_round_trip() -> None:
    event = Event(
        session_id="sess_001",
        task_id="task_001",
        event_type="user_message",
        content="用户希望主攻 Agent 开发。",
        metadata={"source": "conversation"},
    )
    candidate = MemoryCandidate(
        source_event_id=event.event_id,
        memory_type="semantic",
        content="用户当前主攻 Agent 开发。",
        scope="career/agent-dev",
        tags=["career", "agent", "agent"],
        confidence=0.92,
        importance=0.88,
    )
    memory = MemoryItem(
        type="semantic",
        scope=candidate.scope,
        content=candidate.content,
        summary="用户职业方向偏 Agent 开发。",
        source=MemorySource(event_id=event.event_id, source_type="conversation"),
        features=PolicyFeatures(importance=0.88, confidence=0.92, novelty=0.9),
    )
    retrieval = RetrievalResult(
        query="用户当前的职业方向是什么？",
        memory_ids=[memory.id],
        items=[memory],
        scores={memory.id: 0.94},
        token_count=24,
        latency_ms=3.2,
        backend="file",
    )

    assert candidate.tags == ["career", "agent"]
    assert retrieval.memory_ids == [memory.id]
    assert RetrievalResult.model_validate_json(retrieval.model_dump_json()) == retrieval
    assert all(
        schema.timestamp.tzinfo is not None
        for schema in (event, candidate, memory, memory.features, retrieval)
    )


def test_benchmark_schemas_preserve_expected_operations() -> None:
    event = Event(
        session_id="sess_benchmark",
        event_type="benchmark_sample",
        content="Preference update case",
    )
    sample = BenchmarkSample(
        sample_id="case_001",
        events=[event],
        query="用户现在的主要求职方向是什么？",
        gold_answer="Agent 开发和 LLM 应用开发",
        expected_operations=[OperationType.UPDATE, OperationType.RETRIEVE],
        metadata={"task_type": "preference_update"},
    )
    result = BenchmarkResult(
        sample_id=sample.sample_id,
        system_name="schema-smoke-test",
        answer="Agent 开发和 LLM 应用开发",
        is_correct=True,
        operations=["UPDATE", "RETRIEVE"],
        metrics={"accuracy": 1.0, "latency_ms": 2.5},
        token_count=10,
        latency_ms=2.5,
    )

    restored_sample = BenchmarkSample.model_validate_json(sample.model_dump_json())
    restored_result = BenchmarkResult.model_validate_json(result.model_dump_json())

    assert restored_sample.sample_id == "case_001"
    assert restored_sample.expected_operations == [OperationType.UPDATE, OperationType.RETRIEVE]
    assert restored_result.result_id == result.result_id
    assert restored_result.operations == [OperationType.UPDATE, OperationType.RETRIEVE]

