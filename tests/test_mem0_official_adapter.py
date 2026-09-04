"""Tests for the dependency-lazy official Mem0 adapter."""

from __future__ import annotations

from typing import Any

import pytest

from vmp_memos.frameworks import FrameworkRuntimeConfig
from vmp_memos.frameworks.official import Mem0OfficialAdapter, build_mem0_config
from vmp_memos.frameworks.official.mem0 import (
    MEM0_LLM_COMPATIBILITY_VERSION,
    MEM0_MEMORY_STATS_TOP_K,
    Mem0LlmProtocolError,
    _Mem0LlmResponseAdapter,
)
from vmp_memos.longmemeval import LongMemEvalSample, sample_to_session_events


class FakeMem0:
    def __init__(self) -> None:
        self.current: dict[str, Any] = {}
        self.add_calls = 0
        self.reset_calls = 0
        self.get_all_top_k: int | None = None
        self.llm = FakeMem0Llm('{"memory": []}')

    def add(self, messages, *, user_id, metadata, infer):
        self.add_calls += 1
        self.current = {
            "id": "memory_1",
            "memory": messages[0]["content"],
            "metadata": dict(metadata),
            "score": 0.9,
        }
        return {
            "results": [
                {
                    "id": "memory_1",
                    "memory": messages[0]["content"],
                    "event": "ADD" if self.add_calls == 1 else "UPDATE",
                }
            ]
        }

    def get_all(self, *, filters, top_k):
        self.get_all_top_k = top_k
        return {"results": [self.current] if self.current else []}

    def search(self, *, query, filters, top_k):
        return {"results": [self.current] if self.current else []}

    def reset(self):
        self.reset_calls += 1
        self.current = {}


class FakeMem0Llm:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate_response(self, *args, **kwargs) -> str:
        return self.response


class SequencedMem0Llm:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate_response(self, *args, **kwargs) -> str:
        self.calls.append(dict(kwargs))
        return self.responses.pop(0)


def test_mem0_llm_adapter_normalizes_qwen_string_memory_items() -> None:
    delegate = FakeMem0Llm(
        '```json\n{"memory":["Likes hiking",'
        '{"id":"1","text":"Lives in Paris"}]}\n```'
    )
    adapter = _Mem0LlmResponseAdapter(delegate)

    response = adapter.generate_response(messages=[])

    assert response == (
        '{"memory": [{"text": "Likes hiking"}, '
        '{"id": "1", "text": "Lives in Paris"}]}'
    )
    assert adapter.normalized_response_count == 1
    assert adapter.normalized_item_count == 1


def test_mem0_llm_adapter_leaves_native_object_schema_unchanged() -> None:
    native = '{"memory":[{"id":"0","text":"Likes hiking"}]}'
    adapter = _Mem0LlmResponseAdapter(FakeMem0Llm(native))

    assert adapter.generate_response(messages=[]) == native
    assert adapter.normalized_response_count == 0


def test_mem0_llm_adapter_retries_truncated_json_with_larger_budget() -> None:
    delegate = SequencedMem0Llm(
        [
            '{"memory":[{"text":"first fact"}',
            '{"memory":[{"text":"first fact"},{"text":"second fact"}]}',
        ]
    )
    adapter = _Mem0LlmResponseAdapter(delegate, retry_max_tokens=4096)

    response = adapter.generate_response(
        messages=[],
        response_format={"type": "json_object"},
    )

    assert response == '{"memory":[{"text":"first fact"},{"text":"second fact"}]}'
    assert len(delegate.calls) == 2
    assert "max_tokens" not in delegate.calls[0]
    assert delegate.calls[1]["max_tokens"] == 4096
    assert adapter.logical_call_count == 1
    assert adapter.request_count == 2
    assert adapter.initial_invalid_json_count == 1
    assert adapter.retry_attempt_count == 1
    assert adapter.retry_success_count == 1
    assert adapter.unrecovered_invalid_json_count == 0


def test_mem0_llm_adapter_rejects_unrecoverable_json() -> None:
    delegate = SequencedMem0Llm(
        ['{"memory":[', '{"memory":[{"text":"still truncated"}']
    )
    adapter = _Mem0LlmResponseAdapter(delegate, retry_max_tokens=4096)

    with pytest.raises(Mem0LlmProtocolError, match="invalid JSON"):
        adapter.generate_response(
            messages=[],
            response_format={"type": "json_object"},
        )

    assert adapter.unrecovered_invalid_json_count == 1


def test_mem0_llm_adapter_reset_stats_is_per_question() -> None:
    adapter = _Mem0LlmResponseAdapter(
        FakeMem0Llm('{"memory":["Likes hiking"]}'),
        retry_max_tokens=4096,
    )
    adapter.generate_response(
        messages=[],
        response_format={"type": "json_object"},
    )

    adapter.reset_stats()

    assert adapter.logical_call_count == 0
    assert adapter.request_count == 0
    assert adapter.normalized_response_count == 0
    assert adapter.normalized_item_count == 0


def test_mem0_config_uses_same_local_models(tmp_path) -> None:
    runtime = FrameworkRuntimeConfig(
        vllm_base_url="http://127.0.0.1:8000/v1",
        llm_model="Qwen/Qwen2.5-7B-Instruct",
        vllm_api_key="secret-not-for-manifest",
        embedding_model="BAAI/bge-m3",
        embedding_dimension=1024,
        embedding_device="cuda",
        embedding_base_url="http://127.0.0.1:8001/v1",
        embedding_api_key="embedding-secret-not-for-manifest",
    )

    config = build_mem0_config(runtime, store_dir=tmp_path)

    assert config["llm"]["provider"] == "vllm"
    assert config["llm"]["config"]["model"] == runtime.llm_model
    assert config["llm"]["config"]["temperature"] == 0.0
    assert config["llm"]["config"]["max_tokens"] == 2048
    assert config["embedder"]["provider"] == "openai"
    assert config["embedder"]["config"]["model"] == runtime.embedding_model
    assert (
        config["embedder"]["config"]["openai_base_url"]
        == runtime.embedding_base_url
    )
    assert config["embedder"]["config"]["api_key"] == runtime.embedding_api_key
    assert "model_kwargs" not in config["embedder"]["config"]
    assert config["vector_store"]["config"]["embedding_model_dims"] == 1024
    assert "secret-not-for-manifest" not in str(runtime.public_metadata())
    assert "embedding-secret-not-for-manifest" not in str(runtime.public_metadata())


def test_mem0_adapter_preserves_latest_operation_provenance(tmp_path) -> None:
    fake = FakeMem0()
    adapter = Mem0OfficialAdapter(
        runtime=FrameworkRuntimeConfig(),
        memory_factory=lambda config: fake,
    )
    sample = LongMemEvalSample.model_validate(_sample_record())

    adapter.reset(tmp_path / "mem0" / "q1")
    for events in sample_to_session_events(sample):
        adapter.ingest_session(events)
    adapter.finalize_ingestion()
    evidence = adapter.retrieve(sample.question, top_k=5)

    assert evidence[0].source_session_id == "s_new"
    assert evidence[0].memory_type == "mem0_memory"
    assert adapter.stats()["memory_count"] == 1
    assert adapter.stats()["mem0_llm_compatibility_version"] == (
        MEM0_LLM_COMPATIBILITY_VERSION
    )
    assert adapter.stats()["mem0_memory_stats_top_k"] == MEM0_MEMORY_STATS_TOP_K
    assert adapter.stats()["mem0_memory_count_truncated"] is False
    assert fake.get_all_top_k == MEM0_MEMORY_STATS_TOP_K
    assert isinstance(fake.llm, _Mem0LlmResponseAdapter)

    adapter.reset(tmp_path / "mem0" / "q2")
    assert fake.reset_calls == 2
    assert adapter.memory_count == 0


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
            [{"role": "user", "content": "Alex liked hiking."}],
            [{"role": "user", "content": "Alex now prefers swimming."}],
        ],
        "answer_session_ids": ["s_new"],
        "has_answer": True,
    }
