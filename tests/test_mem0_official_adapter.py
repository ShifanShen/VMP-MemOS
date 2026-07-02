"""Tests for the dependency-lazy official Mem0 adapter."""

from __future__ import annotations

from typing import Any

from vmp_memos.frameworks import FrameworkRuntimeConfig
from vmp_memos.frameworks.official import Mem0OfficialAdapter, build_mem0_config
from vmp_memos.longmemeval import LongMemEvalSample, sample_to_session_events


class FakeMem0:
    def __init__(self) -> None:
        self.current: dict[str, Any] = {}
        self.add_calls = 0
        self.reset_calls = 0

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

    def get_all(self, *, filters):
        return {"results": [self.current] if self.current else []}

    def search(self, *, query, filters, top_k):
        return {"results": [self.current] if self.current else []}

    def reset(self):
        self.reset_calls += 1
        self.current = {}


def test_mem0_config_uses_same_local_models(tmp_path) -> None:
    runtime = FrameworkRuntimeConfig(
        vllm_base_url="http://127.0.0.1:8000/v1",
        llm_model="Qwen/Qwen2.5-7B-Instruct",
        vllm_api_key="secret-not-for-manifest",
        embedding_model="BAAI/bge-m3",
        embedding_dimension=1024,
        embedding_device="cuda",
    )

    config = build_mem0_config(runtime, store_dir=tmp_path)

    assert config["llm"]["provider"] == "vllm"
    assert config["llm"]["config"]["model"] == runtime.llm_model
    assert config["llm"]["config"]["temperature"] == 0.0
    assert config["llm"]["config"]["max_tokens"] == 512
    assert config["embedder"]["provider"] == "huggingface"
    assert config["embedder"]["config"]["model"] == runtime.embedding_model
    assert config["vector_store"]["config"]["embedding_model_dims"] == 1024
    assert "secret-not-for-manifest" not in str(runtime.public_metadata())


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
