"""Tests for the dependency-lazy official Letta adapter."""

from __future__ import annotations

import re
from types import SimpleNamespace

from vmp_memos.embeddings import BaseEmbedder
from vmp_memos.frameworks import FrameworkRuntimeConfig
from vmp_memos.frameworks.official import (
    LettaOfficialAdapter,
    build_letta_embedding_config,
    build_letta_llm_config,
)
from vmp_memos.longmemeval import LongMemEvalSample, sample_to_session_events


class FakeEmbedder(BaseEmbedder):
    @property
    def identifier(self) -> str:
        return "fake-letta-embedder"

    @property
    def dimension(self) -> int:
        return 2

    def embed(self, texts):
        return [
            [1.0, 0.0] if "swim" in text.casefold() else [0.0, 1.0]
            for text in texts
        ]


class FakeBlocks:
    def __init__(self) -> None:
        self.value = ""

    def list(self, agent_id: str, *, limit: int):
        assert agent_id == "agent-test"
        assert limit == 200
        return [
            SimpleNamespace(
                id="block-memory",
                label="long_term_memory",
                value=self.value,
            )
        ]


class FakeMessages:
    def __init__(self, blocks: FakeBlocks) -> None:
        self.blocks = blocks

    def create(self, *, agent_id: str, input: str):
        assert agent_id == "agent-test"
        marker = re.search(r"\[SOURCE_SESSION_ID=[^\]]+\]", input)
        assert marker is not None
        fact = (
            "Alex now prefers swimming."
            if "swimming" in input
            else "Alex liked hiking."
        )
        self.blocks.value = f"{fact} {marker.group(0)}"
        return SimpleNamespace(messages=[])


class FakeAgentPassages:
    def list(self, agent_id: str, *, limit: int):
        assert agent_id == "agent-test"
        assert limit == 100
        return []


class FakeGlobalPassages:
    def search(self, *, agent_id: str, query: str, limit: int):
        assert agent_id == "agent-test"
        assert query == "What activity does Alex now prefer?"
        assert limit == 5
        return []


class FakeAgents:
    def __init__(self) -> None:
        self.blocks = FakeBlocks()
        self.messages = FakeMessages(self.blocks)
        self.passages = FakeAgentPassages()
        self.create_calls: list[dict] = []
        self.delete_calls: list[str] = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return SimpleNamespace(id="agent-test")

    def delete(self, agent_id: str):
        self.delete_calls.append(agent_id)


class FakeClient:
    def __init__(self) -> None:
        self.agents = FakeAgents()
        self.passages = FakeGlobalPassages()
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_letta_configs_use_same_local_models() -> None:
    runtime = FrameworkRuntimeConfig(
        vllm_base_url="http://127.0.0.1:8000/v1",
        llm_model="Qwen/Qwen2.5-7B-Instruct",
        embedding_model="BAAI/bge-m3",
        embedding_dimension=1024,
        letta_embedding_base_url="http://127.0.0.1:8001/v1",
    )

    llm = build_letta_llm_config(runtime)
    embedding = build_letta_embedding_config(runtime)

    assert llm["model_endpoint_type"] == "openai"
    assert llm["model_endpoint"] == "http://127.0.0.1:8000/v1"
    assert llm["model"] == runtime.llm_model
    assert llm["temperature"] == 0.0
    assert llm["max_tokens"] == 512
    assert embedding["embedding_endpoint_type"] == "openai"
    assert embedding["embedding_endpoint"] == runtime.letta_embedding_base_url
    assert embedding["embedding_model"] == runtime.embedding_model
    assert embedding["embedding_dim"] == 1024


def test_letta_adapter_exports_managed_core_memory_and_resets(tmp_path) -> None:
    fake = FakeClient()
    runtime = FrameworkRuntimeConfig(
        embedding_dimension=2,
        letta_api_key="server-secret",
    )
    adapter = LettaOfficialAdapter(
        runtime=runtime,
        embedder=FakeEmbedder(),
        client_factory=lambda runtime: fake,
    )
    sample = LongMemEvalSample.model_validate(_sample_record())

    adapter.reset(tmp_path / "letta" / "q1")
    for events in sample_to_session_events(sample):
        adapter.ingest_session(events)
    adapter.finalize_ingestion()
    evidence = adapter.retrieve(sample.question, top_k=5)

    assert evidence[0].content == "Alex now prefers swimming."
    assert evidence[0].source_session_id == "s_new"
    assert evidence[0].source_date == "2024-01-20"
    assert evidence[0].memory_type == "letta_core_memory"
    assert evidence[0].metadata["official_framework"] == "letta"
    assert fake.agents.create_calls[0]["include_base_tools"] is True
    assert adapter.stats()["memory_count"] == 1
    assert "server-secret" not in str(runtime.public_metadata())

    adapter.reset(tmp_path / "letta" / "q2")
    assert fake.agents.delete_calls == ["agent-test"]
    adapter.close()
    assert fake.agents.delete_calls == ["agent-test", "agent-test"]
    assert fake.closed is True
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
