"""Tests for the dependency-lazy official LangMem adapter."""

from __future__ import annotations

from types import SimpleNamespace

from vmp_memos.embeddings import BaseEmbedder
from vmp_memos.frameworks import FrameworkRuntimeConfig
from vmp_memos.frameworks.official import LangMemOfficialAdapter
from vmp_memos.longmemeval import LongMemEvalSample, sample_to_session_events


class FakeEmbedder(BaseEmbedder):
    @property
    def identifier(self) -> str:
        return "fake-langmem-embedder"

    @property
    def dimension(self) -> int:
        return 2

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FakeStore:
    def __init__(self) -> None:
        self.value = None

    def search(self, namespace, *, query=None, limit=10):
        if self.value is None:
            return []
        return [
            SimpleNamespace(
                key="memory_1",
                value={"kind": "Memory", "content": {"content": self.value}},
                score=0.95 if query else None,
            )
        ]


class FakeManager:
    def __init__(self, store: FakeStore) -> None:
        self.store = store

    def invoke(self, payload):
        self.store.value = payload["messages"][0]["content"]
        return []


def test_langmem_adapter_uses_official_manager_and_tracks_updates(tmp_path) -> None:
    store = FakeStore()
    manager = FakeManager(store)
    adapter = LangMemOfficialAdapter(
        runtime=FrameworkRuntimeConfig(embedding_dimension=2),
        embedder=FakeEmbedder(),
        components_factory=lambda runtime, embedder: (store, manager),
    )
    sample = LongMemEvalSample.model_validate(_sample_record())

    adapter.reset(tmp_path / "langmem" / "q1")
    for events in sample_to_session_events(sample):
        adapter.ingest_session(events)
    adapter.finalize_ingestion()
    evidence = adapter.retrieve(sample.question, top_k=5)

    assert evidence[0].source_session_id == "s_new"
    assert evidence[0].memory_type == "langmem_memory"
    assert evidence[0].metadata["official_framework"] == "langmem"
    assert adapter.stats()["memory_count"] == 1


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
