"""Tests for the dependency-lazy official Graphiti adapter."""

from __future__ import annotations

from types import SimpleNamespace

from vmp_memos.embeddings import BaseEmbedder
from vmp_memos.frameworks import FrameworkRuntimeConfig
from vmp_memos.frameworks.official import GraphitiOfficialAdapter
from vmp_memos.longmemeval import LongMemEvalSample, sample_to_session_events


class FakeEmbedder(BaseEmbedder):
    @property
    def identifier(self) -> str:
        return "fake-graphiti-embedder"

    @property
    def dimension(self) -> int:
        return 2

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FakeTracker:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def get_total_usage(self) -> dict[str, int]:
        return {"prompt_tokens": 12, "completion_tokens": 4}


class FakeDriver:
    def __init__(self) -> None:
        self.clear_count = 0

    async def execute_query(self, query: str):
        assert query == "MATCH (n) DETACH DELETE n"
        self.clear_count += 1
        return [], None, None


class FakeGraph:
    def __init__(self) -> None:
        self.driver = FakeDriver()
        self.token_tracker = FakeTracker()
        self.index_count = 0
        self.close_count = 0
        self.episode_uuids: list[str] = []
        self.current_edge = None

    async def build_indices_and_constraints(self) -> None:
        self.index_count += 1

    async def add_episode(self, **kwargs):
        episode_uuid = f"episode_{len(self.episode_uuids) + 1}"
        self.episode_uuids.append(episode_uuid)
        self.current_edge = SimpleNamespace(
            uuid="fact_1",
            fact=kwargs["episode_body"],
            episodes=list(self.episode_uuids),
            invalid_at=None,
        )
        return SimpleNamespace(
            episode=SimpleNamespace(uuid=episode_uuid),
            edges=[self.current_edge],
        )

    async def search(self, query: str, *, num_results: int):
        assert query == "What activity does Alex now prefer?"
        return [self.current_edge][:num_results]

    async def close(self) -> None:
        self.close_count += 1


def test_graphiti_adapter_preserves_episode_provenance_and_resets(tmp_path) -> None:
    graph = FakeGraph()
    runtime = FrameworkRuntimeConfig(
        embedding_dimension=2,
        graphiti_neo4j_password="test-only",
        graphiti_allow_destructive_reset=True,
    )
    assert "test-only" not in str(runtime.public_metadata())
    adapter = GraphitiOfficialAdapter(
        runtime=runtime,
        embedder=FakeEmbedder(),
        components_factory=lambda runtime, embedder: (graph, "message"),
    )
    sample = LongMemEvalSample.model_validate(_sample_record())

    adapter.reset(tmp_path / "graphiti" / "q1")
    for events in sample_to_session_events(sample):
        adapter.ingest_session(events)
    adapter.finalize_ingestion()
    evidence = adapter.retrieve(sample.question, top_k=5)

    assert evidence[0].source_session_id == "s_new"
    assert evidence[0].source_date == "2024-01-20"
    assert evidence[0].memory_type == "graphiti_fact"
    assert evidence[0].metadata["official_framework"] == "graphiti"
    assert adapter.stats()["framework_llm_usage"]["prompt_tokens"] == 12
    assert graph.driver.clear_count == 1
    assert graph.index_count == 1

    adapter.reset(tmp_path / "graphiti" / "q2")
    assert graph.driver.clear_count == 2
    assert graph.index_count == 1
    assert adapter.stats()["memory_count"] == 0
    adapter.close()
    assert graph.close_count == 1


def test_graphiti_requires_explicit_destructive_reset(tmp_path) -> None:
    adapter = GraphitiOfficialAdapter(
        runtime=FrameworkRuntimeConfig(graphiti_neo4j_password="test-only"),
        embedder=FakeEmbedder(),
    )

    try:
        adapter.reset(tmp_path / "graphiti" / "guard")
    except ValueError as exc:
        assert "dedicated Neo4j" in str(exc)
    else:
        raise AssertionError("Graphiti reset must require explicit acknowledgement")


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
