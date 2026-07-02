"""Official Graphiti adapter backed by a dedicated Neo4j database."""

from __future__ import annotations

import asyncio
import json
import math
import os
from collections.abc import Callable, Coroutine, Iterable
from datetime import UTC, datetime
from typing import Any, TypeVar

from pydantic import JsonValue

from vmp_memos.embeddings import BaseEmbedder, SentenceTransformerEmbedder
from vmp_memos.frameworks.base import (
    BaseMemoryFrameworkAdapter,
    FairnessLevel,
    RetrievedMemory,
)
from vmp_memos.frameworks.runtime import FrameworkRuntimeConfig
from vmp_memos.frameworks.text import estimate_tokens, parse_date
from vmp_memos.longmemeval.converter import session_to_text
from vmp_memos.schemas import Event

GraphitiComponentsFactory = Callable[
    [FrameworkRuntimeConfig, BaseEmbedder],
    tuple[Any, Any],
]
T = TypeVar("T")


class GraphitiDependencyError(RuntimeError):
    """Raised when the pinned Graphiti package is unavailable."""


class GraphitiOfficialAdapter(BaseMemoryFrameworkAdapter):
    """Thin synchronous bridge to Graphiti's async official Python API."""

    name = "graphiti"
    fairness_level = FairnessLevel.FULLY_CONTROLLED

    def __init__(
        self,
        *,
        runtime: FrameworkRuntimeConfig | None = None,
        embedder: BaseEmbedder | None = None,
        components_factory: GraphitiComponentsFactory | None = None,
    ) -> None:
        super().__init__()
        self.runtime = runtime or FrameworkRuntimeConfig.from_env()
        self.embedder = embedder
        self._components_factory = components_factory
        self._graph: Any | None = None
        self._episode_type: Any | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._indices_built = False
        self._facts: dict[str, str] = {}
        self._episode_provenance: dict[str, dict[str, str | None]] = {}

    @property
    def memory_count(self) -> int:
        return len(self._facts)

    @property
    def total_tokens(self) -> int:
        return sum(estimate_tokens(fact) for fact in self._facts.values())

    @property
    def storage_size_bytes(self) -> int:
        content_bytes = sum(len(fact.encode("utf-8")) for fact in self._facts.values())
        vector_bytes = 4 * self.runtime.embedding_dimension * len(self._facts)
        return content_bytes + vector_bytes

    def _reset_impl(self) -> None:
        if not self.runtime.graphiti_allow_destructive_reset:
            raise ValueError(
                "Graphiti requires graphiti_allow_destructive_reset=True and a "
                "dedicated Neo4j instance because every benchmark question clears it"
            )
        if not self.runtime.graphiti_neo4j_password:
            raise ValueError("Graphiti requires VMP_GRAPHITI_NEO4J_PASSWORD")
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        if self._graph is None:
            embedder = self.embedder or SentenceTransformerEmbedder(
                self.runtime.embedding_model,
                device=self.runtime.embedding_device,
            )
            self.embedder = embedder
            if self._components_factory is not None:
                self._graph, self._episode_type = self._components_factory(
                    self.runtime,
                    embedder,
                )
            else:
                self._graph, self._episode_type = _create_graphiti_components(
                    self.runtime,
                    embedder,
                )
        self._run(self._clear_graph())
        if not self._indices_built:
            self._run(self._graph.build_indices_and_constraints())
            self._indices_built = True
        self._facts = {}
        self._episode_provenance = {}
        tracker = getattr(self._graph, "token_tracker", None)
        reset_tracker = getattr(tracker, "reset", None)
        if callable(reset_tracker):
            reset_tracker()

    def _ingest_event_impl(self, event: Event) -> None:
        self._add_episode([event])

    def _ingest_session_impl(self, events: list[Event]) -> None:
        if events:
            self._add_episode(events)

    def _retrieve_impl(
        self,
        query: str,
        *,
        top_k: int,
        question_date: str | None,
        metadata: dict[str, JsonValue],
    ) -> list[RetrievedMemory]:
        if self._graph is None:
            raise RuntimeError("Graphiti adapter must be reset before retrieval")
        edges = self._run(self._graph.search(query, num_results=top_k))
        results: list[RetrievedMemory] = []
        for rank, edge in enumerate(edges, start=1):
            memory_id = str(getattr(edge, "uuid", "") or "").strip()
            fact = str(getattr(edge, "fact", "") or "").strip()
            if not memory_id or not fact:
                continue
            provenance = self._latest_provenance(
                list(getattr(edge, "episodes", []) or [])
            )
            results.append(
                RetrievedMemory(
                    memory_id=memory_id,
                    content=fact,
                    score=1.0 / rank,
                    source_session_id=provenance.get("source_session_id"),
                    source_date=provenance.get("source_date"),
                    memory_type="graphiti_fact",
                    token_count=estimate_tokens(fact),
                    metadata={
                        "retrieval_strategy": self.name,
                        "official_framework": "graphiti",
                        "episode_uuids": [
                            str(value)
                            for value in list(getattr(edge, "episodes", []) or [])
                        ],
                    },
                )
            )
        return results[:top_k]

    def stats(self) -> dict[str, JsonValue]:
        stats = super().stats()
        stats["storage_size_is_estimate"] = True
        stats["graphiti_neo4j_uri"] = self.runtime.graphiti_neo4j_uri
        tracker = getattr(self._graph, "token_tracker", None)
        get_usage = getattr(tracker, "get_total_usage", None)
        if callable(get_usage):
            usage = get_usage()
            stats["framework_llm_usage"] = _json_value_or_text(usage)
        return stats

    def close(self) -> None:
        try:
            if self._graph is not None and self._loop is not None:
                self._run(self._graph.close())
        finally:
            if self._loop is not None:
                self._loop.close()
            self._graph = None
            self._loop = None

    async def _clear_graph(self) -> None:
        if self._graph is None:
            return
        await self._graph.driver.execute_query("MATCH (n) DETACH DELETE n")

    def _add_episode(self, events: list[Event]) -> None:
        if self._graph is None or self._episode_type is None:
            raise RuntimeError("Graphiti adapter must be reset before ingestion")
        first = events[0]
        source_session_id = _event_metadata_text(first, "history_session_id")
        source_date = _event_metadata_text(first, "history_date")
        reference_time = parse_date(source_date) or datetime.now(UTC)
        result = self._run(
            self._graph.add_episode(
                name=source_session_id or first.session_id,
                episode_body=session_to_text(events),
                source=self._episode_type,
                source_description="LongMemEval history session",
                reference_time=reference_time,
            )
        )
        episode_uuid = str(getattr(getattr(result, "episode", None), "uuid", "") or "")
        if episode_uuid:
            self._episode_provenance[episode_uuid] = {
                "source_session_id": source_session_id,
                "source_date": source_date,
            }
        for edge in list(getattr(result, "edges", []) or []):
            edge_id = str(getattr(edge, "uuid", "") or "")
            fact = str(getattr(edge, "fact", "") or "")
            invalid_at = getattr(edge, "invalid_at", None)
            if not edge_id:
                continue
            if invalid_at is not None:
                self._facts.pop(edge_id, None)
            elif fact:
                self._facts[edge_id] = fact

    def _latest_provenance(
        self,
        episode_uuids: list[object],
    ) -> dict[str, str | None]:
        candidates = [
            self._episode_provenance[str(episode_uuid)]
            for episode_uuid in episode_uuids
            if str(episode_uuid) in self._episode_provenance
        ]
        if not candidates:
            return {}
        return max(
            candidates,
            key=lambda item: parse_date(item.get("source_date"))
            or datetime.min.replace(tzinfo=UTC),
        )

    def _run(self, coroutine: Coroutine[Any, Any, T]) -> T:
        if self._loop is None:
            raise RuntimeError("Graphiti event loop is not initialized")
        return self._loop.run_until_complete(coroutine)


def _create_graphiti_components(
    runtime: FrameworkRuntimeConfig,
    embedder: BaseEmbedder,
) -> tuple[Any, Any]:
    os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")
    configured_dimension = os.getenv("EMBEDDING_DIM")
    if (
        configured_dimension is not None
        and int(configured_dimension) != runtime.embedding_dimension
    ):
        raise ValueError(
            "EMBEDDING_DIM conflicts with FrameworkRuntimeConfig.embedding_dimension"
        )
    os.environ["EMBEDDING_DIM"] = str(runtime.embedding_dimension)
    try:
        from graphiti_core import Graphiti
        from graphiti_core.cross_encoder.openai_reranker_client import (
            OpenAIRerankerClient,
        )
        from graphiti_core.embedder.client import EmbedderClient
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
        from graphiti_core.nodes import EpisodeType
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise GraphitiDependencyError(
            'Graphiti adapter requires: python -m pip install -e ".[official-graphiti]"'
        ) from exc

    class SharedGraphitiEmbedder(EmbedderClient):
        async def create(
            self,
            input_data: str
            | list[str]
            | Iterable[int]
            | Iterable[Iterable[int]],
        ) -> list[float]:
            if isinstance(input_data, str):
                return _require_dimension(
                    embedder.embed_one(input_data),
                    runtime.embedding_dimension,
                )
            values = list(input_data)
            if values and all(isinstance(value, str) for value in values):
                return _require_dimension(
                    embedder.embed_one(str(values[0])),
                    runtime.embedding_dimension,
                )
            raise TypeError("Graphiti embedder only accepts text input")

        async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
            return [
                _require_dimension(vector, runtime.embedding_dimension)
                for vector in embedder.embed(input_data_list)
            ]

    llm_config = LLMConfig(
        api_key=runtime.vllm_api_key or "local-vllm-key",
        model=runtime.llm_model,
        small_model=runtime.llm_model,
        base_url=runtime.vllm_base_url,
        temperature=runtime.official_llm_temperature,
        max_tokens=runtime.official_llm_max_tokens,
    )
    openai_client = AsyncOpenAI(
        api_key=runtime.vllm_api_key or "local-vllm-key",
        base_url=runtime.vllm_base_url,
    )
    llm_client = OpenAIGenericClient(
        config=llm_config,
        client=openai_client,
        max_tokens=runtime.official_llm_max_tokens,
        structured_output_mode="json_schema",
    )
    graph = Graphiti(
        runtime.graphiti_neo4j_uri,
        runtime.graphiti_neo4j_user,
        runtime.graphiti_neo4j_password,
        llm_client=llm_client,
        embedder=SharedGraphitiEmbedder(),
        cross_encoder=OpenAIRerankerClient(
            config=llm_config,
            client=openai_client,
        ),
        store_raw_episode_content=True,
    )
    return graph, EpisodeType.message


def _event_metadata_text(event: Event, key: str) -> str | None:
    value = event.metadata.get(key)
    return value if isinstance(value, str) and value else None


def _require_dimension(vector: list[float], expected: int) -> list[float]:
    if len(vector) != expected:
        raise ValueError(
            f"Graphiti expected {expected}-D embeddings, received {len(vector)}-D"
        )
    return vector


def _json_value_or_text(value: object) -> JsonValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [_json_value_or_text(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _json_value_or_text(item)
            for key, item in value.items()
        }
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_value_or_text(model_dump(mode="json"))
    return json.dumps(str(value))
