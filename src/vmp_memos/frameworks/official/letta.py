"""Official Letta adapter backed by a self-hosted Letta API server."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import JsonValue

from vmp_memos.embeddings import BaseEmbedder, OpenAICompatibleEmbedder
from vmp_memos.frameworks.base import (
    BaseMemoryFrameworkAdapter,
    FairnessLevel,
    RetrievedMemory,
)
from vmp_memos.frameworks.runtime import FrameworkRuntimeConfig
from vmp_memos.frameworks.text import dense_cosine, estimate_tokens, parse_date
from vmp_memos.longmemeval.converter import session_to_text
from vmp_memos.schemas import Event

LettaClientFactory = Callable[[FrameworkRuntimeConfig], Any]
_PROVENANCE_PATTERN = re.compile(
    r"\[SOURCE_SESSION_ID=(?P<session>[^;\]]+);\s*"
    r"SOURCE_DATE=(?P<date>[^\]]*)\]",
    flags=re.IGNORECASE,
)
_SYSTEM_PROMPT = """You are a memory curator for a longitudinal-memory benchmark.
Each user message contains one historical session and its provenance.
Use Letta's official memory tools to update the writable long_term_memory block
and, when useful, archival memory. Store only durable facts useful in later
questions. Resolve updates instead of retaining obsolete facts.

Every stored fact MUST be one line and MUST end with the exact provenance marker
provided in the input:
[SOURCE_SESSION_ID=<id>; SOURCE_DATE=<date>]

Keep the most recently updated and most important facts near the top of the
long_term_memory block. Do not answer questions during ingestion. After memory
updates are complete, respond only with ACK.
"""


class LettaDependencyError(RuntimeError):
    """Raised when the pinned official Letta client is unavailable."""


class LettaOfficialAdapter(BaseMemoryFrameworkAdapter):
    """Drive an official Letta agent and export its managed memory as evidence."""

    name = "letta"
    fairness_level = FairnessLevel.FULLY_CONTROLLED

    def __init__(
        self,
        *,
        runtime: FrameworkRuntimeConfig | None = None,
        embedder: BaseEmbedder | None = None,
        client_factory: LettaClientFactory | None = None,
    ) -> None:
        super().__init__()
        self.runtime = runtime or FrameworkRuntimeConfig.from_env()
        self.embedder = embedder
        self._client_factory = client_factory
        self._client: Any | None = None
        self._agent_id: str | None = None
        self._memories: list[dict[str, Any]] = []

    @property
    def memory_count(self) -> int:
        return len(self._memories)

    @property
    def total_tokens(self) -> int:
        return sum(estimate_tokens(str(item["content"])) for item in self._memories)

    @property
    def storage_size_bytes(self) -> int:
        content_bytes = sum(
            len(str(item["content"]).encode("utf-8")) for item in self._memories
        )
        vector_bytes = 4 * self.runtime.embedding_dimension * len(self._memories)
        return content_bytes + vector_bytes

    def _reset_impl(self) -> None:
        if not self.runtime.official_memory_infer:
            raise ValueError(
                "Letta requires official_memory_infer=True because its agent-managed "
                "memory is the evaluated mechanism"
            )
        if self._client is None:
            self._client = (
                self._client_factory(self.runtime)
                if self._client_factory is not None
                else _create_letta_client(self.runtime)
            )
        self._delete_agent()
        if self.embedder is None:
            self.embedder = OpenAICompatibleEmbedder(
                base_url=self.runtime.letta_embedding_base_url,
                model=self.runtime.embedding_model,
                dimension=self.runtime.embedding_dimension,
                timeout_seconds=120.0,
            )
        agent = self._client.agents.create(
            name=f"vmp-longmemeval-{uuid4().hex[:16]}",
            agent_type="memgpt_agent",
            system=_SYSTEM_PROMPT,
            llm_config=build_letta_llm_config(self.runtime),
            embedding_config=build_letta_embedding_config(self.runtime),
            memory_blocks=[
                {
                    "label": "long_term_memory",
                    "description": (
                        "Durable benchmark facts, one provenance-tagged fact per line."
                    ),
                    "value": "",
                    "limit": 30_000,
                }
            ],
            include_base_tools=True,
            include_base_tool_rules=True,
            message_buffer_autoclear=True,
            enable_sleeptime=False,
            metadata={
                "benchmark": "LongMemEval",
                "adapter": "vmp-memos",
            },
        )
        self._agent_id = _required_id(agent, "Letta agent")
        self._memories = []

    def _ingest_event_impl(self, event: Event) -> None:
        self._ingest([event])

    def _ingest_session_impl(self, events: list[Event]) -> None:
        if events:
            self._ingest(events)

    def _finalize_ingestion_impl(self) -> None:
        self._memories = self._snapshot()

    def _retrieve_impl(
        self,
        query: str,
        *,
        top_k: int,
        question_date: str | None,
        metadata: dict[str, JsonValue],
    ) -> list[RetrievedMemory]:
        if self._client is None or self._agent_id is None or self.embedder is None:
            raise RuntimeError("Letta adapter must be reset before retrieval")
        core = self._core_memories()
        core_vectors = (
            self.embedder.embed(
                [query, *[str(item["content"]) for item in core]]
            )
            if core
            else []
        )
        candidates: list[RetrievedMemory] = []
        if core_vectors:
            query_vector = core_vectors[0]
            for item, vector in zip(core, core_vectors[1:], strict=True):
                candidates.append(
                    _to_retrieved(
                        item,
                        score=dense_cosine(query_vector, vector),
                        strategy="letta_core_memory_bge",
                    )
                )
        for rank, item in enumerate(self._archival_search(query, top_k=top_k), start=1):
            raw_score = item.get("score")
            score = (
                float(raw_score)
                if isinstance(raw_score, int | float) and math.isfinite(raw_score)
                else 1.0 / rank
            )
            candidates.append(
                _to_retrieved(
                    item,
                    score=score,
                    strategy="letta_archival_search",
                )
            )
        candidates.sort(
            key=lambda item: (
                item.score,
                parse_date(item.source_date)
                or datetime.min.replace(tzinfo=UTC),
            ),
            reverse=True,
        )
        deduplicated: list[RetrievedMemory] = []
        seen: set[str] = set()
        for item in candidates:
            key = f"{item.memory_type}:{item.content.casefold()}"
            if key not in seen:
                seen.add(key)
                deduplicated.append(item)
        return deduplicated[:top_k]

    def stats(self) -> dict[str, JsonValue]:
        stats = super().stats()
        stats["storage_size_is_estimate"] = True
        stats["letta_base_url"] = self.runtime.letta_base_url
        stats["letta_server_version"] = self.runtime.letta_server_version
        return stats

    def close(self) -> None:
        try:
            self._delete_agent()
        finally:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()
            self._client = None

    def _ingest(self, events: list[Event]) -> None:
        if self._client is None or self._agent_id is None:
            raise RuntimeError("Letta adapter must be reset before ingestion")
        first = events[0]
        source_session_id = (
            _event_metadata_text(first, "history_session_id") or first.session_id
        )
        source_date = _event_metadata_text(first, "history_date") or ""
        marker = (
            f"[SOURCE_SESSION_ID={source_session_id}; "
            f"SOURCE_DATE={source_date}]"
        )
        body = (
            "INGEST THIS HISTORICAL SESSION INTO MEMORY.\n"
            f"Required provenance marker: {marker}\n\n"
            f"{session_to_text(events)}"
        )
        _send_letta_message(self._client, self._agent_id, body)
        self._memories = self._snapshot()

    def _snapshot(self) -> list[dict[str, Any]]:
        return [*self._core_memories(), *self._archival_memories()]

    def _core_memories(self) -> list[dict[str, Any]]:
        if self._client is None or self._agent_id is None:
            return []
        raw = self._client.agents.blocks.list(self._agent_id, limit=200)
        memories: list[dict[str, Any]] = []
        for block in _as_items(raw):
            label = str(getattr(block, "label", "") or "")
            if label != "long_term_memory":
                continue
            block_id = str(getattr(block, "id", "") or label)
            value = str(getattr(block, "value", "") or "")
            for line_number, line in enumerate(value.splitlines(), start=1):
                content = line.strip(" \t-*")
                if not content:
                    continue
                provenance = _provenance(content)
                if not provenance:
                    continue
                content = _strip_provenance(content)
                if content:
                    memories.append(
                        {
                            "id": f"{block_id}:{line_number}",
                            "content": content,
                            "source_session_id": provenance.get(
                                "source_session_id"
                            ),
                            "source_date": provenance.get("source_date"),
                            "memory_type": "letta_core_memory",
                        }
                    )
        return memories

    def _archival_memories(self) -> list[dict[str, Any]]:
        if self._client is None or self._agent_id is None:
            return []
        raw = self._client.agents.passages.list(self._agent_id, limit=100)
        return [
            item
            for value in _as_items(raw)
            if (item := _passage_item(value, score=None)) is not None
        ]

    def _archival_search(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        if self._client is None or self._agent_id is None:
            return []
        passages = getattr(self._client, "passages", None)
        search = getattr(passages, "search", None)
        if callable(search):
            raw = search(agent_id=self._agent_id, query=query, limit=top_k)
        else:
            raw = self._client.agents.passages.search(
                self._agent_id,
                query=query,
                limit=top_k,
            )
        return [
            item
            for value in _as_items(raw)
            if (item := _passage_item(value, score=_score(value))) is not None
        ]

    def _delete_agent(self) -> None:
        if self._client is not None and self._agent_id is not None:
            self._client.agents.delete(self._agent_id)
        self._agent_id = None
        self._memories = []


def build_letta_llm_config(runtime: FrameworkRuntimeConfig) -> dict[str, Any]:
    """Return Letta's OpenAI-compatible connection to the shared local vLLM."""

    return {
        "model": runtime.llm_model,
        "model_endpoint_type": "openai",
        "model_endpoint": runtime.vllm_base_url.rstrip("/"),
        "context_window": runtime.letta_context_window,
        "temperature": runtime.official_llm_temperature,
        "max_tokens": runtime.official_llm_max_tokens,
        "put_inner_thoughts_in_kwargs": False,
        "enable_reasoner": False,
        "parallel_tool_calls": False,
    }


def build_letta_embedding_config(
    runtime: FrameworkRuntimeConfig,
) -> dict[str, Any]:
    """Return the shared BGE-M3 OpenAI-compatible embedding connection."""

    return {
        "embedding_endpoint_type": "openai",
        "embedding_endpoint": runtime.letta_embedding_base_url,
        "embedding_model": runtime.embedding_model,
        "embedding_dim": runtime.embedding_dimension,
        "embedding_chunk_size": 300,
        "batch_size": 32,
    }


def _create_letta_client(runtime: FrameworkRuntimeConfig) -> Any:
    try:
        from letta_client import Letta
    except ImportError as exc:
        raise LettaDependencyError(
            'Letta adapter requires: python -m pip install -e ".[official-letta]"'
        ) from exc
    return Letta(
        base_url=runtime.letta_base_url,
        api_key=runtime.letta_api_key,
    )


def _send_letta_message(client: Any, agent_id: str, content: str) -> Any:
    try:
        return client.agents.messages.create(agent_id=agent_id, input=content)
    except TypeError:
        return client.agents.messages.create(
            agent_id=agent_id,
            messages=[{"role": "user", "content": content}],
        )


def _passage_item(value: object, *, score: float | None) -> dict[str, Any] | None:
    passage = getattr(value, "passage", None) or value
    memory_id = str(getattr(passage, "id", "") or "").strip()
    content = str(getattr(passage, "text", "") or "").strip()
    if not memory_id or not content:
        return None
    provenance = _provenance(content)
    stripped = _strip_provenance(content)
    if not stripped:
        return None
    return {
        "id": memory_id,
        "content": stripped,
        "source_session_id": provenance.get("source_session_id"),
        "source_date": provenance.get("source_date"),
        "memory_type": "letta_archival_memory",
        "score": score,
    }


def _to_retrieved(
    item: dict[str, Any],
    *,
    score: float,
    strategy: str,
) -> RetrievedMemory:
    content = str(item["content"])
    return RetrievedMemory(
        memory_id=str(item["id"]),
        content=content,
        score=score,
        source_session_id=_optional_text(item.get("source_session_id")),
        source_date=_optional_text(item.get("source_date")),
        memory_type=str(item["memory_type"]),
        token_count=estimate_tokens(content),
        metadata={
            "retrieval_strategy": strategy,
            "official_framework": "letta",
        },
    )


def _as_items(value: object) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    for attribute in ("items", "results", "data"):
        items = getattr(value, attribute, None)
        if isinstance(items, list):
            return items
    if isinstance(value, Iterable) and not isinstance(value, str | bytes | dict):
        return list(value)
    return []


def _provenance(content: str) -> dict[str, str | None]:
    matches = list(_PROVENANCE_PATTERN.finditer(content))
    if not matches:
        return {}
    match = max(
        matches,
        key=lambda item: parse_date(item.group("date"))
        or datetime.min.replace(tzinfo=UTC),
    )
    return {
        "source_session_id": match.group("session").strip() or None,
        "source_date": match.group("date").strip() or None,
    }


def _strip_provenance(content: str) -> str:
    return _PROVENANCE_PATTERN.sub("", content).strip(" \t-")


def _score(value: object) -> float | None:
    score = getattr(value, "score", None)
    return float(score) if isinstance(score, int | float) else None


def _required_id(value: object, label: str) -> str:
    identifier = str(getattr(value, "id", "") or "").strip()
    if not identifier:
        raise RuntimeError(f"{label} response did not contain an id")
    return identifier


def _event_metadata_text(event: Event, key: str) -> str | None:
    value = event.metadata.get(key)
    return value if isinstance(value, str) and value else None


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None

