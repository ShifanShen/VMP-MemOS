"""Base classes shared by LongMemEval memory-framework adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from time import perf_counter

from pydantic import Field, FiniteFloat, JsonValue

from vmp_memos.frameworks.text import estimate_tokens
from vmp_memos.longmemeval.converter import session_to_text
from vmp_memos.schemas import Event
from vmp_memos.schemas.base import (
    NonEmptyStr,
    NonNegativeFloat,
    NonNegativeInt,
    SchemaModel,
)


class FairnessLevel(str, Enum):
    """Fairness/control labels used in paper tables."""

    FULLY_CONTROLLED = "fully_controlled"
    PARTIALLY_CONTROLLED = "partially_controlled"
    STYLE_REIMPLEMENTATION = "style_reimplementation"
    UNAVAILABLE = "unavailable"


class RetrievedMemory(SchemaModel):
    """Uniform evidence item returned by every memory framework adapter."""

    memory_id: NonEmptyStr
    content: NonEmptyStr
    score: FiniteFloat
    source_session_id: str | None = None
    source_turn_id: str | None = None
    source_date: str | None = None
    memory_type: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    token_count: NonNegativeInt = 0


class MemoryChunk(SchemaModel):
    """Internal indexed memory chunk used by built-in adapters."""

    memory_id: NonEmptyStr
    content: NonEmptyStr
    source_session_id: str | None = None
    source_turn_id: str | None = None
    source_date: str | None = None
    memory_type: str = "session"
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    token_count: NonNegativeInt = 0
    content_embedding: list[FiniteFloat] = Field(default_factory=list)

    def to_retrieved(
        self,
        *,
        score: float,
        metadata: dict[str, JsonValue] | None = None,
    ) -> RetrievedMemory:
        """Convert this indexed chunk into a public retrieval result."""

        merged_metadata = dict(self.metadata)
        if metadata:
            merged_metadata.update(metadata)
        return RetrievedMemory(
            memory_id=self.memory_id,
            content=self.content,
            score=float(score),
            source_session_id=self.source_session_id,
            source_turn_id=self.source_turn_id,
            source_date=self.source_date,
            memory_type=self.memory_type,
            metadata=merged_metadata,
            token_count=self.token_count,
        )


class AdapterStats(SchemaModel):
    """Common stats returned by built-in adapters."""

    name: NonEmptyStr
    fairness_level: FairnessLevel
    memory_count: NonNegativeInt = 0
    total_tokens: NonNegativeInt = 0
    storage_size_bytes: NonNegativeInt = 0
    ingestion_events: NonNegativeInt = 0
    ingestion_sessions: NonNegativeInt = 0
    retrieval_calls: NonNegativeInt = 0
    total_ingest_latency_ms: NonNegativeFloat = 0.0
    total_retrieval_latency_ms: NonNegativeFloat = 0.0
    workspace_dir: str | None = None


class BaseMemoryFrameworkAdapter(ABC):
    """Minimal adapter contract for fair memory framework comparison."""

    name: str
    fairness_level: FairnessLevel = FairnessLevel.FULLY_CONTROLLED

    def __init__(self) -> None:
        self.workspace_dir: Path | None = None
        self._ingestion_events = 0
        self._ingestion_sessions = 0
        self._retrieval_calls = 0
        self._total_ingest_latency_ms = 0.0
        self._total_retrieval_latency_ms = 0.0

    def reset(self, workspace_dir: str | Path) -> None:
        """Reset all sample-local state for a fresh LongMemEval question."""

        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self._ingestion_events = 0
        self._ingestion_sessions = 0
        self._retrieval_calls = 0
        self._total_ingest_latency_ms = 0.0
        self._total_retrieval_latency_ms = 0.0
        self._reset_impl()

    def ingest_event(self, event: Event) -> None:
        """Ingest one event, usually for turn-level experiments."""

        started_at = perf_counter()
        self._ingest_event_impl(event)
        self._ingestion_events += 1
        self._total_ingest_latency_ms += _elapsed_ms(started_at)

    def ingest_session(self, events: list[Event]) -> None:
        """Ingest one session group, usually as a single session-level chunk."""

        started_at = perf_counter()
        self._ingest_session_impl(events)
        self._ingestion_sessions += 1
        self._ingestion_events += len(events)
        self._total_ingest_latency_ms += _elapsed_ms(started_at)

    def finalize_ingestion(self) -> None:
        """Finish deferred indexing while charging its cost to ingestion."""

        started_at = perf_counter()
        self._finalize_ingestion_impl()
        self._total_ingest_latency_ms += _elapsed_ms(started_at)

    def retrieve(
        self,
        query: str,
        top_k: int,
        question_date: str | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ) -> list[RetrievedMemory]:
        """Retrieve evidence using a framework-specific strategy."""

        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        started_at = perf_counter()
        results = self._retrieve_impl(
            query,
            top_k=top_k,
            question_date=question_date,
            metadata=metadata or {},
        )
        self._retrieval_calls += 1
        self._total_retrieval_latency_ms += _elapsed_ms(started_at)
        return results

    def stats(self) -> dict[str, JsonValue]:
        """Return common JSON-safe adapter stats."""

        stats = AdapterStats(
            name=self.name,
            fairness_level=self.fairness_level,
            memory_count=self.memory_count,
            total_tokens=self.total_tokens,
            storage_size_bytes=self.storage_size_bytes,
            ingestion_events=self._ingestion_events,
            ingestion_sessions=self._ingestion_sessions,
            retrieval_calls=self._retrieval_calls,
            total_ingest_latency_ms=self._total_ingest_latency_ms,
            total_retrieval_latency_ms=self._total_retrieval_latency_ms,
            workspace_dir=str(self.workspace_dir) if self.workspace_dir else None,
        )
        return stats.model_dump(mode="json")

    def close(self) -> None:
        """Release resources held by the adapter."""

    @property
    def storage_size_bytes(self) -> int:
        """Return physical workspace bytes when an adapter persists data."""

        if self.workspace_dir is None or not self.workspace_dir.exists():
            return 0
        return sum(
            path.stat().st_size
            for path in self.workspace_dir.rglob("*")
            if path.is_file()
        )

    @property
    @abstractmethod
    def memory_count(self) -> int:
        """Return number of indexed memories."""

    @property
    @abstractmethod
    def total_tokens(self) -> int:
        """Return approximate total indexed tokens."""

    @abstractmethod
    def _reset_impl(self) -> None:
        """Adapter-specific reset hook."""

    @abstractmethod
    def _ingest_event_impl(self, event: Event) -> None:
        """Adapter-specific event ingestion."""

    @abstractmethod
    def _ingest_session_impl(self, events: list[Event]) -> None:
        """Adapter-specific session ingestion."""

    def _finalize_ingestion_impl(self) -> None:
        """Adapter-specific deferred indexing hook."""

    @abstractmethod
    def _retrieve_impl(
        self,
        query: str,
        *,
        top_k: int,
        question_date: str | None,
        metadata: dict[str, JsonValue],
    ) -> list[RetrievedMemory]:
        """Adapter-specific retrieval implementation."""


class InMemorySessionAdapter(BaseMemoryFrameworkAdapter):
    """Convenience base class for deterministic in-memory adapters."""

    memory_type = "session"

    def __init__(self) -> None:
        super().__init__()
        self.chunks: list[MemoryChunk] = []

    @property
    def memory_count(self) -> int:
        """Return number of indexed chunks."""

        return len(self.chunks)

    @property
    def total_tokens(self) -> int:
        """Return approximate indexed token count."""

        return sum(chunk.token_count for chunk in self.chunks)

    @property
    def storage_size_bytes(self) -> int:
        """Return logical UTF-8 content and dense-vector storage size."""

        content_bytes = sum(len(chunk.content.encode("utf-8")) for chunk in self.chunks)
        vector_bytes = sum(4 * len(chunk.content_embedding) for chunk in self.chunks)
        return content_bytes + vector_bytes

    def _reset_impl(self) -> None:
        self.chunks = []

    def _ingest_event_impl(self, event: Event) -> None:
        content = str(event.content).strip()
        if not content:
            return
        self.chunks.append(chunk_from_event(event))

    def _ingest_session_impl(self, events: list[Event]) -> None:
        chunk = chunk_from_session(events, memory_type=self.memory_type)
        if chunk is not None:
            self.chunks.append(chunk)


def chunk_from_event(event: Event) -> MemoryChunk:
    """Create a turn-level chunk from one event."""

    history_session_id = _metadata_str(event, "history_session_id")
    turn_idx = event.metadata.get("turn_idx")
    source_turn_id = None if turn_idx is None else f"{history_session_id}:{turn_idx}"
    content = str(event.content).strip()
    return MemoryChunk(
        memory_id=event.event_id,
        content=content,
        source_session_id=history_session_id or event.session_id,
        source_turn_id=source_turn_id,
        source_date=_metadata_str(event, "history_date"),
        memory_type="turn",
        metadata=_json_metadata(event),
        token_count=estimate_tokens(content),
    )


def chunk_from_session(events: list[Event], *, memory_type: str = "session") -> MemoryChunk | None:
    """Create a session-level chunk from a group of events."""

    if not events:
        return None
    content = session_to_text(events).strip()
    if not content:
        return None
    first = events[0]
    history_session_id = _metadata_str(first, "history_session_id") or first.session_id
    metadata = _json_metadata(first)
    metadata["turn_count"] = len(events)
    metadata["event_ids"] = [event.event_id for event in events]
    metadata["roles"] = [
        str(event.metadata.get("role", event.event_type.value)) for event in events
    ]
    return MemoryChunk(
        memory_id=history_session_id,
        content=content,
        source_session_id=history_session_id,
        source_date=_metadata_str(first, "history_date"),
        memory_type=memory_type,
        metadata=metadata,
        token_count=estimate_tokens(content),
    )


def _metadata_str(event: Event, key: str) -> str | None:
    value = event.metadata.get(key)
    return value if isinstance(value, str) and value else None


def _json_metadata(event: Event) -> dict[str, JsonValue]:
    return {key: value for key, value in event.metadata.items()}


def _elapsed_ms(started_at: float) -> float:
    return (perf_counter() - started_at) * 1000.0
