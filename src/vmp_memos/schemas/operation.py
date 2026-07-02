"""Memory operation and retrieval schemas."""

from __future__ import annotations

from enum import Enum

from pydantic import Field, JsonValue

from vmp_memos.schemas.base import (
    NonEmptyStr,
    NonNegativeFloat,
    NonNegativeInt,
    Score,
    TimestampedSchema,
    new_id,
)
from vmp_memos.schemas.memory import MemoryItem


class OperationType(str, Enum):
    """Memory-operation DSL, including operations reserved for later phases."""

    ADD = "ADD"
    UPDATE = "UPDATE"
    MERGE = "MERGE"
    SPLIT = "SPLIT"
    DELETE = "DELETE"
    ARCHIVE = "ARCHIVE"
    EXPIRE = "EXPIRE"
    LOCK = "LOCK"
    PROMOTE = "PROMOTE"
    DEMOTE = "DEMOTE"
    COMPRESS = "COMPRESS"
    RETRIEVE = "RETRIEVE"
    PIN = "PIN"
    DREAM = "DREAM"
    VERIFY = "VERIFY"
    IGNORE = "IGNORE"


class MemoryOperation(TimestampedSchema):
    """An auditable instruction emitted by a memory policy controller."""

    op_id: NonEmptyStr = Field(default_factory=lambda: new_id("op"), frozen=True)
    op: OperationType
    target_memory_id: NonEmptyStr | None = None
    source_memory_ids: list[NonEmptyStr] = Field(default_factory=list)
    source_event_id: NonEmptyStr | None = None
    reason: NonEmptyStr
    policy_score: Score
    confidence: Score
    scope: NonEmptyStr = "global"
    backend: NonEmptyStr | None = None
    payload: dict[str, JsonValue] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class RetrievalResult(TimestampedSchema):
    """One ranked retrieval response plus its cost and provenance metadata."""

    retrieval_id: NonEmptyStr = Field(default_factory=lambda: new_id("ret"), frozen=True)
    query: NonEmptyStr
    memory_ids: list[NonEmptyStr] = Field(default_factory=list)
    items: list[MemoryItem] = Field(default_factory=list)
    scores: dict[NonEmptyStr, Score] = Field(default_factory=dict)
    token_count: NonNegativeInt = 0
    latency_ms: NonNegativeFloat = 0.0
    backend: NonEmptyStr
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

