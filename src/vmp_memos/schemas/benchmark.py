"""Benchmark input and per-sample result schemas."""

from __future__ import annotations

from pydantic import Field, FiniteFloat, JsonValue

from vmp_memos.schemas.base import (
    NonEmptyStr,
    NonNegativeFloat,
    NonNegativeInt,
    TimestampedSchema,
    new_id,
)
from vmp_memos.schemas.event import Event
from vmp_memos.schemas.operation import OperationType


class BenchmarkSample(TimestampedSchema):
    """A reproducible memory-lifecycle benchmark case."""

    sample_id: NonEmptyStr = Field(default_factory=lambda: new_id("sample"), frozen=True)
    events: list[Event] = Field(default_factory=list)
    query: NonEmptyStr
    gold_answer: NonEmptyStr | list[NonEmptyStr]
    gold_memory_ids: list[NonEmptyStr] = Field(default_factory=list)
    expected_operations: list[OperationType] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class BenchmarkResult(TimestampedSchema):
    """Serializable output produced for one benchmark sample and system."""

    result_id: NonEmptyStr = Field(default_factory=lambda: new_id("result"), frozen=True)
    sample_id: NonEmptyStr
    system_name: NonEmptyStr
    answer: str | None = None
    is_correct: bool | None = None
    retrieved_memory_ids: list[NonEmptyStr] = Field(default_factory=list)
    operations: list[OperationType] = Field(default_factory=list)
    metrics: dict[NonEmptyStr, FiniteFloat] = Field(default_factory=dict)
    token_count: NonNegativeInt = 0
    latency_ms: NonNegativeFloat = 0.0
    error: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

