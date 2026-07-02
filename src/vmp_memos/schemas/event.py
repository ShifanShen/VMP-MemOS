"""Agent event schema."""

from __future__ import annotations

from enum import Enum

from pydantic import Field, JsonValue

from vmp_memos.schemas.base import NonEmptyStr, TimestampedSchema, new_id


class EventType(str, Enum):
    """Event types accepted by the Phase 1 event collector contract."""

    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TASK_RESULT = "task_result"
    USER_CORRECTION = "user_correction"
    FILE_UPLOAD = "file_upload"
    BENCHMARK_SAMPLE = "benchmark_sample"
    BENCHMARK_QUERY = "benchmark_query"
    SYSTEM_FEEDBACK = "system_feedback"


class Event(TimestampedSchema):
    """A normalized event emitted by a user, agent, tool, or environment."""

    event_id: NonEmptyStr = Field(default_factory=lambda: new_id("evt"), frozen=True)
    session_id: NonEmptyStr
    task_id: NonEmptyStr | None = None
    event_type: EventType
    content: JsonValue
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
