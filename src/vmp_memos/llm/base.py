"""Shared LLM request and response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue

from vmp_memos.schemas.base import (
    NonEmptyStr,
    NonNegativeFloat,
    NonNegativeInt,
    SchemaModel,
    TimestampedSchema,
    new_id,
)

ChatRole = Literal["system", "user", "assistant", "tool"]


class ChatMessage(SchemaModel):
    """One OpenAI-compatible chat message."""

    role: ChatRole
    content: NonEmptyStr


class LLMGenerationConfig(SchemaModel):
    """Sampling controls for text generation."""

    max_tokens: NonNegativeInt = 512
    temperature: NonNegativeFloat = 0.2
    top_p: NonNegativeFloat = 0.95
    stop: list[NonEmptyStr] = Field(default_factory=list)


class LLMResponse(TimestampedSchema):
    """Normalized response from an LLM provider."""

    response_id: NonEmptyStr = Field(default_factory=lambda: new_id("llm"), frozen=True)
    provider: NonEmptyStr
    model: NonEmptyStr
    text: str
    finish_reason: str | None = None
    usage: dict[str, JsonValue] = Field(default_factory=dict)
    raw_response: dict[str, JsonValue] = Field(default_factory=dict)
