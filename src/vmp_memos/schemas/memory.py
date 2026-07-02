"""Candidate, policy-feature, and stored-memory schemas."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, ClassVar

from pydantic import AwareDatetime, Field, FiniteFloat, JsonValue, field_validator

from vmp_memos.schemas.base import (
    NonEmptyStr,
    NonNegativeInt,
    SchemaModel,
    Score,
    TimestampedSchema,
    new_id,
    utc_now,
)


class MemoryType(str, Enum):
    """Supported typed-memory categories."""

    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    REFLECTIVE = "reflective"
    RESOURCE = "resource"


class MemoryStatus(str, Enum):
    """Lifecycle states represented by the schema layer."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    EXPIRED = "expired"
    LOCKED = "locked"


class PolicyFeatures(TimestampedSchema):
    """Explainable management-state vector for one candidate or memory item."""

    FEATURE_NAMES: ClassVar[tuple[str, ...]] = (
        "semantic_relevance",
        "importance",
        "confidence",
        "recency",
        "stability",
        "novelty",
        "redundancy",
        "contradiction",
        "staleness",
        "access_frequency",
        "success_contribution",
        "failure_contribution",
        "token_cost",
        "scope_match",
        "actionability",
        "privacy_risk",
    )

    feature_id: NonEmptyStr = Field(default_factory=lambda: new_id("feat"), frozen=True)
    semantic_relevance: Score = 0.0
    importance: Score = 0.0
    confidence: Score = 0.0
    recency: Score = 0.0
    stability: Score = 0.0
    novelty: Score = 0.0
    redundancy: Score = 0.0
    contradiction: Score = 0.0
    staleness: Score = 0.0
    access_frequency: Score = 0.0
    success_contribution: Score = 0.0
    failure_contribution: Score = 0.0
    token_cost: Score = 0.0
    scope_match: Score = 0.0
    actionability: Score = 0.0
    privacy_risk: Score = 0.0

    def as_vector(self) -> tuple[float, ...]:
        """Return features in the canonical order specified by the design document."""

        return tuple(float(getattr(self, name)) for name in self.FEATURE_NAMES)


class MemoryCandidate(TimestampedSchema):
    """A proposed memory extracted from a source event."""

    candidate_id: NonEmptyStr = Field(default_factory=lambda: new_id("cand"), frozen=True)
    source_event_id: NonEmptyStr
    memory_type: MemoryType
    content: NonEmptyStr
    scope: NonEmptyStr = "global"
    tags: list[NonEmptyStr] = Field(default_factory=list)
    confidence: Score = 0.0
    importance: Score = 0.0
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("tags")
    @classmethod
    def deduplicate_tags(cls, tags: list[str]) -> list[str]:
        """Keep tag order stable while removing accidental duplicates."""

        return list(dict.fromkeys(tags))


class MemorySource(SchemaModel):
    """Provenance for a stored memory item."""

    event_id: NonEmptyStr | None = None
    source_type: NonEmptyStr
    uri: NonEmptyStr | None = None


class MemoryMetadata(SchemaModel):
    """Mutable lifecycle metadata stored alongside memory content."""

    created_at: AwareDatetime = Field(default_factory=utc_now)
    updated_at: AwareDatetime = Field(default_factory=utc_now)
    last_accessed_at: AwareDatetime | None = None
    access_count: NonNegativeInt = 0
    version: Annotated[int, Field(ge=1)] = 1
    status: MemoryStatus = MemoryStatus.ACTIVE
    attributes: dict[str, JsonValue] = Field(default_factory=dict)


class MemoryLinks(SchemaModel):
    """Relationships between memory items."""

    related: list[NonEmptyStr] = Field(default_factory=list)
    supersedes: list[NonEmptyStr] = Field(default_factory=list)
    superseded_by: list[NonEmptyStr] = Field(default_factory=list)


class MemoryItem(TimestampedSchema):
    """Canonical representation of one persisted memory."""

    id: NonEmptyStr = Field(default_factory=lambda: new_id("mem"), frozen=True)
    type: MemoryType
    scope: NonEmptyStr = "global"
    content: NonEmptyStr
    summary: NonEmptyStr | None = None
    source: MemorySource
    content_embedding: list[FiniteFloat] = Field(default_factory=list)
    policy_embedding: list[FiniteFloat] = Field(default_factory=list)
    features: PolicyFeatures = Field(default_factory=PolicyFeatures)
    metadata: MemoryMetadata = Field(default_factory=MemoryMetadata)
    links: MemoryLinks = Field(default_factory=MemoryLinks)
