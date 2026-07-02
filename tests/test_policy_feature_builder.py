"""Tests for deterministic policy feature construction."""

from datetime import UTC, datetime, timedelta

from vmp_memos.policy import PolicyFeatureBuilder, PolicyFeatureContext
from vmp_memos.schemas import (
    MemoryCandidate,
    MemoryItem,
    MemorySource,
    MemoryStatus,
    PolicyFeatures,
)


def make_memory(
    *,
    content: str = "User considered Java backend work.",
    scope: str = "career/agent-dev",
    embedding: list[float] | None = None,
    access_count: int = 0,
    updated_at: datetime | None = None,
    status: str = "active",
    features: PolicyFeatures | None = None,
) -> MemoryItem:
    """Create a representative memory item for feature-builder tests."""

    metadata = {
        "access_count": access_count,
        "status": status,
    }
    if updated_at is not None:
        metadata["updated_at"] = updated_at
    return MemoryItem.model_validate(
        {
            "type": "semantic",
            "scope": scope,
            "content": content,
            "source": MemorySource(source_type="test"),
            "content_embedding": embedding or [],
            "features": features or PolicyFeatures(importance=0.6, confidence=0.8),
            "metadata": metadata,
        }
    )


def test_candidate_features_use_embeddings_and_change_heuristics() -> None:
    builder = PolicyFeatureBuilder()
    old_memory = make_memory(embedding=[0.0, 1.0, 0.0])
    candidate = MemoryCandidate(
        source_event_id="evt_policy",
        memory_type="semantic",
        content="User no longer wants all in Java; now focuses on Agent and LLM apps.",
        scope="career/agent-dev",
        confidence=0.92,
        importance=0.88,
    )

    features = builder.build_for_candidate(
        candidate,
        PolicyFeatureContext(
            query="Agent LLM direction",
            target_scope="career/agent-dev",
            query_embedding=[1.0, 0.0, 0.0],
            subject_embedding=[1.0, 0.0, 0.0],
            existing_memories=[old_memory],
            now=candidate.timestamp,
        ),
    )

    assert features.semantic_relevance == 1.0
    assert features.importance == 0.88
    assert features.confidence == 0.92
    assert features.scope_match == 1.0
    assert features.novelty == 1.0
    assert features.contradiction >= 0.8
    assert features.actionability >= 0.6


def test_memory_features_cover_recency_staleness_access_and_token_cost() -> None:
    now = datetime(2026, 6, 25, tzinfo=UTC)
    memory = make_memory(
        content="This is an outdated temporary note. " * 80,
        access_count=20,
        updated_at=now - timedelta(days=120),
    )
    builder = PolicyFeatureBuilder()

    features = builder.build_for_memory(
        memory,
        PolicyFeatureContext(
            target_scope="career/agent-dev",
            now=now,
            token_budget=128,
        ),
    )

    assert 0.45 <= features.recency <= 0.55
    assert features.staleness >= 0.75
    assert features.access_frequency == 1.0
    assert features.token_cost > 0.5
    assert features.stability < 0.7


def test_lexical_redundancy_fallback_does_not_require_embeddings() -> None:
    builder = PolicyFeatureBuilder()
    existing = make_memory(content="Agent memory retrieval needs durable storage.")
    candidate = MemoryCandidate(
        source_event_id="evt_policy",
        memory_type="semantic",
        content="Agent memory retrieval needs durable storage.",
        scope="career/agent-dev",
        confidence=0.9,
        importance=0.9,
    )

    features = builder.build_for_candidate(
        candidate,
        PolicyFeatureContext(existing_memories=[existing]),
    )

    assert features.redundancy >= 0.9
    assert features.novelty <= 0.1


def test_enrich_memory_sets_policy_embedding_without_changing_identity() -> None:
    builder = PolicyFeatureBuilder()
    memory = make_memory(
        content="When debugging this project, reuse the vector backend smoke test.",
        embedding=[1.0, 0.0],
        features=PolicyFeatures(importance=0.7, confidence=0.9),
    )

    enriched = builder.enrich_memory(
        memory,
        PolicyFeatureContext(
            query="debug vector backend",
            query_embedding=[1.0, 0.0],
            target_scope=memory.scope,
            now=memory.metadata.updated_at,
        ),
    )

    assert enriched.id == memory.id
    assert enriched.content == memory.content
    assert enriched.features.semantic_relevance == 1.0
    assert enriched.policy_embedding == list(enriched.features.as_vector())
    assert len(enriched.policy_embedding) == len(PolicyFeatures.FEATURE_NAMES)


def test_privacy_risk_and_archived_status_are_reflected() -> None:
    builder = PolicyFeatureBuilder()
    memory = make_memory(
        content="API token is sk-test-secret and email is user@example.com.",
        status=MemoryStatus.ARCHIVED.value,
    )

    features = builder.build_for_memory(memory)

    assert features.privacy_risk >= 0.9
    assert features.staleness == 1.0


def test_explain_returns_highest_features() -> None:
    features = PolicyFeatures(
        importance=0.9,
        confidence=0.8,
        novelty=0.7,
        privacy_risk=0.95,
    )

    explanation = PolicyFeatureBuilder.explain(features, top_n=2)

    assert list(explanation) == ["privacy_risk", "importance"]
