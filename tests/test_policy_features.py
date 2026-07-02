"""Tests for the explainable policy feature schema."""

from datetime import UTC

import pytest
from pydantic import ValidationError

from vmp_memos.schemas import PolicyFeatures


def test_policy_features_have_canonical_vector_order() -> None:
    features = PolicyFeatures(
        semantic_relevance=0.1,
        importance=0.2,
        confidence=0.3,
        recency=0.4,
        stability=0.5,
        novelty=0.6,
        redundancy=0.7,
        contradiction=0.8,
        staleness=0.9,
        access_frequency=1.0,
        success_contribution=0.9,
        failure_contribution=0.8,
        token_cost=0.7,
        scope_match=0.6,
        actionability=0.5,
        privacy_risk=0.4,
    )

    assert features.as_vector() == (
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        0.9,
        0.8,
        0.7,
        0.6,
        0.5,
        0.4,
    )
    assert features.feature_id.startswith("feat_")
    assert features.timestamp.tzinfo is not None
    assert features.timestamp.utcoffset() == UTC.utcoffset(features.timestamp)


@pytest.mark.parametrize("value", [-0.01, 1.01, float("inf"), float("nan")])
def test_policy_features_reject_values_outside_unit_interval(value: float) -> None:
    with pytest.raises(ValidationError):
        PolicyFeatures(importance=value)


def test_policy_features_json_round_trip_preserves_stable_id() -> None:
    features = PolicyFeatures(importance=0.8)

    restored = PolicyFeatures.model_validate_json(features.model_dump_json())

    assert restored == features
    assert restored.feature_id == features.feature_id
    assert restored.timestamp == features.timestamp
