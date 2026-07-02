"""Tests for the Phase 5 rule-based policy controller."""

import pytest

from vmp_memos.policy import (
    PolicyScoreContext,
    PolicyScoreName,
    RuleBasedPolicyController,
)
from vmp_memos.schemas import OperationType, PolicyFeatures


def test_write_score_formula_and_add_decision() -> None:
    controller = RuleBasedPolicyController()
    features = PolicyFeatures(
        importance=0.9,
        novelty=0.8,
        confidence=0.85,
        actionability=0.7,
        scope_match=1.0,
        redundancy=0.1,
        privacy_risk=0.0,
    )

    score = controller.score_write(features)
    decision = controller.decide_write(features)

    assert score.name == PolicyScoreName.WRITE
    assert score.score == pytest.approx(0.825)
    assert score.passed is True
    assert decision.op == OperationType.ADD
    assert decision.passed is True
    assert "WriteScore" in decision.reason


def test_write_decision_ignores_low_value_or_redundant_memory() -> None:
    controller = RuleBasedPolicyController()
    features = PolicyFeatures(
        importance=0.2,
        novelty=0.1,
        confidence=0.5,
        actionability=0.1,
        scope_match=0.5,
        redundancy=0.8,
        privacy_risk=0.2,
    )

    decision = controller.decide_write(features)

    assert decision.score == pytest.approx(0.07)
    assert decision.op == OperationType.IGNORE
    assert decision.passed is False


def test_retrieve_score_formula_and_decision() -> None:
    controller = RuleBasedPolicyController()
    features = PolicyFeatures(
        semantic_relevance=0.9,
        importance=0.8,
        scope_match=0.9,
        confidence=0.8,
        success_contribution=0.6,
        recency=0.7,
        contradiction=0.1,
        redundancy=0.2,
        token_cost=0.1,
    )

    score = controller.score_retrieve(features)
    decision = controller.decide_retrieve(features)

    assert score.score == pytest.approx(0.745)
    assert decision.op == OperationType.RETRIEVE
    assert decision.feature_snapshot["semantic_relevance"] == 0.9


def test_update_decision_requires_score_similarity_and_contradiction_gates() -> None:
    controller = RuleBasedPolicyController()
    features = PolicyFeatures(
        redundancy=0.8,
        contradiction=0.6,
        recency=1.0,
        confidence=0.9,
    )
    context = PolicyScoreContext(
        semantic_similarity_to_existing=0.8,
        source_priority=0.8,
    )

    decision = controller.decide_update(features, context)

    assert decision.score == pytest.approx(0.785)
    assert decision.op == OperationType.UPDATE
    assert decision.metadata["gates"] == {
        "score": True,
        "semantic_similarity": True,
        "contradiction": True,
    }

    weak_conflict = controller.decide_update(
        PolicyFeatures(
            redundancy=0.9,
            contradiction=0.2,
            recency=1.0,
            confidence=0.9,
        ),
        PolicyScoreContext(semantic_similarity_to_existing=0.9, source_priority=0.8),
    )
    assert weak_conflict.op == OperationType.IGNORE
    assert weak_conflict.metadata["gates"]["contradiction"] is False


def test_merge_archive_and_compress_decisions() -> None:
    controller = RuleBasedPolicyController()

    merge = controller.decide_merge(
        PolicyFeatures(
            redundancy=0.8,
            scope_match=0.9,
            contradiction=0.1,
        ),
        PolicyScoreContext(semantic_similarity=0.85),
    )
    assert merge.op == OperationType.MERGE
    assert merge.score == pytest.approx(0.8525)

    archive = controller.decide_archive(
        PolicyFeatures(
            staleness=1.0,
            redundancy=1.0,
            failure_contribution=1.0,
            importance=0.0,
        ),
        PolicyScoreContext(superseded=1.0),
    )
    assert archive.op == OperationType.ARCHIVE
    assert archive.score == 1.0

    compress = controller.decide_compress(
        PolicyFeatures(
            token_cost=0.9,
            access_frequency=0.8,
            actionability=0.6,
            scope_match=0.9,
        ),
        PolicyScoreContext(information_density=0.7),
    )
    assert compress.op == OperationType.COMPRESS
    assert compress.score == pytest.approx(0.79)


def test_decision_can_be_converted_to_memory_operation() -> None:
    controller = RuleBasedPolicyController()
    features = PolicyFeatures(
        importance=0.9,
        novelty=0.9,
        confidence=0.8,
        actionability=0.8,
        scope_match=1.0,
    )
    decision = controller.decide_write(features)

    operation = controller.to_operation(
        decision,
        target_memory_id="mem_001",
        source_event_id="evt_001",
        scope="career/agent-dev",
        backend="file",
    )

    assert operation.op == OperationType.ADD
    assert operation.policy_score == decision.score
    assert operation.confidence == features.confidence
    assert operation.target_memory_id == "mem_001"
    assert operation.source_event_id == "evt_001"
    assert operation.payload["decision_id"] == decision.decision_id
    assert operation.payload["score_name"] == "WriteScore"


def test_score_and_decision_json_round_trip_preserve_ids() -> None:
    controller = RuleBasedPolicyController()
    features = PolicyFeatures(
        semantic_relevance=0.9,
        importance=0.7,
        confidence=0.8,
    )

    score = controller.score_retrieve(features)
    decision = controller.decide_retrieve(features)

    restored_score = type(score).model_validate_json(score.model_dump_json())
    restored_decision = type(decision).model_validate_json(decision.model_dump_json())

    assert restored_score == score
    assert restored_score.score_id == score.score_id
    assert restored_decision == decision
    assert restored_decision.decision_id == decision.decision_id
