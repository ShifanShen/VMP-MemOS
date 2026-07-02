"""Rule-based policy controller for explainable memory operation decisions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import Enum

from pydantic import Field, FiniteFloat, JsonValue

from vmp_memos.schemas import MemoryOperation, OperationType, PolicyFeatures
from vmp_memos.schemas.base import (
    NonEmptyStr,
    SchemaModel,
    Score,
    TimestampedSchema,
    new_id,
)


class PolicyScoreName(str, Enum):
    """Named rule scores implemented by the Phase 5 policy controller."""

    WRITE = "WriteScore"
    RETRIEVE = "RetrieveScore"
    UPDATE = "UpdateScore"
    MERGE = "MergeScore"
    ARCHIVE = "ArchiveScore"
    COMPRESS = "CompressScore"


class PolicyScoreContext(SchemaModel):
    """Optional external values needed by specific policy scores."""

    semantic_similarity_to_existing: Score | None = None
    source_priority: Score = 0.5
    semantic_similarity: Score | None = None
    information_density: Score | None = None
    superseded: Score = 0.0
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class RuleBasedPolicyControllerConfig(SchemaModel):
    """Thresholds and gates for deterministic policy decisions."""

    write_threshold: Score = 0.65
    retrieve_threshold: Score = 0.05
    update_threshold: Score = 0.65
    update_similarity_threshold: Score = 0.70
    update_contradiction_threshold: Score = 0.45
    merge_threshold: Score = 0.75
    archive_threshold: Score = 0.80
    compress_threshold: Score = 0.65


class PolicyScoreResult(TimestampedSchema):
    """One weighted policy score with its explainable feature contributions."""

    score_id: NonEmptyStr = Field(default_factory=lambda: new_id("score"), frozen=True)
    name: PolicyScoreName
    score: Score
    threshold: Score
    passed: bool
    contributions: dict[NonEmptyStr, FiniteFloat]
    reason: NonEmptyStr
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class PolicyDecision(TimestampedSchema):
    """Controller output consumed later by the operation executor."""

    decision_id: NonEmptyStr = Field(default_factory=lambda: new_id("dec"), frozen=True)
    op: OperationType
    score_name: PolicyScoreName
    score: Score
    threshold: Score
    passed: bool
    confidence: Score
    reason: NonEmptyStr
    contributions: dict[NonEmptyStr, FiniteFloat]
    feature_snapshot: dict[NonEmptyStr, Score]
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class RuleBasedPolicyController:
    """Compute rule-based VMP scores and convert them into operation decisions."""

    def __init__(self, config: RuleBasedPolicyControllerConfig | None = None) -> None:
        self.config = config or RuleBasedPolicyControllerConfig()

    def score_write(self, features: PolicyFeatures) -> PolicyScoreResult:
        """Compute WriteScore for ADD admission."""

        contributions = {
            "importance": 0.30 * features.importance,
            "novelty": 0.25 * features.novelty,
            "confidence": 0.20 * features.confidence,
            "actionability": 0.15 * features.actionability,
            "scope_match": 0.10 * features.scope_match,
            "redundancy_penalty": -0.20 * features.redundancy,
            "privacy_risk_penalty": -0.10 * features.privacy_risk,
        }
        return self._score(
            PolicyScoreName.WRITE,
            contributions,
            threshold=self.config.write_threshold,
        )

    def score_retrieve(self, features: PolicyFeatures) -> PolicyScoreResult:
        """Compute RetrieveScore for retrieval admission and reranking."""

        contributions = {
            "semantic_relevance": 0.30 * features.semantic_relevance,
            "importance": 0.20 * features.importance,
            "scope_match": 0.15 * features.scope_match,
            "confidence": 0.10 * features.confidence,
            "success_contribution": 0.10 * features.success_contribution,
            "recency": 0.10 * features.recency,
            "contradiction_penalty": -0.15 * features.contradiction,
            "redundancy_penalty": -0.05 * features.redundancy,
            "token_cost_penalty": -0.05 * features.token_cost,
        }
        return self._score(
            PolicyScoreName.RETRIEVE,
            contributions,
            threshold=self.config.retrieve_threshold,
        )

    def score_update(
        self,
        features: PolicyFeatures,
        context: PolicyScoreContext | None = None,
    ) -> PolicyScoreResult:
        """Compute UpdateScore for replacing or revising an existing memory."""

        context = context or PolicyScoreContext()
        similarity = context.semantic_similarity_to_existing
        if similarity is None:
            similarity = features.redundancy
        contributions = {
            "semantic_similarity_to_existing": 0.30 * similarity,
            "contradiction": 0.30 * features.contradiction,
            "recency": 0.20 * features.recency,
            "source_priority": 0.15 * context.source_priority,
            "confidence": 0.05 * features.confidence,
        }
        return self._score(
            PolicyScoreName.UPDATE,
            contributions,
            threshold=self.config.update_threshold,
            metadata={
                "semantic_similarity_to_existing": similarity,
                "source_priority": context.source_priority,
            },
        )

    def score_merge(
        self,
        features: PolicyFeatures,
        context: PolicyScoreContext | None = None,
    ) -> PolicyScoreResult:
        """Compute MergeScore for redundant non-conflicting memories."""

        context = context or PolicyScoreContext()
        similarity = context.semantic_similarity
        if similarity is None:
            similarity = features.redundancy
        low_conflict = 1.0 - features.contradiction
        contributions = {
            "semantic_similarity": 0.35 * similarity,
            "redundancy": 0.30 * features.redundancy,
            "scope_match": 0.20 * features.scope_match,
            "low_conflict": 0.15 * low_conflict,
        }
        return self._score(
            PolicyScoreName.MERGE,
            contributions,
            threshold=self.config.merge_threshold,
            metadata={
                "semantic_similarity": similarity,
                "low_conflict": low_conflict,
            },
        )

    def score_archive(
        self,
        features: PolicyFeatures,
        context: PolicyScoreContext | None = None,
    ) -> PolicyScoreResult:
        """Compute ArchiveScore for low-value, stale, or superseded memories."""

        context = context or PolicyScoreContext()
        low_importance = 1.0 - features.importance
        contributions = {
            "staleness": 0.25 * features.staleness,
            "redundancy": 0.25 * features.redundancy,
            "negative_contribution": 0.20 * features.failure_contribution,
            "low_importance": 0.15 * low_importance,
            "superseded": 0.15 * context.superseded,
        }
        return self._score(
            PolicyScoreName.ARCHIVE,
            contributions,
            threshold=self.config.archive_threshold,
            metadata={
                "low_importance": low_importance,
                "superseded": context.superseded,
            },
        )

    def score_compress(
        self,
        features: PolicyFeatures,
        context: PolicyScoreContext | None = None,
    ) -> PolicyScoreResult:
        """Compute CompressScore for high-cost but useful memories."""

        context = context or PolicyScoreContext()
        information_density = context.information_density
        if information_density is None:
            information_density = self._fallback_information_density(features)
        contributions = {
            "token_cost": 0.30 * features.token_cost,
            "access_frequency": 0.25 * features.access_frequency,
            "information_density": 0.20 * information_density,
            "actionability": 0.15 * features.actionability,
            "scope_match": 0.10 * features.scope_match,
        }
        return self._score(
            PolicyScoreName.COMPRESS,
            contributions,
            threshold=self.config.compress_threshold,
            metadata={"information_density": information_density},
        )

    def decide_write(self, features: PolicyFeatures) -> PolicyDecision:
        """Return ADD or IGNORE from WriteScore."""

        result = self.score_write(features)
        return self._decision(
            result,
            features,
            passed=result.passed,
            op_if_passed=OperationType.ADD,
        )

    def decide_retrieve(self, features: PolicyFeatures) -> PolicyDecision:
        """Return RETRIEVE or IGNORE from RetrieveScore."""

        result = self.score_retrieve(features)
        return self._decision(
            result,
            features,
            passed=result.passed,
            op_if_passed=OperationType.RETRIEVE,
        )

    def decide_update(
        self,
        features: PolicyFeatures,
        context: PolicyScoreContext | None = None,
    ) -> PolicyDecision:
        """Return UPDATE or IGNORE using UpdateScore plus conflict gates."""

        context = context or PolicyScoreContext()
        result = self.score_update(features, context)
        similarity = context.semantic_similarity_to_existing
        if similarity is None:
            similarity = features.redundancy
        gates = {
            "score": result.passed,
            "semantic_similarity": similarity >= self.config.update_similarity_threshold,
            "contradiction": features.contradiction >= self.config.update_contradiction_threshold,
        }
        passed = all(gates.values())
        reason = self._append_gate_reason(result.reason, gates)
        return self._decision(
            result,
            features,
            passed=passed,
            op_if_passed=OperationType.UPDATE,
            reason=reason,
            metadata={"gates": gates, **result.metadata},
        )

    def decide_merge(
        self,
        features: PolicyFeatures,
        context: PolicyScoreContext | None = None,
    ) -> PolicyDecision:
        """Return MERGE or IGNORE from MergeScore."""

        result = self.score_merge(features, context)
        return self._decision(
            result,
            features,
            passed=result.passed,
            op_if_passed=OperationType.MERGE,
        )

    def decide_archive(
        self,
        features: PolicyFeatures,
        context: PolicyScoreContext | None = None,
    ) -> PolicyDecision:
        """Return ARCHIVE or IGNORE from ArchiveScore."""

        result = self.score_archive(features, context)
        return self._decision(
            result,
            features,
            passed=result.passed,
            op_if_passed=OperationType.ARCHIVE,
        )

    def decide_compress(
        self,
        features: PolicyFeatures,
        context: PolicyScoreContext | None = None,
    ) -> PolicyDecision:
        """Return COMPRESS or IGNORE from CompressScore."""

        result = self.score_compress(features, context)
        return self._decision(
            result,
            features,
            passed=result.passed,
            op_if_passed=OperationType.COMPRESS,
        )

    def to_operation(
        self,
        decision: PolicyDecision,
        *,
        target_memory_id: str | None = None,
        source_memory_ids: Sequence[str] | None = None,
        source_event_id: str | None = None,
        scope: str = "global",
        backend: str | None = None,
    ) -> MemoryOperation:
        """Convert a policy decision into the auditable operation schema."""

        return MemoryOperation(
            op=decision.op,
            target_memory_id=target_memory_id,
            source_memory_ids=list(source_memory_ids or []),
            source_event_id=source_event_id,
            reason=decision.reason,
            policy_score=decision.score,
            confidence=decision.confidence,
            scope=scope,
            backend=backend,
            payload={
                "decision_id": decision.decision_id,
                "score_name": decision.score_name.value,
                "threshold": decision.threshold,
                "passed": decision.passed,
                "contributions": dict(decision.contributions),
                "feature_snapshot": dict(decision.feature_snapshot),
            },
            metadata=dict(decision.metadata),
        )

    def _score(
        self,
        name: PolicyScoreName,
        contributions: Mapping[str, float],
        *,
        threshold: float,
        metadata: dict[str, JsonValue] | None = None,
    ) -> PolicyScoreResult:
        normalized = {
            key: _finite(value)
            for key, value in contributions.items()
        }
        score = _clamp01(sum(normalized.values()))
        return PolicyScoreResult(
            name=name,
            score=score,
            threshold=_clamp01(threshold),
            passed=score >= threshold,
            contributions=normalized,
            reason=self._reason(name, score, normalized, threshold),
            metadata=metadata or {},
        )

    def _decision(
        self,
        result: PolicyScoreResult,
        features: PolicyFeatures,
        *,
        passed: bool,
        op_if_passed: OperationType,
        reason: str | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ) -> PolicyDecision:
        return PolicyDecision(
            op=op_if_passed if passed else OperationType.IGNORE,
            score_name=result.name,
            score=result.score,
            threshold=result.threshold,
            passed=passed,
            confidence=features.confidence,
            reason=reason or result.reason,
            contributions=result.contributions,
            feature_snapshot=_feature_snapshot(features),
            metadata=metadata or dict(result.metadata),
        )

    @staticmethod
    def _reason(
        name: PolicyScoreName,
        score: float,
        contributions: Mapping[str, float],
        threshold: float,
    ) -> str:
        positives = _top_contributors(contributions, positive=True)
        penalties = _top_contributors(contributions, positive=False)
        status = "passed" if score >= threshold else "did not pass"
        parts = [
            f"{name.value}={score:.3f} {status} threshold {threshold:.3f}.",
            f"Strongest positive signals: {', '.join(positives) or 'none'}.",
        ]
        if penalties:
            parts.append(f"Main penalties: {', '.join(penalties)}.")
        return " ".join(parts)

    @staticmethod
    def _append_gate_reason(reason: str, gates: Mapping[str, bool]) -> str:
        passed = [name for name, ok in gates.items() if ok]
        failed = [name for name, ok in gates.items() if not ok]
        if not failed:
            return f"{reason} Update gates passed: {', '.join(passed)}."
        return f"{reason} Update gates failed: {', '.join(failed)}."

    @staticmethod
    def _fallback_information_density(features: PolicyFeatures) -> float:
        useful = (
            features.importance
            + features.confidence
            + features.actionability
            + features.novelty
        ) / 4.0
        return _clamp01(useful * (1.0 - 0.5 * features.redundancy))


def _feature_snapshot(features: PolicyFeatures) -> dict[str, float]:
    return {
        name: float(getattr(features, name))
        for name in PolicyFeatures.FEATURE_NAMES
    }


def _top_contributors(
    contributions: Mapping[str, float],
    *,
    positive: bool,
    limit: int = 3,
) -> list[str]:
    if positive:
        values = [(key, value) for key, value in contributions.items() if value > 0.0]
        values.sort(key=lambda item: (-item[1], item[0]))
        return [key for key, _ in values[:limit]]
    values = [(key, abs(value)) for key, value in contributions.items() if value < 0.0]
    values.sort(key=lambda item: (-item[1], item[0]))
    return [key for key, _ in values[:limit]]


def _finite(value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("policy score contribution must be finite")
    return numeric


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))
