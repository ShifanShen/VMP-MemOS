"""Rule-based policy feature construction for memory lifecycle decisions."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import JsonValue

from vmp_memos.embeddings import validate_vector
from vmp_memos.schemas import (
    MemoryCandidate,
    MemoryItem,
    MemoryStatus,
    MemoryType,
    PolicyFeatures,
)

_WORD_PATTERN = re.compile(r"[\w-]+", flags=re.UNICODE)
_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_PHONE_PATTERN = re.compile(r"\b(?:\+?\d[\d\s().-]{7,}\d)\b")
_SECRET_PATTERN = re.compile(
    r"\b(?:api[_-]?key|password|passwd|secret|token|private[_-]?key)\b",
    flags=re.IGNORECASE,
)

_CHANGE_KEYWORDS = (
    "不再",
    "不是",
    "改为",
    "改成",
    "转向",
    "现在",
    "instead",
    "rather than",
    "no longer",
    "not anymore",
    "changed",
    "replaced",
    "supersede",
)
_STALE_KEYWORDS = (
    "旧",
    "过期",
    "废弃",
    "不再",
    "outdated",
    "stale",
    "obsolete",
    "deprecated",
    "no longer",
)
_TEMPORARY_KEYWORDS = (
    "临时",
    "暂时",
    "今天",
    "一次性",
    "temporary",
    "tentative",
    "today",
    "one-off",
)
_ACTIONABLE_KEYWORDS = (
    "应该",
    "必须",
    "需要",
    "偏好",
    "主攻",
    "步骤",
    "流程",
    "方案",
    "记住",
    "prefer",
    "should",
    "must",
    "need",
    "workflow",
    "process",
    "fix",
    "debug",
    "reuse",
)


@dataclass(frozen=True)
class PolicyFeatureBuilderConfig:
    """Tunable constants for the deterministic first policy-feature builder."""

    token_budget: int = 2048
    access_count_saturation: int = 20
    half_life_days: Mapping[MemoryType, float] = field(
        default_factory=lambda: {
            MemoryType.SEMANTIC: 120.0,
            MemoryType.EPISODIC: 14.0,
            MemoryType.PROCEDURAL: 240.0,
            MemoryType.REFLECTIVE: 120.0,
            MemoryType.RESOURCE: 60.0,
        }
    )


@dataclass(frozen=True)
class PolicyFeatureContext:
    """External evidence used while constructing policy features."""

    query: str | None = None
    target_scope: str | None = None
    query_embedding: Sequence[float] | None = None
    subject_embedding: Sequence[float] | None = None
    existing_memories: Sequence[MemoryItem] = field(default_factory=tuple)
    now: datetime | None = None
    token_budget: int | None = None


class PolicyFeatureBuilder:
    """Build explainable policy features using deterministic heuristics.

    The first implementation deliberately avoids LLM calls. It consumes vectors
    when callers already have them, otherwise it falls back to lexical and
    metadata-based rules. This keeps Phase 4 runnable on a small local machine
    while preserving clean seams for later NLI, LLM-as-judge, or learned models.
    """

    def __init__(self, config: PolicyFeatureBuilderConfig | None = None) -> None:
        self.config = config or PolicyFeatureBuilderConfig()

    def build_for_candidate(
        self,
        candidate: MemoryCandidate,
        context: PolicyFeatureContext | None = None,
    ) -> PolicyFeatures:
        """Construct features for a proposed memory before it is persisted."""

        return self._build(candidate, context or PolicyFeatureContext())

    def build_for_memory(
        self,
        memory: MemoryItem,
        context: PolicyFeatureContext | None = None,
    ) -> PolicyFeatures:
        """Construct features for an existing memory item."""

        return self._build(memory, context or PolicyFeatureContext())

    def enrich_memory(
        self,
        memory: MemoryItem,
        context: PolicyFeatureContext | None = None,
    ) -> MemoryItem:
        """Return a copy of ``memory`` with refreshed features and policy embedding."""

        features = self.build_for_memory(memory, context)
        payload = memory.model_dump(mode="python")
        payload["features"] = features
        payload["policy_embedding"] = list(features.as_vector())
        return MemoryItem.model_validate(payload)

    @staticmethod
    def explain(features: PolicyFeatures, *, top_n: int = 5) -> dict[str, float]:
        """Return the strongest feature values for logging or debugging."""

        pairs = {
            name: float(getattr(features, name))
            for name in PolicyFeatures.FEATURE_NAMES
        }
        return dict(sorted(pairs.items(), key=lambda item: (-item[1], item[0]))[:top_n])

    def _build(
        self,
        subject: MemoryCandidate | MemoryItem,
        context: PolicyFeatureContext,
    ) -> PolicyFeatures:
        now = _normalize_datetime(context.now or datetime.now(UTC))
        text = _subject_content(subject)
        memory_type = _subject_type(subject)
        subject_embedding = _subject_embedding(subject, context)
        redundancy = self._redundancy(
            subject,
            subject_embedding=subject_embedding,
            existing_memories=context.existing_memories,
        )
        novelty = 1.0 - redundancy
        recency = self._recency(subject, now=now)
        staleness = self._staleness(subject, recency=recency)

        return PolicyFeatures(
            semantic_relevance=self._semantic_relevance(
                text,
                subject_embedding=subject_embedding,
                query=context.query,
                query_embedding=context.query_embedding,
            ),
            importance=_subject_importance(subject),
            confidence=_subject_confidence(subject),
            recency=recency,
            stability=self._stability(text, memory_type=memory_type, recency=recency),
            novelty=novelty,
            redundancy=redundancy,
            contradiction=self._contradiction(
                subject,
                redundancy=redundancy,
                existing_memories=context.existing_memories,
            ),
            staleness=staleness,
            access_frequency=self._access_frequency(subject),
            success_contribution=_subject_feature_or_attribute(
                subject,
                "success_contribution",
            ),
            failure_contribution=_subject_feature_or_attribute(
                subject,
                "failure_contribution",
            ),
            token_cost=self._token_cost(text, context.token_budget),
            scope_match=self._scope_match(
                _subject_scope(subject),
                context.target_scope,
            ),
            actionability=self._actionability(text, memory_type=memory_type),
            privacy_risk=self._privacy_risk(text, _subject_attributes(subject)),
        )

    def _semantic_relevance(
        self,
        text: str,
        *,
        subject_embedding: Sequence[float] | None,
        query: str | None,
        query_embedding: Sequence[float] | None,
    ) -> float:
        if subject_embedding and query_embedding:
            return _cosine_score(query_embedding, subject_embedding)
        if query:
            return _lexical_similarity(query, text)
        return 0.0

    def _redundancy(
        self,
        subject: MemoryCandidate | MemoryItem,
        *,
        subject_embedding: Sequence[float] | None,
        existing_memories: Sequence[MemoryItem],
    ) -> float:
        if not existing_memories:
            return 0.0

        scores: list[float] = []
        subject_id = subject.id if isinstance(subject, MemoryItem) else None
        for memory in existing_memories:
            if subject_id is not None and memory.id == subject_id:
                continue
            if subject_embedding and memory.content_embedding:
                scores.append(_cosine_score(subject_embedding, memory.content_embedding))
            else:
                scores.append(_memory_similarity(subject, memory))
        return max(scores, default=0.0)

    def _contradiction(
        self,
        subject: MemoryCandidate | MemoryItem,
        *,
        redundancy: float,
        existing_memories: Sequence[MemoryItem],
    ) -> float:
        text = _subject_content(subject)
        if not existing_memories:
            return 0.0
        if not _contains_any(text, _CHANGE_KEYWORDS):
            return _clamp01(0.1 * redundancy)

        scope = _subject_scope(subject)
        strongest_relation = redundancy
        for memory in existing_memories:
            strongest_relation = max(
                strongest_relation,
                _lexical_similarity(scope, memory.scope),
                _lexical_similarity(text, memory.content),
            )
        if strongest_relation <= 0.0:
            return 0.0
        return _clamp01(0.45 + 0.45 * strongest_relation)

    def _recency(self, subject: MemoryCandidate | MemoryItem, *, now: datetime) -> float:
        timestamp = _subject_updated_at(subject)
        age_days = max(0.0, (now - timestamp).total_seconds() / 86_400.0)
        half_life = self.config.half_life_days[_subject_type(subject)]
        return _clamp01(0.5 ** (age_days / half_life))

    def _staleness(self, subject: MemoryCandidate | MemoryItem, *, recency: float) -> float:
        if isinstance(subject, MemoryItem):
            if subject.metadata.status in {MemoryStatus.ARCHIVED, MemoryStatus.EXPIRED}:
                return 1.0
        keyword_staleness = (
            0.75 if _contains_any(_subject_content(subject), _STALE_KEYWORDS) else 0.0
        )
        return _clamp01(max(1.0 - recency, keyword_staleness))

    @staticmethod
    def _stability(text: str, *, memory_type: MemoryType, recency: float) -> float:
        base = {
            MemoryType.SEMANTIC: 0.72,
            MemoryType.EPISODIC: 0.35,
            MemoryType.PROCEDURAL: 0.86,
            MemoryType.REFLECTIVE: 0.62,
            MemoryType.RESOURCE: 0.50,
        }[memory_type]
        if _contains_any(text, _TEMPORARY_KEYWORDS):
            base -= 0.25
        return _clamp01(base + 0.12 * (1.0 - recency))

    def _access_frequency(self, subject: MemoryCandidate | MemoryItem) -> float:
        access_count = 0
        if isinstance(subject, MemoryItem):
            access_count = int(subject.metadata.access_count)
        else:
            raw_value = subject.metadata.get("access_count", 0)
            access_count = int(raw_value) if isinstance(raw_value, int | float) else 0
        if access_count <= 0:
            return 0.0
        denominator = math.log1p(max(1, self.config.access_count_saturation))
        return _clamp01(math.log1p(access_count) / denominator)

    def _token_cost(self, text: str, token_budget: int | None) -> float:
        budget = token_budget or self.config.token_budget
        if budget <= 0:
            raise ValueError("token budget must be positive")
        return _clamp01(_estimate_tokens(text) / budget)

    @staticmethod
    def _scope_match(scope: str, target_scope: str | None) -> float:
        if not target_scope:
            return 0.5 if scope == "global" else 0.7
        if scope == target_scope:
            return 1.0
        if scope == "global" or target_scope == "global":
            return 0.65
        if scope.startswith(target_scope) or target_scope.startswith(scope):
            return 0.85
        return _clamp01(0.25 + 0.5 * _lexical_similarity(scope, target_scope))

    @staticmethod
    def _actionability(text: str, *, memory_type: MemoryType) -> float:
        base = {
            MemoryType.SEMANTIC: 0.40,
            MemoryType.EPISODIC: 0.22,
            MemoryType.PROCEDURAL: 0.78,
            MemoryType.REFLECTIVE: 0.58,
            MemoryType.RESOURCE: 0.32,
        }[memory_type]
        if _contains_any(text, _ACTIONABLE_KEYWORDS):
            base += 0.25
        if len(_terms(text)) >= 12:
            base += 0.05
        return _clamp01(base)

    @staticmethod
    def _privacy_risk(text: str, attributes: Mapping[str, JsonValue]) -> float:
        lowered = text.casefold()
        score = 0.0
        if _EMAIL_PATTERN.search(text):
            score = max(score, 0.6)
        if _PHONE_PATTERN.search(text):
            score = max(score, 0.65)
        if _SECRET_PATTERN.search(text):
            score = max(score, 0.9)
        if any(keyword in lowered for keyword in ("身份证", "手机号", "住址", "银行卡")):
            score = max(score, 0.75)
        raw_privacy = attributes.get("privacy_risk")
        if isinstance(raw_privacy, int | float):
            score = max(score, float(raw_privacy))
        raw_sensitive = attributes.get("sensitive")
        if raw_sensitive is True:
            score = max(score, 0.8)
        return _clamp01(score)


def _subject_content(subject: MemoryCandidate | MemoryItem) -> str:
    return subject.content


def _subject_type(subject: MemoryCandidate | MemoryItem) -> MemoryType:
    return subject.type if isinstance(subject, MemoryItem) else subject.memory_type


def _subject_scope(subject: MemoryCandidate | MemoryItem) -> str:
    return subject.scope


def _subject_importance(subject: MemoryCandidate | MemoryItem) -> float:
    if isinstance(subject, MemoryItem):
        return float(subject.features.importance)
    return float(subject.importance)


def _subject_confidence(subject: MemoryCandidate | MemoryItem) -> float:
    if isinstance(subject, MemoryItem):
        return float(subject.features.confidence)
    return float(subject.confidence)


def _subject_updated_at(subject: MemoryCandidate | MemoryItem) -> datetime:
    if isinstance(subject, MemoryItem):
        return _normalize_datetime(subject.metadata.updated_at)
    return _normalize_datetime(subject.timestamp)


def _subject_embedding(
    subject: MemoryCandidate | MemoryItem,
    context: PolicyFeatureContext,
) -> Sequence[float] | None:
    if context.subject_embedding:
        return context.subject_embedding
    if isinstance(subject, MemoryItem) and subject.content_embedding:
        return subject.content_embedding
    return None


def _subject_attributes(subject: MemoryCandidate | MemoryItem) -> Mapping[str, JsonValue]:
    if isinstance(subject, MemoryItem):
        return subject.metadata.attributes
    return subject.metadata


def _subject_feature_or_attribute(
    subject: MemoryCandidate | MemoryItem,
    feature_name: str,
) -> float:
    if isinstance(subject, MemoryItem):
        feature_value = float(getattr(subject.features, feature_name))
        raw_attribute = subject.metadata.attributes.get(feature_name)
    else:
        feature_value = 0.0
        raw_attribute = subject.metadata.get(feature_name)
    if isinstance(raw_attribute, int | float):
        return _clamp01(float(raw_attribute))
    return _clamp01(feature_value)


def _memory_similarity(subject: MemoryCandidate | MemoryItem, memory: MemoryItem) -> float:
    content_similarity = _lexical_similarity(_subject_content(subject), memory.content)
    scope_similarity = _lexical_similarity(_subject_scope(subject), memory.scope)
    type_bonus = 0.1 if _subject_type(subject) == memory.type else 0.0
    return _clamp01(0.8 * content_similarity + 0.1 * scope_similarity + type_bonus)


def _cosine_score(left: Sequence[float], right: Sequence[float]) -> float:
    left_vector = validate_vector(left)
    right_vector = validate_vector(right, expected_dimension=len(left_vector))
    left_norm = math.sqrt(sum(value * value for value in left_vector))
    right_norm = math.sqrt(sum(value * value for value in right_vector))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    raw = sum(a * b for a, b in zip(left_vector, right_vector, strict=True))
    return _clamp01(raw / (left_norm * right_norm))


def _lexical_similarity(left: str, right: str) -> float:
    left_terms = set(_terms(left))
    right_terms = set(_terms(right))
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


def _terms(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _WORD_PATTERN.finditer(text)]


def _contains_any(text: str, keywords: Sequence[str]) -> bool:
    lowered = text.casefold()
    return any(keyword.casefold() in lowered for keyword in keywords)


def _estimate_tokens(text: str) -> int:
    cjk_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    ascii_chars = sum(1 for char in text if ord(char) < 128 and not char.isspace())
    other_chars = max(0, len(text) - cjk_chars - ascii_chars)
    return max(1, math.ceil(cjk_chars / 1.8 + ascii_chars / 4.0 + other_chars / 3.0))


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))
