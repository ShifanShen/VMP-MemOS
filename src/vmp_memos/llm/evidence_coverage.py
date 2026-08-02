"""VMP-v6 atomic evidence schemas and deterministic set coverage."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
from typing import Literal, cast

from pydantic import Field, PositiveInt

from vmp_memos.frameworks.text import terms
from vmp_memos.schemas.base import (
    NonEmptyStr,
    NonNegativeFloat,
    NonNegativeInt,
    SchemaModel,
    Score,
)

EvidenceOperator = Literal["single", "count", "list", "temporal", "latest"]
FactConfidence = Literal["high", "medium", "low"]

_DURATION_PATTERN = re.compile(
    r"\b(?:how\s+many\s+)?(?:days?|weeks?|months?|years?)\b.*\b(?:ago|since|passed)\b",
    flags=re.IGNORECASE,
)
_STOP_TERMS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "at",
        "be",
        "did",
        "do",
        "does",
        "for",
        "from",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "many",
        "me",
        "my",
        "of",
        "on",
        "since",
        "that",
        "the",
        "to",
        "was",
        "what",
        "when",
        "which",
        "with",
    }
)


class QuestionEvidencePlan(SchemaModel):
    """Label-free query requirements used by the deterministic selector."""

    operator: EvidenceOperator
    evidence_needs: list[NonEmptyStr] = Field(min_length=1, max_length=4)
    query_terms: list[NonEmptyStr] = Field(default_factory=list)
    entity_diversity_required: bool = False
    temporal_coverage_required: bool = False


class AtomicEvidenceFact(SchemaModel):
    """One grounded fact extracted from an anonymous candidate excerpt."""

    fact_id: NonEmptyStr
    entity: NonEmptyStr
    relation: NonEmptyStr
    value: NonEmptyStr
    temporal_anchor: str | None = None
    supports_needs: list[NonEmptyStr] = Field(default_factory=list, max_length=4)
    evidence_spans: list[NonEmptyStr] = Field(min_length=1, max_length=4)
    confidence: FactConfidence = "medium"


class CandidateEvidenceProfile(SchemaModel):
    """Auditable atomic facts associated with one local candidate label."""

    candidate_label: NonEmptyStr
    session_id: NonEmptyStr
    rank: PositiveInt
    candidate_relevant: bool = False
    lexical_overlap: Score = 0.0
    facts: list[AtomicEvidenceFact] = Field(default_factory=list)
    extraction_fallback: bool = False
    extraction_failures: list[NonEmptyStr] = Field(default_factory=list)


class EvidenceCoverageSelection(SchemaModel):
    """Deterministic Top-k selected by joint atomic-fact coverage."""

    selected_candidate_labels: list[NonEmptyStr]
    ranked_candidate_labels: list[NonEmptyStr]
    promoted_candidate_labels: list[NonEmptyStr] = Field(default_factory=list)
    displaced_candidate_labels: list[NonEmptyStr] = Field(default_factory=list)
    original_score: NonNegativeFloat
    selected_score: NonNegativeFloat
    gain: float
    original_components: dict[str, float] = Field(default_factory=dict)
    selected_components: dict[str, float] = Field(default_factory=dict)
    combinations_evaluated: NonNegativeInt = 0


def build_question_evidence_plan(question: str) -> QuestionEvidencePlan:
    """Infer a small deterministic answer operator without using labels or answers."""

    normalized = " ".join(question.casefold().split())
    temporal = bool(
        _DURATION_PATTERN.search(normalized)
        or re.search(r"\b(?:when|before|after|ago|since|last visited)\b", normalized)
    )
    count = bool(re.search(r"\b(?:how many|number of|count of)\b", normalized))
    current = bool(
        re.search(r"\b(?:current|currently|latest|most recent|now|still)\b", normalized)
    )
    if temporal:
        operator: EvidenceOperator = "temporal"
        needs = ["N1: query-relevant event or state", "N2: event date or time anchor"]
    elif count:
        operator = "count"
        needs = ["N1: each distinct query-relevant item"]
        if current:
            needs.append("N2: evidence that each item is currently active")
    elif re.search(r"\b(?:list|which|what are|name all|identify all)\b", normalized):
        operator = "list"
        needs = ["N1: each distinct query-relevant item"]
        if current:
            needs.append("N2: latest valid state of each item")
    elif current:
        operator = "latest"
        needs = ["N1: query-relevant state", "N2: latest valid dated state"]
    else:
        operator = "single"
        needs = ["N1: direct answer-supporting fact"]
    query_terms = _meaningful_terms(question)
    return QuestionEvidencePlan(
        operator=operator,
        evidence_needs=needs,
        query_terms=query_terms,
        entity_diversity_required=operator in {"count", "list"},
        temporal_coverage_required=operator in {"temporal", "latest"} or current,
    )


def extract_owned_span_ids(
    values: Sequence[str],
    *,
    owner: str,
    allowed: set[str],
) -> list[str]:
    """Extract owned span labels even when a model appends quoted evidence text."""

    normalized_owner = re.sub(r"\s+", "", owner).upper()
    pattern = re.compile(
        rf"(?<![A-Z0-9-]){re.escape(normalized_owner)}\s*:\s*S0*(\d+)",
        flags=re.IGNORECASE,
    )
    observed: list[str] = []
    for value in values:
        for match in pattern.finditer(value):
            span_id = f"{normalized_owner}:S{int(match.group(1)):02d}"
            if span_id in allowed and span_id not in observed:
                observed.append(span_id)
    return observed


def build_candidate_evidence_profile(
    payload: Mapping[str, object],
    *,
    candidate_label: str,
    session_id: str,
    rank: int,
    plan: QuestionEvidencePlan,
    allowed_span_ids: set[str],
    excerpt: str,
    extraction_fallback: bool = False,
) -> CandidateEvidenceProfile:
    """Validate a model payload and retain only locally grounded atomic facts."""

    raw_facts = payload.get("facts")
    values = raw_facts if isinstance(raw_facts, list) else []
    known_needs = {f"N{index}" for index in range(1, len(plan.evidence_needs) + 1)}
    facts: list[AtomicEvidenceFact] = []
    failures: list[str] = []
    for index, value in enumerate(values, start=1):
        if not isinstance(value, dict):
            failures.append(f"F{index}:invalid_fact")
            continue
        entity = _nonempty(value.get("entity"))
        relation = _nonempty(value.get("relation"))
        fact_value = _nonempty(value.get("value"))
        raw_spans = _string_list(value.get("evidence_spans"))
        spans = extract_owned_span_ids(
            raw_spans,
            owner="X",
            allowed=allowed_span_ids,
        )
        if entity is None or relation is None or fact_value is None:
            failures.append(f"F{index}:missing_fact_field")
            continue
        if not spans:
            failures.append(f"F{index}:span_not_grounded")
            continue
        supports = _normalized_need_ids(
            _string_list(value.get("supports_needs")),
            known=known_needs,
        )
        if not supports:
            failures.append(f"F{index}:missing_supported_need")
            continue
        confidence_value = (_nonempty(value.get("confidence")) or "medium").casefold()
        confidence: FactConfidence = (
            cast(FactConfidence, confidence_value)
            if confidence_value in {"high", "medium", "low"}
            else "medium"
        )
        temporal_anchor = _nonempty(value.get("temporal_anchor"))
        facts.append(
            AtomicEvidenceFact(
                fact_id=f"{candidate_label}:F{index:02d}",
                entity=entity,
                relation=relation,
                value=fact_value,
                temporal_anchor=temporal_anchor,
                supports_needs=supports,
                evidence_spans=spans,
                confidence=confidence,
            )
        )
    overlap = _lexical_overlap(plan.query_terms, excerpt)
    model_relevant = payload.get("candidate_relevant") is True
    if model_relevant and not facts:
        failures.append("relevance_without_grounded_fact")
    return CandidateEvidenceProfile(
        candidate_label=candidate_label,
        session_id=session_id,
        rank=rank,
        candidate_relevant=bool(facts),
        lexical_overlap=overlap,
        facts=facts,
        extraction_fallback=extraction_fallback,
        extraction_failures=failures,
    )


def select_evidence_coverage(
    plan: QuestionEvidencePlan,
    profiles: Sequence[CandidateEvidenceProfile],
    *,
    output_top_k: int,
    protected_top_n: int,
    min_gain: float,
    need_weight: float = 3.0,
    relevance_weight: float = 1.5,
    diversity_weight: float = 1.25,
    temporal_weight: float = 1.25,
    rank_weight: float = 0.08,
) -> EvidenceCoverageSelection:
    """Enumerate all open-slot combinations and choose the best guarded set."""

    if output_top_k < 1:
        raise ValueError("output_top_k must be positive")
    if not 0 <= protected_top_n < output_top_k:
        raise ValueError("protected_top_n must be in [0, output_top_k)")
    ordered = sorted(profiles, key=lambda profile: profile.rank)
    if len(ordered) < output_top_k:
        raise ValueError("not enough evidence profiles for Top-k selection")
    if len({profile.rank for profile in ordered}) != len(ordered):
        raise ValueError("candidate profile ranks must be unique")
    if not math.isfinite(min_gain) or min_gain < 0:
        raise ValueError("min_gain must be finite and non-negative")
    protected = ordered[:protected_top_n]
    open_slots = output_top_k - protected_top_n
    original_open = ordered[protected_top_n:output_top_k]
    original_set = [*protected, *original_open]
    original_score, original_components = _coverage_score(
        plan,
        original_set,
        candidate_count=len(ordered),
        need_weight=need_weight,
        relevance_weight=relevance_weight,
        diversity_weight=diversity_weight,
        temporal_weight=temporal_weight,
        rank_weight=rank_weight,
    )
    best_set = original_set
    best_score = original_score
    best_components = original_components
    evaluated = 0
    for open_group in combinations(ordered[protected_top_n:], open_slots):
        evaluated += 1
        candidate_set = [*protected, *open_group]
        score, components = _coverage_score(
            plan,
            candidate_set,
            candidate_count=len(ordered),
            need_weight=need_weight,
            relevance_weight=relevance_weight,
            diversity_weight=diversity_weight,
            temporal_weight=temporal_weight,
            rank_weight=rank_weight,
        )
        candidate_ranks = tuple(profile.rank for profile in candidate_set)
        best_ranks = tuple(profile.rank for profile in best_set)
        if score > best_score + 1e-9 or (
            abs(score - best_score) <= 1e-9 and candidate_ranks < best_ranks
        ):
            best_set = candidate_set
            best_score = score
            best_components = components
    if best_score - original_score < min_gain:
        best_set = original_set
        best_score = original_score
        best_components = original_components
    selected_labels = [profile.candidate_label for profile in best_set]
    original_labels = [profile.candidate_label for profile in original_set]
    ranked_labels = [
        *selected_labels,
        *[
            profile.candidate_label
            for profile in ordered
            if profile.candidate_label not in selected_labels
        ],
    ]
    return EvidenceCoverageSelection(
        selected_candidate_labels=selected_labels,
        ranked_candidate_labels=ranked_labels,
        promoted_candidate_labels=[
            label for label in selected_labels if label not in original_labels
        ],
        displaced_candidate_labels=[
            label for label in original_labels if label not in selected_labels
        ],
        original_score=original_score,
        selected_score=best_score,
        gain=best_score - original_score,
        original_components=original_components,
        selected_components=best_components,
        combinations_evaluated=evaluated,
    )


def _coverage_score(
    plan: QuestionEvidencePlan,
    profiles: Sequence[CandidateEvidenceProfile],
    *,
    candidate_count: int,
    need_weight: float,
    relevance_weight: float,
    diversity_weight: float,
    temporal_weight: float,
    rank_weight: float,
) -> tuple[float, dict[str, float]]:
    facts = [fact for profile in profiles for fact in profile.facts]
    known_needs = {f"N{index}" for index in range(1, len(plan.evidence_needs) + 1)}
    covered_needs = {
        need
        for fact in facts
        for need in fact.supports_needs
        if need in known_needs
    }
    need_coverage = len(covered_needs) / len(known_needs) if known_needs else 0.0
    confidence_values = {"high": 1.0, "medium": 0.65, "low": 0.2}
    per_candidate_relevance: list[float] = []
    for profile in profiles:
        best_confidence = max(
            (confidence_values[fact.confidence] for fact in profile.facts),
            default=0.0,
        )
        per_candidate_relevance.append(
            (
                0.45 * float(profile.lexical_overlap)
                + 0.35 * best_confidence
                + 0.20 * float(profile.candidate_relevant)
            )
            if profile.facts
            else 0.0
        )
    relevance = sum(per_candidate_relevance)
    relevant_facts = [fact for fact in facts if fact.supports_needs]
    distinct_entities = {
        _normalized_fact_key(fact.entity)
        for fact in relevant_facts
        if _normalized_fact_key(fact.entity)
    }
    diversity = (
        min(len(distinct_entities), len(profiles)) / max(1, len(profiles))
        if plan.entity_diversity_required
        else 0.0
    )
    temporal_facts = [
        fact for fact in relevant_facts if fact.temporal_anchor and fact.temporal_anchor.strip()
    ]
    temporal = (
        min(1.0, len(temporal_facts) / max(1, len(plan.evidence_needs)))
        if plan.temporal_coverage_required
        else 0.0
    )
    signatures = [
        (
            _normalized_fact_key(fact.entity),
            _normalized_fact_key(fact.relation),
            _normalized_fact_key(fact.value),
        )
        for fact in relevant_facts
    ]
    duplicate_count = len(signatures) - len(set(signatures))
    redundancy_penalty = 0.25 * duplicate_count
    rank_prior = sum(
        (candidate_count - profile.rank + 1) / candidate_count
        for profile in profiles
    )
    components = {
        "need_coverage": need_weight * need_coverage,
        "relevance": relevance_weight * relevance,
        "entity_diversity": diversity_weight * diversity,
        "temporal_coverage": temporal_weight * temporal,
        "rank_prior": rank_weight * rank_prior,
        "redundancy_penalty": redundancy_penalty,
    }
    score = (
        components["need_coverage"]
        + components["relevance"]
        + components["entity_diversity"]
        + components["temporal_coverage"]
        + components["rank_prior"]
        - components["redundancy_penalty"]
    )
    return score, components


def _meaningful_terms(value: str) -> list[str]:
    return _ordered_unique(
        token for token in terms(value) if len(token) > 1 and token not in _STOP_TERMS
    )


def _lexical_overlap(query_terms: Sequence[str], excerpt: str) -> float:
    query = set(query_terms)
    if not query:
        return 0.0
    observed = set(_meaningful_terms(excerpt))
    return min(1.0, len(query.intersection(observed)) / len(query))


def _normalized_need_ids(values: Sequence[str], *, known: set[str]) -> list[str]:
    observed: list[str] = []
    for value in values:
        for match in re.finditer(r"\bN([1-4])\b", value, flags=re.IGNORECASE):
            need_id = f"N{int(match.group(1))}"
            if need_id in known and need_id not in observed:
                observed.append(need_id)
    return observed


def _normalized_fact_key(value: str) -> str:
    return " ".join(_meaningful_terms(value))


def _nonempty(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    ]


def _ordered_unique(values: Iterable[str]) -> list[str]:
    observed: list[str] = []
    for value in values:
        if value and value not in observed:
            observed.append(value)
    return observed
