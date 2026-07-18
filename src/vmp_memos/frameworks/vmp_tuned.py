"""Safe hybrid VMP retrieval model tuned only on the LongMemEval dev split."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from pydantic import Field, FiniteFloat, JsonValue, PositiveInt, model_validator

from vmp_memos.embeddings import BaseEmbedder
from vmp_memos.frameworks.base import MemoryChunk, RetrievedMemory
from vmp_memos.frameworks.text import (
    clamp01,
    lexical_jaccard,
    parse_date,
    term_counts,
    terms,
)
from vmp_memos.frameworks.vmp_memos import VMPRuleAdapter
from vmp_memos.schemas import PolicyFeatures
from vmp_memos.schemas.base import NonEmptyStr, NonNegativeFloat, SchemaModel, Score

VMP_TUNED_FEATURES: tuple[str, ...] = (
    "semantic_relevance",
    "importance",
    "scope_match",
    "confidence",
    "success_contribution",
    "recency",
    "contradiction",
    "redundancy",
    "token_cost",
    "staleness",
    "update_signal",
    "action_signal",
)

# V3 treats dense/BM25 retrieval as a safety anchor. Policy weights only supply a
# bounded delta, so a bad lifecycle heuristic cannot destroy the candidate set.
BASELINE_VMP_WEIGHTS: dict[str, float] = {
    "semantic_relevance": 0.0,
    "importance": 0.0,
    "scope_match": 0.0,
    "confidence": 0.0,
    "success_contribution": 0.0,
    "recency": 0.10,
    "contradiction": -0.05,
    "redundancy": -0.05,
    "token_cost": -0.05,
    "staleness": -0.05,
    "update_signal": 0.20,
    "action_signal": 0.05,
}

VMP_TUNED_ABLATIONS: dict[str, tuple[str, str]] = {
    "vmp_tuned__no_recency": ("feature", "recency"),
    "vmp_tuned__no_contradiction": ("feature", "contradiction"),
    "vmp_tuned__no_redundancy": ("feature", "redundancy"),
    "vmp_tuned__no_importance": ("feature", "importance"),
    "vmp_tuned__no_confidence": ("feature", "confidence"),
    "vmp_tuned__no_token_cost": ("feature", "token_cost"),
    "vmp_tuned__no_scope_match": ("feature", "scope_match"),
    "vmp_tuned__no_update_operation": ("operation", "update"),
    "vmp_tuned__no_merge_operation": ("operation", "merge"),
    "vmp_tuned__no_archive_operation": ("operation", "archive"),
}
VMP_TUNED_METHODS: tuple[str, ...] = (
    "vmp_tuned",
    "vmp_full",
    *VMP_TUNED_ABLATIONS,
)
_ALLOWED_ABLATION_FEATURES = {
    "recency",
    "contradiction",
    "redundancy",
    "importance",
    "confidence",
    "token_cost",
    "scope_match",
}
_ALLOWED_ABLATION_OPERATIONS = {"update", "merge", "archive"}
_TEMPORAL_QUERY_MARKERS = (
    "now",
    "currently",
    "current",
    "latest",
    "recent",
    "today",
    "when",
    "before",
    "after",
    "since",
    "still",
    "changed",
    "no longer",
)
_EXPLICIT_UPDATE_MARKERS = (
    " now ",
    "currently",
    "changed",
    "instead of",
    "rather than",
    "no longer",
    "not anymore",
    "replaced",
    "switched",
    "moved from",
    "updated",
)


class VMPTunedAblation(SchemaModel):
    """One frozen-model ablation; no variant is re-tuned."""

    name: NonEmptyStr = "vmp_tuned"
    disabled_features: list[NonEmptyStr] = Field(default_factory=list)
    disabled_operations: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_targets(self) -> VMPTunedAblation:
        unknown_features = set(self.disabled_features) - _ALLOWED_ABLATION_FEATURES
        unknown_operations = set(self.disabled_operations) - _ALLOWED_ABLATION_OPERATIONS
        if unknown_features or unknown_operations:
            raise ValueError(
                "unknown VMP-Tuned ablation targets: "
                f"features={sorted(unknown_features)}, "
                f"operations={sorted(unknown_operations)}"
            )
        if len(self.disabled_features) != len(set(self.disabled_features)):
            raise ValueError("disabled_features cannot contain duplicates")
        if len(self.disabled_operations) != len(set(self.disabled_operations)):
            raise ValueError("disabled_operations cannot contain duplicates")
        return self


class VMPTunedModel(SchemaModel):
    """Portable V3 hybrid ranker with explicit safety and training provenance."""

    schema_version: NonEmptyStr = "1.3"
    model_type: NonEmptyStr = "vmp_v3_safe_hybrid_ranker"
    weights: dict[NonEmptyStr, FiniteFloat]
    intercept: FiniteFloat = 0.0
    retrieve_threshold: Score = 0.0
    semantic_anchor_weight: Score = 0.80
    lexical_anchor_weight: Score = 0.20
    policy_adjustment_limit: Score = 0.10
    candidate_pool_size: PositiveInt = 20
    training_split: NonEmptyStr = "dev"
    split_id: NonEmptyStr
    split_manifest_sha256: NonEmptyStr
    dataset_sha256: NonEmptyStr
    embedding_identifier: str | None = None
    objective: dict[NonEmptyStr, FiniteFloat] = Field(default_factory=dict)
    best_objective: FiniteFloat
    dev_metrics: dict[NonEmptyStr, NonNegativeFloat] = Field(default_factory=dict)
    merge_similarity_threshold: Score = 0.98
    archive_similarity_threshold: Score = 0.55
    archive_update_signal_threshold: Score = 0.55
    archive_score_penalty: Score = 0.03
    duplicate_score_penalty: Score = 0.01
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_frozen_dev_model(self) -> VMPTunedModel:
        """Require a complete V3 model trained on dev, never on test."""

        if self.schema_version != "1.3":
            raise ValueError(
                "VMP-Tuned model schema is obsolete; retrain the frozen dev model"
            )
        if self.training_split != "dev":
            raise ValueError("VMP-Tuned artifacts must be trained only on the dev split")
        missing = set(VMP_TUNED_FEATURES) - set(self.weights)
        extra = set(self.weights) - set(VMP_TUNED_FEATURES)
        if missing or extra:
            raise ValueError(
                f"VMP-Tuned weights mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if self.semantic_anchor_weight + self.lexical_anchor_weight <= 0.0:
            raise ValueError("at least one hybrid anchor weight must be positive")
        return self

    def anchor_score(self, semantic_score: float, lexical_score: float) -> float:
        """Combine shared dense and BM25 signals into the safety anchor."""

        denominator = float(self.semantic_anchor_weight + self.lexical_anchor_weight)
        return clamp01(
            (
                float(self.semantic_anchor_weight) * clamp01(semantic_score)
                + float(self.lexical_anchor_weight) * clamp01(lexical_score)
            )
            / denominator
        )

    def score(
        self,
        features: PolicyFeatures,
        *,
        anchor_score: float = 0.0,
        temporal_intent: bool = True,
        lifecycle_status: str = "active",
        ablation: VMPTunedAblation | None = None,
    ) -> float:
        """Apply a bounded policy delta and soft lifecycle penalty to an anchor."""

        policy_delta = self.policy_delta(
            features,
            temporal_intent=temporal_intent,
            ablation=ablation,
        )
        lifecycle_penalty = self.lifecycle_penalty(lifecycle_status)
        return clamp01(anchor_score + policy_delta - lifecycle_penalty)

    def policy_delta(
        self,
        features: PolicyFeatures,
        *,
        temporal_intent: bool,
        ablation: VMPTunedAblation | None = None,
    ) -> float:
        """Return the independently auditable bounded policy adjustment."""

        active_ablation = ablation or VMPTunedAblation()
        values = vmp_tuned_feature_values(
            features,
            disabled_features=active_ablation.disabled_features,
            disabled_operations=active_ablation.disabled_operations,
            temporal_intent=temporal_intent,
        )
        raw_policy_score = float(self.intercept) + sum(
            float(self.weights[name]) * values[name] for name in VMP_TUNED_FEATURES
        )
        return float(self.policy_adjustment_limit) * math.tanh(raw_policy_score)

    def lifecycle_penalty(self, lifecycle_status: str) -> float:
        """Return the soft, non-destructive status penalty."""

        if lifecycle_status == "superseded":
            return float(self.archive_score_penalty)
        if lifecycle_status == "duplicate":
            return float(self.duplicate_score_penalty)
        return 0.0

    def save(self, path: str | Path) -> Path:
        """Persist the frozen model as JSON."""

        output_path = Path(path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> VMPTunedModel:
        """Load a persisted V3 artifact."""

        model_path = Path(path).expanduser().resolve()
        return cls.model_validate_json(model_path.read_text(encoding="utf-8"))


class VMPTunedAdapter(VMPRuleAdapter):
    """VMP-v3: hybrid candidate generation plus safe policy reranking."""

    name = "vmp_tuned"

    def __init__(
        self,
        *,
        model: VMPTunedModel,
        embedder: BaseEmbedder | None = None,
        ablation: VMPTunedAblation | None = None,
    ) -> None:
        super().__init__(embedder=embedder)
        self.model = model
        self.ablation = ablation or VMPTunedAblation()
        self.name = self.ablation.name
        self._policy_operation_counts = {"update": 0, "merge": 0, "archive": 0}
        self._lifecycle_status_by_index: dict[int, str] = {}

    def _reset_impl(self) -> None:
        super()._reset_impl()
        self._policy_operation_counts = {"update": 0, "merge": 0, "archive": 0}
        self._lifecycle_status_by_index = {}

    def _finalize_ingestion_impl(self) -> None:
        super()._finalize_ingestion_impl()
        self._index_lifecycle_statuses()

    def stats(self) -> dict[str, JsonValue]:
        """Include non-destructive operation and lifecycle audit fields."""

        stats = super().stats()
        counts = {"active": 0, "superseded": 0, "duplicate": 0}
        for index in range(len(self.chunks)):
            status = self._lifecycle_status_by_index.get(index, "active")
            counts[status] = counts.get(status, 0) + 1
        stats["policy_operation_counts"] = dict(self._policy_operation_counts)
        stats["lifecycle_status_counts"] = cast(JsonValue, counts)
        stats["physical_memory_count"] = len(self.chunks)
        stats["active_memory_count"] = counts["active"]
        stats["lifecycle_is_non_destructive"] = True
        stats["ablation"] = self.ablation.model_dump(mode="json")
        stats["model_split_id"] = self.model.split_id
        stats["ranking_pipeline"] = (
            "hybrid_candidate_generation -> bounded_policy_rerank -> "
            "non_destructive_lifecycle"
        )
        return stats

    def _retrieve_impl(
        self,
        query: str,
        *,
        top_k: int,
        question_date: str | None,
        metadata: dict[str, JsonValue],
    ) -> list[RetrievedMemory]:
        rows = self.feature_rows(
            query,
            question_date=question_date,
            metadata=metadata,
        )
        if not self._lifecycle_status_by_index and rows:
            self._index_lifecycle_statuses()
        lexical_scores = normalized_bm25_scores(
            query,
            [chunk.content for chunk, _ in rows],
        )
        temporal_intent = question_has_temporal_intent(query)
        anchor_rows: list[
            tuple[float, int, MemoryChunk, PolicyFeatures, float]
        ] = []
        for index, ((chunk, features), lexical_score) in enumerate(
            zip(rows, lexical_scores, strict=True)
        ):
            anchor_score = self.model.anchor_score(
                float(features.semantic_relevance),
                lexical_score,
            )
            anchor_rows.append((anchor_score, index, chunk, features, lexical_score))
        anchor_rows.sort(key=lambda row: (-row[0], row[2].memory_id))
        candidate_pool = anchor_rows[: max(top_k, self.model.candidate_pool_size)]

        ranked: list[tuple[float, float, RetrievedMemory]] = []
        update_count = 0
        for anchor_score, index, chunk, features, lexical_score in candidate_pool:
            lifecycle_status = self._lifecycle_status_by_index.get(index, "active")
            values = vmp_tuned_feature_values(
                features,
                disabled_features=self.ablation.disabled_features,
                disabled_operations=self.ablation.disabled_operations,
                temporal_intent=temporal_intent,
            )
            if (
                "update" not in self.ablation.disabled_operations
                and values["update_signal"] >= self.model.archive_update_signal_threshold
            ):
                update_count += 1
            score = self.model.score(
                features,
                anchor_score=anchor_score,
                temporal_intent=temporal_intent,
                lifecycle_status=lifecycle_status,
                ablation=self.ablation,
            )
            policy_delta = self.model.policy_delta(
                features,
                temporal_intent=temporal_intent,
                ablation=self.ablation,
            )
            lifecycle_penalty = self.model.lifecycle_penalty(lifecycle_status)
            ranked.append(
                (
                    score,
                    anchor_score,
                    chunk.to_retrieved(
                        score=score,
                        metadata={
                            "retrieval_strategy": self.name,
                            "model_type": self.model.model_type,
                            "split_id": self.model.split_id,
                            "ablation": self.ablation.model_dump(mode="json"),
                            "semantic_anchor_score": float(
                                features.semantic_relevance
                            ),
                            "lexical_anchor_score": lexical_score,
                            "hybrid_anchor_score": anchor_score,
                            "policy_delta": policy_delta,
                            "lifecycle_penalty": lifecycle_penalty,
                            "net_score_delta": score - anchor_score,
                            "policy_adjustment_limit": float(
                                self.model.policy_adjustment_limit
                            ),
                            "temporal_intent": temporal_intent,
                            "lifecycle_status": lifecycle_status,
                            "lifecycle_is_non_destructive": True,
                            "policy_features": {
                                name: float(value) for name, value in values.items()
                            },
                            "policy_contributions": {
                                name: float(self.model.weights[name]) * value
                                for name, value in values.items()
                            },
                        },
                    ),
                )
            )
        self._policy_operation_counts["update"] = update_count
        ranked.sort(key=lambda row: (-row[0], -row[1], row[2].memory_id))
        return [
            memory
            for score, _, memory in ranked[:top_k]
            if score >= self.model.retrieve_threshold
        ]

    def _index_lifecycle_statuses(self) -> None:
        """Build query-independent status annotations without deleting chunks."""

        statuses: dict[int, str] = {}
        candidates = [
            (
                chunk.content,
                chunk.source_date,
                PolicyFeatures(),
            )
            for chunk in self.chunks
        ]
        if "archive" not in self.ablation.disabled_operations:
            for index in superseded_candidate_indices(candidates, model=self.model):
                statuses[index] = "superseded"
        if "merge" not in self.ablation.disabled_operations:
            for index in duplicate_candidate_indices(
                [chunk.content for chunk in self.chunks],
                threshold=self.model.merge_similarity_threshold,
            ):
                statuses.setdefault(index, "duplicate")
        self._lifecycle_status_by_index = statuses
        self._policy_operation_counts["archive"] = sum(
            status == "superseded" for status in statuses.values()
        )
        self._policy_operation_counts["merge"] = sum(
            status == "duplicate" for status in statuses.values()
        )


def vmp_tuned_feature_values(
    features: PolicyFeatures,
    *,
    disabled_features: Sequence[str] = (),
    disabled_operations: Sequence[str] = (),
    temporal_intent: bool = True,
) -> dict[str, float]:
    """Return policy features with temporal signals gated by the query text."""

    disabled = set(disabled_features)
    base = {
        "semantic_relevance": float(features.semantic_relevance),
        "importance": float(features.importance),
        "scope_match": float(features.scope_match),
        "confidence": float(features.confidence),
        "success_contribution": float(features.success_contribution),
        "recency": float(features.recency),
        "contradiction": float(features.contradiction),
        "redundancy": float(features.redundancy),
        "token_cost": float(features.token_cost),
        "staleness": float(features.staleness),
    }
    for feature_name in disabled:
        if feature_name in base:
            base[feature_name] = 0.0
    if not temporal_intent:
        for feature_name in ("recency", "contradiction", "staleness"):
            base[feature_name] = 0.0
    update_signal = base["contradiction"] * base["recency"]
    if "update" in disabled_operations:
        update_signal = 0.0
    return {
        **base,
        "update_signal": update_signal,
        "action_signal": (
            float(features.actionability) * base["recency"]
            if temporal_intent
            else 0.0
        ),
    }


def question_has_temporal_intent(query: str) -> bool:
    """Detect whether recency/update signals are relevant to this question."""

    normalized = f" {query.casefold()} "
    return any(marker in normalized for marker in _TEMPORAL_QUERY_MARKERS)


def normalized_bm25_scores(query: str, documents: Sequence[str]) -> list[float]:
    """Return deterministic per-document BM25 scores normalized to ``[0, 1]``."""

    query_terms = terms(query)
    if not query_terms or not documents:
        return [0.0 for _ in documents]
    counts_by_document = [term_counts(document) for document in documents]
    lengths = [sum(counts.values()) for counts in counts_by_document]
    average_length = sum(lengths) / max(1, len(lengths))
    document_frequency = Counter(
        term for counts in counts_by_document for term in set(counts)
    )
    scores: list[float] = []
    k1 = 1.5
    b = 0.75
    for counts, length in zip(counts_by_document, lengths, strict=True):
        score = 0.0
        for term in query_terms:
            frequency = counts.get(term, 0)
            if frequency <= 0:
                continue
            df = document_frequency[term]
            idf = math.log(
                1.0 + (len(documents) - df + 0.5) / (df + 0.5)
            )
            denominator = frequency + k1 * (
                1.0 - b + b * max(1, length) / max(1.0, average_length)
            )
            score += idf * frequency * (k1 + 1.0) / denominator
        scores.append(score)
    maximum = max(scores, default=0.0)
    return [score / maximum if maximum > 0.0 else 0.0 for score in scores]


def ablation_for_method(name: str) -> VMPTunedAblation:
    """Resolve a registered method name into one explicit ablation."""

    normalized = name.strip().casefold().replace("-", "_")
    if normalized in {"vmp_tuned", "vmp_full"}:
        return VMPTunedAblation(name=normalized)
    target = VMP_TUNED_ABLATIONS.get(normalized)
    if target is None:
        raise ValueError(f"unknown VMP-Tuned method: {name}")
    target_type, target_name = target
    return VMPTunedAblation(
        name=normalized,
        disabled_features=[target_name] if target_type == "feature" else [],
        disabled_operations=[target_name] if target_type == "operation" else [],
    )


def superseded_candidate_indices(
    candidates: Sequence[tuple[str, str | None, PolicyFeatures]],
    *,
    model: VMPTunedModel,
    disabled_features: Sequence[str] = (),
    disabled_operations: Sequence[str] = (),
) -> list[int]:
    """Conservatively mark older chunks superseded; never remove them."""

    del disabled_features
    if "update" in disabled_operations or "archive" in disabled_operations:
        return []
    superseded: list[int] = []
    for index, (content, source_date, _) in enumerate(candidates):
        candidate_date = parse_date(source_date)
        if candidate_date is None:
            continue
        for other_index, (other_content, other_date, _) in enumerate(candidates):
            if other_index == index or not _has_explicit_update_marker(other_content):
                continue
            parsed_other_date = parse_date(other_date)
            if parsed_other_date is None or parsed_other_date <= candidate_date:
                continue
            if lexical_jaccard(content, other_content) >= model.archive_similarity_threshold:
                superseded.append(index)
                break
    return superseded


def duplicate_candidate_indices(
    contents: Sequence[str],
    *,
    threshold: float,
) -> list[int]:
    """Mark later near-duplicates while retaining every source chunk."""

    duplicates: list[int] = []
    seen: list[str] = []
    for index, content in enumerate(contents):
        if is_near_duplicate(content, seen, threshold=threshold):
            duplicates.append(index)
        seen.append(content)
    return duplicates


def is_near_duplicate(
    content: str,
    selected_contents: Sequence[str],
    *,
    threshold: float,
) -> bool:
    """Return whether content is near-duplicate of an earlier chunk."""

    return any(
        lexical_jaccard(content, selected) >= threshold
        for selected in selected_contents
    )


def _has_explicit_update_marker(content: str) -> bool:
    normalized = f" {content.casefold()} "
    return any(marker in normalized for marker in _EXPLICIT_UPDATE_MARKERS)
