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
    dense_cosine,
    heuristic_importance,
    lexical_jaccard,
    parse_date,
    sparse_cosine,
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

# V4 treats dense retrieval as an explicit safety set. Policy weights only supply
# a bounded delta inside that set, so a bad lifecycle heuristic cannot destroy
# Recall@10 and can replace at most one dense Top-5 item by default.
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
_STALE_MARKERS = (
    "outdated",
    "stale",
    "obsolete",
    "deprecated",
    "no longer",
)
_ACTIONABLE_MARKERS = (
    "prefer",
    "should",
    "must",
    "need",
    "workflow",
    "process",
    "fix",
    "debug",
    "reuse",
    "remember",
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


_FULL_VMP_ABLATION = VMPTunedAblation()


class VMPTunedModel(SchemaModel):
    """Portable V4 robust ranker with dense-set safety and provenance."""

    schema_version: NonEmptyStr = "1.4"
    model_type: NonEmptyStr = "vmp_v4_robust_dense_guard_ranker"
    weights: dict[NonEmptyStr, FiniteFloat]
    intercept: FiniteFloat = 0.0
    retrieve_threshold: Score = 0.0
    semantic_anchor_weight: Score = 0.80
    lexical_anchor_weight: Score = 0.20
    policy_adjustment_limit: Score = 0.06
    candidate_pool_size: PositiveInt = 20
    safety_top_k: PositiveInt = 5
    preserve_dense_top_n: PositiveInt = 10
    protected_dense_count: PositiveInt = 4
    promotion_margin: Score = 0.02
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
        """Require a complete V4 model trained on dev, never on test."""

        if self.schema_version != "1.4":
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
        if self.protected_dense_count > self.safety_top_k:
            raise ValueError("protected_dense_count cannot exceed safety_top_k")
        if self.safety_top_k != 5:
            raise ValueError("VMP-v4 safety_top_k must remain 5")
        if self.preserve_dense_top_n < 10:
            raise ValueError("VMP-v4 must preserve at least the dense Top-10 set")
        if self.protected_dense_count < 4:
            raise ValueError("VMP-v4 must protect at least four dense Top-5 items")
        if self.safety_top_k > self.preserve_dense_top_n:
            raise ValueError("safety_top_k cannot exceed preserve_dense_top_n")
        if self.retrieve_threshold != 0.0:
            raise ValueError("VMP-v4 retrieve_threshold must be zero for dense safety")
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

        active_ablation = ablation or _FULL_VMP_ABLATION
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
        """Load a persisted V4 artifact."""

        model_path = Path(path).expanduser().resolve()
        return cls.model_validate_json(model_path.read_text(encoding="utf-8"))


class VMPTunedAdapter(VMPRuleAdapter):
    """VMP-v4: cached features plus dense-set-preserving policy reranking."""

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
        self._static_features: list[PolicyFeatures] = []

    def _reset_impl(self) -> None:
        super()._reset_impl()
        self._policy_operation_counts = {"update": 0, "merge": 0, "archive": 0}
        self._lifecycle_status_by_index = {}
        self._static_features = []

    def _finalize_ingestion_impl(self) -> None:
        super()._finalize_ingestion_impl()
        self._index_lifecycle_statuses()
        self._static_features = build_static_vmp_features(self.chunks)

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
            "dense_top10_safety_set -> guarded_top5_policy_rerank -> "
            "cached_non_destructive_lifecycle"
        )
        stats["dense_safety"] = cast(
            JsonValue,
            {
                "safety_top_k": self.model.safety_top_k,
                "preserve_dense_top_n": self.model.preserve_dense_top_n,
                "protected_dense_count": self.model.protected_dense_count,
                "promotion_margin": float(self.model.promotion_margin),
            },
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
        self._embed_new_chunks()
        if len(self._static_features) != len(self.chunks):
            self._index_lifecycle_statuses()
            self._static_features = build_static_vmp_features(self.chunks)
        query_embedding = self.embedder.embed_one(query) if self.embedder else None
        rows = build_vmp_feature_rows(
            self.chunks,
            query,
            query_embedding=query_embedding,
            question_date=question_date,
            metadata=metadata,
            static_features=self._static_features,
        )
        if not self._lifecycle_status_by_index and rows:
            self._index_lifecycle_statuses()
        lexical_scores = normalized_bm25_scores(
            query,
            [chunk.content for chunk, _ in rows],
        )
        temporal_intent = question_has_temporal_intent(query)
        anchor_rows: list[tuple[float, int, MemoryChunk, PolicyFeatures, float]] = []
        for index, ((chunk, features), lexical_score) in enumerate(
            zip(rows, lexical_scores, strict=True)
        ):
            anchor_score = self.model.anchor_score(
                float(features.semantic_relevance),
                lexical_score,
            )
            anchor_rows.append((anchor_score, index, chunk, features, lexical_score))
        anchor_rows.sort(key=lambda row: (-row[0], row[2].memory_id))
        dense_rows = sorted(
            anchor_rows,
            key=lambda row: (-float(row[3].semantic_relevance), row[2].memory_id),
        )
        pool_size = max(
            top_k,
            self.model.candidate_pool_size,
            self.model.preserve_dense_top_n,
        )
        pool_indices = _ordered_unique_indices(
            [row[1] for row in dense_rows[:pool_size]],
            [row[1] for row in anchor_rows[:pool_size]],
        )
        rows_by_index = {row[1]: row for row in anchor_rows}
        candidate_pool = [rows_by_index[index] for index in pool_indices]

        ranked: list[tuple[float, float, int, RetrievedMemory]] = []
        scores_by_index: dict[int, float] = {}
        anchors_by_index: dict[int, float] = {}
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
            scores_by_index[index] = score
            anchors_by_index[index] = anchor_score
            ranked.append(
                (
                    score,
                    anchor_score,
                    index,
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
                            "dense_rank_guarded": True,
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
        ranked.sort(key=lambda row: (-row[0], -row[1], row[3].memory_id))
        ranked_memory = {index: memory for _, _, index, memory in ranked}
        selected_indices = guarded_ranked_indices(
            dense_ranked_indices=[row[1] for row in dense_rows],
            policy_scores=scores_by_index,
            anchor_scores=anchors_by_index,
            requested_top_k=top_k,
            model=self.model,
        )
        return [
            ranked_memory[index]
            for index in selected_indices
            if index in ranked_memory
            and scores_by_index[index] >= self.model.retrieve_threshold
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


def guarded_ranked_indices(
    *,
    dense_ranked_indices: Sequence[int],
    policy_scores: dict[int, float],
    anchor_scores: dict[int, float],
    requested_top_k: int,
    model: VMPTunedModel,
) -> list[int]:
    """Compose a policy ranking without destroying the dense safety set.

    The first ``safety_top_k`` positions retain at least
    ``protected_dense_count`` items from the dense head. A candidate from dense
    ranks 6--10 may replace the final unprotected head item only when it clears
    ``promotion_margin``. The returned Top-10 set remains exactly the dense
    Top-10 set, so policy reranking cannot regress Recall@10.
    """

    if requested_top_k < 1:
        return []
    available_dense = [
        index for index in dense_ranked_indices if index in policy_scores
    ]
    policy_order = sorted(
        policy_scores,
        key=lambda index: (
            -policy_scores[index],
            -anchor_scores.get(index, 0.0),
            index,
        ),
    )
    preserved = available_dense[: min(model.preserve_dense_top_n, len(available_dense))]
    if not preserved:
        return policy_order[:requested_top_k]

    head_size = min(model.safety_top_k, requested_top_k, len(preserved))
    dense_head = preserved[:head_size]
    protected_count = min(model.protected_dense_count, head_size)
    protected = dense_head[:protected_count]
    open_slots = head_size - protected_count

    unprotected_head = dense_head[protected_count:]
    promotion_floor = min(
        (policy_scores[index] for index in unprotected_head),
        default=float("inf"),
    )
    promotion_candidates = [
        index
        for index in preserved[head_size:]
        if policy_scores[index]
        >= promotion_floor + float(model.promotion_margin)
    ]
    eligible = _ordered_unique_indices(unprotected_head, promotion_candidates)
    eligible.sort(
        key=lambda index: (
            -policy_scores[index],
            -anchor_scores.get(index, 0.0),
            index,
        )
    )
    selected_head = protected + eligible[:open_slots]
    if len(selected_head) < head_size:
        selected_head.extend(
            index
            for index in dense_head
            if index not in selected_head
        )
        selected_head = selected_head[:head_size]
    selected_head.sort(
        key=lambda index: (
            -policy_scores[index],
            -anchor_scores.get(index, 0.0),
            index,
        )
    )

    tail = [index for index in preserved if index not in selected_head]
    tail.sort(
        key=lambda index: (
            -policy_scores[index],
            -anchor_scores.get(index, 0.0),
            index,
        )
    )
    selected = selected_head + tail
    if len(selected) < requested_top_k:
        selected.extend(
            index for index in policy_order if index not in selected
        )
    return selected[:requested_top_k]


def build_static_vmp_features(chunks: Sequence[MemoryChunk]) -> list[PolicyFeatures]:
    """Precompute query-independent VMP features once per ingested memory set."""

    redundancies = _max_pairwise_redundancy(chunks)
    static_features: list[PolicyFeatures] = []
    for chunk, redundancy in zip(chunks, redundancies, strict=True):
        lowered = chunk.content.casefold()
        explicit_update = _has_explicit_update_marker(chunk.content)
        actionability = 0.22
        if any(marker in lowered for marker in _ACTIONABLE_MARKERS):
            actionability += 0.25
        if len(terms(chunk.content)) >= 12:
            actionability += 0.05
        static_features.append(
            PolicyFeatures(
                importance=heuristic_importance(chunk.content),
                confidence=0.85,
                novelty=clamp01(1.0 - redundancy),
                redundancy=redundancy,
                contradiction=(
                    0.90 if explicit_update else clamp01(0.10 * redundancy)
                ),
                token_cost=clamp01(chunk.token_count / 2048.0),
                actionability=clamp01(actionability),
            )
        )
    return static_features


def build_vmp_feature_rows(
    chunks: Sequence[MemoryChunk],
    query: str,
    *,
    query_embedding: Sequence[float] | None,
    question_date: str | None,
    metadata: dict[str, JsonValue],
    static_features: Sequence[PolicyFeatures] | None = None,
) -> list[tuple[MemoryChunk, PolicyFeatures]]:
    """Build V4 query features in O(chunks) after static precomputation."""

    cached = (
        list(static_features)
        if static_features is not None
        else build_static_vmp_features(chunks)
    )
    if len(cached) != len(chunks):
        raise ValueError("static VMP feature count must match chunk count")
    query_counts = term_counts(query)
    raw_budget = metadata.get("token_budget", 2048)
    token_budget = int(raw_budget) if isinstance(raw_budget, int) else 2048
    token_budget = max(1, token_budget)
    target_scope = metadata.get("question_id")
    rows: list[tuple[MemoryChunk, PolicyFeatures]] = []
    for chunk, static in zip(chunks, cached, strict=True):
        semantic = (
            dense_cosine(query_embedding, chunk.content_embedding)
            if query_embedding is not None and chunk.content_embedding
            else sparse_cosine(query_counts, term_counts(chunk.content))
        )
        recency = _episodic_recency(chunk.source_date, question_date)
        lowered = chunk.content.casefold()
        keyword_staleness = (
            0.75 if any(marker in lowered for marker in _STALE_MARKERS) else 0.0
        )
        chunk_scope = chunk.metadata.get("question_id")
        scope_match = (
            1.0
            if isinstance(target_scope, str)
            and isinstance(chunk_scope, str)
            and target_scope == chunk_scope
            else 0.5
        )
        rows.append(
            (
                chunk,
                static.model_copy(
                    update={
                        "semantic_relevance": semantic,
                        "recency": recency,
                        "staleness": max(1.0 - recency, keyword_staleness),
                        "token_cost": clamp01(chunk.token_count / token_budget),
                        "scope_match": scope_match,
                    }
                ),
            )
        )
    return rows


def _max_pairwise_redundancy(chunks: Sequence[MemoryChunk]) -> list[float]:
    if len(chunks) < 2:
        return [0.0 for _ in chunks]
    dimensions = {len(chunk.content_embedding) for chunk in chunks}
    if len(dimensions) == 1 and 0 not in dimensions:
        try:
            import numpy as np  # type: ignore[import-not-found]
        except ImportError:
            pass
        else:
            matrix = np.asarray(
                [chunk.content_embedding for chunk in chunks],
                dtype=np.float32,
            )
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            normalized = matrix / np.maximum(norms, 1e-12)
            similarities = normalized @ normalized.T
            np.fill_diagonal(similarities, -1.0)
            maxima = np.clip(similarities.max(axis=1), 0.0, 1.0)
            return [float(value) for value in maxima.tolist()]

    token_sets = [set(terms(chunk.content)) for chunk in chunks]
    maxima = [0.0 for _ in chunks]
    for left_index, left_terms in enumerate(token_sets):
        for right_index in range(left_index + 1, len(token_sets)):
            right_terms = token_sets[right_index]
            union = left_terms | right_terms
            score = len(left_terms & right_terms) / len(union) if union else 0.0
            maxima[left_index] = max(maxima[left_index], score)
            maxima[right_index] = max(maxima[right_index], score)
    return maxima


def _episodic_recency(source_date: str | None, question_date: str | None) -> float:
    source = parse_date(source_date)
    question = parse_date(question_date)
    if source is None or question is None:
        return 0.5
    age_days = max(0.0, (question - source).total_seconds() / 86_400.0)
    return clamp01(0.5 ** (age_days / 14.0))


def _ordered_unique_indices(*groups: Sequence[int]) -> list[int]:
    return list(dict.fromkeys(index for group in groups for index in group))


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
    parsed_dates = [parse_date(source_date) for _, source_date, _ in candidates]
    token_sets = [set(terms(content)) for content, _, _ in candidates]
    update_markers = [
        _has_explicit_update_marker(content) for content, _, _ in candidates
    ]
    superseded: list[int] = []
    for index, candidate_date in enumerate(parsed_dates):
        if candidate_date is None:
            continue
        for other_index, parsed_other_date in enumerate(parsed_dates):
            if other_index == index or not update_markers[other_index]:
                continue
            if parsed_other_date is None or parsed_other_date <= candidate_date:
                continue
            if (
                _set_jaccard(token_sets[index], token_sets[other_index])
                >= model.archive_similarity_threshold
            ):
                superseded.append(index)
                break
    return superseded


def duplicate_candidate_indices(
    contents: Sequence[str],
    *,
    threshold: float,
) -> list[int]:
    """Mark later near-duplicates while retaining every source chunk."""

    token_sets = [set(terms(content)) for content in contents]
    duplicates: list[int] = []
    for index, content_terms in enumerate(token_sets):
        if any(
            _set_jaccard(content_terms, selected_terms) >= threshold
            for selected_terms in token_sets[:index]
        ):
            duplicates.append(index)
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


def _set_jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0
