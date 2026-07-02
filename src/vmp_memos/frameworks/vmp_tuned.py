"""Frozen VMP retrieval model tuned only on the LongMemEval dev split."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import Field, FiniteFloat, JsonValue, model_validator

from vmp_memos.embeddings import BaseEmbedder
from vmp_memos.frameworks.base import MemoryChunk, RetrievedMemory
from vmp_memos.frameworks.text import clamp01, lexical_jaccard, parse_date
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

BASELINE_VMP_WEIGHTS: dict[str, float] = {
    "semantic_relevance": 0.30,
    "importance": 0.20,
    "scope_match": 0.15,
    "confidence": 0.10,
    "success_contribution": 0.10,
    "recency": 0.10,
    "contradiction": -0.15,
    "redundancy": -0.05,
    "token_cost": -0.05,
    "staleness": 0.0,
    "update_signal": 0.20,
    "action_signal": 0.10,
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
    """Portable ranking artifact with explicit training provenance."""

    schema_version: NonEmptyStr = "1.2"
    model_type: NonEmptyStr = "vmp_tuned_linear_ranker"
    weights: dict[NonEmptyStr, FiniteFloat]
    intercept: FiniteFloat = 0.0
    retrieve_threshold: Score = 0.05
    training_split: NonEmptyStr = "dev"
    split_id: NonEmptyStr
    split_manifest_sha256: NonEmptyStr
    dataset_sha256: NonEmptyStr
    embedding_identifier: str | None = None
    objective: dict[NonEmptyStr, FiniteFloat] = Field(default_factory=dict)
    best_objective: FiniteFloat
    dev_metrics: dict[NonEmptyStr, NonNegativeFloat] = Field(default_factory=dict)
    merge_similarity_threshold: Score = 0.90
    archive_similarity_threshold: Score = 0.15
    archive_update_signal_threshold: Score = 0.10
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_frozen_dev_model(self) -> VMPTunedModel:
        """Require a complete model trained on dev, never on test."""

        if self.schema_version != "1.2":
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
        return self

    def score(
        self,
        features: PolicyFeatures,
        *,
        ablation: VMPTunedAblation | None = None,
    ) -> float:
        """Compute a bounded retrieval score without consulting gold labels."""

        active_ablation = ablation or VMPTunedAblation()
        values = vmp_tuned_feature_values(
            features,
            disabled_features=active_ablation.disabled_features,
            disabled_operations=active_ablation.disabled_operations,
        )
        raw_score = float(self.intercept) + sum(
            float(self.weights[name]) * values[name] for name in VMP_TUNED_FEATURES
        )
        return clamp01(raw_score)

    def save(self, path: str | Path) -> Path:
        """Persist the frozen model as JSON."""

        output_path = Path(path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> VMPTunedModel:
        """Load a frozen VMP-Tuned artifact."""

        model_path = Path(path).expanduser().resolve()
        return cls.model_validate_json(model_path.read_text(encoding="utf-8"))


class VMPTunedAdapter(VMPRuleAdapter):
    """VMP retrieval adapter using a frozen dev-tuned linear ranker."""

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

    def _reset_impl(self) -> None:
        super()._reset_impl()
        self._policy_operation_counts = {"update": 0, "merge": 0, "archive": 0}

    def stats(self) -> dict[str, JsonValue]:
        """Include operation counts and the exact frozen-model ablation."""

        stats = super().stats()
        stats["policy_operation_counts"] = dict(self._policy_operation_counts)
        stats["ablation"] = self.ablation.model_dump(mode="json")
        stats["model_split_id"] = self.model.split_id
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
        archived = (
            set()
            if "archive" in self.ablation.disabled_operations
            else set(
                superseded_candidate_indices(
                    [
                        (chunk.content, chunk.source_date, features)
                        for chunk, features in rows
                    ],
                    model=self.model,
                    disabled_features=self.ablation.disabled_features,
                    disabled_operations=self.ablation.disabled_operations,
                )
            )
        )
        self._policy_operation_counts["archive"] += len(archived)
        ranked: list[tuple[float, int, RetrievedMemory, MemoryChunk]] = []
        for index, (chunk, features) in enumerate(rows):
            if index in archived:
                continue
            values = vmp_tuned_feature_values(
                features,
                disabled_features=self.ablation.disabled_features,
                disabled_operations=self.ablation.disabled_operations,
            )
            score = self.model.score(features, ablation=self.ablation)
            if (
                "update" not in self.ablation.disabled_operations
                and values["update_signal"] >= self.model.archive_update_signal_threshold
            ):
                self._policy_operation_counts["update"] += 1
            ranked.append(
                (
                    score,
                    index,
                    chunk.to_retrieved(
                        score=score,
                        metadata={
                            "retrieval_strategy": self.name,
                            "model_type": self.model.model_type,
                            "split_id": self.model.split_id,
                            "ablation": self.ablation.model_dump(mode="json"),
                            "policy_features": {
                                name: float(value)
                                for name, value in values.items()
                            },
                            "policy_contributions": {
                                name: float(self.model.weights[name]) * value
                                for name, value in values.items()
                            },
                        },
                    ),
                    chunk,
                )
            )
        ranked.sort(key=lambda row: (-row[0], row[2].memory_id))

        selected: list[RetrievedMemory] = []
        active_contents: list[str] = []
        merged: set[int] = set()
        merge_enabled = "merge" not in self.ablation.disabled_operations
        for score, index, memory, chunk in ranked:
            if merge_enabled and is_near_duplicate(
                chunk.content,
                active_contents,
                threshold=self.model.merge_similarity_threshold,
            ):
                self._policy_operation_counts["merge"] += 1
                merged.add(index)
                continue
            active_contents.append(chunk.content)
            if score >= self.model.retrieve_threshold and len(selected) < top_k:
                selected.append(memory)
        removed = archived | merged
        if removed:
            self.chunks = [
                chunk
                for index, chunk in enumerate(self.chunks)
                if index not in removed
            ]
        return selected


def vmp_tuned_feature_values(
    features: PolicyFeatures,
    *,
    disabled_features: Sequence[str] = (),
    disabled_operations: Sequence[str] = (),
) -> dict[str, float]:
    """Return base and update-aware derived features."""

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
    update_signal = base["contradiction"] * base["recency"]
    if "update" in disabled_operations:
        update_signal = 0.0
    return {
        **base,
        "update_signal": update_signal,
        "action_signal": float(features.actionability) * base["recency"],
    }


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
    """Find older evidence superseded by newer update-bearing content."""

    if "update" in disabled_operations:
        return []
    archived: list[int] = []
    for index, (content, source_date, _) in enumerate(candidates):
        candidate_date = parse_date(source_date)
        if candidate_date is None:
            continue
        for other_index, (other_content, other_date, other_features) in enumerate(candidates):
            if other_index == index:
                continue
            parsed_other_date = parse_date(other_date)
            if parsed_other_date is None or parsed_other_date <= candidate_date:
                continue
            feature_values = vmp_tuned_feature_values(
                other_features,
                disabled_features=disabled_features,
                disabled_operations=disabled_operations,
            )
            update_signal = feature_values["update_signal"]
            if update_signal < model.archive_update_signal_threshold:
                continue
            if (
                lexical_jaccard(content, other_content)
                >= model.archive_similarity_threshold
            ):
                archived.append(index)
                break
    return archived


def is_near_duplicate(
    content: str,
    selected_contents: Sequence[str],
    *,
    threshold: float,
) -> bool:
    """Return whether merge would suppress a near-duplicate evidence chunk."""

    return any(
        lexical_jaccard(content, selected) >= threshold
        for selected in selected_contents
    )
