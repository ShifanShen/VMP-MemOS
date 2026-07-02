"""Toy benchmark baselines for deterministic memory-policy evaluation."""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import Field, JsonValue

from vmp_memos.policy import (
    LogisticPolicyModel,
    PolicyFeatureBuilder,
    PolicyFeatureContext,
    PolicyScoreContext,
    RuleBasedPolicyController,
)
from vmp_memos.schemas import (
    BenchmarkSample,
    MemoryItem,
    MemorySource,
    MemoryStatus,
    MemoryType,
    OperationType,
    PolicyFeatures,
)
from vmp_memos.schemas.base import NonEmptyStr, NonNegativeFloat, SchemaModel, utc_now

_TOKEN_PATTERN = re.compile(r"[\w-]+", flags=re.UNICODE)
_FEATURE_NAMES = set(PolicyFeatures.FEATURE_NAMES)


class BaselineOutput(SchemaModel):
    """Raw output from a benchmark baseline before metrics are attached."""

    system_name: NonEmptyStr
    answer: str | None = None
    retrieved_memory_ids: list[NonEmptyStr] = Field(default_factory=list)
    operations: list[OperationType] = Field(default_factory=list)
    token_count: int = Field(default=0, ge=0)
    latency_ms: NonNegativeFloat = 0.0
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class BenchmarkBaseline(ABC):
    """Interface shared by all toy benchmark baselines."""

    name: str

    def __init__(self, *, top_k: int = 3) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.top_k = top_k

    @abstractmethod
    def run(self, sample: BenchmarkSample) -> BaselineOutput:
        """Run the baseline on one sample."""


class ModelBackedBenchmarkBaseline(BenchmarkBaseline):
    """Base class for baselines that need a persisted learned-policy model."""

    def __init__(
        self,
        *,
        top_k: int = 3,
        model_path: str | Path = "outputs/models/learned_policy.json",
    ) -> None:
        super().__init__(top_k=top_k)
        self.model_path = Path(model_path).expanduser()
        self.model = LogisticPolicyModel.load(self.model_path)


class NoMemoryBaseline(BenchmarkBaseline):
    """Answer without reading or writing any memory."""

    name = "no_memory"

    def run(self, sample: BenchmarkSample) -> BaselineOutput:
        started_at = perf_counter()
        return BaselineOutput(
            system_name=self.name,
            answer="No memory available.",
            latency_ms=_elapsed_ms(started_at),
            metadata={
                "memory_count_before": len(_memory_records(sample, "initial_memories")),
                "memory_count_after": len(_memory_records(sample, "initial_memories")),
            },
        )


class FullContextBaseline(BenchmarkBaseline):
    """Retrieve every active memory after naively appending new candidates."""

    name = "full_context"

    def run(self, sample: BenchmarkSample) -> BaselineOutput:
        started_at = perf_counter()
        store = BenchmarkMemoryStore.from_sample(sample)
        operations: list[OperationType] = []
        for record in _memory_records(sample, "candidate_memories"):
            store.add(memory_from_record(record, record_kind="candidate"))
            operations.append(OperationType.ADD)

        retrieved = store.active_memories()
        operations.append(OperationType.RETRIEVE)
        answer_item = _best_answer_memory(sample.query, retrieved)
        return BaselineOutput(
            system_name=self.name,
            answer=_answer_from_memories([answer_item] if answer_item else []),
            retrieved_memory_ids=[item.id for item in retrieved],
            operations=operations,
            token_count=sum(_estimate_tokens(item.content) for item in retrieved),
            latency_ms=_elapsed_ms(started_at),
            metadata={
                "memory_count_before": len(_memory_records(sample, "initial_memories")),
                "memory_count_after": store.active_count(),
                "retrieval_strategy": "all_active_memories",
            },
        )


class SummaryMemoryBaseline(BenchmarkBaseline):
    """Retrieve over compressed summaries instead of full memory content."""

    name = "summary_memory"

    def run(self, sample: BenchmarkSample) -> BaselineOutput:
        started_at = perf_counter()
        store = BenchmarkMemoryStore.from_sample(sample)
        operations: list[OperationType] = []
        for record in _memory_records(sample, "candidate_memories"):
            store.add(memory_from_record(record, record_kind="candidate"))
            operations.append(OperationType.ADD)

        retrieved = store.search_summaries(sample.query, top_k=self.top_k)
        operations.append(OperationType.RETRIEVE)
        return BaselineOutput(
            system_name=self.name,
            answer=_answer_from_memories(retrieved),
            retrieved_memory_ids=[item.id for item in retrieved],
            operations=operations,
            token_count=sum(_estimate_tokens(item.summary or item.content) for item in retrieved),
            latency_ms=_elapsed_ms(started_at),
            metadata={
                "memory_count_before": len(_memory_records(sample, "initial_memories")),
                "memory_count_after": store.active_count(),
                "retrieval_strategy": "summary_lexical_top_k",
            },
        )


class NaiveVectorRAGBaseline(BenchmarkBaseline):
    """Naive content-retrieval baseline using lexical similarity as a local stand-in."""

    name = "naive_vector_rag"

    def run(self, sample: BenchmarkSample) -> BaselineOutput:
        started_at = perf_counter()
        store = BenchmarkMemoryStore.from_sample(sample)
        operations: list[OperationType] = []
        for record in _memory_records(sample, "candidate_memories"):
            store.add(memory_from_record(record, record_kind="candidate"))
            operations.append(OperationType.ADD)
        retrieved = store.search(sample.query, top_k=self.top_k)
        operations.append(OperationType.RETRIEVE)
        return BaselineOutput(
            system_name=self.name,
            answer=_answer_from_memories(retrieved),
            retrieved_memory_ids=[item.id for item in retrieved],
            operations=operations,
            token_count=sum(_estimate_tokens(item.content) for item in retrieved),
            latency_ms=_elapsed_ms(started_at),
            metadata={
                "memory_count_before": len(_memory_records(sample, "initial_memories")),
                "memory_count_after": store.active_count(),
            },
        )


class VectorRAGRecencyBaseline(BenchmarkBaseline):
    """Naive vector-style retrieval reranked with a recency prior."""

    name = "vector_rag_recency"

    def run(self, sample: BenchmarkSample) -> BaselineOutput:
        started_at = perf_counter()
        store = BenchmarkMemoryStore.from_sample(sample)
        operations: list[OperationType] = []
        for record in _memory_records(sample, "candidate_memories"):
            store.add(memory_from_record(record, record_kind="candidate"))
            operations.append(OperationType.ADD)

        retrieved = store.search_weighted(
            sample.query,
            top_k=self.top_k,
            lexical_weight=0.70,
            recency_weight=0.25,
            importance_weight=0.05,
        )
        operations.append(OperationType.RETRIEVE)
        return _retrieval_output(
            self.name,
            sample,
            store,
            retrieved,
            operations,
            started_at,
            retrieval_strategy="lexical_plus_recency",
        )


class VectorRAGImportanceBaseline(BenchmarkBaseline):
    """Naive vector-style retrieval reranked with an importance prior."""

    name = "vector_rag_importance"

    def run(self, sample: BenchmarkSample) -> BaselineOutput:
        started_at = perf_counter()
        store = BenchmarkMemoryStore.from_sample(sample)
        operations: list[OperationType] = []
        for record in _memory_records(sample, "candidate_memories"):
            store.add(memory_from_record(record, record_kind="candidate"))
            operations.append(OperationType.ADD)

        retrieved = store.search_weighted(
            sample.query,
            top_k=self.top_k,
            lexical_weight=0.70,
            recency_weight=0.05,
            importance_weight=0.25,
        )
        operations.append(OperationType.RETRIEVE)
        return _retrieval_output(
            self.name,
            sample,
            store,
            retrieved,
            operations,
            started_at,
            retrieval_strategy="lexical_plus_importance",
        )


class VMPRuleBaseline(BenchmarkBaseline):
    """Rule-based VMP baseline using Phase 4 features and Phase 5 controller."""

    name = "vmp_rule"

    def __init__(
        self,
        *,
        top_k: int = 3,
        disabled_features: Sequence[str] = (),
        system_name: str | None = None,
    ) -> None:
        super().__init__(top_k=top_k)
        unknown = sorted(set(disabled_features) - _FEATURE_NAMES)
        if unknown:
            raise ValueError(f"Unknown policy features for ablation: {', '.join(unknown)}")
        self.disabled_features = tuple(dict.fromkeys(disabled_features))
        self.name = system_name or self.name
        self.feature_builder = PolicyFeatureBuilder()
        self.controller = RuleBasedPolicyController()

    def run(self, sample: BenchmarkSample) -> BaselineOutput:
        started_at = perf_counter()
        store = BenchmarkMemoryStore.from_sample(sample)
        operations: list[OperationType] = []

        for record in _memory_records(sample, "candidate_memories"):
            candidate = memory_from_record(record, record_kind="candidate")
            operations.extend(self._admit_candidate(candidate, store, sample))

        operations.extend(self._archive_stale_memories(store, sample))
        retrieved = self._retrieve(sample, store)
        operations.append(OperationType.RETRIEVE)
        return BaselineOutput(
            system_name=self.name,
            answer=_answer_from_memories(retrieved),
            retrieved_memory_ids=[item.id for item in retrieved],
            operations=operations,
            token_count=sum(_estimate_tokens(item.content) for item in retrieved),
            latency_ms=_elapsed_ms(started_at),
            metadata={
                "memory_count_before": len(_memory_records(sample, "initial_memories")),
                "memory_count_after": store.active_count(),
                "disabled_features": list(self.disabled_features),
            },
        )

    def _admit_candidate(
        self,
        candidate: MemoryItem,
        store: "BenchmarkMemoryStore",
        sample: BenchmarkSample,
    ) -> list[OperationType]:
        context = PolicyFeatureContext(
            query=sample.query,
            target_scope=str(sample.metadata.get("target_scope", candidate.scope)),
            existing_memories=store.active_memories(),
        )
        features = _apply_feature_overrides(
            self.feature_builder.build_for_memory(candidate, context),
            candidate,
        )
        features = _mask_features(features, self.disabled_features)
        target, similarity = store.best_match(candidate)
        if target is not None:
            effective_similarity = _effective_similarity(similarity, features, target, candidate)
            update_decision = self.controller.decide_update(
                features,
                PolicyScoreContext(
                    semantic_similarity_to_existing=effective_similarity,
                    source_priority=_source_priority(candidate),
                ),
            )
            if update_decision.op == OperationType.UPDATE:
                store.update(
                    target.id,
                    _update_patch(candidate, features),
                )
                return [OperationType.UPDATE]

            merge_decision = self.controller.decide_merge(
                features,
                PolicyScoreContext(semantic_similarity=effective_similarity),
            )
            if merge_decision.op == OperationType.MERGE:
                store.update(target.id, _merge_patch(target, candidate, features))
                return [OperationType.MERGE]

        write_decision = self.controller.decide_write(features)
        if write_decision.op == OperationType.ADD:
            store.add(_with_features(candidate, features))
            return [OperationType.ADD]
        return [OperationType.IGNORE]

    def _archive_stale_memories(
        self,
        store: "BenchmarkMemoryStore",
        sample: BenchmarkSample,
    ) -> list[OperationType]:
        operations: list[OperationType] = []
        superseded_ids = set(_string_list(sample.metadata.get("superseded_memory_ids", [])))
        for item in list(store.active_memories()):
            features = _apply_feature_overrides(
                self.feature_builder.build_for_memory(
                    item,
                    PolicyFeatureContext(query=sample.query, target_scope=item.scope),
                ),
                item,
            )
            features = _mask_features(features, self.disabled_features)
            decision = self.controller.decide_archive(
                features,
                PolicyScoreContext(superseded=float(item.id in superseded_ids)),
            )
            if decision.op == OperationType.ARCHIVE:
                store.archive(item.id)
                operations.append(OperationType.ARCHIVE)
        return operations

    def _retrieve(self, sample: BenchmarkSample, store: "BenchmarkMemoryStore") -> list[MemoryItem]:
        ranked: list[tuple[float, MemoryItem]] = []
        target_scope = str(sample.metadata.get("target_scope", "global"))
        for item in store.active_memories():
            features = _apply_feature_overrides(
                self.feature_builder.build_for_memory(
                    item,
                    PolicyFeatureContext(query=sample.query, target_scope=target_scope),
                ),
                item,
            )
            features = _mask_features(features, self.disabled_features)
            decision = self.controller.decide_retrieve(features)
            if decision.op == OperationType.RETRIEVE:
                ranked.append((decision.score, item))
        ranked.sort(key=lambda pair: (-pair[0], pair[1].id))
        return [item for _, item in ranked[: self.top_k]]


class LearnedPolicyBaseline(ModelBackedBenchmarkBaseline):
    """Learned operation-policy baseline backed by a trained logistic model."""

    name = "learned_policy"

    def __init__(
        self,
        *,
        top_k: int = 3,
        model_path: str | Path = "outputs/models/learned_policy.json",
        retrieve_threshold: float = 0.05,
    ) -> None:
        super().__init__(top_k=top_k, model_path=model_path)
        self.feature_builder = PolicyFeatureBuilder()
        self.retrieve_threshold = retrieve_threshold

    def run(self, sample: BenchmarkSample) -> BaselineOutput:
        started_at = perf_counter()
        store = BenchmarkMemoryStore.from_sample(sample)
        operations: list[OperationType] = []
        predictions: list[dict[str, JsonValue]] = []

        for record in _memory_records(sample, "candidate_memories"):
            candidate = memory_from_record(record, record_kind="candidate")
            op, prediction = self._admit_candidate(candidate, store, sample)
            operations.append(op)
            predictions.append(prediction)

        for op, prediction in self._archive_stale_memories(store, sample):
            operations.append(op)
            predictions.append(prediction)

        retrieved, retrieve_predictions = self._retrieve(sample, store)
        operations.append(OperationType.RETRIEVE)
        predictions.extend(retrieve_predictions)
        return BaselineOutput(
            system_name=self.name,
            answer=_answer_from_memories(retrieved),
            retrieved_memory_ids=[item.id for item in retrieved],
            operations=operations,
            token_count=sum(_estimate_tokens(item.content) for item in retrieved),
            latency_ms=_elapsed_ms(started_at),
            metadata={
                "memory_count_before": len(_memory_records(sample, "initial_memories")),
                "memory_count_after": store.active_count(),
                "model_path": str(self.model_path),
                "model_type": str(self.model.metadata.get("model_type", "unknown")),
                "predictions": predictions[:20],
            },
        )

    def _admit_candidate(
        self,
        candidate: MemoryItem,
        store: "BenchmarkMemoryStore",
        sample: BenchmarkSample,
    ) -> tuple[OperationType, dict[str, JsonValue]]:
        context = PolicyFeatureContext(
            query=sample.query,
            target_scope=str(sample.metadata.get("target_scope", candidate.scope)),
            existing_memories=store.active_memories(),
        )
        features = _apply_feature_overrides(
            self.feature_builder.build_for_memory(candidate, context),
            candidate,
        )
        target, _ = store.best_match(candidate)
        prediction = self.model.predict(features)
        feasible = [OperationType.ADD, OperationType.IGNORE]
        if target is not None:
            feasible.extend([OperationType.UPDATE, OperationType.MERGE])
        op = _best_feasible_operation(prediction.probabilities, feasible)
        if op == OperationType.ADD:
            store.add(_with_features(candidate, features))
        elif op == OperationType.UPDATE and target is not None:
            store.update(target.id, _update_patch(candidate, features))
        elif op == OperationType.MERGE and target is not None:
            store.update(target.id, _merge_patch(target, candidate, features))
        return op, _prediction_metadata("candidate_admission", candidate.id, prediction)

    def _archive_stale_memories(
        self,
        store: "BenchmarkMemoryStore",
        sample: BenchmarkSample,
    ) -> list[tuple[OperationType, dict[str, JsonValue]]]:
        outputs: list[tuple[OperationType, dict[str, JsonValue]]] = []
        target_scope = str(sample.metadata.get("target_scope", "global"))
        for item in list(store.active_memories()):
            features = _apply_feature_overrides(
                self.feature_builder.build_for_memory(
                    item,
                    PolicyFeatureContext(query=sample.query, target_scope=target_scope),
                ),
                item,
            )
            prediction = self.model.predict(features)
            op = _best_feasible_operation(
                prediction.probabilities,
                [OperationType.ARCHIVE, OperationType.IGNORE],
            )
            if op == OperationType.ARCHIVE:
                store.archive(item.id)
                outputs.append(
                    (op, _prediction_metadata("archive_review", item.id, prediction))
                )
            else:
                outputs.append(
                    (
                        OperationType.IGNORE,
                        _prediction_metadata("archive_review", item.id, prediction),
                    )
                )
        return outputs

    def _retrieve(
        self,
        sample: BenchmarkSample,
        store: "BenchmarkMemoryStore",
    ) -> tuple[list[MemoryItem], list[dict[str, JsonValue]]]:
        ranked: list[tuple[float, MemoryItem]] = []
        predictions: list[dict[str, JsonValue]] = []
        target_scope = str(sample.metadata.get("target_scope", "global"))
        for item in store.active_memories():
            features = _apply_feature_overrides(
                self.feature_builder.build_for_memory(
                    item,
                    PolicyFeatureContext(query=sample.query, target_scope=target_scope),
                ),
                item,
            )
            prediction = self.model.predict(features)
            retrieve_probability = float(
                prediction.probabilities.get(OperationType.RETRIEVE.value, 0.0)
            )
            ranked.append((retrieve_probability, item))
            predictions.append(_prediction_metadata("retrieval", item.id, prediction))
        ranked.sort(key=lambda pair: (-pair[0], pair[1].id))
        retrieved = [
            item
            for probability, item in ranked[: self.top_k]
            if probability >= self.retrieve_threshold
        ]
        if not retrieved and ranked:
            retrieved = [ranked[0][1]]
        return retrieved, predictions


class BenchmarkMemoryStore:
    """Small in-memory store for deterministic benchmark baselines."""

    def __init__(self, memories: Sequence[MemoryItem] = ()) -> None:
        self.memories: dict[str, MemoryItem] = {item.id: item for item in memories}

    @classmethod
    def from_sample(cls, sample: BenchmarkSample) -> "BenchmarkMemoryStore":
        return cls(
            memory_from_record(record, record_kind="initial")
            for record in _memory_records(sample, "initial_memories")
        )

    def add(self, item: MemoryItem) -> MemoryItem:
        self.memories[item.id] = item
        return item

    def update(self, memory_id: str, patch: Mapping[str, Any]) -> MemoryItem:
        current = self.memories[memory_id]
        payload = _deep_merge(current.model_dump(mode="python"), patch)
        metadata = dict(payload["metadata"])
        metadata["version"] = current.metadata.version + 1
        metadata["created_at"] = current.metadata.created_at
        metadata["updated_at"] = utc_now()
        payload["metadata"] = metadata
        updated = MemoryItem.model_validate(payload)
        self.memories[memory_id] = updated
        return updated

    def archive(self, memory_id: str) -> MemoryItem:
        current = self.memories[memory_id]
        payload = current.model_dump(mode="python")
        metadata = dict(payload["metadata"])
        metadata["version"] = current.metadata.version + 1
        metadata["updated_at"] = utc_now()
        metadata["status"] = MemoryStatus.ARCHIVED
        payload["metadata"] = metadata
        archived = MemoryItem.model_validate(payload)
        self.memories[memory_id] = archived
        return archived

    def active_memories(self) -> list[MemoryItem]:
        return [
            item
            for item in self.memories.values()
            if item.metadata.status == MemoryStatus.ACTIVE
        ]

    def active_count(self) -> int:
        return len(self.active_memories())

    def search(self, query: str, *, top_k: int) -> list[MemoryItem]:
        ranked = [
            (_lexical_similarity(query, item.content + " " + (item.summary or "")), item)
            for item in self.active_memories()
        ]
        ranked.sort(key=lambda pair: (-pair[0], pair[1].id))
        return [item for score, item in ranked[:top_k] if score > 0.0]

    def search_summaries(self, query: str, *, top_k: int) -> list[MemoryItem]:
        ranked = [
            (_lexical_similarity(query, item.summary or item.content), item)
            for item in self.active_memories()
        ]
        ranked.sort(key=lambda pair: (-pair[0], pair[1].id))
        return [item for score, item in ranked[:top_k] if score > 0.0]

    def search_weighted(
        self,
        query: str,
        *,
        top_k: int,
        lexical_weight: float,
        recency_weight: float,
        importance_weight: float,
    ) -> list[MemoryItem]:
        ranked = []
        for item in self.active_memories():
            lexical = _lexical_similarity(query, item.content + " " + (item.summary or ""))
            score = (
                lexical_weight * lexical
                + recency_weight * item.features.recency
                + importance_weight * item.features.importance
            )
            ranked.append((_clamp01(score), item))
        ranked.sort(key=lambda pair: (-pair[0], pair[1].id))
        return [item for score, item in ranked[:top_k] if score > 0.0]

    def best_match(self, candidate: MemoryItem) -> tuple[MemoryItem | None, float]:
        ranked = [
            (_memory_similarity(candidate, item), item)
            for item in self.active_memories()
            if item.id != candidate.id
        ]
        if not ranked:
            return None, 0.0
        ranked.sort(key=lambda pair: (-pair[0], pair[1].id))
        score, item = ranked[0]
        return item, score


def baseline_for_name(name: str, *, top_k: int = 3) -> BenchmarkBaseline:
    """Instantiate a configured baseline by CLI name."""

    return baseline_for_name_with_options(name, top_k=top_k)


def baseline_for_name_with_options(
    name: str,
    *,
    top_k: int = 3,
    model_path: str | Path = "outputs/models/learned_policy.json",
) -> BenchmarkBaseline:
    """Instantiate a configured baseline by CLI name and optional model path."""

    normalized = name.strip().casefold()
    if normalized == NoMemoryBaseline.name:
        return NoMemoryBaseline(top_k=top_k)
    if normalized == FullContextBaseline.name:
        return FullContextBaseline(top_k=top_k)
    if normalized == SummaryMemoryBaseline.name:
        return SummaryMemoryBaseline(top_k=top_k)
    if normalized == NaiveVectorRAGBaseline.name:
        return NaiveVectorRAGBaseline(top_k=top_k)
    if normalized == "vector_rag":
        return NaiveVectorRAGBaseline(top_k=top_k)
    if normalized == VectorRAGRecencyBaseline.name:
        return VectorRAGRecencyBaseline(top_k=top_k)
    if normalized == VectorRAGImportanceBaseline.name:
        return VectorRAGImportanceBaseline(top_k=top_k)
    if normalized == VMPRuleBaseline.name:
        return VMPRuleBaseline(top_k=top_k)
    if normalized in {LearnedPolicyBaseline.name, "learned"}:
        return LearnedPolicyBaseline(top_k=top_k, model_path=model_path)
    raise ValueError(f"Unknown benchmark baseline: {name}")


def memory_from_record(
    record: Mapping[str, Any],
    *,
    record_kind: str | None = None,
) -> MemoryItem:
    """Convert structured benchmark metadata into a MemoryItem."""

    raw_features = _features_from_record(record, record_kind=record_kind)
    attributes = dict(_mapping(record.get("attributes")))
    if record_kind:
        attributes["benchmark_record_kind"] = record_kind
    tags = _string_list(record.get("tags", []))
    if tags:
        attributes["tags"] = tags
    if raw_features:
        attributes["feature_overrides"] = raw_features
    metadata = {
        "status": str(record.get("status", MemoryStatus.ACTIVE.value)),
        "attributes": attributes,
    }
    if "access_count" in record:
        metadata["access_count"] = record["access_count"]
    values = {
        "id": str(record["id"]),
        "type": str(record.get("type", MemoryType.SEMANTIC.value)),
        "scope": str(record.get("scope", "global")),
        "content": str(record["content"]),
        "summary": record.get("summary"),
        "source": MemorySource(
            event_id=record.get("source_event_id"),
            source_type=str(record.get("source_type", "benchmark")),
        ),
        "features": PolicyFeatures.model_validate(raw_features or {}),
        "metadata": metadata,
    }
    return MemoryItem.model_validate(values)


def _memory_records(sample: BenchmarkSample, key: str) -> list[Mapping[str, Any]]:
    raw_value = sample.metadata.get(key, [])
    if not isinstance(raw_value, list):
        raise ValueError(f"sample.metadata[{key!r}] must be a list")
    records: list[Mapping[str, Any]] = []
    for index, item in enumerate(raw_value):
        if not isinstance(item, dict):
            raise ValueError(f"sample.metadata[{key!r}][{index}] must be an object")
        records.append(item)
    return records


def _features_from_record(
    record: Mapping[str, Any],
    *,
    record_kind: str | None = None,
) -> dict[str, float]:
    raw_features = dict(_mapping(record.get("features")))
    for key in ("importance", "confidence", "recency"):
        if key in record and key not in raw_features:
            raw_features[key] = record[key]
    if "recency" not in raw_features:
        if record_kind == "candidate":
            raw_features["recency"] = 1.0
        elif record_kind == "initial":
            raw_features["recency"] = 0.35
    return {
        key: float(value)
        for key, value in raw_features.items()
        if key in _FEATURE_NAMES and isinstance(value, int | float)
    }


def _apply_feature_overrides(features: PolicyFeatures, item: MemoryItem) -> PolicyFeatures:
    raw_overrides = item.metadata.attributes.get("feature_overrides", {})
    overrides = {
        key: value
        for key, value in _mapping(raw_overrides).items()
        if key in _FEATURE_NAMES and isinstance(value, int | float)
    }
    if not overrides:
        return features
    payload = features.model_dump(mode="python")
    payload.update(overrides)
    return PolicyFeatures.model_validate(payload)


def _with_features(item: MemoryItem, features: PolicyFeatures) -> MemoryItem:
    payload = item.model_dump(mode="python")
    payload["features"] = features
    payload["policy_embedding"] = list(features.as_vector())
    return MemoryItem.model_validate(payload)


def _mask_features(features: PolicyFeatures, disabled_features: Sequence[str]) -> PolicyFeatures:
    if not disabled_features:
        return features
    payload = features.model_dump(mode="python")
    for feature_name in disabled_features:
        payload[feature_name] = 0.0
    return PolicyFeatures.model_validate(payload)


def _update_patch(candidate: MemoryItem, features: PolicyFeatures) -> dict[str, Any]:
    return {
        "content": candidate.content,
        "summary": candidate.summary,
        "features": features.model_dump(mode="python"),
        "policy_embedding": list(features.as_vector()),
        "metadata": {
            "attributes": {
                **candidate.metadata.attributes,
                "updated_from_candidate_id": candidate.id,
            }
        },
    }


def _merge_patch(
    target: MemoryItem,
    candidate: MemoryItem,
    features: PolicyFeatures,
) -> dict[str, Any]:
    content = _merge_texts([target.content, candidate.content])
    summary = _merge_texts([target.summary or "", candidate.summary or ""]).replace("\n\n", "; ")
    return {
        "content": content,
        "summary": summary or target.summary or candidate.summary,
        "features": features.model_dump(mode="python"),
        "policy_embedding": list(features.as_vector()),
        "metadata": {
            "attributes": {
                **target.metadata.attributes,
                "merged_candidate_ids": [candidate.id],
            }
        },
    }


def _answer_from_memories(memories: Sequence[MemoryItem]) -> str | None:
    if not memories:
        return None
    return memories[0].summary or memories[0].content


def _best_answer_memory(query: str, memories: Sequence[MemoryItem]) -> MemoryItem | None:
    if not memories:
        return None
    ranked = [
        (
            _lexical_similarity(query, item.content + " " + (item.summary or ""))
            + 0.10 * item.features.recency
            + 0.05 * item.features.importance,
            item,
        )
        for item in memories
    ]
    ranked.sort(key=lambda pair: (-pair[0], pair[1].id))
    return ranked[0][1]


def _retrieval_output(
    system_name: str,
    sample: BenchmarkSample,
    store: BenchmarkMemoryStore,
    retrieved: Sequence[MemoryItem],
    operations: Sequence[OperationType],
    started_at: float,
    *,
    retrieval_strategy: str,
) -> BaselineOutput:
    return BaselineOutput(
        system_name=system_name,
        answer=_answer_from_memories(retrieved),
        retrieved_memory_ids=[item.id for item in retrieved],
        operations=list(operations),
        token_count=sum(_estimate_tokens(item.content) for item in retrieved),
        latency_ms=_elapsed_ms(started_at),
        metadata={
            "memory_count_before": len(_memory_records(sample, "initial_memories")),
            "memory_count_after": store.active_count(),
            "retrieval_strategy": retrieval_strategy,
        },
    )


def _best_feasible_operation(
    probabilities: Mapping[str, float],
    feasible: Sequence[OperationType],
) -> OperationType:
    ranked = [
        (float(probabilities.get(operation.value, 0.0)), operation)
        for operation in feasible
    ]
    ranked.sort(key=lambda pair: (-pair[0], pair[1].value))
    return ranked[0][1]


def _prediction_metadata(
    decision_stage: str,
    memory_id: str,
    prediction: Any,
) -> dict[str, JsonValue]:
    return {
        "decision_stage": decision_stage,
        "memory_id": memory_id,
        "predicted_op": prediction.predicted_op.value,
        "confidence": prediction.confidence,
        "probabilities": dict(prediction.probabilities),
    }


def _effective_similarity(
    similarity: float,
    features: PolicyFeatures,
    target: MemoryItem,
    candidate: MemoryItem,
) -> float:
    if target.scope == candidate.scope and features.contradiction >= 0.45:
        return max(similarity, 0.8)
    return similarity


def _source_priority(item: MemoryItem) -> float:
    raw_value = item.metadata.attributes.get("source_priority", 0.8)
    return _clamp01(float(raw_value)) if isinstance(raw_value, int | float) else 0.8


def _memory_similarity(left: MemoryItem, right: MemoryItem) -> float:
    content = _lexical_similarity(left.content, right.content)
    scope = 1.0 if left.scope == right.scope else _lexical_similarity(left.scope, right.scope)
    type_bonus = 0.1 if left.type == right.type else 0.0
    return _clamp01(0.65 * content + 0.25 * scope + type_bonus)


def _lexical_similarity(left: str, right: str) -> float:
    left_terms = set(_terms(left))
    right_terms = set(_terms(right))
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


def _terms(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _TOKEN_PATTERN.finditer(text)]


def _merge_texts(values: Sequence[str]) -> str:
    paragraphs: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(value.split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            paragraphs.append(value.strip())
    return "\n\n".join(paragraphs)


def _deep_merge(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in patch.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _elapsed_ms(started_at: float) -> float:
    return (perf_counter() - started_at) * 1000.0


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))
