"""Training-data builders for learned memory policies."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from vmp_memos.benchmark.baselines import (
    BenchmarkMemoryStore,
    _apply_feature_overrides,
    _memory_records,
    _merge_patch,
    _string_list,
    _update_patch,
    _with_features,
    memory_from_record,
)
from vmp_memos.policy import PolicyFeatureBuilder, PolicyFeatureContext
from vmp_memos.policy.learned import PolicyTrainingExample, features_to_mapping
from vmp_memos.schemas import BenchmarkSample, MemoryItem, OperationType, PolicyFeatures


def build_policy_training_examples(
    samples: Sequence[BenchmarkSample],
) -> list[PolicyTrainingExample]:
    """Construct supervised examples from benchmark gold operations."""

    builder = PolicyFeatureBuilder()
    examples: list[PolicyTrainingExample] = []
    for sample in samples:
        store = BenchmarkMemoryStore.from_sample(sample)
        examples.extend(_candidate_examples(sample, store, builder))
        examples.extend(_archive_examples(sample, store, builder))
        examples.extend(_retrieval_examples(sample, store, builder))
    return examples


def load_policy_training_examples_from_operation_logs(
    paths: Iterable[str | Path],
) -> list[PolicyTrainingExample]:
    """Load examples from operation JSONL logs with feature snapshots."""

    examples: list[PolicyTrainingExample] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                loaded = json.loads(line)
                if not isinstance(loaded, dict):
                    continue
                example = _example_from_operation_record(loaded, path, line_number)
                if example is not None:
                    examples.append(example)
    return examples


def write_policy_training_examples(
    path: str | Path,
    examples: Sequence[PolicyTrainingExample],
) -> Path:
    """Write training examples as JSONL for auditability."""

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for example in examples:
            stream.write(example.to_json_line())
            stream.write("\n")
    return output_path


def _candidate_examples(
    sample: BenchmarkSample,
    store: BenchmarkMemoryStore,
    builder: PolicyFeatureBuilder,
) -> list[PolicyTrainingExample]:
    examples: list[PolicyTrainingExample] = []
    for record in _memory_records(sample, "candidate_memories"):
        candidate = memory_from_record(record, record_kind="candidate")
        target_scope = str(sample.metadata.get("target_scope", candidate.scope))
        context = PolicyFeatureContext(
            query=sample.query,
            target_scope=target_scope,
            existing_memories=store.active_memories(),
        )
        features = _apply_feature_overrides(
            builder.build_for_memory(candidate, context),
            candidate,
        )
        target, _ = store.best_match(candidate)
        label = _candidate_label(sample, target)
        examples.append(
            _example(
                features,
                label,
                sample,
                candidate.id,
                decision_stage="candidate_admission",
            )
        )
        _apply_expected_candidate_mutation(store, candidate, features, label, target)
    return examples


def _archive_examples(
    sample: BenchmarkSample,
    store: BenchmarkMemoryStore,
    builder: PolicyFeatureBuilder,
) -> list[PolicyTrainingExample]:
    superseded_ids = set(_string_list(sample.metadata.get("superseded_memory_ids", [])))
    stale_ids = set(_string_list(sample.metadata.get("stale_memory_ids", [])))
    target_scope = str(sample.metadata.get("target_scope", "global"))
    examples: list[PolicyTrainingExample] = []
    for item in list(store.active_memories()):
        features = _apply_feature_overrides(
            builder.build_for_memory(
                item,
                PolicyFeatureContext(query=sample.query, target_scope=target_scope),
            ),
            item,
        )
        should_archive = item.id in superseded_ids or item.id in stale_ids
        label = (
            OperationType.ARCHIVE
            if should_archive and OperationType.ARCHIVE in sample.expected_operations
            else OperationType.IGNORE
        )
        examples.append(
            _example(
                features,
                label,
                sample,
                item.id,
                decision_stage="archive_review",
            )
        )
        if label == OperationType.ARCHIVE:
            store.archive(item.id)
    return examples


def _retrieval_examples(
    sample: BenchmarkSample,
    store: BenchmarkMemoryStore,
    builder: PolicyFeatureBuilder,
) -> list[PolicyTrainingExample]:
    gold_ids = set(sample.gold_memory_ids)
    target_scope = str(sample.metadata.get("target_scope", "global"))
    examples: list[PolicyTrainingExample] = []
    for item in store.active_memories():
        features = _apply_feature_overrides(
            builder.build_for_memory(
                item,
                PolicyFeatureContext(query=sample.query, target_scope=target_scope),
            ),
            item,
        )
        label = OperationType.RETRIEVE if item.id in gold_ids else OperationType.IGNORE
        examples.append(
            _example(
                features,
                label,
                sample,
                item.id,
                decision_stage="retrieval",
            )
        )
    return examples


def _candidate_label(
    sample: BenchmarkSample,
    target: MemoryItem | None,
) -> OperationType:
    expected = set(sample.expected_operations)
    if target is not None and OperationType.UPDATE in expected:
        return OperationType.UPDATE
    if target is not None and OperationType.MERGE in expected:
        return OperationType.MERGE
    if OperationType.ADD in expected or OperationType.COMPRESS in expected:
        return OperationType.ADD
    return OperationType.IGNORE


def _apply_expected_candidate_mutation(
    store: BenchmarkMemoryStore,
    candidate: MemoryItem,
    features: PolicyFeatures,
    label: OperationType,
    target: MemoryItem | None,
) -> None:
    if label == OperationType.ADD:
        store.add(_with_features(candidate, features))
    elif label == OperationType.UPDATE and target is not None:
        store.update(target.id, _update_patch(candidate, features))
    elif label == OperationType.MERGE and target is not None:
        store.update(target.id, _merge_patch(target, candidate, features))


def _example(
    features: PolicyFeatures,
    label: OperationType,
    sample: BenchmarkSample,
    memory_id: str,
    *,
    decision_stage: str,
) -> PolicyTrainingExample:
    return PolicyTrainingExample(
        features=features_to_mapping(features),
        label=label,
        sample_id=sample.sample_id,
        memory_id=memory_id,
        source="benchmark",
        metadata={
            "decision_stage": decision_stage,
            "task_type": str(sample.metadata.get("task_type", "unknown")),
        },
    )


def _example_from_operation_record(
    record: Mapping[str, Any],
    path: Path,
    line_number: int,
) -> PolicyTrainingExample | None:
    raw_op = record.get("op")
    payload = record.get("payload", {})
    if not isinstance(raw_op, str) or not isinstance(payload, dict):
        return None
    raw_features = payload.get("feature_snapshot")
    if not isinstance(raw_features, dict):
        return None
    try:
        label = OperationType(raw_op)
        features = _coerce_features(raw_features)
    except ValueError:
        return None
    return PolicyTrainingExample(
        features=features,
        label=label,
        sample_id=_optional_str(record.get("source_event_id")),
        memory_id=_optional_str(record.get("target_memory_id")),
        source="operation_log",
        metadata={
            "path": str(path),
            "line_number": line_number,
            "decision_id": str(payload.get("decision_id", "")),
        },
    )


def _coerce_features(raw_features: Mapping[str, Any]) -> dict[str, float]:
    return {
        name: _clamp01(float(raw_features.get(name, 0.0)))
        for name in PolicyFeatures.FEATURE_NAMES
    }


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None and str(value).strip() else None


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))
