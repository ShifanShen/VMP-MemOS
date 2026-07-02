"""Tests for backend-agnostic execution of policy decisions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from vmp_memos.backends.base import BaseMemoryBackend, MemoryNotFoundError
from vmp_memos.operations import (
    MemoryOperationExecutor,
    MergePlan,
    OperationExecutionError,
    OperationExecutionStatus,
    RetrievalPlan,
)
from vmp_memos.policy import PolicyScoreContext, RuleBasedPolicyController
from vmp_memos.schemas import (
    MemoryItem,
    MemoryOperation,
    MemorySource,
    MemoryStatus,
    OperationType,
    PolicyFeatures,
)
from vmp_memos.schemas.base import utc_now


class InMemoryBackend(BaseMemoryBackend):
    """Small backend implementation used to isolate executor tests."""

    backend_name = "in_memory"

    def __init__(self, root: Path) -> None:
        self.items: dict[str, MemoryItem] = {}
        self.operation_log_path = root / "logs" / "operations.jsonl"
        self.retrieval_log_path = root / "logs" / "retrievals.jsonl"
        self.operation_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.operation_log_path.touch()
        self.retrieval_log_path.touch()

    def add(
        self,
        memory_item: MemoryItem,
        *,
        reason: str = "Added memory item.",
        policy_score: float | None = None,
        confidence: float | None = None,
    ) -> MemoryItem:
        self.items[memory_item.id] = memory_item
        self._log(OperationType.ADD, memory_item, reason, policy_score, confidence)
        return memory_item

    def update(
        self,
        memory_id: str,
        patch: Mapping[str, Any],
        *,
        reason: str = "Updated memory item.",
        policy_score: float | None = None,
        confidence: float | None = None,
    ) -> MemoryItem:
        current = self.get(memory_id)
        payload = self._deep_merge(current.model_dump(mode="python"), patch)
        metadata = dict(payload["metadata"])
        metadata["version"] = current.metadata.version + 1
        metadata["created_at"] = current.metadata.created_at
        metadata["updated_at"] = utc_now()
        payload["metadata"] = metadata
        updated = MemoryItem.model_validate(payload)
        self.items[memory_id] = updated
        self._log(OperationType.UPDATE, updated, reason, policy_score, confidence)
        return updated

    def get(self, memory_id: str) -> MemoryItem:
        try:
            return self.items[memory_id]
        except KeyError as exc:
            raise MemoryNotFoundError(memory_id) from exc

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: Mapping[str, Any] | None = None,
    ) -> list[MemoryItem]:
        lowered = query.casefold()
        results = [
            item
            for item in self.list(filters)
            if lowered in item.content.casefold() or lowered in (item.summary or "").casefold()
        ][:top_k]
        MemoryOperation(
            op=OperationType.RETRIEVE,
            reason=f"Retrieved {len(results)} item(s).",
            policy_score=1.0 if results else 0.0,
            confidence=1.0,
            backend=self.backend_name,
            payload={"query": query, "result_ids": [item.id for item in results]},
        ).append_jsonl(self.operation_log_path)
        return results

    def list(self, filters: Mapping[str, Any] | None = None) -> list[MemoryItem]:
        criteria = dict(filters or {})
        requested_status = criteria.get("status")
        include_archived = bool(criteria.get("include_archived", False))
        if isinstance(requested_status, MemoryStatus):
            requested_status = requested_status.value
        if requested_status == MemoryStatus.ARCHIVED.value:
            include_archived = True
        items = []
        for item in self.items.values():
            if not include_archived and item.metadata.status != MemoryStatus.ACTIVE:
                continue
            if requested_status is not None and item.metadata.status.value != requested_status:
                continue
            items.append(item)
        return sorted(items, key=lambda item: item.id)

    def archive(
        self,
        memory_id: str,
        *,
        reason: str = "Archived memory item.",
        policy_score: float | None = None,
        confidence: float | None = None,
    ) -> MemoryItem:
        current = self.get(memory_id)
        payload = current.model_dump(mode="python")
        metadata = dict(payload["metadata"])
        metadata["version"] = current.metadata.version + 1
        metadata["updated_at"] = utc_now()
        metadata["status"] = MemoryStatus.ARCHIVED
        payload["metadata"] = metadata
        archived = MemoryItem.model_validate(payload)
        self.items[memory_id] = archived
        self._log(OperationType.ARCHIVE, archived, reason, policy_score, confidence)
        return archived

    def delete(self, memory_id: str, *, reason: str = "Delete requested.") -> MemoryItem:
        return self.archive(memory_id, reason=f"{reason} Physical deletion is disabled.")

    def persist(self) -> None:
        return None

    def _log(
        self,
        op: OperationType,
        item: MemoryItem,
        reason: str,
        policy_score: float | None,
        confidence: float | None,
    ) -> None:
        MemoryOperation(
            op=op,
            target_memory_id=item.id,
            source_event_id=item.source.event_id,
            reason=reason,
            policy_score=item.features.importance if policy_score is None else policy_score,
            confidence=item.features.confidence if confidence is None else confidence,
            scope=item.scope,
            backend=self.backend_name,
        ).append_jsonl(self.operation_log_path)

    @classmethod
    def _deep_merge(cls, base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
        merged = deepcopy(dict(base))
        for key, value in patch.items():
            current = merged.get(key)
            if isinstance(current, Mapping) and isinstance(value, Mapping):
                merged[key] = cls._deep_merge(current, value)
            else:
                merged[key] = deepcopy(value)
        return merged


def make_memory(content: str, *, summary: str | None = None) -> MemoryItem:
    """Create a representative memory for executor tests."""

    return MemoryItem(
        type="semantic",
        scope="career/agent-dev",
        content=content,
        summary=summary,
        source=MemorySource(event_id="evt_executor", source_type="test"),
        features=PolicyFeatures(importance=0.8, confidence=0.9),
    )


def read_operations(path: Path) -> list[dict]:
    """Read JSONL operation records."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_add_update_archive_retrieve_and_ignore_decisions(tmp_path) -> None:
    backend = InMemoryBackend(tmp_path)
    executor = MemoryOperationExecutor(backend)
    controller = RuleBasedPolicyController()
    memory = make_memory("Agent memory systems matter.", summary="Agent memory")

    add_result = executor.execute(
        controller.decide_write(
            PolicyFeatures(
                importance=0.9,
                novelty=0.9,
                confidence=0.9,
                actionability=0.8,
                scope_match=1.0,
            )
        ),
        memory_item=memory,
    )
    assert add_result.applied
    assert backend.get(memory.id) == memory

    update_result = executor.execute(
        controller.decide_update(
            PolicyFeatures(redundancy=0.9, contradiction=0.8, recency=1.0, confidence=0.9),
            PolicyScoreContext(semantic_similarity_to_existing=0.9, source_priority=1.0),
        ),
        target_memory_id=memory.id,
        patch={"content": "Agent memory systems and policy layers matter."},
    )
    assert update_result.items[0].metadata.version == 2
    assert "policy layers" in backend.get(memory.id).content

    retrieve_result = executor.execute(
        controller.decide_retrieve(
            PolicyFeatures(semantic_relevance=1.0, importance=0.8, confidence=0.9)
        ),
        retrieval_plan=RetrievalPlan(query="policy layers", top_k=3),
    )
    assert retrieve_result.op == OperationType.RETRIEVE
    assert retrieve_result.item_ids == [memory.id]

    archive_result = executor.execute(
        controller.decide_archive(
            PolicyFeatures(staleness=1.0, redundancy=1.0, failure_contribution=1.0)
        ),
        target_memory_id=memory.id,
    )
    assert archive_result.items[0].metadata.status == MemoryStatus.ARCHIVED

    ignore_result = executor.execute(
        controller.decide_write(PolicyFeatures(importance=0.1, redundancy=1.0)),
        target_memory_id=memory.id,
        scope="career/agent-dev",
    )
    assert ignore_result.status == OperationExecutionStatus.IGNORED
    assert ignore_result.op == OperationType.IGNORE
    assert read_operations(backend.operation_log_path)[-1]["op"] == "IGNORE"


def test_merge_updates_target_archives_sources_and_logs_merge(tmp_path) -> None:
    backend = InMemoryBackend(tmp_path)
    executor = MemoryOperationExecutor(backend)
    controller = RuleBasedPolicyController()
    target = backend.add(make_memory("User focuses on Agent work.", summary="Agent work"))
    source = backend.add(
        make_memory("User focuses on LLM applications.", summary="LLM applications")
    )

    result = executor.execute(
        controller.decide_merge(
            PolicyFeatures(redundancy=0.9, scope_match=1.0, contradiction=0.0),
            PolicyScoreContext(semantic_similarity=0.9),
        ),
        merge_plan=MergePlan(target_memory_id=target.id, source_memory_ids=[source.id]),
    )

    updated = backend.get(target.id)
    archived = backend.get(source.id)
    operations = read_operations(backend.operation_log_path)

    assert result.op == OperationType.MERGE
    assert result.applied
    assert "LLM applications" in updated.content
    assert source.id in updated.links.supersedes
    assert archived.metadata.status == MemoryStatus.ARCHIVED
    assert operations[-1]["op"] == "MERGE"
    assert operations[-1]["source_memory_ids"] == [source.id]


def test_executor_rejects_missing_inputs_and_unsupported_ops(tmp_path) -> None:
    backend = InMemoryBackend(tmp_path)
    executor = MemoryOperationExecutor(backend)
    controller = RuleBasedPolicyController()

    with pytest.raises(OperationExecutionError, match="ADD execution requires"):
        executor.execute(
            controller.decide_write(
                PolicyFeatures(
                    importance=0.9,
                    novelty=0.9,
                    confidence=0.9,
                    actionability=0.8,
                    scope_match=1.0,
                )
            )
        )

    with pytest.raises(OperationExecutionError, match="Unsupported operation"):
        executor.execute(
            controller.decide_compress(
                PolicyFeatures(
                    token_cost=1.0,
                    access_frequency=1.0,
                    actionability=1.0,
                    scope_match=1.0,
                )
            )
        )


def test_execution_result_json_round_trip_preserves_identity(tmp_path) -> None:
    backend = InMemoryBackend(tmp_path)
    executor = MemoryOperationExecutor(backend)
    controller = RuleBasedPolicyController()
    memory = make_memory("Agent memory systems matter.")

    result = executor.execute(
        controller.decide_write(
            PolicyFeatures(
                importance=0.9,
                novelty=0.9,
                confidence=0.9,
                actionability=0.8,
                scope_match=1.0,
            )
        ),
        memory_item=memory,
    )

    restored = type(result).model_validate_json(result.model_dump_json())

    assert restored == result
    assert restored.execution_id == result.execution_id
    assert restored.applied
