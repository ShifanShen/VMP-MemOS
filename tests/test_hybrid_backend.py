"""Tests for the hybrid file-plus-vector backend orchestration."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from vmp_memos.backends import (
    HybridMemoryBackend,
    MemoryAlreadyExistsError,
    MemoryNotFoundError,
)
from vmp_memos.backends.base import BaseMemoryBackend
from vmp_memos.schemas import (
    MemoryItem,
    MemoryOperation,
    MemorySource,
    MemoryStatus,
    OperationType,
    PolicyFeatures,
)
from vmp_memos.schemas.base import utc_now


class FakeBackend(BaseMemoryBackend):
    """Simple backend used to test HybridMemoryBackend without optional deps."""

    def __init__(self, root: Path, *, backend_name: str, enrich_vectors: bool = False) -> None:
        self.backend_name = backend_name
        self.enrich_vectors = enrich_vectors
        self.items: dict[str, MemoryItem] = {}
        self.operation_log_path = root / f"{backend_name}_operations.jsonl"
        self.retrieval_log_path = root / f"{backend_name}_retrievals.jsonl"
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
        if memory_item.id in self.items:
            raise MemoryAlreadyExistsError(memory_item.id)
        item = self._maybe_enrich(memory_item)
        self.items[item.id] = item
        self._log(OperationType.ADD, item, reason, policy_score, confidence)
        return item

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
        item = self._maybe_enrich(MemoryItem.model_validate(payload))
        self.items[memory_id] = item
        self._log(OperationType.UPDATE, item, reason, policy_score, confidence)
        return item

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
        del query, filters
        return list(self.items.values())[:top_k]

    def list(self, filters: Mapping[str, Any] | None = None) -> list[MemoryItem]:
        include_archived = bool((filters or {}).get("include_archived", False))
        return [
            item
            for item in self.items.values()
            if include_archived or item.metadata.status == MemoryStatus.ACTIVE
        ]

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
        metadata["status"] = MemoryStatus.ARCHIVED
        metadata["updated_at"] = utc_now()
        payload["metadata"] = metadata
        archived = MemoryItem.model_validate(payload)
        self.items[memory_id] = archived
        self._log(OperationType.ARCHIVE, archived, reason, policy_score, confidence)
        return archived

    def delete(self, memory_id: str, *, reason: str = "Delete requested.") -> MemoryItem:
        return self.archive(memory_id, reason=reason)

    def persist(self) -> None:
        return None

    def _maybe_enrich(self, item: MemoryItem) -> MemoryItem:
        if not self.enrich_vectors or item.content_embedding:
            return item
        payload = item.model_dump(mode="python")
        payload["content_embedding"] = [1.0, 0.0]
        return MemoryItem.model_validate(payload)

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


def make_memory(content: str = "Agent memory systems matter.") -> MemoryItem:
    """Create a memory item for hybrid backend tests."""

    return MemoryItem(
        type="semantic",
        scope="career/agent-dev",
        content=content,
        summary=content,
        source=MemorySource(event_id="evt_hybrid", source_type="test"),
        features=PolicyFeatures(importance=0.8, confidence=0.9),
    )


def read_jsonl(path: Path) -> list[dict]:
    """Read JSONL records from ``path``."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def make_hybrid(tmp_path) -> tuple[HybridMemoryBackend, FakeBackend, FakeBackend]:
    """Create a hybrid backend with fake file and vector components."""

    file_backend = FakeBackend(tmp_path, backend_name="file")
    vector_backend = FakeBackend(tmp_path, backend_name="vector", enrich_vectors=True)
    hybrid = HybridMemoryBackend(
        tmp_path / "workspace",
        file_backend=file_backend,
        vector_backend=vector_backend,
    )
    return hybrid, file_backend, vector_backend


def test_add_writes_vector_enriched_memory_to_file_source_of_truth(tmp_path) -> None:
    hybrid, file_backend, vector_backend = make_hybrid(tmp_path)
    memory = make_memory()

    stored = hybrid.add(memory, reason="Hybrid add.")

    assert stored.id == memory.id
    assert stored.content_embedding == [1.0, 0.0]
    assert file_backend.get(memory.id) == stored
    assert vector_backend.get(memory.id).content_embedding == [1.0, 0.0]

    with pytest.raises(MemoryAlreadyExistsError):
        hybrid.add(memory)


def test_update_refreshes_vector_then_updates_file_once(tmp_path) -> None:
    hybrid, file_backend, vector_backend = make_hybrid(tmp_path)
    memory = hybrid.add(make_memory("Old Java direction."))

    updated = hybrid.update(
        memory.id,
        {"content": "Current Agent and LLM direction."},
        reason="Hybrid update.",
    )

    assert updated.metadata.version == 2
    assert file_backend.get(memory.id).metadata.version == 2
    assert vector_backend.get(memory.id).metadata.version == 2
    assert updated.content_embedding == [1.0, 0.0]
    assert "Agent" in updated.content


def test_search_uses_vector_order_and_hydrates_from_file(tmp_path) -> None:
    hybrid, file_backend, vector_backend = make_hybrid(tmp_path)
    first = hybrid.add(make_memory("Agent memory systems matter."))
    second = hybrid.add(make_memory("LLM app development matters."))

    # Force vector order to differ from insertion order. Hybrid should preserve
    # vector ranking but hydrate the returned objects from file storage.
    vector_backend.items = {
        second.id: vector_backend.items[second.id],
        first.id: vector_backend.items[first.id],
    }

    results = hybrid.search("Agent", top_k=2, filters={"scope": "career/agent-dev"})

    assert results == [file_backend.get(second.id), file_backend.get(first.id)]
    retrievals = read_jsonl(hybrid.retrieval_log_path)
    operations = read_jsonl(hybrid.operation_log_path)
    assert retrievals[-1]["backend"] == "hybrid"
    assert retrievals[-1]["metadata"]["retrieval_method"] == "hybrid_vector_hydrate"
    assert operations[-1]["op"] == "RETRIEVE"
    assert operations[-1]["backend"] == "hybrid"


def test_archive_and_delete_sync_both_components(tmp_path) -> None:
    hybrid, file_backend, vector_backend = make_hybrid(tmp_path)
    memory = hybrid.add(make_memory())

    archived = hybrid.delete(memory.id, reason="Delete request.")

    assert archived.metadata.status == MemoryStatus.ARCHIVED
    assert file_backend.get(memory.id).metadata.status == MemoryStatus.ARCHIVED
    assert vector_backend.get(memory.id).metadata.status == MemoryStatus.ARCHIVED


def test_reindex_adds_missing_vector_entries(tmp_path) -> None:
    hybrid, file_backend, vector_backend = make_hybrid(tmp_path)
    memory = file_backend.add(make_memory("Manual Markdown import."))

    count = hybrid.reindex()

    assert count == 1
    assert vector_backend.get(memory.id).content == memory.content
