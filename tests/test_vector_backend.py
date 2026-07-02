"""Tests for the SQLite VectorMemoryBackend."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence

import pytest

from vmp_memos.backends import (
    InvalidMemoryIdError,
    MemoryAlreadyExistsError,
    MemoryNotFoundError,
    VectorDimensionError,
    VectorMemoryBackend,
    VectorStoreError,
    cosine_similarity,
)
from vmp_memos.embeddings import BaseEmbedder, EmbeddingDimensionError
from vmp_memos.schemas import MemoryItem, MemorySource, MemoryStatus, PolicyFeatures


class KeywordEmbedder(BaseEmbedder):
    """Deterministic keyword embedder used instead of a real model in tests."""

    def __init__(
        self,
        *,
        identifier: str = "test-keyword-embedder",
        dimension: int = 4,
    ) -> None:
        self._identifier = identifier
        self._dimension = dimension
        self.calls: list[str] = []

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        normalized = self.validate_texts(texts)
        self.calls.extend(normalized)
        return [self._vector(text) for text in normalized]

    def _vector(self, text: str) -> list[float]:
        lower = text.casefold()
        values = [
            1.0 if "agent" in lower or "llm" in lower else 0.0,
            1.0 if "java" in lower else 0.0,
            1.0 if "memory" in lower or "backend" in lower else 0.0,
            1.0 if "file" in lower else 0.0,
        ]
        return values[: self._dimension]


def make_memory(
    *,
    memory_id: str | None = None,
    content: str = "Agent memory systems need durable retrieval.",
    scope: str = "career/agent-dev",
    tags: list[str] | None = None,
) -> MemoryItem:
    """Create a semantic memory for vector backend tests."""

    values = {
        "type": "semantic",
        "scope": scope,
        "content": content,
        "summary": content.split(".")[0],
        "source": MemorySource(event_id="evt_vector", source_type="test"),
        "features": PolicyFeatures(importance=0.8, confidence=0.9),
        "metadata": {"attributes": {"tags": tags or ["agent", "memory"]}},
    }
    if memory_id is not None:
        values["id"] = memory_id
    return MemoryItem.model_validate(values)


def read_jsonl(path) -> list[dict]:
    """Read a JSONL file into dictionaries for assertions."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_add_get_and_reopen_vector_store(tmp_path) -> None:
    embedder = KeywordEmbedder()
    backend = VectorMemoryBackend(tmp_path / "workspace", embedder=embedder, use_cache=False)
    item = make_memory()

    stored = backend.add(item, reason="Persist vector memory.")

    assert stored.id == item.id
    assert stored.content_embedding == [1.0, 0.0, 1.0, 0.0]
    assert backend.get(item.id) == stored
    assert read_jsonl(backend.operation_log_path)[0]["op"] == "ADD"

    reopened = VectorMemoryBackend(
        tmp_path / "workspace",
        embedder=KeywordEmbedder(),
        use_cache=False,
    )
    assert reopened.get(item.id) == stored


def test_add_rejects_duplicates_and_unsafe_ids(tmp_path) -> None:
    backend = VectorMemoryBackend(
        tmp_path / "workspace",
        embedder=KeywordEmbedder(),
        use_cache=False,
    )
    item = make_memory()
    backend.add(item)

    with pytest.raises(MemoryAlreadyExistsError):
        backend.add(item)
    with pytest.raises(InvalidMemoryIdError):
        backend.add(make_memory(memory_id="../outside"))


def test_cosine_search_ranks_results_and_logs_retrieval(tmp_path) -> None:
    backend = VectorMemoryBackend(
        tmp_path / "workspace",
        embedder=KeywordEmbedder(),
        use_cache=False,
    )
    agent = backend.add(make_memory(content="Agent memory retrieval for LLM apps."))
    backend.add(
        make_memory(
            content="Java services use typed APIs.",
            scope="career/java",
            tags=["java"],
        )
    )
    file_backend = backend.add(
        make_memory(
            content="File backend stores memory records as Markdown.",
            scope="project/vmp-memos",
            tags=["project"],
        )
    )

    results = backend.search("agent memory", top_k=3)

    assert results[:2] == [agent, file_backend]
    retrievals = read_jsonl(backend.retrieval_log_path)
    operations = read_jsonl(backend.operation_log_path)
    assert retrievals[-1]["memory_ids"][:2] == [agent.id, file_backend.id]
    assert retrievals[-1]["metadata"]["retrieval_method"] == "cosine"
    assert operations[-1]["op"] == "RETRIEVE"


def test_update_retains_version_and_refreshes_embedding(tmp_path) -> None:
    backend = VectorMemoryBackend(
        tmp_path / "workspace",
        embedder=KeywordEmbedder(),
        use_cache=False,
    )
    original = backend.add(make_memory(content="Java backend work."))

    updated = backend.update(
        original.id,
        {
            "content": "Agent memory work.",
            "features": {"importance": 0.95},
            "metadata": {"attributes": {"verified": True}},
        },
        reason="New preference supersedes old one.",
    )

    assert updated.id == original.id
    assert updated.timestamp == original.timestamp
    assert updated.metadata.created_at == original.metadata.created_at
    assert updated.metadata.version == 2
    assert updated.content_embedding == [1.0, 0.0, 1.0, 0.0]
    assert updated.metadata.attributes == {"tags": ["agent", "memory"], "verified": True}
    assert updated.features.importance == 0.95

    with sqlite3.connect(backend.db_path) as connection:
        version_count = connection.execute(
            "SELECT COUNT(*) FROM memory_versions WHERE memory_id = ?",
            (original.id,),
        ).fetchone()[0]
    assert version_count == 1

    with pytest.raises(ValueError, match="immutable field"):
        backend.update(original.id, {"id": "mem_replaced"})
    with pytest.raises(ValueError, match="managed metadata"):
        backend.update(original.id, {"metadata": {"version": 99}})
    with pytest.raises(ValueError, match="cannot be empty"):
        backend.update(original.id, {})


def test_archive_and_filters(tmp_path) -> None:
    backend = VectorMemoryBackend(
        tmp_path / "workspace",
        embedder=KeywordEmbedder(),
        use_cache=False,
    )
    item = backend.add(make_memory(tags=["career", "agent"]))

    assert backend.list({"tags": ["career"]}) == [item]

    archived = backend.delete(item.id, reason="Test delete request.")

    assert archived.metadata.status == MemoryStatus.ARCHIVED
    assert backend.get(item.id) == archived
    assert backend.list() == []
    assert backend.list({"status": "archived"}) == [archived]
    assert "Physical deletion is disabled" in read_jsonl(backend.operation_log_path)[-1][
        "reason"
    ]

    with pytest.raises(MemoryNotFoundError):
        backend.update(item.id, {"content": "cannot update archived"})


def test_cosine_similarity_handles_zero_vectors_and_dimension_errors() -> None:
    assert cosine_similarity([1.0, 0.0], [[1.0, 0.0], [0.0, 0.0]]) == [1.0, 0.0]

    with pytest.raises(EmbeddingDimensionError, match="Expected embedding dimension"):
        cosine_similarity([1.0, 0.0], [[1.0]])


def test_existing_store_rejects_incompatible_embedding_namespace(tmp_path) -> None:
    backend = VectorMemoryBackend(
        tmp_path / "workspace",
        embedder=KeywordEmbedder(),
        use_cache=False,
    )
    backend.add(make_memory())

    with pytest.raises(VectorStoreError, match="namespace mismatch"):
        VectorMemoryBackend(
            tmp_path / "workspace",
            embedder=KeywordEmbedder(identifier="another-embedder"),
            use_cache=False,
        )


def test_existing_store_rejects_dimension_mismatch(tmp_path) -> None:
    backend = VectorMemoryBackend(
        tmp_path / "workspace",
        embedder=KeywordEmbedder(),
        use_cache=False,
    )
    backend.add(make_memory())

    with pytest.raises(VectorDimensionError):
        VectorMemoryBackend(
            tmp_path / "workspace",
            embedder=KeywordEmbedder(dimension=3),
            use_cache=False,
        )
