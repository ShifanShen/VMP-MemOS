"""Tests for the Markdown FileMemoryBackend."""

import json

import pytest

from vmp_memos.backends import (
    FileMemoryBackend,
    InvalidMemoryFileError,
    InvalidMemoryIdError,
    MemoryAlreadyExistsError,
    MemoryNotFoundError,
)
from vmp_memos.schemas import MemoryItem, MemorySource, MemoryStatus, PolicyFeatures


def make_memory(
    *,
    memory_id: str | None = None,
    content: str = "用户当前主攻 Agent 开发。",
    scope: str = "career/agent-dev",
) -> MemoryItem:
    """Create a representative semantic memory for backend tests."""

    values = {
        "type": "semantic",
        "scope": scope,
        "content": content,
        "summary": "用户职业方向偏 Agent 开发",
        "source": MemorySource(event_id="evt_test", source_type="test"),
        "content_embedding": [0.1, 0.2, 0.3],
        "policy_embedding": [0.8, 0.9],
        "features": PolicyFeatures(importance=0.88, confidence=0.92, novelty=0.9),
        "metadata": {"attributes": {"tags": ["career", "agent"]}},
    }
    if memory_id is not None:
        values["id"] = memory_id
    return MemoryItem.model_validate(values)


def read_jsonl(path) -> list[dict]:
    """Read a JSONL file into dictionaries for assertions."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_add_round_trips_markdown_and_updates_index(tmp_path) -> None:
    backend = FileMemoryBackend(tmp_path / "workspace")
    item = make_memory()

    stored = backend.add(item, reason="High-value preference.")

    markdown_path = backend.memories_dir / f"{item.id}.md"
    markdown = markdown_path.read_text(encoding="utf-8")
    assert stored == item
    assert markdown.startswith("---\nschema_version: 1\n")
    assert markdown.endswith(f"\n{item.content}\n")
    assert backend.get(item.id) == item
    assert item.id in backend.index_path.read_text(encoding="utf-8")
    assert read_jsonl(backend.operation_log_path)[0]["op"] == "ADD"


def test_add_rejects_duplicates_and_unsafe_ids(tmp_path) -> None:
    backend = FileMemoryBackend(tmp_path / "workspace")
    item = make_memory()
    backend.add(item)

    with pytest.raises(MemoryAlreadyExistsError):
        backend.add(item)
    with pytest.raises(InvalidMemoryIdError):
        backend.add(make_memory(memory_id="../outside"))


def test_update_retains_version_and_protects_managed_fields(tmp_path) -> None:
    backend = FileMemoryBackend(tmp_path / "workspace")
    original = backend.add(make_memory())

    updated = backend.update(
        original.id,
        {
            "content": "用户现在主攻 Agent 和 LLM 应用开发。",
            "features": {"importance": 0.95},
            "metadata": {"attributes": {"verified": True}},
        },
        reason="New preference supersedes the old one.",
    )

    version_path = backend.versions_dir / original.id / "v000001.md"
    assert updated.id == original.id
    assert updated.timestamp == original.timestamp
    assert updated.metadata.created_at == original.metadata.created_at
    assert updated.metadata.version == 2
    assert updated.metadata.attributes == {"tags": ["career", "agent"], "verified": True}
    assert updated.features.importance == 0.95
    assert version_path.is_file()
    assert backend._read_item(version_path) == original
    assert [record["op"] for record in read_jsonl(backend.operation_log_path)] == [
        "ADD",
        "UPDATE",
    ]

    with pytest.raises(ValueError, match="immutable field"):
        backend.update(original.id, {"id": "mem_replaced"})
    with pytest.raises(ValueError, match="managed metadata"):
        backend.update(original.id, {"metadata": {"version": 99}})
    with pytest.raises(ValueError, match="cannot be empty"):
        backend.update(original.id, {})


def test_archive_and_delete_never_physically_delete_memory(tmp_path) -> None:
    backend = FileMemoryBackend(tmp_path / "workspace")
    item = backend.add(make_memory())

    archived = backend.delete(item.id, reason="Test delete request.")

    assert archived.metadata.status == MemoryStatus.ARCHIVED
    assert archived.metadata.version == 2
    assert not (backend.memories_dir / f"{item.id}.md").exists()
    assert (backend.archive_dir / f"{item.id}.md").is_file()
    assert backend.get(item.id) == archived
    assert backend.list() == []
    assert backend.list({"status": "archived"}) == [archived]
    assert read_jsonl(backend.operation_log_path)[-1]["op"] == "ARCHIVE"
    assert "Physical deletion is disabled" in read_jsonl(backend.operation_log_path)[-1][
        "reason"
    ]

    with pytest.raises(MemoryNotFoundError):
        backend.update(item.id, {"content": "cannot update archived"})


def test_lexical_search_applies_filters_and_logs_retrieval(tmp_path) -> None:
    backend = FileMemoryBackend(tmp_path / "workspace")
    career = backend.add(make_memory())
    backend.add(
        make_memory(
            content="项目正在编写文件记忆后端。",
            scope="project/vmp-memos",
        )
    )

    results = backend.search("Agent 开发", filters={"scope": "career/agent-dev"})

    assert results == [career]
    retrievals = read_jsonl(backend.retrieval_log_path)
    operations = read_jsonl(backend.operation_log_path)
    assert retrievals[-1]["memory_ids"] == [career.id]
    assert retrievals[-1]["metadata"]["retrieval_method"] == "lexical"
    assert operations[-1]["op"] == "RETRIEVE"


def test_invalid_frontmatter_is_reported_clearly(tmp_path) -> None:
    backend = FileMemoryBackend(tmp_path / "workspace")
    malformed = backend.memories_dir / "mem_broken.md"
    malformed.write_text("not frontmatter", encoding="utf-8")

    with pytest.raises(InvalidMemoryFileError, match="Missing YAML frontmatter"):
        backend.list()
