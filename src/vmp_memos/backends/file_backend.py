"""Human-readable Markdown implementation of the memory backend contract."""

from __future__ import annotations

import re
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any, Final

import yaml
from pydantic import ValidationError

from vmp_memos.backends.base import (
    BaseMemoryBackend,
    InvalidMemoryFileError,
    InvalidMemoryIdError,
    MemoryAlreadyExistsError,
    MemoryNotFoundError,
)
from vmp_memos.schemas import (
    MemoryItem,
    MemoryOperation,
    MemoryStatus,
    MemoryType,
    OperationType,
    RetrievalResult,
)
from vmp_memos.schemas.base import utc_now

_SAFE_MEMORY_ID: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SUPPORTED_FILTERS: Final = {"include_archived", "scope", "status", "tags", "type"}
_PROTECTED_UPDATE_FIELDS: Final = {"id", "timestamp"}
_PROTECTED_METADATA_FIELDS: Final = {"created_at", "status", "updated_at", "version"}


class FileMemoryBackend(BaseMemoryBackend):
    """Store each memory as Markdown plus YAML frontmatter.

    Active items live under ``memories/``. Archived items move to ``archive/``;
    previous revisions are retained under ``versions/<memory-id>/``. Every public
    mutation is immediately persisted and appended to ``logs/operations.jsonl``.
    """

    backend_name: Final = "file"

    def __init__(self, workspace: str | Path = "memory_workspace") -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.memories_dir = self.workspace / "memories"
        self.archive_dir = self.workspace / "archive"
        self.versions_dir = self.workspace / "versions"
        self.logs_dir = self.workspace / "logs"
        self.index_path = self.workspace / "INDEX.md"
        self.operation_log_path = self.logs_dir / "operations.jsonl"
        self.retrieval_log_path = self.logs_dir / "retrievals.jsonl"

        for directory in (
            self.workspace,
            self.memories_dir,
            self.archive_dir,
            self.versions_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.operation_log_path.touch(exist_ok=True)
        self.retrieval_log_path.touch(exist_ok=True)
        if not self.index_path.exists():
            self._write_index()

    def add(
        self,
        memory_item: MemoryItem,
        *,
        reason: str = "Added memory item.",
        policy_score: float | None = None,
        confidence: float | None = None,
    ) -> MemoryItem:
        """Write a new active memory and record an ``ADD`` operation."""

        self._validate_memory_id(memory_item.id)
        if memory_item.metadata.status != MemoryStatus.ACTIVE:
            raise ValueError("New memory items must have status='active'")
        already_active = self._active_path(memory_item.id).exists()
        already_archived = self._archive_path(memory_item.id).exists()
        if already_active or already_archived:
            raise MemoryAlreadyExistsError(f"Memory already exists: {memory_item.id}")

        operation = self._make_operation(
            op=OperationType.ADD,
            item=memory_item,
            reason=reason,
            policy_score=policy_score,
            confidence=confidence,
        )
        self._write_item(self._active_path(memory_item.id), memory_item)
        self._write_index()
        operation.append_jsonl(self.operation_log_path)
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
        """Validate a partial update and retain the prior Markdown revision."""

        current = self._get_active(memory_id)
        self._validate_patch(patch)
        merged = self._deep_merge(current.model_dump(mode="python"), patch)
        raw_metadata = merged.get("metadata")
        if not isinstance(raw_metadata, Mapping):
            raise ValueError("metadata must remain a mapping after an update")
        metadata = dict(raw_metadata)
        metadata["version"] = current.metadata.version + 1
        metadata["created_at"] = current.metadata.created_at
        metadata["updated_at"] = utc_now()
        metadata["status"] = MemoryStatus.ACTIVE
        merged["metadata"] = metadata

        try:
            updated = MemoryItem.model_validate(merged)
        except ValidationError as exc:
            raise ValueError(f"Invalid memory update for {memory_id}: {exc}") from exc

        operation = self._make_operation(
            op=OperationType.UPDATE,
            item=updated,
            reason=reason,
            policy_score=policy_score,
            confidence=confidence,
            payload={"changed_fields": sorted(patch)},
        )
        self._retain_version(current)
        self._write_item(self._active_path(memory_id), updated)
        self._write_index()
        operation.append_jsonl(self.operation_log_path)
        return updated

    def get(self, memory_id: str) -> MemoryItem:
        """Return an active memory, falling back to the archive."""

        self._validate_memory_id(memory_id)
        active_path = self._active_path(memory_id)
        if active_path.exists():
            return self._read_item(active_path)
        archive_path = self._archive_path(memory_id)
        if archive_path.exists():
            return self._read_item(archive_path)
        raise MemoryNotFoundError(f"Memory not found: {memory_id}")

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: Mapping[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Run deterministic lexical retrieval until vector search is introduced."""

        normalized_query = query.strip().casefold()
        if not normalized_query:
            raise ValueError("query cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        started_at = perf_counter()
        terms = re.findall(r"[\w-]+", normalized_query, flags=re.UNICODE)
        ranked: list[tuple[float, MemoryItem]] = []
        for item in self.list(filters):
            haystack = " ".join(
                value
                for value in (item.content, item.summary or "", item.scope)
                if value
            ).casefold()
            matched_terms = sum(term in haystack for term in terms)
            score = matched_terms / len(terms) if terms else 0.0
            if normalized_query in haystack:
                score = 1.0
            if score > 0.0:
                ranked.append((score, item))

        ranked.sort(key=lambda pair: (-pair[0], pair[1].id))
        selected = ranked[:top_k]
        items = [item for _, item in selected]
        scores = {item.id: score for score, item in selected}
        latency_ms = (perf_counter() - started_at) * 1000.0
        token_count = sum(max(1, len(item.content) // 4) for item in items)

        retrieval = RetrievalResult(
            query=query,
            memory_ids=[item.id for item in items],
            items=items,
            scores=scores,
            token_count=token_count,
            latency_ms=latency_ms,
            backend=self.backend_name,
            metadata={"retrieval_method": "lexical", "top_k": top_k},
        )
        retrieval.append_jsonl(self.retrieval_log_path)

        scope = str(filters.get("scope", "global")) if filters else "global"
        MemoryOperation(
            op=OperationType.RETRIEVE,
            reason=f"Lexical file search returned {len(items)} memory item(s).",
            policy_score=max(scores.values(), default=0.0),
            confidence=1.0,
            scope=scope,
            backend=self.backend_name,
            payload={
                "query": query,
                "result_ids": [item.id for item in items],
                "top_k": top_k,
            },
        ).append_jsonl(self.operation_log_path)
        return items

    def list(self, filters: Mapping[str, Any] | None = None) -> list[MemoryItem]:
        """List active memories, optionally including or selecting archived items."""

        criteria = dict(filters or {})
        unknown_filters = set(criteria) - _SUPPORTED_FILTERS
        if unknown_filters:
            names = ", ".join(sorted(unknown_filters))
            raise ValueError(f"Unsupported file-backend filter(s): {names}")

        include_archived = bool(criteria.pop("include_archived", False))
        requested_status = criteria.get("status")
        if isinstance(requested_status, MemoryStatus):
            requested_status = requested_status.value
        include_archived = include_archived or requested_status == MemoryStatus.ARCHIVED.value

        items = self._read_directory(self.memories_dir)
        if include_archived:
            items.extend(self._read_directory(self.archive_dir))
        return sorted(
            (item for item in items if self._matches(item, criteria)),
            key=lambda item: item.id,
        )

    def archive(
        self,
        memory_id: str,
        *,
        reason: str = "Archived memory item.",
        policy_score: float | None = None,
        confidence: float | None = None,
    ) -> MemoryItem:
        """Move an active memory to ``archive/`` and preserve its prior version."""

        self._validate_memory_id(memory_id)
        active_path = self._active_path(memory_id)
        archive_path = self._archive_path(memory_id)
        if not active_path.exists():
            if archive_path.exists():
                return self._read_item(archive_path)
            raise MemoryNotFoundError(f"Memory not found: {memory_id}")

        current = self._read_item(active_path)
        payload = current.model_dump(mode="python")
        metadata = dict(payload["metadata"])
        metadata["version"] = current.metadata.version + 1
        metadata["updated_at"] = utc_now()
        metadata["status"] = MemoryStatus.ARCHIVED
        payload["metadata"] = metadata
        archived = MemoryItem.model_validate(payload)
        operation = self._make_operation(
            op=OperationType.ARCHIVE,
            item=archived,
            reason=reason,
            policy_score=policy_score,
            confidence=confidence,
        )

        self._retain_version(current)
        self._write_item(archive_path, archived)
        active_path.unlink()
        self._write_index()
        operation.append_jsonl(self.operation_log_path)
        return archived

    def delete(self, memory_id: str, *, reason: str = "Delete requested.") -> MemoryItem:
        """Translate deletion into an auditable archive operation in Phase 2."""

        return self.archive(memory_id, reason=f"{reason} Physical deletion is disabled.")

    def persist(self) -> None:
        """Refresh the generated index; item writes are already immediate."""

        self._write_index()

    def _get_active(self, memory_id: str) -> MemoryItem:
        self._validate_memory_id(memory_id)
        path = self._active_path(memory_id)
        if not path.exists():
            if self._archive_path(memory_id).exists():
                raise MemoryNotFoundError(f"Memory is archived and cannot be updated: {memory_id}")
            raise MemoryNotFoundError(f"Memory not found: {memory_id}")
        return self._read_item(path)

    def _active_path(self, memory_id: str) -> Path:
        self._validate_memory_id(memory_id)
        return self.memories_dir / f"{memory_id}.md"

    def _archive_path(self, memory_id: str) -> Path:
        self._validate_memory_id(memory_id)
        return self.archive_dir / f"{memory_id}.md"

    @staticmethod
    def _validate_memory_id(memory_id: str) -> None:
        if not _SAFE_MEMORY_ID.fullmatch(memory_id):
            raise InvalidMemoryIdError(
                "Memory IDs may contain only letters, numbers, '.', '_' and '-'"
            )

    @staticmethod
    def _validate_patch(patch: Mapping[str, Any]) -> None:
        if not patch:
            raise ValueError("update patch cannot be empty")
        protected = set(patch) & _PROTECTED_UPDATE_FIELDS
        if protected:
            names = ", ".join(sorted(protected))
            raise ValueError(f"Cannot update immutable field(s): {names}")
        metadata_patch = patch.get("metadata")
        if isinstance(metadata_patch, Mapping):
            protected_metadata = set(metadata_patch) & _PROTECTED_METADATA_FIELDS
            if protected_metadata:
                names = ", ".join(sorted(protected_metadata))
                raise ValueError(f"Cannot directly update managed metadata field(s): {names}")

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

    def _retain_version(self, item: MemoryItem) -> None:
        version_dir = self.versions_dir / item.id
        version_path = version_dir / f"v{item.metadata.version:06d}.md"
        if not version_path.exists():
            self._write_item(version_path, item)

    def _read_directory(self, directory: Path) -> list[MemoryItem]:
        return [self._read_item(path) for path in sorted(directory.glob("*.md"))]

    def _read_item(self, path: Path) -> MemoryItem:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise InvalidMemoryFileError(f"Cannot read memory file {path}: {exc}") from exc

        lines = text.splitlines(keepends=True)
        if not lines or lines[0].strip() != "---":
            raise InvalidMemoryFileError(f"Missing YAML frontmatter in {path}")
        closing_index = next(
            (index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"),
            None,
        )
        if closing_index is None:
            raise InvalidMemoryFileError(f"Unclosed YAML frontmatter in {path}")

        try:
            frontmatter = yaml.safe_load("".join(lines[1:closing_index]))
        except yaml.YAMLError as exc:
            raise InvalidMemoryFileError(f"Invalid YAML frontmatter in {path}: {exc}") from exc
        if not isinstance(frontmatter, dict):
            raise InvalidMemoryFileError(f"Frontmatter must be a mapping in {path}")

        payload = dict(frontmatter)
        payload.pop("schema_version", None)
        body = "".join(lines[closing_index + 1 :])
        if body.startswith("\n") or body.startswith("\r\n"):
            body = body.split("\n", maxsplit=1)[1]
        payload["content"] = body.rstrip("\r\n")
        try:
            item = MemoryItem.model_validate(payload)
        except ValidationError as exc:
            raise InvalidMemoryFileError(f"Invalid memory schema in {path}: {exc}") from exc
        if path.parent in {self.memories_dir, self.archive_dir} and path.stem != item.id:
            raise InvalidMemoryFileError(
                f"Memory filename '{path.stem}' does not match frontmatter ID '{item.id}'"
            )
        return item

    def _write_item(self, path: Path, item: MemoryItem) -> None:
        payload = item.model_dump(mode="json")
        content = payload.pop("content")
        frontmatter = {"schema_version": 1, **payload}
        yaml_text = yaml.safe_dump(
            frontmatter,
            allow_unicode=True,
            sort_keys=False,
            width=100,
        )
        document = f"---\n{yaml_text}---\n\n{content}\n"
        self._atomic_write(path, document)

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        try:
            temporary.write_text(content, encoding="utf-8", newline="\n")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def _make_operation(
        self,
        *,
        op: OperationType,
        item: MemoryItem,
        reason: str,
        policy_score: float | None,
        confidence: float | None,
        payload: dict[str, Any] | None = None,
    ) -> MemoryOperation:
        return MemoryOperation(
            op=op,
            target_memory_id=item.id,
            source_event_id=item.source.event_id,
            reason=reason,
            policy_score=item.features.importance if policy_score is None else policy_score,
            confidence=item.features.confidence if confidence is None else confidence,
            scope=item.scope,
            backend=self.backend_name,
            payload=payload or {},
        )

    @staticmethod
    def _matches(item: MemoryItem, criteria: Mapping[str, Any]) -> bool:
        for key, expected in criteria.items():
            if key == "tags":
                raw_tags = item.metadata.attributes.get("tags", [])
                actual_tags = raw_tags if isinstance(raw_tags, list) else []
                if isinstance(expected, str):
                    expected_tags = [expected]
                elif isinstance(expected, (list, set, tuple)):
                    expected_tags = list(expected)
                else:
                    raise ValueError("tags filter must be a string or a collection of strings")
                if not all(tag in actual_tags for tag in expected_tags):
                    return False
                continue
            actual = item.metadata.status if key == "status" else getattr(item, key)
            actual_value = (
                actual.value if isinstance(actual, (MemoryStatus, MemoryType)) else actual
            )
            expected_value = (
                expected.value
                if isinstance(expected, (MemoryStatus, MemoryType))
                else expected
            )
            if actual_value != expected_value:
                return False
        return True

    def _write_index(self) -> None:
        items = self.list({"include_archived": True})
        active = [item for item in items if item.metadata.status == MemoryStatus.ACTIVE]
        archived = [item for item in items if item.metadata.status == MemoryStatus.ARCHIVED]
        lines = [
            "# Memory Workspace Index",
            "",
            "> Automatically generated by FileMemoryBackend. Edit memory Markdown files instead.",
            "",
            f"## Active Memories ({len(active)})",
            "",
        ]
        lines.extend(self._index_entries(active, archived=False) or ["_No active memories._"])
        lines.extend(["", f"## Archived Memories ({len(archived)})", ""])
        lines.extend(self._index_entries(archived, archived=True) or ["_No archived memories._"])
        lines.append("")
        self._atomic_write(self.index_path, "\n".join(lines))

    def _index_entries(self, items: list[MemoryItem], *, archived: bool) -> list[str]:
        directory = self.archive_dir if archived else self.memories_dir
        entries: list[str] = []
        for item in items:
            raw_label = item.summary or item.content.splitlines()[0]
            label = " ".join(raw_label.split()).replace("[", "\\[").replace("]", "\\]")
            relative_path = (directory / f"{item.id}.md").relative_to(self.workspace).as_posix()
            entries.append(
                f"- [{label}]({relative_path}) — `{item.type.value}` · `{item.scope}` · "
                f"v{item.metadata.version}"
            )
        return entries
