"""SQLite vector-memory backend with cached local text embeddings."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any, Final

from pydantic import ValidationError

from vmp_memos.backends.base import (
    BaseMemoryBackend,
    InvalidMemoryIdError,
    MemoryAlreadyExistsError,
    MemoryBackendError,
    MemoryNotFoundError,
)
from vmp_memos.embeddings import (
    BaseEmbedder,
    CachedEmbedder,
    EmbeddingDimensionError,
    SQLiteEmbeddingCache,
    SentenceTransformerEmbedder,
    validate_vector,
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
_VECTOR_SCHEMA_VERSION: Final = "1"
_DEFAULT_EMBEDDING_MODEL: Final = "sentence-transformers/all-MiniLM-L6-v2"


class VectorStoreError(MemoryBackendError):
    """Raised when a persisted vector store is invalid or incompatible."""


class VectorDimensionError(VectorStoreError):
    """Raised when stored and generated vectors do not share one dimension."""


def cosine_similarity(
    query_vector: Sequence[float],
    candidate_vectors: Sequence[Sequence[float]],
) -> list[float]:
    """Return raw cosine similarity scores in candidate order.

    Zero-norm query or candidate vectors are treated as similarity ``0.0``.
    Dimension mismatches are surfaced as embedding-dimension errors because they
    indicate that a backend is mixing incompatible embedding spaces.
    """

    query = validate_vector(query_vector)
    query_norm = math.sqrt(sum(value * value for value in query))
    if query_norm == 0.0:
        return [0.0 for _ in candidate_vectors]

    scores: list[float] = []
    for candidate_vector in candidate_vectors:
        candidate = validate_vector(candidate_vector, expected_dimension=len(query))
        candidate_norm = math.sqrt(sum(value * value for value in candidate))
        if candidate_norm == 0.0:
            scores.append(0.0)
            continue
        dot_product = sum(left * right for left, right in zip(query, candidate, strict=True))
        scores.append(dot_product / (query_norm * candidate_norm))
    return scores


class VectorMemoryBackend(BaseMemoryBackend):
    """Persist memories and embeddings in SQLite, then rank with cosine search."""

    backend_name: Final = "vector"

    def __init__(
        self,
        workspace: str | Path = "memory_workspace",
        *,
        db_path: str | Path | None = None,
        embedder: BaseEmbedder | None = None,
        cache_path: str | Path | None = None,
        use_cache: bool = True,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.vector_dir = self.workspace / "vector"
        self.cache_dir = self.workspace / "cache"
        self.logs_dir = self.workspace / "logs"
        self.db_path = Path(db_path).expanduser().resolve() if db_path else (
            self.vector_dir / "memories.sqlite3"
        )
        self.operation_log_path = self.logs_dir / "operations.jsonl"
        self.retrieval_log_path = self.logs_dir / "retrievals.jsonl"

        base_embedder = embedder or SentenceTransformerEmbedder(_DEFAULT_EMBEDDING_MODEL)
        if use_cache and not isinstance(base_embedder, CachedEmbedder):
            cache = SQLiteEmbeddingCache(cache_path or self.cache_dir / "embeddings.sqlite3")
            self.embedder: BaseEmbedder = CachedEmbedder(base_embedder, cache)
        else:
            self.embedder = base_embedder

        for directory in (self.workspace, self.vector_dir, self.cache_dir, self.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.operation_log_path.touch(exist_ok=True)
        self.retrieval_log_path.touch(exist_ok=True)
        self._initialize()

    def add(
        self,
        memory_item: MemoryItem,
        *,
        reason: str = "Added memory item.",
        policy_score: float | None = None,
        confidence: float | None = None,
    ) -> MemoryItem:
        """Persist a new active memory plus its content embedding."""

        self._validate_memory_id(memory_item.id)
        if memory_item.metadata.status != MemoryStatus.ACTIVE:
            raise ValueError("New memory items must have status='active'")

        vector = self._embedding_for_item(memory_item)
        stored_item = self._with_embedding(memory_item, vector)
        operation = self._make_operation(
            op=OperationType.ADD,
            item=stored_item,
            reason=reason,
            policy_score=policy_score,
            confidence=confidence,
        )

        with self._connect() as connection:
            self._ensure_dimension(connection, len(vector))
            if self._row_exists(connection, stored_item.id):
                raise MemoryAlreadyExistsError(f"Memory already exists: {stored_item.id}")
            self._insert_current(connection, stored_item, vector)
        operation.append_jsonl(self.operation_log_path)
        return stored_item

    def update(
        self,
        memory_id: str,
        patch: Mapping[str, Any],
        *,
        reason: str = "Updated memory item.",
        policy_score: float | None = None,
        confidence: float | None = None,
    ) -> MemoryItem:
        """Patch an active memory, retain its prior version, and refresh vectors."""

        current, current_vector = self._get_active_with_vector(memory_id)
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

        vector = self._embedding_for_update(updated, patch)
        stored_item = self._with_embedding(updated, vector)
        operation = self._make_operation(
            op=OperationType.UPDATE,
            item=stored_item,
            reason=reason,
            policy_score=policy_score,
            confidence=confidence,
            payload={"changed_fields": sorted(patch)},
        )

        with self._connect() as connection:
            self._ensure_dimension(connection, len(vector))
            self._retain_version(connection, current, current_vector)
            self._update_current(connection, stored_item, vector)
        operation.append_jsonl(self.operation_log_path)
        return stored_item

    def get(self, memory_id: str) -> MemoryItem:
        """Return an active or archived memory by ID."""

        item, _ = self._get_with_vector(memory_id)
        return item

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: Mapping[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Embed ``query`` and return top cosine-ranked active memories."""

        if not query.strip():
            raise ValueError("query cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        started_at = perf_counter()
        query_vector = self.embedder.embed_one(query)
        with self._connect() as connection:
            self._ensure_dimension(connection, len(query_vector))
            candidates = self._list_records(connection, filters)

        candidate_vectors = [vector for _, vector in candidates]
        raw_scores = cosine_similarity(query_vector, candidate_vectors)
        ranked = [
            (self._retrieval_score(score), item)
            for score, (item, _) in zip(raw_scores, candidates, strict=True)
            if self._retrieval_score(score) > 0.0
        ]
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
            metadata={
                "embedding_model": self.embedder.identifier,
                "retrieval_method": "cosine",
                "top_k": top_k,
            },
        )
        retrieval.append_jsonl(self.retrieval_log_path)

        scope = str(filters.get("scope", "global")) if filters else "global"
        MemoryOperation(
            op=OperationType.RETRIEVE,
            reason=f"Vector search returned {len(items)} memory item(s).",
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
        """List current memories matching backend-neutral filters."""

        with self._connect() as connection:
            records = self._list_records(connection, filters)
        return [item for item, _ in records]

    def archive(
        self,
        memory_id: str,
        *,
        reason: str = "Archived memory item.",
        policy_score: float | None = None,
        confidence: float | None = None,
    ) -> MemoryItem:
        """Mark an active memory archived while keeping it retrievable by ID."""

        try:
            current, current_vector = self._get_active_with_vector(memory_id)
        except MemoryNotFoundError:
            archived, _ = self._get_archived_with_vector(memory_id)
            return archived

        payload = current.model_dump(mode="python")
        metadata = dict(payload["metadata"])
        metadata["version"] = current.metadata.version + 1
        metadata["updated_at"] = utc_now()
        metadata["status"] = MemoryStatus.ARCHIVED
        payload["metadata"] = metadata
        archived = MemoryItem.model_validate(payload)
        archived = self._with_embedding(archived, current_vector)
        operation = self._make_operation(
            op=OperationType.ARCHIVE,
            item=archived,
            reason=reason,
            policy_score=policy_score,
            confidence=confidence,
        )

        with self._connect() as connection:
            self._retain_version(connection, current, current_vector)
            self._update_current(connection, archived, current_vector)
        operation.append_jsonl(self.operation_log_path)
        return archived

    def delete(self, memory_id: str, *, reason: str = "Delete requested.") -> MemoryItem:
        """Translate deletion into an auditable archive operation."""

        return self.archive(memory_id, reason=f"{reason} Physical deletion is disabled.")

    def persist(self) -> None:
        """Flush SQLite WAL pages when possible."""

        with self._connect() as connection:
            connection.execute("PRAGMA wal_checkpoint(FULL)")

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS vector_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    item_json TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    dimension INTEGER NOT NULL CHECK(dimension > 0),
                    status TEXT NOT NULL,
                    type TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    version INTEGER NOT NULL CHECK(version > 0),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memory_versions (
                    memory_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    item_json TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    dimension INTEGER NOT NULL CHECK(dimension > 0),
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(memory_id, version)
                );
                CREATE INDEX IF NOT EXISTS idx_memories_status
                    ON memories(status);
                CREATE INDEX IF NOT EXISTS idx_memories_scope
                    ON memories(scope);
                CREATE INDEX IF NOT EXISTS idx_memories_type
                    ON memories(type);
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO vector_settings(key, value) VALUES (?, ?)",
                ("schema_version", _VECTOR_SCHEMA_VERSION),
            )
            self._ensure_embedding_namespace(connection)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _ensure_embedding_namespace(self, connection: sqlite3.Connection) -> None:
        existing = self._get_setting(connection, "embedding_namespace")
        if existing is None:
            self._set_setting(connection, "embedding_namespace", self.embedder.identifier)
            return
        if existing != self.embedder.identifier:
            raise VectorStoreError(
                "Vector store embedding namespace mismatch: "
                f"stored={existing!r}, current={self.embedder.identifier!r}"
            )

        stored_dimension = self._get_setting(connection, "embedding_dimension")
        if stored_dimension is not None and self.embedder.dimension is not None:
            if int(stored_dimension) != self.embedder.dimension:
                raise VectorDimensionError(
                    "Vector store dimension mismatch: "
                    f"stored={stored_dimension}, current={self.embedder.dimension}"
                )

    def _ensure_dimension(self, connection: sqlite3.Connection, dimension: int) -> None:
        if self.embedder.dimension is not None and self.embedder.dimension != dimension:
            raise VectorDimensionError(
                f"Embedder reports dimension {self.embedder.dimension}, got {dimension}"
            )
        stored_dimension = self._get_setting(connection, "embedding_dimension")
        if stored_dimension is None:
            self._set_setting(connection, "embedding_dimension", str(dimension))
            return
        if int(stored_dimension) != dimension:
            raise VectorDimensionError(
                f"Expected vector dimension {stored_dimension}, got {dimension}"
            )

    @staticmethod
    def _get_setting(connection: sqlite3.Connection, key: str) -> str | None:
        row = connection.execute(
            "SELECT value FROM vector_settings WHERE key = ?",
            (key,),
        ).fetchone()
        return None if row is None else str(row["value"])

    @staticmethod
    def _set_setting(connection: sqlite3.Connection, key: str, value: str) -> None:
        connection.execute(
            """
            INSERT INTO vector_settings(key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )

    def _embedding_for_item(self, item: MemoryItem) -> list[float]:
        if item.content_embedding:
            return validate_vector(item.content_embedding)
        return self.embedder.embed_one(item.content)

    def _embedding_for_update(
        self,
        item: MemoryItem,
        patch: Mapping[str, Any],
    ) -> list[float]:
        content_changed = "content" in patch
        embedding_changed = "content_embedding" in patch
        if item.content_embedding and (not content_changed or embedding_changed):
            return validate_vector(item.content_embedding)
        return self.embedder.embed_one(item.content)

    @staticmethod
    def _with_embedding(item: MemoryItem, vector: Sequence[float]) -> MemoryItem:
        payload = item.model_dump(mode="python")
        payload["content_embedding"] = validate_vector(vector)
        return MemoryItem.model_validate(payload)

    def _get_with_vector(self, memory_id: str) -> tuple[MemoryItem, list[float]]:
        self._validate_memory_id(memory_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
        if row is None:
            raise MemoryNotFoundError(f"Memory not found: {memory_id}")
        return self._record_from_row(row)

    def _get_active_with_vector(self, memory_id: str) -> tuple[MemoryItem, list[float]]:
        self._validate_memory_id(memory_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM memories WHERE id = ? AND status = ?",
                (memory_id, MemoryStatus.ACTIVE.value),
            ).fetchone()
            archived_exists = connection.execute(
                "SELECT 1 FROM memories WHERE id = ? AND status = ?",
                (memory_id, MemoryStatus.ARCHIVED.value),
            ).fetchone()
        if row is None:
            if archived_exists is not None:
                raise MemoryNotFoundError(f"Memory is archived and cannot be updated: {memory_id}")
            raise MemoryNotFoundError(f"Memory not found: {memory_id}")
        return self._record_from_row(row)

    def _get_archived_with_vector(self, memory_id: str) -> tuple[MemoryItem, list[float]]:
        self._validate_memory_id(memory_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM memories WHERE id = ? AND status = ?",
                (memory_id, MemoryStatus.ARCHIVED.value),
            ).fetchone()
        if row is None:
            raise MemoryNotFoundError(f"Memory not found: {memory_id}")
        return self._record_from_row(row)

    def _list_records(
        self,
        connection: sqlite3.Connection,
        filters: Mapping[str, Any] | None = None,
    ) -> list[tuple[MemoryItem, list[float]]]:
        criteria = dict(filters or {})
        unknown_filters = set(criteria) - _SUPPORTED_FILTERS
        if unknown_filters:
            names = ", ".join(sorted(unknown_filters))
            raise ValueError(f"Unsupported vector-backend filter(s): {names}")

        include_archived = bool(criteria.pop("include_archived", False))
        requested_status = criteria.get("status")
        if isinstance(requested_status, MemoryStatus):
            requested_status = requested_status.value

        rows = connection.execute("SELECT * FROM memories ORDER BY id").fetchall()
        records: list[tuple[MemoryItem, list[float]]] = []
        for row in rows:
            item, vector = self._record_from_row(row)
            if not include_archived and requested_status is None:
                if item.metadata.status != MemoryStatus.ACTIVE:
                    continue
            if self._matches(item, criteria):
                records.append((item, vector))
        return records

    def _record_from_row(self, row: sqlite3.Row) -> tuple[MemoryItem, list[float]]:
        try:
            item = MemoryItem.model_validate_json(row["item_json"])
        except ValidationError as exc:
            raise VectorStoreError(f"Invalid memory JSON in vector store: {exc}") from exc
        if row["id"] != item.id:
            raise VectorStoreError(
                f"Vector row ID {row['id']!r} does not match memory ID {item.id!r}"
            )
        vector = self._vector_from_json(row["vector_json"], int(row["dimension"]))
        if item.content_embedding != vector:
            item = self._with_embedding(item, vector)
        return item, vector

    def _insert_current(
        self,
        connection: sqlite3.Connection,
        item: MemoryItem,
        vector: Sequence[float],
    ) -> None:
        vector_json, dimension = self._vector_to_json(vector)
        connection.execute(
            """
            INSERT INTO memories (
                id, item_json, vector_json, dimension, status, type, scope,
                version, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            self._current_row_values(item, vector_json, dimension),
        )

    def _update_current(
        self,
        connection: sqlite3.Connection,
        item: MemoryItem,
        vector: Sequence[float],
    ) -> None:
        vector_json, dimension = self._vector_to_json(vector)
        connection.execute(
            """
            UPDATE memories
            SET item_json = ?,
                vector_json = ?,
                dimension = ?,
                status = ?,
                type = ?,
                scope = ?,
                version = ?,
                created_at = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                item.model_dump_json(),
                vector_json,
                dimension,
                item.metadata.status.value,
                item.type.value,
                item.scope,
                item.metadata.version,
                item.metadata.created_at.isoformat(),
                item.metadata.updated_at.isoformat(),
                item.id,
            ),
        )

    def _retain_version(
        self,
        connection: sqlite3.Connection,
        item: MemoryItem,
        vector: Sequence[float],
    ) -> None:
        vector_json, dimension = self._vector_to_json(vector)
        connection.execute(
            """
            INSERT OR IGNORE INTO memory_versions (
                memory_id, version, item_json, vector_json, dimension, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                item.id,
                item.metadata.version,
                item.model_dump_json(),
                vector_json,
                dimension,
                utc_now().isoformat(),
            ),
        )

    @staticmethod
    def _current_row_values(
        item: MemoryItem,
        vector_json: str,
        dimension: int,
    ) -> tuple[str, str, str, int, str, str, str, int, str, str]:
        return (
            item.id,
            item.model_dump_json(),
            vector_json,
            dimension,
            item.metadata.status.value,
            item.type.value,
            item.scope,
            item.metadata.version,
            item.metadata.created_at.isoformat(),
            item.metadata.updated_at.isoformat(),
        )

    @staticmethod
    def _vector_to_json(vector: Sequence[float]) -> tuple[str, int]:
        normalized = validate_vector(vector)
        return json.dumps(normalized, separators=(",", ":")), len(normalized)

    @staticmethod
    def _vector_from_json(vector_json: str, dimension: int) -> list[float]:
        try:
            raw_vector = json.loads(vector_json)
        except json.JSONDecodeError as exc:
            raise VectorStoreError("Invalid vector JSON in vector store") from exc
        if not isinstance(raw_vector, list):
            raise VectorStoreError("Stored vector must be a JSON list")
        try:
            return validate_vector(raw_vector, expected_dimension=dimension)
        except EmbeddingDimensionError as exc:
            raise VectorDimensionError(str(exc)) from exc

    @staticmethod
    def _row_exists(connection: sqlite3.Connection, memory_id: str) -> bool:
        row = connection.execute(
            "SELECT 1 FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()
        return row is not None

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
    def _retrieval_score(raw_cosine: float) -> float:
        if not math.isfinite(raw_cosine):
            return 0.0
        return max(0.0, min(1.0, raw_cosine))
