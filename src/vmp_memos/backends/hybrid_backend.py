"""Hybrid backend combining readable Markdown storage with vector retrieval."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any, Final

from vmp_memos.backends.base import (
    BaseMemoryBackend,
    MemoryAlreadyExistsError,
    MemoryBackendError,
    MemoryNotFoundError,
)
from vmp_memos.embeddings import BaseEmbedder
from vmp_memos.schemas import (
    MemoryItem,
    MemoryOperation,
    OperationType,
    RetrievalResult,
)


class HybridBackendError(MemoryBackendError):
    """Raised when the file and vector components cannot be kept in sync."""


class HybridMemoryBackend(BaseMemoryBackend):
    """Use file storage as source of truth and vector storage as retrieval index."""

    backend_name: Final = "hybrid"

    def __init__(
        self,
        workspace: str | Path = "memory_workspace",
        *,
        file_backend: BaseMemoryBackend | None = None,
        vector_backend: BaseMemoryBackend | None = None,
        embedder: BaseEmbedder | None = None,
        use_cache: bool = True,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.file_backend = file_backend or self._create_file_backend(self.workspace)
        self.vector_backend = vector_backend or self._create_vector_backend(
            self.workspace,
            embedder=embedder,
            use_cache=use_cache,
        )
        self.logs_dir = self.workspace / "logs"
        self.operation_log_path = self.logs_dir / "operations.jsonl"
        self.retrieval_log_path = self.logs_dir / "retrievals.jsonl"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.operation_log_path.touch(exist_ok=True)
        self.retrieval_log_path.touch(exist_ok=True)

    def add(
        self,
        memory_item: MemoryItem,
        *,
        reason: str = "Added memory item.",
        policy_score: float | None = None,
        confidence: float | None = None,
    ) -> MemoryItem:
        """Add memory to both vector index and readable file storage."""

        self._ensure_absent(memory_item.id)
        try:
            indexed = self.vector_backend.add(
                memory_item,
                reason=reason,
                policy_score=policy_score,
                confidence=confidence,
            )
        except Exception as exc:
            raise HybridBackendError(f"Vector add failed for {memory_item.id}: {exc}") from exc
        try:
            return self.file_backend.add(
                indexed,
                reason=reason,
                policy_score=policy_score,
                confidence=confidence,
            )
        except Exception as exc:
            self._best_effort_archive_index(indexed.id, reason="File add failed after vector add.")
            raise HybridBackendError(f"File add failed for {memory_item.id}: {exc}") from exc

    def update(
        self,
        memory_id: str,
        patch: Mapping[str, Any],
        *,
        reason: str = "Updated memory item.",
        policy_score: float | None = None,
        confidence: float | None = None,
    ) -> MemoryItem:
        """Update source-of-truth file storage and refresh the vector index."""

        try:
            updated_index = self.vector_backend.update(
                memory_id,
                patch,
                reason=reason,
                policy_score=policy_score,
                confidence=confidence,
            )
        except MemoryNotFoundError:
            updated_file = self.file_backend.update(
                memory_id,
                patch,
                reason=reason,
                policy_score=policy_score,
                confidence=confidence,
            )
            self.vector_backend.add(
                updated_file,
                reason=f"Re-index after missing vector entry. {reason}",
                policy_score=policy_score,
                confidence=confidence,
            )
            return updated_file
        except Exception as exc:
            raise HybridBackendError(f"Vector update failed for {memory_id}: {exc}") from exc

        file_patch = dict(patch)
        if updated_index.content_embedding:
            file_patch["content_embedding"] = updated_index.content_embedding
        return self.file_backend.update(
            memory_id,
            file_patch,
            reason=reason,
            policy_score=policy_score,
            confidence=confidence,
        )

    def get(self, memory_id: str) -> MemoryItem:
        """Return source-of-truth memory from the file component."""

        return self.file_backend.get(memory_id)

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: Mapping[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Rank with vector backend, hydrate results from file storage, and log hybrid retrieval."""

        if not query.strip():
            raise ValueError("query cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        started_at = perf_counter()
        indexed_results = self.vector_backend.search(query, top_k=top_k, filters=filters)
        hydrated: list[MemoryItem] = []
        missing_ids: list[str] = []
        for indexed in indexed_results:
            try:
                hydrated.append(self.file_backend.get(indexed.id))
            except MemoryNotFoundError:
                missing_ids.append(indexed.id)

        scores = {
            item.id: max(0.0, 1.0 - rank / max(1, len(hydrated)))
            for rank, item in enumerate(hydrated)
        }
        latency_ms = (perf_counter() - started_at) * 1000.0
        token_count = sum(max(1, len(item.content) // 4) for item in hydrated)
        RetrievalResult(
            query=query,
            memory_ids=[item.id for item in hydrated],
            items=hydrated,
            scores=scores,
            token_count=token_count,
            latency_ms=latency_ms,
            backend=self.backend_name,
            metadata={
                "retrieval_method": "hybrid_vector_hydrate",
                "top_k": top_k,
                "missing_file_ids": missing_ids,
            },
        ).append_jsonl(self.retrieval_log_path)
        MemoryOperation(
            op=OperationType.RETRIEVE,
            reason=f"Hybrid search returned {len(hydrated)} hydrated memory item(s).",
            policy_score=max(scores.values(), default=0.0),
            confidence=1.0,
            scope=_scope_from_filters(filters),
            backend=self.backend_name,
            payload={
                "query": query,
                "result_ids": [item.id for item in hydrated],
                "missing_file_ids": missing_ids,
                "top_k": top_k,
            },
        ).append_jsonl(self.operation_log_path)
        return hydrated

    def list(self, filters: Mapping[str, Any] | None = None) -> list[MemoryItem]:
        """List source-of-truth file memories."""

        return self.file_backend.list(filters)

    def archive(
        self,
        memory_id: str,
        *,
        reason: str = "Archived memory item.",
        policy_score: float | None = None,
        confidence: float | None = None,
    ) -> MemoryItem:
        """Archive memory in both components, returning the file copy."""

        archived = self.file_backend.archive(
            memory_id,
            reason=reason,
            policy_score=policy_score,
            confidence=confidence,
        )
        try:
            self.vector_backend.archive(
                memory_id,
                reason=reason,
                policy_score=policy_score,
                confidence=confidence,
            )
        except MemoryNotFoundError:
            pass
        except Exception as exc:
            raise HybridBackendError(f"Vector archive failed for {memory_id}: {exc}") from exc
        return archived

    def delete(self, memory_id: str, *, reason: str = "Delete requested.") -> MemoryItem:
        """Translate deletion into archive, preserving the no-physical-delete rule."""

        return self.archive(memory_id, reason=f"{reason} Physical deletion is disabled.")

    def persist(self) -> None:
        """Flush both backend components."""

        self.file_backend.persist()
        self.vector_backend.persist()

    def reindex(self) -> int:
        """Ensure active file memories exist in the vector component.

        This is useful after manual Markdown edits or after importing a workspace.
        Existing vector items are updated, missing vector items are added.
        """

        count = 0
        for item in self.file_backend.list():
            try:
                self.vector_backend.update(
                    item.id,
                    _index_patch(item),
                    reason="Hybrid reindex from file source of truth.",
                    policy_score=item.features.importance,
                    confidence=item.features.confidence,
                )
            except MemoryNotFoundError:
                self.vector_backend.add(
                    item,
                    reason="Hybrid reindex from file source of truth.",
                    policy_score=item.features.importance,
                    confidence=item.features.confidence,
                )
            count += 1
        return count

    @staticmethod
    def _create_file_backend(workspace: Path) -> BaseMemoryBackend:
        from vmp_memos.backends.file_backend import FileMemoryBackend

        return FileMemoryBackend(workspace)

    @staticmethod
    def _create_vector_backend(
        workspace: Path,
        *,
        embedder: BaseEmbedder | None,
        use_cache: bool,
    ) -> BaseMemoryBackend:
        from vmp_memos.backends.vector_backend import VectorMemoryBackend

        return VectorMemoryBackend(workspace, embedder=embedder, use_cache=use_cache)

    def _ensure_absent(self, memory_id: str) -> None:
        exists_in: list[str] = []
        for label, backend in (
            ("file", self.file_backend),
            ("vector", self.vector_backend),
        ):
            if _exists(backend, memory_id):
                exists_in.append(label)
        if exists_in:
            locations = ", ".join(exists_in)
            raise MemoryAlreadyExistsError(
                f"Memory already exists in hybrid component(s): {locations}"
            )

    def _best_effort_archive_index(self, memory_id: str, *, reason: str) -> None:
        try:
            self.vector_backend.archive(memory_id, reason=reason)
        except Exception:
            return


def _exists(backend: BaseMemoryBackend, memory_id: str) -> bool:
    try:
        backend.get(memory_id)
    except MemoryNotFoundError:
        return False
    return True


def _scope_from_filters(filters: Mapping[str, Any] | None) -> str:
    if not filters:
        return "global"
    raw_scope = filters.get("scope")
    return raw_scope if isinstance(raw_scope, str) and raw_scope else "global"


def _index_patch(item: MemoryItem) -> dict[str, Any]:
    metadata_patch: dict[str, Any] = {
        "access_count": item.metadata.access_count,
        "last_accessed_at": item.metadata.last_accessed_at,
        "attributes": item.metadata.attributes,
    }
    return {
        "type": item.type,
        "scope": item.scope,
        "content": item.content,
        "summary": item.summary,
        "source": item.source.model_dump(mode="python"),
        "content_embedding": item.content_embedding,
        "policy_embedding": item.policy_embedding,
        "features": item.features.model_dump(mode="python"),
        "metadata": metadata_patch,
        "links": item.links.model_dump(mode="python"),
    }
