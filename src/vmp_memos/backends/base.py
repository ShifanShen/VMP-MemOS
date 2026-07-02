"""Backend-neutral memory persistence contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from vmp_memos.schemas import MemoryItem


class MemoryBackendError(RuntimeError):
    """Base exception for persistence and backend contract failures."""


class MemoryNotFoundError(MemoryBackendError):
    """Raised when a requested memory ID does not exist."""


class MemoryAlreadyExistsError(MemoryBackendError):
    """Raised when adding a memory whose ID already exists."""


class InvalidMemoryIdError(MemoryBackendError):
    """Raised when an ID cannot safely be mapped to backend storage."""


class InvalidMemoryFileError(MemoryBackendError):
    """Raised when a persisted memory file cannot be parsed or validated."""


class BaseMemoryBackend(ABC):
    """Common CRUD/search interface implemented by every memory backend."""

    @abstractmethod
    def add(
        self,
        memory_item: MemoryItem,
        *,
        reason: str = "Added memory item.",
        policy_score: float | None = None,
        confidence: float | None = None,
    ) -> MemoryItem:
        """Persist a new memory and log the operation."""

    @abstractmethod
    def update(
        self,
        memory_id: str,
        patch: Mapping[str, Any],
        *,
        reason: str = "Updated memory item.",
        policy_score: float | None = None,
        confidence: float | None = None,
    ) -> MemoryItem:
        """Apply a validated patch while retaining the previous version."""

    @abstractmethod
    def get(self, memory_id: str) -> MemoryItem:
        """Load an active or archived memory by ID."""

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: Mapping[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Return backend-ranked memories and record the retrieval."""

    @abstractmethod
    def list(self, filters: Mapping[str, Any] | None = None) -> list[MemoryItem]:
        """List memories matching backend-neutral filters."""

    @abstractmethod
    def archive(
        self,
        memory_id: str,
        *,
        reason: str = "Archived memory item.",
        policy_score: float | None = None,
        confidence: float | None = None,
    ) -> MemoryItem:
        """Move a memory to archived lifecycle state without deleting it."""

    @abstractmethod
    def delete(self, memory_id: str, *, reason: str = "Delete requested.") -> MemoryItem:
        """Handle a delete request; Phase 2 backends must archive instead."""

    @abstractmethod
    def persist(self) -> None:
        """Flush any backend metadata or indexes."""

