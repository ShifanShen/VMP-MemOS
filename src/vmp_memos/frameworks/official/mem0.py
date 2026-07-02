"""Official Mem0 OSS adapter using local vLLM and Hugging Face embeddings."""

from __future__ import annotations

import gc
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from vmp_memos.frameworks.base import (
    BaseMemoryFrameworkAdapter,
    FairnessLevel,
    RetrievedMemory,
)
from vmp_memos.frameworks.runtime import FrameworkRuntimeConfig
from vmp_memos.frameworks.text import estimate_tokens
from vmp_memos.schemas import Event

MemoryFactory = Callable[[dict[str, Any]], Any]


class Mem0DependencyError(RuntimeError):
    """Raised when the pinned official Mem0 package is unavailable."""


class Mem0OfficialAdapter(BaseMemoryFrameworkAdapter):
    """Thin adapter around ``mem0.Memory`` without reimplementing its algorithm."""

    name = "mem0"
    fairness_level = FairnessLevel.FULLY_CONTROLLED

    def __init__(
        self,
        *,
        runtime: FrameworkRuntimeConfig | None = None,
        memory_factory: MemoryFactory | None = None,
    ) -> None:
        super().__init__()
        self.runtime = runtime or FrameworkRuntimeConfig.from_env()
        self._memory_factory = memory_factory
        self._memory: Any | None = None
        self._store_dir: Path | None = None
        self._user_id = "longmemeval"
        self._all_memories: list[dict[str, Any]] = []
        self._provenance: dict[str, dict[str, str | None]] = {}

    @property
    def memory_count(self) -> int:
        return len(self._all_memories)

    @property
    def total_tokens(self) -> int:
        return sum(
            estimate_tokens(_memory_text(memory))
            for memory in self._all_memories
            if _memory_text(memory)
        )

    @property
    def storage_size_bytes(self) -> int:
        if self._store_dir is None or not self._store_dir.exists():
            return 0
        return sum(
            path.stat().st_size
            for path in self._store_dir.rglob("*")
            if path.is_file()
        )

    def stats(self) -> dict[str, JsonValue]:
        """Mark shared reset-store allocation as an approximate sample cost."""

        stats = super().stats()
        stats["storage_size_is_estimate"] = True
        stats["storage_size_note"] = "shared reset-store allocated bytes"
        return stats

    def _reset_impl(self) -> None:
        if self.workspace_dir is None:
            raise RuntimeError("workspace_dir must be set before Mem0 reset")
        self._user_id = _safe_entity_id(self.workspace_dir.name)
        self._all_memories = []
        self._provenance = {}
        if self._memory is None:
            self._store_dir = self.workspace_dir.parent / "_mem0_store"
            self._store_dir.mkdir(parents=True, exist_ok=True)
            self._memory = self._create_memory(self._store_dir)
        self._memory.reset()

    def _ingest_event_impl(self, event: Event) -> None:
        self._add_messages([event])

    def _ingest_session_impl(self, events: list[Event]) -> None:
        if events:
            self._add_messages(events)

    def _finalize_ingestion_impl(self) -> None:
        self._all_memories = self._get_all()

    def _retrieve_impl(
        self,
        query: str,
        *,
        top_k: int,
        question_date: str | None,
        metadata: dict[str, JsonValue],
    ) -> list[RetrievedMemory]:
        if self._memory is None:
            raise RuntimeError("Mem0 adapter must be reset before retrieval")
        raw = _mem0_search(
            self._memory,
            query=query,
            user_id=self._user_id,
            top_k=top_k,
        )
        results: list[RetrievedMemory] = []
        for item in _result_items(raw):
            memory_id = str(item.get("id") or item.get("memory_id") or "").strip()
            content = _memory_text(item).strip()
            if not memory_id or not content:
                continue
            item_metadata = _item_metadata(item)
            provenance = self._provenance.get(memory_id, {})
            source_session_id = _first_text(
                provenance.get("source_session_id"),
                item_metadata.get("source_session_id"),
                item.get("source_session_id"),
            )
            source_date = _first_text(
                provenance.get("source_date"),
                item_metadata.get("source_date"),
                item.get("source_date"),
            )
            results.append(
                RetrievedMemory(
                    memory_id=memory_id,
                    content=content,
                    score=_score(item),
                    source_session_id=source_session_id,
                    source_date=source_date,
                    memory_type="mem0_memory",
                    token_count=estimate_tokens(content),
                    metadata={
                        "retrieval_strategy": self.name,
                        "official_framework": "mem0",
                        "framework_event": str(item.get("event") or ""),
                        "raw_metadata": _json_mapping(item_metadata),
                    },
                )
            )
        return results[:top_k]

    def close(self) -> None:
        if self._memory is None:
            return
        vector_store = getattr(self._memory, "vector_store", None)
        client = getattr(vector_store, "client", None)
        close = getattr(client, "close", None)
        if callable(close):
            close()
        self._memory = None
        gc.collect()

    def _create_memory(self, store_dir: Path) -> Any:
        config = build_mem0_config(self.runtime, store_dir=store_dir)
        if self._memory_factory is not None:
            return self._memory_factory(config)
        os.environ.setdefault("MEM0_TELEMETRY", "false")
        try:
            from mem0 import Memory
        except ImportError as exc:
            raise Mem0DependencyError(
                'Mem0 adapter requires: python -m pip install -e ".[official-mem0]"'
            ) from exc
        return Memory.from_config(config)

    def _add_messages(self, events: list[Event]) -> None:
        if self._memory is None:
            raise RuntimeError("Mem0 adapter must be reset before ingestion")
        first = events[0]
        source_session_id = _event_metadata_text(first, "history_session_id")
        source_date = _event_metadata_text(first, "history_date")
        messages = [
            {
                "role": str(event.metadata.get("role") or event.event_type.value),
                "content": str(event.content),
            }
            for event in events
            if str(event.content).strip()
        ]
        if not messages:
            return
        response = self._memory.add(
            messages,
            user_id=self._user_id,
            metadata={
                "source_session_id": source_session_id,
                "source_date": source_date,
                "question_id": _event_metadata_text(first, "question_id"),
            },
            infer=self.runtime.official_memory_infer,
        )
        for item in _result_items(response):
            memory_id = str(item.get("id") or item.get("memory_id") or "").strip()
            event = str(item.get("event") or "").upper()
            if not memory_id:
                continue
            if event in {"DELETE", "DELETED"}:
                self._provenance.pop(memory_id, None)
            else:
                self._provenance[memory_id] = {
                    "source_session_id": source_session_id,
                    "source_date": source_date,
                }

    def _get_all(self) -> list[dict[str, Any]]:
        if self._memory is None:
            return []
        try:
            raw = self._memory.get_all(filters={"user_id": self._user_id})
        except TypeError:
            raw = self._memory.get_all(user_id=self._user_id)
        return _result_items(raw)


def build_mem0_config(
    runtime: FrameworkRuntimeConfig,
    *,
    store_dir: Path,
) -> dict[str, Any]:
    """Build the pinned Mem0 OSS config used in paper runs."""

    return {
        "version": "v1.1",
        "llm": {
            "provider": "vllm",
            "config": {
                "model": runtime.llm_model,
                "vllm_base_url": runtime.vllm_base_url,
                "api_key": runtime.vllm_api_key or "local-vllm-key",
                "temperature": runtime.official_llm_temperature,
                "top_p": 1.0,
                "max_tokens": runtime.official_llm_max_tokens,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": runtime.embedding_model,
                "embedding_dims": runtime.embedding_dimension,
                "model_kwargs": {
                    "device": runtime.embedding_device,
                    "trust_remote_code": True,
                },
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "vmp_longmemeval_mem0",
                "embedding_model_dims": runtime.embedding_dimension,
                "path": str(store_dir / "qdrant"),
                "on_disk": True,
            },
        },
        "history_db_path": str(store_dir / "history.db"),
    }


def _mem0_search(memory: Any, *, query: str, user_id: str, top_k: int) -> Any:
    try:
        return memory.search(
            query=query,
            filters={"user_id": user_id},
            top_k=top_k,
        )
    except TypeError:
        return memory.search(query, user_id=user_id, limit=top_k)


def _result_items(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        raw = raw.get("results", [])
    if not isinstance(raw, list):
        return []
    items: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            items.append(item)
        elif callable(getattr(item, "model_dump", None)):
            dumped = item.model_dump(mode="python")
            if isinstance(dumped, dict):
                items.append(dumped)
    return items


def _memory_text(item: dict[str, Any]) -> str:
    for key in ("memory", "data", "text", "content"):
        value = item.get(key)
        if isinstance(value, str):
            return value
    return ""


def _item_metadata(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _score(item: dict[str, Any]) -> float:
    value = item.get("score", item.get("similarity", 0.0))
    score = float(value) if isinstance(value, int | float) else 0.0
    return score if math.isfinite(score) else 0.0


def _first_text(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def _event_metadata_text(event: Event, key: str) -> str | None:
    value = event.metadata.get(key)
    return value if isinstance(value, str) and value else None


def _json_mapping(values: dict[str, Any]) -> dict[str, JsonValue]:
    return {
        str(key): value
        for key, value in values.items()
        if _is_json_value(value)
    }


def _is_json_value(value: object) -> bool:
    if value is None or isinstance(value, str | int | float | bool):
        return True
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_json_value(item)
            for key, item in value.items()
        )
    return False


def _safe_entity_id(value: str) -> str:
    normalized = "".join(
        char if char.isalnum() or char in "_.-" else "_"
        for char in value
    )
    return normalized.strip("._") or "longmemeval"
