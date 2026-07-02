"""Official LangMem adapter using its memory manager and LangGraph store."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any

from pydantic import JsonValue

from vmp_memos.embeddings import BaseEmbedder, SentenceTransformerEmbedder
from vmp_memos.frameworks.base import (
    BaseMemoryFrameworkAdapter,
    FairnessLevel,
    RetrievedMemory,
)
from vmp_memos.frameworks.runtime import FrameworkRuntimeConfig
from vmp_memos.frameworks.text import estimate_tokens
from vmp_memos.schemas import Event

ComponentsFactory = Callable[
    [FrameworkRuntimeConfig, BaseEmbedder],
    tuple[Any, Any],
]


class LangMemDependencyError(RuntimeError):
    """Raised when the pinned official LangMem packages are unavailable."""


class LangMemOfficialAdapter(BaseMemoryFrameworkAdapter):
    """Thin wrapper around LangMem's official background memory manager."""

    name = "langmem"
    fairness_level = FairnessLevel.FULLY_CONTROLLED
    namespace = ("memories",)

    def __init__(
        self,
        *,
        runtime: FrameworkRuntimeConfig | None = None,
        embedder: BaseEmbedder | None = None,
        components_factory: ComponentsFactory | None = None,
    ) -> None:
        super().__init__()
        self.runtime = runtime or FrameworkRuntimeConfig.from_env()
        self.embedder = embedder
        self._components_factory = components_factory
        self._store: Any | None = None
        self._manager: Any | None = None
        self._memories: list[dict[str, Any]] = []
        self._provenance: dict[str, dict[str, str | None]] = {}

    @property
    def memory_count(self) -> int:
        return len(self._memories)

    @property
    def total_tokens(self) -> int:
        return sum(estimate_tokens(str(memory["content"])) for memory in self._memories)

    @property
    def storage_size_bytes(self) -> int:
        content_bytes = sum(
            len(
                json.dumps(memory, ensure_ascii=False, sort_keys=True).encode("utf-8")
            )
            for memory in self._memories
        )
        vector_bytes = 4 * self.runtime.embedding_dimension * len(self._memories)
        return content_bytes + vector_bytes

    def stats(self) -> dict[str, JsonValue]:
        """Mark logical content/vector bytes as an estimate."""

        stats = super().stats()
        stats["storage_size_is_estimate"] = True
        return stats

    def _reset_impl(self) -> None:
        if not self.runtime.official_memory_infer:
            raise ValueError(
                "LangMem requires official_memory_infer=True because its native "
                "memory manager is the evaluated mechanism"
            )
        embedder = self.embedder or SentenceTransformerEmbedder(
            self.runtime.embedding_model,
            device=self.runtime.embedding_device,
        )
        self.embedder = embedder
        if self._components_factory is not None:
            self._store, self._manager = self._components_factory(
                self.runtime,
                embedder,
            )
        else:
            self._store, self._manager = _create_langmem_components(
                self.runtime,
                embedder,
            )
        self._memories = []
        self._provenance = {}

    def _ingest_event_impl(self, event: Event) -> None:
        self._manage([event])

    def _ingest_session_impl(self, events: list[Event]) -> None:
        if events:
            self._manage(events)

    def _finalize_ingestion_impl(self) -> None:
        self._memories = self._snapshot()

    def _retrieve_impl(
        self,
        query: str,
        *,
        top_k: int,
        question_date: str | None,
        metadata: dict[str, JsonValue],
    ) -> list[RetrievedMemory]:
        if self._store is None:
            raise RuntimeError("LangMem adapter must be reset before retrieval")
        items = self._store.search(self.namespace, query=query, limit=top_k)
        results: list[RetrievedMemory] = []
        for item in items:
            memory_id = str(getattr(item, "key", "") or "").strip()
            content = _langmem_content(getattr(item, "value", None)).strip()
            if not memory_id or not content:
                continue
            provenance = self._provenance.get(memory_id, {})
            score = getattr(item, "score", 0.0)
            numeric_score = float(score) if isinstance(score, int | float) else 0.0
            if not math.isfinite(numeric_score):
                numeric_score = 0.0
            results.append(
                RetrievedMemory(
                    memory_id=memory_id,
                    content=content,
                    score=numeric_score,
                    source_session_id=provenance.get("source_session_id"),
                    source_date=provenance.get("source_date"),
                    memory_type="langmem_memory",
                    token_count=estimate_tokens(content),
                    metadata={
                        "retrieval_strategy": self.name,
                        "official_framework": "langmem",
                        "namespace": list(self.namespace),
                    },
                )
            )
        return results[:top_k]

    def _manage(self, events: list[Event]) -> None:
        if self._manager is None:
            raise RuntimeError("LangMem adapter must be reset before ingestion")
        before = {memory["id"]: memory["content"] for memory in self._snapshot()}
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
        self._manager.invoke({"messages": messages, "max_steps": 1})
        after = self._snapshot()
        first = events[0]
        source_session_id = _event_metadata_text(first, "history_session_id")
        source_date = _event_metadata_text(first, "history_date")
        after_ids = {memory["id"] for memory in after}
        self._provenance = {
            memory_id: provenance
            for memory_id, provenance in self._provenance.items()
            if memory_id in after_ids
        }
        for memory in after:
            memory_id = str(memory["id"])
            if before.get(memory_id) != memory["content"]:
                self._provenance[memory_id] = {
                    "source_session_id": source_session_id,
                    "source_date": source_date,
                }
        self._memories = after

    def _snapshot(self) -> list[dict[str, Any]]:
        if self._store is None:
            return []
        items = self._store.search(self.namespace, limit=10_000)
        snapshot: list[dict[str, Any]] = []
        for item in items:
            memory_id = str(getattr(item, "key", "") or "").strip()
            content = _langmem_content(getattr(item, "value", None)).strip()
            if memory_id and content:
                snapshot.append({"id": memory_id, "content": content})
        return snapshot


def _create_langmem_components(
    runtime: FrameworkRuntimeConfig,
    embedder: BaseEmbedder,
) -> tuple[Any, Any]:
    try:
        from langchain_core.embeddings import Embeddings
        from langchain_openai import ChatOpenAI
        from langgraph.store.memory import InMemoryStore
        from langmem import create_memory_store_manager
    except ImportError as exc:
        raise LangMemDependencyError(
            'LangMem adapter requires: python -m pip install -e ".[official-langmem]"'
        ) from exc

    class SharedEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return embedder.embed(texts)

        def embed_query(self, text: str) -> list[float]:
            return embedder.embed_one(text)

    store = InMemoryStore(
        index={
            "dims": runtime.embedding_dimension,
            "embed": SharedEmbeddings(),
        }
    )
    model = ChatOpenAI(
        model=runtime.llm_model,
        base_url=runtime.vllm_base_url,
        api_key=runtime.vllm_api_key or "local-vllm-key",
        temperature=runtime.official_llm_temperature,
        top_p=1.0,
        max_tokens=runtime.official_llm_max_tokens,
    )
    manager = create_memory_store_manager(
        model,
        namespace=LangMemOfficialAdapter.namespace,
        store=store,
        enable_inserts=True,
        enable_deletes=True,
        query_limit=10,
    )
    return store, manager


def _langmem_content(value: object) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return ""
    content = value.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        nested = content.get("content")
        if isinstance(nested, str):
            return nested
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _event_metadata_text(event: Event, key: str) -> str | None:
    value = event.metadata.get(key)
    return value if isinstance(value, str) and value else None
