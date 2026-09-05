"""Official Mem0 OSS adapter using local OpenAI-compatible model services."""

from __future__ import annotations

import gc
import json
import logging
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
LOGGER = logging.getLogger(__name__)
MEM0_LLM_COMPATIBILITY_VERSION = "mem0_v2010_json_transport_v4"
MEM0_MEMORY_STATS_TOP_K = 10_000


class Mem0DependencyError(RuntimeError):
    """Raised when the pinned official Mem0 package is unavailable."""


class Mem0LlmProtocolError(RuntimeError):
    """Raised when Mem0's JSON response is still invalid after one retry."""


class _Mem0LlmResponseAdapter:
    """Harden the Mem0 2.0.10/Qwen JSON transport without changing semantics.

    Mem0's additive extractor consumes ``memory`` as a list of objects with a
    ``text`` field. Local instruction models can preserve the semantic facts
    while returning that list as strings. Long extraction responses can also
    reach a generation ceiling before the JSON object closes. This adapter
    normalizes only the known string wire shape and retries invalid JSON once
    with a larger, frozen output budget. It never repairs or invents facts.
    Version 4 records validation failure categories and uses a paper-pipeline
    guard to keep the 8192-token retry ceiling from being overridden by stale
    shell state.
    """

    def __init__(self, delegate: Any, *, retry_max_tokens: int = 8192) -> None:
        self._delegate = delegate
        self.retry_max_tokens = retry_max_tokens
        self.reset_stats()

    def reset_stats(self) -> None:
        """Reset counters so every retrieval record contains per-question costs."""

        self.logical_call_count = 0
        self.request_count = 0
        self.json_mode_call_count = 0
        self.initial_invalid_json_count = 0
        self.retry_attempt_count = 0
        self.retry_success_count = 0
        self.unrecovered_invalid_json_count = 0
        self.request_exception_count = 0
        self.normalized_response_count = 0
        self.normalized_item_count = 0
        self.output_character_count = 0
        self.max_response_characters = 0
        self.initial_invalid_reason_counts: dict[str, int] = {}
        self.unrecovered_invalid_reason_counts: dict[str, int] = {}
        self.initial_invalid_max_response_characters = 0
        self.unrecovered_invalid_max_response_characters = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def generate_response(self, *args: Any, **kwargs: Any) -> Any:
        self.logical_call_count += 1
        json_mode = _is_json_object_response_format(kwargs.get("response_format"))
        if json_mode:
            self.json_mode_call_count += 1
        response = self._request(*args, **kwargs)
        validation_error = _json_object_validation_error(response)
        if json_mode and validation_error is not None:
            self.initial_invalid_json_count += 1
            _increment(self.initial_invalid_reason_counts, validation_error)
            self.initial_invalid_max_response_characters = max(
                self.initial_invalid_max_response_characters,
                _response_characters(response),
            )
            self.retry_attempt_count += 1
            if self.retry_attempt_count == 1:
                LOGGER.warning(
                    "Mem0 returned invalid JSON; retrying once with max_tokens=%d.",
                    self.retry_max_tokens,
                )
            retry_kwargs = dict(kwargs)
            retry_kwargs["max_tokens"] = self.retry_max_tokens
            response = self._request(*args, **retry_kwargs)
            retry_validation_error = _json_object_validation_error(response)
            if retry_validation_error is not None:
                self.unrecovered_invalid_json_count += 1
                _increment(
                    self.unrecovered_invalid_reason_counts,
                    retry_validation_error,
                )
                self.unrecovered_invalid_max_response_characters = max(
                    self.unrecovered_invalid_max_response_characters,
                    _response_characters(response),
                )
                raise Mem0LlmProtocolError(
                    "Mem0 extraction returned invalid JSON after one larger-budget "
                    f"retry (reason={retry_validation_error}, "
                    f"characters={_response_characters(response)})"
                )
            self.retry_success_count += 1
        normalized, item_count = _normalize_mem0_memory_items(response)
        if item_count:
            self.normalized_response_count += 1
            self.normalized_item_count += item_count
            if self.normalized_response_count == 1:
                LOGGER.warning(
                    "Applied %s to normalize Mem0 string-form memory items.",
                    MEM0_LLM_COMPATIBILITY_VERSION,
                )
        return normalized

    def stats(self) -> dict[str, JsonValue]:
        """Return manifest-safe per-question transport counters."""

        return {
            "mem0_llm_compatibility_version": MEM0_LLM_COMPATIBILITY_VERSION,
            "mem0_llm_logical_calls": self.logical_call_count,
            "mem0_llm_requests": self.request_count,
            "mem0_llm_json_mode_calls": self.json_mode_call_count,
            "mem0_llm_initial_invalid_json": self.initial_invalid_json_count,
            "mem0_llm_retry_attempts": self.retry_attempt_count,
            "mem0_llm_retry_successes": self.retry_success_count,
            "mem0_llm_unrecovered_invalid_json": (
                self.unrecovered_invalid_json_count
            ),
            "mem0_llm_request_exceptions": self.request_exception_count,
            "mem0_llm_normalized_responses": self.normalized_response_count,
            "mem0_llm_normalized_items": self.normalized_item_count,
            "mem0_llm_output_characters": self.output_character_count,
            "mem0_llm_max_response_characters": self.max_response_characters,
            "mem0_llm_retry_max_tokens": self.retry_max_tokens,
            "mem0_llm_initial_invalid_reason_counts": dict(
                self.initial_invalid_reason_counts
            ),
            "mem0_llm_unrecovered_invalid_reason_counts": dict(
                self.unrecovered_invalid_reason_counts
            ),
            "mem0_llm_initial_invalid_max_response_characters": (
                self.initial_invalid_max_response_characters
            ),
            "mem0_llm_unrecovered_invalid_max_response_characters": (
                self.unrecovered_invalid_max_response_characters
            ),
        }

    def _request(self, *args: Any, **kwargs: Any) -> Any:
        self.request_count += 1
        try:
            response = self._delegate.generate_response(*args, **kwargs)
        except Exception:
            self.request_exception_count += 1
            raise
        if isinstance(response, str):
            response_characters = len(response)
            self.output_character_count += response_characters
            self.max_response_characters = max(
                self.max_response_characters,
                response_characters,
            )
        return response


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
        llm = getattr(self._memory, "llm", None)
        if isinstance(llm, _Mem0LlmResponseAdapter):
            stats.update(llm.stats())
        else:
            stats["mem0_llm_compatibility_version"] = (
                MEM0_LLM_COMPATIBILITY_VERSION
            )
        stats["mem0_memory_stats_top_k"] = MEM0_MEMORY_STATS_TOP_K
        stats["mem0_memory_count_truncated"] = (
            len(self._all_memories) >= MEM0_MEMORY_STATS_TOP_K
        )
        stats["mem0_bm25_enabled"] = _mem0_bm25_enabled(self._memory)
        stats["mem0_spacy_lemma_enabled"] = _mem0_spacy_lemma_enabled()
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
        llm = getattr(self._memory, "llm", None)
        if isinstance(llm, _Mem0LlmResponseAdapter):
            llm.reset_stats()

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
            memory = self._memory_factory(config)
        else:
            os.environ.setdefault("MEM0_TELEMETRY", "false")
            try:
                from mem0 import Memory  # type: ignore[import-not-found]
            except ImportError as exc:
                raise Mem0DependencyError(
                    'Mem0 adapter requires: python -m pip install -e ".[official-mem0]"'
                ) from exc
            memory = Memory.from_config(config)
        _install_mem0_llm_response_adapter(
            memory,
            retry_max_tokens=self.runtime.official_llm_retry_max_tokens,
        )
        return memory

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
            raw = self._memory.get_all(
                filters={"user_id": self._user_id},
                top_k=MEM0_MEMORY_STATS_TOP_K,
            )
        except TypeError:
            try:
                raw = self._memory.get_all(filters={"user_id": self._user_id})
            except TypeError:
                raw = self._memory.get_all(
                    user_id=self._user_id,
                    limit=MEM0_MEMORY_STATS_TOP_K,
                )
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
            # Keep Mem0's official provider/factory while serving the shared
            # BGE-M3 model out of process.  This prevents Mem0 from importing a
            # second torch/transformers stack into the vLLM experiment process.
            "provider": "openai",
            "config": {
                "model": runtime.embedding_model,
                "embedding_dims": runtime.embedding_dimension,
                "openai_base_url": runtime.embedding_base_url,
                "api_key": runtime.embedding_api_key or "local-embedding-key",
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


def _install_mem0_llm_response_adapter(
    memory: Any,
    *,
    retry_max_tokens: int,
) -> None:
    llm = getattr(memory, "llm", None)
    if llm is not None and not isinstance(llm, _Mem0LlmResponseAdapter):
        memory.llm = _Mem0LlmResponseAdapter(
            llm,
            retry_max_tokens=retry_max_tokens,
        )


def _is_json_object_response_format(value: Any) -> bool:
    return isinstance(value, dict) and value.get("type") in {
        "json_object",
        "json_schema",
    }


def _is_valid_json_object(response: Any) -> bool:
    return _json_object_validation_error(response) is None


def _json_object_validation_error(response: Any) -> str | None:
    if not isinstance(response, str):
        return "non_string_response"
    if not response.strip():
        return "empty_response"
    start = response.find("{")
    end = response.rfind("}")
    if start < 0:
        return "missing_json_object"
    if end <= start:
        return "unterminated_json_object"
    fragment = response[start : end + 1]
    if _has_unclosed_json_delimiters(fragment):
        return "unterminated_json_object"
    try:
        payload = json.loads(fragment, strict=False)
    except (json.JSONDecodeError, TypeError):
        return "json_decode_error"
    return None if isinstance(payload, dict) else "non_object_json"


def _increment(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _response_characters(response: Any) -> int:
    return len(response) if isinstance(response, str) else 0


def _has_unclosed_json_delimiters(value: str) -> bool:
    stack: list[str] = []
    in_string = False
    escaped = False
    pairs = {"}": "{", "]": "["}
    for character in value:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "{[":
            stack.append(character)
        elif character in "}]":
            if not stack or stack[-1] != pairs[character]:
                return False
            stack.pop()
    return in_string or bool(stack)


def _normalize_mem0_memory_items(response: Any) -> tuple[Any, int]:
    if not isinstance(response, str):
        return response, 0
    start = response.find("{")
    end = response.rfind("}")
    if start < 0 or end <= start:
        return response, 0
    try:
        payload = json.loads(response[start : end + 1], strict=False)
    except (json.JSONDecodeError, TypeError):
        return response, 0
    if not isinstance(payload, dict):
        return response, 0
    memories = payload.get("memory")
    if not isinstance(memories, list):
        return response, 0
    normalized: list[dict[str, Any]] = []
    normalized_count = 0
    for item in memories:
        if isinstance(item, str):
            normalized.append({"text": item})
            normalized_count += 1
        elif isinstance(item, dict):
            normalized.append(item)
        else:
            return response, 0
    if not normalized_count:
        return response, 0
    payload["memory"] = normalized
    return json.dumps(payload, ensure_ascii=False), normalized_count


def _mem0_search(memory: Any, *, query: str, user_id: str, top_k: int) -> Any:
    try:
        return memory.search(
            query=query,
            filters={"user_id": user_id},
            top_k=top_k,
        )
    except TypeError:
        return memory.search(query, user_id=user_id, limit=top_k)


def _mem0_bm25_enabled(memory: Any) -> bool | None:
    vector_store = getattr(memory, "vector_store", None)
    if vector_store is None:
        return None
    if not bool(getattr(vector_store, "_has_bm25_slot", False)):
        return False
    encoder = getattr(vector_store, "_bm25_encoder", None)
    return encoder is not None and encoder is not False


def _mem0_spacy_lemma_enabled() -> bool | None:
    try:
        from mem0.utils import spacy_models  # type: ignore[import-not-found]
    except ImportError:
        return None
    return getattr(spacy_models, "_nlp_lemma", None) is not None


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
