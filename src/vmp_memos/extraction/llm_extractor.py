"""LLM-backed memory candidate extraction."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import Field

from vmp_memos.llm import ChatMessage, LLMGenerationConfig, VLLMClient
from vmp_memos.schemas import Event, MemoryCandidate, MemoryType
from vmp_memos.schemas.base import NonEmptyStr, SchemaModel


class LLMMemoryExtractorConfig(SchemaModel):
    """Prompt and decoding settings for LLM memory extraction."""

    default_scope: NonEmptyStr = "global"
    max_candidates: int = Field(default=5, ge=1)
    system_prompt: NonEmptyStr = (
        "You extract durable memory candidates for a long-running agent. "
        "Return strict JSON only."
    )
    generation: LLMGenerationConfig = Field(
        default_factory=lambda: LLMGenerationConfig(
            max_tokens=768,
            temperature=0.0,
            top_p=1.0,
        )
    )


class LLMMemoryExtractor:
    """Extract ``MemoryCandidate`` objects with a vLLM-backed chat model."""

    def __init__(
        self,
        client: VLLMClient | None = None,
        config: LLMMemoryExtractorConfig | None = None,
    ) -> None:
        self.client = client or VLLMClient()
        self.config = config or LLMMemoryExtractorConfig()

    def extract(self, event: Event) -> list[MemoryCandidate]:
        """Call the LLM and validate returned candidates."""

        response = self.client.chat(
            [
                ChatMessage(role="system", content=self.config.system_prompt),
                ChatMessage(role="user", content=self._prompt(event)),
            ],
            generation=self.config.generation,
        )
        records = _candidate_records_from_text(response.text)
        candidates: list[MemoryCandidate] = []
        for record in records[: self.config.max_candidates]:
            candidate = _candidate_from_record(
                record,
                event=event,
                default_scope=self.config.default_scope,
            )
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def _prompt(self, event: Event) -> str:
        return (
            "Extract memory candidates from the event below.\n\n"
            "Rules:\n"
            "- Keep only stable preferences, project facts, procedures, lessons, "
            "or resources useful in future sessions.\n"
            "- Do not include transient chatter.\n"
            "- Use memory_type: semantic, episodic, procedural, reflective, or resource.\n"
            "- confidence and importance must be numbers between 0 and 1.\n"
            "- Return JSON with shape: "
            "{\"candidates\":[{\"memory_type\":\"semantic\","
            "\"content\":\"...\",\"scope\":\"global\",\"tags\":[\"...\"],"
            "\"confidence\":0.8,\"importance\":0.7,\"metadata\":{}}]}.\n\n"
            f"Event type: {event.event_type.value}\n"
            f"Event id: {event.event_id}\n"
            f"Session id: {event.session_id}\n"
            f"Default scope: {self.config.default_scope}\n"
            f"Content:\n{event.content}"
        )


def _candidate_records_from_text(text: str) -> list[Mapping[str, Any]]:
    loaded = _load_json_object(text)
    raw_candidates: object
    if isinstance(loaded, dict):
        raw_candidates = loaded.get("candidates", [])
    elif isinstance(loaded, list):
        raw_candidates = loaded
    else:
        raw_candidates = []
    if not isinstance(raw_candidates, list):
        return []
    return [record for record in raw_candidates if isinstance(record, Mapping)]


def _candidate_from_record(
    record: Mapping[str, Any],
    *,
    event: Event,
    default_scope: str,
) -> MemoryCandidate | None:
    content = str(record.get("content", "") or "").strip()
    if not content:
        return None
    memory_type = _memory_type(str(record.get("memory_type", "semantic")))
    tags = _string_list(record.get("tags", []))
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return MemoryCandidate(
        source_event_id=event.event_id,
        memory_type=memory_type,
        content=content,
        scope=str(record.get("scope") or default_scope),
        tags=tags,
        confidence=_score(record.get("confidence", 0.7)),
        importance=_score(record.get("importance", 0.5)),
        metadata={
            **metadata,
            "extractor": "llm",
            "provider": "vllm",
        },
    )


def _load_json_object(text: str) -> object:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        starts = [index for index in (stripped.find("{"), stripped.find("[")) if index >= 0]
        if not starts:
            raise ValueError("LLM output did not contain JSON") from None
        start = min(starts)
        end = max(stripped.rfind("}"), stripped.rfind("]"))
        if end <= start:
            raise ValueError("LLM output did not contain JSON") from None
        return json.loads(stripped[start : end + 1])


def _memory_type(value: str) -> MemoryType:
    normalized = value.strip().casefold()
    for memory_type in MemoryType:
        if memory_type.value == normalized:
            return memory_type
    return MemoryType.SEMANTIC


def _score(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, numeric))


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value if str(item).strip()]
    return []
