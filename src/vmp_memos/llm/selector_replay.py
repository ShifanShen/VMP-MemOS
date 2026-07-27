"""Exact first-stage selector replay for boundary-only LongMemEval experiments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pydantic import JsonValue

from vmp_memos.frameworks import RetrievedMemory
from vmp_memos.llm.base import ChatMessage, LLMGenerationConfig, LLMResponse
from vmp_memos.llm.reranker import (
    LONGMEMEVAL_BOUNDARY_SYSTEM_PROMPT,
    LONGMEMEVAL_RERANK_PROMPT_VERSION,
    LONGMEMEVAL_RERANK_SYSTEM_PROMPT,
    LongMemEvalRerankerConfig,
    RerankerChatClient,
    build_longmemeval_rerank_prompt,
    prepare_longmemeval_rerank_candidates,
)


@dataclass(frozen=True)
class SelectorReplayCache:
    """Validated selector responses indexed by their exact user-prompt hash."""

    source_run: Path
    selector_run: Path
    selector_manifest_sha256: str
    source_manifest_sha256: str
    responses: dict[str, LLMResponse]
    record_count: int
    provider: str
    model: str


class SelectorReplayClient:
    """Replay selector responses exactly and delegate boundary calls to vLLM."""

    def __init__(
        self,
        delegate: RerankerChatClient,
        cache: SelectorReplayCache,
    ) -> None:
        self.delegate = delegate
        self.cache = cache
        self.selector_replay_hits = 0
        self.boundary_live_calls = 0
        self.used_prompt_hashes: set[str] = set()

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        generation: LLMGenerationConfig | None = None,
    ) -> LLMResponse:
        """Return an exact cached selector response or perform a live boundary call."""

        if len(messages) != 2:
            raise ValueError("selector replay requires exactly one system and one user message")
        system_prompt = messages[0].content
        if system_prompt == LONGMEMEVAL_RERANK_SYSTEM_PROMPT:
            prompt_sha256 = hashlib.sha256(messages[1].content.encode("utf-8")).hexdigest()
            response = self.cache.responses.get(prompt_sha256)
            if response is None:
                raise ValueError(
                    "Selector replay cache miss. Candidate order or selector prompt settings "
                    f"changed (prompt_sha256={prompt_sha256})."
                )
            self.selector_replay_hits += 1
            self.used_prompt_hashes.add(prompt_sha256)
            return response.model_copy(deep=True)
        if system_prompt != LONGMEMEVAL_BOUNDARY_SYSTEM_PROMPT:
            raise ValueError("selector replay client received an unknown prompt stage")
        self.boundary_live_calls += 1
        return self.delegate.chat(messages, generation=generation)


def load_selector_replay_cache(
    selector_run: str | Path,
    *,
    source_run: str | Path,
    methods: Sequence[str],
    expected_model: str | None = None,
) -> SelectorReplayCache:
    """Load and validate selector responses from a completed rerank run."""

    source_dir = Path(source_run).expanduser().resolve()
    selector_dir = Path(selector_run).expanduser().resolve()
    source_manifest_path = source_dir / "manifest.json"
    selector_manifest_path = selector_dir / "manifest.json"
    source_manifest = _read_json_object(source_manifest_path)
    selector_manifest = _read_json_object(selector_manifest_path)
    if source_manifest.get("status") != "completed":
        raise ValueError(f"Selector replay source run is not completed: {source_dir}")
    if selector_manifest.get("status") != "completed":
        raise ValueError(f"Selector replay run is not completed: {selector_dir}")

    source_manifest_sha256 = _sha256(source_manifest_path)
    if (
        selector_manifest.get("source_retrieval_manifest_sha256")
        != source_manifest_sha256
    ):
        raise ValueError(
            "Selector replay run was not produced from the requested candidate manifest"
        )

    normalized_methods = _ordered_unique(_normalize_method(method) for method in methods)
    if not normalized_methods:
        raise ValueError("selector replay requires at least one source method")
    expected_count = _integer(source_manifest.get("sample_count"))
    if expected_count < 1:
        raise ValueError("selector replay source manifest has no samples")

    responses: dict[str, LLMResponse] = {}
    record_count = 0
    observed: set[tuple[str, str]] = set()
    for method in normalized_methods:
        retrieval_path = _selector_records_path(selector_dir, method)
        method_count = 0
        with retrieval_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected JSON object in {retrieval_path}")
                metadata = payload.get("rerank_metadata")
                if not isinstance(metadata, dict):
                    raise ValueError(f"Missing rerank_metadata in {retrieval_path}")
                if metadata.get("source_method") != method:
                    raise ValueError(
                        f"Selector replay source method mismatch in {retrieval_path}"
                    )
                if metadata.get("prompt_version") != LONGMEMEVAL_RERANK_PROMPT_VERSION:
                    raise ValueError("Selector replay prompt version is incompatible")
                prompt_sha256 = metadata.get("prompt_sha256")
                provider = metadata.get("provider")
                model = metadata.get("model")
                response_text = metadata.get("response_text")
                if not isinstance(prompt_sha256, str) or len(prompt_sha256) != 64:
                    raise ValueError("Selector replay record has an invalid prompt SHA-256")
                if not isinstance(provider, str) or not provider:
                    raise ValueError("Selector replay record is missing its provider")
                if not isinstance(model, str) or not model:
                    raise ValueError("Selector replay record is missing its model")
                if not isinstance(response_text, str):
                    raise ValueError("Selector replay record is missing its response text")
                if expected_model is not None and model != expected_model:
                    raise ValueError(
                        f"Selector replay model {model!r} differs from requested "
                        f"model {expected_model!r}"
                    )
                observed.add((provider, model))
                usage_value = metadata.get("usage")
                usage = (
                    cast(dict[str, JsonValue], usage_value)
                    if isinstance(usage_value, dict)
                    else {}
                )
                finish_reason_value = metadata.get("finish_reason")
                finish_reason = (
                    finish_reason_value
                    if isinstance(finish_reason_value, str)
                    else None
                )
                response = LLMResponse(
                    provider=provider,
                    model=model,
                    text=response_text,
                    finish_reason=finish_reason,
                    usage=usage,
                    raw_response={
                        "selector_replay": True,
                        "selector_run": str(selector_dir),
                        "prompt_sha256": prompt_sha256,
                    },
                )
                existing = responses.get(prompt_sha256)
                if existing is not None and (
                    existing.text != response.text
                    or existing.provider != response.provider
                    or existing.model != response.model
                ):
                    raise ValueError(
                        f"Conflicting selector responses for prompt {prompt_sha256}"
                    )
                responses[prompt_sha256] = response
                method_count += 1
                record_count += 1
        if method_count != expected_count:
            raise ValueError(
                f"Selector replay method {method!r} has {method_count} records; "
                f"expected {expected_count}"
            )
    if len(observed) != 1:
        raise ValueError(
            f"Selector replay must contain one provider/model pair, observed {sorted(observed)}"
        )
    provider, model = next(iter(observed))
    return SelectorReplayCache(
        source_run=source_dir,
        selector_run=selector_dir,
        selector_manifest_sha256=_sha256(selector_manifest_path),
        source_manifest_sha256=source_manifest_sha256,
        responses=responses,
        record_count=record_count,
        provider=provider,
        model=model,
    )


def validate_selector_replay_source(
    cache: SelectorReplayCache,
    *,
    source_run: str | Path,
    methods: Sequence[str],
    config: LongMemEvalRerankerConfig,
    limit: int | None = None,
) -> int:
    """Verify every current selector prompt is cached before any live LLM call."""

    source_dir = Path(source_run).expanduser().resolve()
    if source_dir != cache.source_run:
        raise ValueError("Selector replay preflight source differs from the cache source")
    checked = 0
    for method in _ordered_unique(_normalize_method(method) for method in methods):
        path = source_dir / method / "retrieval.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        method_checked = 0
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                if limit is not None and method_checked >= limit:
                    break
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected JSON object in {path}")
                question = payload.get("question")
                question_date_value = payload.get("question_date")
                memory_values = payload.get("retrieved_memories")
                if not isinstance(question, str) or not question:
                    raise ValueError(f"Selector replay source has no question in {path}")
                if question_date_value is not None and not isinstance(
                    question_date_value,
                    str,
                ):
                    raise ValueError(f"Selector replay source has invalid question_date in {path}")
                if not isinstance(memory_values, list):
                    raise ValueError(f"Selector replay source has no memories in {path}")
                parsed_memories = [
                    RetrievedMemory.model_validate(value)
                    for value in memory_values
                ]
                memories = prepare_longmemeval_rerank_candidates(
                    parsed_memories,
                    candidate_count=config.candidate_count,
                )
                prompt = build_longmemeval_rerank_prompt(
                    question=question,
                    question_date=question_date_value,
                    candidates=memories,
                    config=config,
                )
                prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                if prompt_sha256 not in cache.responses:
                    question_id = payload.get("question_id")
                    raise ValueError(
                        "Selector replay preflight cache miss for "
                        f"method={method!r}, question_id={question_id!r}, "
                        f"prompt_sha256={prompt_sha256}"
                    )
                checked += 1
                method_checked += 1
    return checked


def _selector_records_path(selector_run: Path, source_method: str) -> Path:
    candidates = (
        selector_run / f"{source_method}__vllm_boundary" / "retrieval.jsonl",
        selector_run / f"{source_method}__vllm_rerank" / "retrieval.jsonl",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No selector records for method {source_method!r} in {selector_run}"
    )


def _read_json_object(path: Path) -> dict[str, JsonValue]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return cast(dict[str, JsonValue], payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer(value: object) -> int:
    return int(value) if isinstance(value, int | float) else 0


def _normalize_method(method: str) -> str:
    return method.strip().casefold().replace("-", "_")


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))
