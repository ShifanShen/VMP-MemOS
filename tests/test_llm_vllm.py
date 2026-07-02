"""Tests for vLLM client configuration and LLM memory extraction."""

from pathlib import Path
from typing import Any

from vmp_memos.extraction import LLMMemoryExtractor, LLMMemoryExtractorConfig
from vmp_memos.llm import LLMResponse, VLLMClientConfig, load_vllm_config
from vmp_memos.schemas import Event, MemoryType


def test_load_vllm_config_ignores_provider_field() -> None:
    config = load_vllm_config(Path("configs/llm.yaml"))

    assert isinstance(config, VLLMClientConfig)
    assert config.base_url.endswith("/v1")
    assert config.model
    assert config.generation.max_tokens == 512


def test_llm_memory_extractor_parses_json_without_network() -> None:
    class FakeClient:
        def chat(
            self,
            messages: list[Any],
            *,
            generation: Any = None,
        ) -> LLMResponse:
            return LLMResponse(
                provider="fake",
                model="fake-model",
                text=(
                    '{"candidates":[{"memory_type":"procedural",'
                    '"content":"Close SQLite connections before temp cleanup.",'
                    '"scope":"project/tests","tags":["sqlite","windows"],'
                    '"confidence":0.91,"importance":0.84}]}'
                ),
            )

    event = Event(
        session_id="sess_llm_test",
        event_type="user_message",
        content="A Windows temp cleanup failed because sqlite was still open.",
    )

    extractor = LLMMemoryExtractor(
        FakeClient(),
        LLMMemoryExtractorConfig(default_scope="project/tests"),
    )
    candidates = extractor.extract(event)

    assert len(candidates) == 1
    assert candidates[0].memory_type == MemoryType.PROCEDURAL
    assert candidates[0].source_event_id == event.event_id
    assert candidates[0].scope == "project/tests"
    assert "sqlite" in candidates[0].tags
