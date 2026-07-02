"""Thin, fixed-prompt LongMemEval reader over the shared LLM client."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pydantic import Field, JsonValue

from vmp_memos.frameworks import RetrievedMemory
from vmp_memos.frameworks.text import estimate_tokens
from vmp_memos.llm.base import ChatMessage, LLMGenerationConfig, LLMResponse
from vmp_memos.schemas.base import (
    NonEmptyStr,
    NonNegativeInt,
    SchemaModel,
)

LONGMEMEVAL_SYSTEM_PROMPT = (
    "You answer LongMemEval questions using only the supplied retrieved memory."
)

LONGMEMEVAL_USER_PROMPT = (
    "You are answering a LongMemEval question using retrieved long-term memory.\n\n"
    "Question date:\n"
    "{question_date}\n\n"
    "Question:\n"
    "{question}\n\n"
    "Retrieved memory:\n"
    "{memory_context}\n\n"
    "Instructions:\n"
    "- Answer using only the retrieved memory.\n"
    "- Prefer newer evidence when memories conflict.\n"
    '- If the answer is not supported by the retrieved memory, say "I don\'t know".\n'
    "- Keep the answer concise."
)


class ChatClient(Protocol):
    """Structural interface implemented by VLLMClient and test doubles."""

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        generation: LLMGenerationConfig | None = None,
    ) -> LLMResponse:
        """Generate one chat response."""


class LongMemEvalReaderConfig(SchemaModel):
    """Fixed reader settings shared by every compared memory method."""

    top_k: NonNegativeInt = 5
    generation: LLMGenerationConfig = Field(
        default_factory=lambda: LLMGenerationConfig(
            max_tokens=128,
            temperature=0.0,
            top_p=1.0,
        )
    )


class ReaderOutput(SchemaModel):
    """Reader answer plus usage needed for paper cost analysis."""

    answer: str
    model: NonEmptyStr
    provider: NonEmptyStr
    finish_reason: str | None = None
    prompt: NonEmptyStr
    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt
    usage: dict[str, JsonValue] = Field(default_factory=dict)


class LongMemEvalReader:
    """Apply one prompt and one generation config to all retrieval methods."""

    def __init__(
        self,
        client: ChatClient,
        config: LongMemEvalReaderConfig | None = None,
    ) -> None:
        self.client = client
        self.config = config or LongMemEvalReaderConfig()
        if self.config.top_k < 1:
            raise ValueError("reader top_k must be at least 1")

    def answer(
        self,
        *,
        question: str,
        question_date: str | None,
        memories: Sequence[RetrievedMemory],
    ) -> ReaderOutput:
        """Answer one question from the first top-k retrieved memories."""

        prompt = build_longmemeval_prompt(
            question=question,
            question_date=question_date,
            memories=memories[: self.config.top_k],
        )
        response = self.client.chat(
            [
                ChatMessage(role="system", content=LONGMEMEVAL_SYSTEM_PROMPT),
                ChatMessage(role="user", content=prompt),
            ],
            generation=self.config.generation,
        )
        input_tokens = _usage_tokens(
            response.usage,
            "prompt_tokens",
            fallback=estimate_tokens(LONGMEMEVAL_SYSTEM_PROMPT + "\n" + prompt),
        )
        output_tokens = _usage_tokens(
            response.usage,
            "completion_tokens",
            fallback=estimate_tokens(response.text) if response.text else 0,
        )
        return ReaderOutput(
            answer=response.text.strip(),
            model=response.model,
            provider=response.provider,
            finish_reason=response.finish_reason,
            prompt=prompt,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usage=response.usage,
        )


def build_longmemeval_prompt(
    *,
    question: str,
    question_date: str | None,
    memories: Sequence[RetrievedMemory],
) -> str:
    """Render the immutable paper QA prompt."""

    memory_context = "\n\n".join(
        _format_memory(rank, memory)
        for rank, memory in enumerate(memories, start=1)
    )
    if not memory_context:
        memory_context = "(No memory retrieved.)"
    return LONGMEMEVAL_USER_PROMPT.format(
        question_date=question_date or "unknown",
        question=question,
        memory_context=memory_context,
    )


def _format_memory(rank: int, memory: RetrievedMemory) -> str:
    date = memory.source_date or "unknown"
    session_id = memory.source_session_id or "unknown"
    return (
        f"[Memory {rank} | session={session_id} | date={date}]\n"
        f"{memory.content}"
    )


def _usage_tokens(
    usage: dict[str, JsonValue],
    key: str,
    *,
    fallback: int,
) -> int:
    value = usage.get(key)
    return int(value) if isinstance(value, int | float) and value >= 0 else fallback
