"""Regression tests for the hybrid LongMemEval QA-v2.1 evidence protocol."""

from __future__ import annotations

from collections.abc import Sequence

from vmp_memos.frameworks import RetrievedMemory
from vmp_memos.llm import (
    LONGMEMEVAL_HYBRID_READER_PROMPT_VERSION,
    ChatMessage,
    LLMGenerationConfig,
    LLMResponse,
    LongMemEvalReader,
    LongMemEvalReaderConfig,
    build_query_evidence_windows,
)


class CapturingClient:
    """Capture the hybrid prompt without contacting vLLM."""

    def __init__(self) -> None:
        self.messages: list[ChatMessage] = []

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        generation: LLMGenerationConfig | None = None,
    ) -> LLMResponse:
        self.messages = list(messages)
        return LLMResponse(
            provider="fake-vllm",
            model="fake-reader",
            text="Golden Retriever",
            finish_reason="stop",
            usage={"prompt_tokens": 180, "completion_tokens": 4},
        )


def test_query_window_keeps_the_answer_sentence_next_to_a_lexical_anchor() -> None:
    memory = _memory(
        "s1",
        "user: I finally adopted a dog after months of searching. "
        "She is a Golden Retriever named Sunny. We went hiking yesterday.\n"
        "assistant: That sounds wonderful.",
    )

    windows = build_query_evidence_windows(
        question="What breed is my dog?",
        memories=[memory],
    )

    assert len(windows) == 1
    selected = [span.text for span in windows[0].spans]
    assert "I finally adopted a dog after months of searching." in selected
    assert "She is a Golden Retriever named Sunny." in selected


def test_hybrid_reader_places_facts_and_windows_before_the_question() -> None:
    client = CapturingClient()
    reader = LongMemEvalReader(
        client,
        LongMemEvalReaderConfig(
            top_k=5,
            prompt_version=LONGMEMEVAL_HYBRID_READER_PROMPT_VERSION,
            evidence_mode="reranker_facts_with_query_windows",
        ),
    )

    output = reader.answer(
        question="What breed is my dog?",
        question_date="2024-02-01",
        memories=[
            _memory(
                "s1",
                "user: I finally adopted a dog. She is a Golden Retriever named Sunny.",
            )
        ],
    )

    prompt = client.messages[-1].content
    assert prompt.index("Grounded Facts:") < prompt.index("Evidence Windows:")
    assert prompt.index("Evidence Windows:") < prompt.index("Question:")
    assert "Golden Retriever" in prompt
    assert prompt.rstrip().endswith("Answer:")
    assert output.prompt_version == LONGMEMEVAL_HYBRID_READER_PROMPT_VERSION
    assert output.evidence_mode == "reranker_facts_with_query_windows"
    assert output.evidence_window_count == 1
    assert output.evidence_span_count == 2


def test_query_window_keeps_the_previous_user_turn_across_a_long_reply() -> None:
    filler = " ".join(f"Organization tip {index}." for index in range(40))
    memory = _memory(
        "s1",
        "user: I have been using the Cartwheel app from Target.\n"
        f"assistant: {filler}\n"
        "user: I redeemed a $5 coupon on coffee creamer last Sunday.\n"
        "assistant: That was a useful discount.",
    )

    windows = build_query_evidence_windows(
        question="Where did I redeem a $5 coupon on coffee creamer?",
        memories=[memory],
    )

    selected = [span.text for window in windows for span in window.spans]
    assert "I have been using the Cartwheel app from Target." in selected
    assert "I redeemed a $5 coupon on coffee creamer last Sunday." in selected


def test_query_window_total_budget_is_deterministic() -> None:
    memories = [
        _memory(
            f"s{index}",
            "user: I bought feed for the animals. "
            + "The bag weighed 40 pounds. " * 100,
        )
        for index in range(1, 6)
    ]

    first = build_query_evidence_windows(
        question="What was the total weight of the feed?",
        memories=memories,
        max_chars_per_memory=240,
        total_max_chars=600,
    )
    second = build_query_evidence_windows(
        question="What was the total weight of the feed?",
        memories=memories,
        max_chars_per_memory=240,
        total_max_chars=600,
    )

    assert first == second
    assert sum(window.char_count for window in first) <= 600


def _memory(session_id: str, content: str) -> RetrievedMemory:
    return RetrievedMemory(
        memory_id=session_id,
        source_session_id=session_id,
        source_date="2024-01-20",
        content=content,
        score=1.0,
        token_count=100,
    )
