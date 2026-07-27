"""Network-free tests for the shared LongMemEval evidence reranker."""

from __future__ import annotations

from typing import Any

import pytest

from vmp_memos.frameworks import RetrievedMemory
from vmp_memos.llm import (
    LONGMEMEVAL_ATOMIC_BOUNDARY_PROMPT_VERSION,
    LONGMEMEVAL_SYMBOLIC_BOUNDARY_PROMPT_VERSION,
    LLMGenerationConfig,
    LLMResponse,
    LongMemEvalEvidenceReranker,
    LongMemEvalRerankerConfig,
    candidate_excerpt,
    guarded_session_ranking,
    prepare_longmemeval_rerank_candidates,
)


class FakeRerankClient:
    def __init__(self, text: str | list[str]) -> None:
        self.responses = [text] if isinstance(text, str) else list(text)
        self.messages: list[Any] = []
        self.all_messages: list[list[Any]] = []
        self.generation: LLMGenerationConfig | None = None
        self.calls = 0

    def chat(
        self,
        messages: list[Any],
        *,
        generation: LLMGenerationConfig | None = None,
    ) -> LLMResponse:
        self.messages = messages
        self.all_messages.append(messages)
        self.generation = generation
        response_text = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return LLMResponse(
            provider="vllm",
            model="Qwen/Qwen2.5-7B-Instruct",
            text=response_text,
            usage={"prompt_tokens": 100, "completion_tokens": 20},
        )


def test_shared_reranker_promotes_one_candidate_behind_protected_top_four() -> None:
    client = FakeRerankClient(
        """
        ```json
        {
          "evidence_needs": ["current activity", "change date"],
          "selected_session_ids": ["s6", "s2", "s1", "s3", "s4"],
          "ranked_session_ids": ["s6", "s2", "s1", "s3", "s4", "s5"]
        }
        ```
        """
    )
    reranker = LongMemEvalEvidenceReranker(
        client,
        LongMemEvalRerankerConfig(candidate_count=6, ranked_output_count=6),
    )

    decision = reranker.rerank(
        question="What activity does Alex currently prefer?",
        question_date="2024-02-01",
        candidates=_memories(6),
    )

    assert decision.selected_session_ids == ["s1", "s2", "s3", "s4", "s6"]
    assert decision.parse_fallback is False
    assert decision.invalid_session_ids == []
    assert decision.input_tokens == 100
    assert client.generation is not None
    assert client.generation.temperature == 0.0
    prompt = client.messages[1].content
    assert "gold" not in prompt.casefold()
    assert "vmp_hierarchical" not in prompt
    assert "session_id=s6" in prompt


def test_reranker_fails_closed_to_original_order_on_malformed_output() -> None:
    reranker = LongMemEvalEvidenceReranker(
        FakeRerankClient("not JSON"),
        LongMemEvalRerankerConfig(candidate_count=6, ranked_output_count=6),
    )

    decision = reranker.rerank(
        question="What happened?",
        question_date=None,
        candidates=_memories(6),
    )

    assert decision.parse_fallback is True
    assert decision.selected_session_ids == ["s1", "s2", "s3", "s4", "s5"]
    assert decision.parse_fallback_reason


def test_guard_and_excerpt_are_deterministic_and_framework_agnostic() -> None:
    assert guarded_session_ranking(
        original_session_ids=["a", "b", "c", "d", "e", "f"],
        proposed_session_ids=["f", "f", "unknown", "e"],
        output_top_k=5,
        protected_top_n=4,
    )[:5] == ["a", "b", "c", "d", "f"]
    excerpt = candidate_excerpt(
        "Which city did Alex move to?",
        "user: Alex likes tea.\nassistant: General reply.\n"
        "user: Alex moved to Paris yesterday.\nassistant: Noted.",
        max_chars=100,
        max_turns=2,
    )
    assert "Paris" in excerpt
    assert len(excerpt) <= 100


def test_candidate_depth_is_applied_after_session_deduplication() -> None:
    memories = _memories(6)
    duplicate = memories[0].model_copy(update={"memory_id": "duplicate-s1"})

    prepared = prepare_longmemeval_rerank_candidates(
        [memories[0], duplicate, *memories[1:]],
        candidate_count=6,
    )

    assert [memory.source_session_id for memory in prepared] == [
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
    ]


def test_paper_reranker_rejects_non_deterministic_sampling() -> None:
    with pytest.raises(ValueError, match="temperature=0"):
        LongMemEvalRerankerConfig(
            generation=LLMGenerationConfig(
                max_tokens=64,
                temperature=0.1,
                top_p=1.0,
            )
        )


def test_v53_boundary_verifier_can_promote_two_complementary_sessions() -> None:
    client = FakeRerankClient(
        [
            json_response(
                selected=["s1", "s2", "s6", "s7"],
                ranked=["s1", "s2", "s6", "s7", "s3"],
            ),
            """
            {
              "evidence_needs": ["first missing fact", "second missing fact"],
              "selected_boundary_session_ids": ["s6", "s7"],
              "decision": "replace_two",
              "confidence": "high"
            }
            """,
        ]
    )
    reranker = LongMemEvalEvidenceReranker(
        client,
        LongMemEvalRerankerConfig(
            candidate_count=7,
            protected_top_n=3,
            ranked_output_count=7,
            boundary_verification=True,
        ),
    )

    decision = reranker.rerank(
        question="Which two facts jointly answer the question?",
        question_date="2024-02-01",
        candidates=_memories(7),
    )

    assert client.calls == 2
    assert decision.selected_session_ids == ["s1", "s2", "s3", "s6", "s7"]
    assert decision.boundary is not None
    assert decision.boundary.replacement_accepted is True
    assert decision.boundary.parse_fallback is False
    boundary_prompt = client.all_messages[1][1].content
    assert "gold" not in boundary_prompt.casefold()
    assert "vmp_hierarchical" not in boundary_prompt
    assert "proposed-promotion | session_id=s6" in boundary_prompt


def test_v53_boundary_verifier_rejects_low_confidence_replacement() -> None:
    client = FakeRerankClient(
        [
            json_response(
                selected=["s1", "s6", "s2", "s3", "s4"],
                ranked=["s1", "s6", "s2", "s3", "s4"],
            ),
            """
            {
              "evidence_needs": ["uncertain fact"],
              "selected_boundary_session_ids": ["s4", "s6"],
              "decision": "replace_one",
              "confidence": "low"
            }
            """,
        ]
    )
    reranker = LongMemEvalEvidenceReranker(
        client,
        LongMemEvalRerankerConfig(
            candidate_count=6,
            protected_top_n=3,
            ranked_output_count=6,
            boundary_verification=True,
        ),
    )

    decision = reranker.rerank(
        question="What happened?",
        question_date=None,
        candidates=_memories(6),
    )

    assert decision.selected_session_ids == ["s1", "s2", "s3", "s4", "s5"]
    assert decision.boundary is not None
    assert decision.boundary.policy_rejected is True
    assert decision.boundary.replacement_accepted is False
    assert decision.parse_fallback is False


def test_v531_symbolic_boundary_promotes_without_exposing_session_ids() -> None:
    client = FakeRerankClient(
        [
            json_response(
                selected=["s1", "s2", "s6", "s7"],
                ranked=["s1", "s2", "s6", "s7", "s3"],
            ),
            """
            {
              "evidence_needs": ["first missing fact", "second missing fact"],
              "needs_missing_after_locked": ["first missing fact", "second missing fact"],
              "selected_slots": ["p1", " P2 "],
              "confidence": "high"
            }
            """,
        ]
    )
    reranker = LongMemEvalEvidenceReranker(
        client,
        LongMemEvalRerankerConfig(
            candidate_count=7,
            protected_top_n=3,
            ranked_output_count=7,
            boundary_verification=True,
            boundary_prompt_version=LONGMEMEVAL_SYMBOLIC_BOUNDARY_PROMPT_VERSION,
        ),
    )

    decision = reranker.rerank(
        question="Which two facts jointly answer the question?",
        question_date="2024-02-01",
        candidates=_memories(7),
    )

    assert decision.selected_session_ids == ["s1", "s2", "s3", "s6", "s7"]
    assert decision.boundary is not None
    assert decision.boundary.prompt_version == LONGMEMEVAL_SYMBOLIC_BOUNDARY_PROMPT_VERSION
    assert decision.boundary.raw_selected_slot_labels == ["p1", "P2"]
    assert decision.boundary.selected_slot_labels == ["P1", "P2"]
    assert decision.boundary.slot_session_ids == {
        "B1": "s4",
        "B2": "s5",
        "P1": "s6",
        "P2": "s7",
    }
    assert decision.boundary.decision == "replace_two"
    assert decision.boundary.replacement_accepted is True
    assert decision.boundary.parse_fallback is False
    prompt = client.all_messages[1][1].content
    assert "session_id=" not in prompt
    assert "[LOCKED-1 |" in prompt
    assert "[B1 |" in prompt
    assert "[P1 |" in prompt
    assert '"selected_slots"' in prompt


def test_v531_symbolic_boundary_fails_closed_on_locked_label() -> None:
    client = FakeRerankClient(
        [
            json_response(
                selected=["s1", "s2", "s6", "s7"],
                ranked=["s1", "s2", "s6", "s7", "s3"],
            ),
            """
            {
              "evidence_needs": ["missing fact"],
              "needs_missing_after_locked": ["missing fact"],
              "selected_slots": ["LOCKED-1", "P1"],
              "confidence": "high"
            }
            """,
        ]
    )
    reranker = LongMemEvalEvidenceReranker(
        client,
        LongMemEvalRerankerConfig(
            candidate_count=7,
            protected_top_n=3,
            ranked_output_count=7,
            boundary_verification=True,
            boundary_prompt_version=LONGMEMEVAL_SYMBOLIC_BOUNDARY_PROMPT_VERSION,
        ),
    )

    decision = reranker.rerank(
        question="What happened?",
        question_date=None,
        candidates=_memories(7),
    )

    assert decision.selected_session_ids == ["s1", "s2", "s3", "s4", "s5"]
    assert decision.boundary is not None
    assert decision.boundary.invalid_slot_labels == ["LOCKED-1"]
    assert decision.boundary.parse_fallback is True
    assert decision.boundary.replacement_accepted is False


def test_v532_atomic_boundary_accepts_only_quote_grounded_promotion() -> None:
    client = FakeRerankClient(
        [
            json_response(
                selected=["s1", "s2", "s6", "s3", "s4"],
                ranked=["s1", "s2", "s6", "s3", "s4", "s5"],
            ),
            """
            {
              "evidence_needs": ["N1: missing preference"],
              "needs_missing_after_locked": ["N1"],
              "slot_assessments": [
                {
                  "slot": "P1",
                  "supports_needs": ["N1"],
                  "evidence_quote": "Session 6 discusses a preference.",
                  "adds_missing_evidence": true
                }
              ],
              "selected_slots": ["B1", "P1"],
              "confidence": "high"
            }
            """,
        ]
    )
    reranker = LongMemEvalEvidenceReranker(
        client,
        LongMemEvalRerankerConfig(
            candidate_count=6,
            protected_top_n=3,
            ranked_output_count=6,
            boundary_verification=True,
            boundary_prompt_version=LONGMEMEVAL_ATOMIC_BOUNDARY_PROMPT_VERSION,
        ),
    )

    decision = reranker.rerank(
        question="Which missing preference is required?",
        question_date="2024-02-01",
        candidates=_memories(6),
    )

    assert decision.selected_session_ids == ["s1", "s2", "s3", "s4", "s6"]
    assert decision.boundary is not None
    assert decision.boundary.replacement_accepted is True
    assert decision.boundary.policy_rejected is False
    assert decision.boundary.atomic_support_failures == []
    assert decision.boundary.slot_assessments == [
        {
            "slot": "P1",
            "supports_needs": ["N1"],
            "evidence_quote": "Session 6 discusses a preference.",
            "adds_missing_evidence": True,
            "quote_valid": True,
        }
    ]
    prompt = client.all_messages[1][1].content
    assert "session_id=" not in prompt
    assert '"slot_assessments"' in prompt
    assert "verbatim" in prompt.casefold()
    assert "complete Top-5" in prompt


def test_v532_atomic_boundary_rejects_hallucinated_promotion_quote() -> None:
    client = FakeRerankClient(
        [
            json_response(
                selected=["s1", "s2", "s6", "s3", "s4"],
                ranked=["s1", "s2", "s6", "s3", "s4", "s5"],
            ),
            """
            {
              "evidence_needs": ["N1: missing preference"],
              "needs_missing_after_locked": ["N1"],
              "slot_assessments": [
                {
                  "slot": "P1",
                  "supports_needs": ["N1"],
                  "evidence_quote": "This quote does not occur in the candidate.",
                  "adds_missing_evidence": true
                }
              ],
              "selected_slots": ["B1", "P1"],
              "confidence": "high"
            }
            """,
        ]
    )
    reranker = LongMemEvalEvidenceReranker(
        client,
        LongMemEvalRerankerConfig(
            candidate_count=6,
            protected_top_n=3,
            ranked_output_count=6,
            boundary_verification=True,
            boundary_prompt_version=LONGMEMEVAL_ATOMIC_BOUNDARY_PROMPT_VERSION,
        ),
    )

    decision = reranker.rerank(
        question="Which missing preference is required?",
        question_date="2024-02-01",
        candidates=_memories(6),
    )

    assert decision.selected_session_ids == ["s1", "s2", "s3", "s4", "s5"]
    assert decision.boundary is not None
    assert decision.boundary.replacement_accepted is False
    assert decision.boundary.policy_rejected is True
    assert decision.boundary.atomic_support_failures == ["P1:quote_not_grounded"]
    assert decision.boundary.fallback_reason == (
        "atomic promotion evidence failed validation: P1:quote_not_grounded"
    )


def json_response(*, selected: list[str], ranked: list[str]) -> str:
    import json

    return json.dumps(
        {
            "evidence_needs": ["evidence"],
            "selected_session_ids": selected,
            "ranked_session_ids": ranked,
        }
    )


def _memories(count: int) -> list[RetrievedMemory]:
    return [
        RetrievedMemory(
            memory_id=f"s{index}",
            source_session_id=f"s{index}",
            source_date=f"2024-01-{index:02d}",
            content=(
                f"user: Session {index} discusses a preference.\n"
                f"assistant: Session {index} supporting evidence."
            ),
            score=1.0 / index,
            token_count=20,
        )
        for index in range(1, count + 1)
    ]
