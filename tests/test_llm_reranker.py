"""Network-free tests for the shared LongMemEval evidence reranker."""

from __future__ import annotations

from typing import Any

import pytest

from vmp_memos.frameworks import RetrievedMemory
from vmp_memos.llm import (
    LONGMEMEVAL_ATOMIC_BOUNDARY_PROMPT_VERSION,
    LONGMEMEVAL_ROLE_AWARE_EXCERPT_V3_VERSION,
    LONGMEMEVAL_ROLE_AWARE_EXCERPT_VERSION,
    LONGMEMEVAL_SYMBOLIC_BOUNDARY_PROMPT_VERSION,
    LONGMEMEVAL_V6_ATOMIC_FACT_SELECTOR_PROMPT_VERSION,
    LONGMEMEVAL_V6_SET_COVERAGE_BOUNDARY_VERSION,
    LONGMEMEVAL_V55_DUAL_VIEW_CANDIDATE_PLANNER_VERSION,
    LONGMEMEVAL_V62_ATOMIC_FACT_SELECTOR_PROMPT_VERSION,
    LONGMEMEVAL_V62_SET_COVERAGE_BOUNDARY_VERSION,
    LONGMEMEVAL_V551_COMPLETE_CHALLENGER_SELECTOR_PROMPT_VERSION,
    LONGMEMEVAL_V552_PAIRWISE_BOUNDARY_PROMPT_VERSION,
    LONGMEMEVAL_V552_PAIRWISE_SELECTOR_PROMPT_VERSION,
    LLMGenerationConfig,
    LLMResponse,
    LongMemEvalEvidenceReranker,
    LongMemEvalRerankerConfig,
    candidate_excerpt,
    guarded_session_ranking,
    prepare_longmemeval_rerank_candidates,
)

V54_SELECTOR_PROMPT_VERSION = "vmp_v54_symbolic_span_selector_v1"
V54_BOUNDARY_PROMPT_VERSION = "vmp_v54_symbolic_span_boundary_v1"


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


def test_v54_selector_and_boundary_versions_must_be_enabled_together() -> None:
    with pytest.raises(ValueError, match="must be enabled together"):
        LongMemEvalRerankerConfig(
            prompt_version=V54_SELECTOR_PROMPT_VERSION,
            boundary_verification=True,
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


def test_v54_symbolic_selector_and_boundary_use_grounded_span_ids() -> None:
    client = FakeRerankClient(
        [
            """
            {
              "evidence_needs": ["N1: missing preference"],
              "evidence_selections": [
                {
                  "supports_needs": ["N1"],
                  "evidence_spans": ["C06:S01"]
                }
              ],
              "selected_candidates": ["C01", "C02", "C03", "C06", "C04"],
              "ranked_candidates": ["C01", "C02", "C03", "C06", "C04", "C05"]
            }
            """,
            """
            {
              "evidence_needs": ["N1: missing preference"],
              "needs_missing_after_locked": ["N1"],
              "slot_assessments": [
                {
                  "slot": "P1",
                  "supports_needs": ["N1"],
                  "evidence_spans": ["P1:S01"],
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
            prompt_version=V54_SELECTOR_PROMPT_VERSION,
            candidate_count=6,
            protected_top_n=3,
            ranked_output_count=6,
            boundary_verification=True,
            boundary_prompt_version=V54_BOUNDARY_PROMPT_VERSION,
        ),
    )

    decision = reranker.rerank(
        question="Which missing preference is required?",
        question_date="2024-02-01",
        candidates=_memories(6),
    )

    assert decision.selected_session_ids == ["s1", "s2", "s3", "s4", "s6"]
    assert decision.raw_selected_candidate_labels == [
        "C01",
        "C02",
        "C03",
        "C06",
        "C04",
    ]
    assert decision.invalid_candidate_labels == []
    assert decision.selector_span_binding_failures == []
    assert decision.selector_grounded_promotion_labels == ["C06"]
    assert decision.candidate_label_session_ids["C06"] == "s6"
    assert decision.boundary is not None
    assert decision.boundary.replacement_accepted is True
    assert decision.boundary.atomic_support_failures == []
    assert decision.boundary.slot_assessments == [
        {
            "slot": "P1",
            "supports_needs": ["N1"],
            "evidence_spans": ["P1:S01"],
            "adds_missing_evidence": True,
            "span_valid": True,
        }
    ]
    selector_prompt = client.all_messages[0][1].content
    boundary_prompt = client.all_messages[1][1].content
    assert "session_id=" not in selector_prompt
    assert "s6" not in selector_prompt
    assert "[C06 |" in selector_prompt
    assert "[C06:S01]" in selector_prompt
    assert "[P1:S01]" in boundary_prompt
    assert "evidence_spans" in boundary_prompt


def test_v54_uses_span_owner_instead_of_mismatched_selected_label() -> None:
    client = FakeRerankClient(
        [
            """
            {
              "evidence_needs": ["N1: missing preference"],
              "evidence_selections": [
                {
                  "supports_needs": ["N1"],
                  "evidence_spans": ["C07:S01"]
                }
              ],
              "selected_candidates": ["C01", "C02", "C03", "C06", "C04"],
              "ranked_candidates": ["C01", "C02", "C03", "C06", "C04", "C05", "C07"]
            }
            """,
            """
            {
              "evidence_needs": ["N1: missing preference"],
              "needs_missing_after_locked": ["N1"],
              "slot_assessments": [
                {
                  "slot": "P1",
                  "supports_needs": ["N1"],
                  "evidence_spans": ["P1:S01"],
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
            prompt_version=V54_SELECTOR_PROMPT_VERSION,
            candidate_count=7,
            protected_top_n=3,
            ranked_output_count=7,
            boundary_verification=True,
            boundary_prompt_version=V54_BOUNDARY_PROMPT_VERSION,
        ),
    )

    decision = reranker.rerank(
        question="Which missing preference is required?",
        question_date="2024-02-01",
        candidates=_memories(7),
    )

    assert decision.selector_grounded_promotion_labels == ["C07"]
    assert decision.selected_session_ids == ["s1", "s2", "s3", "s4", "s7"]
    assert "s6" not in decision.selected_session_ids


def test_v54_boundary_rejects_span_owned_by_another_slot() -> None:
    client = FakeRerankClient(
        [
            """
            {
              "evidence_needs": ["N1: missing preference"],
              "evidence_selections": [
                {
                  "supports_needs": ["N1"],
                  "evidence_spans": ["C06:S01"]
                }
              ],
              "selected_candidates": ["C01", "C02", "C03", "C06", "C04"],
              "ranked_candidates": ["C01", "C02", "C03", "C06", "C04", "C05"]
            }
            """,
            """
            {
              "evidence_needs": ["N1: missing preference"],
              "needs_missing_after_locked": ["N1"],
              "slot_assessments": [
                {
                  "slot": "P1",
                  "supports_needs": ["N1"],
                  "evidence_spans": ["B1:S01"],
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
            prompt_version=V54_SELECTOR_PROMPT_VERSION,
            candidate_count=6,
            protected_top_n=3,
            ranked_output_count=6,
            boundary_verification=True,
            boundary_prompt_version=V54_BOUNDARY_PROMPT_VERSION,
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
    assert decision.boundary.atomic_support_failures == ["P1:span_not_grounded"]
    assert decision.boundary.fallback_reason == (
        "symbolic span evidence failed validation: P1:span_not_grounded"
    )


def test_v54_selector_rejects_ungrounded_promotion_span_before_boundary() -> None:
    client = FakeRerankClient(
        """
        {
          "evidence_needs": ["N1: missing preference"],
          "evidence_selections": [
            {
              "supports_needs": ["N1"],
              "evidence_spans": ["C99:S01"]
            }
          ],
          "selected_candidates": ["C01", "C02", "C03", "C06", "C04"],
          "ranked_candidates": ["C01", "C02", "C03", "C06", "C04", "C05"]
        }
        """
    )
    reranker = LongMemEvalEvidenceReranker(
        client,
        LongMemEvalRerankerConfig(
            prompt_version=V54_SELECTOR_PROMPT_VERSION,
            candidate_count=6,
            protected_top_n=3,
            ranked_output_count=6,
            boundary_verification=True,
            boundary_prompt_version=V54_BOUNDARY_PROMPT_VERSION,
        ),
    )

    decision = reranker.rerank(
        question="Which missing preference is required?",
        question_date="2024-02-01",
        candidates=_memories(6),
    )

    assert client.calls == 1
    assert decision.selected_session_ids == ["s1", "s2", "s3", "s4", "s5"]
    assert decision.selector_grounded_promotion_labels == []
    assert decision.selector_span_binding_failures == ["E1:span_not_grounded"]
    assert decision.boundary is not None
    assert decision.boundary.call_made is False


def test_v55_challenger_scan_promotes_grounded_late_candidate() -> None:
    client = FakeRerankClient(
        [
            """
            {
              "evidence_needs": ["N1: missing preference"],
              "challenger_assessments": [
                {"candidate":"C06","supports_needs":[],"evidence_spans":[],"adds_missing_evidence":false},
                {"candidate":"C07","supports_needs":["N1"],"evidence_spans":["C07:S01"],"adds_missing_evidence":true},
                {"candidate":"C08","supports_needs":[],"evidence_spans":[],"adds_missing_evidence":false},
                {"candidate":"C09","supports_needs":[],"evidence_spans":[],"adds_missing_evidence":false},
                {"candidate":"C10","supports_needs":[],"evidence_spans":[],"adds_missing_evidence":false}
              ],
              "selected_candidates": ["C01", "C02", "C03", "C07", "C04"],
              "ranked_candidates": ["C01", "C02", "C03", "C07", "C04", "C05"]
            }
            """,
            """
            {
              "evidence_needs": ["N1: missing preference"],
              "needs_missing_after_locked": ["N1"],
              "slot_assessments": [
                {
                  "slot": "P1",
                  "supports_needs": ["N1"],
                  "evidence_spans": ["P1:S01"],
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
            candidate_planner_version=(
                LONGMEMEVAL_V55_DUAL_VIEW_CANDIDATE_PLANNER_VERSION
            ),
            prompt_version=(
                LONGMEMEVAL_V551_COMPLETE_CHALLENGER_SELECTOR_PROMPT_VERSION
            ),
            candidate_count=10,
            protected_top_n=3,
            ranked_output_count=10,
            boundary_verification=True,
            boundary_prompt_version=V54_BOUNDARY_PROMPT_VERSION,
        ),
    )

    decision = reranker.rerank(
        question="Which missing preference is required?",
        question_date="2024-02-01",
        candidates=_memories(10),
    )

    assert decision.selected_session_ids == ["s1", "s2", "s3", "s4", "s7"]
    assert decision.selector_grounded_promotion_labels == ["C07"]
    assert decision.selector_span_binding_failures == []
    assert decision.boundary is not None
    assert decision.boundary.replacement_accepted is True
    selector_prompt = client.all_messages[0][1].content
    assert "Inspect every challenger" in selector_prompt
    for label in ("C06", "C07", "C08", "C09", "C10"):
        assert f'"candidate":"{label}"' in selector_prompt
    assert "[C06:S01]" in selector_prompt
    assert "[C10:S01]" in selector_prompt
    assert "session_id=" not in selector_prompt


def test_v55_challenger_scan_fails_closed_when_assessment_is_missing() -> None:
    client = FakeRerankClient(
        """
        {
          "evidence_needs": ["N1: missing preference"],
          "challenger_assessments": [
            {"candidate":"C06","supports_needs":[],"evidence_spans":[],"adds_missing_evidence":false},
            {"candidate":"C07","supports_needs":["N1"],"evidence_spans":["C07:S01"],"adds_missing_evidence":true}
          ],
          "selected_candidates": ["C01", "C02", "C03", "C07", "C04"],
          "ranked_candidates": ["C01", "C02", "C03", "C07", "C04", "C05"]
        }
        """
    )
    reranker = LongMemEvalEvidenceReranker(
        client,
        LongMemEvalRerankerConfig(
            candidate_planner_version=(
                LONGMEMEVAL_V55_DUAL_VIEW_CANDIDATE_PLANNER_VERSION
            ),
            prompt_version=(
                LONGMEMEVAL_V551_COMPLETE_CHALLENGER_SELECTOR_PROMPT_VERSION
            ),
            candidate_count=10,
            protected_top_n=3,
            ranked_output_count=10,
            boundary_verification=True,
            boundary_prompt_version=V54_BOUNDARY_PROMPT_VERSION,
        ),
    )

    decision = reranker.rerank(
        question="Which missing preference is required?",
        question_date="2024-02-01",
        candidates=_memories(10),
    )

    assert client.calls == 1
    assert decision.parse_fallback is True
    assert decision.selected_session_ids == ["s1", "s2", "s3", "s4", "s5"]
    assert decision.selector_grounded_promotion_labels == []
    assert "missing_assessments:C08,C09,C10" in decision.selector_span_binding_failures
    assert decision.boundary is not None
    assert decision.boundary.call_made is False


def test_role_aware_excerpt_keeps_concise_user_fact_over_verbose_assistant() -> None:
    verbose = " ".join(["magazine subscription advice"] * 120)
    content = (
        "assistant: " + verbose + "\n"
        "user: Yesterday I attended the Nordstrom friends and family sale.\n"
        "assistant: " + verbose
    )

    excerpt = candidate_excerpt(
        "How many weeks ago was the Nordstrom friends and family sale?",
        content,
        max_chars=500,
        max_turns=4,
        excerpt_version=LONGMEMEVAL_ROLE_AWARE_EXCERPT_VERSION,
    )

    assert "Yesterday I attended the Nordstrom friends and family sale" in excerpt
    assert excerpt.count("magazine subscription advice") < 20


def test_role_aware_v3_excerpt_recovers_publication_subscription_facts() -> None:
    content = (
        "assistant: Do you have suggestions for eco-friendly gifts and styling?\n"
        "user: I finished buying my last National Geographic issue on March 15th.\n"
        "assistant: Do you have ideas for a garden trellis or wall color?\n"
        "user: By the way, I'm also getting Architectural Digest.\n"
        "assistant: Here are several unrelated decorating suggestions."
    )

    excerpt = candidate_excerpt(
        "How many magazine subscriptions do I currently have?",
        content,
        max_chars=500,
        max_turns=4,
        excerpt_version="role_aware_fact_v3",
    )

    assert "National Geographic issue" in excerpt
    assert "Architectural Digest" in excerpt


def test_v552_pairwise_selector_combines_two_grounded_challengers() -> None:
    reject = """
    {
      "decision":"reject",
      "evidence_needs":["N1: first fact","N2: second fact"],
      "supports_needs":[],
      "challenger_spans":[],
      "displaced_slot":null,
      "adds_missing_evidence":false,
      "displaced_slot_redundant":false,
      "confidence":"high"
    }
    """
    replace_b1 = """
    {
      "decision":"replace_B1",
      "evidence_needs":["N1: first fact","N2: second fact"],
      "supports_needs":["N1"],
      "challenger_spans":["X:S01"],
      "displaced_slot":"B1",
      "adds_missing_evidence":true,
      "displaced_slot_redundant":true,
      "confidence":"high"
    }
    """
    replace_b2 = """
    {
      "decision":"replace_B2",
      "evidence_needs":["N1: first fact","N2: second fact"],
      "supports_needs":["N2"],
      "challenger_spans":["X:S01"],
      "displaced_slot":"B2",
      "adds_missing_evidence":true,
      "displaced_slot_redundant":true,
      "confidence":"high"
    }
    """
    client = FakeRerankClient([reject, replace_b1, reject, reject, replace_b2])
    reranker = LongMemEvalEvidenceReranker(
        client,
        LongMemEvalRerankerConfig(
            candidate_planner_version=(
                LONGMEMEVAL_V55_DUAL_VIEW_CANDIDATE_PLANNER_VERSION
            ),
            prompt_version=LONGMEMEVAL_V552_PAIRWISE_SELECTOR_PROMPT_VERSION,
            candidate_excerpt_version=LONGMEMEVAL_ROLE_AWARE_EXCERPT_VERSION,
            candidate_count=10,
            protected_top_n=3,
            ranked_output_count=10,
            boundary_verification=True,
            boundary_prompt_version=LONGMEMEVAL_V552_PAIRWISE_BOUNDARY_PROMPT_VERSION,
        ),
    )

    decision = reranker.rerank(
        question="Which two missing facts are required?",
        question_date="2024-02-01",
        candidates=_memories(10),
    )

    assert client.calls == 5
    assert decision.selector_call_count == 5
    assert decision.selector_call_fallbacks == 0
    assert decision.selected_session_ids == ["s1", "s2", "s3", "s7", "s10"]
    assert decision.selector_grounded_promotion_labels == ["C07", "C10"]
    assert decision.boundary is not None
    assert decision.boundary.replacement_accepted is True
    assert decision.boundary.proposed_promotion_session_ids == ["s7", "s10"]
    assert [item["slot"] for item in decision.boundary.slot_assessments] == [
        "P1",
        "P2",
    ]
    assert all(
        item["span_valid"] is True for item in decision.boundary.slot_assessments
    )
    for messages in client.all_messages:
        prompt = messages[1].content
        assert "[X |" in prompt
        assert "C06" not in prompt
        assert "C07" not in prompt
        assert '"decision":"replace_B1"' not in prompt


def test_v552_pairwise_selector_rejects_invalid_owned_spans_per_candidate() -> None:
    invalid = """
    {
      "decision":"replace_B1",
      "evidence_needs":["N1: missing fact"],
      "supports_needs":["N1"],
      "challenger_spans":["B1:S01"],
      "displaced_slot":"B1",
      "adds_missing_evidence":true,
      "displaced_slot_redundant":true,
      "confidence":"high"
    }
    """
    client = FakeRerankClient([invalid] * 5)
    reranker = LongMemEvalEvidenceReranker(
        client,
        LongMemEvalRerankerConfig(
            candidate_planner_version=(
                LONGMEMEVAL_V55_DUAL_VIEW_CANDIDATE_PLANNER_VERSION
            ),
            prompt_version=LONGMEMEVAL_V552_PAIRWISE_SELECTOR_PROMPT_VERSION,
            candidate_excerpt_version=LONGMEMEVAL_ROLE_AWARE_EXCERPT_VERSION,
            candidate_count=10,
            protected_top_n=3,
            ranked_output_count=10,
            boundary_verification=True,
            boundary_prompt_version=LONGMEMEVAL_V552_PAIRWISE_BOUNDARY_PROMPT_VERSION,
        ),
    )

    decision = reranker.rerank(
        question="Which missing fact is required?",
        question_date="2024-02-01",
        candidates=_memories(10),
    )

    assert client.calls == 5
    assert decision.selected_session_ids == ["s1", "s2", "s3", "s4", "s5"]
    assert decision.selector_grounded_promotion_labels == []
    assert len(decision.selector_span_binding_failures) == 5
    assert decision.parse_fallback is False


def test_v6_atomic_fact_coverage_combines_two_challengers() -> None:
    empty = '{"candidate_relevant":false,"facts":[]}'
    national_geographic = """
    {
      "candidate_relevant": true,
      "facts": [{
        "entity": "National Geographic",
        "relation": "has_subscription",
        "value": "active",
        "temporal_anchor": "current",
        "supports_needs": ["N1", "N2"],
        "evidence_spans": ["[X:S01] user stated the publication"],
        "confidence": "high"
      }]
    }
    """
    architectural_digest = """
    {
      "candidate_relevant": true,
      "facts": [{
        "entity": "Architectural Digest",
        "relation": "has_subscription",
        "value": "active",
        "temporal_anchor": "current",
        "supports_needs": ["N1", "N2"],
        "evidence_spans": ["X:S01"],
        "confidence": "high"
      }]
    }
    """
    responses = [empty] * 6 + [national_geographic] + [empty] * 2 + [
        architectural_digest
    ]
    client = FakeRerankClient(responses)
    reranker = LongMemEvalEvidenceReranker(
        client,
        LongMemEvalRerankerConfig(
            candidate_planner_version=(
                LONGMEMEVAL_V55_DUAL_VIEW_CANDIDATE_PLANNER_VERSION
            ),
            prompt_version=LONGMEMEVAL_V6_ATOMIC_FACT_SELECTOR_PROMPT_VERSION,
            candidate_excerpt_version=LONGMEMEVAL_ROLE_AWARE_EXCERPT_VERSION,
            candidate_count=10,
            protected_top_n=3,
            ranked_output_count=10,
            boundary_verification=True,
            boundary_prompt_version=LONGMEMEVAL_V6_SET_COVERAGE_BOUNDARY_VERSION,
            coverage_min_gain=0.1,
        ),
    )

    decision = reranker.rerank(
        question="How many magazine subscriptions do I currently have?",
        question_date="2024-02-01",
        candidates=_memories(10),
    )

    assert client.calls == 10
    assert decision.selector_call_count == 10
    assert decision.selector_call_fallbacks == 0
    assert decision.selected_session_ids == ["s1", "s2", "s3", "s7", "s10"]
    assert decision.selector_grounded_promotion_labels == ["C07", "C10"]
    assert decision.question_evidence_plan is not None
    assert decision.question_evidence_plan.operator == "count"
    assert decision.coverage_selection is not None
    assert decision.coverage_selection.gain > 0.1
    assert decision.boundary is not None
    assert decision.boundary.replacement_accepted is True
    assert decision.boundary.proposed_promotion_session_ids == ["s7", "s10"]
    for messages in client.all_messages:
        prompt = messages[1].content
        assert "[X |" in prompt
        assert "C01" not in prompt
        assert "C07" not in prompt
        assert "replace_B1" not in prompt


def test_v62_extracts_partial_fact_with_scalar_compatibility() -> None:
    empty = '{"candidate_relevant":false,"facts":[]}'
    partial_museum = """
    {
      "candidate_relevant": true,
      "facts": [{
        "entity": "History Museum lecture",
        "relation": "visited",
        "value": "attended a lecture at the History Museum",
        "temporal_anchor": null,
        "supports_needs": "N1",
        "evidence_spans": "X:S01",
        "confidence": "high"
      }]
    }
    """
    client = FakeRerankClient([empty] * 7 + [partial_museum] + [empty] * 2)
    reranker = LongMemEvalEvidenceReranker(
        client,
        LongMemEvalRerankerConfig(
            candidate_planner_version=(
                LONGMEMEVAL_V55_DUAL_VIEW_CANDIDATE_PLANNER_VERSION
            ),
            prompt_version=LONGMEMEVAL_V62_ATOMIC_FACT_SELECTOR_PROMPT_VERSION,
            candidate_excerpt_version=LONGMEMEVAL_ROLE_AWARE_EXCERPT_V3_VERSION,
            candidate_count=10,
            protected_top_n=3,
            ranked_output_count=10,
            boundary_verification=True,
            boundary_prompt_version=LONGMEMEVAL_V62_SET_COVERAGE_BOUNDARY_VERSION,
            coverage_min_gain=0.1,
        ),
    )

    decision = reranker.rerank(
        question="Which museum did I visit with a friend?",
        question_date="2024-02-01",
        candidates=_memories(10),
    )

    assert client.calls == 10
    assert "s8" in decision.selected_session_ids
    assert decision.selector_grounded_promotion_labels == ["C08"]
    assert all("partial" in messages[0].content for messages in client.all_messages)
    assert all(
        "does not need to satisfy every qualifier" in messages[1].content
        for messages in client.all_messages
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
