"""Shared local-vLLM evidence-set reranker for LongMemEval.

The reranker is deliberately framework-agnostic. Every compared memory method
supplies the same number of ``RetrievedMemory`` candidates and receives the
same prompt, generation settings, parser, and guarded Top-k policy.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from time import perf_counter
from typing import Protocol, cast

from pydantic import Field, JsonValue, PositiveInt, model_validator

from vmp_memos.frameworks import RetrievedMemory
from vmp_memos.frameworks.text import estimate_tokens, terms
from vmp_memos.llm.base import ChatMessage, LLMGenerationConfig, LLMResponse
from vmp_memos.llm.candidate_planner import (
    LONGMEMEVAL_CANDIDATE_PLANNER_VERSIONS,
    LONGMEMEVAL_IDENTITY_CANDIDATE_PLANNER_VERSION,
    LONGMEMEVAL_V55_DUAL_VIEW_CANDIDATE_PLANNER_VERSION,
)
from vmp_memos.schemas.base import (
    NonEmptyStr,
    NonNegativeFloat,
    NonNegativeInt,
    SchemaModel,
)

LONGMEMEVAL_RERANK_PROMPT_VERSION = "vmp_v52_evidence_set_v1"
LONGMEMEVAL_SYMBOLIC_SPAN_SELECTOR_PROMPT_VERSION = (
    "vmp_v54_symbolic_span_selector_v1"
)
LONGMEMEVAL_V55_CHALLENGER_SELECTOR_PROMPT_VERSION = (
    "vmp_v55_challenger_span_selector_v1"
)
LONGMEMEVAL_RERANK_PROMPT_VERSIONS = frozenset(
    {
        LONGMEMEVAL_RERANK_PROMPT_VERSION,
        LONGMEMEVAL_SYMBOLIC_SPAN_SELECTOR_PROMPT_VERSION,
        LONGMEMEVAL_V55_CHALLENGER_SELECTOR_PROMPT_VERSION,
    }
)
LONGMEMEVAL_BOUNDARY_PROMPT_VERSION = "vmp_v53_selective_boundary_v1"
LONGMEMEVAL_SYMBOLIC_BOUNDARY_PROMPT_VERSION = "vmp_v531_symbolic_boundary_v1"
LONGMEMEVAL_ATOMIC_BOUNDARY_PROMPT_VERSION = "vmp_v532_atomic_set_boundary_v1"
LONGMEMEVAL_SYMBOLIC_SPAN_BOUNDARY_PROMPT_VERSION = (
    "vmp_v54_symbolic_span_boundary_v1"
)
LONGMEMEVAL_BOUNDARY_PROMPT_VERSIONS = frozenset(
    {
        LONGMEMEVAL_BOUNDARY_PROMPT_VERSION,
        LONGMEMEVAL_SYMBOLIC_BOUNDARY_PROMPT_VERSION,
        LONGMEMEVAL_ATOMIC_BOUNDARY_PROMPT_VERSION,
        LONGMEMEVAL_SYMBOLIC_SPAN_BOUNDARY_PROMPT_VERSION,
    }
)
LONGMEMEVAL_RERANK_SYSTEM_PROMPT = (
    "You rank long-term memory evidence. Do not answer the question. "
    "Return only the requested JSON object."
)
LONGMEMEVAL_RERANK_USER_PROMPT = """\
Select a joint set of memory sessions that would be sufficient to answer the
question. Candidate session IDs are opaque identifiers.

Question date:
{question_date}

Question:
{question}

Candidate sessions:
{candidate_context}

Instructions:
- First decompose the question into at most four atomic evidence needs.
- For temporal or knowledge-update questions, reason over session dates and
  prefer the latest valid state; retain earlier evidence only when the sequence
  itself is needed.
- For multi-session questions, select sessions that jointly cover every
  evidence need instead of repeating the same fact.
- Never invent a session ID and never answer the question.
- Return exactly one JSON object with this shape:
  {{"evidence_needs":["..."],"selected_session_ids":["id1", "..."],
    "ranked_session_ids":["id1", "..."]}}
- selected_session_ids must contain the best {output_top_k} distinct candidates.
- ranked_session_ids may contain up to {ranked_output_count} distinct candidates.
"""
LONGMEMEVAL_SYMBOLIC_SPAN_SELECTOR_USER_PROMPT = """\
Select a joint set of long-term-memory evidence sufficient to answer the
question. Candidate labels and evidence-span labels are opaque identifiers.

Question date:
{question_date}

Question:
{question}

Candidate sessions:
{candidate_context}

Evidence-first procedure:
1. Decompose the question into at most four atomic needs, named N1 through N4.
2. Select the best {output_top_k} distinct candidate labels. The first
   {protected_top_n} candidates are protected by the server, but you must still
   return a complete selection.
3. For every selected candidate outside the original Top-5 (C01 through C05),
   add one evidence_selections entry containing the exact evidence-span labels
   that support its missing need. Do not copy or paraphrase evidence text.
4. For temporal and knowledge-update questions, prefer the latest valid state
   while retaining earlier dated states only when the sequence is required.
5. For multi-session questions, cover distinct evidence needs rather than
   selecting repeated versions of the same fact.

Never output a session ID or answer the question. Return one JSON object:
  {{"evidence_needs":["N1: ..."],
    "evidence_selections":[
      {{"supports_needs":["N1"],"evidence_spans":["C06:S02"]}}
    ],
    "selected_candidates":["C01","C02","C03","C04","C05"],
    "ranked_candidates":["C01","C02","C03","C04","C05"]}}

selected_candidates must contain exactly {output_top_k} distinct labels.
ranked_candidates may contain up to {ranked_output_count} distinct labels.
Evidence-span ownership is checked locally; an out-of-Top-5 candidate without
a valid owned evidence span cannot be promoted.
"""
LONGMEMEVAL_V55_CHALLENGER_SELECTOR_USER_PROMPT = """\
Audit whether any challenger supplies essential long-term-memory evidence
missing from the original Top-5. Candidate and evidence-span labels are opaque.

Question date:
{question_date}

Question:
{question}

Original Top-5:
{original_context}

Challengers:
{challenger_context}

Evidence-first procedure:
1. Decompose the question into at most four atomic needs, named N1 through N4.
2. Inspect every challenger from C06 through C10. Return exactly one
   challenger_assessments entry for each, in label order.
3. When a challenger adds essential missing evidence, set
   adds_missing_evidence=true and return one or more exact span labels owned by
   that challenger. Otherwise return empty supports_needs and evidence_spans.
4. Counting, list, duration, and ordering questions may need several sessions.
   Do not treat the first matching session as complete coverage.
5. For relative-time questions, use the question date and session dates.
6. The first {protected_top_n} candidates are protected by the server. Return a
   complete Top-{output_top_k} selection, but do not invent evidence.

Never output a session ID or answer the question. Return one JSON object:
  {{"evidence_needs":["N1: ..."],
    "challenger_assessments":[
      {{"candidate":"C06","supports_needs":[],
        "evidence_spans":[],"adds_missing_evidence":false}},
      {{"candidate":"C07","supports_needs":["N1"],
        "evidence_spans":["C07:S02"],"adds_missing_evidence":true}}
    ],
    "selected_candidates":["C01","C02","C03","C04","C05"],
    "ranked_candidates":["C01","C02","C03","C04","C05"]}}

selected_candidates must contain exactly {output_top_k} distinct labels.
ranked_candidates may contain up to {ranked_output_count} distinct labels.
Evidence-span ownership and complete challenger coverage are checked locally.
"""
LONGMEMEVAL_BOUNDARY_SYSTEM_PROMPT = (
    "You conservatively verify the boundary of a long-term memory evidence set. "
    "Do not answer the question. Return only the requested JSON object."
)
LONGMEMEVAL_BOUNDARY_USER_PROMPT = """\
Audit whether proposed memory sessions should replace either of the two original
boundary sessions in a Top-5 evidence set.

Question date:
{question_date}

Question:
{question}

The first {protected_top_n} protected sessions are always retained:
{protected_context}

Original boundary sessions:
{original_boundary_context}

Proposed promotion sessions:
{promotion_context}

Instructions:
- Choose exactly {open_slots} distinct IDs from the original-boundary and
  proposed-promotion sessions. Together with the protected sessions, they must
  form the most complete evidence set for answering the question.
- Default to the original boundary sessions. Promote a new session only when it
  adds essential evidence missing from the original Top-5.
- Do not replace a session merely because another candidate repeats the same
  fact more fluently.
- Multi-session questions may require two complementary promotions. Temporal
  and knowledge-update questions must preserve the dates/states needed to infer
  the requested sequence or latest valid fact.
- Confidence must be "high" only when every replacement is directly supported
  by the candidate excerpts. Otherwise keep the originals and use "low".
- Never invent a session ID and never answer the question.
- Return exactly one JSON object:
  {{"evidence_needs":["..."],
    "selected_boundary_session_ids":["id1","id2"],
    "decision":"keep|replace_one|replace_two",
    "confidence":"high|low"}}
"""
LONGMEMEVAL_SYMBOLIC_BOUNDARY_USER_PROMPT = """\
Choose the two open slots of a Top-5 long-term-memory evidence set. The first
{protected_top_n} sessions are locked and will always remain in the final set.

Question date:
{question_date}

Question:
{question}

Locked evidence (context only; LOCKED labels are not selectable):
{protected_context}

Original open-slot options:
{original_boundary_context}

Promotion options:
{promotion_context}

Selection procedure:
1. Decompose the question into at most four atomic evidence needs.
2. Identify which needs are already covered by the locked evidence.
3. Choose exactly {open_slots} distinct labels from the selectable labels
   {selectable_labels}. The locked evidence plus those choices must cover as
   many different evidence needs as possible.
4. Select a P option when it supplies essential evidence that is missing from
   the locked evidence and the retained B options. Multi-session questions may
   require both P options when they supply two complementary missing facts.
5. For temporal and knowledge-update questions, preserve all dated states
   needed to infer the sequence or latest valid fact.
6. If a P option only repeats evidence already covered, keep the B options.
7. Use confidence "high" only when every selected P option is directly and
   unambiguously required. Low-confidence promotions will be rejected.

The labels are opaque. Never output a LOCKED label, a session ID, or an answer
to the question. Return exactly one JSON object:
  {{"evidence_needs":["..."],
    "needs_missing_after_locked":["..."],
    "selected_slots":["B1","B2"],
    "confidence":"high|low"}}
"""
LONGMEMEVAL_ATOMIC_BOUNDARY_USER_PROMPT = """\
Audit the complete Top-5 long-term-memory evidence set. The first
{protected_top_n} sessions are locked. You must choose the remaining
{open_slots} slots from the original B options and selector-proposed P options.

Question date:
{question_date}

Question:
{question}

Locked evidence already in both candidate sets (LOCKED labels are not selectable):
{protected_context}

Original Top-5 open slots:
{original_boundary_context}

Selector-proposed alternatives:
{promotion_context}

Evidence-first procedure:
1. Decompose the question into at most four atomic needs, named N1 through N4.
2. Read the complete locked+B and locked+P evidence sets. Do not judge a slot
   from topic similarity alone.
3. For every P slot you select, copy one short verbatim quote from that slot's
   excerpt. The quote must directly support a missing need and must not be
   invented or paraphrased.
4. Keep B by default. Select P only when its quoted fact or dated state adds
   essential evidence not supplied by the locked evidence and retained B slots.
5. Multi-session counting/list questions require every distinct contributing
   fact. Temporal questions require the dated states needed for comparison.
6. Use confidence "high" only when every selected P has a directly grounded
   quote. Low-confidence changes are rejected.

Labels are opaque. Never output a LOCKED label, a session ID, or a final answer.
Return exactly one JSON object:
  {{"evidence_needs":["N1: ..."],
    "needs_missing_after_locked":["N1"],
    "slot_assessments":[
      {{"slot":"P1","supports_needs":["N1"],
        "evidence_quote":"verbatim text copied from P1",
        "adds_missing_evidence":true}}
    ],
    "selected_slots":["B1","P1"],
    "confidence":"high|low"}}

selected_slots must contain exactly {open_slots} distinct labels from
{selectable_labels}. Include a slot_assessments entry for every selected P slot.
"""
LONGMEMEVAL_SYMBOLIC_SPAN_BOUNDARY_USER_PROMPT = """\
Audit the complete Top-5 long-term-memory evidence set. The first
{protected_top_n} sessions are locked. Choose the remaining {open_slots} slots
from the original B options and selector-proposed P options.

Question date:
{question_date}

Question:
{question}

Locked evidence present in both sets (LOCKED labels are not selectable):
{protected_context}

Original Top-5 open slots:
{original_boundary_context}

Selector-proposed alternatives:
{promotion_context}

Evidence-first procedure:
1. Decompose the question into at most four atomic needs, named N1 through N4.
2. Read the complete locked+B and locked+P evidence sets.
3. For every selected P slot, return one or more exact evidence-span labels
   owned by that P slot. Do not copy evidence text and do not use a span owned
   by another slot.
4. Keep B by default. Select P only when its grounded span adds essential
   evidence missing from the locked evidence and retained B slots.
5. Multi-session questions require complementary facts. Temporal and
   knowledge-update questions require the dated states needed for comparison.
6. Use confidence "high" only when every selected P is grounded by its own
   span. Low-confidence changes are rejected.

Never output a LOCKED label, session ID, or final answer. Return one JSON object:
  {{"evidence_needs":["N1: ..."],
    "needs_missing_after_locked":["N1"],
    "slot_assessments":[
      {{"slot":"P1","supports_needs":["N1"],
        "evidence_spans":["P1:S02"],"adds_missing_evidence":true}}
    ],
    "selected_slots":["B1","P1"],
    "confidence":"high|low"}}

selected_slots must contain exactly {open_slots} distinct labels from
{selectable_labels}. Include a slot_assessments entry for every selected P slot.
"""

_JSON_FENCE_PATTERN = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)


class RerankerChatClient(Protocol):
    """Structural client interface used by vLLM and deterministic test doubles."""

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        generation: LLMGenerationConfig | None = None,
    ) -> LLMResponse:
        """Generate one evidence-ranking response."""


class LongMemEvalRerankerConfig(SchemaModel):
    """Immutable shared reranker settings used by every compared framework."""

    prompt_version: NonEmptyStr = LONGMEMEVAL_RERANK_PROMPT_VERSION
    candidate_planner_version: NonEmptyStr = (
        LONGMEMEVAL_IDENTITY_CANDIDATE_PLANNER_VERSION
    )
    candidate_planner_rrf_k: NonNegativeInt = 60
    candidate_planner_hierarchical_weight: NonNegativeFloat = 0.8
    candidate_count: PositiveInt = 30
    output_top_k: PositiveInt = 5
    protected_top_n: NonNegativeInt = 4
    ranked_output_count: PositiveInt = 10
    max_candidate_chars: PositiveInt = 1200
    max_excerpt_turns: PositiveInt = 4
    boundary_verification: bool = False
    boundary_prompt_version: NonEmptyStr = LONGMEMEVAL_BOUNDARY_PROMPT_VERSION
    boundary_protected_top_n: NonNegativeInt = 3
    boundary_max_promotions: PositiveInt = 2
    boundary_min_confidence: NonEmptyStr = "high"
    generation: LLMGenerationConfig = Field(
        default_factory=lambda: LLMGenerationConfig(
            max_tokens=512,
            temperature=0.0,
            top_p=1.0,
        )
    )
    boundary_generation: LLMGenerationConfig = Field(
        default_factory=lambda: LLMGenerationConfig(
            max_tokens=256,
            temperature=0.0,
            top_p=1.0,
        )
    )

    @model_validator(mode="after")
    def validate_fair_reranking(self) -> LongMemEvalRerankerConfig:
        """Reject settings that break the fixed V5.2 comparison contract."""

        if self.prompt_version not in LONGMEMEVAL_RERANK_PROMPT_VERSIONS:
            raise ValueError("unsupported LongMemEval reranker prompt version")
        if self.candidate_planner_version not in LONGMEMEVAL_CANDIDATE_PLANNER_VERSIONS:
            raise ValueError("unsupported LongMemEval candidate planner version")
        if float(self.candidate_planner_hierarchical_weight) > 1.0:
            raise ValueError("candidate planner hierarchical weight must be in [0, 1]")
        if self.candidate_count < self.output_top_k:
            raise ValueError("candidate_count must be at least output_top_k")
        if self.protected_top_n >= self.output_top_k:
            raise ValueError("V5.2 must leave exactly one or more Top-k slots open")
        if self.ranked_output_count < self.output_top_k:
            raise ValueError("ranked_output_count must be at least output_top_k")
        if self.ranked_output_count > self.candidate_count:
            raise ValueError("ranked_output_count cannot exceed candidate_count")
        if self.boundary_prompt_version not in LONGMEMEVAL_BOUNDARY_PROMPT_VERSIONS:
            raise ValueError("unsupported LongMemEval boundary prompt version")
        symbolic_span_selector = self.prompt_version in {
            LONGMEMEVAL_SYMBOLIC_SPAN_SELECTOR_PROMPT_VERSION,
            LONGMEMEVAL_V55_CHALLENGER_SELECTOR_PROMPT_VERSION,
        }
        symbolic_span_boundary = (
            self.boundary_prompt_version
            == LONGMEMEVAL_SYMBOLIC_SPAN_BOUNDARY_PROMPT_VERSION
        )
        if self.boundary_verification and (
            symbolic_span_selector != symbolic_span_boundary
        ):
            raise ValueError(
                "V5.4 symbolic span selector and boundary versions must be enabled together"
            )
        if self.prompt_version == LONGMEMEVAL_V55_CHALLENGER_SELECTOR_PROMPT_VERSION:
            if (
                self.candidate_planner_version
                != LONGMEMEVAL_V55_DUAL_VIEW_CANDIDATE_PLANNER_VERSION
            ):
                raise ValueError("V5.5 challenger selector requires the dual-view planner")
            if (
                self.candidate_count != 10
                or self.output_top_k != 5
                or self.protected_top_n != 3
                or self.ranked_output_count != 10
            ):
                raise ValueError(
                    "V5.5 paper contract requires candidates=10, top_k=5, "
                    "protected=3, ranked=10"
                )
        if self.boundary_verification:
            if self.boundary_protected_top_n >= self.output_top_k:
                raise ValueError("V5.3 boundary verification must leave open Top-k slots")
            open_slots = self.output_top_k - self.boundary_protected_top_n
            if self.boundary_max_promotions < open_slots:
                raise ValueError("V5.3 boundary_max_promotions must cover every open Top-k slot")
            if self.boundary_min_confidence != "high":
                raise ValueError("V5.3 paper policy requires high-confidence promotion")
        if float(self.generation.temperature) != 0.0:
            raise ValueError("paper reranking requires temperature=0")
        if float(self.generation.top_p) != 1.0:
            raise ValueError("paper reranking requires top_p=1")
        if float(self.boundary_generation.temperature) != 0.0:
            raise ValueError("paper boundary verification requires temperature=0")
        if float(self.boundary_generation.top_p) != 1.0:
            raise ValueError("paper boundary verification requires top_p=1")
        return self


class LongMemEvalBoundaryDecision(SchemaModel):
    """One conservative second-stage boundary decision."""

    call_made: bool = False
    skipped_reason: str | None = None
    prompt_version: NonEmptyStr = LONGMEMEVAL_BOUNDARY_PROMPT_VERSION
    prompt_sha256: str | None = None
    provider: str | None = None
    model: str | None = None
    finish_reason: str | None = None
    evidence_needs: list[str] = Field(default_factory=list)
    needs_missing_after_locked: list[str] = Field(default_factory=list)
    original_boundary_session_ids: list[NonEmptyStr] = Field(default_factory=list)
    proposed_promotion_session_ids: list[NonEmptyStr] = Field(default_factory=list)
    slot_session_ids: dict[NonEmptyStr, NonEmptyStr] = Field(default_factory=dict)
    raw_selected_slot_labels: list[NonEmptyStr] = Field(default_factory=list)
    selected_slot_labels: list[NonEmptyStr] = Field(default_factory=list)
    invalid_slot_labels: list[NonEmptyStr] = Field(default_factory=list)
    raw_selected_boundary_session_ids: list[NonEmptyStr] = Field(default_factory=list)
    selected_boundary_session_ids: list[NonEmptyStr] = Field(default_factory=list)
    invalid_session_ids: list[NonEmptyStr] = Field(default_factory=list)
    slot_assessments: list[dict[str, JsonValue]] = Field(default_factory=list)
    atomic_support_failures: list[NonEmptyStr] = Field(default_factory=list)
    decision: str | None = None
    confidence: str | None = None
    replacement_accepted: bool = False
    parse_fallback: bool = False
    policy_rejected: bool = False
    fallback_reason: str | None = None
    input_tokens: NonNegativeInt = 0
    output_tokens: NonNegativeInt = 0
    usage: dict[str, JsonValue] = Field(default_factory=dict)
    response_text: str = ""
    latency_ms: NonNegativeFloat = 0.0


class LongMemEvalRerankDecision(SchemaModel):
    """One parsed, guarded, and auditable LLM ranking decision."""

    prompt_version: NonEmptyStr
    prompt_sha256: NonEmptyStr
    provider: NonEmptyStr
    model: NonEmptyStr
    finish_reason: str | None = None
    evidence_needs: list[str] = Field(default_factory=list)
    candidate_label_session_ids: dict[NonEmptyStr, NonEmptyStr] = Field(
        default_factory=dict
    )
    raw_selected_candidate_labels: list[NonEmptyStr] = Field(default_factory=list)
    raw_ranked_candidate_labels: list[NonEmptyStr] = Field(default_factory=list)
    invalid_candidate_labels: list[NonEmptyStr] = Field(default_factory=list)
    selector_evidence_selections: list[dict[str, JsonValue]] = Field(
        default_factory=list
    )
    selector_span_binding_failures: list[NonEmptyStr] = Field(default_factory=list)
    selector_grounded_promotion_labels: list[NonEmptyStr] = Field(
        default_factory=list
    )
    raw_selected_session_ids: list[NonEmptyStr] = Field(default_factory=list)
    raw_ranked_session_ids: list[NonEmptyStr] = Field(default_factory=list)
    selector_selected_session_ids: list[NonEmptyStr] = Field(default_factory=list)
    ranked_session_ids: list[NonEmptyStr] = Field(default_factory=list)
    selected_session_ids: list[NonEmptyStr] = Field(default_factory=list)
    invalid_session_ids: list[NonEmptyStr] = Field(default_factory=list)
    parse_fallback: bool = False
    parse_fallback_reason: str | None = None
    input_tokens: NonNegativeInt = 0
    output_tokens: NonNegativeInt = 0
    usage: dict[str, JsonValue] = Field(default_factory=dict)
    response_text: str = ""
    selector_replay: bool = False
    selector_prompt_sha256_match: bool | None = None
    selector_source_prompt_sha256: str | None = None
    selector_current_prompt_sha256: str | None = None
    boundary: LongMemEvalBoundaryDecision | None = None


class LongMemEvalEvidenceReranker:
    """Use one local vLLM prompt to select a guarded evidence set."""

    def __init__(
        self,
        client: RerankerChatClient,
        config: LongMemEvalRerankerConfig | None = None,
    ) -> None:
        self.client = client
        self.config = config or LongMemEvalRerankerConfig()

    def rerank(
        self,
        *,
        question_id: str | None = None,
        source_method: str | None = None,
        question: str,
        question_date: str | None,
        candidates: Sequence[RetrievedMemory],
    ) -> LongMemEvalRerankDecision:
        """Rerank one framework's candidates without observing gold labels."""

        unique_candidates = prepare_longmemeval_rerank_candidates(
            candidates,
            candidate_count=self.config.candidate_count,
        )
        if not unique_candidates:
            raise ValueError("at least one retrieval candidate is required")
        original_ids = [_session_id(memory) for memory in unique_candidates]
        prompt = build_longmemeval_rerank_prompt(
            question=question,
            question_date=question_date,
            candidates=unique_candidates,
            config=self.config,
        )
        replay_context_setter = getattr(
            self.client,
            "set_selector_replay_context",
            None,
        )
        if replay_context_setter is not None:
            if question_id is None or source_method is None:
                raise ValueError("selector replay requires source_method and question_id context")
            replay_context_setter(
                source_method=source_method,
                question_id=question_id,
                question=question,
                question_date=question_date,
                candidate_session_ids=original_ids,
            )
        response = self.client.chat(
            [
                ChatMessage(role="system", content=LONGMEMEVAL_RERANK_SYSTEM_PROMPT),
                ChatMessage(role="user", content=prompt),
            ],
            generation=self.config.generation,
        )
        parsed, fallback_reason = _parse_rerank_response(response.text)
        evidence_needs = _string_list(parsed.get("evidence_needs"))[:4]
        candidate_label_session_ids: dict[str, str] = {}
        raw_selected_candidate_labels: list[str] = []
        raw_ranked_candidate_labels: list[str] = []
        invalid_candidate_labels: list[str] = []
        selector_evidence_selections: list[dict[str, JsonValue]] = []
        selector_span_binding_failures: list[str] = []
        selector_grounded_promotion_labels: list[str] = []
        selector_protocol_fallback = False
        if self.config.prompt_version in {
            LONGMEMEVAL_SYMBOLIC_SPAN_SELECTOR_PROMPT_VERSION,
            LONGMEMEVAL_V55_CHALLENGER_SELECTOR_PROMPT_VERSION,
        }:
            candidate_label_session_ids = _candidate_label_session_map(unique_candidates)
            raw_selected_candidate_labels = [
                _normalized_candidate_label(value)
                for value in _string_list(parsed.get("selected_candidates"))
            ]
            raw_ranked_candidate_labels = [
                _normalized_candidate_label(value)
                for value in _string_list(parsed.get("ranked_candidates"))
            ]
            raw_candidate_labels = _ordered_unique(
                [*raw_selected_candidate_labels, *raw_ranked_candidate_labels]
            )
            invalid_candidate_labels = [
                label
                for label in raw_candidate_labels
                if label not in candidate_label_session_ids
            ]
            if (
                self.config.prompt_version
                == LONGMEMEVAL_V55_CHALLENGER_SELECTOR_PROMPT_VERSION
            ):
                (
                    selector_evidence_selections,
                    selector_span_binding_failures,
                    grounded_labels,
                    challenger_scan_complete,
                ) = _validate_challenger_assessments(
                    parsed,
                    candidates=unique_candidates,
                    question=question,
                    config=self.config,
                )
                selector_protocol_fallback = not challenger_scan_complete
                if selector_protocol_fallback:
                    # Fail closed: a partial challenger scan must not count any
                    # promotion as grounded or influence the final ranking.
                    grounded_labels = []
            else:
                (
                    selector_evidence_selections,
                    selector_span_binding_failures,
                    grounded_labels,
                ) = _validate_selector_evidence_selections(
                    parsed,
                    candidates=unique_candidates,
                    question=question,
                    config=self.config,
                )
            original_top_k_labels = {
                _candidate_label(index)
                for index in range(
                    1,
                    min(self.config.output_top_k, len(unique_candidates)) + 1,
                )
            }
            selector_grounded_promotion_labels = [
                label for label in grounded_labels if label not in original_top_k_labels
            ]
            valid_labels = [
                *selector_grounded_promotion_labels,
                *[
                    label
                    for label in raw_candidate_labels
                    if label in original_top_k_labels
                ],
            ]
            valid_proposed = _ordered_unique(
                candidate_label_session_ids[label]
                for label in valid_labels
                if label in candidate_label_session_ids
            )
            raw_selected = [
                candidate_label_session_ids[label]
                for label in raw_selected_candidate_labels
                if label in candidate_label_session_ids
            ]
            raw_ranked = [
                candidate_label_session_ids[label]
                for label in raw_ranked_candidate_labels
                if label in candidate_label_session_ids
            ]
            invalid_ids: list[str] = []
        else:
            raw_selected = _string_list(parsed.get("selected_session_ids"))
            raw_ranked = _string_list(parsed.get("ranked_session_ids"))
            allowed = set(original_ids)
            proposed = _ordered_unique([*raw_selected, *raw_ranked])
            invalid_ids = [session_id for session_id in proposed if session_id not in allowed]
            valid_proposed = [
                session_id for session_id in proposed if session_id in allowed
            ]
        parse_fallback = selector_protocol_fallback or not valid_proposed
        if parse_fallback:
            valid_proposed = list(original_ids)
            fallback_reason = fallback_reason or (
                "challenger scan did not assess every candidate"
                if selector_protocol_fallback
                else (
                    "response contained no grounded candidate spans"
                    if self.config.prompt_version
                    in {
                        LONGMEMEVAL_SYMBOLIC_SPAN_SELECTOR_PROMPT_VERSION,
                        LONGMEMEVAL_V55_CHALLENGER_SELECTOR_PROMPT_VERSION,
                    }
                    else "response contained no valid candidate IDs"
                )
            )
        complete_llm_order = _ordered_unique([*valid_proposed, *original_ids])
        selector_ranked_ids = guarded_session_ranking(
            original_session_ids=original_ids,
            proposed_session_ids=complete_llm_order,
            output_top_k=self.config.output_top_k,
            protected_top_n=self.config.protected_top_n,
        )
        input_tokens = _usage_tokens(
            response.usage,
            "prompt_tokens",
            fallback=estimate_tokens(LONGMEMEVAL_RERANK_SYSTEM_PROMPT + "\n" + prompt),
        )
        output_tokens = _usage_tokens(
            response.usage,
            "completion_tokens",
            fallback=estimate_tokens(response.text) if response.text else 0,
        )
        boundary: LongMemEvalBoundaryDecision | None = None
        ranked_ids = selector_ranked_ids
        if self.config.boundary_verification:
            boundary, ranked_ids = self._verify_boundary(
                question=question,
                question_date=question_date,
                candidates=unique_candidates,
                valid_proposed_session_ids=[] if parse_fallback else valid_proposed,
                original_session_ids=original_ids,
                selector_response=response,
            )
            input_tokens += boundary.input_tokens
            output_tokens += boundary.output_tokens
            if boundary.parse_fallback:
                parse_fallback = True
                fallback_reason = boundary.fallback_reason or fallback_reason
        return LongMemEvalRerankDecision(
            prompt_version=self.config.prompt_version,
            prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            provider=response.provider,
            model=response.model,
            finish_reason=response.finish_reason,
            evidence_needs=evidence_needs,
            candidate_label_session_ids=candidate_label_session_ids,
            raw_selected_candidate_labels=raw_selected_candidate_labels,
            raw_ranked_candidate_labels=raw_ranked_candidate_labels,
            invalid_candidate_labels=invalid_candidate_labels,
            selector_evidence_selections=selector_evidence_selections,
            selector_span_binding_failures=selector_span_binding_failures,
            selector_grounded_promotion_labels=selector_grounded_promotion_labels,
            raw_selected_session_ids=raw_selected,
            raw_ranked_session_ids=raw_ranked,
            selector_selected_session_ids=selector_ranked_ids[: self.config.output_top_k],
            ranked_session_ids=ranked_ids,
            selected_session_ids=ranked_ids[: self.config.output_top_k],
            invalid_session_ids=invalid_ids,
            parse_fallback=parse_fallback,
            parse_fallback_reason=fallback_reason if parse_fallback else None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usage=response.usage,
            response_text=response.text.strip(),
            selector_replay=response.raw_response.get("selector_replay") is True,
            selector_prompt_sha256_match=_optional_bool(
                response.raw_response.get("prompt_sha256_match")
            ),
            selector_source_prompt_sha256=_optional_string(
                response.raw_response.get("source_prompt_sha256")
            ),
            selector_current_prompt_sha256=_optional_string(
                response.raw_response.get("current_prompt_sha256")
            ),
            boundary=boundary,
        )

    def _verify_boundary(
        self,
        *,
        question: str,
        question_date: str | None,
        candidates: Sequence[RetrievedMemory],
        valid_proposed_session_ids: Sequence[str],
        original_session_ids: Sequence[str],
        selector_response: LLMResponse,
    ) -> tuple[LongMemEvalBoundaryDecision, list[str]]:
        """Conservatively verify at most two promotions around the Top-5 boundary."""

        protected = list(original_session_ids[: self.config.boundary_protected_top_n])
        original_top_k = list(original_session_ids[: self.config.output_top_k])
        original_boundary = original_top_k[self.config.boundary_protected_top_n :]
        promotions = [
            session_id
            for session_id in _ordered_unique(valid_proposed_session_ids)
            if session_id not in original_top_k
        ][: self.config.boundary_max_promotions]
        if not promotions:
            ranked_ids = _ordered_unique([*original_top_k, *original_session_ids])
            return (
                LongMemEvalBoundaryDecision(
                    call_made=False,
                    skipped_reason="selector proposed no out-of-Top-5 candidates",
                    prompt_version=self.config.boundary_prompt_version,
                    original_boundary_session_ids=original_boundary,
                    selected_boundary_session_ids=original_boundary,
                ),
                ranked_ids,
            )

        by_session = {_session_id(memory): memory for memory in candidates}
        prompt = build_longmemeval_boundary_prompt(
            question=question,
            question_date=question_date,
            protected=[by_session[session_id] for session_id in protected],
            original_boundary=[
                by_session[session_id]
                for session_id in original_boundary
                if session_id in by_session
            ],
            promotions=[
                by_session[session_id] for session_id in promotions if session_id in by_session
            ],
            config=self.config,
        )
        started_at = perf_counter()
        response = self.client.chat(
            [
                ChatMessage(role="system", content=LONGMEMEVAL_BOUNDARY_SYSTEM_PROMPT),
                ChatMessage(role="user", content=prompt),
            ],
            generation=self.config.boundary_generation,
        )
        latency_ms = (perf_counter() - started_at) * 1000.0
        if (
            response.provider != selector_response.provider
            or response.model != selector_response.model
        ):
            raise ValueError("selector and boundary verifier must use one provider/model")
        parsed, fallback_reason = _parse_rerank_response(response.text)
        open_slots = self.config.output_top_k - self.config.boundary_protected_top_n
        confidence = _normalized_string(parsed.get("confidence"))
        slot_session_ids = _boundary_slot_map(original_boundary, promotions)
        raw_slot_labels: list[str] = []
        selected_slot_labels: list[str] = []
        invalid_slot_labels: list[str] = []
        slot_assessments: list[dict[str, JsonValue]] = []
        atomic_support_failures: list[str] = []
        decision_name: str | None
        if _uses_symbolic_boundary_slots(self.config.boundary_prompt_version):
            raw_slot_labels = _string_list(parsed.get("selected_slots"))
            normalized_labels = [_normalized_slot_label(label) for label in raw_slot_labels]
            invalid_slot_labels = [
                raw
                for raw, normalized in zip(
                    raw_slot_labels,
                    normalized_labels,
                    strict=True,
                )
                if normalized not in slot_session_ids
            ]
            selected_slot_labels = _ordered_unique(
                label for label in normalized_labels if label in slot_session_ids
            )
            raw_selected = [slot_session_ids[label] for label in selected_slot_labels]
            invalid_ids: list[str] = []
            valid_selected = list(raw_selected)
            parse_fallback = len(selected_slot_labels) != open_slots or bool(invalid_slot_labels)
            decision_name = _boundary_decision_name(
                selected_slot_labels,
            )
        else:
            raw_selected = _string_list(parsed.get("selected_boundary_session_ids"))
            allowed_boundary = set([*original_boundary, *promotions])
            invalid_ids = [
                session_id for session_id in raw_selected if session_id not in allowed_boundary
            ]
            valid_selected = _ordered_unique(
                session_id for session_id in raw_selected if session_id in allowed_boundary
            )
            parse_fallback = len(valid_selected) != open_slots or bool(invalid_ids)
            decision_name = _normalized_string(parsed.get("decision"))
        proposed_change = set(valid_selected) != set(original_boundary)
        if (
            self.config.boundary_prompt_version == LONGMEMEVAL_ATOMIC_BOUNDARY_PROMPT_VERSION
            and not parse_fallback
            and proposed_change
        ):
            slot_assessments, atomic_support_failures = _validate_atomic_slot_assessments(
                parsed,
                selected_slot_labels=selected_slot_labels,
                slot_session_ids=slot_session_ids,
                by_session=by_session,
                question=question,
                config=self.config,
            )
        elif (
            self.config.boundary_prompt_version
            == LONGMEMEVAL_SYMBOLIC_SPAN_BOUNDARY_PROMPT_VERSION
            and not parse_fallback
            and proposed_change
        ):
            slot_assessments, atomic_support_failures = (
                _validate_symbolic_span_slot_assessments(
                    parsed,
                    selected_slot_labels=selected_slot_labels,
                    slot_session_ids=slot_session_ids,
                    by_session=by_session,
                    question=question,
                    config=self.config,
                )
            )
        policy_rejected = (
            not parse_fallback
            and proposed_change
            and (confidence != self.config.boundary_min_confidence or bool(atomic_support_failures))
        )
        if parse_fallback:
            selected_boundary = original_boundary
            fallback_reason = fallback_reason or (
                f"boundary response must select exactly {open_slots} valid distinct "
                + (
                    "slot labels"
                    if _uses_symbolic_boundary_slots(self.config.boundary_prompt_version)
                    else "IDs"
                )
            )
        elif policy_rejected:
            selected_boundary = original_boundary
            if atomic_support_failures:
                validation_name = (
                    "symbolic span evidence"
                    if self.config.boundary_prompt_version
                    == LONGMEMEVAL_SYMBOLIC_SPAN_BOUNDARY_PROMPT_VERSION
                    else "atomic promotion evidence"
                )
                fallback_reason = (
                    f"{validation_name} failed validation: "
                    + ", ".join(atomic_support_failures)
                )
            else:
                fallback_reason = (
                    f"replacement confidence below {self.config.boundary_min_confidence!r}"
                )
        else:
            selected_boundary = valid_selected if proposed_change else original_boundary
        if _uses_symbolic_boundary_slots(self.config.boundary_prompt_version):
            selected_slot_labels = (
                _slot_labels_for_sessions(slot_session_ids, selected_boundary)
                if not parse_fallback and not policy_rejected
                else [f"B{index}" for index in range(1, len(original_boundary) + 1)]
            )
        final_top_k = _ordered_unique([*protected, *selected_boundary])
        ranked_ids = _ordered_unique(
            [
                *final_top_k,
                *valid_proposed_session_ids,
                *original_session_ids,
            ]
        )
        input_tokens = _usage_tokens(
            response.usage,
            "prompt_tokens",
            fallback=estimate_tokens(LONGMEMEVAL_BOUNDARY_SYSTEM_PROMPT + "\n" + prompt),
        )
        output_tokens = _usage_tokens(
            response.usage,
            "completion_tokens",
            fallback=estimate_tokens(response.text) if response.text else 0,
        )
        return (
            LongMemEvalBoundaryDecision(
                call_made=True,
                prompt_version=self.config.boundary_prompt_version,
                prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                provider=response.provider,
                model=response.model,
                finish_reason=response.finish_reason,
                evidence_needs=_string_list(parsed.get("evidence_needs"))[:4],
                needs_missing_after_locked=_string_list(parsed.get("needs_missing_after_locked"))[
                    :4
                ],
                original_boundary_session_ids=original_boundary,
                proposed_promotion_session_ids=promotions,
                slot_session_ids=slot_session_ids,
                raw_selected_slot_labels=raw_slot_labels,
                selected_slot_labels=selected_slot_labels,
                invalid_slot_labels=invalid_slot_labels,
                raw_selected_boundary_session_ids=raw_selected,
                selected_boundary_session_ids=selected_boundary,
                invalid_session_ids=invalid_ids,
                slot_assessments=slot_assessments,
                atomic_support_failures=atomic_support_failures,
                decision=decision_name,
                confidence=confidence,
                replacement_accepted=selected_boundary != original_boundary,
                parse_fallback=parse_fallback,
                policy_rejected=policy_rejected,
                fallback_reason=fallback_reason if parse_fallback or policy_rejected else None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                usage=response.usage,
                response_text=response.text.strip(),
                latency_ms=latency_ms,
            ),
            ranked_ids,
        )


def build_longmemeval_rerank_prompt(
    *,
    question: str,
    question_date: str | None,
    candidates: Sequence[RetrievedMemory],
    config: LongMemEvalRerankerConfig,
) -> str:
    """Render the fixed framework-agnostic evidence-selection prompt."""

    if config.prompt_version == LONGMEMEVAL_V55_CHALLENGER_SELECTOR_PROMPT_VERSION:
        rendered = [
            _format_symbolic_span_candidate(
                rank,
                memory,
                question=question,
                max_chars=config.max_candidate_chars,
                max_turns=config.max_excerpt_turns,
            )
            for rank, memory in enumerate(
                candidates[: config.candidate_count],
                start=1,
            )
        ]
        return LONGMEMEVAL_V55_CHALLENGER_SELECTOR_USER_PROMPT.format(
            question_date=question_date or "unknown",
            question=question,
            original_context="\n\n".join(rendered[: config.output_top_k]),
            challenger_context="\n\n".join(rendered[config.output_top_k :]),
            output_top_k=config.output_top_k,
            protected_top_n=config.protected_top_n,
            ranked_output_count=config.ranked_output_count,
        ).strip()

    if config.prompt_version == LONGMEMEVAL_SYMBOLIC_SPAN_SELECTOR_PROMPT_VERSION:
        candidate_context = "\n\n".join(
            _format_symbolic_span_candidate(
                rank,
                memory,
                question=question,
                max_chars=config.max_candidate_chars,
                max_turns=config.max_excerpt_turns,
            )
            for rank, memory in enumerate(
                candidates[: config.candidate_count],
                start=1,
            )
        )
        return LONGMEMEVAL_SYMBOLIC_SPAN_SELECTOR_USER_PROMPT.format(
            question_date=question_date or "unknown",
            question=question,
            candidate_context=candidate_context,
            output_top_k=config.output_top_k,
            protected_top_n=config.protected_top_n,
            ranked_output_count=config.ranked_output_count,
        ).strip()

    candidate_context = "\n\n".join(
        _format_candidate(
            rank,
            memory,
            question=question,
            max_chars=config.max_candidate_chars,
            max_turns=config.max_excerpt_turns,
        )
        for rank, memory in enumerate(candidates[: config.candidate_count], start=1)
    )
    return LONGMEMEVAL_RERANK_USER_PROMPT.format(
        question_date=question_date or "unknown",
        question=question,
        candidate_context=candidate_context,
        output_top_k=config.output_top_k,
        ranked_output_count=config.ranked_output_count,
    ).strip()


def prepare_longmemeval_rerank_candidates(
    candidates: Sequence[RetrievedMemory],
    *,
    candidate_count: int,
) -> list[RetrievedMemory]:
    """Deduplicate sessions before applying the shared candidate-depth limit."""

    if candidate_count < 1:
        raise ValueError("candidate_count must be at least 1")
    return _unique_memories(candidates)[:candidate_count]


def build_longmemeval_boundary_prompt(
    *,
    question: str,
    question_date: str | None,
    protected: Sequence[RetrievedMemory],
    original_boundary: Sequence[RetrievedMemory],
    promotions: Sequence[RetrievedMemory],
    config: LongMemEvalRerankerConfig,
) -> str:
    """Render one fixed framework-agnostic boundary verification prompt."""

    if (
        config.boundary_prompt_version
        == LONGMEMEVAL_SYMBOLIC_SPAN_BOUNDARY_PROMPT_VERSION
    ):
        return _build_symbolic_span_boundary_prompt(
            question=question,
            question_date=question_date,
            protected=protected,
            original_boundary=original_boundary,
            promotions=promotions,
            config=config,
        )
    if config.boundary_prompt_version == LONGMEMEVAL_ATOMIC_BOUNDARY_PROMPT_VERSION:
        return _build_atomic_boundary_prompt(
            question=question,
            question_date=question_date,
            protected=protected,
            original_boundary=original_boundary,
            promotions=promotions,
            config=config,
        )
    if config.boundary_prompt_version == LONGMEMEVAL_SYMBOLIC_BOUNDARY_PROMPT_VERSION:
        return _build_symbolic_boundary_prompt(
            question=question,
            question_date=question_date,
            protected=protected,
            original_boundary=original_boundary,
            promotions=promotions,
            config=config,
        )

    def context(role: str, memories: Sequence[RetrievedMemory]) -> str:
        if not memories:
            return "(none)"
        blocks: list[str] = []
        for memory in memories:
            excerpt = candidate_excerpt(
                question,
                memory.content,
                max_chars=config.max_candidate_chars,
                max_turns=config.max_excerpt_turns,
            )
            blocks.append(
                f"[{role} | session_id={_session_id(memory)} | "
                f"date={memory.source_date or 'unknown'}]\n{excerpt}"
            )
        return "\n\n".join(blocks)

    return LONGMEMEVAL_BOUNDARY_USER_PROMPT.format(
        question_date=question_date or "unknown",
        question=question,
        protected_top_n=config.boundary_protected_top_n,
        protected_context=context("protected", protected),
        original_boundary_context=context("original-boundary", original_boundary),
        promotion_context=context("proposed-promotion", promotions),
        open_slots=config.output_top_k - config.boundary_protected_top_n,
    ).strip()


def _build_symbolic_boundary_prompt(
    *,
    question: str,
    question_date: str | None,
    protected: Sequence[RetrievedMemory],
    original_boundary: Sequence[RetrievedMemory],
    promotions: Sequence[RetrievedMemory],
    config: LongMemEvalRerankerConfig,
) -> str:
    """Render V5.3.1 without exposing selectable session IDs to the model."""

    def context(
        prefix: str,
        memories: Sequence[RetrievedMemory],
        *,
        description: str,
    ) -> str:
        if not memories:
            return "(none)"
        blocks: list[str] = []
        for index, memory in enumerate(memories, start=1):
            excerpt = candidate_excerpt(
                question,
                memory.content,
                max_chars=config.max_candidate_chars,
                max_turns=config.max_excerpt_turns,
            )
            blocks.append(
                f"[{prefix}{index} | {description} | "
                f"date={memory.source_date or 'unknown'}]\n{excerpt}"
            )
        return "\n\n".join(blocks)

    slot_labels = [
        *[f"B{index}" for index in range(1, len(original_boundary) + 1)],
        *[f"P{index}" for index in range(1, len(promotions) + 1)],
    ]
    return LONGMEMEVAL_SYMBOLIC_BOUNDARY_USER_PROMPT.format(
        question_date=question_date or "unknown",
        question=question,
        protected_top_n=config.boundary_protected_top_n,
        protected_context=context(
            "LOCKED-",
            protected,
            description="already retained",
        ),
        original_boundary_context=context(
            "B",
            original_boundary,
            description="selectable original",
        ),
        promotion_context=context(
            "P",
            promotions,
            description="selectable promotion",
        ),
        open_slots=config.output_top_k - config.boundary_protected_top_n,
        selectable_labels=", ".join(slot_labels),
    ).strip()


def _build_atomic_boundary_prompt(
    *,
    question: str,
    question_date: str | None,
    protected: Sequence[RetrievedMemory],
    original_boundary: Sequence[RetrievedMemory],
    promotions: Sequence[RetrievedMemory],
    config: LongMemEvalRerankerConfig,
) -> str:
    """Render V5.3.2's quote-grounded complete-set boundary audit."""

    def context(
        prefix: str,
        memories: Sequence[RetrievedMemory],
        *,
        description: str,
    ) -> str:
        if not memories:
            return "(none)"
        blocks: list[str] = []
        for index, memory in enumerate(memories, start=1):
            excerpt = candidate_excerpt(
                question,
                memory.content,
                max_chars=config.max_candidate_chars,
                max_turns=config.max_excerpt_turns,
            )
            blocks.append(
                f"[{prefix}{index} | {description} | "
                f"date={memory.source_date or 'unknown'}]\n{excerpt}"
            )
        return "\n\n".join(blocks)

    slot_labels = [
        *[f"B{index}" for index in range(1, len(original_boundary) + 1)],
        *[f"P{index}" for index in range(1, len(promotions) + 1)],
    ]
    return LONGMEMEVAL_ATOMIC_BOUNDARY_USER_PROMPT.format(
        question_date=question_date or "unknown",
        question=question,
        protected_top_n=config.boundary_protected_top_n,
        protected_context=context(
            "LOCKED-",
            protected,
            description="present in every complete set",
        ),
        original_boundary_context=context(
            "B",
            original_boundary,
            description="selectable original",
        ),
        promotion_context=context(
            "P",
            promotions,
            description="selectable promotion",
        ),
        open_slots=config.output_top_k - config.boundary_protected_top_n,
        selectable_labels=", ".join(slot_labels),
    ).strip()


def _build_symbolic_span_boundary_prompt(
    *,
    question: str,
    question_date: str | None,
    protected: Sequence[RetrievedMemory],
    original_boundary: Sequence[RetrievedMemory],
    promotions: Sequence[RetrievedMemory],
    config: LongMemEvalRerankerConfig,
) -> str:
    """Render V5.4 with locally owned evidence-span identifiers."""

    def context(
        prefix: str,
        memories: Sequence[RetrievedMemory],
        *,
        description: str,
    ) -> str:
        if not memories:
            return "(none)"
        return "\n\n".join(
            _format_labeled_span_memory(
                f"{prefix}{index}",
                memory,
                question=question,
                description=description,
                max_chars=config.max_candidate_chars,
                max_turns=config.max_excerpt_turns,
            )
            for index, memory in enumerate(memories, start=1)
        )

    slot_labels = [
        *[f"B{index}" for index in range(1, len(original_boundary) + 1)],
        *[f"P{index}" for index in range(1, len(promotions) + 1)],
    ]
    return LONGMEMEVAL_SYMBOLIC_SPAN_BOUNDARY_USER_PROMPT.format(
        question_date=question_date or "unknown",
        question=question,
        protected_top_n=config.boundary_protected_top_n,
        protected_context=context(
            "LOCKED-",
            protected,
            description="present in every complete set",
        ),
        original_boundary_context=context(
            "B",
            original_boundary,
            description="selectable original",
        ),
        promotion_context=context(
            "P",
            promotions,
            description="selectable promotion",
        ),
        open_slots=config.output_top_k - config.boundary_protected_top_n,
        selectable_labels=", ".join(slot_labels),
    ).strip()


def _boundary_slot_map(
    original_boundary: Sequence[str],
    promotions: Sequence[str],
) -> dict[str, str]:
    return {
        **{f"B{index}": session_id for index, session_id in enumerate(original_boundary, start=1)},
        **{f"P{index}": session_id for index, session_id in enumerate(promotions, start=1)},
    }


def _uses_symbolic_boundary_slots(prompt_version: str) -> bool:
    return prompt_version in {
        LONGMEMEVAL_SYMBOLIC_BOUNDARY_PROMPT_VERSION,
        LONGMEMEVAL_ATOMIC_BOUNDARY_PROMPT_VERSION,
        LONGMEMEVAL_SYMBOLIC_SPAN_BOUNDARY_PROMPT_VERSION,
    }


def _validate_atomic_slot_assessments(
    parsed: dict[str, object],
    *,
    selected_slot_labels: Sequence[str],
    slot_session_ids: dict[str, str],
    by_session: dict[str, RetrievedMemory],
    question: str,
    config: LongMemEvalRerankerConfig,
) -> tuple[list[dict[str, JsonValue]], list[str]]:
    raw_assessments = parsed.get("slot_assessments")
    values = raw_assessments if isinstance(raw_assessments, list) else []
    missing_need_ids = _evidence_need_ids(_string_list(parsed.get("needs_missing_after_locked")))
    assessments: list[dict[str, JsonValue]] = []
    by_slot: dict[str, dict[str, JsonValue]] = {}
    for value in values:
        if not isinstance(value, dict):
            continue
        slot_value = value.get("slot")
        if not isinstance(slot_value, str):
            continue
        slot = _normalized_slot_label(slot_value)
        session_id = slot_session_ids.get(slot)
        if session_id is None or slot in by_slot:
            continue
        supports_needs = _string_list(value.get("supports_needs"))[:4]
        quote_value = value.get("evidence_quote")
        quote = (
            quote_value.strip() if isinstance(quote_value, str) and quote_value.strip() else None
        )
        adds_missing_evidence = value.get("adds_missing_evidence") is True
        memory = by_session.get(session_id)
        quote_valid = bool(
            quote
            and memory is not None
            and _quote_is_grounded(
                quote,
                candidate_excerpt(
                    question,
                    memory.content,
                    max_chars=config.max_candidate_chars,
                    max_turns=config.max_excerpt_turns,
                ),
            )
        )
        assessment: dict[str, JsonValue] = {
            "slot": slot,
            "supports_needs": cast(JsonValue, supports_needs),
            "evidence_quote": quote,
            "adds_missing_evidence": adds_missing_evidence,
            "quote_valid": quote_valid,
        }
        assessments.append(assessment)
        by_slot[slot] = assessment

    failures: list[str] = []
    for slot in selected_slot_labels:
        if not slot.startswith("P"):
            continue
        selected_assessment = by_slot.get(slot)
        if selected_assessment is None:
            failures.append(f"{slot}:missing_assessment")
            continue
        selected_needs = selected_assessment.get("supports_needs")
        if not isinstance(selected_needs, list) or not selected_needs:
            failures.append(f"{slot}:missing_need")
        elif not missing_need_ids.intersection(
            _evidence_need_ids([value for value in selected_needs if isinstance(value, str)])
        ):
            failures.append(f"{slot}:not_linked_to_missing_need")
        if selected_assessment.get("adds_missing_evidence") is not True:
            failures.append(f"{slot}:not_missing_evidence")
        selected_quote = selected_assessment.get("evidence_quote")
        if not isinstance(selected_quote, str) or not selected_quote:
            failures.append(f"{slot}:missing_quote")
        elif selected_assessment.get("quote_valid") is not True:
            failures.append(f"{slot}:quote_not_grounded")
    return assessments, failures


def _validate_selector_evidence_selections(
    parsed: dict[str, object],
    *,
    candidates: Sequence[RetrievedMemory],
    question: str,
    config: LongMemEvalRerankerConfig,
) -> tuple[list[dict[str, JsonValue]], list[str], list[str]]:
    raw_selections = parsed.get("evidence_selections")
    values = raw_selections if isinstance(raw_selections, list) else []
    need_ids = _evidence_need_ids(_string_list(parsed.get("evidence_needs")))
    span_owners = _selector_span_owner_map(
        candidates,
        question=question,
        config=config,
    )
    selections: list[dict[str, JsonValue]] = []
    failures: list[str] = []
    grounded_labels: list[str] = []
    for index, value in enumerate(values, start=1):
        if not isinstance(value, dict):
            failures.append(f"E{index}:invalid_assessment")
            continue
        supports_needs = _string_list(value.get("supports_needs"))[:4]
        raw_span_ids = _string_list(value.get("evidence_spans"))[:4]
        span_ids = [_normalized_evidence_span_id(span_id) for span_id in raw_span_ids]
        invalid_span_ids = [span_id for span_id in span_ids if span_id not in span_owners]
        owner_labels = _ordered_unique(
            span_owners[span_id]
            for span_id in span_ids
            if span_id in span_owners
        )
        supports_known_need = bool(
            need_ids.intersection(_evidence_need_ids(supports_needs))
        )
        span_valid = bool(span_ids) and not invalid_span_ids and len(owner_labels) == 1
        selection: dict[str, JsonValue] = {
            "supports_needs": cast(JsonValue, supports_needs),
            "evidence_spans": cast(JsonValue, span_ids),
            "owner_candidate_labels": cast(JsonValue, owner_labels),
            "span_valid": span_valid,
        }
        selections.append(selection)
        if not supports_needs:
            failures.append(f"E{index}:missing_need")
        elif not supports_known_need:
            failures.append(f"E{index}:unknown_need")
        if not span_ids:
            failures.append(f"E{index}:missing_span")
        elif invalid_span_ids or len(owner_labels) != 1:
            failures.append(f"E{index}:span_not_grounded")
        if span_valid and supports_known_need:
            grounded_labels.extend(owner_labels)
    return selections, failures, _ordered_unique(grounded_labels)


def _validate_challenger_assessments(
    parsed: dict[str, object],
    *,
    candidates: Sequence[RetrievedMemory],
    question: str,
    config: LongMemEvalRerankerConfig,
) -> tuple[list[dict[str, JsonValue]], list[str], list[str], bool]:
    raw_assessments = parsed.get("challenger_assessments")
    values = raw_assessments if isinstance(raw_assessments, list) else []
    need_ids = _evidence_need_ids(_string_list(parsed.get("evidence_needs")))
    span_owners = _selector_span_owner_map(
        candidates,
        question=question,
        config=config,
    )
    expected_labels = [
        _candidate_label(index)
        for index in range(config.output_top_k + 1, config.candidate_count + 1)
    ]
    expected = set(expected_labels)
    assessments: list[dict[str, JsonValue]] = []
    failures: list[str] = []
    grounded_labels: list[str] = []
    observed: set[str] = set()
    complete = True
    for index, value in enumerate(values, start=1):
        if not isinstance(value, dict):
            failures.append(f"E{index}:invalid_assessment")
            complete = False
            continue
        candidate_value = value.get("candidate")
        candidate = (
            _normalized_candidate_label(candidate_value)
            if isinstance(candidate_value, str)
            else ""
        )
        if candidate not in expected:
            failures.append(f"E{index}:invalid_challenger")
            complete = False
            continue
        if candidate in observed:
            failures.append(f"{candidate}:duplicate_assessment")
            complete = False
            continue
        observed.add(candidate)
        supports_needs = _string_list(value.get("supports_needs"))[:4]
        span_ids = [
            _normalized_evidence_span_id(span_id)
            for span_id in _string_list(value.get("evidence_spans"))[:4]
        ]
        adds_missing_evidence = value.get("adds_missing_evidence") is True
        supports_known_need = bool(
            need_ids.intersection(_evidence_need_ids(supports_needs))
        )
        span_valid = bool(span_ids) and all(
            span_owners.get(span_id) == candidate for span_id in span_ids
        )
        assessment: dict[str, JsonValue] = {
            "candidate": candidate,
            "supports_needs": cast(JsonValue, supports_needs),
            "evidence_spans": cast(JsonValue, span_ids),
            "adds_missing_evidence": adds_missing_evidence,
            "span_valid": span_valid,
        }
        assessments.append(assessment)
        if not adds_missing_evidence:
            continue
        if not supports_needs:
            failures.append(f"{candidate}:missing_need")
        elif not supports_known_need:
            failures.append(f"{candidate}:unknown_need")
        if not span_ids:
            failures.append(f"{candidate}:missing_span")
        elif not span_valid:
            failures.append(f"{candidate}:span_not_grounded")
        if supports_known_need and span_valid:
            grounded_labels.append(candidate)

    missing = [label for label in expected_labels if label not in observed]
    if missing:
        failures.append("missing_assessments:" + ",".join(missing))
        complete = False
    assessments.sort(
        key=lambda assessment: expected_labels.index(str(assessment["candidate"]))
    )
    return assessments, failures, _ordered_unique(grounded_labels), complete


def _validate_symbolic_span_slot_assessments(
    parsed: dict[str, object],
    *,
    selected_slot_labels: Sequence[str],
    slot_session_ids: dict[str, str],
    by_session: dict[str, RetrievedMemory],
    question: str,
    config: LongMemEvalRerankerConfig,
) -> tuple[list[dict[str, JsonValue]], list[str]]:
    raw_assessments = parsed.get("slot_assessments")
    values = raw_assessments if isinstance(raw_assessments, list) else []
    missing_need_ids = _evidence_need_ids(
        _string_list(parsed.get("needs_missing_after_locked"))
    )
    span_owners: dict[str, str] = {}
    for slot, session_id in slot_session_ids.items():
        memory = by_session.get(session_id)
        if memory is None:
            continue
        for span_id in _evidence_span_ids(
            slot,
            memory,
            question=question,
            config=config,
        ):
            span_owners[span_id] = slot

    assessments: list[dict[str, JsonValue]] = []
    by_slot: dict[str, dict[str, JsonValue]] = {}
    for value in values:
        if not isinstance(value, dict):
            continue
        slot_value = value.get("slot")
        if not isinstance(slot_value, str):
            continue
        slot = _normalized_slot_label(slot_value)
        if slot not in slot_session_ids or slot in by_slot:
            continue
        supports_needs = _string_list(value.get("supports_needs"))[:4]
        span_ids = [
            _normalized_evidence_span_id(span_id)
            for span_id in _string_list(value.get("evidence_spans"))[:4]
        ]
        adds_missing_evidence = value.get("adds_missing_evidence") is True
        span_valid = bool(span_ids) and all(
            span_owners.get(span_id) == slot for span_id in span_ids
        )
        assessment: dict[str, JsonValue] = {
            "slot": slot,
            "supports_needs": cast(JsonValue, supports_needs),
            "evidence_spans": cast(JsonValue, span_ids),
            "adds_missing_evidence": adds_missing_evidence,
            "span_valid": span_valid,
        }
        assessments.append(assessment)
        by_slot[slot] = assessment

    failures: list[str] = []
    for slot in selected_slot_labels:
        if not slot.startswith("P"):
            continue
        selected_assessment = by_slot.get(slot)
        if selected_assessment is None:
            failures.append(f"{slot}:missing_assessment")
            continue
        selected_needs = selected_assessment.get("supports_needs")
        if not isinstance(selected_needs, list) or not selected_needs:
            failures.append(f"{slot}:missing_need")
        elif not missing_need_ids.intersection(
            _evidence_need_ids(
                [value for value in selected_needs if isinstance(value, str)]
            )
        ):
            failures.append(f"{slot}:not_linked_to_missing_need")
        if selected_assessment.get("adds_missing_evidence") is not True:
            failures.append(f"{slot}:not_missing_evidence")
        selected_spans = selected_assessment.get("evidence_spans")
        if not isinstance(selected_spans, list) or not selected_spans:
            failures.append(f"{slot}:missing_span")
        elif selected_assessment.get("span_valid") is not True:
            failures.append(f"{slot}:span_not_grounded")
    return assessments, failures


def _evidence_need_ids(values: Sequence[str]) -> set[str]:
    identifiers: set[str] = set()
    for value in values:
        match = re.match(r"\s*(N[1-4])(?:\s*:|\s*$)", value, flags=re.IGNORECASE)
        if match:
            identifiers.add(match.group(1).upper())
    return identifiers


def _quote_is_grounded(quote: str, excerpt: str) -> bool:
    normalized_quote = _normalized_quote_text(quote)
    if len(normalized_quote) < 12:
        return False
    return normalized_quote in _normalized_quote_text(excerpt)


def _normalized_quote_text(value: str) -> str:
    quote_marks = " \t\r\n\"'\u201c\u201d\u2018\u2019`"
    return " ".join(value.casefold().strip(quote_marks).split())


def _normalized_candidate_label(value: str) -> str:
    compact = re.sub(r"\s+", "", value).upper()
    match = re.fullmatch(r"C0*(\d+)", compact)
    if not match:
        return compact
    return _candidate_label(int(match.group(1)))


def _normalized_evidence_span_id(value: str) -> str:
    compact = re.sub(r"\s+", "", value).upper()
    candidate = re.fullmatch(r"C0*(\d+):S0*(\d+)", compact)
    if candidate:
        return f"{_candidate_label(int(candidate.group(1)))}:S{int(candidate.group(2)):02d}"
    boundary = re.fullmatch(r"([BP])0*(\d+):S0*(\d+)", compact)
    if boundary:
        return f"{boundary.group(1)}{int(boundary.group(2))}:S{int(boundary.group(3)):02d}"
    locked = re.fullmatch(r"LOCKED-0*(\d+):S0*(\d+)", compact)
    if locked:
        return f"LOCKED-{int(locked.group(1))}:S{int(locked.group(2)):02d}"
    return compact


def _normalized_slot_label(value: str) -> str:
    return re.sub(r"\s+", "", value).upper()


def _slot_labels_for_sessions(
    slot_session_ids: dict[str, str],
    selected_session_ids: Sequence[str],
) -> list[str]:
    by_session = {session_id: label for label, session_id in slot_session_ids.items()}
    return [
        by_session[session_id] for session_id in selected_session_ids if session_id in by_session
    ]


def _boundary_decision_name(
    selected_slot_labels: Sequence[str],
) -> str:
    promotions = sum(label.startswith("P") for label in selected_slot_labels)
    if promotions == 0:
        return "keep"
    return "replace_one" if promotions == 1 else "replace_two"


def guarded_session_ranking(
    *,
    original_session_ids: Sequence[str],
    proposed_session_ids: Sequence[str],
    output_top_k: int,
    protected_top_n: int,
) -> list[str]:
    """Apply one shared safe-membership rule to an LLM-proposed ranking."""

    if output_top_k < 1:
        raise ValueError("output_top_k must be at least 1")
    if not 0 <= protected_top_n < output_top_k:
        raise ValueError("protected_top_n must be in [0, output_top_k)")
    original = _ordered_unique(original_session_ids)
    allowed = set(original)
    proposed = _ordered_unique(
        session_id for session_id in proposed_session_ids if session_id in allowed
    )
    protected = original[: min(protected_top_n, len(original))]
    open_slots = max(0, min(output_top_k, len(original)) - len(protected))
    promoted = [session_id for session_id in proposed if session_id not in protected][:open_slots]
    if len(promoted) < open_slots:
        promoted.extend(
            session_id
            for session_id in original
            if session_id not in protected and session_id not in promoted
        )
        promoted = promoted[:open_slots]
    selected = [*protected, *promoted]
    tail = [session_id for session_id in [*proposed, *original] if session_id not in selected]
    return _ordered_unique([*selected, *tail])


def reorder_memories(
    memories: Sequence[RetrievedMemory],
    ranked_session_ids: Sequence[str],
) -> list[RetrievedMemory]:
    """Materialize one session-ID ranking as ``RetrievedMemory`` records."""

    by_session = {_session_id(memory): memory for memory in _unique_memories(memories)}
    return [by_session[session_id] for session_id in ranked_session_ids if session_id in by_session]


def candidate_excerpt(
    question: str,
    content: str,
    *,
    max_chars: int,
    max_turns: int,
) -> str:
    """Select lexical query-matching turns with the same rule for every method."""

    if max_chars < 1 or max_turns < 1:
        raise ValueError("excerpt limits must be positive")
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return _balanced_excerpt(content, max_chars=max_chars)
    query_terms = set(terms(question))
    scored = []
    for index, line in enumerate(lines):
        line_terms = set(terms(line))
        overlap = len(query_terms.intersection(line_terms))
        scored.append((overlap, -len(line_terms), index, line))
    matched = [row for row in scored if row[0] > 0]
    if matched:
        selected_indices: list[int] = []
        for anchor in sorted(matched, key=lambda row: (-row[0], row[1], row[2])):
            for candidate_index in (anchor[2] - 1, anchor[2], anchor[2] + 1):
                if 0 <= candidate_index < len(lines) and candidate_index not in selected_indices:
                    selected_indices.append(candidate_index)
                if len(selected_indices) >= max_turns:
                    break
            if len(selected_indices) >= max_turns:
                break
        excerpt = "\n".join(lines[index] for index in sorted(selected_indices))
    else:
        edge_lines = _ordered_unique([*lines[:2], *lines[-2:]])
        excerpt = "\n".join(edge_lines[:max_turns])
    return _balanced_excerpt(excerpt, max_chars=max_chars)


def candidate_evidence_spans(
    question: str,
    content: str,
    *,
    max_chars: int,
    max_turns: int,
    max_span_chars: int = 360,
) -> list[str]:
    """Split the deterministic candidate excerpt into locally addressable spans."""

    if max_span_chars < 32:
        raise ValueError("max_span_chars must be at least 32")
    excerpt = candidate_excerpt(
        question,
        content,
        max_chars=max_chars,
        max_turns=max_turns,
    )
    pieces = [
        piece.strip()
        for piece in re.split(r"(?<=[.!?])\s+|\n+", excerpt)
        if piece.strip() and piece.strip() != "..."
    ]
    spans: list[str] = []
    for piece in pieces or [excerpt]:
        spans.extend(_split_evidence_span(piece, max_chars=max_span_chars))
    return spans or ["(empty candidate)"]


def _format_symbolic_span_candidate(
    rank: int,
    memory: RetrievedMemory,
    *,
    question: str,
    max_chars: int,
    max_turns: int,
) -> str:
    label = _candidate_label(rank)
    return _format_labeled_span_memory(
        label,
        memory,
        question=question,
        description=f"candidate_rank={rank}",
        max_chars=max_chars,
        max_turns=max_turns,
    )


def _format_labeled_span_memory(
    label: str,
    memory: RetrievedMemory,
    *,
    question: str,
    description: str,
    max_chars: int,
    max_turns: int,
) -> str:
    spans = candidate_evidence_spans(
        question,
        memory.content,
        max_chars=max_chars,
        max_turns=max_turns,
    )
    evidence = "\n".join(
        f"[{label}:S{index:02d}] {span}"
        for index, span in enumerate(spans, start=1)
    )
    return (
        f"[{label} | {description} | date={memory.source_date or 'unknown'}]\n"
        f"{evidence}"
    )


def _format_candidate(
    rank: int,
    memory: RetrievedMemory,
    *,
    question: str,
    max_chars: int,
    max_turns: int,
) -> str:
    excerpt = candidate_excerpt(
        question,
        memory.content,
        max_chars=max_chars,
        max_turns=max_turns,
    )
    return (
        f"[Candidate {rank} | session_id={_session_id(memory)} | "
        f"date={memory.source_date or 'unknown'}]\n{excerpt}"
    )


def _candidate_label(rank: int) -> str:
    if rank < 1:
        raise ValueError("candidate rank must be positive")
    return f"C{rank:02d}"


def _candidate_label_session_map(
    candidates: Sequence[RetrievedMemory],
) -> dict[str, str]:
    return {
        _candidate_label(index): _session_id(memory)
        for index, memory in enumerate(candidates, start=1)
    }


def _selector_span_owner_map(
    candidates: Sequence[RetrievedMemory],
    *,
    question: str,
    config: LongMemEvalRerankerConfig,
) -> dict[str, str]:
    owners: dict[str, str] = {}
    for index, memory in enumerate(candidates, start=1):
        label = _candidate_label(index)
        for span_id in _evidence_span_ids(
            label,
            memory,
            question=question,
            config=config,
        ):
            owners[span_id] = label
    return owners


def _evidence_span_ids(
    label: str,
    memory: RetrievedMemory,
    *,
    question: str,
    config: LongMemEvalRerankerConfig,
) -> list[str]:
    spans = candidate_evidence_spans(
        question,
        memory.content,
        max_chars=config.max_candidate_chars,
        max_turns=config.max_excerpt_turns,
    )
    return [f"{label}:S{index:02d}" for index in range(1, len(spans) + 1)]


def _parse_rerank_response(text: str) -> tuple[dict[str, object], str | None]:
    stripped = text.strip()
    fence_match = _JSON_FENCE_PATTERN.match(stripped)
    if fence_match:
        stripped = fence_match.group(1).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        return {}, "response did not contain a JSON object"
    try:
        payload = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        return {}, f"invalid JSON response: {exc.msg}"
    if not isinstance(payload, dict):
        return {}, "JSON response was not an object"
    return {str(key): value for key, value in payload.items()}, None


def _unique_memories(memories: Sequence[RetrievedMemory]) -> list[RetrievedMemory]:
    unique: list[RetrievedMemory] = []
    seen: set[str] = set()
    for memory in memories:
        session_id = _session_id(memory)
        if session_id in seen:
            continue
        seen.add(session_id)
        unique.append(memory)
    return unique


def _session_id(memory: RetrievedMemory) -> str:
    return memory.source_session_id or memory.memory_id


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [stripped for item in value if isinstance(item, str) and (stripped := item.strip())]


def _normalized_string(value: object) -> str | None:
    return value.strip().casefold() if isinstance(value, str) and value.strip() else None


def _optional_string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _ordered_unique(values: Iterable[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def _balanced_excerpt(text: str, *, max_chars: int) -> str:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped or "(empty candidate)"
    if max_chars < 32:
        return stripped[:max_chars]
    head = (max_chars - 5) // 2
    tail = max_chars - 5 - head
    return f"{stripped[:head]}\n...\n{stripped[-tail:]}"


def _split_evidence_span(text: str, *, max_chars: int) -> list[str]:
    stripped = " ".join(text.split())
    if len(stripped) <= max_chars:
        return [stripped] if stripped else []
    words = stripped.split()
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for word in words:
        added = len(word) + int(bool(current))
        if current and current_length + added > max_chars:
            chunks.append(" ".join(current))
            current = []
            current_length = 0
        if len(word) > max_chars:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_length = 0
            chunks.extend(
                word[index : index + max_chars]
                for index in range(0, len(word), max_chars)
            )
            continue
        current.append(word)
        current_length += len(word) + int(len(current) > 1)
    if current:
        chunks.append(" ".join(current))
    return chunks


def _usage_tokens(
    usage: dict[str, JsonValue],
    key: str,
    *,
    fallback: int,
) -> int:
    value = usage.get(key)
    return int(value) if isinstance(value, int | float) and value >= 0 else fallback
