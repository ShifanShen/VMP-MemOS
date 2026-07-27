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
from typing import Protocol

from pydantic import Field, JsonValue, PositiveInt, model_validator

from vmp_memos.frameworks import RetrievedMemory
from vmp_memos.frameworks.text import estimate_tokens, terms
from vmp_memos.llm.base import ChatMessage, LLMGenerationConfig, LLMResponse
from vmp_memos.schemas.base import (
    NonEmptyStr,
    NonNegativeFloat,
    NonNegativeInt,
    SchemaModel,
)

LONGMEMEVAL_RERANK_PROMPT_VERSION = "vmp_v52_evidence_set_v1"
LONGMEMEVAL_BOUNDARY_PROMPT_VERSION = "vmp_v53_selective_boundary_v1"
LONGMEMEVAL_SYMBOLIC_BOUNDARY_PROMPT_VERSION = "vmp_v531_symbolic_boundary_v1"
LONGMEMEVAL_BOUNDARY_PROMPT_VERSIONS = frozenset(
    {
        LONGMEMEVAL_BOUNDARY_PROMPT_VERSION,
        LONGMEMEVAL_SYMBOLIC_BOUNDARY_PROMPT_VERSION,
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

        if self.prompt_version != LONGMEMEVAL_RERANK_PROMPT_VERSION:
            raise ValueError("unsupported LongMemEval reranker prompt version")
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
        if self.boundary_verification:
            if self.boundary_protected_top_n >= self.output_top_k:
                raise ValueError("V5.3 boundary verification must leave open Top-k slots")
            open_slots = self.output_top_k - self.boundary_protected_top_n
            if self.boundary_max_promotions < open_slots:
                raise ValueError(
                    "V5.3 boundary_max_promotions must cover every open Top-k slot"
                )
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
    original_boundary_session_ids: list[NonEmptyStr] = Field(default_factory=list)
    proposed_promotion_session_ids: list[NonEmptyStr] = Field(default_factory=list)
    slot_session_ids: dict[NonEmptyStr, NonEmptyStr] = Field(default_factory=dict)
    raw_selected_slot_labels: list[NonEmptyStr] = Field(default_factory=list)
    selected_slot_labels: list[NonEmptyStr] = Field(default_factory=list)
    invalid_slot_labels: list[NonEmptyStr] = Field(default_factory=list)
    raw_selected_boundary_session_ids: list[NonEmptyStr] = Field(default_factory=list)
    selected_boundary_session_ids: list[NonEmptyStr] = Field(default_factory=list)
    invalid_session_ids: list[NonEmptyStr] = Field(default_factory=list)
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
                raise ValueError(
                    "selector replay requires source_method and question_id context"
                )
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
        raw_selected = _string_list(parsed.get("selected_session_ids"))
        raw_ranked = _string_list(parsed.get("ranked_session_ids"))
        evidence_needs = _string_list(parsed.get("evidence_needs"))[:4]
        allowed = set(original_ids)
        proposed = _ordered_unique([*raw_selected, *raw_ranked])
        invalid_ids = [session_id for session_id in proposed if session_id not in allowed]
        valid_proposed = [session_id for session_id in proposed if session_id in allowed]
        parse_fallback = not valid_proposed
        if parse_fallback:
            valid_proposed = list(original_ids)
            fallback_reason = fallback_reason or "response contained no valid candidate IDs"
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
        decision_name: str | None
        if (
            self.config.boundary_prompt_version
            == LONGMEMEVAL_SYMBOLIC_BOUNDARY_PROMPT_VERSION
        ):
            raw_slot_labels = _string_list(parsed.get("selected_slots"))
            normalized_labels = [
                _normalized_slot_label(label) for label in raw_slot_labels
            ]
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
            raw_selected = [
                slot_session_ids[label]
                for label in selected_slot_labels
            ]
            invalid_ids: list[str] = []
            valid_selected = list(raw_selected)
            parse_fallback = (
                len(selected_slot_labels) != open_slots
                or bool(invalid_slot_labels)
            )
            decision_name = _boundary_decision_name(
                selected_slot_labels,
            )
        else:
            raw_selected = _string_list(parsed.get("selected_boundary_session_ids"))
            allowed_boundary = set([*original_boundary, *promotions])
            invalid_ids = [
                session_id
                for session_id in raw_selected
                if session_id not in allowed_boundary
            ]
            valid_selected = _ordered_unique(
                session_id
                for session_id in raw_selected
                if session_id in allowed_boundary
            )
            parse_fallback = len(valid_selected) != open_slots or bool(invalid_ids)
            decision_name = _normalized_string(parsed.get("decision"))
        proposed_change = set(valid_selected) != set(original_boundary)
        policy_rejected = (
            not parse_fallback
            and proposed_change
            and confidence != self.config.boundary_min_confidence
        )
        if parse_fallback:
            selected_boundary = original_boundary
            fallback_reason = fallback_reason or (
                f"boundary response must select exactly {open_slots} valid distinct "
                + (
                    "slot labels"
                    if self.config.boundary_prompt_version
                    == LONGMEMEVAL_SYMBOLIC_BOUNDARY_PROMPT_VERSION
                    else "IDs"
                )
            )
        elif policy_rejected:
            selected_boundary = original_boundary
            fallback_reason = (
                "replacement confidence below "
                f"{self.config.boundary_min_confidence!r}"
            )
        else:
            selected_boundary = valid_selected if proposed_change else original_boundary
        if (
            self.config.boundary_prompt_version
            == LONGMEMEVAL_SYMBOLIC_BOUNDARY_PROMPT_VERSION
        ):
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
                original_boundary_session_ids=original_boundary,
                proposed_promotion_session_ids=promotions,
                slot_session_ids=slot_session_ids,
                raw_selected_slot_labels=raw_slot_labels,
                selected_slot_labels=selected_slot_labels,
                invalid_slot_labels=invalid_slot_labels,
                raw_selected_boundary_session_ids=raw_selected,
                selected_boundary_session_ids=selected_boundary,
                invalid_session_ids=invalid_ids,
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
    )


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
        == LONGMEMEVAL_SYMBOLIC_BOUNDARY_PROMPT_VERSION
    ):
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
    )


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
    )


def _boundary_slot_map(
    original_boundary: Sequence[str],
    promotions: Sequence[str],
) -> dict[str, str]:
    return {
        **{
            f"B{index}": session_id
            for index, session_id in enumerate(original_boundary, start=1)
        },
        **{
            f"P{index}": session_id
            for index, session_id in enumerate(promotions, start=1)
        },
    }


def _normalized_slot_label(value: str) -> str:
    return re.sub(r"\s+", "", value).upper()


def _slot_labels_for_sessions(
    slot_session_ids: dict[str, str],
    selected_session_ids: Sequence[str],
) -> list[str]:
    by_session = {
        session_id: label for label, session_id in slot_session_ids.items()
    }
    return [
        by_session[session_id]
        for session_id in selected_session_ids
        if session_id in by_session
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


def _usage_tokens(
    usage: dict[str, JsonValue],
    key: str,
    *,
    fallback: int,
) -> int:
    value = usage.get(key)
    return int(value) if isinstance(value, int | float) and value >= 0 else fallback
