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
from typing import Protocol

from pydantic import Field, JsonValue, PositiveInt, model_validator

from vmp_memos.frameworks import RetrievedMemory
from vmp_memos.frameworks.text import estimate_tokens, terms
from vmp_memos.llm.base import ChatMessage, LLMGenerationConfig, LLMResponse
from vmp_memos.schemas.base import (
    NonEmptyStr,
    NonNegativeInt,
    SchemaModel,
)

LONGMEMEVAL_RERANK_PROMPT_VERSION = "vmp_v52_evidence_set_v1"
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
    generation: LLMGenerationConfig = Field(
        default_factory=lambda: LLMGenerationConfig(
            max_tokens=512,
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
        if float(self.generation.temperature) != 0.0:
            raise ValueError("paper reranking requires temperature=0")
        if float(self.generation.top_p) != 1.0:
            raise ValueError("paper reranking requires top_p=1")
        return self


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
    ranked_session_ids: list[NonEmptyStr] = Field(default_factory=list)
    selected_session_ids: list[NonEmptyStr] = Field(default_factory=list)
    invalid_session_ids: list[NonEmptyStr] = Field(default_factory=list)
    parse_fallback: bool = False
    parse_fallback_reason: str | None = None
    input_tokens: NonNegativeInt = 0
    output_tokens: NonNegativeInt = 0
    usage: dict[str, JsonValue] = Field(default_factory=dict)
    response_text: str = ""


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
        question: str,
        question_date: str | None,
        candidates: Sequence[RetrievedMemory],
    ) -> LongMemEvalRerankDecision:
        """Rerank one framework's candidates without observing gold labels."""

        unique_candidates = _unique_memories(candidates)[: self.config.candidate_count]
        if not unique_candidates:
            raise ValueError("at least one retrieval candidate is required")
        original_ids = [_session_id(memory) for memory in unique_candidates]
        prompt = build_longmemeval_rerank_prompt(
            question=question,
            question_date=question_date,
            candidates=unique_candidates,
            config=self.config,
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
        ranked_ids = guarded_session_ranking(
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
        return LongMemEvalRerankDecision(
            prompt_version=self.config.prompt_version,
            prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            provider=response.provider,
            model=response.model,
            finish_reason=response.finish_reason,
            evidence_needs=evidence_needs,
            raw_selected_session_ids=raw_selected,
            raw_ranked_session_ids=raw_ranked,
            ranked_session_ids=ranked_ids,
            selected_session_ids=ranked_ids[: self.config.output_top_k],
            invalid_session_ids=invalid_ids,
            parse_fallback=parse_fallback,
            parse_fallback_reason=fallback_reason if parse_fallback else None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usage=response.usage,
            response_text=response.text.strip(),
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
