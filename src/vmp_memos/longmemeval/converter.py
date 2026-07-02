"""Convert LongMemEval records into VMP-MemOS benchmark primitives."""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import JsonValue

from vmp_memos.longmemeval.schema import LongMemEvalSample, LongMemEvalSession
from vmp_memos.schemas import BenchmarkSample, Event, EventType

_SAFE_ID_PATTERN = re.compile(r"[^A-Za-z0-9_.:-]+")


def sample_to_events(sample: LongMemEvalSample) -> list[Event]:
    """Convert every LongMemEval history turn into an Event."""

    events: list[Event] = []
    for session in sample.sessions:
        events.extend(_session_to_events(sample, session))
    return events


def sample_to_session_events(sample: LongMemEvalSample) -> list[list[Event]]:
    """Convert a sample into event groups, one group per history session."""

    return [_session_to_events(sample, session) for session in sample.sessions]


def sample_to_query_event(sample: LongMemEvalSample) -> Event:
    """Represent the benchmark question as a query event."""

    return Event(
        event_id=f"lme_{_safe_id(sample.question_id)}_query",
        session_id=f"lme_eval_{_safe_id(sample.question_id)}",
        task_id=sample.question_id,
        event_type=EventType.BENCHMARK_QUERY,
        content=sample.question,
        metadata={
            "dataset": "longmemeval",
            "question_id": sample.question_id,
            "question_type": sample.question_type,
            "question_date": sample.question_date,
            "answer_session_ids": list(sample.answer_session_ids),
            "is_abstention": sample.is_abstention,
        },
    )


def sample_to_benchmark_sample(sample: LongMemEvalSample) -> BenchmarkSample:
    """Create the generic BenchmarkSample used by existing evaluation utilities."""

    return BenchmarkSample(
        sample_id=f"lme_{_safe_id(sample.question_id)}",
        events=sample_to_events(sample),
        query=sample.question,
        gold_answer=sample.answer,
        gold_memory_ids=list(sample.answer_session_ids),
        metadata={
            "dataset": "longmemeval",
            "question_id": sample.question_id,
            "question_type": sample.question_type,
            "question_date": sample.question_date,
            "is_abstention": sample.is_abstention,
            "session_count": sample.session_count,
            "turn_count": sample.turn_count,
        },
    )


def session_to_text(events: Sequence[Event]) -> str:
    """Format a session event group as the text indexed by retrieval adapters."""

    lines: list[str] = []
    for event in events:
        role = str(event.metadata.get("role") or event.event_type.value)
        content = _content_as_text(event.content)
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _session_to_events(sample: LongMemEvalSample, session: LongMemEvalSession) -> list[Event]:
    events: list[Event] = []
    question_id = _safe_id(sample.question_id)
    session_id = _safe_id(session.session_id)
    event_session_id = f"{question_id}_{session_id}"
    for turn_idx, turn in enumerate(session.turns):
        event_type = _event_type_for_role(turn.role)
        events.append(
            Event(
                event_id=f"lme_{question_id}_{session_id}_{turn_idx}",
                session_id=event_session_id,
                task_id=sample.question_id,
                event_type=event_type,
                content=turn.content,
                metadata={
                    "dataset": "longmemeval",
                    "question_id": sample.question_id,
                    "question_type": sample.question_type,
                    "history_session_id": session.session_id,
                    "history_date": session.date,
                    "turn_idx": turn_idx,
                    "role": turn.role,
                    "has_answer": turn.has_answer,
                },
            )
        )
    return events


def _event_type_for_role(role: str) -> EventType:
    normalized = role.casefold()
    if "assistant" in normalized or normalized in {"ai", "bot", "agent"}:
        return EventType.ASSISTANT_MESSAGE
    return EventType.USER_MESSAGE


def _safe_id(value: str) -> str:
    normalized = _SAFE_ID_PATTERN.sub("_", value.strip())
    return normalized.strip("_") or "unknown"


def _content_as_text(content: JsonValue) -> str:
    if isinstance(content, str):
        return content
    return str(content)
