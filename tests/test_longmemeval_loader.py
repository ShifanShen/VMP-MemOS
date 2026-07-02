"""Tests for LongMemEval loading and conversion."""

from __future__ import annotations

import json

from vmp_memos.longmemeval import (
    inspect_longmemeval,
    load_longmemeval,
    sample_to_benchmark_sample,
    sample_to_events,
    sample_to_query_event,
    sample_to_session_events,
    session_to_text,
)
from vmp_memos.schemas import EventType


def test_load_and_convert_longmemeval_sample(tmp_path) -> None:
    path = tmp_path / "longmemeval_s_cleaned.json"
    path.write_text(json.dumps([_sample_record()]), encoding="utf-8")

    samples = load_longmemeval(path)
    assert len(samples) == 1
    sample = samples[0]
    assert sample.question_id == "q1"
    assert sample.session_count == 2
    assert sample.turn_count == 4
    assert not sample.is_abstention

    stats = inspect_longmemeval(path)
    assert stats.sample_count == 1
    assert stats.question_types == {"knowledge_update": 1}
    assert stats.session_count_avg == 2

    events = sample_to_events(sample)
    assert len(events) == 4
    assert events[0].event_type == EventType.USER_MESSAGE
    assert events[1].event_type == EventType.ASSISTANT_MESSAGE
    assert events[0].metadata["history_session_id"] == "s_old"

    session_events = sample_to_session_events(sample)
    assert len(session_events) == 2
    assert "hiking" in session_to_text(session_events[0])

    query_event = sample_to_query_event(sample)
    assert query_event.event_type == EventType.BENCHMARK_QUERY
    assert query_event.metadata["answer_session_ids"] == ["s_new"]

    benchmark_sample = sample_to_benchmark_sample(sample)
    assert benchmark_sample.sample_id == "lme_q1"
    assert benchmark_sample.gold_memory_ids == ["s_new"]


def test_load_normalizes_numeric_answer_to_text(tmp_path) -> None:
    record = _sample_record()
    record["answer"] = 3
    answer_list_record = _sample_record()
    answer_list_record["question_id"] = "q2"
    answer_list_record["answer"] = [3, "three"]
    path = tmp_path / "longmemeval_numeric_answer.json"
    path.write_text(json.dumps([record, answer_list_record]), encoding="utf-8")

    samples = load_longmemeval(path)

    assert samples[0].answer == "3"
    assert sample_to_benchmark_sample(samples[0]).gold_answer == "3"
    assert samples[1].answer == ["3", "three"]


def _sample_record() -> dict:
    return {
        "question_id": "q1",
        "question_type": "knowledge_update",
        "question": "What activity does Alex now prefer?",
        "answer": "swimming",
        "question_date": "2024-02-01",
        "haystack_session_ids": ["s_old", "s_new"],
        "haystack_dates": ["2024-01-01", "2024-01-20"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "Alex said hiking was fun."},
                {"role": "assistant", "content": "I will remember Alex liked hiking."},
            ],
            [
                {"role": "user", "content": "Alex now prefers swimming instead of hiking."},
                {"role": "assistant", "content": "Updated: Alex prefers swimming."},
            ],
        ],
        "answer_session_ids": ["s_new"],
        "has_answer": True,
    }
