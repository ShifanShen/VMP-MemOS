"""Tests for offline LongMemEval cost and efficiency analysis."""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from vmp_memos.llm import (
    ChatMessage,
    LLMGenerationConfig,
    LLMResponse,
    LongMemEvalReader,
    LongMemEvalReaderConfig,
)
from vmp_memos.longmemeval import LongMemEvalRunConfig
from vmp_memos.longmemeval.cost import (
    _summarize_cost,
    analyze_longmemeval_cost,
    export_longmemeval_cost,
)
from vmp_memos.longmemeval.qa_runner import (
    LongMemEvalQARunConfig,
    run_longmemeval_qa,
)
from vmp_memos.longmemeval.retrieval_runner import (
    RetrievalSampleRecord,
    run_longmemeval_retrieval,
)


class CostFakeChatClient:
    """Return locally exact answers with deterministic usage."""

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        generation: LLMGenerationConfig | None = None,
    ) -> LLMResponse:
        prompt = messages[-1].content
        answer = "I don't know" if "favorite color" in prompt else "swimming"
        return LLMResponse(
            provider="fake-vllm",
            model="fake-reader",
            text=answer,
            finish_reason="stop",
            usage={"prompt_tokens": 100, "completion_tokens": 3},
        )


def test_cost_analysis_requires_aligned_qa_and_exports_all_formats(tmp_path) -> None:
    data_path = tmp_path / "longmemeval.json"
    data_path.write_text(
        json.dumps([_answerable_record(), _abstention_record()]),
        encoding="utf-8",
    )
    retrieval = run_longmemeval_retrieval(
        LongMemEvalRunConfig(
            data_path=data_path,
            methods=["bm25"],
            output_dir=tmp_path / "outputs",
        ),
        run_id="cost",
    )
    with pytest.raises(ValueError, match="QA artifacts"):
        analyze_longmemeval_cost(retrieval.run_dir)

    reader = LongMemEvalReader(
        CostFakeChatClient(),
        LongMemEvalReaderConfig(top_k=5),
    )
    run_longmemeval_qa(
        LongMemEvalQARunConfig(
            retrieval_run=retrieval.run_dir,
            methods=["bm25"],
            top_k=5,
        ),
        reader=reader,
    )
    report = analyze_longmemeval_cost(retrieval.run_dir)
    summary = report.methods["bm25"]

    assert report.qa_complete is True
    assert summary.samples == 2
    assert summary.correct_answers == 2
    assert summary.total_reader_input_tokens == 200
    assert summary.total_reader_output_tokens == 6
    assert summary.total_observed_tokens == 206
    assert summary.observed_tokens_per_correct == 103
    assert summary.framework_usage_coverage == 1.0
    assert summary.memory_retention_ratio == 1.0

    outputs = export_longmemeval_cost(
        retrieval.run_dir,
        output_dir=tmp_path / "tables",
    )
    assert len(outputs) == 4
    assert all(path.exists() for path in outputs.values())
    csv_text = outputs["cost_csv"].read_text(encoding="utf-8")
    assert "observed_tokens_per_correct" in csv_text
    assert "bm25" in csv_text


def test_missing_official_framework_usage_is_not_imputed_as_zero() -> None:
    record = RetrievalSampleRecord(
        question_id="q1",
        question_type="single_session_user",
        question="Question?",
        answer="Answer",
        method="mem0",
        is_abstention=False,
        adapter_stats={
            "memory_count": 1,
            "ingestion_sessions": 1,
            "storage_size_bytes": 0,
            "storage_size_is_estimate": True,
        },
    )

    summary = _summarize_cost(
        "mem0",
        retrieval_records=[record],
        qa_records=[],
    )

    assert summary.framework_llm_tokens is None
    assert summary.framework_usage_coverage == 0.0
    assert summary.storage_size_coverage == 0.0


def _answerable_record() -> dict[str, object]:
    return {
        "question_id": "q1",
        "question_type": "knowledge_update",
        "question": "What activity does Alex now prefer?",
        "answer": "swimming",
        "question_date": "2024-02-01",
        "haystack_session_ids": ["s_old", "s_new"],
        "haystack_dates": ["2024-01-01", "2024-01-20"],
        "haystack_sessions": [
            [{"role": "user", "content": "Alex liked hiking."}],
            [{"role": "user", "content": "Alex now prefers swimming."}],
        ],
        "answer_session_ids": ["s_new"],
        "has_answer": True,
    }


def _abstention_record() -> dict[str, object]:
    return {
        "question_id": "q2",
        "question_type": "single_session_user",
        "question": "What is Taylor's favorite color?",
        "answer": "I don't know",
        "question_date": "2024-02-01",
        "haystack_session_ids": ["s_other"],
        "haystack_dates": ["2024-01-01"],
        "haystack_sessions": [
            [{"role": "user", "content": "Taylor discussed weekend plans."}]
        ],
        "answer_session_ids": [],
        "has_answer": False,
    }
