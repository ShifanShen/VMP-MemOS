"""Synthetic QA runner tests that never contact a real model."""

from __future__ import annotations

import json
from collections.abc import Sequence

from vmp_memos.llm import (
    ChatMessage,
    LLMGenerationConfig,
    LLMResponse,
    LongMemEvalReader,
    LongMemEvalReaderConfig,
)
from vmp_memos.longmemeval import LongMemEvalRunConfig
from vmp_memos.longmemeval.qa_runner import (
    LongMemEvalQARunConfig,
    run_longmemeval_qa,
)
from vmp_memos.longmemeval.retrieval_runner import run_longmemeval_retrieval
from vmp_memos.longmemeval.tables import export_retrieval_tables


class FakeChatClient:
    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        generation: LLMGenerationConfig | None = None,
    ) -> LLMResponse:
        self.calls += 1
        self.prompts.append(messages[-1].content)
        answer = "I don't know" if "favorite color" in messages[-1].content else "swimming"
        return LLMResponse(
            provider="fake-vllm",
            model="fake-reader",
            text=answer,
            finish_reason="stop",
            usage={"prompt_tokens": 100, "completion_tokens": 3},
        )


def test_qa_runner_writes_metrics_hypotheses_and_resumes(tmp_path) -> None:
    retrieval_run = _build_retrieval_run(tmp_path)
    client = FakeChatClient()
    reader = LongMemEvalReader(
        client,
        LongMemEvalReaderConfig(top_k=5),
    )
    config = LongMemEvalQARunConfig(
        retrieval_run=retrieval_run,
        methods=["bm25"],
        top_k=5,
        reader_metadata={"provider": "fake-vllm", "model": "fake-reader"},
    )

    result = run_longmemeval_qa(config, reader=reader)

    assert client.calls == 2
    assert result.summaries["bm25"].metrics["contains_answer"] == 1.0
    assert result.summaries["bm25"].metrics["abstention_accuracy"] == 1.0
    hypotheses = _read_jsonl(retrieval_run / "hypotheses" / "bm25.jsonl")
    assert hypotheses == [
        {"question_id": "q1", "hypothesis": "swimming"},
        {"question_id": "q2", "hypothesis": "I don't know"},
    ]

    resumed = config.model_copy(update={"resume": True})
    run_longmemeval_qa(resumed, reader=reader)
    assert client.calls == 2


def test_retrieval_table_export_writes_all_formats(tmp_path) -> None:
    retrieval_run = _build_retrieval_run(tmp_path)

    outputs = export_retrieval_tables(
        retrieval_run,
        output_dir=tmp_path / "tables",
    )

    assert len(outputs) == 6
    assert all(path.exists() for path in outputs.values())
    csv_text = outputs["table1_retrieval_overall_csv"].read_text(encoding="utf-8")
    assert "recall_at_5" in csv_text
    assert "bm25" in csv_text


def _build_retrieval_run(tmp_path):
    data_path = tmp_path / "longmemeval.json"
    data_path.write_text(
        json.dumps([_answerable_record(), _abstention_record()]),
        encoding="utf-8",
    )
    result = run_longmemeval_retrieval(
        LongMemEvalRunConfig(
            data_path=data_path,
            methods=["bm25"],
            output_dir=tmp_path / "outputs",
        ),
        run_id="synthetic",
    )
    return result.run_dir


def _read_jsonl(path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _answerable_record() -> dict:
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


def _abstention_record() -> dict:
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
