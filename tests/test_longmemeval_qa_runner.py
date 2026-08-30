"""Synthetic QA runner tests that never contact a real model."""

from __future__ import annotations

import json
from collections.abc import Sequence

from vmp_memos.llm import (
    LONGMEMEVAL_FACT_READER_PROMPT_VERSION,
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
        {"question_id": "q2_abs", "hypothesis": "I don't know"},
    ]

    resumed = config.model_copy(update={"resume": True})
    run_longmemeval_qa(resumed, reader=reader)
    assert client.calls == 2


def test_qa_runner_writes_fact_reader_to_a_versioned_subdirectory(tmp_path) -> None:
    retrieval_run = _build_retrieval_run(tmp_path)
    _add_reranker_facts(retrieval_run / "bm25" / "retrieval.jsonl")
    client = FakeChatClient()
    reader = LongMemEvalReader(
        client,
        LongMemEvalReaderConfig(
            top_k=5,
            prompt_version=LONGMEMEVAL_FACT_READER_PROMPT_VERSION,
            evidence_mode="reranker_facts",
        ),
    )
    config = LongMemEvalQARunConfig(
        retrieval_run=retrieval_run,
        methods=["bm25"],
        top_k=5,
        qa_subdir="qa_v2_dev",
        reader_metadata={"provider": "fake-vllm", "model": "fake-reader"},
    )

    result = run_longmemeval_qa(config, reader=reader)

    assert result.qa_dir == retrieval_run / "qa_v2_dev"
    assert result.summaries["bm25"].answerable_fact_coverage_rate == 1.0
    records = _read_jsonl(result.qa_dir / "bm25.jsonl")
    assert records[0]["reader_prompt_version"] == LONGMEMEVAL_FACT_READER_PROMPT_VERSION
    assert records[0]["evidence_fact_count"] == 1
    assert (result.qa_dir / "hypotheses" / "bm25.jsonl").exists()

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
    assert "recall_all@5" in csv_text
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


def _add_reranker_facts(path) -> None:
    records = _read_jsonl(path)
    records[0]["rerank_metadata"] = {
        "selector_evidence_selections": [
            {
                "candidate_label": "C01",
                "session_id": "s_new",
                "rank": 1,
                "candidate_relevant": True,
                "lexical_overlap": 1.0,
                "facts": [
                    {
                        "fact_id": "C01:F01",
                        "entity": "Alex",
                        "relation": "now_prefers",
                        "value": "swimming",
                        "temporal_anchor": "2024-01-20",
                        "supports_needs": ["N1"],
                        "evidence_spans": ["X:S01"],
                        "confidence": "high",
                    }
                ],
                "extraction_fallback": False,
                "extraction_failures": [],
            }
        ]
    }
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


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
        "question_id": "q2_abs",
        "question_type": "single_session_user",
        "question": "What is Taylor's favorite color?",
        "answer": "The information provided is not enough.",
        "question_date": "2024-02-01",
        "haystack_session_ids": ["s_other"],
        "haystack_dates": ["2024-01-01"],
        "haystack_sessions": [
            [{"role": "user", "content": "Taylor discussed weekend plans."}]
        ],
        "answer_session_ids": ["s_other"],
    }
