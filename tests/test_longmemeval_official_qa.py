"""Network-free tests for the official-prompt-compatible local QA judge."""

from __future__ import annotations

import json
from collections.abc import Sequence

from vmp_memos.llm import ChatMessage, LLMGenerationConfig, LLMResponse
from vmp_memos.longmemeval.official_qa import (
    LOCAL_JUDGE_SCORE_KIND,
    LongMemEvalOfficialJudgeConfig,
    build_official_qa_judge_prompt,
    parse_official_qa_judge_response,
    run_longmemeval_official_judge,
)
from vmp_memos.longmemeval.qa_runner import QASampleRecord


class FakeJudgeClient:
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
        prompt = messages[-1].content
        self.prompts.append(prompt)
        text = "yes" if "expected-positive" in prompt else "no"
        return LLMResponse(
            provider="fake-vllm",
            model="shared-local-judge",
            text=text,
            finish_reason="stop",
            usage={"prompt_tokens": 80, "completion_tokens": 1},
        )


def test_official_prompt_builder_uses_all_task_specific_protocols() -> None:
    temporal = build_official_qa_judge_prompt(
        "temporal-reasoning",
        "How many days?",
        "18 days",
        "19 days",
        is_abstention=False,
    )
    preference = build_official_qa_judge_prompt(
        "single-session-preference",
        "What should I choose?",
        "Use the user's hiking preference",
        "Choose hiking boots",
        is_abstention=False,
    )
    abstention = build_official_qa_judge_prompt(
        "multi-session",
        "What color?",
        "The information is incomplete.",
        "I don't know",
        is_abstention=True,
    )

    assert "off-by-one errors" in temporal
    assert "Rubric:" in preference
    assert "unanswerable question" in abstention
    assert parse_official_qa_judge_response("YES") == (True, "yes")
    assert parse_official_qa_judge_response("no") == (False, "no")
    assert parse_official_qa_judge_response("yes and no") == (True, "ambiguous")
    assert parse_official_qa_judge_response("maybe") == (False, "missing")


def test_official_judge_writes_summaries_provenance_and_resumes(tmp_path) -> None:
    qa_run, reference_data = _write_qa_fixture(tmp_path)
    client = FakeJudgeClient()
    generation = LLMGenerationConfig(max_tokens=10, temperature=0.0, top_p=1.0)
    config = LongMemEvalOfficialJudgeConfig(
        qa_run=qa_run,
        reference_data=reference_data,
        methods=["reference", "comparator"],
        judge_metadata={"provider": "fake-vllm", "model": "shared-local-judge"},
    )

    result = run_longmemeval_official_judge(
        config,
        client=client,
        generation=generation,
    )

    assert client.calls == 4
    assert result.summaries["reference"].accuracy == 1.0
    assert result.summaries["comparator"].accuracy == 0.5
    assert result.summaries["reference"].abstention_accuracy == 1.0
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["signature"]["score_kind"] == LOCAL_JUDGE_SCORE_KIND
    assert manifest["signature"]["directly_comparable_to_published_gpt4o_scores"] is False
    assert manifest["signature"]["protocol"]["gold_answers_visible_to_reader"] is False
    assert manifest["signature"]["protocol"]["gold_answers_visible_to_judge"] is True
    records = _read_jsonl(result.judge_dir / "reference.jsonl")
    assert records[0]["autoeval_label"] == {
        "model": "shared-local-judge",
        "label": True,
    }
    assert records[0]["judge_input_tokens"] == 80

    run_longmemeval_official_judge(
        config.model_copy(update={"resume": True}),
        client=client,
        generation=generation,
    )
    assert client.calls == 4


def _write_qa_fixture(tmp_path):
    reference_data = tmp_path / "longmemeval.json"
    reference_data.write_text(
        json.dumps([_reference_record(), _abstention_record()]),
        encoding="utf-8",
    )
    qa_run = tmp_path / "qa_v21_test"
    qa_run.mkdir()
    methods = ["reference", "comparator"]
    (qa_run / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "signature": {
                    "methods": methods,
                    "protocol": {
                        "prompt_version": "reader-v1",
                        "gold_answers_visible_to_reader": False,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    hypotheses = {
        "reference": ["expected-positive", "expected-positive"],
        "comparator": ["expected-negative", "expected-positive"],
    }
    for method in methods:
        records = [
            _qa_record(method, "q1", "knowledge-update", hypotheses[method][0]),
            _qa_record(method, "q2_abs", "multi-session", hypotheses[method][1], True),
        ]
        (qa_run / f"{method}.jsonl").write_text(
            "".join(record.model_dump_json() + "\n" for record in records),
            encoding="utf-8",
        )
    return qa_run, reference_data


def _qa_record(
    method: str,
    question_id: str,
    question_type: str,
    prediction: str,
    is_abstention: bool = False,
) -> QASampleRecord:
    return QASampleRecord(
        question_id=question_id,
        question_type=question_type,
        method=method,
        question=("What activity?" if question_id == "q1" else "What color?"),
        gold_answer="tampered QA gold is ignored",
        prediction=prediction,
        is_abstention=is_abstention,
        metrics={},
        reader_provider="fake-vllm",
        reader_model="shared-reader",
        prompt_sha256=f"prompt-{method}-{question_id}",
    )


def _reference_record() -> dict:
    return {
        "question_id": "q1",
        "question_type": "knowledge-update",
        "question": "What activity?",
        "answer": "swimming",
        "question_date": "2024-02-01",
        "haystack_session_ids": ["s1"],
        "haystack_dates": ["2024-01-01"],
        "haystack_sessions": [[{"role": "user", "content": "Swimming."}]],
        "answer_session_ids": ["s1"],
    }


def _abstention_record() -> dict:
    return {
        "question_id": "q2_abs",
        "question_type": "multi-session",
        "question": "What color?",
        "answer": "The information is incomplete.",
        "question_date": "2024-02-01",
        "haystack_session_ids": ["s2"],
        "haystack_dates": ["2024-01-01"],
        "haystack_sessions": [[{"role": "user", "content": "No color."}]],
        "answer_session_ids": ["s2"],
    }


def _read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
