"""End-to-end synthetic tests for the LongMemEval retrieval runner."""

from __future__ import annotations

import json

from vmp_memos.longmemeval import LongMemEvalRunConfig
from vmp_memos.longmemeval.retrieval_runner import run_longmemeval_retrieval


def test_retrieval_runner_writes_manifest_records_and_summary(tmp_path) -> None:
    data_path = tmp_path / "longmemeval.json"
    data_path.write_text(
        json.dumps([_answerable_record(), _abstention_record()]),
        encoding="utf-8",
    )
    config = LongMemEvalRunConfig(
        data_path=data_path,
        methods=["bm25", "empty"],
        top_k=5,
        retrieval_depth=10,
        output_dir=tmp_path / "outputs",
    )

    result = run_longmemeval_retrieval(config, run_id="synthetic")

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["sample_count"] == 2
    assert len(manifest["data_sha256"]) == 64

    bm25_records = _read_jsonl(result.run_dir / "bm25" / "retrieval.jsonl")
    assert len(bm25_records) == 2
    assert bm25_records[0]["retrieved_session_ids"][0] == "s_new"
    assert bm25_records[0]["metrics"]["recall_at_1"] == 1.0
    assert bm25_records[1]["evaluation_skipped"] is True
    assert bm25_records[1]["skip_reason"] == "abstention"
    assert result.summaries["bm25"].evaluated_questions == 1
    assert result.summaries["empty"].metrics["recall_at_5"] == 0.0


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
            [
                {"role": "user", "content": "Alex said hiking was fun."},
                {"role": "assistant", "content": "Alex liked hiking."},
            ],
            [
                {"role": "user", "content": "Alex now prefers swimming."},
                {"role": "assistant", "content": "Alex prefers swimming."},
            ],
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
