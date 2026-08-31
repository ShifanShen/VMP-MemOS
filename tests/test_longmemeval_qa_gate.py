"""Tests for the Dev-only grounded-reader quality gate."""

from __future__ import annotations

import hashlib
import json

from vmp_memos.llm import (
    LONGMEMEVAL_FACT_READER_PROMPT_VERSION,
    LONGMEMEVAL_HYBRID_READER_PROMPT_VERSION,
    LONGMEMEVAL_QUERY_WINDOW_VERSION,
)
from vmp_memos.longmemeval.qa_gate import (
    LongMemEvalQAGateConfig,
    evaluate_longmemeval_qa_gate,
)
from vmp_memos.longmemeval.qa_runner import QASampleRecord


def test_qa_gate_passes_a_complete_grounded_dev_run(tmp_path) -> None:
    retrieval_run = _write_gate_fixture(tmp_path)
    result = evaluate_longmemeval_qa_gate(_gate_config(retrieval_run))

    assert result.status == "passed"
    assert all(result.checks.values())
    assert result.test_labels_used is False


def test_qa_gate_rejects_answerable_refusal_collapse(tmp_path) -> None:
    retrieval_run = _write_gate_fixture(tmp_path)
    summary_path = retrieval_run / "qa_v2_dev" / "method.summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["answerable_refusal_rate"] = 0.9
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = evaluate_longmemeval_qa_gate(_gate_config(retrieval_run))

    assert result.status == "failed"
    assert result.checks["answerable_refusal_rate"] is False


def test_hybrid_qa_gate_requires_query_window_evidence_coverage(tmp_path) -> None:
    retrieval_run = _write_gate_fixture(tmp_path)
    source = retrieval_run / "qa_v2_dev"
    qa_dir = source.rename(retrieval_run / "qa_v21_dev")
    records = [
        json.loads(line)
        for line in (qa_dir / "method.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    for record in records:
        record["reader_prompt_version"] = LONGMEMEVAL_HYBRID_READER_PROMPT_VERSION
        record["reader_evidence_mode"] = "reranker_facts_with_query_windows"
    (qa_dir / "method.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = json.loads(
        (qa_dir / "method.summary.json").read_text(encoding="utf-8")
    )
    summary["answerable_evidence_coverage_rate"] = 0.5
    (qa_dir / "method.summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    manifest = json.loads((qa_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["signature"]["qa_subdir"] = "qa_v21_dev"
    manifest["signature"]["protocol"]["prompt_version"] = (
        LONGMEMEVAL_HYBRID_READER_PROMPT_VERSION
    )
    manifest["signature"]["protocol"]["evidence_mode"] = (
        "reranker_facts_with_query_windows"
    )
    manifest["signature"]["protocol"]["query_windows"] = {
        "version": LONGMEMEVAL_QUERY_WINDOW_VERSION
    }
    manifest["signature"]["reader"] = {"protocol_selection_split": "dev"}
    (qa_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = evaluate_longmemeval_qa_gate(
        LongMemEvalQAGateConfig(
            retrieval_run=retrieval_run,
            qa_subdir="qa_v21_dev",
            methods=["method"],
            expected_prompt_version=LONGMEMEVAL_HYBRID_READER_PROMPT_VERSION,
            expected_evidence_mode="reranker_facts_with_query_windows",
        )
    )

    assert result.status == "failed"
    assert result.checks["answerable_evidence_coverage"] is False
    assert result.checks["query_window_version_matches"] is True
    assert result.checks["dev_selection_declared"] is True


def _write_gate_fixture(tmp_path):
    retrieval_run = tmp_path / "retrieval"
    qa_dir = retrieval_run / "qa_v2_dev"
    qa_dir.mkdir(parents=True)
    retrieval_manifest_path = retrieval_run / "manifest.json"
    retrieval_manifest_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "sample_count": 2,
                "split": {"name": "dev"},
                "test_labels_used": False,
            }
        ),
        encoding="utf-8",
    )
    records = [
        QASampleRecord(
            question_id="q1",
            question_type="single-session-user",
            method="method",
            question="What activity?",
            gold_answer="swimming",
            prediction="swimming",
            is_abstention=False,
            metrics={
                "normalized_exact_match": 1.0,
                "token_f1": 1.0,
                "contains_answer": 1.0,
            },
            reader_provider="fake-vllm",
            reader_model="fake-reader",
            reader_prompt_version=LONGMEMEVAL_FACT_READER_PROMPT_VERSION,
            reader_evidence_mode="reranker_facts",
            prompt_sha256="prompt-hash",
            evidence_profile_count=1,
            evidence_fact_count=1,
        ),
        QASampleRecord(
            question_id="q2_abs",
            question_type="single-session-user",
            method="method",
            question="What color?",
            gold_answer="I don't know",
            prediction="I don't know",
            is_abstention=True,
            metrics={"abstention_accuracy": 1.0},
            reader_provider="fake-vllm",
            reader_model="fake-reader",
            reader_prompt_version=LONGMEMEVAL_FACT_READER_PROMPT_VERSION,
            reader_evidence_mode="reranker_facts",
            prompt_sha256="prompt-hash-2",
        ),
    ]
    (qa_dir / "method.jsonl").write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
    )
    (qa_dir / "method.summary.json").write_text(
        json.dumps(
            {
                "processed_questions": 2,
                "answerable_questions": 1,
                "abstention_questions": 1,
                "answerable_refusal_rate": 0.0,
                "answerable_fact_coverage_rate": 1.0,
                "metrics": {
                    "normalized_exact_match": 1.0,
                    "token_f1": 1.0,
                    "contains_answer": 1.0,
                    "abstention_accuracy": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    retrieval_sha = hashlib.sha256(retrieval_manifest_path.read_bytes()).hexdigest()
    (qa_dir / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "retrieval_manifest_sha256": retrieval_sha,
                "signature": {
                    "qa_subdir": "qa_v2_dev",
                    "methods": ["method"],
                    "protocol": {
                        "prompt_version": LONGMEMEVAL_FACT_READER_PROMPT_VERSION,
                        "evidence_mode": "reranker_facts",
                        "gold_answers_visible_to_reader": False,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return retrieval_run


def _gate_config(retrieval_run):
    return LongMemEvalQAGateConfig(
        retrieval_run=retrieval_run,
        qa_subdir="qa_v2_dev",
        methods=["method"],
        expected_prompt_version=LONGMEMEVAL_FACT_READER_PROMPT_VERSION,
        expected_evidence_mode="reranker_facts",
    )
