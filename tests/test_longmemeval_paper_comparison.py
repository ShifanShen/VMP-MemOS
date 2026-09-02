"""Tests for immutable cross-run LongMemEval paper comparisons."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from vmp_memos.longmemeval.cost import analyze_longmemeval_cost
from vmp_memos.longmemeval.official_qa import (
    OfficialAutoEvalLabel,
    OfficialJudgeRecord,
)
from vmp_memos.longmemeval.paper_comparison import merge_longmemeval_paper_runs
from vmp_memos.longmemeval.paper_efficiency import export_official_judge_efficiency
from vmp_memos.longmemeval.qa_runner import QASampleRecord
from vmp_memos.longmemeval.qa_statistics import (
    LongMemEvalQAReportConfig,
    export_longmemeval_qa_report,
)
from vmp_memos.longmemeval.retrieval_runner import RetrievalSampleRecord
from vmp_memos.longmemeval.tables import export_retrieval_tables

QA_SUBDIR = "qa_v21_test"
JUDGE_SUBDIR = "official_judge_local_vllm_v1"
VMP_METHOD = "vmp_hierarchical__vllm_boundary"
MEM0_METHOD = "mem0_official__vllm_boundary"


def test_merge_paper_runs_supports_retrieval_cost_and_qa_exports(tmp_path) -> None:
    vmp = _source_run(tmp_path / "vmp", VMP_METHOD, correct=True)
    mem0 = _source_run(tmp_path / "mem0", MEM0_METHOD, correct=False)

    output = merge_longmemeval_paper_runs(
        [vmp, mem0],
        output_dir=tmp_path / "comparison",
    )

    manifest = _read_json(output / "manifest.json")
    assert manifest["config"]["methods"] == [VMP_METHOD, MEM0_METHOD]
    assert manifest["signature"]["merged_without_rerunning_models"] is True
    qa_manifest = _read_json(output / QA_SUBDIR / "manifest.json")
    assert qa_manifest["retrieval_manifest_sha256"] == _sha256(
        output / "manifest.json"
    )
    judge_manifest = _read_json(
        output / QA_SUBDIR / JUDGE_SUBDIR / "manifest.json"
    )
    assert judge_manifest["signature"]["methods"] == [VMP_METHOD, MEM0_METHOD]

    retrieval_outputs = export_retrieval_tables(
        output,
        output_dir=output / "paper",
    )
    assert all(path.exists() for path in retrieval_outputs.values())
    cost = analyze_longmemeval_cost(
        output,
        qa_subdir=QA_SUBDIR,
        reference_method=VMP_METHOD,
    )
    assert cost.reference_method == VMP_METHOD
    assert cost.methods[MEM0_METHOD].framework_llm_tokens is None
    qa_report = export_longmemeval_qa_report(
        LongMemEvalQAReportConfig(
            judge_run=output / QA_SUBDIR / JUDGE_SUBDIR,
            reference_method=VMP_METHOD,
            bootstrap_samples=100,
        )
    )
    assert qa_report.comparisons[0].accuracy_delta == 1.0
    efficiency_outputs = export_official_judge_efficiency(
        output,
        qa_subdir=QA_SUBDIR,
        judge_subdir=JUDGE_SUBDIR,
        reference_method=VMP_METHOD,
        output_dir=output / "paper",
    )
    assert all(path.exists() for path in efficiency_outputs.values())
    efficiency = _read_json(
        efficiency_outputs["official_judge_efficiency_json"]
    )
    assert efficiency["methods"][0]["token_accounting_complete"] is True
    assert efficiency["methods"][1]["token_accounting_complete"] is False


def test_merge_paper_runs_rejects_reader_model_drift_without_partial_output(
    tmp_path,
) -> None:
    vmp = _source_run(tmp_path / "vmp", VMP_METHOD, correct=True)
    mem0 = _source_run(
        tmp_path / "mem0",
        MEM0_METHOD,
        correct=False,
        model="different-reader",
    )
    output = tmp_path / "comparison"

    with pytest.raises(ValueError, match="not comparable"):
        merge_longmemeval_paper_runs([vmp, mem0], output_dir=output)

    assert not output.exists()


def _source_run(
    path: Path,
    method: str,
    *,
    correct: bool,
    model: str = "Qwen/Qwen2.5-7B-Instruct",
) -> Path:
    method_dir = path / method
    method_dir.mkdir(parents=True)
    retrieval = RetrievalSampleRecord(
        question_id="q1",
        question_type="multi-session",
        question="What activity does Alex prefer?",
        answer="swimming",
        method=method,
        is_abstention=False,
        gold_session_ids=["s1"],
        retrieved_session_ids=["s1"],
        metrics={"recall_all@5": 1.0, "mrr": 1.0},
        adapter_stats={
            "memory_count": 2,
            "total_ingest_latency_ms": 10.0,
            "total_retrieval_latency_ms": 5.0,
        },
        rerank_metadata={"input_tokens": 20, "output_tokens": 2},
    )
    _write_jsonl(method_dir / "retrieval.jsonl", [retrieval])
    rerank_manifest = {
        "status": "completed",
        "dataset": "longmemeval-cleaned",
        "data_sha256": "dataset-sha",
        "sample_count": 1,
        "split": {"name": "test", "split_id": "split-42", "question_count": 1},
        "signature": {
            "methods": [method.split("__", maxsplit=1)[0]],
            "reranked_methods": [method],
            "limit": None,
            "question_ids": [],
            "reranker": {
                "prompt_version": "vmp_v64_high_recall_atomic_fact_extractor_v4",
                "candidate_count": 10,
                "output_top_k": 5,
                "generation": {"max_tokens": 512, "temperature": 0.0},
            },
            "test_labels_used": False,
        },
        "config": {"methods": [method]},
        "observed_reranker": {"provider": "vllm", "model": model},
    }
    manifest_path = path / "manifest.json"
    _write_json(manifest_path, rerank_manifest)

    qa_dir = path / QA_SUBDIR
    qa_dir.mkdir()
    qa = QASampleRecord(
        question_id="q1",
        question_type="multi-session",
        method=method,
        question="What activity does Alex prefer?",
        gold_answer="swimming",
        prediction="swimming" if correct else "hiking",
        is_abstention=False,
        metrics={"normalized_exact_match": float(correct)},
        reader_provider="vllm",
        reader_model=model,
        prompt_sha256="reader-prompt-sha",
        reader_input_tokens=100,
        reader_output_tokens=2,
        end_to_end_latency_ms=20.0,
    )
    _write_jsonl(qa_dir / f"{method}.jsonl", [qa])
    qa_manifest_path = qa_dir / "manifest.json"
    qa_manifest = {
        "status": "completed",
        "signature": {
            "methods": [method],
            "top_k": 5,
            "limit": None,
            "qa_subdir": QA_SUBDIR,
            "generation": {"max_tokens": 256, "temperature": 0.0},
            "protocol": {
                "prompt_version": "longmemeval_hybrid_evidence_reader_v21",
                "gold_answers_visible_to_reader": False,
            },
            "reader": {"provider": "vllm", "model": model},
        },
        "retrieval_manifest_sha256": _sha256(manifest_path),
        "observed_reader": {"provider": "vllm", "model": model},
    }
    _write_json(qa_manifest_path, qa_manifest)

    judge_dir = qa_dir / JUDGE_SUBDIR
    judge_dir.mkdir()
    judge = OfficialJudgeRecord(
        question_id="q1",
        question_type="multi-session",
        method=method,
        hypothesis=qa.prediction,
        autoeval_label=OfficialAutoEvalLabel(model=model, label=correct),
        judge_provider="vllm",
        judge_model=model,
        judge_response="yes" if correct else "no",
        parse_status="yes" if correct else "no",
        prompt_sha256="judge-prompt-sha",
    )
    _write_jsonl(judge_dir / f"{method}.jsonl", [judge])
    _write_json(
        judge_dir / "manifest.json",
        {
            "status": "completed",
            "signature": {
                "qa_manifest_sha256": _sha256(qa_manifest_path),
                "reference_data_sha256": "dataset-sha",
                "methods": [method],
                "limit": None,
                "score_kind": "official_prompt_local_vllm_judge",
                "directly_comparable_to_published_gpt4o_scores": False,
                "protocol": {"prompt_version": "longmemeval_official_qa_judge_v1"},
                "generation": {"max_tokens": 10, "temperature": 0.0},
                "judge": {"provider": "vllm", "model": model},
            },
            "observed_judge": {"provider": "vllm", "model": model},
        },
    )
    return path


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(
    path: Path,
    records: list[RetrievalSampleRecord | QASampleRecord | OfficialJudgeRecord],
) -> None:
    path.write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
