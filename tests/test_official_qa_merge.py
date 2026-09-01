"""Tests for cross-run official-judge comparison artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vmp_memos.longmemeval.official_qa import (
    OfficialAutoEvalLabel,
    OfficialJudgeRecord,
)
from vmp_memos.longmemeval.official_qa_merge import merge_official_judge_runs
from vmp_memos.longmemeval.qa_statistics import (
    LongMemEvalQAReportConfig,
    export_longmemeval_qa_report,
)


def test_merge_official_judges_preserves_methods_and_shared_provenance(tmp_path) -> None:
    first = _judge_run(tmp_path / "first", "vmp", [True, False])
    second = _judge_run(tmp_path / "second", "mem0", [False, False])

    output = merge_official_judge_runs(
        [first, second],
        output_dir=tmp_path / "comparison",
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["signature"]["methods"] == ["vmp", "mem0"]
    assert manifest["signature"]["question_count"] == 2
    assert manifest["signature"]["merged_without_rejudging"] is True
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["methods"]["vmp"]["accuracy"] == 0.5
    assert summary["methods"]["mem0"]["accuracy"] == 0.0
    report = export_longmemeval_qa_report(
        LongMemEvalQAReportConfig(
            judge_run=output,
            reference_method="vmp",
            bootstrap_samples=100,
        )
    )
    assert report.comparisons[0].comparator_method == "mem0"
    assert report.comparisons[0].accuracy_delta == 0.5


def test_merge_official_judges_rejects_different_judge_model(tmp_path) -> None:
    first = _judge_run(tmp_path / "first", "vmp", [True])
    second = _judge_run(
        tmp_path / "second",
        "mem0",
        [True],
        model="different-model",
    )

    with pytest.raises(ValueError, match="not comparable"):
        merge_official_judge_runs(
            [first, second],
            output_dir=tmp_path / "comparison",
        )


def test_merge_official_judges_rejects_question_coverage_drift(tmp_path) -> None:
    first = _judge_run(tmp_path / "first", "vmp", [True, False])
    second = _judge_run(tmp_path / "second", "mem0", [True, True])
    records_path = second / "mem0.jsonl"
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
    ]
    records[1]["question_id"] = "different-question"
    records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ordered question coverage"):
        merge_official_judge_runs(
            [first, second],
            output_dir=tmp_path / "comparison",
        )


def _judge_run(
    path: Path,
    method: str,
    labels: list[bool],
    *,
    model: str = "Qwen/Qwen2.5-7B-Instruct",
) -> Path:
    path.mkdir()
    qa_path = path / "qa"
    judge_path = qa_path / "judge"
    judge_path.mkdir(parents=True)
    rerank_manifest_path = path / "manifest.json"
    rerank_manifest_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "dataset": "longmemeval-cleaned",
                "data_sha256": "dataset-sha",
                "sample_count": len(labels),
                "split": {
                    "name": "test",
                    "split_id": "split-42",
                    "question_count": len(labels),
                },
                "signature": {
                    "limit": None,
                    "question_ids": [],
                    "reranker": {"prompt_version": "vmp-v64"},
                    "test_labels_used": False,
                },
                "observed_reranker": {"provider": "vllm", "model": model},
            }
        ),
        encoding="utf-8",
    )
    qa_manifest_path = qa_path / "manifest.json"
    qa_manifest_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "signature": {
                    "top_k": 5,
                    "limit": None,
                    "generation": {
                        "max_tokens": 256,
                        "temperature": 0.0,
                        "top_p": 1.0,
                    },
                    "protocol": {"prompt_version": "reader-v21"},
                    "reader": {"provider": "vllm", "model": model},
                },
                "retrieval_manifest_sha256": _sha256(rerank_manifest_path),
                "observed_reader": {"provider": "vllm", "model": model},
            }
        ),
        encoding="utf-8",
    )
    records = [
        OfficialJudgeRecord(
            question_id=f"q{index}",
            question_type="multi-session",
            method=method,
            hypothesis="answer",
            autoeval_label=OfficialAutoEvalLabel(model=model, label=label),
            judge_provider="vllm",
            judge_model=model,
            judge_response="yes" if label else "no",
            parse_status="yes" if label else "no",
            prompt_sha256=f"sha-{index}",
        )
        for index, label in enumerate(labels, start=1)
    ]
    (judge_path / f"{method}.jsonl").write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
    )
    signature = {
        "qa_manifest_sha256": _sha256(qa_manifest_path),
        "reference_data_sha256": "dataset-sha",
        "methods": [method],
        "limit": None,
        "score_kind": "official_prompt_local_vllm_judge",
        "directly_comparable_to_published_gpt4o_scores": False,
        "protocol": {"prompt_version": "longmemeval_official_qa_judge_v1"},
        "generation": {"max_tokens": 10, "temperature": 0.0, "top_p": 1.0},
        "judge": {"provider": "vllm", "model": model},
    }
    (judge_path / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "signature": signature,
                "observed_judge": {"provider": "vllm", "model": model},
            }
        ),
        encoding="utf-8",
    )
    return judge_path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
