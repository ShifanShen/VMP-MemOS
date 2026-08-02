"""Tests for the Dev-only VMP rerank diagnostic report."""

from __future__ import annotations

import json

import pytest
from scripts.analyze_vmp_rerank import analyze_dev_rerank


def test_analyzer_reports_candidate_oracle_recoverable_failure(tmp_path) -> None:
    run = tmp_path / "run"
    method = "vmp_hierarchical__vllm_boundary"
    method_dir = run / method
    method_dir.mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps({"status": "completed", "split": {"name": "dev"}}),
        encoding="utf-8",
    )
    record = {
        "question_id": "q1",
        "question_type": "multi-session",
        "question": "Which two subscriptions are current?",
        "gold_session_ids": ["s7", "s10"],
        "rerank_metadata": {
            "transition_vs_source": "stable_failure",
            "candidate_oracle_recoverable": True,
            "question_evidence_plan": {"operator": "list"},
            "coverage_gain": 0.5,
            "candidate_label_session_ids": {"C07": "s7", "C10": "s10"},
            "selected_session_ids": ["s1", "s2", "s3", "s7", "s9"],
            "coverage_selection": {"promoted_candidate_labels": ["C07"]},
            "selector_span_binding_failures": ["C10:F1:span_not_grounded"],
        },
    }
    (method_dir / "retrieval.jsonl").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )

    report = analyze_dev_rerank(run, method=method)

    assert report["transitions"] == {"stable_failure": 1}
    assert report["positive_coverage_gain_questions"] == 1
    assert report["candidate_oracle_recoverable_stable_failure_count"] == 1
    case = report["candidate_oracle_recoverable_stable_failures"][0]
    assert case["missing_gold_candidate_labels"] == ["C10"]
    assert report["extraction_failure_reasons"] == {"F1:span_not_grounded": 1}


def test_analyzer_refuses_sealed_test_split(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps({"status": "completed", "split": {"name": "test"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="restricted to the Dev split"):
        analyze_dev_rerank(run, method="method")
