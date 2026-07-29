"""Tests for the strict Dev-only VMP-v5.3 boundary gate."""

from __future__ import annotations

import hashlib
import json

import pytest

from vmp_memos.longmemeval.boundary_gate import evaluate_v53_gate
from vmp_memos.longmemeval.rerank_gate import write_gate_receipt


def test_v53_gate_requires_large_zero_regression_dev_gain(tmp_path) -> None:
    candidate_run, rerank_run = _write_runs(tmp_path)

    report = evaluate_v53_gate(candidate_run, rerank_run)

    assert report["status"] == "passed"
    assert report["checks"]["shared_two_stage_prompts"] is True
    assert report["checks"]["zero_or_bounded_regression"] is True
    assert report["metrics"]["delta_vs_raw_v5"] == pytest.approx(0.04)
    receipt = write_gate_receipt(report, tmp_path / "v53-gate.json")
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "passed"


def test_v53_gate_rejects_even_one_regressed_dev_question(tmp_path) -> None:
    candidate_run, rerank_run = _write_runs(tmp_path, regressed_questions=1)

    report = evaluate_v53_gate(candidate_run, rerank_run)

    assert report["status"] == "failed"
    assert report["checks"]["zero_or_bounded_regression"] is False
    with pytest.raises(ValueError, match="failed report"):
        write_gate_receipt(report, tmp_path / "v53-gate.json")


def test_v531_gate_accepts_only_the_symbolic_prompt_version(tmp_path) -> None:
    prompt_version = "vmp_v531_symbolic_boundary_v1"
    candidate_run, rerank_run = _write_runs(
        tmp_path,
        boundary_prompt_version=prompt_version,
    )

    report = evaluate_v53_gate(
        candidate_run,
        rerank_run,
        expected_boundary_prompt_version=prompt_version,
    )

    assert report["status"] == "passed"
    assert report["reranker"]["expected_boundary_prompt_version"] == prompt_version


def test_v532_gate_accepts_only_the_atomic_set_prompt_version(tmp_path) -> None:
    prompt_version = "vmp_v532_atomic_set_boundary_v1"
    candidate_run, rerank_run = _write_runs(
        tmp_path,
        boundary_prompt_version=prompt_version,
    )

    report = evaluate_v53_gate(
        candidate_run,
        rerank_run,
        expected_boundary_prompt_version=prompt_version,
    )

    assert report["status"] == "passed"
    assert report["reranker"]["expected_boundary_prompt_version"] == prompt_version


def test_v54_gate_accepts_symbolic_span_selector_and_boundary(tmp_path) -> None:
    selector_prompt_version = "vmp_v54_symbolic_span_selector_v1"
    boundary_prompt_version = "vmp_v54_symbolic_span_boundary_v1"
    candidate_run, rerank_run = _write_runs(
        tmp_path,
        selector_prompt_version=selector_prompt_version,
        boundary_prompt_version=boundary_prompt_version,
    )

    report = evaluate_v53_gate(
        candidate_run,
        rerank_run,
        expected_selector_prompt_version=selector_prompt_version,
        expected_boundary_prompt_version=boundary_prompt_version,
    )

    assert report["status"] == "passed"
    assert report["reranker"]["expected_selector_prompt_version"] == (
        selector_prompt_version
    )
    assert report["reranker"]["expected_boundary_prompt_version"] == (
        boundary_prompt_version
    )


def _write_runs(
    tmp_path,
    *,
    regressed_questions: int = 0,
    selector_prompt_version: str = "vmp_v52_evidence_set_v1",
    boundary_prompt_version: str = "vmp_v53_selective_boundary_v1",
):
    candidate_run = tmp_path / "candidates"
    rerank_run = tmp_path / "reranked"
    candidate_run.mkdir()
    rerank_run.mkdir()
    candidate_manifest = candidate_run / "manifest.json"
    _write_json(
        candidate_manifest,
        {
            "status": "completed",
            "sample_count": 100,
            "split": {"name": "dev"},
            "config": {"allow_dev_model_selection": True},
        },
    )
    _write_json(
        rerank_run / "manifest.json",
        {
            "status": "completed",
            "split": {"name": "dev"},
            "source_retrieval_manifest_sha256": _sha256(candidate_manifest),
            "fairness": {
                "two_stage_boundary_verification": True,
                "symbolic_selector_labels": (
                    selector_prompt_version == "vmp_v54_symbolic_span_selector_v1"
                ),
                "selector_evidence_span_binding": (
                    selector_prompt_version == "vmp_v54_symbolic_span_selector_v1"
                ),
                "boundary_evidence_span_binding": (
                    boundary_prompt_version == "vmp_v54_symbolic_span_boundary_v1"
                ),
            },
            "test_labels_used": False,
        },
    )
    _write_summary(candidate_run / "vmp_hierarchical" / "summary.json", 0.90)
    _write_summary(candidate_run / "vmp_tuned" / "summary.json", 0.89)
    _write_summary(
        rerank_run / "vmp_hierarchical__vllm_boundary" / "summary.json",
        0.94,
        reranked=True,
        recovered_questions=4,
        regressed_questions=regressed_questions,
        selector_prompt_version=selector_prompt_version,
        boundary_prompt_version=boundary_prompt_version,
    )
    _write_summary(
        rerank_run / "vmp_tuned__vllm_boundary" / "summary.json",
        0.90,
        reranked=True,
        recovered_questions=3,
        selector_prompt_version=selector_prompt_version,
        boundary_prompt_version=boundary_prompt_version,
    )
    return candidate_run, rerank_run


def _write_summary(
    path,
    recall: float,
    *,
    reranked: bool = False,
    recovered_questions: int = 0,
    regressed_questions: int = 0,
    selector_prompt_version: str = "vmp_v52_evidence_set_v1",
    boundary_prompt_version: str = "vmp_v53_selective_boundary_v1",
) -> None:
    payload = {
        "processed_questions": 100,
        "evaluated_questions": 94,
        "metrics": {"recall_all@5": recall},
        "by_question_type": {
            "multi-session": {"recall_all@5": recall},
            "temporal-reasoning": {"recall_all@5": recall},
        },
    }
    if reranked:
        payload.update(
            {
                "reranker_provider": "vllm",
                "reranker_model": "Qwen/Qwen2.5-7B-Instruct",
                "prompt_version": selector_prompt_version,
                "boundary_prompt_version": boundary_prompt_version,
                "boundary_verification": True,
                "candidate_count": 30,
                "min_observed_candidate_count": 30,
                "output_top_k": 5,
                "protected_top_n": 3,
                "boundary_protected_top_n": 3,
                "parse_fallback_rate": 0.0,
                "boundary_parse_fallback_rate": 0.0,
                "recovered_questions": recovered_questions,
                "regressed_questions": regressed_questions,
            }
        )
    _write_json(path, payload)


def _write_json(path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
