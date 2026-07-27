"""Tests for the sealed Dev-only VMP-v5.2 quality gate."""

from __future__ import annotations

import hashlib
import json

import pytest

from vmp_memos.longmemeval.rerank_gate import (
    evaluate_v52_gate,
    write_gate_receipt,
)


def test_v52_gate_requires_shared_vllm_gain_and_writes_receipt(tmp_path) -> None:
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
            "test_labels_used": False,
        },
    )
    _write_summary(candidate_run / "vmp_hierarchical" / "summary.json", 0.90)
    _write_summary(candidate_run / "vmp_tuned" / "summary.json", 0.89)
    _write_summary(
        rerank_run / "vmp_hierarchical__vllm_rerank" / "summary.json",
        0.93,
        reranked=True,
    )
    _write_summary(
        rerank_run / "vmp_tuned__vllm_rerank" / "summary.json",
        0.90,
        reranked=True,
    )

    report = evaluate_v52_gate(candidate_run, rerank_run)

    assert report["status"] == "passed"
    assert report["checks"]["shared_local_vllm"] is True
    assert report["metrics"]["delta_vs_raw_v5"] == pytest.approx(0.03)
    receipt = write_gate_receipt(report, tmp_path / "gate.json")
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "passed"


def test_v52_gate_fails_on_parser_fallbacks(tmp_path) -> None:
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
            "test_labels_used": False,
        },
    )
    _write_summary(candidate_run / "vmp_hierarchical" / "summary.json", 0.90)
    _write_summary(candidate_run / "vmp_tuned" / "summary.json", 0.89)
    _write_summary(
        rerank_run / "vmp_hierarchical__vllm_rerank" / "summary.json",
        0.93,
        reranked=True,
        fallback_rate=0.10,
    )
    _write_summary(
        rerank_run / "vmp_tuned__vllm_rerank" / "summary.json",
        0.90,
        reranked=True,
    )

    report = evaluate_v52_gate(candidate_run, rerank_run)

    assert report["status"] == "failed"
    assert report["checks"]["parse_fallback_rate"] is False
    with pytest.raises(ValueError, match="failed report"):
        write_gate_receipt(report, tmp_path / "gate.json")


def _write_summary(
    path,
    recall: float,
    *,
    reranked: bool = False,
    fallback_rate: float = 0.0,
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
                "prompt_version": "vmp_v52_evidence_set_v1",
                "candidate_count": 30,
                "min_observed_candidate_count": 30,
                "output_top_k": 5,
                "protected_top_n": 4,
                "parse_fallback_rate": fallback_rate,
            }
        )
    _write_json(path, payload)


def _write_json(path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
