"""Tests for the label-free Mem0 transport and runtime quality gate."""

from __future__ import annotations

import json

from vmp_memos.longmemeval.mem0_protocol import audit_mem0_protocol_run


def test_mem0_protocol_audit_accepts_recovered_json_and_complete_runtime(tmp_path) -> None:
    run = _write_run(tmp_path, final_failures=0, bm25=True, spacy=True)

    report = audit_mem0_protocol_run(run, max_initial_invalid_rate=0.20)

    assert report["status"] == "passed"
    assert report["checks"]["unrecovered_failure_rate"] is True
    assert report["observed"]["logical_calls"] == 6
    assert report["observed"]["retry_successes"] == 1
    assert report["observed"]["initial_invalid_reason_counts"] == {
        "unterminated_json_object": 1,
    }
    assert report["test_labels_used"] is False


def test_mem0_protocol_audit_rejects_silent_extraction_loss(tmp_path) -> None:
    run = _write_run(tmp_path, final_failures=1, bm25=True, spacy=True)

    report = audit_mem0_protocol_run(run, max_initial_invalid_rate=0.20)

    assert report["status"] == "failed"
    assert report["checks"]["unrecovered_failure_rate"] is False


def test_mem0_protocol_audit_rejects_disabled_native_hybrid_dependencies(tmp_path) -> None:
    run = _write_run(tmp_path, final_failures=0, bm25=False, spacy=False)

    report = audit_mem0_protocol_run(run, max_initial_invalid_rate=0.20)

    assert report["status"] == "failed"
    assert report["checks"]["bm25_enabled"] is False
    assert report["checks"]["spacy_lemma_enabled"] is False


def _write_run(tmp_path, *, final_failures: int, bm25: bool, spacy: bool):
    run = tmp_path / "run"
    method = run / "mem0_official"
    method.mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "sample_count": 2,
                "split": {"name": "dev"},
                "config": {
                    "metadata": {"test_labels_used_for_training": False},
                },
                "official_framework_runtime": {
                    "official_llm_max_tokens": 2048,
                    "official_llm_retry_max_tokens": 8192,
                    "official_llm_context_window": 32768,
                },
            }
        ),
        encoding="utf-8",
    )
    rows = []
    for index in range(2):
        rows.append(
            {
                "question_id": f"q{index}",
                "adapter_stats": {
                    "mem0_llm_compatibility_version": "mem0_v2010_json_transport_v4",
                    "mem0_llm_logical_calls": 3,
                    "mem0_llm_json_mode_calls": 3,
                    "mem0_llm_requests": 4 if index == 0 else 3,
                    "mem0_llm_initial_invalid_json": 1 if index == 0 else 0,
                    "mem0_llm_retry_attempts": 1 if index == 0 else 0,
                    "mem0_llm_retry_successes": 1 if index == 0 else 0,
                    "mem0_llm_unrecovered_invalid_json": (
                        final_failures if index == 1 else 0
                    ),
                    "mem0_llm_initial_invalid_reason_counts": (
                        {"unterminated_json_object": 1} if index == 0 else {}
                    ),
                    "mem0_llm_unrecovered_invalid_reason_counts": (
                        {"unterminated_json_object": final_failures}
                        if index == 1
                        else {}
                    ),
                    "mem0_llm_initial_invalid_max_response_characters": (
                        8192 if index == 0 else 0
                    ),
                    "mem0_llm_unrecovered_invalid_max_response_characters": (
                        16384 if index == 1 and final_failures else 0
                    ),
                    "mem0_llm_request_exceptions": 0,
                    "mem0_memory_count_truncated": False,
                    "mem0_bm25_enabled": bm25,
                    "mem0_spacy_lemma_enabled": spacy,
                },
            }
        )
    (method / "retrieval.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return run
