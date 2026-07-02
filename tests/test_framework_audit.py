"""Tests for external framework controllability audit records."""

from __future__ import annotations

import json

import vmp_memos.frameworks.audit as audit_module
from vmp_memos.frameworks import FairnessLevel, audit_known_frameworks


def test_audit_known_frameworks_is_conservative() -> None:
    reports = audit_known_frameworks(
        ["mem0", "unknown_framework"],
        vllm_base_url="http://127.0.0.1:8000/v1",
        llm_model="Qwen/Qwen2.5-7B-Instruct",
        embedding_model="BAAI/bge-m3",
        embedding_dimension=1024,
    )
    assert [report.framework_name for report in reports] == [
        "mem0",
        "unknown_framework",
    ]
    assert not any(report.main_table_eligible for report in reports)
    assert reports[0].adapter_implemented is True
    assert reports[1].adapter_implemented is False
    assert reports[1].fairness_level == FairnessLevel.UNAVAILABLE
    assert reports[0].official_repo


def test_mem0_becomes_eligible_only_with_matching_smoke(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(audit_module, "find_spec", lambda package: object())
    monkeypatch.setattr(
        audit_module,
        "_installed_version",
        lambda distribution: "2.0.10",
    )
    (tmp_path / "mem0_smoke.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "framework_version": "2.0.10",
                "vllm_base_url": "http://127.0.0.1:8000/v1",
                "llm_model": "Qwen/Qwen2.5-7B-Instruct",
                "embedding_model": "BAAI/bge-m3",
                "embedding_dimension": 1024,
                "official_llm_max_tokens": 512,
                "official_llm_temperature": 0.0,
            }
        ),
        encoding="utf-8",
    )

    report = audit_known_frameworks(
        ["mem0"],
        vllm_base_url="http://127.0.0.1:8000/v1",
        llm_model="Qwen/Qwen2.5-7B-Instruct",
        embedding_model="BAAI/bge-m3",
        embedding_dimension=1024,
        verification_dir=tmp_path,
    )[0]

    assert report.adapter_smoke_verified is True
    assert report.main_table_eligible is True
    assert report.fairness_level == FairnessLevel.FULLY_CONTROLLED


def test_graphiti_is_an_implemented_pinned_official_adapter(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(audit_module, "find_spec", lambda package: object())
    monkeypatch.setattr(
        audit_module,
        "_installed_version",
        lambda distribution: "0.29.2",
    )
    (tmp_path / "graphiti_smoke.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "framework_version": "0.29.2",
                "vllm_base_url": "http://127.0.0.1:8000/v1",
                "llm_model": "Qwen/Qwen2.5-7B-Instruct",
                "embedding_model": "BAAI/bge-m3",
                "embedding_dimension": 1024,
                "official_llm_max_tokens": 512,
                "official_llm_temperature": 0.0,
            }
        ),
        encoding="utf-8",
    )

    report = audit_known_frameworks(
        ["graphiti"],
        vllm_base_url="http://127.0.0.1:8000/v1",
        llm_model="Qwen/Qwen2.5-7B-Instruct",
        embedding_model="BAAI/bge-m3",
        embedding_dimension=1024,
        verification_dir=tmp_path,
    )[0]

    assert report.adapter_implemented is True
    assert report.version_or_commit == "0.29.2"
    assert report.main_table_eligible is True
    assert report.license == "Apache-2.0"


def test_letta_requires_matching_client_and_server_versions(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(audit_module, "find_spec", lambda package: object())
    monkeypatch.setattr(
        audit_module,
        "_installed_version",
        lambda distribution: "1.12.1",
    )
    payload = {
        "status": "passed",
        "framework_version": "1.12.1",
        "server_version": "0.16.8",
        "vllm_base_url": "http://127.0.0.1:8000/v1",
        "llm_model": "Qwen/Qwen2.5-7B-Instruct",
        "embedding_model": "BAAI/bge-m3",
        "embedding_dimension": 1024,
        "official_llm_max_tokens": 512,
        "official_llm_temperature": 0.0,
    }
    path = tmp_path / "letta_smoke.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = audit_known_frameworks(
        ["letta"],
        vllm_base_url="http://127.0.0.1:8000/v1",
        llm_model="Qwen/Qwen2.5-7B-Instruct",
        embedding_model="BAAI/bge-m3",
        embedding_dimension=1024,
        verification_dir=tmp_path,
    )[0]
    assert report.main_table_eligible is True
    assert report.notes["supported_server_version"] == "0.16.8"
    assert report.token_latency_stats_supported is False

    payload["server_version"] = "0.16.7"
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = audit_known_frameworks(
        ["letta"],
        vllm_base_url="http://127.0.0.1:8000/v1",
        llm_model="Qwen/Qwen2.5-7B-Instruct",
        embedding_model="BAAI/bge-m3",
        embedding_dimension=1024,
        verification_dir=tmp_path,
    )[0]
    assert report.main_table_eligible is False
