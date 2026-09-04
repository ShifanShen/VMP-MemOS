"""Strict, label-free audit for official Mem0 experiment runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from pydantic import JsonValue

from vmp_memos.frameworks.official.mem0 import (
    MEM0_LLM_COMPATIBILITY_VERSION,
)


def audit_mem0_protocol_run(
    run_dir: str | Path,
    *,
    method: str = "mem0_official",
    max_unrecovered_failure_rate: float = 0.0,
    max_initial_invalid_rate: float = 0.02,
    require_bm25: bool = True,
    require_spacy: bool = True,
    expected_llm_max_tokens: int = 2048,
    expected_llm_retry_max_tokens: int = 4096,
    expected_llm_context_window: int = 32_768,
) -> dict[str, JsonValue]:
    """Audit one completed Mem0 run without reading answer labels.

    The initial-invalid rate is diagnostic: a successfully retried response is
    not semantic loss. Unrecovered JSON, request exceptions, truncated memory
    statistics, or a disabled native retrieval dependency make a run ineligible.
    """

    if not 0.0 <= max_unrecovered_failure_rate <= 1.0:
        raise ValueError("max_unrecovered_failure_rate must be in [0, 1]")
    if not 0.0 <= max_initial_invalid_rate <= 1.0:
        raise ValueError("max_initial_invalid_rate must be in [0, 1]")

    root = Path(run_dir)
    manifest = _read_json_object(root / "manifest.json")
    records = _read_jsonl(root / method / "retrieval.jsonl")
    runtime = _mapping(manifest.get("official_framework_runtime"))
    config = _mapping(manifest.get("config"))
    metadata = _mapping(config.get("metadata"))

    stats = [_mapping(record.get("adapter_stats")) for record in records]
    question_ids = [str(record.get("question_id") or "") for record in records]
    logical_calls = _sum_int(stats, "mem0_llm_logical_calls")
    requests = _sum_int(stats, "mem0_llm_requests")
    json_mode_calls = _sum_int(stats, "mem0_llm_json_mode_calls")
    initial_invalid = _sum_int(stats, "mem0_llm_initial_invalid_json")
    retry_attempts = _sum_int(stats, "mem0_llm_retry_attempts")
    retry_successes = _sum_int(stats, "mem0_llm_retry_successes")
    unrecovered = _sum_int(stats, "mem0_llm_unrecovered_invalid_json")
    request_exceptions = _sum_int(stats, "mem0_llm_request_exceptions")
    final_failures = unrecovered + request_exceptions
    initial_invalid_rate = initial_invalid / json_mode_calls if json_mode_calls else 0.0
    unrecovered_failure_rate = final_failures / logical_calls if logical_calls else 0.0

    expected_count = _as_int(manifest.get("sample_count"))
    versions = {
        str(item.get("mem0_llm_compatibility_version") or "")
        for item in stats
    }
    required_stat_fields = {
        "mem0_llm_compatibility_version",
        "mem0_llm_logical_calls",
        "mem0_llm_requests",
        "mem0_llm_json_mode_calls",
        "mem0_llm_initial_invalid_json",
        "mem0_llm_retry_attempts",
        "mem0_llm_retry_successes",
        "mem0_llm_unrecovered_invalid_json",
        "mem0_llm_request_exceptions",
        "mem0_memory_count_truncated",
        "mem0_bm25_enabled",
        "mem0_spacy_lemma_enabled",
    }
    incomplete_stats = sum(
        not required_stat_fields.issubset(item) for item in stats
    )
    memory_stats_truncated = sum(
        bool(item.get("mem0_memory_count_truncated")) for item in stats
    )
    bm25_values = [item.get("mem0_bm25_enabled") for item in stats]
    spacy_values = [item.get("mem0_spacy_lemma_enabled") for item in stats]

    checks: dict[str, bool] = {
        "run_completed": manifest.get("status") == "completed",
        "record_count": bool(records) and len(records) == expected_count,
        "unique_nonempty_question_ids": (
            len(question_ids) == len(set(question_ids)) and all(question_ids)
        ),
        "test_labels_not_used": not bool(
            metadata.get("test_labels_used_for_training", False)
        ),
        "protocol_version": versions == {MEM0_LLM_COMPATIBILITY_VERSION},
        "complete_protocol_stats": incomplete_stats == 0,
        "logical_calls_observed": logical_calls > 0,
        "json_mode_calls_observed": json_mode_calls > 0,
        "request_accounting": requests == logical_calls + retry_attempts,
        "initial_invalid_accounting": initial_invalid == retry_attempts,
        "retry_accounting": retry_successes + unrecovered == retry_attempts,
        "initial_invalid_rate": initial_invalid_rate <= max_initial_invalid_rate,
        "unrecovered_failure_rate": (
            unrecovered_failure_rate <= max_unrecovered_failure_rate
        ),
        "memory_stats_not_truncated": memory_stats_truncated == 0,
        "bm25_enabled": not require_bm25
        or (
            bool(bm25_values)
            and all(value is True for value in bm25_values)
        ),
        "spacy_lemma_enabled": not require_spacy
        or (
            bool(spacy_values)
            and all(value is True for value in spacy_values)
        ),
        "llm_max_tokens": (
            _as_int(runtime.get("official_llm_max_tokens"))
            == expected_llm_max_tokens
        ),
        "llm_retry_max_tokens": (
            _as_int(runtime.get("official_llm_retry_max_tokens"))
            == expected_llm_retry_max_tokens
        ),
        "llm_context_window": (
            _as_int(runtime.get("official_llm_context_window"))
            == expected_llm_context_window
        ),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "run": str(root),
        "method": method,
        "split": _mapping(manifest.get("split")).get("name"),
        "observed": cast(
            JsonValue,
            {
                "records": len(records),
                "logical_calls": logical_calls,
                "json_mode_calls": json_mode_calls,
                "physical_requests": requests,
                "initial_invalid_json": initial_invalid,
                "initial_invalid_rate": initial_invalid_rate,
                "retry_attempts": retry_attempts,
                "retry_successes": retry_successes,
                "unrecovered_invalid_json": unrecovered,
                "request_exceptions": request_exceptions,
                "unrecovered_failure_rate": unrecovered_failure_rate,
                "memory_stats_truncated_questions": memory_stats_truncated,
                "incomplete_protocol_stats_questions": incomplete_stats,
                "protocol_versions": sorted(versions),
                "bm25_enabled_questions": sum(value is True for value in bm25_values),
                "spacy_lemma_enabled_questions": sum(
                    value is True for value in spacy_values
                ),
            },
        ),
        "required": cast(
            JsonValue,
            {
                "max_unrecovered_failure_rate": max_unrecovered_failure_rate,
                "max_initial_invalid_rate": max_initial_invalid_rate,
                "require_bm25": require_bm25,
                "require_spacy": require_spacy,
                "llm_max_tokens": expected_llm_max_tokens,
                "llm_retry_max_tokens": expected_llm_retry_max_tokens,
                "llm_context_window": expected_llm_context_window,
                "protocol_version": MEM0_LLM_COMPATIBILITY_VERSION,
            },
        ),
        "checks": cast(JsonValue, checks),
        "test_labels_used": bool(
            metadata.get("test_labels_used_for_training", False)
        ),
    }


def _read_json_object(path: Path) -> dict[str, JsonValue]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Missing Mem0 protocol input: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return cast(dict[str, JsonValue], payload)


def _read_jsonl(path: Path) -> list[dict[str, JsonValue]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ValueError(f"Missing Mem0 protocol input: {path}") from exc
    records: list[dict[str, JsonValue]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload: Any = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a JSON object at {path}:{line_number}")
        records.append(cast(dict[str, JsonValue], payload))
    return records


def _mapping(value: object) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], value) if isinstance(value, dict) else {}


def _as_int(value: object) -> int:
    return int(value) if isinstance(value, int | float) else 0


def _sum_int(rows: list[dict[str, JsonValue]], key: str) -> int:
    return sum(_as_int(row.get(key)) for row in rows)
