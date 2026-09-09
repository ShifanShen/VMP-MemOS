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
    max_partial_recovery_rate: float = 0.01,
    require_bm25: bool = True,
    require_spacy: bool = True,
    expected_sample_count: int = 20,
    expected_llm_max_tokens: int = 2048,
    expected_llm_retry_max_tokens: int = 16384,
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
    if not 0.0 <= max_partial_recovery_rate <= 1.0:
        raise ValueError("max_partial_recovery_rate must be in [0, 1]")
    if expected_sample_count <= 0:
        raise ValueError("expected_sample_count must be positive")

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
    initial_invalid_schema = _sum_int(
        stats,
        "mem0_llm_initial_invalid_schema",
    )
    unrecovered_invalid_schema = _sum_int(
        stats,
        "mem0_llm_unrecovered_invalid_schema",
    )
    request_exceptions = _sum_int(stats, "mem0_llm_request_exceptions")
    normalized_responses = _sum_int(stats, "mem0_llm_normalized_responses")
    normalized_items = _sum_int(stats, "mem0_llm_normalized_items")
    ignored_null_items = _sum_int(stats, "mem0_llm_ignored_null_items")
    partial_recoveries = _sum_int(stats, "mem0_llm_partial_recoveries")
    partial_recovered_items = _sum_int(
        stats,
        "mem0_llm_partial_recovered_items",
    )
    partial_discarded_characters = _sum_int(
        stats,
        "mem0_llm_partial_discarded_characters",
    )
    max_partial_discarded_characters = _max_int(
        stats,
        "mem0_llm_max_partial_discarded_characters",
    )
    partial_recovery_reasons = _sum_counter(
        stats,
        "mem0_llm_partial_recovery_reason_counts",
    )
    initial_invalid_reasons = _sum_counter(
        stats,
        "mem0_llm_initial_invalid_reason_counts",
    )
    unrecovered_invalid_reasons = _sum_counter(
        stats,
        "mem0_llm_unrecovered_invalid_reason_counts",
    )
    initial_invalid_max_characters = _max_int(
        stats,
        "mem0_llm_initial_invalid_max_response_characters",
    )
    unrecovered_invalid_max_characters = _max_int(
        stats,
        "mem0_llm_unrecovered_invalid_max_response_characters",
    )
    final_failures = unrecovered + request_exceptions
    initial_invalid_rate = initial_invalid / json_mode_calls if json_mode_calls else 0.0
    unrecovered_failure_rate = final_failures / logical_calls if logical_calls else 0.0
    partial_recovery_rate = partial_recoveries / logical_calls if logical_calls else 0.0

    manifest_sample_count = _as_int(manifest.get("sample_count"))
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
        "mem0_llm_initial_invalid_schema",
        "mem0_llm_unrecovered_invalid_schema",
        "mem0_llm_request_exceptions",
        "mem0_llm_normalized_responses",
        "mem0_llm_normalized_items",
        "mem0_llm_ignored_null_items",
        "mem0_llm_partial_recoveries",
        "mem0_llm_partial_recovered_items",
        "mem0_llm_partial_discarded_characters",
        "mem0_llm_max_partial_discarded_characters",
        "mem0_llm_partial_recovery_reason_counts",
        "mem0_llm_initial_invalid_reason_counts",
        "mem0_llm_unrecovered_invalid_reason_counts",
        "mem0_llm_initial_invalid_max_response_characters",
        "mem0_llm_unrecovered_invalid_max_response_characters",
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
        "manifest_sample_count": manifest_sample_count == expected_sample_count,
        "record_count": len(records) == expected_sample_count,
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
        "partial_recovery_accounting": 0 <= partial_recoveries <= retry_successes,
        "partial_recovery_reason_accounting": (
            sum(partial_recovery_reasons.values()) == partial_recoveries
            and all(count >= 0 for count in partial_recovery_reasons.values())
        ),
        "partial_recovery_character_accounting": (
            partial_recovered_items >= 0
            and partial_discarded_characters >= 0
            and 0 <= max_partial_discarded_characters <= partial_discarded_characters
        ),
        "schema_failure_accounting": (
            initial_invalid_schema <= initial_invalid
            and unrecovered_invalid_schema <= unrecovered
        ),
        "initial_invalid_reason_accounting": (
            sum(initial_invalid_reasons.values()) == initial_invalid
        ),
        "unrecovered_invalid_reason_accounting": (
            sum(unrecovered_invalid_reasons.values()) == unrecovered
        ),
        "initial_invalid_rate": initial_invalid_rate <= max_initial_invalid_rate,
        "partial_recovery_rate": (
            partial_recovery_rate <= max_partial_recovery_rate
        ),
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
                "manifest_sample_count": manifest_sample_count,
                "logical_calls": logical_calls,
                "json_mode_calls": json_mode_calls,
                "physical_requests": requests,
                "initial_invalid_json": initial_invalid,
                "initial_invalid_rate": initial_invalid_rate,
                "retry_attempts": retry_attempts,
                "retry_successes": retry_successes,
                "complete_retry_successes": retry_successes - partial_recoveries,
                "unrecovered_invalid_json": unrecovered,
                "initial_invalid_schema": initial_invalid_schema,
                "unrecovered_invalid_schema": unrecovered_invalid_schema,
                "request_exceptions": request_exceptions,
                "normalized_responses": normalized_responses,
                "normalized_items": normalized_items,
                "ignored_null_items": ignored_null_items,
                "partial_recoveries": partial_recoveries,
                "partial_recovery_rate": partial_recovery_rate,
                "partial_recovered_items": partial_recovered_items,
                "partial_discarded_characters": partial_discarded_characters,
                "max_partial_discarded_characters": (
                    max_partial_discarded_characters
                ),
                "partial_recovery_reason_counts": partial_recovery_reasons,
                "initial_invalid_reason_counts": initial_invalid_reasons,
                "unrecovered_invalid_reason_counts": unrecovered_invalid_reasons,
                "initial_invalid_max_response_characters": (
                    initial_invalid_max_characters
                ),
                "unrecovered_invalid_max_response_characters": (
                    unrecovered_invalid_max_characters
                ),
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
                "max_partial_recovery_rate": max_partial_recovery_rate,
                "require_bm25": require_bm25,
                "require_spacy": require_spacy,
                "sample_count": expected_sample_count,
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


def _max_int(rows: list[dict[str, JsonValue]], key: str) -> int:
    return max((_as_int(row.get(key)) for row in rows), default=0)


def _sum_counter(
    rows: list[dict[str, JsonValue]],
    key: str,
) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, dict):
            continue
        for name, count in value.items():
            if isinstance(name, str):
                result[name] = result.get(name, 0) + _as_int(count)
    return result
