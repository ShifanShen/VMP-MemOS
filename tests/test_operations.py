"""Tests for operation validation and shared JSONL support."""

import json

import pytest
from pydantic import ValidationError

from vmp_memos.schemas import MemoryOperation, OperationType


def test_memory_operation_serializes_with_enum_values() -> None:
    operation = MemoryOperation(
        op=OperationType.UPDATE,
        target_memory_id="mem_001",
        source_event_id="evt_002",
        reason="A newer preference supersedes the old value.",
        policy_score=0.82,
        confidence=0.91,
        scope="career/agent-dev",
        backend="file",
    )

    payload = json.loads(operation.to_json_line())

    assert payload["op"] == "UPDATE"
    assert payload["target_memory_id"] == "mem_001"
    assert payload["op_id"].startswith("op_")
    assert payload["timestamp"].endswith("Z")


def test_schema_appends_and_reads_jsonl(tmp_path) -> None:
    log_path = tmp_path / "nested" / "operations.jsonl"
    first = MemoryOperation(
        op="ADD",
        reason="High-value new information.",
        policy_score=0.8,
        confidence=0.9,
    )
    second = MemoryOperation(
        op="IGNORE",
        reason="Low-value duplicate.",
        policy_score=0.2,
        confidence=0.7,
    )

    first.append_jsonl(log_path)
    second.append_jsonl(log_path)
    records = log_path.read_text(encoding="utf-8").splitlines()

    assert len(records) == 2
    assert MemoryOperation.from_json_line(records[0]) == first
    assert MemoryOperation.from_json_line(records[1]) == second


def test_operation_id_is_immutable() -> None:
    operation = MemoryOperation(
        op="ADD",
        reason="Useful information.",
        policy_score=0.8,
        confidence=0.9,
    )

    with pytest.raises(ValidationError):
        operation.op_id = "op_replaced"


def test_operation_rejects_invalid_scores() -> None:
    with pytest.raises(ValidationError):
        MemoryOperation(
            op="ADD",
            reason="Invalid score.",
            policy_score=1.2,
            confidence=0.9,
        )

