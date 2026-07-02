"""Tests for deterministic, leakage-safe LongMemEval splits."""

from __future__ import annotations

import json

import pytest

from vmp_memos.longmemeval.splits import (
    LongMemEvalSplitManifest,
    create_longmemeval_split,
    load_split_samples,
)


def test_split_is_exact_deterministic_and_disjoint(tmp_path) -> None:
    data_path = tmp_path / "longmemeval.json"
    data_path.write_text(
        json.dumps([_record(index) for index in range(8)]),
        encoding="utf-8",
    )

    first = create_longmemeval_split(data_path, dev_size=2, test_size=6, seed=42)
    second = create_longmemeval_split(data_path, dev_size=2, test_size=6, seed=42)

    assert first.splits == second.splits
    assert len(first.splits["dev"]) == 2
    assert len(first.splits["test"]) == 6
    assert set(first.splits["dev"]).isdisjoint(first.splits["test"])
    assert first.metadata["assignment_uses_labels"] is False

    manifest_path = first.save(tmp_path / "split.json")
    loaded = LongMemEvalSplitManifest.load(manifest_path)
    samples, checked = load_split_samples(data_path, manifest_path, "test")
    assert checked.split_id == loaded.split_id
    assert [sample.question_id for sample in samples] == loaded.splits["test"]


def test_split_rejects_modified_dataset_bytes(tmp_path) -> None:
    data_path = tmp_path / "longmemeval.json"
    data_path.write_text(json.dumps([_record(1), _record(2)]), encoding="utf-8")
    manifest = create_longmemeval_split(data_path, dev_size=1, test_size=1)
    manifest_path = manifest.save(tmp_path / "split.json")
    data_path.write_text(
        json.dumps([_record(1), _record(2)], indent=2),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SHA-256"):
        load_split_samples(data_path, manifest_path, "dev")


def _record(index: int) -> dict:
    return {
        "question_id": f"q{index}",
        "question_type": "knowledge_update" if index % 2 else "single_session_user",
        "question": f"What does user {index} prefer?",
        "answer": f"answer {index}",
        "question_date": "2024-02-01",
        "haystack_session_ids": [f"s{index}"],
        "haystack_dates": ["2024-01-20"],
        "haystack_sessions": [
            [{"role": "user", "content": f"User {index} prefers answer {index}."}]
        ],
        "answer_session_ids": [f"s{index}"],
        "has_answer": True,
    }
