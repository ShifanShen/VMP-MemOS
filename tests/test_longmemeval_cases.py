"""Tests for deterministic qualitative paper-case export."""

from __future__ import annotations

import json
from pathlib import Path

from vmp_memos.frameworks import RetrievedMemory
from vmp_memos.longmemeval.cases import export_longmemeval_cases
from vmp_memos.longmemeval.qa_runner import QASampleRecord
from vmp_memos.longmemeval.retrieval_runner import RetrievalSampleRecord


def test_case_export_selects_four_distinct_auditable_cases(tmp_path) -> None:
    main_dir = tmp_path / "main"
    ablation_dir = tmp_path / "ablation"
    main_methods = ["naive_vector", "vmp_tuned"]
    ablation_methods = [
        "vmp_tuned",
        "vmp_tuned__no_archive_operation",
    ]
    _write_manifest(main_dir, main_methods)
    _write_manifest(ablation_dir, ablation_methods)

    vector_records = [
        _retrieval("q1", "naive_vector", [_memory("q1_old", "2024-01-01")], recall=0),
        _retrieval(
            "q2",
            "naive_vector",
            [
                _memory("q2_old", "2024-01-01"),
                _memory("q2_new", "2024-01-20"),
            ],
            recall=1,
        ),
        _retrieval("q3", "naive_vector", [_memory("q3_new", "2024-01-20")], recall=1),
        _retrieval("q4", "naive_vector", [_memory("q4_new", "2024-01-20")], recall=1),
    ]
    vmp_records = [
        _retrieval("q1", "vmp_tuned", [_memory("q1_new", "2024-01-20")], recall=1),
        _retrieval("q2", "vmp_tuned", [_memory("q2_new", "2024-01-20")], recall=1),
        _retrieval("q3", "vmp_tuned", [_memory("q3_new", "2024-01-20")], recall=1),
        _retrieval("q4", "vmp_tuned", [_memory("q4_old", "2024-01-01")], recall=0),
    ]
    _write_jsonl(main_dir / "naive_vector" / "retrieval.jsonl", vector_records)
    _write_jsonl(main_dir / "vmp_tuned" / "retrieval.jsonl", vmp_records)
    _write_qa_manifest(main_dir)
    _write_jsonl(
        main_dir / "qa" / "naive_vector.jsonl",
        [
            _qa("q1", "naive_vector", correct=True),
            _qa("q2", "naive_vector", correct=False),
            _qa("q3", "naive_vector", correct=True),
            _qa("q4", "naive_vector", correct=True),
        ],
    )
    _write_jsonl(
        main_dir / "qa" / "vmp_tuned.jsonl",
        [
            _qa("q1", "vmp_tuned", correct=True),
            _qa("q2", "vmp_tuned", correct=True),
            _qa("q3", "vmp_tuned", correct=True),
            _qa("q4", "vmp_tuned", correct=False),
        ],
    )

    full_ablation = [
        _retrieval(f"q{index}", "vmp_tuned", [_memory(f"q{index}_new", "2024-01-20")], recall=1)
        for index in range(1, 5)
    ]
    full_ablation[2] = _retrieval(
        "q3",
        "vmp_tuned",
        [_memory("q3_new", "2024-01-20")],
        recall=1,
        memory_count=1,
        archive_count=1,
    )
    no_archive = [
        _retrieval(
            f"q{index}",
            "vmp_tuned__no_archive_operation",
            [_memory(f"q{index}_new", "2024-01-20")],
            recall=1,
        )
        for index in range(1, 5)
    ]
    no_archive[2] = _retrieval(
        "q3",
        "vmp_tuned__no_archive_operation",
        [
            _memory("q3_new", "2024-01-20"),
            _memory("q3_old", "2024-01-01"),
        ],
        recall=1,
        memory_count=2,
    )
    _write_jsonl(ablation_dir / "vmp_tuned" / "retrieval.jsonl", full_ablation)
    _write_jsonl(
        ablation_dir / "vmp_tuned__no_archive_operation" / "retrieval.jsonl",
        no_archive,
    )

    outputs = export_longmemeval_cases(
        main_dir,
        ablation_run=ablation_dir,
        output_dir=tmp_path / "cases",
    )

    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["selected_question_ids"] == {
        "case1_knowledge_update": "q1",
        "case2_stale_vector_retrieval": "q2",
        "case3_archive_suppression": "q3",
        "case4_vmp_error": "q4",
    }
    markdown = outputs["paper_markdown"].read_text(encoding="utf-8")
    assert "VMP correctly handles a knowledge update" in markdown
    assert "VMP failure case" in markdown
    assert len(outputs) == 7


def _write_manifest(run_dir: Path, methods: list[str]) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "data_sha256": "dataset-sha",
                "split": {"name": "test", "split_id": "split-42"},
                "vmp_tuned_model": {"sha256": "model-sha"},
                "config": {"methods": methods},
            }
        ),
        encoding="utf-8",
    )


def _write_qa_manifest(run_dir: Path) -> None:
    qa_dir = run_dir / "qa"
    qa_dir.mkdir(parents=True)
    (qa_dir / "manifest.json").write_text(
        json.dumps({"status": "completed"}),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[SchemaRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
    )


def _retrieval(
    question_id: str,
    method: str,
    memories: list[RetrievedMemory],
    *,
    recall: float,
    memory_count: int = 1,
    archive_count: int = 0,
) -> RetrievalSampleRecord:
    return RetrievalSampleRecord(
        question_id=question_id,
        question_type="knowledge_update",
        question=f"What is the current preference for {question_id}?",
        answer="new answer",
        question_date="2024-02-01",
        method=method,
        is_abstention=False,
        gold_session_ids=[f"{question_id}_new"],
        retrieved_session_ids=[
            memory.source_session_id
            for memory in memories
            if memory.source_session_id is not None
        ],
        retrieved_memories=memories,
        metrics={"recall_all@5": recall, "mrr": recall},
        retrieved_tokens=sum(memory.token_count for memory in memories),
        adapter_stats={
            "memory_count": memory_count,
            "total_tokens": 20 * memory_count,
            "storage_size_bytes": 100 * memory_count,
            "policy_operation_counts": {
                "update": archive_count,
                "merge": 0,
                "archive": archive_count,
            },
        },
    )


def _memory(session_id: str, source_date: str) -> RetrievedMemory:
    return RetrievedMemory(
        memory_id=session_id,
        content=f"Evidence from {session_id}",
        score=0.9,
        source_session_id=session_id,
        source_date=source_date,
        token_count=5,
        metadata={
            "policy_features": {"recency": 0.9},
            "policy_contributions": {"recency": 0.09},
        },
    )


def _qa(question_id: str, method: str, *, correct: bool) -> QASampleRecord:
    return QASampleRecord(
        question_id=question_id,
        question_type="knowledge_update",
        method=method,
        question=f"What is the current preference for {question_id}?",
        gold_answer="new answer",
        prediction="new answer" if correct else "old answer",
        is_abstention=False,
        metrics={
            "normalized_exact_match": float(correct),
            "token_f1": float(correct),
            "contains_answer": float(correct),
        },
        reader_provider="fake-vllm",
        reader_model="fake-reader",
        prompt_sha256="a" * 64,
    )


SchemaRecord = RetrievalSampleRecord | QASampleRecord
