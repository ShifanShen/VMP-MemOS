"""Tests for frozen-model LongMemEval component ablations."""

from __future__ import annotations

import json
from pathlib import Path

from vmp_memos.frameworks import RetrievedMemory, adapter_for_name
from vmp_memos.frameworks.vmp_tuned import (
    BASELINE_VMP_WEIGHTS,
    VMP_TUNED_ABLATIONS,
    VMPTunedModel,
    ablation_for_method,
    vmp_tuned_feature_values,
)
from vmp_memos.longmemeval import (
    LongMemEvalRunConfig,
    LongMemEvalSample,
    sample_to_session_events,
)
from vmp_memos.longmemeval.ablation import export_longmemeval_ablation_table
from vmp_memos.longmemeval.retrieval_runner import run_longmemeval_retrieval
from vmp_memos.longmemeval.splits import create_longmemeval_split
from vmp_memos.longmemeval.tuning import train_vmp_tuned
from vmp_memos.schemas import PolicyFeatures


def test_feature_and_operation_ablation_targets_are_independent() -> None:
    features = PolicyFeatures(
        recency=0.8,
        contradiction=0.5,
        actionability=0.5,
    )
    no_recency = ablation_for_method("vmp_tuned__no_recency")
    no_update = ablation_for_method("vmp_tuned__no_update_operation")

    recency_values = vmp_tuned_feature_values(
        features,
        disabled_features=no_recency.disabled_features,
    )
    update_values = vmp_tuned_feature_values(
        features,
        disabled_operations=no_update.disabled_operations,
    )
    non_temporal_values = vmp_tuned_feature_values(
        features,
        temporal_intent=False,
    )

    assert recency_values["recency"] == 0.0
    assert recency_values["update_signal"] == 0.0
    assert recency_values["action_signal"] == 0.0
    assert update_values["recency"] == 0.8
    assert update_values["contradiction"] == 0.5
    assert update_values["update_signal"] == 0.0
    assert non_temporal_values["recency"] == 0.0
    assert non_temporal_values["contradiction"] == 0.0
    assert non_temporal_values["update_signal"] == 0.0
    assert non_temporal_values["action_signal"] == 0.0


def test_lifecycle_operations_are_non_destructive_during_retrieval(tmp_path) -> None:
    model_path = _model().save(tmp_path / "model.json")
    update_sample = LongMemEvalSample.model_validate(_record(0))

    full, full_stats = _retrieve(
        "vmp_tuned",
        update_sample,
        model_path=model_path,
        workspace=tmp_path / "full",
    )
    no_archive, no_archive_stats = _retrieve(
        "vmp_tuned__no_archive_operation",
        update_sample,
        model_path=model_path,
        workspace=tmp_path / "no_archive",
    )
    assert {memory.source_session_id for memory in full} == {
        "q0_old",
        "q0_new",
    }
    assert {memory.source_session_id for memory in no_archive} == {
        "q0_old",
        "q0_new",
    }
    assert full_stats["memory_count"] == 2
    assert no_archive_stats["memory_count"] == 2
    assert full_stats["lifecycle_status_counts"]["superseded"] == 1
    assert no_archive_stats["lifecycle_status_counts"]["superseded"] == 0

    duplicate_sample = LongMemEvalSample.model_validate(_duplicate_record())
    merged, merged_stats = _retrieve(
        "vmp_tuned",
        duplicate_sample,
        model_path=model_path,
        workspace=tmp_path / "merged",
    )
    no_merge, no_merge_stats = _retrieve(
        "vmp_tuned__no_merge_operation",
        duplicate_sample,
        model_path=model_path,
        workspace=tmp_path / "no_merge",
    )
    assert len(merged) == 2
    assert len(no_merge) == 2
    assert merged_stats["memory_count"] == 2
    assert no_merge_stats["memory_count"] == 2
    assert merged_stats["lifecycle_status_counts"]["duplicate"] == 1
    assert no_merge_stats["lifecycle_status_counts"]["duplicate"] == 0


def test_ablation_run_exports_delta_table_from_test_split(tmp_path) -> None:
    data_path = tmp_path / "longmemeval.json"
    data_path.write_text(
        json.dumps([_record(index) for index in range(4)]),
        encoding="utf-8",
    )
    split = create_longmemeval_split(data_path, dev_size=2, test_size=2, seed=42)
    split_path = split.save(tmp_path / "split.json")
    tuning = train_vmp_tuned(
        data_path,
        split_path,
        embedder=None,
        trials=2,
        tuning_seed=7,
    )
    model_path = tuning.model.save(tmp_path / "vmp_tuned.json")
    methods = ["vmp_tuned", *VMP_TUNED_ABLATIONS]
    config = LongMemEvalRunConfig(
        data_path=data_path,
        methods=methods,
        top_k=5,
        retrieval_depth=10,
        output_dir=tmp_path / "outputs",
        split_manifest_path=split_path,
        split_name="test",
        vmp_tuned_model_path=model_path,
    )

    result = run_longmemeval_retrieval(config, run_id="ablation")
    outputs = export_longmemeval_ablation_table(result.run_dir)

    csv_text = outputs["ablation_csv"].read_text(encoding="utf-8")
    assert "VMP-full" in csv_text
    assert "vmp_tuned__no_update_operation" in csv_text
    assert "delta_recall_all@5" in csv_text
    assert outputs["ablation_markdown"].exists()
    assert outputs["ablation_latex"].exists()


def _retrieve(
    method: str,
    sample: LongMemEvalSample,
    *,
    model_path: Path,
    workspace: Path,
) -> tuple[list[RetrievedMemory], dict[str, object]]:
    adapter = adapter_for_name(
        method,
        vmp_tuned_model_path=str(model_path),
    )
    adapter.reset(workspace)
    for events in sample_to_session_events(sample):
        adapter.ingest_session(events)
    results = adapter.retrieve(
        sample.question,
        top_k=10,
        question_date=sample.question_date,
        metadata={
            "question_id": sample.question_id,
            "question_type": "single_session_user",
        },
    )
    stats = adapter.stats()
    adapter.close()
    return results, stats


def _model() -> VMPTunedModel:
    return VMPTunedModel(
        weights=BASELINE_VMP_WEIGHTS,
        retrieve_threshold=0.0,
        archive_similarity_threshold=0.45,
        split_id="test_split",
        split_manifest_sha256="test_manifest",
        dataset_sha256="test_dataset",
        best_objective=0.0,
    )


def _record(index: int) -> dict[str, object]:
    return {
        "question_id": f"q{index}",
        "question_type": "knowledge_update",
        "question": f"What activity does person {index} now prefer?",
        "answer": "swimming",
        "question_date": "2024-02-01",
        "haystack_session_ids": [f"q{index}_old", f"q{index}_new"],
        "haystack_dates": ["2024-01-01", "2024-01-20"],
        "haystack_sessions": [
            [{"role": "user", "content": f"Person {index} prefers hiking."}],
            [
                {
                    "role": "user",
                    "content": (
                        f"Person {index} now prefers swimming instead of hiking."
                    ),
                }
            ],
        ],
        "answer_session_ids": [f"q{index}_new"],
        "has_answer": True,
    }


def _duplicate_record() -> dict[str, object]:
    return {
        "question_id": "duplicate",
        "question_type": "single_session_user",
        "question": "What drink does Alex prefer?",
        "answer": "tea",
        "question_date": "2024-02-01",
        "haystack_session_ids": ["duplicate_1", "duplicate_2"],
        "haystack_dates": ["2024-01-01", "2024-01-02"],
        "haystack_sessions": [
            [{"role": "user", "content": "Alex prefers tea."}],
            [{"role": "user", "content": "Alex prefers tea."}],
        ],
        "answer_session_ids": ["duplicate_2"],
        "has_answer": True,
    }
