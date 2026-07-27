"""Tests for VMP-v5 hierarchical session/turn retrieval."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import timedelta

import pytest

from vmp_memos.embeddings import BaseEmbedder
from vmp_memos.frameworks import (
    VMPHierarchicalAdapter,
    VMPHierarchicalModel,
    VMPTunedAdapter,
    VMPTunedModel,
    adapter_for_name,
)
from vmp_memos.frameworks.vmp_hierarchical import (
    hierarchical_fusion_score,
    pool_top_scores,
)
from vmp_memos.frameworks.vmp_tuned import VMP_TUNED_FEATURES
from vmp_memos.longmemeval import LongMemEvalRunConfig, LongMemEvalSample
from vmp_memos.longmemeval.converter import sample_to_session_events
from vmp_memos.longmemeval.hierarchical_tuning import (
    hierarchical_parameter_grid,
    train_vmp_hierarchical,
)
from vmp_memos.longmemeval.retrieval_runner import run_longmemeval_retrieval
from vmp_memos.longmemeval.splits import (
    create_longmemeval_split,
    sha256_file,
)


class HierarchyTestEmbedder(BaseEmbedder):
    """Make the relevant long session weak but its relevant turn exact."""

    @property
    def identifier(self) -> str:
        return "hierarchy-test-embedder"

    @property
    def dimension(self) -> int:
        return 2

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in self.validate_texts(texts):
            lowered = text.casefold()
            if "which hidden fact" in lowered or ("needle fact" in lowered and "\n" not in text):
                vectors.append([1.0, 0.0])
            elif "needle fact" in lowered:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.6, 0.8])
        return vectors


def test_hierarchical_turn_signal_recovers_a_buried_session(tmp_path) -> None:
    sample = LongMemEvalSample.model_validate(_hierarchical_sample())
    embedder = HierarchyTestEmbedder()
    base_model = _base_model(embedder.identifier)
    hierarchical_model = VMPHierarchicalModel(
        base_model=base_model,
        session_semantic_weight=0.2,
        turn_semantic_weight=0.8,
        turn_lexical_weight=0.0,
        turn_pooling_top_n=1,
        split_id=base_model.split_id,
        split_manifest_sha256=base_model.split_manifest_sha256,
        split_assignment_sha256="assignment",
        dataset_sha256=base_model.dataset_sha256,
        embedding_identifier=base_model.embedding_identifier,
        best_objective=1.0,
        metadata={"test_labels_used": False},
    )

    dense = VMPTunedAdapter(model=base_model, embedder=embedder)
    hierarchical = VMPHierarchicalAdapter(
        model=hierarchical_model,
        embedder=embedder,
    )
    for adapter in (dense, hierarchical):
        adapter.reset(tmp_path / adapter.name)
        for events in sample_to_session_events(sample):
            adapter.ingest_session(events)
        adapter.finalize_ingestion()

    dense_result = dense.retrieve(sample.question, top_k=2)
    hierarchical_result = hierarchical.retrieve(sample.question, top_k=2)

    assert dense_result[0].source_session_id == "distractor"
    assert hierarchical_result[0].source_session_id == "relevant"
    assert (
        hierarchical_result[0].metadata["hierarchical_fused_score"]
        > hierarchical_result[1].metadata["hierarchical_fused_score"]
    )
    stats = hierarchical.stats()
    assert stats["session_memory_count"] == 2
    assert stats["turn_memory_count"] == 4
    assert stats["memory_count"] == 6


def test_hierarchical_model_and_registry_require_safe_provenance(tmp_path) -> None:
    base_model = _base_model("hierarchy-test-embedder")
    model = VMPHierarchicalModel(
        base_model=base_model,
        session_semantic_weight=0.5,
        turn_semantic_weight=0.5,
        turn_lexical_weight=0.0,
        split_id=base_model.split_id,
        split_manifest_sha256=base_model.split_manifest_sha256,
        split_assignment_sha256="assignment",
        dataset_sha256=base_model.dataset_sha256,
        embedding_identifier=base_model.embedding_identifier,
        best_objective=1.0,
        metadata={"test_labels_used": False},
    )
    path = model.save(tmp_path / "vmp_v5.json")

    adapter = adapter_for_name(
        "vmp-v5",
        embedder=HierarchyTestEmbedder(),
        vmp_hierarchical_model_path=str(path),
    )

    assert adapter.name == "vmp_hierarchical"
    with pytest.raises(ValueError, match="test labels"):
        VMPHierarchicalModel.model_validate(
            {
                **model.model_dump(mode="python"),
                "metadata": {"test_labels_used": True},
            }
        )


def test_hierarchical_config_requires_session_ingestion_and_model_path() -> None:
    with pytest.raises(ValueError, match="model_path"):
        LongMemEvalRunConfig(
            data_path="data.json",
            methods=["vmp_hierarchical"],
        )
    with pytest.raises(ValueError, match="session ingestion"):
        LongMemEvalRunConfig(
            data_path="data.json",
            methods=["vmp_hierarchical"],
            ingestion_granularity="turn",
            vmp_hierarchical_model_path="model.json",
        )


def test_hierarchical_fusion_and_grid_are_deterministic() -> None:
    assert pool_top_scores([0.9, 0.5, 0.1], top_n=2) == pytest.approx(0.7)
    assert hierarchical_fusion_score(
        session_semantic=0.2,
        turn_semantic=0.8,
        turn_lexical=0.5,
        session_semantic_weight=0.25,
        turn_semantic_weight=0.50,
        turn_lexical_weight=0.25,
    ) == pytest.approx(0.575)
    grid = hierarchical_parameter_grid(
        step=0.5,
        turn_pooling_options=(1, 2),
    )
    assert len(grid) == 12
    assert grid[0].as_payload() == {
        "session_semantic_weight": 1.0,
        "turn_semantic_weight": 0.0,
        "turn_lexical_weight": 0.0,
        "turn_pooling_top_n": 1,
    }


def test_hierarchical_tuning_and_runner_keep_dev_test_isolated(
    tmp_path,
) -> None:
    records = [_hierarchical_training_record(index) for index in range(4)]
    data_path = tmp_path / "longmemeval.json"
    data_path.write_text(
        json.dumps(records),
        encoding="utf-8",
    )
    split = create_longmemeval_split(
        data_path,
        dev_size=2,
        test_size=2,
        seed=42,
    )
    split_path = split.save(tmp_path / "split.json")
    original_manifest_sha256 = sha256_file(split_path)
    embedder = HierarchyTestEmbedder()
    base_model = VMPTunedModel(
        weights={name: 0.0 for name in VMP_TUNED_FEATURES},
        semantic_anchor_weight=1.0,
        lexical_anchor_weight=0.0,
        policy_adjustment_limit=0.0,
        protected_dense_count=5,
        promotion_margin=0.0,
        promotion_ranker=None,
        split_id=split.split_id,
        split_manifest_sha256=original_manifest_sha256,
        dataset_sha256=split.dataset_sha256,
        embedding_identifier=embedder.identifier,
        best_objective=1.0,
        dev_metrics={"recall_all@5": 0.0},
        metadata={"test_labels_used": False},
    )
    base_path = base_model.save(tmp_path / "vmp_v43.json")
    regenerated_split = split.model_copy(
        update={
            "created_at": split.created_at + timedelta(seconds=1),
            "dataset_path": str(tmp_path / "moved" / data_path.name),
        }
    )
    split_path.write_text(
        regenerated_split.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    assert regenerated_split.split_id == split.split_id
    assert regenerated_split.splits == split.splits
    assert sha256_file(split_path) != original_manifest_sha256

    tuning = train_vmp_hierarchical(
        data_path,
        split_path,
        base_path,
        embedder=embedder,
        grid_step=0.5,
        turn_pooling_options=(1,),
        enable_promotion=True,
        promotion_margins=(0.0, 0.5),
        stability_folds=2,
    )
    model_path = tuning.model.save(tmp_path / "vmp_v5.json")

    assert tuning.trials_evaluated == 6
    assert tuning.model.schema_version == "2.1"
    assert tuning.model.model_type == ("vmp_v5_1_hierarchical_guarded_promotion_ranker")
    assert tuning.model.base_model.promotion_ranker is not None
    assert tuning.model.base_model.protected_dense_count == 4
    assert tuning.model.training_split == "dev"
    assert tuning.model.metadata["test_labels_used"] is False
    assert tuning.model.metadata["skipped_empty_turn_count"] == 2
    assert tuning.model.metadata["dev_metrics_source"] == ("leave_one_question_out_promotion")
    assert tuning.model.metadata["promotion_geometry"] == ("hierarchical_fused_session_turn")
    assert len(tuning.promotion_trial_summaries) == 2
    assert len(tuning.dev_audit) == 2
    assert all(row["test_labels_used"] is False for row in tuning.dev_audit)
    assert float(tuning.model.turn_semantic_weight) + float(tuning.model.turn_lexical_weight) > 0.0

    config = LongMemEvalRunConfig(
        data_path=data_path,
        methods=["vmp_hierarchical"],
        output_dir=tmp_path / "outputs",
        split_manifest_path=split_path,
        split_name="test",
        vmp_hierarchical_model_path=model_path,
    )
    result = run_longmemeval_retrieval(
        config,
        embedder=embedder,
        run_id="vmp_v5_test",
    )

    assert result.summaries["vmp_hierarchical"].processed_questions == 2
    assert result.summaries["vmp_hierarchical"].metrics["recall_all@5"] == 1.0
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["vmp_hierarchical_model"]["sha256"]
    invalid_provenance = VMPHierarchicalModel.model_validate(
        {
            **tuning.model.model_dump(mode="python"),
            "split_assignment_sha256": "different_assignment",
        }
    )
    invalid_provenance_path = invalid_provenance.save(tmp_path / "vmp_v5_invalid_provenance.json")
    with pytest.raises(ValueError, match="semantic split assignment differs"):
        run_longmemeval_retrieval(
            config.model_copy(update={"vmp_hierarchical_model_path": invalid_provenance_path}),
            embedder=embedder,
            run_id="vmp_v5_invalid_provenance",
        )
    different_split = create_longmemeval_split(
        data_path,
        dev_size=2,
        test_size=2,
        seed=7,
    )
    different_split_path = different_split.save(tmp_path / "different_split.json")
    with pytest.raises(ValueError, match="evaluation split manifest differ"):
        run_longmemeval_retrieval(
            config.model_copy(update={"split_manifest_path": different_split_path}),
            embedder=embedder,
            run_id="vmp_v5_different_split",
        )
    with pytest.raises(ValueError, match="training split"):
        run_longmemeval_retrieval(
            config.model_copy(update={"split_name": "dev"}),
            embedder=embedder,
            run_id="vmp_v5_must_not_run",
        )


def _base_model(embedding_identifier: str) -> VMPTunedModel:
    return VMPTunedModel(
        weights={name: 0.0 for name in VMP_TUNED_FEATURES},
        semantic_anchor_weight=1.0,
        lexical_anchor_weight=0.0,
        policy_adjustment_limit=0.0,
        protected_dense_count=5,
        promotion_margin=0.0,
        promotion_ranker=None,
        split_id="split",
        split_manifest_sha256="manifest",
        dataset_sha256="dataset",
        embedding_identifier=embedding_identifier,
        best_objective=1.0,
        metadata={"test_labels_used": False},
    )


def _hierarchical_sample() -> dict:
    return {
        "question_id": "q_hierarchy",
        "question_type": "single-session-user",
        "question": "Which hidden fact should I remember?",
        "answer": "needle fact",
        "question_date": "2024-02-01",
        "haystack_session_ids": ["distractor", "relevant"],
        "haystack_dates": ["2024-01-01", "2024-01-02"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "General unrelated discussion."},
                {"role": "assistant", "content": "More unrelated details."},
            ],
            [
                {
                    "role": "user",
                    "content": "A long preamble eventually contains the needle fact.",
                },
                {
                    "role": "assistant",
                    "content": "The rest of the session is generic filler.",
                },
            ],
        ],
        "answer_session_ids": ["relevant"],
        "has_answer": True,
    }


def _hierarchical_training_record(index: int) -> dict:
    distractor_ids = [f"d{index}_{item}" for item in range(5)]
    relevant_id = f"relevant_{index}"
    return {
        "question_id": f"q_hierarchy_{index}",
        "question_type": "single-session-user",
        "question": "Which hidden fact should I remember?",
        "answer": "needle fact",
        "question_date": "2024-02-01",
        "haystack_session_ids": [*distractor_ids, relevant_id],
        "haystack_dates": ["2024-01-01"] * 6,
        "haystack_sessions": [
            [
                {"role": "user", "content": "General unrelated discussion."},
                {"role": "assistant", "content": "More unrelated details."},
            ]
            for _ in distractor_ids
        ]
        + [
            [
                {
                    "role": "assistant",
                    "content": "   ",
                },
                {
                    "role": "user",
                    "content": "A long preamble eventually contains the needle fact.",
                },
                {
                    "role": "assistant",
                    "content": "The rest of the session is generic filler.",
                },
            ]
        ],
        "answer_session_ids": [relevant_id],
        "has_answer": True,
    }
