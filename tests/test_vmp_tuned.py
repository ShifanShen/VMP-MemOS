"""Tests for dev-only VMP tuning and frozen test evaluation."""

from __future__ import annotations

import json

import pytest

from vmp_memos.frameworks import VMPTunedModel, adapter_for_name
from vmp_memos.frameworks.vmp_tuned import guarded_ranked_indices
from vmp_memos.longmemeval import LongMemEvalRunConfig
from vmp_memos.longmemeval.retrieval_runner import run_longmemeval_retrieval
from vmp_memos.longmemeval.splits import create_longmemeval_split
from vmp_memos.longmemeval.tuning import train_vmp_tuned
from vmp_memos.schemas import PolicyFeatures


def test_vmp_tuned_trains_on_dev_and_runs_only_on_test(tmp_path) -> None:
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
        trials=3,
        tuning_seed=7,
    )
    model_path = tuning.model.save(tmp_path / "vmp_tuned.json")

    assert tuning.model.training_split == "dev"
    assert tuning.model.split_id == split.split_id
    assert tuning.model.metadata["test_labels_used"] is False
    assert tuning.trials_evaluated == 3
    assert VMPTunedModel.load(model_path).weights == tuning.model.weights
    assert adapter_for_name(
        "vmp-tuned",
        vmp_tuned_model_path=str(model_path),
    ).name == "vmp_tuned"

    config = LongMemEvalRunConfig(
        data_path=data_path,
        methods=["vmp_tuned"],
        top_k=5,
        retrieval_depth=10,
        output_dir=tmp_path / "outputs",
        split_manifest_path=split_path,
        split_name="test",
        vmp_tuned_model_path=model_path,
    )
    result = run_longmemeval_retrieval(config, run_id="vmp_tuned_test")
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["split"]["name"] == "test"
    assert manifest["split"]["split_id"] == split.split_id
    assert manifest["vmp_tuned_model"]["sha256"]
    assert result.summaries["vmp_tuned"].processed_questions == 2

    dev_config = config.model_copy(update={"split_name": "dev"})
    with pytest.raises(ValueError, match="training split"):
        run_longmemeval_retrieval(dev_config, run_id="must_not_run")


def test_vmp_tuned_rejects_models_with_obsolete_feature_semantics(tmp_path) -> None:
    payload = {
        "schema_version": "1.1",
        "weights": {
            name: 0.0
            for name in (
                "semantic_relevance",
                "importance",
                "scope_match",
                "confidence",
                "success_contribution",
                "recency",
                "contradiction",
                "redundancy",
                "token_cost",
                "staleness",
                "update_signal",
                "action_signal",
            )
        },
        "split_id": "old",
        "split_manifest_sha256": "old",
        "dataset_sha256": "old",
        "best_objective": 0.0,
    }
    model_path = tmp_path / "old_model.json"
    model_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="schema is obsolete"):
        VMPTunedModel.load(model_path)


def test_vmp_v4_safe_trial_is_dense_only_and_model_records_safety_bounds(
    tmp_path,
) -> None:
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
        trials=1,
        tuning_seed=7,
    )

    assert tuning.model.schema_version == "1.4"
    assert tuning.model.semantic_anchor_weight == 1.0
    assert tuning.model.lexical_anchor_weight == 0.0
    assert tuning.model.policy_adjustment_limit == 0.0
    assert tuning.model.retrieve_threshold == 0.0
    assert tuning.model.protected_dense_count == 5
    assert tuning.model.preserve_dense_top_n == 10
    assert tuning.model.metadata["ranking_pipeline"] == (
        "dense_top10_safety_set -> guarded_top5_policy_rerank -> "
        "cached_non_destructive_lifecycle"
    )
    assert "macro_type_recall_all@5" in tuning.model.dev_metrics
    assert "min_fold_recall_all@5" in tuning.model.dev_metrics


def test_vmp_v4_policy_delta_is_bounded_independently_of_lifecycle() -> None:
    model = VMPTunedModel(
        weights={
            name: 1.0
            for name in (
                "semantic_relevance",
                "importance",
                "scope_match",
                "confidence",
                "success_contribution",
                "recency",
                "contradiction",
                "redundancy",
                "token_cost",
                "staleness",
                "update_signal",
                "action_signal",
            )
        },
        policy_adjustment_limit=0.05,
        archive_score_penalty=0.03,
        split_id="split",
        split_manifest_sha256="manifest",
        dataset_sha256="dataset",
        best_objective=0.0,
    )
    features = PolicyFeatures(
        semantic_relevance=1.0,
        importance=1.0,
        confidence=1.0,
        recency=1.0,
        contradiction=1.0,
        redundancy=1.0,
        token_cost=1.0,
        staleness=1.0,
        actionability=1.0,
    )

    delta = model.policy_delta(features, temporal_intent=True)
    active = model.score(
        features,
        anchor_score=0.5,
        temporal_intent=True,
    )
    superseded = model.score(
        features,
        anchor_score=0.5,
        temporal_intent=True,
        lifecycle_status="superseded",
    )

    assert 0.0 <= delta <= 0.05
    assert active == pytest.approx(0.5 + delta)
    assert superseded == pytest.approx(active - 0.03)


def test_vmp_v4_guard_preserves_dense_top10_and_four_dense_head_items() -> None:
    model = VMPTunedModel(
        weights={
            name: 0.0
            for name in (
                "semantic_relevance",
                "importance",
                "scope_match",
                "confidence",
                "success_contribution",
                "recency",
                "contradiction",
                "redundancy",
                "token_cost",
                "staleness",
                "update_signal",
                "action_signal",
            )
        },
        protected_dense_count=4,
        promotion_margin=0.01,
        split_id="split",
        split_manifest_sha256="manifest",
        dataset_sha256="dataset",
        best_objective=0.0,
    )
    dense = list(range(12))
    scores = {index: 1.0 - index * 0.01 for index in dense}
    scores[6] = 1.05
    selected = guarded_ranked_indices(
        dense_ranked_indices=dense,
        policy_scores=scores,
        anchor_scores=scores,
        requested_top_k=10,
        model=model,
    )

    assert set(selected) == set(range(10))
    assert len(set(selected[:5]) & set(range(5))) >= 4
    assert 6 in selected[:5]


def _record(index: int) -> dict:
    old_session = f"q{index}_old"
    new_session = f"q{index}_new"
    return {
        "question_id": f"q{index}",
        "question_type": "knowledge_update",
        "question": f"What activity does person {index} now prefer?",
        "answer": "swimming",
        "question_date": "2024-02-01",
        "haystack_session_ids": [old_session, new_session],
        "haystack_dates": ["2024-01-01", "2024-01-20"],
        "haystack_sessions": [
            [{"role": "user", "content": f"Person {index} liked hiking."}],
            [
                {
                    "role": "user",
                    "content": f"Person {index} now prefers swimming instead of hiking.",
                }
            ],
        ],
        "answer_session_ids": [new_session],
        "has_answer": True,
    }
