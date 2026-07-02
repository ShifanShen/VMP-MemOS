"""Tests for Phase 9 learned-policy training and inference."""

from pathlib import Path

from vmp_memos.benchmark import build_policy_training_examples, load_benchmark_samples
from vmp_memos.benchmark.baselines import LearnedPolicyBaseline
from vmp_memos.policy import LogisticPolicyModel
from vmp_memos.schemas import OperationType

DATASET_PATH = Path("data/benchmarks/memory_policy_toy.jsonl")


def test_benchmark_training_examples_cover_policy_operations() -> None:
    samples = load_benchmark_samples(DATASET_PATH)

    examples = build_policy_training_examples(samples)
    labels = {example.label for example in examples}

    assert len(examples) >= 20
    assert OperationType.ADD in labels
    assert OperationType.UPDATE in labels
    assert OperationType.MERGE in labels
    assert OperationType.ARCHIVE in labels
    assert OperationType.RETRIEVE in labels
    assert OperationType.IGNORE in labels
    assert all(len(example.as_vector()) == 16 for example in examples)


def test_logistic_policy_model_saves_loads_and_predicts(tmp_path) -> None:
    samples = load_benchmark_samples(DATASET_PATH)
    examples = build_policy_training_examples(samples)

    model = LogisticPolicyModel.train(examples, epochs=120, learning_rate=0.35)
    model_path = model.save(tmp_path / "learned_policy.json")
    loaded = LogisticPolicyModel.load(model_path)
    prediction = loaded.predict(examples[0].features)

    assert model_path.is_file()
    assert prediction.predicted_op in loaded.labels
    assert abs(sum(prediction.probabilities.values()) - 1.0) < 1e-6
    assert loaded.metadata["num_examples"] == len(examples)


def test_learned_policy_baseline_runs_with_trained_model(tmp_path) -> None:
    samples = load_benchmark_samples(DATASET_PATH)
    examples = build_policy_training_examples(samples)
    model_path = LogisticPolicyModel.train(
        examples,
        epochs=200,
        learning_rate=0.35,
    ).save(tmp_path / "learned_policy.json")

    output = LearnedPolicyBaseline(top_k=3, model_path=model_path).run(samples[0])

    assert output.system_name == "learned_policy"
    assert OperationType.RETRIEVE in output.operations
    assert output.retrieved_memory_ids
    assert output.metadata["model_type"] == "multiclass_logistic_regression"
