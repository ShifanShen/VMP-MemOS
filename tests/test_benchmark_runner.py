"""Tests for the toy benchmark runner."""

import json
from pathlib import Path

from vmp_memos.benchmark import (
    BenchmarkRunConfig,
    BenchmarkRunner,
    FullContextBaseline,
    SummaryMemoryBaseline,
    VMPRuleBaseline,
    VectorRAGImportanceBaseline,
    VectorRAGRecencyBaseline,
    aggregate_results,
    baseline_for_name,
    load_benchmark_samples,
    per_sample_metrics,
)
from vmp_memos.schemas import BenchmarkResult, OperationType

DATASET_PATH = Path("data/benchmarks/memory_policy_toy.jsonl")


def test_load_default_toy_dataset() -> None:
    samples = load_benchmark_samples(DATASET_PATH)

    assert len(samples) == 8
    assert samples[0].sample_id == "case_001_preference_update"
    assert samples[0].expected_operations == [OperationType.UPDATE, OperationType.RETRIEVE]
    assert samples[0].metadata["task_type"] == "preference_update"


def test_vmp_rule_baseline_handles_preference_update() -> None:
    sample = load_benchmark_samples(DATASET_PATH)[0]

    output = VMPRuleBaseline(top_k=3).run(sample)

    assert OperationType.UPDATE in output.operations
    assert OperationType.RETRIEVE in output.operations
    assert output.retrieved_memory_ids[0] == "mem_career_direction"
    assert "Agent and LLM" in (output.answer or "")


def test_phase8_baselines_and_vector_alias_are_available() -> None:
    sample = load_benchmark_samples(DATASET_PATH)[0]

    vector_alias = baseline_for_name("vector_rag", top_k=3).run(sample)
    full_context = FullContextBaseline(top_k=3).run(sample)
    summary = SummaryMemoryBaseline(top_k=3).run(sample)
    recency = VectorRAGRecencyBaseline(top_k=3).run(sample)
    importance = VectorRAGImportanceBaseline(top_k=3).run(sample)

    assert vector_alias.system_name == "naive_vector_rag"
    assert full_context.system_name == "full_context"
    assert full_context.token_count >= vector_alias.token_count
    assert "mem_career_direction" in full_context.retrieved_memory_ids
    assert summary.metadata["retrieval_strategy"] == "summary_lexical_top_k"
    assert recency.metadata["retrieval_strategy"] == "lexical_plus_recency"
    assert importance.metadata["retrieval_strategy"] == "lexical_plus_importance"


def test_per_sample_metrics_cover_answer_evidence_and_operations() -> None:
    sample = load_benchmark_samples(DATASET_PATH)[0]
    result = BenchmarkResult(
        sample_id=sample.sample_id,
        system_name="test",
        answer="Current career direction: Agent and LLM application development.",
        retrieved_memory_ids=["mem_career_direction"],
        operations=[OperationType.UPDATE, OperationType.RETRIEVE],
        token_count=12,
        latency_ms=3.0,
        metadata={"memory_count_after": 1},
    )

    metrics = per_sample_metrics(sample, result)

    assert metrics["accuracy"] == 1.0
    assert metrics["evidence_precision"] == 1.0
    assert metrics["operation_recall"] == 1.0
    assert metrics["update_accuracy"] == 1.0
    assert metrics["conflict_resolution_accuracy"] == 1.0
    assert metrics["conflict_retrieval_rate"] == 0.0


def test_benchmark_runner_writes_jsonl_and_markdown_report(tmp_path) -> None:
    config = BenchmarkRunConfig(
        dataset_path=DATASET_PATH,
        output_dir=tmp_path / "runs",
        report_dir=tmp_path / "reports",
        baselines=["no_memory", "naive_vector_rag", "vmp_rule"],
        top_k=3,
        run_id="unit_run",
    )

    summary = BenchmarkRunner(config).run()

    assert summary.run_id == "unit_run"
    assert summary.num_samples == 8
    assert summary.result_path == (tmp_path / "runs" / "unit_run" / "results.jsonl")
    assert summary.report_path == (tmp_path / "reports" / "unit_run.md")
    assert summary.result_path.is_file()
    assert summary.report_path.is_file()

    rows = [
        json.loads(line)
        for line in summary.result_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 24
    assert {row["system_name"] for row in rows} == {
        "no_memory",
        "naive_vector_rag",
        "vmp_rule",
    }
    assert "Aggregate metrics" in summary.report_path.read_text(encoding="utf-8")


def test_aggregate_results_groups_by_system() -> None:
    results = [
        BenchmarkResult(
            sample_id="case_1",
            system_name="a",
            metrics={"accuracy": 1.0},
            latency_ms=1.0,
        ),
        BenchmarkResult(
            sample_id="case_2",
            system_name="a",
            metrics={"accuracy": 0.0},
            latency_ms=3.0,
        ),
        BenchmarkResult(
            sample_id="case_1",
            system_name="b",
            metrics={"accuracy": 1.0},
            latency_ms=2.0,
        ),
    ]

    summary = aggregate_results(results)

    assert summary["a"]["accuracy"] == 0.5
    assert summary["a"]["p50_latency_ms"] == 2.0
    assert summary["b"]["accuracy"] == 1.0
