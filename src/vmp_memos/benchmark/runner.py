"""Benchmark runner that wires datasets, baselines, metrics, and reports."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from pydantic import Field

from vmp_memos.benchmark.baselines import BaselineOutput, baseline_for_name_with_options
from vmp_memos.benchmark.datasets import load_benchmark_samples, write_benchmark_results
from vmp_memos.benchmark.metrics import aggregate_results, per_sample_metrics
from vmp_memos.benchmark.reports import write_markdown_report
from vmp_memos.schemas import BenchmarkResult
from vmp_memos.schemas.base import NonEmptyStr, SchemaModel, new_id


class BenchmarkRunConfig(SchemaModel):
    """Configuration for one benchmark run."""

    dataset_path: Path = Path("data/benchmarks/memory_policy_toy.jsonl")
    output_dir: Path = Path("outputs/runs")
    report_dir: Path = Path("outputs/reports")
    baselines: list[NonEmptyStr] = Field(
        default_factory=lambda: [
            "no_memory",
            "full_context",
            "summary_memory",
            "naive_vector_rag",
            "vector_rag_recency",
            "vector_rag_importance",
            "vmp_rule",
        ]
    )
    top_k: int = Field(default=3, ge=1)
    run_id: NonEmptyStr | None = None
    policy_model_path: Path = Path("outputs/models/learned_policy.json")


class BenchmarkRunSummary(SchemaModel):
    """Paths and aggregate metrics produced by one benchmark run."""

    run_id: NonEmptyStr
    dataset_path: Path
    result_path: Path
    report_path: Path
    num_samples: int
    baselines: list[NonEmptyStr]
    aggregate_metrics: dict[str, dict[str, float]]


class BenchmarkRunner:
    """Run configured baselines over a validated benchmark dataset."""

    def __init__(self, config: BenchmarkRunConfig) -> None:
        self.config = config

    def run(self) -> BenchmarkRunSummary:
        """Execute the full benchmark and write result artifacts."""

        run_id = self.config.run_id or _default_run_id()
        samples = load_benchmark_samples(self.config.dataset_path)
        baselines = [
            baseline_for_name_with_options(
                name,
                top_k=self.config.top_k,
                model_path=self.config.policy_model_path,
            )
            for name in self.config.baselines
        ]
        results: list[BenchmarkResult] = []
        for sample in samples:
            for baseline in baselines:
                started_at = perf_counter()
                try:
                    output = baseline.run(sample)
                    result = self._result_from_output(sample.sample_id, output)
                except Exception as exc:
                    result = BenchmarkResult(
                        sample_id=sample.sample_id,
                        system_name=baseline.name,
                        error=str(exc),
                        latency_ms=(perf_counter() - started_at) * 1000.0,
                    )
                metrics = per_sample_metrics(sample, result)
                result = result.model_copy(
                    update={
                        "is_correct": bool(metrics["accuracy"]),
                        "metrics": metrics,
                    }
                )
                results.append(result)

        run_dir = self.config.output_dir / run_id
        result_path = write_benchmark_results(run_dir / "results.jsonl", results)
        aggregate = aggregate_results(results)
        report_path = write_markdown_report(
            self.config.report_dir / f"{run_id}.md",
            run_id=run_id,
            dataset_path=str(self.config.dataset_path),
            results=results,
            aggregate_metrics=aggregate,
        )
        return BenchmarkRunSummary(
            run_id=run_id,
            dataset_path=self.config.dataset_path,
            result_path=result_path,
            report_path=report_path,
            num_samples=len(samples),
            baselines=[baseline.name for baseline in baselines],
            aggregate_metrics=aggregate,
        )

    @staticmethod
    def _result_from_output(sample_id: str, output: BaselineOutput) -> BenchmarkResult:
        return BenchmarkResult(
            sample_id=sample_id,
            system_name=output.system_name,
            answer=output.answer,
            retrieved_memory_ids=output.retrieved_memory_ids,
            operations=output.operations,
            token_count=output.token_count,
            latency_ms=output.latency_ms,
            metadata=output.metadata,
        )


def _default_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{new_id('run')}"
