"""Feature-ablation runner and report generation."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from pydantic import Field

from vmp_memos.benchmark.baselines import (
    BaselineOutput,
    VMPRuleBaseline,
    baseline_for_name,
)
from vmp_memos.benchmark.datasets import load_benchmark_samples, write_benchmark_results
from vmp_memos.benchmark.metrics import aggregate_results, per_sample_metrics
from vmp_memos.schemas import BenchmarkResult, BenchmarkSample, PolicyFeatures
from vmp_memos.schemas.base import NonEmptyStr, SchemaModel, new_id

DEFAULT_ABLATION_FEATURES: tuple[str, ...] = (
    "recency",
    "contradiction",
    "redundancy",
    "success_contribution",
    "token_cost",
)


class AblationRunConfig(SchemaModel):
    """Configuration for one feature-ablation run."""

    dataset_path: Path = Path("data/benchmarks/memory_policy_toy.jsonl")
    output_dir: Path = Path("outputs/runs")
    report_path: Path = Path("outputs/reports/ablation.md")
    baseline_names: list[NonEmptyStr] = Field(
        default_factory=lambda: ["no_memory", "vector_rag", "vmp_rule"]
    )
    disabled_features: list[NonEmptyStr] = Field(
        default_factory=lambda: list(DEFAULT_ABLATION_FEATURES)
    )
    top_k: int = Field(default=3, ge=1)
    run_id: NonEmptyStr | None = None
    max_error_cases: int = Field(default=10, ge=0)


class AblationRunSummary(SchemaModel):
    """Paths and aggregate metrics produced by an ablation run."""

    run_id: NonEmptyStr
    dataset_path: Path
    result_path: Path
    report_path: Path
    num_samples: int
    systems: list[NonEmptyStr]
    disabled_features: list[NonEmptyStr]
    aggregate_metrics: dict[str, dict[str, float]]


def run_ablation(config: AblationRunConfig) -> AblationRunSummary:
    """Run baseline comparison plus one-feature-at-a-time VMP ablations."""

    _validate_features(config.disabled_features)
    run_id = config.run_id or _default_ablation_run_id()
    samples = load_benchmark_samples(config.dataset_path)
    baselines = [
        baseline_for_name(name, top_k=config.top_k)
        for name in config.baseline_names
    ]
    ablations = [
        VMPRuleBaseline(
            top_k=config.top_k,
            disabled_features=[feature],
            system_name=f"vmp_rule__no_{feature}",
        )
        for feature in dict.fromkeys(config.disabled_features)
    ]
    all_systems = [*baselines, *ablations]

    results: list[BenchmarkResult] = []
    for sample in samples:
        for system in all_systems:
            started_at = perf_counter()
            try:
                output = system.run(sample)
                result = _result_from_output(sample.sample_id, output)
            except Exception as exc:
                result = BenchmarkResult(
                    sample_id=sample.sample_id,
                    system_name=system.name,
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

    result_path = write_benchmark_results(
        config.output_dir / run_id / "results.jsonl",
        results,
    )
    aggregate = aggregate_results(results)
    report_path = write_ablation_report(
        config.report_path,
        run_id=run_id,
        config=config,
        samples=samples,
        results=results,
        aggregate_metrics=aggregate,
    )
    return AblationRunSummary(
        run_id=run_id,
        dataset_path=config.dataset_path,
        result_path=result_path,
        report_path=report_path,
        num_samples=len(samples),
        systems=[system.name for system in all_systems],
        disabled_features=list(dict.fromkeys(config.disabled_features)),
        aggregate_metrics=aggregate,
    )


def write_ablation_report(
    path: str | Path,
    *,
    run_id: str,
    config: AblationRunConfig,
    samples: Sequence[BenchmarkSample],
    results: Sequence[BenchmarkResult],
    aggregate_metrics: dict[str, dict[str, float]],
) -> Path:
    """Write the Phase 10 Markdown ablation report."""

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            _report_lines(
                run_id,
                config,
                samples,
                results,
                aggregate_metrics,
            )
        ),
        encoding="utf-8",
        newline="\n",
    )
    return output_path


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


def _report_lines(
    run_id: str,
    config: AblationRunConfig,
    samples: Sequence[BenchmarkSample],
    results: Sequence[BenchmarkResult],
    aggregate_metrics: dict[str, dict[str, float]],
) -> list[str]:
    baseline_systems = _baseline_system_names(config.baseline_names)
    ablation_systems = [
        f"vmp_rule__no_{feature}"
        for feature in dict.fromkeys(config.disabled_features)
    ]
    lines = [
        f"# VMP-MemOS Ablation Report: `{run_id}`",
        "",
        "## Experiment settings",
        "",
        f"- Dataset: `{config.dataset_path}`",
        f"- Samples: {len(samples)}",
        f"- Top-K: {config.top_k}",
        f"- Baselines: `{', '.join(baseline_systems)}`",
        f"- Disabled features: `{', '.join(dict.fromkeys(config.disabled_features))}`",
        "",
        "## Baseline comparison",
        "",
    ]
    lines.extend(_metrics_table(aggregate_metrics, baseline_systems))
    lines.extend(["", "## Ablation comparison", ""])
    lines.extend(_ablation_delta_table(aggregate_metrics, ablation_systems))
    lines.extend(["", "## Full metrics table", ""])
    lines.extend(_metrics_table(aggregate_metrics, sorted(aggregate_metrics)))
    lines.extend(["", "## Error cases", ""])
    lines.extend(_error_case_lines(results, max_cases=config.max_error_cases))
    lines.extend(["", "## Memory operation examples", ""])
    lines.extend(_operation_example_lines(results, ablation_systems))
    lines.append("")
    return lines


def _metrics_table(
    aggregate_metrics: dict[str, dict[str, float]],
    systems: Sequence[str],
) -> list[str]:
    metric_names = _ordered_metric_names(aggregate_metrics, systems)
    lines = ["| System | " + " | ".join(metric_names) + " |"]
    lines.append("|---|" + "|".join("---:" for _ in metric_names) + "|")
    for system_name in systems:
        metrics = aggregate_metrics.get(system_name)
        if metrics is None:
            continue
        values = [_format_metric(metrics.get(name, 0.0)) for name in metric_names]
        lines.append(f"| {_cell(system_name)} | " + " | ".join(values) + " |")
    return lines


def _ablation_delta_table(
    aggregate_metrics: dict[str, dict[str, float]],
    ablation_systems: Sequence[str],
) -> list[str]:
    reference = aggregate_metrics.get("vmp_rule", {})
    metrics = [
        "accuracy",
        "evidence_recall",
        "operation_recall",
        "conflict_retrieval_rate",
        "stale_memory_usage_rate",
        "token_cost",
    ]
    lines = [
        "| Ablation | Disabled feature | "
        + " | ".join(f"Δ {metric}" for metric in metrics)
        + " |"
    ]
    lines.append("|---|---|" + "|".join("---:" for _ in metrics) + "|")
    for system_name in ablation_systems:
        system_metrics = aggregate_metrics.get(system_name, {})
        feature_name = system_name.removeprefix("vmp_rule__no_")
        deltas = [
            _format_metric(system_metrics.get(metric, 0.0) - reference.get(metric, 0.0))
            for metric in metrics
        ]
        lines.append(
            f"| {_cell(system_name)} | `{_cell(feature_name)}` | "
            + " | ".join(deltas)
            + " |"
        )
    return lines


def _error_case_lines(
    results: Sequence[BenchmarkResult],
    *,
    max_cases: int,
) -> list[str]:
    if max_cases == 0:
        return ["Error-case listing disabled."]
    cases = [
        result
        for result in results
        if result.error is not None or result.is_correct is False
    ][:max_cases]
    if not cases:
        return ["No incorrect or errored rows in this run."]
    lines = ["| Sample | System | Error | Answer | Retrieved | Operations |"]
    lines.append("|---|---|---|---|---|---|")
    for result in cases:
        lines.append(
            f"| {_cell(result.sample_id)} | {_cell(result.system_name)} | "
            f"{_cell(result.error or '')} | {_cell(result.answer or '')} | "
            f"{_cell(', '.join(result.retrieved_memory_ids) or '-')} | "
            f"{_cell(_operation_list(result))} |"
        )
    return lines


def _operation_example_lines(
    results: Sequence[BenchmarkResult],
    ablation_systems: Sequence[str],
) -> list[str]:
    by_key = {
        (result.sample_id, result.system_name): result
        for result in results
    }
    lines = ["| Sample | VMP operations | Ablation | Ablation operations |"]
    lines.append("|---|---|---|---|")
    emitted = 0
    for (sample_id, system_name), result in sorted(by_key.items()):
        if system_name != "vmp_rule":
            continue
        base_ops = _operation_list(result)
        for ablation_name in ablation_systems:
            ablated = by_key.get((sample_id, ablation_name))
            if ablated is None:
                continue
            ablated_ops = _operation_list(ablated)
            if ablated_ops == base_ops:
                continue
            lines.append(
                f"| {_cell(sample_id)} | {_cell(base_ops)} | "
                f"{_cell(ablation_name)} | {_cell(ablated_ops)} |"
            )
            emitted += 1
            if emitted >= 12:
                return lines
    if emitted == 0:
        lines.append("| - | No operation differences observed. | - | - |")
    return lines


def _baseline_system_names(names: Sequence[str]) -> list[str]:
    system_names: list[str] = []
    for name in names:
        normalized = name.strip().casefold()
        if normalized == "vector_rag":
            system_names.append("naive_vector_rag")
        else:
            system_names.append(normalized)
    return list(dict.fromkeys(system_names))


def _ordered_metric_names(
    aggregate_metrics: dict[str, dict[str, float]],
    systems: Sequence[str],
) -> list[str]:
    preferred = [
        "accuracy",
        "evidence_precision",
        "evidence_recall",
        "operation_recall",
        "conflict_retrieval_rate",
        "stale_memory_usage_rate",
        "memory_growth_rate",
        "token_cost",
        "error_rate",
    ]
    available = {
        metric
        for system in systems
        for metric in aggregate_metrics.get(system, {})
    }
    ordered = [metric for metric in preferred if metric in available]
    ordered.extend(sorted(available - set(ordered) - {"num_samples"}))
    return ["num_samples", *ordered]


def _operation_list(result: BenchmarkResult) -> str:
    return ", ".join(
        operation.value if hasattr(operation, "value") else str(operation)
        for operation in result.operations
    ) or "-"


def _validate_features(features: Sequence[str]) -> None:
    unknown = sorted(set(features) - set(PolicyFeatures.FEATURE_NAMES))
    if unknown:
        raise ValueError(f"Unknown policy features for ablation: {', '.join(unknown)}")


def _format_metric(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.3f}"


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _default_ablation_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{new_id('ablation')}"
