"""Markdown report generation for benchmark runs."""

from __future__ import annotations

from pathlib import Path

from vmp_memos.schemas import BenchmarkResult


def write_markdown_report(
    path: str | Path,
    *,
    run_id: str,
    dataset_path: str,
    results: list[BenchmarkResult],
    aggregate_metrics: dict[str, dict[str, float]],
) -> Path:
    """Write a compact Markdown report for one benchmark run."""

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(_report_lines(run_id, dataset_path, results, aggregate_metrics)),
        encoding="utf-8",
        newline="\n",
    )
    return output_path


def _report_lines(
    run_id: str,
    dataset_path: str,
    results: list[BenchmarkResult],
    aggregate_metrics: dict[str, dict[str, float]],
) -> list[str]:
    lines = [
        f"# VMP-MemOS Benchmark Report: `{run_id}`",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Results: {len(results)} sample/system rows",
        "",
        "## Aggregate metrics",
        "",
    ]
    metric_names = _ordered_metric_names(aggregate_metrics)
    lines.append("| System | " + " | ".join(metric_names) + " |")
    lines.append("|---|" + "|".join("---" for _ in metric_names) + "|")
    for system_name, metrics in sorted(aggregate_metrics.items()):
        values = [_format_metric(metrics.get(name, 0.0)) for name in metric_names]
        lines.append(f"| {system_name} | " + " | ".join(values) + " |")

    lines.extend(["", "## Per-sample results", ""])
    lines.append("| Sample | System | Correct | Retrieved | Operations | Error |")
    lines.append("|---|---|---:|---|---|---|")
    for result in results:
        retrieved = ", ".join(result.retrieved_memory_ids) or "-"
        operations = (
            ", ".join(op.value if hasattr(op, "value") else str(op) for op in result.operations)
            or "-"
        )
        correct = "-" if result.is_correct is None else str(int(result.is_correct))
        error = result.error or ""
        lines.append(
            f"| {result.sample_id} | {result.system_name} | {correct} | "
            f"{retrieved} | {operations} | {error} |"
        )
    lines.append("")
    return lines


def _ordered_metric_names(aggregate_metrics: dict[str, dict[str, float]]) -> list[str]:
    preferred = [
        "accuracy",
        "evidence_precision",
        "evidence_recall",
        "operation_recall",
        "write_precision",
        "update_accuracy",
        "conflict_resolution_accuracy",
        "conflict_retrieval_rate",
        "stale_memory_usage_rate",
        "memory_growth_rate",
        "token_cost",
        "p50_latency_ms",
        "p95_latency_ms",
        "error_rate",
    ]
    available = {name for metrics in aggregate_metrics.values() for name in metrics}
    ordered = [name for name in preferred if name in available]
    ordered.extend(sorted(available - set(ordered) - {"num_samples"}))
    return ["num_samples", *ordered]


def _format_metric(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.3f}"
