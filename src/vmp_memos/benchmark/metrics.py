"""Deterministic metrics for the toy memory-policy benchmark."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence

from vmp_memos.schemas import BenchmarkResult, BenchmarkSample, OperationType

_TOKEN_PATTERN = re.compile(r"[\w-]+", flags=re.UNICODE)


def per_sample_metrics(sample: BenchmarkSample, result: BenchmarkResult) -> dict[str, float]:
    """Compute per-sample metrics without external judge calls."""

    gold_ids = set(sample.gold_memory_ids)
    retrieved_ids = set(result.retrieved_memory_ids)
    expected_ops = [
        op.value if isinstance(op, OperationType) else str(op)
        for op in sample.expected_operations
    ]
    actual_ops = [
        op.value if isinstance(op, OperationType) else str(op)
        for op in result.operations
    ]
    stale_ids = set(_string_list(sample.metadata.get("stale_memory_ids", [])))

    evidence_precision = (
        len(gold_ids & retrieved_ids) / len(retrieved_ids)
        if retrieved_ids
        else float(not gold_ids)
    )
    evidence_recall = (
        len(gold_ids & retrieved_ids) / len(gold_ids)
        if gold_ids
        else float(not retrieved_ids)
    )
    operation_recall = (
        len(set(expected_ops) & set(actual_ops)) / len(set(expected_ops))
        if expected_ops
        else 1.0
    )
    stale_usage = (
        len(stale_ids & retrieved_ids) / len(retrieved_ids)
        if retrieved_ids
        else 0.0
    )
    memory_count_before = float(sample.metadata.get("memory_count_before", 0.0) or 0.0)
    memory_count_after = float(result.metadata.get("memory_count_after", memory_count_before))

    return {
        "accuracy": _answer_accuracy(result.answer, sample.gold_answer),
        "evidence_precision": _clamp01(evidence_precision),
        "evidence_recall": _clamp01(evidence_recall),
        "operation_recall": _clamp01(operation_recall),
        "write_precision": _write_precision(actual_ops, expected_ops),
        "update_accuracy": _operation_present_score(actual_ops, expected_ops, "UPDATE"),
        "conflict_resolution_accuracy": _conflict_resolution_score(
            sample,
            actual_ops,
            retrieved_ids,
        ),
        "conflict_retrieval_rate": _conflict_retrieval_rate(sample, retrieved_ids),
        "stale_memory_usage_rate": _clamp01(stale_usage),
        "memory_growth_rate": _memory_growth(memory_count_before, memory_count_after),
        "token_cost": float(result.token_count),
        "latency_ms": float(result.latency_ms),
    }


def aggregate_results(results: Sequence[BenchmarkResult]) -> dict[str, dict[str, float]]:
    """Aggregate per-system metrics from benchmark results."""

    grouped: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for result in results:
        grouped[result.system_name].append(result)

    summary: dict[str, dict[str, float]] = {}
    for system_name, system_results in sorted(grouped.items()):
        metric_names = sorted({key for result in system_results for key in result.metrics})
        values: dict[str, float] = {
            "num_samples": float(len(system_results)),
            "error_rate": sum(result.error is not None for result in system_results)
            / len(system_results),
        }
        for metric_name in metric_names:
            metric_values = [
                float(result.metrics.get(metric_name, 0.0))
                for result in system_results
            ]
            values[metric_name] = _mean(metric_values)
        latencies = [float(result.latency_ms) for result in system_results]
        values["p50_latency_ms"] = _percentile(latencies, 50)
        values["p95_latency_ms"] = _percentile(latencies, 95)
        summary[system_name] = values
    return summary


def _answer_accuracy(answer: str | None, gold_answer: str | Sequence[str]) -> float:
    if answer is None:
        return 0.0
    gold_values = [gold_answer] if isinstance(gold_answer, str) else list(gold_answer)
    normalized_answer = _normalize(answer)
    for gold in gold_values:
        normalized_gold = _normalize(gold)
        if not normalized_gold:
            continue
        if normalized_gold in normalized_answer:
            return 1.0
        overlap = _token_overlap(normalized_answer, normalized_gold)
        if overlap >= 0.75:
            return 1.0
    return 0.0


def _write_precision(actual_ops: Sequence[str], expected_ops: Sequence[str]) -> float:
    write_ops = {"ADD", "UPDATE", "MERGE", "ARCHIVE", "COMPRESS"}
    actual_writes = [op for op in actual_ops if op in write_ops]
    if not actual_writes:
        return float(not any(op in write_ops for op in expected_ops))
    correct = sum(op in expected_ops for op in actual_writes)
    return correct / len(actual_writes)


def _operation_present_score(
    actual_ops: Sequence[str],
    expected_ops: Sequence[str],
    op_name: str,
) -> float:
    if op_name not in expected_ops:
        return 1.0
    return float(op_name in actual_ops)


def _conflict_resolution_score(
    sample: BenchmarkSample,
    actual_ops: Sequence[str],
    retrieved_ids: set[str],
) -> float:
    if sample.metadata.get("task_type") not in {"conflict_resolution", "preference_update"}:
        return 1.0
    stale_ids = set(_string_list(sample.metadata.get("stale_memory_ids", [])))
    updated_or_merged = "UPDATE" in actual_ops or "MERGE" in actual_ops
    stale_suppressed = not (stale_ids & retrieved_ids)
    return float(updated_or_merged and stale_suppressed)


def _conflict_retrieval_rate(sample: BenchmarkSample, retrieved_ids: set[str]) -> float:
    if sample.metadata.get("task_type") not in {"conflict_resolution", "preference_update"}:
        return 0.0
    stale_ids = set(_string_list(sample.metadata.get("stale_memory_ids", [])))
    return float(bool(stale_ids & retrieved_ids))


def _memory_growth(before: float, after: float) -> float:
    denominator = max(1.0, before)
    return max(0.0, (after - before) / denominator)


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _token_overlap(left: str, right: str) -> float:
    left_terms = set(_TOKEN_PATTERN.findall(left))
    right_terms = set(_TOKEN_PATTERN.findall(right))
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(right_terms)


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _mean(values: Iterable[float]) -> float:
    numbers = list(values)
    return sum(numbers) / len(numbers) if numbers else 0.0


def _percentile(values: Sequence[float], percentile: int) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = (len(sorted_values) - 1) * percentile / 100
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[int(index)]
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))
