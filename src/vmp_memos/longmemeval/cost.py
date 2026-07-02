"""Offline cost and efficiency analysis for completed LongMemEval runs."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from pydantic import Field, JsonValue

from vmp_memos.longmemeval.qa_runner import QASampleRecord
from vmp_memos.longmemeval.retrieval_runner import RetrievalSampleRecord
from vmp_memos.schemas.base import (
    NonEmptyStr,
    NonNegativeFloat,
    NonNegativeInt,
    SchemaModel,
    Score,
)

_OFFICIAL_METHODS = {
    "mem0",
    "mem0_official",
    "langmem",
    "langmem_official",
    "graphiti",
    "graphiti_official",
    "letta",
    "letta_official",
}

_TABLE_COLUMNS = (
    "method",
    "samples",
    "correct_answers",
    "local_exact_accuracy",
    "mean_ingest_latency_ms",
    "mean_retrieval_latency_ms",
    "mean_reader_latency_ms",
    "mean_end_to_end_latency_ms",
    "delta_end_to_end_latency_ms",
    "p95_end_to_end_latency_ms",
    "mean_retrieved_tokens",
    "mean_reader_input_tokens",
    "mean_reader_output_tokens",
    "delta_reader_tokens",
    "framework_llm_tokens",
    "framework_usage_coverage",
    "mean_memory_count",
    "delta_memory_count",
    "memory_retention_ratio",
    "mean_storage_size_mb",
    "storage_size_coverage",
    "storage_estimate_fraction",
    "observed_tokens_per_correct",
    "milliseconds_per_correct",
)


class CostMethodSummary(SchemaModel):
    """Cost metrics for one method over an aligned question set."""

    method: NonEmptyStr
    samples: NonNegativeInt
    qa_samples: NonNegativeInt
    correct_answers: NonNegativeInt
    local_exact_accuracy: Score
    mean_ingest_latency_ms: NonNegativeFloat
    p95_ingest_latency_ms: NonNegativeFloat
    mean_retrieval_latency_ms: NonNegativeFloat
    p95_retrieval_latency_ms: NonNegativeFloat
    mean_reader_latency_ms: NonNegativeFloat
    p95_reader_latency_ms: NonNegativeFloat
    mean_end_to_end_latency_ms: NonNegativeFloat
    p95_end_to_end_latency_ms: NonNegativeFloat
    total_retrieved_tokens: NonNegativeInt
    mean_retrieved_tokens: NonNegativeFloat
    total_reader_input_tokens: NonNegativeInt
    mean_reader_input_tokens: NonNegativeFloat
    total_reader_output_tokens: NonNegativeInt
    mean_reader_output_tokens: NonNegativeFloat
    framework_llm_input_tokens: NonNegativeInt | None = None
    framework_llm_output_tokens: NonNegativeInt | None = None
    framework_llm_tokens: NonNegativeInt | None = None
    framework_usage_coverage: Score
    total_observed_tokens: NonNegativeInt
    mean_memory_count: NonNegativeFloat
    mean_memory_tokens: NonNegativeFloat
    memory_retention_ratio: NonNegativeFloat
    mean_storage_size_bytes: NonNegativeFloat
    mean_storage_size_mb: NonNegativeFloat
    storage_size_coverage: Score
    storage_estimate_fraction: Score
    update_operations: NonNegativeInt = 0
    merge_operations: NonNegativeInt = 0
    archive_operations: NonNegativeInt = 0
    observed_tokens_per_correct: NonNegativeFloat | None = None
    milliseconds_per_correct: NonNegativeFloat | None = None


class CostAnalysisReport(SchemaModel):
    """Replayable cost report tied to retrieval and QA manifest hashes."""

    schema_version: NonEmptyStr = "1.0"
    retrieval_run: NonEmptyStr
    retrieval_manifest_sha256: NonEmptyStr
    qa_manifest_sha256: str | None = None
    qa_complete: bool
    reference_method: str | None = None
    generated_at: datetime
    methods: dict[NonEmptyStr, CostMethodSummary]
    definitions: dict[str, JsonValue] = Field(default_factory=dict)


def analyze_longmemeval_cost(
    retrieval_run: str | Path,
    *,
    require_qa: bool = True,
) -> CostAnalysisReport:
    """Aggregate existing artifacts without running retrieval or generation."""

    run_dir = Path(retrieval_run).expanduser().resolve()
    retrieval_manifest_path = run_dir / "manifest.json"
    retrieval_manifest = _read_json(retrieval_manifest_path)
    if retrieval_manifest.get("status") != "completed":
        raise ValueError(f"retrieval run is not completed: {run_dir}")
    methods = _manifest_methods(retrieval_manifest)
    if not methods:
        raise ValueError("retrieval run contains no methods")

    qa_manifest_path = run_dir / "qa" / "manifest.json"
    qa_complete = False
    if qa_manifest_path.exists():
        qa_complete = _read_json(qa_manifest_path).get("status") == "completed"
    if require_qa and not qa_complete:
        raise ValueError("completed QA artifacts are required for cost analysis")

    summaries: dict[str, CostMethodSummary] = {}
    for method in methods:
        retrieval_records = _read_retrieval_records(
            run_dir / method / "retrieval.jsonl"
        )
        qa_records = (
            _read_qa_records(run_dir / "qa" / f"{method}.jsonl")
            if qa_complete
            else []
        )
        if require_qa:
            _validate_alignment(method, retrieval_records, qa_records)
        summaries[method] = _summarize_cost(
            method,
            retrieval_records=retrieval_records,
            qa_records=qa_records,
        )

    return CostAnalysisReport(
        retrieval_run=str(run_dir),
        retrieval_manifest_sha256=_sha256(retrieval_manifest_path),
        qa_manifest_sha256=_sha256(qa_manifest_path) if qa_complete else None,
        qa_complete=qa_complete,
        reference_method=_reference_method(methods),
        generated_at=datetime.now(UTC),
        methods=summaries,
        definitions={
            "token_accounting": (
                "Reader input already contains retrieved evidence, so retrieved "
                "tokens are reported separately and are not double-counted in "
                "total_observed_tokens."
            ),
            "correct_answer": (
                "normalized_exact_match == 1 for answerable questions; "
                "abstention_accuracy == 1 for abstention questions."
            ),
            "cost_per_correct_answer": (
                "Total observed reader/framework tokens and total measured "
                "pipeline milliseconds divided by locally correct answers."
            ),
            "framework_usage": (
                "Missing official-framework internal LLM usage remains null, "
                "with coverage reported explicitly; it is never imputed as zero."
            ),
            "storage_coverage": (
                "Remote or graph-backed storage without a measurable byte size "
                "is reported with zero coverage, not interpreted as zero cost."
            ),
            "monetary_cost": (
                "Not estimated because all primary experiments use local vLLM."
            ),
        },
    )


def export_longmemeval_cost(
    retrieval_run: str | Path,
    *,
    output_dir: str | Path | None = None,
    require_qa: bool = True,
) -> dict[str, Path]:
    """Write JSON plus CSV/Markdown/LaTeX Table 5 artifacts."""

    report = analyze_longmemeval_cost(retrieval_run, require_qa=require_qa)
    run_dir = Path(report.retrieval_run)
    target = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else run_dir.parent.parent / "tables"
    )
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "cost_analysis.json"
    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    table_paths = _write_table(
        target / "table5_cost",
        [
            _table_row(
                summary,
                reference=(
                    report.methods.get(report.reference_method)
                    if report.reference_method is not None
                    else None
                ),
            )
            for summary in report.methods.values()
        ],
    )
    return {"cost_analysis_json": json_path, **table_paths}


def _summarize_cost(
    method: str,
    *,
    retrieval_records: list[RetrievalSampleRecord],
    qa_records: list[QASampleRecord],
) -> CostMethodSummary:
    sample_count = len(retrieval_records)
    qa_count = len(qa_records)
    ingest = [_adapter_number(record, "total_ingest_latency_ms") for record in retrieval_records]
    retrieval = [
        _adapter_number(record, "total_retrieval_latency_ms")
        for record in retrieval_records
    ]
    reader = [float(record.reader_latency_ms) for record in qa_records]
    end_to_end = [float(record.end_to_end_latency_ms) for record in qa_records]
    retrieved_tokens = [record.retrieved_tokens for record in qa_records]
    if not qa_records:
        retrieved_tokens = [record.retrieved_tokens for record in retrieval_records]
    reader_input = [record.reader_input_tokens for record in qa_records]
    reader_output = [record.reader_output_tokens for record in qa_records]
    correct = sum(_is_locally_correct(record) for record in qa_records)

    framework_usage = [
        usage
        for record in retrieval_records
        if (usage := _framework_usage(record)) is not None
    ]
    is_official = method in _OFFICIAL_METHODS
    coverage = (
        len(framework_usage) / sample_count
        if sample_count
        else 0.0
    )
    if not is_official:
        coverage = 1.0
    framework_input = (
        sum(usage[0] for usage in framework_usage)
        if framework_usage
        else None if is_official else 0
    )
    framework_output = (
        sum(usage[1] for usage in framework_usage)
        if framework_usage
        else None if is_official else 0
    )
    framework_total = (
        sum(usage[2] for usage in framework_usage)
        if framework_usage
        else None if is_official else 0
    )
    observed_tokens = sum(reader_input) + sum(reader_output) + (framework_total or 0)

    memory_counts = [_adapter_number(record, "memory_count") for record in retrieval_records]
    memory_tokens = [_adapter_number(record, "total_tokens") for record in retrieval_records]
    storage_sizes = [
        _adapter_number(record, "storage_size_bytes")
        for record in retrieval_records
    ]
    retention_ratios: list[float] = []
    for record in retrieval_records:
        ratio = _memory_retention_ratio(record)
        if ratio is not None:
            retention_ratios.append(ratio)
    storage_estimates = [
        bool(record.adapter_stats.get("storage_size_is_estimate", False))
        for record in retrieval_records
    ]
    storage_available = [
        not is_estimate or size > 0
        for size, is_estimate in zip(
            storage_sizes,
            storage_estimates,
            strict=True,
        )
    ]
    operation_counts = {
        operation: sum(_operation_count(record, operation) for record in retrieval_records)
        for operation in ("update", "merge", "archive")
    }
    total_pipeline_ms = sum(end_to_end)
    return CostMethodSummary(
        method=method,
        samples=sample_count,
        qa_samples=qa_count,
        correct_answers=correct,
        local_exact_accuracy=correct / qa_count if qa_count else 0.0,
        mean_ingest_latency_ms=_mean(ingest),
        p95_ingest_latency_ms=_percentile(ingest, 0.95),
        mean_retrieval_latency_ms=_mean(retrieval),
        p95_retrieval_latency_ms=_percentile(retrieval, 0.95),
        mean_reader_latency_ms=_mean(reader),
        p95_reader_latency_ms=_percentile(reader, 0.95),
        mean_end_to_end_latency_ms=_mean(end_to_end),
        p95_end_to_end_latency_ms=_percentile(end_to_end, 0.95),
        total_retrieved_tokens=sum(retrieved_tokens),
        mean_retrieved_tokens=_mean(retrieved_tokens),
        total_reader_input_tokens=sum(reader_input),
        mean_reader_input_tokens=_mean(reader_input),
        total_reader_output_tokens=sum(reader_output),
        mean_reader_output_tokens=_mean(reader_output),
        framework_llm_input_tokens=framework_input,
        framework_llm_output_tokens=framework_output,
        framework_llm_tokens=framework_total,
        framework_usage_coverage=coverage,
        total_observed_tokens=observed_tokens,
        mean_memory_count=_mean(memory_counts),
        mean_memory_tokens=_mean(memory_tokens),
        memory_retention_ratio=_mean(retention_ratios),
        mean_storage_size_bytes=_mean(storage_sizes),
        mean_storage_size_mb=_mean(storage_sizes) / (1024.0 * 1024.0),
        storage_size_coverage=_mean([float(value) for value in storage_available]),
        storage_estimate_fraction=_mean([float(value) for value in storage_estimates]),
        update_operations=operation_counts["update"],
        merge_operations=operation_counts["merge"],
        archive_operations=operation_counts["archive"],
        observed_tokens_per_correct=(
            observed_tokens / correct if correct else None
        ),
        milliseconds_per_correct=(
            total_pipeline_ms / correct if correct else None
        ),
    )


def _table_row(
    summary: CostMethodSummary,
    *,
    reference: CostMethodSummary | None,
) -> dict[str, object]:
    row = {
        column: getattr(summary, column)
        for column in _TABLE_COLUMNS
        if hasattr(summary, column)
    }
    row["delta_end_to_end_latency_ms"] = (
        summary.mean_end_to_end_latency_ms - reference.mean_end_to_end_latency_ms
        if reference is not None
        else None
    )
    row["delta_reader_tokens"] = (
        (
            summary.mean_reader_input_tokens
            + summary.mean_reader_output_tokens
            - reference.mean_reader_input_tokens
            - reference.mean_reader_output_tokens
        )
        if reference is not None
        else None
    )
    row["delta_memory_count"] = (
        summary.mean_memory_count - reference.mean_memory_count
        if reference is not None
        else None
    )
    return row


def _write_table(
    base_path: Path,
    rows: list[dict[str, object]],
) -> dict[str, Path]:
    csv_path = base_path.with_suffix(".csv")
    markdown_path = base_path.with_suffix(".md")
    latex_path = base_path.with_suffix(".tex")
    formatted = _formatted_rows(rows)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(formatted)

    markdown_lines = [
        "| " + " | ".join(_TABLE_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in _TABLE_COLUMNS) + " |",
    ]
    markdown_lines.extend(
        "| " + " | ".join(str(row[column]) for column in _TABLE_COLUMNS) + " |"
        for row in formatted
    )
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    latex_lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{LongMemEval cost and efficiency results}",
        "\\label{tab:longmemeval-cost}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{" + "l" * len(_TABLE_COLUMNS) + "}",
        "\\toprule",
        " & ".join(_latex_escape(column) for column in _TABLE_COLUMNS) + " \\\\",
        "\\midrule",
    ]
    latex_lines.extend(
        " & ".join(_latex_escape(str(row[column])) for column in _TABLE_COLUMNS)
        + " \\\\"
        for row in formatted
    )
    latex_lines.extend(
        ["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}"]
    )
    latex_path.write_text("\n".join(latex_lines) + "\n", encoding="utf-8")
    return {
        "cost_csv": csv_path,
        "cost_markdown": markdown_path,
        "cost_latex": latex_path,
    }


def _formatted_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            column: (
                ""
                if row.get(column) is None
                else f"{row[column]:.4f}"
                if isinstance(row.get(column), float)
                else str(row.get(column, "")).replace("|", "\\|")
            )
            for column in _TABLE_COLUMNS
        }
        for row in rows
    ]


def _manifest_methods(manifest: dict[str, object]) -> list[str]:
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("retrieval manifest is missing config")
    methods = config.get("methods")
    if not isinstance(methods, list) or not all(
        isinstance(method, str) for method in methods
    ):
        raise ValueError("retrieval manifest config is missing methods")
    return list(dict.fromkeys(str(method) for method in methods))


def _reference_method(methods: list[str]) -> str | None:
    for candidate in ("vmp_tuned", "vmp_full", "vmp_rule"):
        if candidate in methods:
            return candidate
    return None


def _read_retrieval_records(path: Path) -> list[RetrievalSampleRecord]:
    return [
        RetrievalSampleRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_qa_records(path: Path) -> list[QASampleRecord]:
    if not path.exists():
        return []
    return [
        QASampleRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _validate_alignment(
    method: str,
    retrieval_records: list[RetrievalSampleRecord],
    qa_records: list[QASampleRecord],
) -> None:
    retrieval_ids = [record.question_id for record in retrieval_records]
    qa_ids = [record.question_id for record in qa_records]
    if len(retrieval_ids) != len(set(retrieval_ids)):
        raise ValueError(f"duplicate retrieval question IDs for {method}")
    if len(qa_ids) != len(set(qa_ids)):
        raise ValueError(f"duplicate QA question IDs for {method}")
    if retrieval_ids != qa_ids:
        raise ValueError(f"retrieval and QA question order differs for {method}")
    if any(record.method != method for record in qa_records):
        raise ValueError(f"QA record method mismatch for {method}")


def _is_locally_correct(record: QASampleRecord) -> int:
    metric = (
        "abstention_accuracy"
        if record.is_abstention
        else "normalized_exact_match"
    )
    return int(float(record.metrics.get(metric, 0.0)) >= 1.0)


def _adapter_number(record: RetrievalSampleRecord, name: str) -> float:
    value = record.adapter_stats.get(name, 0.0)
    return float(value) if isinstance(value, int | float) else 0.0


def _memory_retention_ratio(record: RetrievalSampleRecord) -> float | None:
    active = _adapter_number(record, "memory_count")
    sessions = _adapter_number(record, "ingestion_sessions")
    events = _adapter_number(record, "ingestion_events")
    denominator = sessions if sessions > 0 else events
    return active / denominator if denominator > 0 else None


def _operation_count(record: RetrievalSampleRecord, operation: str) -> int:
    payload = record.adapter_stats.get("policy_operation_counts")
    if not isinstance(payload, dict):
        return 0
    value = payload.get(operation, 0)
    return int(value) if isinstance(value, int | float) and value >= 0 else 0


def _framework_usage(
    record: RetrievalSampleRecord,
) -> tuple[int, int, int] | None:
    payload = record.adapter_stats.get("framework_llm_usage")
    if not isinstance(payload, dict):
        return None
    input_tokens = _recursive_token_value(
        payload,
        {"prompt_tokens", "input_tokens"},
    )
    output_tokens = _recursive_token_value(
        payload,
        {"completion_tokens", "output_tokens"},
    )
    total_tokens = _recursive_token_value(payload, {"total_tokens"})
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None
    input_value = input_tokens or 0
    output_value = output_tokens or 0
    resolved_total = total_tokens if total_tokens is not None else input_value + output_value
    return input_value, output_value, resolved_total


def _recursive_token_value(
    payload: object,
    names: set[str],
) -> int | None:
    if not isinstance(payload, Mapping):
        return None
    total = 0
    found = False
    for key, value in payload.items():
        if str(key).casefold() in names and isinstance(value, int | float):
            total += max(0, int(value))
            found = True
        elif isinstance(value, Mapping):
            nested = _recursive_token_value(value, names)
            if nested is not None:
                total += nested
                found = True
    return total if found else None


def _mean(values: list[int | float]) -> float:
    return sum(float(value) for value in values) / len(values) if values else 0.0


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
    }
    return "".join(replacements.get(char, char) for char in value)
