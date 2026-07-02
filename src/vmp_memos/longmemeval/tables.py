"""Export LongMemEval retrieval summaries as CSV, Markdown, and LaTeX."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from vmp_memos.evaluation import aggregate_retrieval_metrics
from vmp_memos.longmemeval.retrieval_runner import (
    RetrievalMethodSummary,
    RetrievalSampleRecord,
)

_OVERALL_COLUMNS = (
    "method",
    "evaluated_questions",
    "recall_all@5",
    "recall_all@10",
    "recall_any@5",
    "recall_any@10",
    "ndcg_any@5",
    "ndcg_any@10",
    "fractional_recall@5",
    "precision@5",
    "mrr",
    "standard_ndcg@5",
    "mean_retrieved_tokens",
    "mean_ingest_latency_ms",
    "mean_retrieval_latency_ms",
    "mean_memory_count",
    "mean_storage_size_bytes",
    "embedding_cache_hit_rate",
    "embedding_cache_misses",
)

_BY_TYPE_COLUMNS = (
    "method",
    "question_type",
    "evaluated_questions",
    "recall_all@5",
    "recall_all@10",
    "recall_any@5",
    "recall_any@10",
    "ndcg_any@5",
    "ndcg_any@10",
    "fractional_recall@5",
    "precision@5",
    "mrr",
    "standard_ndcg@5",
)


def export_retrieval_tables(
    retrieval_run: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Export overall and question-type retrieval tables in three formats."""

    run_dir = Path(retrieval_run).expanduser().resolve()
    manifest = _read_json_object(run_dir / "manifest.json")
    if manifest.get("status") != "completed":
        raise ValueError(f"Retrieval run is not completed: {run_dir}")
    methods = _manifest_methods(manifest)
    target = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else run_dir.parent.parent / "tables"
    )
    target.mkdir(parents=True, exist_ok=True)

    overall_rows = [_overall_row(run_dir, method) for method in methods]
    by_type_rows = [
        row
        for method in methods
        for row in _question_type_rows(run_dir, method)
    ]
    outputs: dict[str, Path] = {}
    outputs.update(
        _write_table_formats(
            target / "table1_retrieval_overall",
            columns=_OVERALL_COLUMNS,
            rows=overall_rows,
            caption="Overall LongMemEval Retrieval Results",
            label="tab:longmemeval-retrieval-overall",
        )
    )
    outputs.update(
        _write_table_formats(
            target / "table2_by_question_type",
            columns=_BY_TYPE_COLUMNS,
            rows=by_type_rows,
            caption="LongMemEval Retrieval Results by Question Type",
            label="tab:longmemeval-retrieval-by-type",
        )
    )
    return outputs


def _overall_row(run_dir: Path, method: str) -> dict[str, object]:
    payload = _read_json_object(run_dir / method / "summary.json")
    summary = RetrievalMethodSummary.model_validate(payload)
    row: dict[str, object] = {
        "method": method,
        "evaluated_questions": summary.evaluated_questions,
        "mean_retrieved_tokens": summary.mean_retrieved_tokens,
        "mean_ingest_latency_ms": summary.mean_ingest_latency_ms,
        "mean_retrieval_latency_ms": summary.mean_retrieval_latency_ms,
        "mean_memory_count": summary.mean_memory_count,
        "mean_storage_size_bytes": summary.mean_storage_size_bytes,
        "embedding_cache_hit_rate": summary.embedding_cache_hit_rate,
        "embedding_cache_misses": summary.embedding_cache_misses,
    }
    row.update(summary.metrics)
    return row


def _question_type_rows(run_dir: Path, method: str) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    path = run_dir / method / "retrieval.jsonl"
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = RetrievalSampleRecord.model_validate_json(line)
            if record.evaluation_skipped:
                continue
            grouped[record.question_type].append(
                {name: float(value) for name, value in record.metrics.items()}
            )
    rows: list[dict[str, object]] = []
    for question_type, metric_rows in sorted(grouped.items()):
        row: dict[str, object] = {
            "method": method,
            "question_type": question_type,
            "evaluated_questions": len(metric_rows),
        }
        row.update(aggregate_retrieval_metrics(metric_rows))
        rows.append(row)
    return rows


def _write_table_formats(
    base_path: Path,
    *,
    columns: tuple[str, ...],
    rows: list[dict[str, object]],
    caption: str,
    label: str,
) -> dict[str, Path]:
    csv_path = base_path.with_suffix(".csv")
    markdown_path = base_path.with_suffix(".md")
    latex_path = base_path.with_suffix(".tex")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_formatted_rows(columns, rows, markdown=False))

    markdown_lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    markdown_lines.extend(
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for row in _formatted_rows(columns, rows, markdown=True)
    )
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    latex_lines = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\toprule",
        " & ".join(_latex_escape(column) for column in columns) + " \\\\",
        "\\midrule",
    ]
    latex_lines.extend(
        " & ".join(_latex_escape(str(row[column])) for column in columns) + " \\\\"
        for row in _formatted_rows(columns, rows, markdown=True)
    )
    latex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    latex_path.write_text("\n".join(latex_lines) + "\n", encoding="utf-8")
    return {
        f"{base_path.name}_csv": csv_path,
        f"{base_path.name}_markdown": markdown_path,
        f"{base_path.name}_latex": latex_path,
    }


def _formatted_rows(
    columns: tuple[str, ...],
    rows: list[dict[str, object]],
    *,
    markdown: bool,
) -> list[dict[str, object]]:
    return [
        {
            column: _format_value(row.get(column, ""), markdown=markdown)
            for column in columns
        }
        for row in rows
    ]


def _format_value(value: object, *, markdown: bool) -> object:
    if isinstance(value, float):
        return f"{value:.4f}"
    if markdown:
        return str(value).replace("|", "\\|")
    return value


def _manifest_methods(manifest: dict[str, object]) -> list[str]:
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("Retrieval manifest is missing config")
    methods = config.get("methods")
    if not isinstance(methods, list) or not all(isinstance(method, str) for method in methods):
        raise ValueError("Retrieval manifest config is missing methods")
    return list(dict.fromkeys(methods))


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


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
