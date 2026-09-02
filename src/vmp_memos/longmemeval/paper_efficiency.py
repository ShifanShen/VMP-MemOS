"""Join official-prompt correctness with end-to-end resource measurements."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from pydantic import Field

from vmp_memos.longmemeval.cost import analyze_longmemeval_cost
from vmp_memos.longmemeval.official_qa import OfficialJudgeMethodSummary
from vmp_memos.schemas.base import (
    NonEmptyStr,
    NonNegativeFloat,
    NonNegativeInt,
    SchemaModel,
    Score,
)

_COLUMNS = (
    "method",
    "questions",
    "official_correct",
    "official_accuracy",
    "task_averaged_accuracy",
    "mean_end_to_end_latency_ms",
    "p95_end_to_end_latency_ms",
    "mean_retrieved_tokens",
    "mean_reader_input_tokens",
    "mean_reader_output_tokens",
    "reranker_tokens",
    "reranker_usage_coverage",
    "framework_llm_tokens",
    "framework_usage_coverage",
    "mean_memory_count",
    "mean_storage_size_mb",
    "storage_size_coverage",
    "observed_tokens_per_official_correct",
    "milliseconds_per_official_correct",
    "token_accounting_complete",
)


class OfficialJudgeEfficiencyRow(SchemaModel):
    """One method's official correctness and measured resource trade-off."""

    method: NonEmptyStr
    questions: NonNegativeInt
    official_correct: NonNegativeInt
    official_accuracy: Score
    task_averaged_accuracy: Score
    mean_end_to_end_latency_ms: NonNegativeFloat
    p95_end_to_end_latency_ms: NonNegativeFloat
    mean_retrieved_tokens: NonNegativeFloat
    mean_reader_input_tokens: NonNegativeFloat
    mean_reader_output_tokens: NonNegativeFloat
    reranker_tokens: NonNegativeInt
    reranker_usage_coverage: Score
    framework_llm_tokens: NonNegativeInt | None = None
    framework_usage_coverage: Score
    mean_memory_count: NonNegativeFloat
    mean_storage_size_mb: NonNegativeFloat
    storage_size_coverage: Score
    observed_tokens_per_official_correct: NonNegativeFloat | None = None
    milliseconds_per_official_correct: NonNegativeFloat | None = None
    token_accounting_complete: bool


class OfficialJudgeEfficiencyReport(SchemaModel):
    """Paper efficiency rows with explicit accounting limitations."""

    schema_version: NonEmptyStr = "1.0"
    retrieval_run: NonEmptyStr
    qa_subdir: NonEmptyStr
    judge_subdir: NonEmptyStr
    reference_method: NonEmptyStr
    methods: list[OfficialJudgeEfficiencyRow] = Field(default_factory=list)
    notes: dict[str, str] = Field(default_factory=dict)


def export_official_judge_efficiency(
    retrieval_run: str | Path,
    *,
    qa_subdir: str,
    judge_subdir: str,
    reference_method: str,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Export Table 6 using official labels rather than lexical QA metrics."""

    run = Path(retrieval_run).expanduser().resolve()
    judge_dir = run / qa_subdir / judge_subdir
    manifest = _read_object(judge_dir / "manifest.json")
    if manifest.get("status") != "completed":
        raise ValueError(f"Official judge run is not completed: {judge_dir}")
    signature = manifest.get("signature")
    methods = signature.get("methods") if isinstance(signature, dict) else None
    if not isinstance(methods, list) or not all(isinstance(item, str) for item in methods):
        raise ValueError("Official judge manifest is missing methods")
    if reference_method not in methods:
        raise ValueError(f"Reference method is absent from judge run: {reference_method}")

    cost = analyze_longmemeval_cost(
        run,
        qa_subdir=qa_subdir,
        reference_method=reference_method,
    )
    if set(methods) != set(cost.methods):
        raise ValueError("Cost and official-judge methods differ")
    rows: list[OfficialJudgeEfficiencyRow] = []
    for method in methods:
        judged = OfficialJudgeMethodSummary.model_validate(
            _read_object(judge_dir / f"{method}.summary.json")
        )
        measured = cost.methods[method]
        if judged.questions != measured.qa_samples:
            raise ValueError(f"Judge and cost question counts differ: {method}")
        correct = judged.correct
        rows.append(
            OfficialJudgeEfficiencyRow(
                method=method,
                questions=judged.questions,
                official_correct=correct,
                official_accuracy=judged.accuracy,
                task_averaged_accuracy=judged.task_averaged_accuracy,
                mean_end_to_end_latency_ms=measured.mean_end_to_end_latency_ms,
                p95_end_to_end_latency_ms=measured.p95_end_to_end_latency_ms,
                mean_retrieved_tokens=measured.mean_retrieved_tokens,
                mean_reader_input_tokens=measured.mean_reader_input_tokens,
                mean_reader_output_tokens=measured.mean_reader_output_tokens,
                reranker_tokens=measured.reranker_tokens,
                reranker_usage_coverage=measured.reranker_usage_coverage,
                framework_llm_tokens=measured.framework_llm_tokens,
                framework_usage_coverage=measured.framework_usage_coverage,
                mean_memory_count=measured.mean_memory_count,
                mean_storage_size_mb=measured.mean_storage_size_mb,
                storage_size_coverage=measured.storage_size_coverage,
                observed_tokens_per_official_correct=(
                    measured.total_observed_tokens / correct if correct else None
                ),
                milliseconds_per_official_correct=(
                    measured.mean_end_to_end_latency_ms * measured.qa_samples / correct
                    if correct
                    else None
                ),
                token_accounting_complete=(
                    measured.reranker_usage_coverage == 1.0
                    and measured.framework_usage_coverage == 1.0
                ),
            )
        )
    report = OfficialJudgeEfficiencyReport(
        retrieval_run=str(run),
        qa_subdir=qa_subdir,
        judge_subdir=judge_subdir,
        reference_method=reference_method,
        methods=rows,
        notes={
            "correctness": "LongMemEval official-prompt local-vLLM binary judge.",
            "token_accounting": (
                "Observed tokens include reranker, reader, and available native "
                "framework LLM usage; incomplete native usage is never imputed as zero."
            ),
            "latency": (
                "End-to-end latency is ingestion + retrieval/reranking + reader; "
                "judge latency is excluded."
            ),
        },
    )
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "official_judge_efficiency.json"
    csv_path = target / "table6_official_judge_efficiency.csv"
    markdown_path = target / "table6_official_judge_efficiency.md"
    latex_path = target / "table6_official_judge_efficiency.tex"
    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    _write_csv(csv_path, rows)
    _write_markdown(markdown_path, rows)
    _write_latex(latex_path, rows)
    return {
        "official_judge_efficiency_json": json_path,
        "table6_official_judge_efficiency_csv": csv_path,
        "table6_official_judge_efficiency_markdown": markdown_path,
        "table6_official_judge_efficiency_latex": latex_path,
    }


def _write_csv(path: Path, rows: list[OfficialJudgeEfficiencyRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(_rows(rows, markdown=False))


def _write_markdown(path: Path, rows: list[OfficialJudgeEfficiencyRow]) -> None:
    lines = [
        "| " + " | ".join(_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in _COLUMNS) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(row[column]) for column in _COLUMNS) + " |"
        for row in _rows(rows, markdown=True)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex(path: Path, rows: list[OfficialJudgeEfficiencyRow]) -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{LongMemEval official-judge accuracy and efficiency}",
        "\\label{tab:longmemeval-official-efficiency}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{" + "l" * len(_COLUMNS) + "}",
        "\\toprule",
        " & ".join(_latex_escape(column) for column in _COLUMNS) + " \\\\",
        "\\midrule",
    ]
    lines.extend(
        " & ".join(_latex_escape(str(row[column])) for column in _COLUMNS)
        + " \\\\"
        for row in _rows(rows, markdown=True)
    )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rows(
    rows: list[OfficialJudgeEfficiencyRow],
    *,
    markdown: bool,
) -> list[dict[str, object]]:
    return [
        {
            column: _format_value(
                row.model_dump(mode="python").get(column),
                markdown=markdown,
            )
            for column in _COLUMNS
        }
        for row in rows
    ]


def _format_value(value: object, *, markdown: bool) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    if markdown:
        return str(value).replace("|", "\\|")
    return value


def _read_object(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
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
