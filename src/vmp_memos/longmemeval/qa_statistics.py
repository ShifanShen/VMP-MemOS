"""Paper-ready tables and paired statistics for LongMemEval QA judge runs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from pathlib import Path

from pydantic import Field

from vmp_memos.longmemeval.official_qa import (
    OfficialJudgeMethodSummary,
    OfficialJudgeRecord,
)
from vmp_memos.schemas.base import NonEmptyStr, NonNegativeInt, SchemaModel, Score

_OVERALL_COLUMNS = (
    "method",
    "questions",
    "accuracy",
    "task_averaged_accuracy",
    "abstention_accuracy",
    "parse_fallback_rate",
    "mean_judge_input_tokens",
    "mean_judge_output_tokens",
    "mean_judge_latency_ms",
)
_BY_TYPE_COLUMNS = (
    "method",
    "question_type",
    "questions",
    "correct",
    "accuracy",
)
_PAIRWISE_COLUMNS = (
    "reference_method",
    "comparator_method",
    "questions",
    "reference_accuracy",
    "comparator_accuracy",
    "accuracy_delta",
    "ci_low",
    "ci_high",
    "reference_only_correct",
    "comparator_only_correct",
    "mcnemar_exact_p",
)


class LongMemEvalQAReportConfig(SchemaModel):
    """Deterministic settings for paper table and significance export."""

    judge_run: Path
    reference_method: NonEmptyStr
    output_dir: Path | None = None
    bootstrap_samples: int = Field(default=10_000, ge=100)
    confidence_level: Score = 0.95
    seed: NonNegativeInt = 42


class JudgePairwiseComparison(SchemaModel):
    """Paired binary comparison against one declared reference method."""

    reference_method: NonEmptyStr
    comparator_method: NonEmptyStr
    questions: NonNegativeInt
    reference_accuracy: Score
    comparator_accuracy: Score
    accuracy_delta: float
    ci_low: float
    ci_high: float
    reference_only_correct: NonNegativeInt
    comparator_only_correct: NonNegativeInt
    mcnemar_exact_p: Score
    bootstrap_samples: NonNegativeInt
    confidence_level: Score
    seed: NonNegativeInt


class LongMemEvalQAReportResult(SchemaModel):
    """Generated report payload and paths."""

    report_path: Path
    outputs: dict[str, Path]
    comparisons: list[JudgePairwiseComparison] = Field(default_factory=list)


def export_longmemeval_qa_report(
    config: LongMemEvalQAReportConfig,
) -> LongMemEvalQAReportResult:
    """Export official-prompt judge results with paired uncertainty estimates."""

    judge_run = config.judge_run.expanduser().resolve()
    manifest_path = judge_run / "manifest.json"
    manifest = _read_json_object(manifest_path)
    if manifest.get("status") != "completed":
        raise ValueError(f"Judge run is not completed: {judge_run}")
    signature = manifest.get("signature")
    methods = signature.get("methods") if isinstance(signature, dict) else None
    if not isinstance(methods, list) or not all(isinstance(item, str) for item in methods):
        raise ValueError("Judge manifest signature is missing methods")
    methods = list(dict.fromkeys(methods))
    if config.reference_method not in methods:
        raise ValueError("reference_method is absent from the judge run")

    summaries = {
        method: OfficialJudgeMethodSummary.model_validate(
            _read_json_object(judge_run / f"{method}.summary.json")
        )
        for method in methods
    }
    records = {
        method: _read_judge_records(judge_run / f"{method}.jsonl")
        for method in methods
    }
    _validate_paired_coverage(records)
    comparisons = [
        compare_official_judge_methods(
            config.reference_method,
            records[config.reference_method],
            method,
            records[method],
            bootstrap_samples=config.bootstrap_samples,
            confidence_level=float(config.confidence_level),
            seed=config.seed,
        )
        for method in methods
        if method != config.reference_method
    ]

    output_dir = (
        config.output_dir.expanduser().resolve()
        if config.output_dir is not None
        else judge_run / "paper"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    overall_rows = [
        {
            "method": method,
            "questions": summary.questions,
            "accuracy": summary.accuracy,
            "task_averaged_accuracy": summary.task_averaged_accuracy,
            "abstention_accuracy": summary.abstention_accuracy,
            "parse_fallback_rate": summary.parse_fallback_rate,
            "mean_judge_input_tokens": summary.mean_judge_input_tokens,
            "mean_judge_output_tokens": summary.mean_judge_output_tokens,
            "mean_judge_latency_ms": summary.mean_judge_latency_ms,
        }
        for method, summary in summaries.items()
    ]
    by_type_rows = [
        {
            "method": method,
            "question_type": question_type,
            "questions": type_summary.questions,
            "correct": type_summary.correct,
            "accuracy": type_summary.accuracy,
        }
        for method, summary in summaries.items()
        for question_type, type_summary in summary.by_question_type.items()
    ]
    pairwise_rows = [comparison.model_dump(mode="json") for comparison in comparisons]
    outputs: dict[str, Path] = {}
    outputs.update(
        _write_table_formats(
            output_dir / "table3_qa_official_prompt_overall",
            columns=_OVERALL_COLUMNS,
            rows=overall_rows,
            caption="LongMemEval QA Results with the Shared Official-Prompt Judge",
            label="tab:longmemeval-qa-overall",
        )
    )
    outputs.update(
        _write_table_formats(
            output_dir / "table4_qa_official_prompt_by_type",
            columns=_BY_TYPE_COLUMNS,
            rows=by_type_rows,
            caption="LongMemEval QA Accuracy by Question Type",
            label="tab:longmemeval-qa-by-type",
        )
    )
    outputs.update(
        _write_table_formats(
            output_dir / "table5_qa_paired_significance",
            columns=_PAIRWISE_COLUMNS,
            rows=pairwise_rows,
            caption="Paired LongMemEval QA Comparisons",
            label="tab:longmemeval-qa-significance",
        )
    )

    report_path = output_dir / "qa_paper_report.json"
    _write_json(
        report_path,
        {
            "schema_version": "1.0",
            "judge_run": str(judge_run),
            "judge_manifest_sha256": _sha256(manifest_path),
            "score_kind": signature.get("score_kind") if isinstance(signature, dict) else None,
            "directly_comparable_to_published_gpt4o_scores": (
                signature.get("directly_comparable_to_published_gpt4o_scores")
                if isinstance(signature, dict)
                else False
            ),
            "reference_method": config.reference_method,
            "bootstrap": {
                "samples": config.bootstrap_samples,
                "confidence_level": config.confidence_level,
                "seed": config.seed,
            },
            "summaries": {
                method: summary.model_dump(mode="json")
                for method, summary in summaries.items()
            },
            "comparisons": [
                comparison.model_dump(mode="json") for comparison in comparisons
            ],
            "outputs": {name: str(path) for name, path in outputs.items()},
        },
    )
    outputs["qa_paper_report_json"] = report_path
    return LongMemEvalQAReportResult(
        report_path=report_path,
        outputs=outputs,
        comparisons=comparisons,
    )


def compare_official_judge_methods(
    reference_method: str,
    reference_records: list[OfficialJudgeRecord],
    comparator_method: str,
    comparator_records: list[OfficialJudgeRecord],
    *,
    bootstrap_samples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> JudgePairwiseComparison:
    """Compute a paired bootstrap CI and exact McNemar test on binary labels."""

    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    reference = {record.question_id: record for record in reference_records}
    comparator = {record.question_id: record for record in comparator_records}
    if list(reference) != list(comparator):
        raise ValueError("Paired judge records must have identical ordered question IDs")
    question_ids = list(reference)
    if not question_ids:
        raise ValueError("Paired judge records cannot be empty")
    deltas = [
        float(reference[question_id].autoeval_label.label)
        - float(comparator[question_id].autoeval_label.label)
        for question_id in question_ids
    ]
    reference_only = sum(delta > 0 for delta in deltas)
    comparator_only = sum(delta < 0 for delta in deltas)
    rng = random.Random(seed)
    sample_count = len(deltas)
    bootstrap = sorted(
        sum(deltas[rng.randrange(sample_count)] for _ in range(sample_count))
        / sample_count
        for _ in range(bootstrap_samples)
    )
    alpha = 1.0 - confidence_level
    reference_accuracy = _mean(
        [float(record.autoeval_label.label) for record in reference_records]
    )
    comparator_accuracy = _mean(
        [float(record.autoeval_label.label) for record in comparator_records]
    )
    return JudgePairwiseComparison(
        reference_method=reference_method,
        comparator_method=comparator_method,
        questions=sample_count,
        reference_accuracy=reference_accuracy,
        comparator_accuracy=comparator_accuracy,
        accuracy_delta=reference_accuracy - comparator_accuracy,
        ci_low=_quantile(bootstrap, alpha / 2.0),
        ci_high=_quantile(bootstrap, 1.0 - alpha / 2.0),
        reference_only_correct=reference_only,
        comparator_only_correct=comparator_only,
        mcnemar_exact_p=_mcnemar_exact_p(reference_only, comparator_only),
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed,
    )


def _mcnemar_exact_p(reference_only: int, comparator_only: int) -> float:
    discordant = reference_only + comparator_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(reference_only, comparator_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a quantile of an empty sequence")
    position = probability * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _validate_paired_coverage(
    records: dict[str, list[OfficialJudgeRecord]],
) -> None:
    expected: list[str] | None = None
    for method, method_records in records.items():
        ids = [record.question_id for record in method_records]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate judge question IDs for method: {method}")
        if expected is None:
            expected = ids
        elif ids != expected:
            raise ValueError("Judge methods do not share identical ordered coverage")


def _read_judge_records(path: Path) -> list[OfficialJudgeRecord]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [
        OfficialJudgeRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
    formatted = _formatted_rows(columns, rows)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(formatted)
    markdown_lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    markdown_lines.extend(
        "| " + " | ".join(str(row[column]).replace("|", "\\|") for column in columns)
        + " |"
        for row in formatted
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
        for row in formatted
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
) -> list[dict[str, object]]:
    return [
        {
            column: f"{value:.4f}" if isinstance(value, float) else value
            for column in columns
            for value in [row.get(column, "")]
        }
        for row in rows
    ]


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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
