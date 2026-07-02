"""Paper-table export for frozen-model LongMemEval ablations."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from vmp_memos.frameworks.vmp_tuned import VMP_TUNED_ABLATIONS
from vmp_memos.longmemeval.qa_runner import QAMethodSummary
from vmp_memos.longmemeval.retrieval_runner import RetrievalMethodSummary

_COLUMNS = (
    "method",
    "ablation_type",
    "disabled_component",
    "recall_all@5",
    "delta_recall_all@5",
    "mrr",
    "delta_mrr",
    "ndcg_any@5",
    "delta_ndcg_any@5",
    "mean_retrieved_tokens",
    "delta_mean_retrieved_tokens",
    "normalized_exact_match",
    "delta_normalized_exact_match",
    "token_f1",
    "delta_token_f1",
)


def export_longmemeval_ablation_table(
    retrieval_run: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Export retrieval and optional QA deltas against frozen VMP-full."""

    run_dir = Path(retrieval_run).expanduser().resolve()
    manifest = _read_json(run_dir / "manifest.json")
    if manifest.get("status") != "completed":
        raise ValueError(f"retrieval run is not completed: {run_dir}")
    split = manifest.get("split")
    if not isinstance(split, dict) or split.get("name") != "test":
        raise ValueError("LongMemEval ablation tables must come from the test split")
    if not isinstance(manifest.get("vmp_tuned_model"), dict):
        raise ValueError("ablation run is missing frozen VMP-Tuned model provenance")

    methods = _ablation_methods(manifest)
    reference_method = "vmp_tuned" if "vmp_tuned" in methods else "vmp_full"
    if reference_method not in methods:
        raise ValueError("ablation run must contain vmp_tuned or vmp_full")
    expected = set(VMP_TUNED_ABLATIONS)
    missing = sorted(expected - set(methods))
    if missing:
        raise ValueError(f"ablation run is missing methods: {', '.join(missing)}")

    retrieval = {
        method: _load_retrieval_summary(run_dir, method)
        for method in methods
    }
    qa = _load_qa_summaries(run_dir, methods)
    rows = [
        _row(
            method,
            retrieval=summary,
            reference=retrieval[reference_method],
            qa=qa.get(method),
            qa_reference=qa.get(reference_method),
        )
        for method, summary in retrieval.items()
    ]
    target = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else run_dir.parent.parent / "tables"
    )
    target.mkdir(parents=True, exist_ok=True)
    base_path = target / "table4_ablation"
    return _write_formats(base_path, rows)


def _row(
    method: str,
    *,
    retrieval: RetrievalMethodSummary,
    reference: RetrievalMethodSummary,
    qa: QAMethodSummary | None,
    qa_reference: QAMethodSummary | None,
) -> dict[str, object]:
    target_type, target_name = VMP_TUNED_ABLATIONS.get(
        method,
        ("full", "none"),
    )
    row: dict[str, object] = {
        "method": "VMP-full" if target_type == "full" else method,
        "ablation_type": target_type,
        "disabled_component": target_name,
        "recall_all@5": _metric(retrieval.metrics, "recall_all@5"),
        "mrr": _metric(retrieval.metrics, "mrr"),
        "ndcg_any@5": _metric(retrieval.metrics, "ndcg_any@5"),
        "mean_retrieved_tokens": retrieval.mean_retrieved_tokens,
    }
    for metric in ("recall_all@5", "mrr", "ndcg_any@5"):
        row[f"delta_{metric}"] = _metric(retrieval.metrics, metric) - _metric(
            reference.metrics,
            metric,
        )
    row["delta_mean_retrieved_tokens"] = (
        retrieval.mean_retrieved_tokens - reference.mean_retrieved_tokens
    )
    for metric in ("normalized_exact_match", "token_f1"):
        value = _metric(qa.metrics, metric) if qa is not None else None
        reference_value = (
            _metric(qa_reference.metrics, metric)
            if qa_reference is not None
            else None
        )
        row[metric] = value
        row[f"delta_{metric}"] = (
            value - reference_value
            if value is not None and reference_value is not None
            else None
        )
    return row


def _ablation_methods(manifest: dict[str, object]) -> list[str]:
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("retrieval manifest is missing config")
    methods = config.get("methods")
    if not isinstance(methods, list):
        raise ValueError("retrieval manifest config is missing methods")
    return [
        method
        for method in methods
        if isinstance(method, str)
        and (method in {"vmp_tuned", "vmp_full"} or method in VMP_TUNED_ABLATIONS)
    ]


def _load_retrieval_summary(
    run_dir: Path,
    method: str,
) -> RetrievalMethodSummary:
    return RetrievalMethodSummary.model_validate(
        _read_json(run_dir / method / "summary.json")
    )


def _load_qa_summaries(
    run_dir: Path,
    methods: list[str],
) -> dict[str, QAMethodSummary]:
    manifest_path = run_dir / "qa" / "manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "completed":
        return {}
    summaries: dict[str, QAMethodSummary] = {}
    for method in methods:
        path = run_dir / "qa" / f"{method}.summary.json"
        if path.exists():
            summaries[method] = QAMethodSummary.model_validate(_read_json(path))
    return summaries


def _write_formats(
    base_path: Path,
    rows: list[dict[str, object]],
) -> dict[str, Path]:
    csv_path = base_path.with_suffix(".csv")
    markdown_path = base_path.with_suffix(".md")
    latex_path = base_path.with_suffix(".tex")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(_formatted_rows(rows))

    markdown_lines = [
        "| " + " | ".join(_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in _COLUMNS) + " |",
    ]
    markdown_lines.extend(
        "| " + " | ".join(str(row[column]) for column in _COLUMNS) + " |"
        for row in _formatted_rows(rows)
    )
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    latex_lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{VMP component ablation on the LongMemEval test split}",
        "\\label{tab:vmp-ablation}",
        "\\begin{tabular}{" + "l" * len(_COLUMNS) + "}",
        "\\toprule",
        " & ".join(_latex_escape(column) for column in _COLUMNS) + " \\\\",
        "\\midrule",
    ]
    latex_lines.extend(
        " & ".join(_latex_escape(str(row[column])) for column in _COLUMNS) + " \\\\"
        for row in _formatted_rows(rows)
    )
    latex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    latex_path.write_text("\n".join(latex_lines) + "\n", encoding="utf-8")
    return {
        "ablation_csv": csv_path,
        "ablation_markdown": markdown_path,
        "ablation_latex": latex_path,
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
            for column in _COLUMNS
        }
        for row in rows
    ]


def _metric(metrics: dict[str, float], name: str) -> float:
    return float(metrics.get(name, 0.0))


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
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
