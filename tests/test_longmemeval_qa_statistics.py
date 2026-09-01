"""Tests for deterministic paper QA tables and paired statistics."""

from __future__ import annotations

import json
from pathlib import Path

from vmp_memos.longmemeval.official_qa import (
    OfficialAutoEvalLabel,
    OfficialJudgeRecord,
    summarize_official_judge_method,
)
from vmp_memos.longmemeval.qa_statistics import (
    LongMemEvalQAReportConfig,
    compare_official_judge_methods,
    export_longmemeval_qa_report,
)

PAPER_SCRIPT_PATH = Path("scripts/run_vmp_v64_paper_qa_eval.sh")


def test_paired_comparison_reports_delta_interval_and_exact_test() -> None:
    reference = [
        _record("reference", "q1", True),
        _record("reference", "q2", True),
        _record("reference", "q3", False),
        _record("reference", "q4_abs", True),
    ]
    comparator = [
        _record("comparator", "q1", False),
        _record("comparator", "q2", True),
        _record("comparator", "q3", True),
        _record("comparator", "q4_abs", False),
    ]

    comparison = compare_official_judge_methods(
        "reference",
        reference,
        "comparator",
        comparator,
        bootstrap_samples=500,
        seed=7,
    )

    assert comparison.reference_accuracy == 0.75
    assert comparison.comparator_accuracy == 0.5
    assert comparison.accuracy_delta == 0.25
    assert comparison.reference_only_correct == 2
    assert comparison.comparator_only_correct == 1
    assert comparison.ci_low <= comparison.accuracy_delta <= comparison.ci_high
    assert comparison.mcnemar_exact_p == 1.0


def test_report_export_writes_json_csv_markdown_and_latex(tmp_path) -> None:
    judge_run = tmp_path / "judge"
    judge_run.mkdir()
    methods = ["reference", "comparator"]
    (judge_run / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "signature": {
                    "methods": methods,
                    "score_kind": "official_prompt_local_vllm_judge",
                    "directly_comparable_to_published_gpt4o_scores": False,
                },
            }
        ),
        encoding="utf-8",
    )
    records = {
        "reference": [
            _record("reference", "q1", True),
            _record("reference", "q2_abs", True),
        ],
        "comparator": [
            _record("comparator", "q1", False),
            _record("comparator", "q2_abs", True),
        ],
    }
    for method in methods:
        (judge_run / f"{method}.jsonl").write_text(
            "".join(record.model_dump_json() + "\n" for record in records[method]),
            encoding="utf-8",
        )
        summary = summarize_official_judge_method(method, records[method])
        (judge_run / f"{method}.summary.json").write_text(
            summary.model_dump_json(),
            encoding="utf-8",
        )

    result = export_longmemeval_qa_report(
        LongMemEvalQAReportConfig(
            judge_run=judge_run,
            reference_method="reference",
            bootstrap_samples=500,
        )
    )

    assert len(result.outputs) == 10
    assert all(path.exists() for path in result.outputs.values())
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["reference_method"] == "reference"
    assert report["directly_comparable_to_published_gpt4o_scores"] is False
    assert report["comparisons"][0]["accuracy_delta"] == 0.5
    markdown = result.outputs[
        "table3_qa_official_prompt_overall_markdown"
    ].read_text(encoding="utf-8")
    assert "task_averaged_accuracy" in markdown
    assert "reference" in markdown


def test_paper_script_freezes_shared_judge_and_explicit_score_provenance() -> None:
    script = PAPER_SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'STAGE="${STAGE:-judge_smoke}"' in script
    assert "run_longmemeval_official_judge.py" in script
    assert "export_longmemeval_qa_report.py" in script
    assert 'REFERENCE_METHOD="${REFERENCE_METHOD:-vmp_hierarchical__vllm_boundary}"' in script
    assert "Scores are official-prompt compatible, not published GPT-4o judge scores." in script
    assert 'JUDGE_RESUME="${JUDGE_RESUME:-1}"' in script


def _record(method: str, question_id: str, label: bool) -> OfficialJudgeRecord:
    return OfficialJudgeRecord(
        question_id=question_id,
        question_type=("multi-session" if question_id.endswith("_abs") else "knowledge-update"),
        method=method,
        hypothesis="answer",
        autoeval_label=OfficialAutoEvalLabel(model="local-judge", label=label),
        judge_provider="fake-vllm",
        judge_model="local-judge",
        judge_response="yes" if label else "no",
        parse_status="yes" if label else "no",
        prompt_sha256=f"hash-{method}-{question_id}",
    )
