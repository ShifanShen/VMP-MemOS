"""Tests for Phase 10 feature ablation runner."""

from pathlib import Path

from vmp_memos.benchmark import AblationRunConfig, load_benchmark_samples, run_ablation
from vmp_memos.benchmark.baselines import VMPRuleBaseline

DATASET_PATH = Path("data/benchmarks/memory_policy_toy.jsonl")


def test_vmp_rule_baseline_masks_disabled_features() -> None:
    sample = load_benchmark_samples(DATASET_PATH)[0]

    output = VMPRuleBaseline(
        disabled_features=["recency", "contradiction"],
        system_name="vmp_rule__mask_test",
    ).run(sample)

    assert output.system_name == "vmp_rule__mask_test"
    assert output.metadata["disabled_features"] == ["recency", "contradiction"]


def test_ablation_runner_writes_report_and_results(tmp_path) -> None:
    summary = run_ablation(
        AblationRunConfig(
            dataset_path=DATASET_PATH,
            output_dir=tmp_path / "runs",
            report_path=tmp_path / "reports" / "ablation.md",
            baseline_names=["no_memory", "vector_rag", "vmp_rule"],
            disabled_features=["recency", "contradiction"],
            run_id="unit_ablation",
        )
    )

    assert summary.run_id == "unit_ablation"
    assert summary.num_samples == 8
    assert summary.result_path == tmp_path / "runs" / "unit_ablation" / "results.jsonl"
    assert summary.report_path == tmp_path / "reports" / "ablation.md"
    assert summary.result_path.is_file()
    assert summary.report_path.is_file()
    assert "vmp_rule__no_recency" in summary.systems
    assert "vmp_rule__no_contradiction" in summary.systems

    rows = summary.result_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 40

    report = summary.report_path.read_text(encoding="utf-8")
    assert "Experiment settings" in report
    assert "Baseline comparison" in report
    assert "Ablation comparison" in report
    assert "Error cases" in report
    assert "Memory operation examples" in report
