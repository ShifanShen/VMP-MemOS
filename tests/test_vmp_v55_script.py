"""Static contract tests for the staged VMP-v5.5 paper experiment."""

from pathlib import Path

SCRIPT_PATH = Path("scripts/run_vmp_v55_experiment.sh")
V551_SCRIPT_PATH = Path("scripts/run_vmp_v551_experiment.sh")
V552_SCRIPT_PATH = Path("scripts/run_vmp_v552_experiment.sh")
V6_SCRIPT_PATH = Path("scripts/run_vmp_v6_experiment.sh")
V61_SCRIPT_PATH = Path("scripts/run_vmp_v61_experiment.sh")
V62_SCRIPT_PATH = Path("scripts/run_vmp_v62_experiment.sh")
V63_SCRIPT_PATH = Path("scripts/run_vmp_v63_experiment.sh")


def test_v55_reuses_frozen_dev_candidates_and_keeps_outcome_gates() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert (
        'DEV_CANDIDATE_RUN_ID="${DEV_CANDIDATE_RUN_ID:-lme_dev_vmp_v532_candidates_seed42}"'
    ) in script
    assert 'MIN_DEV_RECALL_ALL_5="${MIN_DEV_RECALL_ALL_5:-0.93}"' in script
    assert 'MIN_DEV_DELTA_VS_SHARED_V43="${MIN_DEV_DELTA_VS_SHARED_V43:-0.03}"' in script
    assert 'MAX_REGRESSED_QUESTIONS="${MAX_REGRESSED_QUESTIONS:-0}"' in script


def test_v55_uses_audited_dual_view_planner_and_fixed_ten_candidate_prompt() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'CANDIDATE_COUNT="${CANDIDATE_COUNT:-10}"' in script
    assert (
        'CANDIDATE_PLANNER_VERSION="${CANDIDATE_PLANNER_VERSION:-vmp_v55_dual_view_rrf_v1}"'
    ) in script
    assert "--candidate-planner-version" in script
    assert "--expected-candidate-planner-version" in script
    assert 'SELECTOR_PROMPT_VERSION="${SELECTOR_PROMPT_VERSION:-' in script
    assert "vmp_v55_challenger_span_selector_v1" in script


def test_v551_uses_distinct_prompt_and_artifact_names() -> None:
    script = V551_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "vmp_v551_complete_challenger_selector_v1" in script
    assert "lme_dev_vmp_v551_rerank_seed42" in script
    assert "vmp_v551_seed42_dev_pass.json" in script
    assert "run_vmp_v55_experiment.sh" in script


def test_v552_uses_anonymous_pairwise_protocol_and_role_aware_excerpt() -> None:
    script = V552_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "vmp_v552_anonymous_pairwise_selector_v1" in script
    assert "vmp_v552_integrated_pairwise_boundary_v1" in script
    assert "role_aware_fact_v2" in script
    assert "lme_dev_vmp_v552_rerank_seed42" in script
    assert "run_vmp_v55_experiment.sh" in script


def test_v6_uses_atomic_fact_extraction_and_deterministic_coverage() -> None:
    script = V6_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "vmp_v6_anonymous_atomic_fact_extractor_v1" in script
    assert "vmp_v6_deterministic_set_coverage_v1" in script
    assert "role_aware_fact_v2" in script
    assert "lme_dev_vmp_v6_rerank_seed42" in script
    assert "COVERAGE_DIVERSITY_WEIGHT" in script
    assert "run_vmp_v55_experiment.sh" in script


def test_v61_loads_frozen_dev_tuned_weights_into_a_new_run() -> None:
    script = V61_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "vmp_v6_coverage_seed42.json" in script
    assert "COVERAGE_MIN_GAIN" in script
    assert "COVERAGE_TEMPORAL_WEIGHT" in script
    assert "lme_dev_vmp_v61_rerank_seed42" in script
    assert "run_vmp_v6_experiment.sh" in script
    assert 'best.get("gate_feasible") is not True' in script


def test_v62_uses_partial_fact_prompt_v3_excerpt_and_distinct_artifacts() -> None:
    script = V62_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "vmp_v62_partial_atomic_fact_extractor_v2" in script
    assert "vmp_v62_deterministic_set_coverage_v2" in script
    assert "role_aware_fact_v3" in script
    assert "lme_dev_vmp_v62_rerank_seed42" in script
    assert "vmp_v62_seed42_dev_pass.json" in script
    assert "run_vmp_v6_experiment.sh" in script
    assert "af082822,1a8a66a6,0bc8ad92" in script
    assert 'STAGE:-}" == "dev_smoke"' in script


def test_v63_versions_grounding_guards_without_changing_coverage_weights() -> None:
    script = V63_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "vmp_v63_grounded_atomic_fact_extractor_v3" in script
    assert "vmp_v63_deterministic_set_coverage_v3" in script
    assert "role_aware_fact_v4" in script
    assert "lme_dev_vmp_v63_rerank_seed42" in script
    assert "vmp_v63_seed42_dev_pass.json" in script
    assert 'COVERAGE_MIN_GAIN="${COVERAGE_MIN_GAIN:-0.25}"' in script
    assert 'COVERAGE_DIVERSITY_WEIGHT="${COVERAGE_DIVERSITY_WEIGHT:-1.25}"' in script
    assert "run_vmp_v6_experiment.sh" in script
    assert "af082822,1a8a66a6,0bc8ad92" in script
