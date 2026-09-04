"""Static contracts for the official-framework paper pipeline."""

from pathlib import Path

import yaml

SCRIPT_PATH = Path("scripts/run_official_framework_paper_experiment.sh")
CONFIG_PATH = Path("configs/official_frameworks.yaml")


def test_official_pipeline_is_staged_and_resumable() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    for stage in (
        "smoke)",
        "audit)",
        "dev_protocol)",
        "test_candidates)",
        "test_rerank)",
        "test_qa)",
        "test_judge)",
        "status)",
    ):
        assert stage in script
    assert "--require-main-table-eligible" in script
    assert "--resume" in script
    assert 'if [[ "${RETRIEVAL_RESUME}" == "1" && -d "${CANDIDATE_RUN}" ]]' in script
    assert "Each completed question is durably checkpointed" in script
    assert "audit_mem0_protocol.py" in script
    assert "lme_test_${FRAMEWORK}_official_v2_candidates_seed42" in script


def test_official_pipeline_freezes_the_shared_v64_protocol() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "vmp_v55_dual_view_rrf_v1" in script
    assert "vmp_v64_high_recall_atomic_fact_extractor_v4" in script
    assert "vmp_v64_deterministic_set_coverage_v4" in script
    assert "role_aware_fact_v4" in script
    assert "longmemeval_hybrid_evidence_reader_v21" in script
    assert "reranker_facts_with_query_windows" in script
    assert "--protocol-selection-split dev" in script
    assert "run_longmemeval_official_judge.py" in script


def test_official_pipeline_never_passes_vmp_models_or_test_training_override() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "--vmp-tuned-model" not in script
    assert "--vmp-hierarchical-model" not in script
    assert "--allow-dev-model-selection" not in script
    assert "--split test" in script


def test_official_config_matches_pinned_optional_dependencies() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["frameworks"]["mem0"]["version"] == "2.0.10"
    assert config["frameworks"]["langmem"]["version"] == "0.0.30"
    assert config["frameworks"]["graphiti"]["version"] == "0.29.2"
    assert config["frameworks"]["letta"]["client_version"] == "1.12.1"
    assert config["frameworks"]["letta"]["server_version"] == "0.16.8"
    assert config["shared_models"]["embedding"] == "BAAI/bge-m3"
    assert config["shared_models"]["memory_max_tokens"] == 2048
    assert config["shared_models"]["memory_retry_max_tokens"] == 4096
    assert config["shared_models"]["context_window"] == 32768
    assert config["mem0_protocol_gate"]["dev_samples"] == 20
    assert config["mem0_protocol_gate"]["max_unrecovered_failure_rate"] == 0.0
    assert config["mem0_protocol_gate"]["require_native_bm25"] is True
    assert config["test_labels_visible_to_memory_or_reader"] is False
