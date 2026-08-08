#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${STAGE:-}" == "dev_smoke" ]]; then
  export CANDIDATE_METHODS="${CANDIDATE_METHODS:-vmp_hierarchical}"
  export RERANK_QUESTION_IDS="${RERANK_QUESTION_IDS:-af082822,1a8a66a6,0bc8ad92}"
  export DEV_RERANK_RUN_ID="${DEV_RERANK_RUN_ID:-lme_dev_vmp_v62_smoke_seed42}"
fi

# VMP-v6.2 preserves the frozen candidate pool and deterministic coverage
# policy while fixing evidence perception: scalar one-item fields are accepted,
# role-aware excerpt V3 ignores generic question words, and the extractor is
# explicitly required to retain useful partial evidence.
export PAPER_VERSION_NAME="${PAPER_VERSION_NAME:-VMP-v6.2}"
export SELECTOR_PROMPT_VERSION="${SELECTOR_PROMPT_VERSION:-vmp_v62_partial_atomic_fact_extractor_v2}"
export BOUNDARY_PROMPT_VERSION="${BOUNDARY_PROMPT_VERSION:-vmp_v62_deterministic_set_coverage_v2}"
export CANDIDATE_EXCERPT_VERSION="${CANDIDATE_EXCERPT_VERSION:-role_aware_fact_v3}"
export RERANK_MAX_TOKENS="${RERANK_MAX_TOKENS:-512}"
export COVERAGE_MIN_GAIN="${COVERAGE_MIN_GAIN:-0.25}"
export COVERAGE_NEED_WEIGHT="${COVERAGE_NEED_WEIGHT:-3.0}"
export COVERAGE_RELEVANCE_WEIGHT="${COVERAGE_RELEVANCE_WEIGHT:-1.5}"
export COVERAGE_DIVERSITY_WEIGHT="${COVERAGE_DIVERSITY_WEIGHT:-1.25}"
export COVERAGE_TEMPORAL_WEIGHT="${COVERAGE_TEMPORAL_WEIGHT:-1.25}"
export COVERAGE_RANK_WEIGHT="${COVERAGE_RANK_WEIGHT:-0.08}"
export DEV_RERANK_RUN_ID="${DEV_RERANK_RUN_ID:-lme_dev_vmp_v62_rerank_seed42}"
export TEST_CANDIDATE_RUN_ID="${TEST_CANDIDATE_RUN_ID:-lme_test_vmp_v62_candidates_seed42}"
export TEST_RERANK_RUN_ID="${TEST_RERANK_RUN_ID:-lme_test_vmp_v62_rerank_seed42}"
export GATE_RECEIPT="${GATE_RECEIPT:-outputs/longmemeval/gates/vmp_v62_seed42_dev_pass.json}"
export LOG_PATH="${LOG_PATH:-outputs/longmemeval/logs/vmp_v62_${STAGE:-dev_rerank}.log}"

exec bash "${PROJECT_ROOT}/scripts/run_vmp_v6_experiment.sh"
