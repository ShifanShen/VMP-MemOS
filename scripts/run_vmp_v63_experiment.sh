#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${STAGE:-}" == "dev_smoke" ]]; then
  export CANDIDATE_METHODS="${CANDIDATE_METHODS:-vmp_hierarchical}"
  export RERANK_QUESTION_IDS="${RERANK_QUESTION_IDS:-8e91e7d9,af082822,1a8a66a6,0bc8ad92}"
  export DEV_RERANK_RUN_ID="${DEV_RERANK_RUN_ID:-lme_dev_vmp_v63_smoke_seed42}"
fi

# VMP-v6.3 keeps the frozen V6.2 candidates, model, Top-k policy, and default
# coverage weights. It version-isolates three grounding corrections: list-valued
# facts are canonicalized, evidence-coordinate entities are rejected, and bare
# enumeration markers cannot become candidate evidence.
export PAPER_VERSION_NAME="${PAPER_VERSION_NAME:-VMP-v6.3}"
export SELECTOR_PROMPT_VERSION="${SELECTOR_PROMPT_VERSION:-vmp_v63_grounded_atomic_fact_extractor_v3}"
export BOUNDARY_PROMPT_VERSION="${BOUNDARY_PROMPT_VERSION:-vmp_v63_deterministic_set_coverage_v3}"
export CANDIDATE_EXCERPT_VERSION="${CANDIDATE_EXCERPT_VERSION:-role_aware_fact_v4}"
export RERANK_MAX_TOKENS="${RERANK_MAX_TOKENS:-512}"
export COVERAGE_MIN_GAIN="${COVERAGE_MIN_GAIN:-0.25}"
export COVERAGE_NEED_WEIGHT="${COVERAGE_NEED_WEIGHT:-3.0}"
export COVERAGE_RELEVANCE_WEIGHT="${COVERAGE_RELEVANCE_WEIGHT:-1.5}"
export COVERAGE_DIVERSITY_WEIGHT="${COVERAGE_DIVERSITY_WEIGHT:-1.25}"
export COVERAGE_TEMPORAL_WEIGHT="${COVERAGE_TEMPORAL_WEIGHT:-1.25}"
export COVERAGE_RANK_WEIGHT="${COVERAGE_RANK_WEIGHT:-0.08}"
export DEV_RERANK_RUN_ID="${DEV_RERANK_RUN_ID:-lme_dev_vmp_v63_rerank_seed42}"
export TEST_CANDIDATE_RUN_ID="${TEST_CANDIDATE_RUN_ID:-lme_test_vmp_v63_candidates_seed42}"
export TEST_RERANK_RUN_ID="${TEST_RERANK_RUN_ID:-lme_test_vmp_v63_rerank_seed42}"
export GATE_RECEIPT="${GATE_RECEIPT:-outputs/longmemeval/gates/vmp_v63_seed42_dev_pass.json}"
export LOG_PATH="${LOG_PATH:-outputs/longmemeval/logs/vmp_v63_${STAGE:-dev_rerank}.log}"

exec bash "${PROJECT_ROOT}/scripts/run_vmp_v6_experiment.sh"
