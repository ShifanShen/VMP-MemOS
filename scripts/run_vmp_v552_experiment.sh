#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# V5.5.2 keeps the frozen Dev candidate pool and strict outcome gates. It
# removes the observed fixed-label bias by comparing C06-C10 in five identical
# anonymous X positions. Each call integrates evidence binding and conservative
# B1/B2 replacement validation, so no separate boundary LLM call is made.
export PAPER_VERSION_NAME="${PAPER_VERSION_NAME:-VMP-v5.5.2}"
export SELECTOR_PROMPT_VERSION="${SELECTOR_PROMPT_VERSION:-vmp_v552_anonymous_pairwise_selector_v1}"
export BOUNDARY_PROMPT_VERSION="${BOUNDARY_PROMPT_VERSION:-vmp_v552_integrated_pairwise_boundary_v1}"
export CANDIDATE_EXCERPT_VERSION="${CANDIDATE_EXCERPT_VERSION:-role_aware_fact_v2}"
export RERANK_MAX_TOKENS="${RERANK_MAX_TOKENS:-384}"
export DEV_RERANK_RUN_ID="${DEV_RERANK_RUN_ID:-lme_dev_vmp_v552_rerank_seed42}"
export TEST_CANDIDATE_RUN_ID="${TEST_CANDIDATE_RUN_ID:-lme_test_vmp_v552_candidates_seed42}"
export TEST_RERANK_RUN_ID="${TEST_RERANK_RUN_ID:-lme_test_vmp_v552_rerank_seed42}"
export GATE_RECEIPT="${GATE_RECEIPT:-outputs/longmemeval/gates/vmp_v552_seed42_dev_pass.json}"
export LOG_PATH="${LOG_PATH:-outputs/longmemeval/logs/vmp_v552_${STAGE:-dev_rerank}.log}"

exec bash "${PROJECT_ROOT}/scripts/run_vmp_v55_experiment.sh"
