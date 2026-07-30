#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# V5.5.1 is the protocol-corrected V5.5 rerun. It changes only the selector
# prompt version and artifact names: the planner, candidates, model settings,
# symbolic boundary, and strict outcome gates remain frozen.
export PAPER_VERSION_NAME="${PAPER_VERSION_NAME:-VMP-v5.5.1}"
export SELECTOR_PROMPT_VERSION="${SELECTOR_PROMPT_VERSION:-vmp_v551_complete_challenger_selector_v1}"
export DEV_RERANK_RUN_ID="${DEV_RERANK_RUN_ID:-lme_dev_vmp_v551_rerank_seed42}"
export TEST_CANDIDATE_RUN_ID="${TEST_CANDIDATE_RUN_ID:-lme_test_vmp_v551_candidates_seed42}"
export TEST_RERANK_RUN_ID="${TEST_RERANK_RUN_ID:-lme_test_vmp_v551_rerank_seed42}"
export GATE_RECEIPT="${GATE_RECEIPT:-outputs/longmemeval/gates/vmp_v551_seed42_dev_pass.json}"
export LOG_PATH="${LOG_PATH:-outputs/longmemeval/logs/vmp_v551_${STAGE:-dev_rerank}.log}"

exec bash "${PROJECT_ROOT}/scripts/run_vmp_v55_experiment.sh"
