#!/usr/bin/env bash
set -euo pipefail

export ENABLE_PROMOTION="${ENABLE_PROMOTION:-1}"
export EXPERIMENT_LABEL="${EXPERIMENT_LABEL:-VMP-v5.1}"
export GRID_STEP="${GRID_STEP:-0.05}"
export MODEL_PATH="${MODEL_PATH:-outputs/longmemeval/models/vmp_v51_seed42.json}"
export SEARCH_REPORT="${SEARCH_REPORT:-outputs/longmemeval/models/vmp_v51_seed42_search.json}"
export DEV_AUDIT_PATH="${DEV_AUDIT_PATH:-outputs/longmemeval/models/vmp_v51_seed42_dev_audit.jsonl}"
export RUN_ID="${RUN_ID:-lme_test_vmp_v51_$(date -u +%Y%m%dT%H%M%SZ)}"

# With 94 answerable Dev questions, a 0.005 delta requires at least one
# additional fully recalled question over V4.3 and pre-promotion V5.
export MIN_DEV_RECALL_ALL_5="${MIN_DEV_RECALL_ALL_5:-0.91}"
export MIN_DEV_DELTA_VS_V43="${MIN_DEV_DELTA_VS_V43:-0.005}"
export MIN_DEV_DELTA_VS_PRE_PROMOTION="${MIN_DEV_DELTA_VS_PRE_PROMOTION:-0.005}"
export MIN_DEV_MACRO_DELTA_VS_V43="${MIN_DEV_MACRO_DELTA_VS_V43:-0.0}"
export MAX_DEV_WORST_TYPE_REGRESSION_VS_V43="${MAX_DEV_WORST_TYPE_REGRESSION_VS_V43:-0.03}"

exec bash "$(dirname "${BASH_SOURCE[0]}")/run_vmp_v5_experiment.sh"
