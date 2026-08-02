#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COVERAGE_MODEL_PATH="${COVERAGE_MODEL_PATH:-outputs/longmemeval/models/vmp_v6_coverage_seed42.json}"

if [[ -f "${COVERAGE_MODEL_PATH}" ]]; then
  COVERAGE_MODEL_RESOLVED="${COVERAGE_MODEL_PATH}"
elif [[ -f "${PROJECT_ROOT}/${COVERAGE_MODEL_PATH}" ]]; then
  COVERAGE_MODEL_RESOLVED="${PROJECT_ROOT}/${COVERAGE_MODEL_PATH}"
else
  echo "Missing tuned V6 coverage model: ${COVERAGE_MODEL_PATH}" >&2
  exit 2
fi

mapfile -t COVERAGE_VALUES < <(
  python - "${COVERAGE_MODEL_RESOLVED}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
best = payload.get("best", {})
weights = best.get("weights", {})
required = (
    "min_gain",
    "need_weight",
    "relevance_weight",
    "diversity_weight",
    "temporal_weight",
    "rank_weight",
)
missing = [name for name in required if name not in weights]
if missing:
    raise SystemExit(f"Coverage model is missing weights: {missing}")
for name in required:
    print(weights[name])
PY
)

export COVERAGE_MIN_GAIN="${COVERAGE_VALUES[0]}"
export COVERAGE_NEED_WEIGHT="${COVERAGE_VALUES[1]}"
export COVERAGE_RELEVANCE_WEIGHT="${COVERAGE_VALUES[2]}"
export COVERAGE_DIVERSITY_WEIGHT="${COVERAGE_VALUES[3]}"
export COVERAGE_TEMPORAL_WEIGHT="${COVERAGE_VALUES[4]}"
export COVERAGE_RANK_WEIGHT="${COVERAGE_VALUES[5]}"
export VMP_COVERAGE_MODEL_PATH="${COVERAGE_MODEL_RESOLVED}"
export VMP_COVERAGE_MODEL_SHA256="$(sha256sum "${COVERAGE_MODEL_RESOLVED}" | awk '{print $1}')"
export PAPER_VERSION_NAME="${PAPER_VERSION_NAME:-VMP-v6.1}"
export DEV_RERANK_RUN_ID="${DEV_RERANK_RUN_ID:-lme_dev_vmp_v61_rerank_seed42}"
export TEST_CANDIDATE_RUN_ID="${TEST_CANDIDATE_RUN_ID:-lme_test_vmp_v61_candidates_seed42}"
export TEST_RERANK_RUN_ID="${TEST_RERANK_RUN_ID:-lme_test_vmp_v61_rerank_seed42}"
export GATE_RECEIPT="${GATE_RECEIPT:-outputs/longmemeval/gates/vmp_v61_seed42_dev_pass.json}"
export LOG_PATH="${LOG_PATH:-outputs/longmemeval/logs/vmp_v61_${STAGE:-dev_rerank}.log}"

exec bash "${PROJECT_ROOT}/scripts/run_vmp_v6_experiment.sh"
