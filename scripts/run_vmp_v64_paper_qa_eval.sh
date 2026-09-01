#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${STAGE:-judge_smoke}"
DATA_PATH="${DATA_PATH:-data/longmemeval/longmemeval_s_cleaned.json}"
QA_RUN="${QA_RUN:-outputs/longmemeval/runs/lme_test_vmp_v64_rerank_seed42/qa_v21_test}"
QA_METHODS="${QA_METHODS:-vmp_tuned__vllm_boundary,vmp_hierarchical__vllm_boundary}"
REFERENCE_METHOD="${REFERENCE_METHOD:-vmp_hierarchical__vllm_boundary}"
VMP_LLM_BASE_URL="${VMP_LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
VMP_LLM_MODEL="${VMP_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
JUDGE_SUBDIR="${JUDGE_SUBDIR:-official_judge_local_vllm_v1}"
JUDGE_SMOKE_SUBDIR="${JUDGE_SMOKE_SUBDIR:-official_judge_local_vllm_v1_smoke}"
JUDGE_RESUME="${JUDGE_RESUME:-1}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-42}"
LOG_DIR="${LOG_DIR:-outputs/longmemeval/logs}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/vmp_v64_paper_qa_${STAGE}.log}"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

log_stage() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

on_exit() {
  exit_code=$?
  if [[ "${exit_code}" -eq 0 ]]; then
    log_stage "VMP-v6.4 paper QA stage ${STAGE} completed successfully."
  else
    log_stage "VMP-v6.4 paper QA stage ${STAGE} failed or was interrupted (exit_code=${exit_code})."
  fi
}
trap on_exit EXIT

judge() {
  local output_subdir="$1"
  local limit="${2:-}"
  local args=(
    --qa-run "${QA_RUN}"
    --reference-data "${DATA_PATH}"
    --methods "${QA_METHODS}"
    --output-subdir "${output_subdir}"
    --base-url "${VMP_LLM_BASE_URL}"
    --model "${VMP_LLM_MODEL}"
  )
  if [[ -n "${VMP_LLM_API_KEY:-}" ]]; then
    args+=(--api-key "${VMP_LLM_API_KEY}")
  fi
  if [[ -n "${limit}" ]]; then
    args+=(--limit "${limit}")
  fi
  if [[ "${JUDGE_RESUME}" == "1" ]]; then
    args+=(--resume)
  fi
  python scripts/run_longmemeval_official_judge.py "${args[@]}"
}

report() {
  python scripts/export_longmemeval_qa_report.py \
    --judge-run "${QA_RUN}/${JUDGE_SUBDIR}" \
    --reference-method "${REFERENCE_METHOD}" \
    --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
    --seed "${BOOTSTRAP_SEED}"
}

case "${STAGE}" in
  judge_smoke)
    log_stage "Judging the first 10 saved Test predictions with the official prompt family."
    log_stage "This uses the shared local vLLM judge and does not regenerate answers."
    judge "${JUDGE_SMOKE_SUBDIR}" 10
    ;;
  judge)
    log_stage "Judging every saved prediction with one shared local vLLM model."
    log_stage "Scores are official-prompt compatible, not published GPT-4o judge scores."
    judge "${JUDGE_SUBDIR}"
    ;;
  report)
    log_stage "Exporting deterministic paper tables and paired significance tests."
    report
    ;;
  all)
    log_stage "Running the complete shared local judge, then exporting paper tables."
    judge "${JUDGE_SUBDIR}"
    report
    ;;
  *)
    echo "Unknown STAGE=${STAGE}. Expected judge_smoke, judge, report, or all." >&2
    exit 2
    ;;
esac
