#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${STAGE:-dev_smoke}"
VMP_LLM_BASE_URL="${VMP_LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
VMP_LLM_MODEL="${VMP_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
QA_METHODS="${QA_METHODS:-vmp_tuned__vllm_boundary,vmp_hierarchical__vllm_boundary}"
READER_PROMPT_VERSION="${READER_PROMPT_VERSION:-longmemeval_grounded_fact_reader_v2}"
READER_EVIDENCE_MODE="${READER_EVIDENCE_MODE:-reranker_facts}"
QA_TOP_K="${QA_TOP_K:-5}"
QA_MAX_TOKENS="${QA_MAX_TOKENS:-256}"
QA_RESUME="${QA_RESUME:-1}"
DEV_RETRIEVAL_RUN="${DEV_RETRIEVAL_RUN:-outputs/longmemeval/runs/lme_dev_vmp_v64_rerank_seed42}"
TEST_RETRIEVAL_RUN="${TEST_RETRIEVAL_RUN:-outputs/longmemeval/runs/lme_test_vmp_v64_rerank_seed42}"
DEV_SMOKE_SUBDIR="${DEV_SMOKE_SUBDIR:-qa_v2_smoke}"
DEV_QA_SUBDIR="${DEV_QA_SUBDIR:-qa_v2_dev}"
TEST_QA_SUBDIR="${TEST_QA_SUBDIR:-qa_v2_test}"
DEV_GATE_RECEIPT="${DEV_GATE_RECEIPT:-outputs/longmemeval/gates/vmp_v64_qa_v2_seed42_dev_pass.json}"
LOG_DIR="${LOG_DIR:-outputs/longmemeval/logs}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/vmp_v64_qa_v2_${STAGE}.log}"

MAX_ANSWERABLE_REFUSAL_RATE="${MAX_ANSWERABLE_REFUSAL_RATE:-0.25}"
MIN_ANSWERABLE_FACT_COVERAGE="${MIN_ANSWERABLE_FACT_COVERAGE:-0.90}"
MIN_TOKEN_F1="${MIN_TOKEN_F1:-0.25}"
MIN_CONTAINS_ANSWER="${MIN_CONTAINS_ANSWER:-0.10}"
MIN_ABSTENTION_ACCURACY="${MIN_ABSTENTION_ACCURACY:-0.50}"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

log_stage() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

on_exit() {
  exit_code=$?
  if [[ "${exit_code}" -eq 0 ]]; then
    log_stage "VMP-v6.4 QA-v2 stage ${STAGE} completed successfully."
  elif [[ "${exit_code}" -eq 3 ]]; then
    log_stage "VMP-v6.4 QA-v2 completed, but the Dev gate failed (exit_code=3)."
  else
    log_stage "VMP-v6.4 QA-v2 stage ${STAGE} failed or was interrupted (exit_code=${exit_code})."
  fi
}
trap on_exit EXIT

require_completed_run() {
  local run_dir="$1"
  python - "${run_dir}/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "completed":
    raise SystemExit(f"Run is not completed: {path}")
PY
}

run_qa() {
  local retrieval_run="$1"
  local qa_subdir="$2"
  local limit="${3:-}"
  local args=(
    --retrieval-run "${retrieval_run}"
    --methods "${QA_METHODS}"
    --base-url "${VMP_LLM_BASE_URL}"
    --model "${VMP_LLM_MODEL}"
    --top-k "${QA_TOP_K}"
    --prompt-version "${READER_PROMPT_VERSION}"
    --evidence-mode "${READER_EVIDENCE_MODE}"
    --qa-subdir "${qa_subdir}"
    --temperature 0
    --top-p 1
    --max-tokens "${QA_MAX_TOKENS}"
  )
  if [[ -n "${VMP_LLM_API_KEY:-}" ]]; then
    args+=(--api-key "${VMP_LLM_API_KEY}")
  fi
  if [[ -n "${limit}" ]]; then
    args+=(--limit "${limit}")
  fi
  if [[ "${QA_RESUME}" == "1" ]]; then
    args+=(--resume)
  fi
  python scripts/run_longmemeval_qa.py "${args[@]}"
}

check_dev_gate() {
  python scripts/check_longmemeval_qa_gate.py \
    --retrieval-run "${DEV_RETRIEVAL_RUN}" \
    --qa-subdir "${DEV_QA_SUBDIR}" \
    --methods "${QA_METHODS}" \
    --expected-prompt-version "${READER_PROMPT_VERSION}" \
    --expected-evidence-mode "${READER_EVIDENCE_MODE}" \
    --output "${DEV_GATE_RECEIPT}" \
    --max-answerable-refusal-rate "${MAX_ANSWERABLE_REFUSAL_RATE}" \
    --min-answerable-fact-coverage "${MIN_ANSWERABLE_FACT_COVERAGE}" \
    --min-token-f1 "${MIN_TOKEN_F1}" \
    --min-contains-answer "${MIN_CONTAINS_ANSWER}" \
    --min-abstention-accuracy "${MIN_ABSTENTION_ACCURACY}"
}

case "${STAGE}" in
  dev_smoke)
    require_completed_run "${DEV_RETRIEVAL_RUN}"
    log_stage "Smoke-running grounded QA-v2 on the first 10 frozen Dev questions."
    log_stage "The failed Test QA directory is not read or modified."
    run_qa "${DEV_RETRIEVAL_RUN}" "${DEV_SMOKE_SUBDIR}" 10
    ;;
  dev)
    require_completed_run "${DEV_RETRIEVAL_RUN}"
    log_stage "Running grounded QA-v2 on all frozen Dev questions."
    run_qa "${DEV_RETRIEVAL_RUN}" "${DEV_QA_SUBDIR}"
    log_stage "Enforcing the Dev-only refusal, fact-coverage, and local-quality gate."
    check_dev_gate
    ;;
  test)
    log_stage "Rechecking the frozen Dev QA-v2 gate before sealed Test generation."
    check_dev_gate
    require_completed_run "${TEST_RETRIEVAL_RUN}"
    log_stage "Generating Test answers with the exact Dev-approved reader protocol."
    run_qa "${TEST_RETRIEVAL_RUN}" "${TEST_QA_SUBDIR}"
    ;;
  *)
    echo "Unknown STAGE=${STAGE}. Expected dev_smoke, dev, or test." >&2
    exit 2
    ;;
esac
