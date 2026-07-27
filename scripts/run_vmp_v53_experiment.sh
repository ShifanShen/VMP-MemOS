#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${STAGE:-}"
DATA_PATH="${DATA_PATH:-data/longmemeval/longmemeval_s_cleaned.json}"
SPLIT_PATH="${SPLIT_PATH:-outputs/longmemeval/splits/dev_test_seed42.json}"
V43_MODEL_PATH="${V43_MODEL_PATH:-outputs/longmemeval/models/vmp_v43_seed42.json}"
V5_MODEL_PATH="${V5_MODEL_PATH:-outputs/longmemeval/models/vmp_v5_seed42.json}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-BAAI/bge-m3}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cuda}"
EMBEDDING_CACHE_DIR="${EMBEDDING_CACHE_DIR:-${HOME}/.cache/huggingface}"
EMBEDDING_CACHE_DB="${EMBEDDING_CACHE_DB:-outputs/longmemeval/cache/bge_m3.sqlite3}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-2}"
VMP_LLM_BASE_URL="${VMP_LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
VMP_LLM_MODEL="${VMP_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
CANDIDATE_METHODS="${CANDIDATE_METHODS:-vmp_tuned,vmp_hierarchical}"
CANDIDATE_COUNT="${CANDIDATE_COUNT:-${VMP_V53_CANDIDATE_COUNT:-30}}"
OUTPUT_TOP_K="${OUTPUT_TOP_K:-5}"
SELECTOR_PROTECTED_TOP_N="${SELECTOR_PROTECTED_TOP_N:-${VMP_V53_SELECTOR_PROTECTED_TOP_N:-3}}"
BOUNDARY_PROTECTED_TOP_N="${BOUNDARY_PROTECTED_TOP_N:-${VMP_V53_BOUNDARY_PROTECTED_TOP_N:-3}}"
BOUNDARY_MAX_PROMOTIONS="${BOUNDARY_MAX_PROMOTIONS:-${VMP_V53_BOUNDARY_MAX_PROMOTIONS:-2}}"
MAX_CANDIDATE_CHARS="${MAX_CANDIDATE_CHARS:-1200}"
MAX_EXCERPT_TURNS="${MAX_EXCERPT_TURNS:-4}"
RERANK_MAX_TOKENS="${RERANK_MAX_TOKENS:-512}"
BOUNDARY_MAX_TOKENS="${BOUNDARY_MAX_TOKENS:-256}"
RERANK_RESUME="${RERANK_RESUME:-0}"
DEV_CANDIDATE_RUN_ID="${DEV_CANDIDATE_RUN_ID:-lme_dev_vmp_v53_candidates_seed42}"
DEV_RERANK_RUN_ID="${DEV_RERANK_RUN_ID:-lme_dev_vmp_v53_rerank_seed42}"
TEST_CANDIDATE_RUN_ID="${TEST_CANDIDATE_RUN_ID:-lme_test_vmp_v53_candidates_seed42}"
TEST_RERANK_RUN_ID="${TEST_RERANK_RUN_ID:-lme_test_vmp_v53_rerank_seed42}"
DEV_CANDIDATE_RUN="outputs/longmemeval/runs/${DEV_CANDIDATE_RUN_ID}"
DEV_RERANK_RUN="outputs/longmemeval/runs/${DEV_RERANK_RUN_ID}"
TEST_CANDIDATE_RUN="outputs/longmemeval/runs/${TEST_CANDIDATE_RUN_ID}"
TEST_RERANK_RUN="outputs/longmemeval/runs/${TEST_RERANK_RUN_ID}"
GATE_RECEIPT="${GATE_RECEIPT:-outputs/longmemeval/gates/vmp_v53_seed42_dev_pass.json}"
LOG_DIR="${LOG_DIR:-outputs/longmemeval/logs}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/vmp_v53_${STAGE:-missing_stage}.log}"
TABLE_DIR="${TABLE_DIR:-outputs/longmemeval/tables/${TEST_RERANK_RUN_ID}}"
RUN_QA="${RUN_QA:-0}"

MIN_DEV_RECALL_ALL_5="${MIN_DEV_RECALL_ALL_5:-0.93}"
MIN_DEV_DELTA_VS_RAW_V5="${MIN_DEV_DELTA_VS_RAW_V5:-0.025}"
MIN_DEV_DELTA_VS_SHARED_V43="${MIN_DEV_DELTA_VS_SHARED_V43:-0.03}"
MIN_DEV_MACRO_DELTA_VS_RAW_V5="${MIN_DEV_MACRO_DELTA_VS_RAW_V5:-0.0}"
MAX_DEV_TYPE_REGRESSION_VS_RAW_V5="${MAX_DEV_TYPE_REGRESSION_VS_RAW_V5:-0.03}"
MAX_PARSE_FALLBACK_RATE="${MAX_PARSE_FALLBACK_RATE:-0.02}"
MAX_BOUNDARY_FALLBACK_RATE="${MAX_BOUNDARY_FALLBACK_RATE:-0.02}"
MAX_REGRESSED_QUESTIONS="${MAX_REGRESSED_QUESTIONS:-0}"
MIN_RECOVERED_QUESTIONS="${MIN_RECOVERED_QUESTIONS:-3}"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

log_stage() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

on_exit() {
  exit_code=$?
  if [[ "${exit_code}" -eq 0 ]]; then
    log_stage "VMP-v5.3 stage ${STAGE:-unset} completed successfully."
  else
    log_stage "VMP-v5.3 stage ${STAGE:-unset} failed or was interrupted (exit_code=${exit_code})."
  fi
}
trap on_exit EXIT

if [[ -z "${STAGE}" ]]; then
  echo "Set STAGE to dev_candidates, dev_rerank, test_candidates, or test_rerank." >&2
  exit 2
fi
if [[ ! -f "${V43_MODEL_PATH}" || ! -f "${V5_MODEL_PATH}" ]]; then
  echo "Missing frozen V4.3 or V5 model artifact." >&2
  exit 2
fi

create_split() {
  python scripts/create_longmemeval_split.py \
    --data "${DATA_PATH}" \
    --output "${SPLIT_PATH}" \
    --seed 42 \
    --dev-size 100 \
    --test-size 400
}

run_candidates() {
  local split_name="$1"
  local run_id="$2"
  local dev_args=()
  if [[ "${split_name}" == "dev" ]]; then
    dev_args=(--allow-dev-model-selection)
  fi
  python scripts/run_longmemeval_retrieval.py \
    --data "${DATA_PATH}" \
    --split-manifest "${SPLIT_PATH}" \
    --split "${split_name}" \
    --vmp-tuned-model "${V43_MODEL_PATH}" \
    --vmp-hierarchical-model "${V5_MODEL_PATH}" \
    --methods "${CANDIDATE_METHODS}" \
    --top-k "${OUTPUT_TOP_K}" \
    --retrieval-depth "${CANDIDATE_COUNT}" \
    --ingestion-granularity session \
    --embedding-model "${EMBEDDING_MODEL}" \
    --embedding-device "${EMBEDDING_DEVICE}" \
    --embedding-cache-dir "${EMBEDDING_CACHE_DIR}" \
    --embedding-cache-db "${EMBEDDING_CACHE_DB}" \
    --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
    --prewarm-embeddings \
    --run-id "${run_id}" \
    "${dev_args[@]}"
}

run_rerank() {
  local source_run="$1"
  local run_id="$2"
  local resume_args=()
  if [[ "${RERANK_RESUME}" == "1" ]]; then
    resume_args=(--resume)
  fi
  python scripts/run_longmemeval_rerank.py \
    --source-run "${source_run}" \
    --run-id "${run_id}" \
    --methods "${CANDIDATE_METHODS}" \
    --base-url "${VMP_LLM_BASE_URL}" \
    --model "${VMP_LLM_MODEL}" \
    --candidate-count "${CANDIDATE_COUNT}" \
    --output-top-k "${OUTPUT_TOP_K}" \
    --protected-top-n "${SELECTOR_PROTECTED_TOP_N}" \
    --ranked-output-count 10 \
    --max-candidate-chars "${MAX_CANDIDATE_CHARS}" \
    --max-excerpt-turns "${MAX_EXCERPT_TURNS}" \
    --max-tokens "${RERANK_MAX_TOKENS}" \
    --boundary-verification \
    --boundary-protected-top-n "${BOUNDARY_PROTECTED_TOP_N}" \
    --boundary-max-promotions "${BOUNDARY_MAX_PROMOTIONS}" \
    --boundary-max-tokens "${BOUNDARY_MAX_TOKENS}" \
    --boundary-min-confidence high \
    "${resume_args[@]}"
}

check_dev_gate() {
  python scripts/check_vmp_v53_gate.py \
    --candidate-run "${DEV_CANDIDATE_RUN}" \
    --rerank-run "${DEV_RERANK_RUN}" \
    --vmp-method vmp_hierarchical \
    --baseline-method vmp_tuned \
    --receipt "${GATE_RECEIPT}" \
    --min-recall-all-at-5 "${MIN_DEV_RECALL_ALL_5}" \
    --min-delta-vs-raw-v5 "${MIN_DEV_DELTA_VS_RAW_V5}" \
    --min-delta-vs-shared-v43 "${MIN_DEV_DELTA_VS_SHARED_V43}" \
    --min-macro-delta-vs-raw-v5 "${MIN_DEV_MACRO_DELTA_VS_RAW_V5}" \
    --max-type-regression-vs-raw-v5 "${MAX_DEV_TYPE_REGRESSION_VS_RAW_V5}" \
    --max-parse-fallback-rate "${MAX_PARSE_FALLBACK_RATE}" \
    --max-boundary-fallback-rate "${MAX_BOUNDARY_FALLBACK_RATE}" \
    --max-regressed-questions "${MAX_REGRESSED_QUESTIONS}" \
    --min-recovered-questions "${MIN_RECOVERED_QUESTIONS}" \
    --min-candidate-count "${CANDIDATE_COUNT}"
}

case "${STAGE}" in
  dev_candidates)
    log_stage "Stage 1/4: creating V4.3/V5 Dev Top-${CANDIDATE_COUNT} pools."
    log_stage "Keep vLLM stopped; this stage uses BGE-M3 on ${EMBEDDING_DEVICE}."
    create_split
    run_candidates dev "${DEV_CANDIDATE_RUN_ID}"
    ;;
  dev_rerank)
    log_stage "Stage 2/4: shared selector plus conservative boundary verification on Dev."
    log_stage "source=${DEV_CANDIDATE_RUN} model=${VMP_LLM_MODEL}"
    run_rerank "${DEV_CANDIDATE_RUN}" "${DEV_RERANK_RUN_ID}"
    log_stage "Enforcing strict V5.3 Dev-only quality gate."
    check_dev_gate
    ;;
  test_candidates)
    log_stage "Stage 3/4: rechecking the sealed V5.3 Dev gate."
    check_dev_gate
    log_stage "Generating frozen Test candidates. Stop vLLM for this BGE-M3 stage."
    create_split
    run_candidates test "${TEST_CANDIDATE_RUN_ID}"
    ;;
  test_rerank)
    log_stage "Stage 4/4: rechecking V5.3 Dev gate before sealed Test reranking."
    check_dev_gate
    log_stage "Applying the same two-stage local-vLLM policy to every Test method."
    run_rerank "${TEST_CANDIDATE_RUN}" "${TEST_RERANK_RUN_ID}"
    RERANKED_METHODS="$(
      python - "${CANDIDATE_METHODS}" <<'PY'
import sys

methods = [item.strip().lower().replace("-", "_") for item in sys.argv[1].split(",")]
print(",".join(f"{method}__vllm_boundary" for method in methods if method))
PY
    )"
    if [[ "${RUN_QA}" == "1" ]]; then
      log_stage "Generating QA with the same local vLLM after boundary verification."
      python scripts/run_longmemeval_qa.py \
        --retrieval-run "${TEST_RERANK_RUN}" \
        --methods "${RERANKED_METHODS}" \
        --base-url "${VMP_LLM_BASE_URL}" \
        --model "${VMP_LLM_MODEL}" \
        --top-k "${OUTPUT_TOP_K}" \
        --temperature 0 \
        --top-p 1 \
        --max-tokens 128
    else
      log_stage "QA skipped because RUN_QA=${RUN_QA}."
    fi
    python scripts/export_longmemeval_tables.py \
      --retrieval-run "${TEST_RERANK_RUN}" \
      --output-dir "${TABLE_DIR}"
    ;;
  *)
    echo "Unknown STAGE=${STAGE}. Expected dev_candidates, dev_rerank, test_candidates, or test_rerank." >&2
    exit 2
    ;;
esac
