#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_PATH="${DATA_PATH:-data/longmemeval/longmemeval_s_cleaned.json}"
SPLIT_PATH="${SPLIT_PATH:-outputs/longmemeval/splits/dev_test_seed42.json}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-outputs/longmemeval/models/vmp_v43_seed42.json}"
MODEL_PATH="${MODEL_PATH:-outputs/longmemeval/models/vmp_v5_seed42.json}"
SEARCH_REPORT="${SEARCH_REPORT:-outputs/longmemeval/models/vmp_v5_seed42_search.json}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-BAAI/bge-m3}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cuda}"
EMBEDDING_CACHE_DIR="${EMBEDDING_CACHE_DIR:-${HOME}/.cache/huggingface}"
EMBEDDING_CACHE_DB="${EMBEDDING_CACHE_DB:-outputs/longmemeval/cache/bge_m3.sqlite3}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-4}"
RUN_ID="${RUN_ID:-lme_test_vmp_v5_$(date -u +%Y%m%dT%H%M%SZ)}"
METHODS="${METHODS:-bm25,naive_vector,vector_importance,vmp_tuned,vmp_hierarchical}"
GRID_STEP="${GRID_STEP:-0.2}"
TURN_POOLING="${TURN_POOLING:-1,2,3}"
MIN_DEV_RECALL_ALL_5="${MIN_DEV_RECALL_ALL_5:-0.91}"
MIN_DEV_DELTA_VS_V43="${MIN_DEV_DELTA_VS_V43:-0.0}"
MIN_DEV_DELTA_VS_SESSION_ONLY="${MIN_DEV_DELTA_VS_SESSION_ONLY:-0.0}"
MIN_TURN_WEIGHT="${MIN_TURN_WEIGHT:-0.2}"
MAX_DEV_FOLD_RECALL_STDDEV="${MAX_DEV_FOLD_RECALL_STDDEV:-0.20}"
VMP_LLM_BASE_URL="${VMP_LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
VMP_LLM_MODEL="${VMP_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
RUN_QA="${RUN_QA:-0}"
LOG_DIR="${LOG_DIR:-outputs/longmemeval/logs}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/${RUN_ID}.log}"
TABLE_DIR="${TABLE_DIR:-outputs/longmemeval/tables/${RUN_ID}}"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

log_stage() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

on_exit() {
  exit_code=$?
  if [[ "${exit_code}" -eq 0 ]]; then
    log_stage "Experiment completed successfully."
  else
    log_stage "Experiment failed or was interrupted (exit_code=${exit_code})."
  fi
}
trap on_exit EXIT

if [[ ! -f "${BASE_MODEL_PATH}" ]]; then
  echo "Missing frozen VMP-v4.3 base model: ${BASE_MODEL_PATH}" >&2
  exit 2
fi

log_stage "Starting VMP-v5 hierarchical session-turn experiment."
log_stage "run_id=${RUN_ID} data=${DATA_PATH} base_model=${BASE_MODEL_PATH}"
log_stage "embedding_model=${EMBEDDING_MODEL} device=${EMBEDDING_DEVICE} batch=${EMBEDDING_BATCH_SIZE}"
log_stage "grid_step=${GRID_STEP} turn_pooling=${TURN_POOLING} methods=${METHODS}"
log_stage "run_qa=${RUN_QA} log=${LOG_PATH}"

log_stage "Phase 1/5: creating deterministic LongMemEval split."
python scripts/create_longmemeval_split.py \
  --data "${DATA_PATH}" \
  --output "${SPLIT_PATH}" \
  --seed 42 \
  --dev-size 100 \
  --test-size 400

log_stage "Phase 2/5: tuning hierarchical fusion on Dev only."
python scripts/train_vmp_hierarchical.py \
  --data "${DATA_PATH}" \
  --split-manifest "${SPLIT_PATH}" \
  --base-model "${BASE_MODEL_PATH}" \
  --output "${MODEL_PATH}" \
  --report "${SEARCH_REPORT}" \
  --embedding-model "${EMBEDDING_MODEL}" \
  --embedding-device "${EMBEDDING_DEVICE}" \
  --embedding-cache-dir "${EMBEDDING_CACHE_DIR}" \
  --embedding-cache-db "${EMBEDDING_CACHE_DB}" \
  --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
  --grid-step "${GRID_STEP}" \
  --turn-pooling "${TURN_POOLING}" \
  --retrieval-depth 10 \
  --qa-top-k 5 \
  --stability-folds 5

log_stage "Phase 3/5: enforcing V5 Dev-only quality gates."
python scripts/check_vmp_v5_gate.py \
  --model "${MODEL_PATH}" \
  --min-recall-all-at-5 "${MIN_DEV_RECALL_ALL_5}" \
  --min-delta-vs-v43 "${MIN_DEV_DELTA_VS_V43}" \
  --min-delta-vs-session-only "${MIN_DEV_DELTA_VS_SESSION_ONLY}" \
  --min-turn-weight "${MIN_TURN_WEIGHT}" \
  --max-fold-recall-stddev "${MAX_DEV_FOLD_RECALL_STDDEV}"

log_stage "Phase 4/5: evaluating frozen V4.3 and V5 methods on Test."
python scripts/run_longmemeval_retrieval.py \
  --data "${DATA_PATH}" \
  --split-manifest "${SPLIT_PATH}" \
  --split test \
  --vmp-tuned-model "${BASE_MODEL_PATH}" \
  --vmp-hierarchical-model "${MODEL_PATH}" \
  --methods "${METHODS}" \
  --top-k 5 \
  --retrieval-depth 10 \
  --ingestion-granularity session \
  --embedding-model "${EMBEDDING_MODEL}" \
  --embedding-device "${EMBEDDING_DEVICE}" \
  --embedding-cache-dir "${EMBEDDING_CACHE_DIR}" \
  --embedding-cache-db "${EMBEDDING_CACHE_DB}" \
  --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
  --prewarm-embeddings \
  --run-id "${RUN_ID}"

if [[ "${RUN_QA}" == "1" ]]; then
  log_stage "Phase 5/5: generating answers with the shared vLLM reader."
  python scripts/run_longmemeval_qa.py \
    --retrieval-run "outputs/longmemeval/runs/${RUN_ID}" \
    --methods "${METHODS}" \
    --base-url "${VMP_LLM_BASE_URL}" \
    --model "${VMP_LLM_MODEL}" \
    --top-k 5 \
    --temperature 0 \
    --top-p 1 \
    --max-tokens 128
else
  log_stage "Phase 5/5: QA skipped because RUN_QA=${RUN_QA}."
fi

log_stage "Exporting paper tables."
python scripts/export_longmemeval_tables.py \
  --retrieval-run "outputs/longmemeval/runs/${RUN_ID}" \
  --output-dir "${TABLE_DIR}"

echo "Completed VMP-v5 test run: outputs/longmemeval/runs/${RUN_ID}"
