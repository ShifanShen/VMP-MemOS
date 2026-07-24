#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_PATH="${DATA_PATH:-data/longmemeval/longmemeval_s_cleaned.json}"
SPLIT_PATH="${SPLIT_PATH:-outputs/longmemeval/splits/dev_test_seed42.json}"
MODEL_PATH="${MODEL_PATH:-outputs/longmemeval/models/vmp_v4_seed42.json}"
SEARCH_REPORT="${SEARCH_REPORT:-outputs/longmemeval/models/vmp_v4_seed42_search.json}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-BAAI/bge-m3}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cuda}"
EMBEDDING_CACHE_DIR="${EMBEDDING_CACHE_DIR:-${HOME}/.cache/huggingface}"
EMBEDDING_CACHE_DB="${EMBEDDING_CACHE_DB:-outputs/longmemeval/cache/bge_m3.sqlite3}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-8}"
RUN_ID="${RUN_ID:-lme_test_vmp_v4_$(date -u +%Y%m%dT%H%M%SZ)}"
METHODS="${METHODS:-empty,bm25,naive_vector,vector_recency,vector_importance,vmp_rule,vmp_tuned}"
TUNING_TRIALS="${TUNING_TRIALS:-512}"
STABILITY_FOLDS="${STABILITY_FOLDS:-5}"
MIN_DEV_RECALL_ALL_5="${MIN_DEV_RECALL_ALL_5:-0.90}"
MIN_DEV_DELTA_VS_DENSE="${MIN_DEV_DELTA_VS_DENSE:-0.02}"
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

log_stage "Starting VMP-v4 robust dense-guard experiment."
log_stage "run_id=${RUN_ID} data=${DATA_PATH} model=${EMBEDDING_MODEL} device=${EMBEDDING_DEVICE}"
log_stage "embedding_batch_size=${EMBEDDING_BATCH_SIZE} prewarm_embeddings=true"
log_stage "methods=${METHODS} trials=${TUNING_TRIALS} folds=${STABILITY_FOLDS} run_qa=${RUN_QA}"
log_stage "dense_safety=preserve_top10 protected_top5>=4 log=${LOG_PATH}"

log_stage "Phase 1/5: creating deterministic LongMemEval split."
python scripts/create_longmemeval_split.py \
  --data "${DATA_PATH}" \
  --output "${SPLIT_PATH}" \
  --seed 42 \
  --dev-size 100 \
  --test-size 400

log_stage "Phase 2/5: precomputing Dev features and robustly tuning VMP-v4."
python scripts/train_vmp_tuned.py \
  --data "${DATA_PATH}" \
  --split-manifest "${SPLIT_PATH}" \
  --output "${MODEL_PATH}" \
  --report "${SEARCH_REPORT}" \
  --embedding-model "${EMBEDDING_MODEL}" \
  --embedding-device "${EMBEDDING_DEVICE}" \
  --embedding-cache-dir "${EMBEDDING_CACHE_DIR}" \
  --embedding-cache-db "${EMBEDDING_CACHE_DB}" \
  --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
  --trials "${TUNING_TRIALS}" \
  --stability-folds "${STABILITY_FOLDS}" \
  --tuning-seed 2025 \
  --retrieval-depth 10 \
  --qa-top-k 5

log_stage "Phase 3/5: enforcing robust Dev gates before Test."
python scripts/check_vmp_v4_gate.py \
  --model "${MODEL_PATH}" \
  --min-recall-all-at-5 "${MIN_DEV_RECALL_ALL_5}" \
  --min-delta-vs-dense "${MIN_DEV_DELTA_VS_DENSE}"

log_stage "Phase 4/5: evaluating frozen methods on Test."
python scripts/run_longmemeval_retrieval.py \
  --data "${DATA_PATH}" \
  --split-manifest "${SPLIT_PATH}" \
  --split test \
  --vmp-tuned-model "${MODEL_PATH}" \
  --methods "${METHODS}" \
  --top-k 5 \
  --retrieval-depth 10 \
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
  python scripts/export_longmemeval_cost.py \
    --retrieval-run "outputs/longmemeval/runs/${RUN_ID}"
else
  log_stage "Phase 5/5: QA skipped because RUN_QA=${RUN_QA}."
fi

log_stage "Exporting paper tables."
python scripts/export_longmemeval_tables.py \
  --retrieval-run "outputs/longmemeval/runs/${RUN_ID}" \
  --output-dir "${TABLE_DIR}"

echo "Completed VMP-v4 test run: outputs/longmemeval/runs/${RUN_ID}"
