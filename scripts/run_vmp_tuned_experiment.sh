#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_PATH="${DATA_PATH:-data/longmemeval/longmemeval_s_cleaned.json}"
SPLIT_PATH="${SPLIT_PATH:-outputs/longmemeval/splits/dev_test_seed42.json}"
MODEL_PATH="${MODEL_PATH:-outputs/longmemeval/models/vmp_tuned_seed42.json}"
SEARCH_REPORT="${SEARCH_REPORT:-outputs/longmemeval/models/vmp_tuned_seed42_search.json}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-BAAI/bge-m3}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cuda}"
EMBEDDING_CACHE_DIR="${EMBEDDING_CACHE_DIR:-${HOME}/.cache/huggingface}"
EMBEDDING_CACHE_DB="${EMBEDDING_CACHE_DB:-outputs/longmemeval/cache/bge_m3.sqlite3}"
RUN_ID="${RUN_ID:-lme_test_vmp_tuned_$(date -u +%Y%m%dT%H%M%SZ)}"
METHODS="${METHODS:-empty,bm25,naive_vector,vector_recency,vector_importance,vmp_rule,vmp_tuned}"
VMP_LLM_BASE_URL="${VMP_LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
VMP_LLM_MODEL="${VMP_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
RUN_QA="${RUN_QA:-0}"

cd "${PROJECT_ROOT}"

python scripts/create_longmemeval_split.py \
  --data "${DATA_PATH}" \
  --output "${SPLIT_PATH}" \
  --seed 42 \
  --dev-size 100 \
  --test-size 400

python scripts/train_vmp_tuned.py \
  --data "${DATA_PATH}" \
  --split-manifest "${SPLIT_PATH}" \
  --output "${MODEL_PATH}" \
  --report "${SEARCH_REPORT}" \
  --embedding-model "${EMBEDDING_MODEL}" \
  --embedding-device "${EMBEDDING_DEVICE}" \
  --embedding-cache-dir "${EMBEDDING_CACHE_DIR}" \
  --embedding-cache-db "${EMBEDDING_CACHE_DB}" \
  --trials 64 \
  --tuning-seed 2025 \
  --retrieval-depth 10 \
  --qa-top-k 5

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
  --run-id "${RUN_ID}"

if [[ "${RUN_QA}" == "1" ]]; then
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
fi

python scripts/export_longmemeval_tables.py \
  --retrieval-run "outputs/longmemeval/runs/${RUN_ID}"

echo "Completed VMP-Tuned test run: outputs/longmemeval/runs/${RUN_ID}"
if [[ "${RUN_QA}" != "1" ]]; then
  echo "QA was skipped. Start vLLM, then run run_longmemeval_qa.py for this run."
fi
