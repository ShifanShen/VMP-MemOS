#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_PATH="${DATA_PATH:-data/longmemeval/longmemeval_s_cleaned.json}"
SPLIT_PATH="${SPLIT_PATH:-outputs/longmemeval/splits/dev_test_seed42.json}"
MODEL_PATH="${MODEL_PATH:-outputs/longmemeval/models/vmp_v3_seed42.json}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-BAAI/bge-m3}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cuda}"
EMBEDDING_CACHE_DIR="${EMBEDDING_CACHE_DIR:-${HOME}/.cache/huggingface}"
EMBEDDING_CACHE_DB="${EMBEDDING_CACHE_DB:-outputs/longmemeval/cache/bge_m3.sqlite3}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-1}"
RUN_ID="${RUN_ID:-lme_test_ablation_$(date -u +%Y%m%dT%H%M%SZ)}"
TABLE_DIR="${TABLE_DIR:-outputs/longmemeval/tables/${RUN_ID}}"
RUN_QA="${RUN_QA:-0}"
VMP_LLM_BASE_URL="${VMP_LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
VMP_LLM_MODEL="${VMP_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
METHODS="$(
  printf '%s' \
    'vmp_tuned,' \
    'vmp_tuned__no_recency,' \
    'vmp_tuned__no_contradiction,' \
    'vmp_tuned__no_redundancy,' \
    'vmp_tuned__no_importance,' \
    'vmp_tuned__no_confidence,' \
    'vmp_tuned__no_token_cost,' \
    'vmp_tuned__no_scope_match,' \
    'vmp_tuned__no_update_operation,' \
    'vmp_tuned__no_merge_operation,' \
    'vmp_tuned__no_archive_operation'
)"

cd "${PROJECT_ROOT}"

if [[ ! -f "${SPLIT_PATH}" ]]; then
  echo "Missing split manifest: ${SPLIT_PATH}" >&2
  echo "Run scripts/run_vmp_tuned_experiment.sh first." >&2
  exit 2
fi
if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Missing frozen VMP-Tuned model: ${MODEL_PATH}" >&2
  echo "Run scripts/run_vmp_tuned_experiment.sh first." >&2
  exit 2
fi

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
  python scripts/run_longmemeval_qa.py \
    --retrieval-run "outputs/longmemeval/runs/${RUN_ID}" \
    --methods "${METHODS}" \
    --base-url "${VMP_LLM_BASE_URL}" \
    --model "${VMP_LLM_MODEL}" \
    --top-k 5 \
    --temperature 0 \
    --top-p 1 \
    --max-tokens 128
fi

python scripts/export_longmemeval_ablation.py \
  --retrieval-run "outputs/longmemeval/runs/${RUN_ID}" \
  --output-dir "${TABLE_DIR}"

echo "Completed LongMemEval ablation run: outputs/longmemeval/runs/${RUN_ID}"
if [[ "${RUN_QA}" != "1" ]]; then
  echo "QA deltas are blank. Start vLLM, run QA, then export the table again."
fi
