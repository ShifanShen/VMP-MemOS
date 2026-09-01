#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRAMEWORK="${FRAMEWORK:-}"
STAGE="${STAGE:-status}"

case "${FRAMEWORK}" in
  mem0|langmem|graphiti|letta) ;;
  *)
    echo "Set FRAMEWORK to exactly one of: mem0, langmem, graphiti, letta." >&2
    exit 2
    ;;
esac

METHOD="${FRAMEWORK}_official"
RERANKED_METHOD="${METHOD}__vllm_boundary"
DATA_PATH="${DATA_PATH:-data/longmemeval/longmemeval_s_cleaned.json}"
SPLIT_PATH="${SPLIT_PATH:-outputs/longmemeval/splits/dev_test_seed42.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/longmemeval}"
AUDIT_DIR="${AUDIT_DIR:-${OUTPUT_DIR}/audit}"
FRAMEWORK_AUDIT_OUTPUT="${FRAMEWORK_AUDIT_OUTPUT:-${OUTPUT_DIR}/framework_audits/${FRAMEWORK}}"
VMP_LLM_BASE_URL="${VMP_LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
VMP_LLM_MODEL="${VMP_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-BAAI/bge-m3}"
EMBEDDING_DIMENSION="${EMBEDDING_DIMENSION:-1024}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cuda}"
EMBEDDING_CACHE_DIR="${EMBEDDING_CACHE_DIR:-${HOME}/.cache/huggingface}"
EMBEDDING_CACHE_DB="${EMBEDDING_CACHE_DB:-${OUTPUT_DIR}/cache/bge_m3.sqlite3}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-2}"
OFFICIAL_LLM_MAX_TOKENS="${VMP_OFFICIAL_LLM_MAX_TOKENS:-512}"
OFFICIAL_LLM_TEMPERATURE="${VMP_OFFICIAL_LLM_TEMPERATURE:-0.0}"

CANDIDATE_POOL_COUNT="${CANDIDATE_POOL_COUNT:-40}"
CANDIDATE_COUNT="${CANDIDATE_COUNT:-10}"
OUTPUT_TOP_K="${OUTPUT_TOP_K:-5}"
CANDIDATE_PLANNER_VERSION="${CANDIDATE_PLANNER_VERSION:-vmp_v55_dual_view_rrf_v1}"
SELECTOR_PROMPT_VERSION="${SELECTOR_PROMPT_VERSION:-vmp_v64_high_recall_atomic_fact_extractor_v4}"
BOUNDARY_PROMPT_VERSION="${BOUNDARY_PROMPT_VERSION:-vmp_v64_deterministic_set_coverage_v4}"
CANDIDATE_EXCERPT_VERSION="${CANDIDATE_EXCERPT_VERSION:-role_aware_fact_v4}"
READER_PROMPT_VERSION="${READER_PROMPT_VERSION:-longmemeval_hybrid_evidence_reader_v21}"
READER_EVIDENCE_MODE="${READER_EVIDENCE_MODE:-reranker_facts_with_query_windows}"

CANDIDATE_RUN_ID="${CANDIDATE_RUN_ID:-lme_test_${FRAMEWORK}_official_candidates_seed42}"
RERANK_RUN_ID="${RERANK_RUN_ID:-lme_test_${FRAMEWORK}_official_v64_rerank_seed42}"
CANDIDATE_RUN="${OUTPUT_DIR}/runs/${CANDIDATE_RUN_ID}"
RERANK_RUN="${OUTPUT_DIR}/runs/${RERANK_RUN_ID}"
QA_SUBDIR="${QA_SUBDIR:-qa_v21_test}"
JUDGE_SUBDIR="${JUDGE_SUBDIR:-official_judge_local_vllm_v1}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/official_${FRAMEWORK}_${STAGE}.log}"
RETRIEVAL_RESUME="${RETRIEVAL_RESUME:-1}"
RERANK_RESUME="${RERANK_RESUME:-1}"
QA_RESUME="${QA_RESUME:-1}"
JUDGE_RESUME="${JUDGE_RESUME:-1}"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

log_stage() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ${FRAMEWORK}: $*"
}

on_exit() {
  local exit_code=$?
  if [[ "${exit_code}" -eq 0 ]]; then
    log_stage "stage ${STAGE} completed successfully."
  elif [[ "${exit_code}" -eq 3 ]]; then
    log_stage "stage ${STAGE} completed, but a strict eligibility gate failed."
  else
    log_stage "stage ${STAGE} failed or was interrupted (exit_code=${exit_code})."
  fi
}
trap on_exit EXIT

framework_service_args() {
  FRAMEWORK_SERVICE_ARGS=()
  if [[ "${FRAMEWORK}" == "graphiti" ]]; then
    FRAMEWORK_SERVICE_ARGS+=(
      --graphiti-neo4j-uri "${VMP_GRAPHITI_NEO4J_URI:-bolt://127.0.0.1:7687}"
      --graphiti-neo4j-user "${VMP_GRAPHITI_NEO4J_USER:-neo4j}"
      --graphiti-allow-destructive-reset
    )
  elif [[ "${FRAMEWORK}" == "letta" ]]; then
    FRAMEWORK_SERVICE_ARGS+=(
      --letta-base-url "${VMP_LETTA_BASE_URL:-http://127.0.0.1:8283}"
      --letta-embedding-base-url "${VMP_LETTA_EMBEDDING_BASE_URL:-http://127.0.0.1:8001/v1}"
      --letta-context-window "${VMP_LETTA_CONTEXT_WINDOW:-16384}"
    )
  fi
}
framework_service_args

require_completed_run() {
  local run_dir="$1"
  python - "${run_dir}/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(f"Missing run manifest: {path}")
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "completed":
    raise SystemExit(f"Run is not completed: {path}")
PY
}

create_split() {
  python scripts/create_longmemeval_split.py \
    --data "${DATA_PATH}" \
    --output "${SPLIT_PATH}" \
    --seed 42 \
    --dev-size 100 \
    --test-size 400
}

run_smoke() {
  python scripts/run_official_framework_smoke.py \
    --framework "${FRAMEWORK}" \
    --vllm-base-url "${VMP_LLM_BASE_URL}" \
    --vllm-model "${VMP_LLM_MODEL}" \
    --embedding-model "${EMBEDDING_MODEL}" \
    --embedding-dimension "${EMBEDDING_DIMENSION}" \
    --embedding-device "${EMBEDDING_DEVICE}" \
    --official-llm-max-tokens "${OFFICIAL_LLM_MAX_TOKENS}" \
    --official-llm-temperature "${OFFICIAL_LLM_TEMPERATURE}" \
    --output-dir "${AUDIT_DIR}" \
    "${FRAMEWORK_SERVICE_ARGS[@]}"
}

run_audit() {
  python scripts/audit_frameworks.py \
    --frameworks "${FRAMEWORK}" \
    --vllm-base-url "${VMP_LLM_BASE_URL}" \
    --llm-model "${VMP_LLM_MODEL}" \
    --embedding-model "${EMBEDDING_MODEL}" \
    --embedding-dimension "${EMBEDDING_DIMENSION}" \
    --official-llm-max-tokens "${OFFICIAL_LLM_MAX_TOKENS}" \
    --official-llm-temperature "${OFFICIAL_LLM_TEMPERATURE}" \
    --verification-dir "${AUDIT_DIR}" \
    --output-dir "${FRAMEWORK_AUDIT_OUTPUT}" \
    --require-main-table-eligible
}

run_candidates() {
  local resume_args=()
  if [[ "${RETRIEVAL_RESUME}" == "1" && -d "${CANDIDATE_RUN}" ]]; then
    resume_args+=(--resume)
  fi
  python scripts/run_longmemeval_retrieval.py \
    --data "${DATA_PATH}" \
    --split-manifest "${SPLIT_PATH}" \
    --split test \
    --methods "${METHOD}" \
    --top-k "${OUTPUT_TOP_K}" \
    --retrieval-depth "${CANDIDATE_POOL_COUNT}" \
    --ingestion-granularity session \
    --embedding-model "${EMBEDDING_MODEL}" \
    --embedding-device "${EMBEDDING_DEVICE}" \
    --embedding-cache-dir "${EMBEDDING_CACHE_DIR}" \
    --embedding-cache-db "${EMBEDDING_CACHE_DB}" \
    --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
    --prewarm-embeddings \
    --vllm-base-url "${VMP_LLM_BASE_URL}" \
    --vllm-model "${VMP_LLM_MODEL}" \
    --official-memory-infer \
    --official-llm-max-tokens "${OFFICIAL_LLM_MAX_TOKENS}" \
    --official-llm-temperature "${OFFICIAL_LLM_TEMPERATURE}" \
    --run-id "${CANDIDATE_RUN_ID}" \
    "${FRAMEWORK_SERVICE_ARGS[@]}" \
    "${resume_args[@]}"
}

run_rerank() {
  local resume_args=()
  if [[ "${RERANK_RESUME}" == "1" && -d "${RERANK_RUN}" ]]; then
    resume_args+=(--resume)
  fi
  python scripts/run_longmemeval_rerank.py \
    --source-run "${CANDIDATE_RUN}" \
    --run-id "${RERANK_RUN_ID}" \
    --methods "${METHOD}" \
    --base-url "${VMP_LLM_BASE_URL}" \
    --model "${VMP_LLM_MODEL}" \
    --candidate-count "${CANDIDATE_COUNT}" \
    --candidate-planner-version "${CANDIDATE_PLANNER_VERSION}" \
    --candidate-planner-rrf-k 60 \
    --candidate-planner-hierarchical-weight 0.8 \
    --output-top-k "${OUTPUT_TOP_K}" \
    --protected-top-n 3 \
    --ranked-output-count 10 \
    --max-candidate-chars 1200 \
    --max-excerpt-turns 4 \
    --candidate-excerpt-version "${CANDIDATE_EXCERPT_VERSION}" \
    --max-tokens 512 \
    --prompt-version "${SELECTOR_PROMPT_VERSION}" \
    --boundary-verification \
    --boundary-prompt-version "${BOUNDARY_PROMPT_VERSION}" \
    --boundary-protected-top-n 3 \
    --boundary-max-promotions 2 \
    --boundary-max-tokens 512 \
    --boundary-min-confidence high \
    --coverage-min-gain 0.25 \
    --coverage-need-weight 3.0 \
    --coverage-relevance-weight 1.5 \
    --coverage-diversity-weight 1.25 \
    --coverage-temporal-weight 1.25 \
    --coverage-rank-weight 0.08 \
    "${resume_args[@]}"
}

run_qa() {
  local resume_args=()
  if [[ "${QA_RESUME}" == "1" ]]; then
    resume_args+=(--resume)
  fi
  python scripts/run_longmemeval_qa.py \
    --retrieval-run "${RERANK_RUN}" \
    --methods "${RERANKED_METHOD}" \
    --base-url "${VMP_LLM_BASE_URL}" \
    --model "${VMP_LLM_MODEL}" \
    --top-k "${OUTPUT_TOP_K}" \
    --prompt-version "${READER_PROMPT_VERSION}" \
    --evidence-mode "${READER_EVIDENCE_MODE}" \
    --qa-subdir "${QA_SUBDIR}" \
    --protocol-selection-split dev \
    --temperature 0 \
    --top-p 1 \
    --max-tokens 256 \
    "${resume_args[@]}"
}

run_judge() {
  local resume_args=()
  if [[ "${JUDGE_RESUME}" == "1" ]]; then
    resume_args+=(--resume)
  fi
  python scripts/run_longmemeval_official_judge.py \
    --qa-run "${RERANK_RUN}/${QA_SUBDIR}" \
    --reference-data "${DATA_PATH}" \
    --methods "${RERANKED_METHOD}" \
    --output-subdir "${JUDGE_SUBDIR}" \
    --base-url "${VMP_LLM_BASE_URL}" \
    --model "${VMP_LLM_MODEL}" \
    "${resume_args[@]}"
}

show_status() {
  python - "${FRAMEWORK}" "${AUDIT_DIR}" "${CANDIDATE_RUN}" "${RERANK_RUN}" "${QA_SUBDIR}" "${JUDGE_SUBDIR}" <<'PY'
import json
import sys
from pathlib import Path

framework, audit_dir, candidate, rerank, qa_subdir, judge_subdir = sys.argv[1:]
paths = {
    "smoke": Path(audit_dir) / f"{framework}_smoke.json",
    "candidates": Path(candidate) / "manifest.json",
    "rerank": Path(rerank) / "manifest.json",
    "qa": Path(rerank) / qa_subdir / "manifest.json",
    "judge": Path(rerank) / qa_subdir / judge_subdir / "manifest.json",
}
for name, path in paths.items():
    status = "missing"
    if path.exists():
        try:
            status = str(json.loads(path.read_text(encoding="utf-8")).get("status", "unknown"))
        except (OSError, json.JSONDecodeError):
            status = "unreadable"
    print(f"{name:12} {status:10} {path}")
PY
}

case "${STAGE}" in
  smoke)
    log_stage "Verifying the pinned official adapter with local vLLM and BGE-M3."
    run_smoke
    ;;
  audit)
    log_stage "Enforcing package-version, endpoint, model, and smoke-receipt equality."
    run_audit
    ;;
  test_candidates)
    log_stage "Rechecking the official-adapter eligibility gate."
    run_audit
    create_split
    log_stage "Running native official memory ingestion/retrieval on frozen Test."
    log_stage "Each completed question is durably checkpointed; rerun this stage to resume."
    run_candidates
    python scripts/export_longmemeval_tables.py \
      --retrieval-run "${CANDIDATE_RUN}"
    ;;
  test_rerank)
    run_audit
    require_completed_run "${CANDIDATE_RUN}"
    log_stage "Applying the exact frozen VMP-v6.4 evidence-selection protocol."
    run_rerank
    python scripts/export_longmemeval_tables.py \
      --retrieval-run "${RERANK_RUN}"
    ;;
  test_qa)
    run_audit
    require_completed_run "${RERANK_RUN}"
    log_stage "Generating answers with the shared frozen QA-v2.1 reader."
    run_qa
    python scripts/export_longmemeval_cost.py \
      --retrieval-run "${RERANK_RUN}" \
      --qa-subdir "${QA_SUBDIR}"
    ;;
  test_judge)
    run_audit
    require_completed_run "${RERANK_RUN}/${QA_SUBDIR}"
    log_stage "Judging saved answers with the shared official-prompt local judge."
    run_judge
    ;;
  status)
    show_status
    ;;
  *)
    echo "Unknown STAGE=${STAGE}. Expected smoke, audit, test_candidates, test_rerank, test_qa, test_judge, or status." >&2
    exit 2
    ;;
esac
