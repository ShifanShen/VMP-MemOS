#!/usr/bin/env bash
set -euo pipefail

MODEL="${VMP_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
HOST="${VMP_VLLM_HOST:-0.0.0.0}"
PORT="${VMP_VLLM_PORT:-8000}"
DTYPE="${VMP_VLLM_DTYPE:-auto}"
API_KEY="${VMP_LLM_API_KEY:-}"

cmd=(vllm serve "$MODEL" --host "$HOST" --port "$PORT" --dtype "$DTYPE")

if [[ "${VMP_VLLM_ENABLE_TOOL_CALLING:-1}" == "1" ]]; then
  cmd+=(--enable-auto-tool-choice)
  cmd+=(--tool-call-parser "${VMP_VLLM_TOOL_CALL_PARSER:-hermes}")
fi

if [[ -n "$API_KEY" ]]; then
  cmd+=(--api-key "$API_KEY")
fi

if [[ -n "${VMP_VLLM_GPU_MEMORY_UTILIZATION:-}" ]]; then
  cmd+=(--gpu-memory-utilization "$VMP_VLLM_GPU_MEMORY_UTILIZATION")
fi

if [[ -n "${VMP_VLLM_MAX_MODEL_LEN:-}" ]]; then
  cmd+=(--max-model-len "$VMP_VLLM_MAX_MODEL_LEN")
fi

if [[ -n "${VMP_VLLM_TENSOR_PARALLEL_SIZE:-}" ]]; then
  cmd+=(--tensor-parallel-size "$VMP_VLLM_TENSOR_PARALLEL_SIZE")
fi

echo "Starting vLLM OpenAI-compatible server:"
printf ' %q' "${cmd[@]}"
echo
exec "${cmd[@]}" ${VMP_VLLM_EXTRA_ARGS:-}
