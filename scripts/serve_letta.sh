#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${VMP_LETTA_CONTAINER:-vmp-letta}"
IMAGE="${VMP_LETTA_IMAGE:-letta/letta:0.16.8}"
VOLUME="${VMP_LETTA_VOLUME:-vmp-letta-data}"
LABEL="vmp-memos.letta.dedicated=true"
BASE_URL="${VMP_LETTA_BASE_URL:-http://127.0.0.1:8283}"
VLLM_ROOT="${VMP_LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
VLLM_ROOT="${VLLM_ROOT%/v1}"
EMBEDDING_ROOT="${VMP_LETTA_EMBEDDING_BASE_URL:-http://127.0.0.1:8001/v1}"
EMBEDDING_ROOT="${EMBEDDING_ROOT%/v1}"
LOCAL_API_KEY="${VMP_LLM_API_KEY:-local-vllm-key}"
CURL_AUTH=()

if [[ -n "${VMP_LLM_API_KEY:-}" ]]; then
  CURL_AUTH=(-H "Authorization: Bearer ${VMP_LLM_API_KEY}")
fi
if ! curl --fail --silent "${CURL_AUTH[@]}" "${VLLM_ROOT}/v1/models" >/dev/null; then
  echo "vLLM is not reachable at ${VLLM_ROOT}/v1." >&2
  exit 2
fi
if ! curl --fail --silent "${EMBEDDING_ROOT}/health" >/dev/null; then
  echo "Embedding server is not reachable at ${EMBEDDING_ROOT}." >&2
  exit 2
fi

wait_for_letta() {
  for _ in $(seq 1 90); do
    if curl --fail --silent "${BASE_URL}/v1/health" >/dev/null 2>&1; then
      echo "Letta is ready at ${BASE_URL}."
      return 0
    fi
    sleep 2
  done
  echo "Letta did not become ready within 180 seconds." >&2
  return 4
}

if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
  dedicated="$(
    docker container inspect \
      --format '{{ index .Config.Labels "vmp-memos.letta.dedicated" }}' \
      "$CONTAINER"
  )"
  if [[ "$dedicated" != "true" ]]; then
    echo "Refusing to use unlabelled container: $CONTAINER" >&2
    exit 3
  fi
  configured_image="$(
    docker container inspect --format '{{.Config.Image}}' "$CONTAINER"
  )"
  if [[ "$configured_image" != "$IMAGE" ]]; then
    echo "Container image is ${configured_image}; expected pinned ${IMAGE}." >&2
    exit 3
  fi
  if [[ "$(docker container inspect --format '{{.State.Running}}' "$CONTAINER")" == "true" ]]; then
    echo "Dedicated Letta container is already running: $CONTAINER"
  else
    docker start "$CONTAINER" >/dev/null
    echo "Started dedicated Letta container: $CONTAINER"
  fi
  wait_for_letta
  exit 0
fi

docker run -d \
  --name "$CONTAINER" \
  --label "$LABEL" \
  --restart unless-stopped \
  --network host \
  --env "VLLM_API_BASE=${VLLM_ROOT}" \
  --env "OPENAI_API_KEY=${LOCAL_API_KEY}" \
  --env "LETTA_DISABLE_TRACING=true" \
  --env "LETTA_LLM_REQUEST_TIMEOUT_SECONDS=600" \
  --volume "${VOLUME}:/var/lib/postgresql/data" \
  "$IMAGE" >/dev/null

echo "Started dedicated Letta container: $CONTAINER ($IMAGE)"
wait_for_letta
