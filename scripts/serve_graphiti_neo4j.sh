#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${VMP_GRAPHITI_NEO4J_CONTAINER:-vmp-graphiti-neo4j}"
IMAGE="${VMP_GRAPHITI_NEO4J_IMAGE:-neo4j:5.26-community}"
USER="${VMP_GRAPHITI_NEO4J_USER:-neo4j}"
PASSWORD="${VMP_GRAPHITI_NEO4J_PASSWORD:-}"
HTTP_PORT="${VMP_GRAPHITI_NEO4J_HTTP_PORT:-7474}"
BOLT_PORT="${VMP_GRAPHITI_NEO4J_BOLT_PORT:-7687}"
VOLUME="${VMP_GRAPHITI_NEO4J_VOLUME:-${CONTAINER}-data}"
LABEL="vmp-memos.graphiti.dedicated=true"

wait_for_neo4j() {
  for _ in $(seq 1 60); do
    if docker exec "$CONTAINER" \
      cypher-shell -u "$USER" -p "$PASSWORD" "RETURN 1;" >/dev/null 2>&1; then
      echo "Neo4j is ready."
      return 0
    fi
    sleep 2
  done
  echo "Neo4j did not become ready within 120 seconds." >&2
  return 4
}

if [[ -z "$PASSWORD" ]]; then
  echo "Set VMP_GRAPHITI_NEO4J_PASSWORD before starting Neo4j." >&2
  exit 2
fi

if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
  dedicated="$(
    docker container inspect \
      --format '{{ index .Config.Labels "vmp-memos.graphiti.dedicated" }}' \
      "$CONTAINER"
  )"
  if [[ "$dedicated" != "true" ]]; then
    echo "Refusing to use unlabelled container: $CONTAINER" >&2
    echo "Graphiti benchmark resets delete every node in the connected database." >&2
    exit 3
  fi
  if [[ "$(docker container inspect --format '{{.State.Running}}' "$CONTAINER")" == "true" ]]; then
    echo "Dedicated Graphiti Neo4j container is already running: $CONTAINER"
  else
    docker start "$CONTAINER" >/dev/null
    echo "Started dedicated Graphiti Neo4j container: $CONTAINER"
  fi
  wait_for_neo4j
  exit 0
fi

docker run -d \
  --name "$CONTAINER" \
  --label "$LABEL" \
  --restart unless-stopped \
  --publish "${HTTP_PORT}:7474" \
  --publish "${BOLT_PORT}:7687" \
  --env "NEO4J_AUTH=${USER}/${PASSWORD}" \
  --volume "${VOLUME}:/data" \
  "$IMAGE" >/dev/null

echo "Started dedicated Graphiti Neo4j container: $CONTAINER"
echo "Bolt endpoint: bolt://127.0.0.1:${BOLT_PORT}"
echo "WARNING: VMP-MemOS Graphiti runs clear every node before each question."
wait_for_neo4j
