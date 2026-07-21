#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible entry point. New experiments use the VMP-v4 pipeline.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${PROJECT_ROOT}/scripts/run_vmp_v4_experiment.sh" "$@"
