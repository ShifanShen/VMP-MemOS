#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VMP_EXTRAS="${VMP_EXTRAS:-dev}"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[${VMP_EXTRAS}]"
python scripts/init_workspace.py
python -m pytest

echo "VMP-MemOS environment is ready with extras: ${VMP_EXTRAS}."
