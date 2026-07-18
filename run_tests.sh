#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ARGS=("$@")
if (( ${#ARGS[@]} == 0 )); then
    ARGS=("tests/")
fi
    
exec python -m pytest "${ARGS[@]}"
