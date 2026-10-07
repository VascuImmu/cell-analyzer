#!/bin/bash
# Starts Cell Analyzer (Linux). Run in a terminal:  bash start_linux.sh
cd "$(dirname "$0")" || exit 1
source tools/find_conda.sh
PY="$(find_env_python)" || PY="$(find_env_python "$OLD_ENV_NAME")"
if [ -z "$PY" ]; then
  echo "  ✗ The 'cell-analyzer' environment was not found -- run install_linux.sh first."
  pause_if_terminal; exit 1
fi
echo "Starting Cell Analyzer with $PY"
echo "(Keep this window open while you work -- closing it closes the program.)"
"$PY" -m cell_analyzer
