#!/bin/bash
# Starts Cell Analyzer (macOS). Double-click, or in Terminal:  bash start_mac.command
# After running install_mac.command once you can also double-click "Cell Analyzer.app".
cd "$(dirname "$0")" || exit 1
source tools/find_conda.sh
PY="$(find_env_python)" || PY="$(find_env_python "$OLD_ENV_NAME")"
if [ -z "$PY" ]; then
  echo "  ✗ The 'cell-analyzer' environment was not found -- run install_mac.command first."
  pause_if_terminal; exit 1
fi
echo "Starting Cell Analyzer with $PY"
echo "(Keep this window open while you work -- closing it closes the program.)"
"$PY" -m cell_analyzer
