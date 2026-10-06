# Shared helper for the macOS/Linux install/start files (sourced, not run).
#   CONDA_BASE   the conda installation to use for installing
#   find_env_python [name] -> prints the python of the 'cell-analyzer' environment (or [name]), searching ALL conda installs
#                       (important when several are installed, e.g. miniforge3 AND miniconda3)
# Double-clicked scripts don't get the PATH of your Terminal, so we look in the usual places.

ENV_NAME="cell-analyzer"
OLD_ENV_NAME="cellpipeline"   # name used before the rename to Cell Analyzer
CANDIDATE_BASES=(
  "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/.anaconda"
  "$HOME/opt/miniforge3" "$HOME/opt/miniconda3" "$HOME/opt/anaconda3"
  "/opt/homebrew/Caskroom/miniforge/base" "/opt/homebrew/Caskroom/miniconda/base"
  "/usr/local/Caskroom/miniforge/base" "/usr/local/Caskroom/miniconda/base"
  "/opt/miniforge3" "/opt/miniconda3" "/opt/anaconda3" "/usr/local/miniconda3" "/usr/local/anaconda3"
)

CONDA_BASE=""
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null)"
fi
if [ -z "$CONDA_BASE" ] || [ ! -d "$CONDA_BASE" ]; then
  for c in "${CANDIDATE_BASES[@]}"; do
    if [ -x "$c/bin/conda" ]; then CONDA_BASE="$c"; break; fi
  done
fi

find_env_python() {
  local ENV_NAME="${1:-$ENV_NAME}"
  local b
  for b in "$CONDA_BASE" "${CANDIDATE_BASES[@]}"; do
    if [ -n "$b" ] && [ -x "$b/envs/$ENV_NAME/bin/python" ]; then echo "$b/envs/$ENV_NAME/bin/python"; return 0; fi
  done
  if [ -x "$HOME/.conda/envs/$ENV_NAME/bin/python" ]; then echo "$HOME/.conda/envs/$ENV_NAME/bin/python"; return 0; fi
  # last resort: ask every conda we can find where its environments are
  for b in "$CONDA_BASE" "${CANDIDATE_BASES[@]}"; do
    if [ -n "$b" ] && [ -x "$b/bin/conda" ]; then
      local p
      p="$("$b/bin/conda" env list 2>/dev/null | awk -v n="$ENV_NAME" '$1==n {print $NF}' | head -1)"
      if [ -n "$p" ] && [ -x "$p/bin/python" ]; then echo "$p/bin/python"; return 0; fi
    fi
  done
  return 1
}

pause_if_terminal() {
  # only wait for Enter when a person is looking at a Terminal window
  if [ -t 0 ]; then read -r -p "  Press Enter to close." _; fi
}
