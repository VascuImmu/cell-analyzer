#!/bin/bash
# Installs everything Cell Analyzer needs (Linux).
# Run in a terminal:  bash install_linux.sh
# Needs Miniforge (or Anaconda/Miniconda) installed first -- see USER_GUIDE.html, section 3.
cd "$(dirname "$0")" || exit 1
DIR="$(pwd)"

echo "== Making the launcher files executable =="
# files downloaded from the internet are 'quarantined' by macOS and may lose their execute bit
chmod +x "$DIR"/*.command "$DIR"/*.sh 2>/dev/null
echo "  ✓ done"

echo "== Installing / updating the 'cell-analyzer' environment (5-15 min the first time) =="
source tools/find_conda.sh
if [ -z "$CONDA_BASE" ]; then
  echo ""
  echo "  ✗ Conda was not found. Install Miniforge first (USER_GUIDE.html, section 3), then run this again."
  pause_if_terminal; exit 1
fi
CONDA="$CONDA_BASE/bin/conda"
echo "  using conda in $CONDA_BASE"

create_env() { "$CONDA" env create -y -f environment.yml 2>/dev/null || "$CONDA" env create -f environment.yml; }
remove_env() { "$CONDA" env remove -y -p "$1" 2>/dev/null || "$CONDA" env remove -y -n "$ENV_NAME"; }
check_env()  { local py; py="$(find_env_python)"; [ -n "$py" ] && "$py" tools/check_env.py; }

EXISTING="$(find_env_python)"
if [ -n "$EXISTING" ] && [ "$1" = "--fresh" ]; then
  echo "  --fresh: removing the old environment first"
  remove_env "$(dirname "$(dirname "$EXISTING")")"; EXISTING=""
fi
if [ -n "$EXISTING" ]; then
  ENV_PREFIX="$(dirname "$(dirname "$EXISTING")")"
  echo "  updating existing environment $ENV_PREFIX"
  "$CONDA" env update -p "$ENV_PREFIX" -f environment.yml --prune
else
  create_env
fi

echo "  checking the environment..."
if ! check_env; then
  if [ -n "$EXISTING" ]; then
    echo "  The existing environment is damaged -- rebuilding it from scratch (5-15 min)..."
    remove_env "$(dirname "$(dirname "$(find_env_python)")")"
    create_env
    echo "  checking the environment again..."
  fi
  if ! check_env; then
    echo ""; echo "  ✗ Installing the environment failed -- see the messages above (USER_GUIDE.html, 'Troubleshooting')."
    pause_if_terminal; exit 1
  fi
fi

OLD_PY="$(find_env_python "$OLD_ENV_NAME")"
if [ -n "$OLD_PY" ]; then
  echo ""
  echo "  Note: the old environment '$OLD_ENV_NAME' (from before the rename) is no longer needed."
  echo "  To free disk space:  \"$CONDA\" env remove -n $OLD_ENV_NAME"
fi

echo ""
echo "  ✓ Done."
echo "  Start the program with:  bash start_linux.sh"
pause_if_terminal
