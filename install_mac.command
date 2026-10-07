#!/bin/bash
# Installs everything Cell Analyzer needs (macOS).
# Double-click it -- or, if macOS refuses to open it, open Terminal, type  bash  (with a space),
# drag this file into the Terminal window and press Enter.
# Needs Miniforge (or Anaconda/Miniconda) installed first -- see USER_GUIDE.html, section 3.
cd "$(dirname "$0")" || exit 1
DIR="$(pwd)"

echo "== 1/3  Unblocking the launcher files =="
# files downloaded from the internet are 'quarantined' by macOS and may lose their execute bit
xattr -dr com.apple.quarantine "$DIR" 2>/dev/null
chmod +x "$DIR"/*.command "$DIR"/*.sh 2>/dev/null
echo "  ✓ done"

echo "== 2/3  Installing / updating the 'cell-analyzer' environment (5-15 min the first time) =="
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

echo "== 3/3  Creating 'Cell Analyzer.app' =="
APP="$DIR/Cell Analyzer.app"
LOG="\$HOME/Library/Logs/cell_analyzer_launcher.log"
rm -rf "$APP"
TMP_SCPT="${TMPDIR:-/tmp}/cell_analyzer_launcher_$$.applescript"
cat > "$TMP_SCPT" <<APPLESCRIPT
-- Launcher created by install_mac.command: runs start_mac.command from the folder the app is in
-- (or, if the app was moved elsewhere, from the folder it was created in).
on run
	set appPath to POSIX path of (path to me)
	set sh to "APP=" & quoted form of appPath & "; D=\$(dirname \"\$APP\"); if [ ! -d \"\$D/cell_analyzer\" ]; then D=" & quoted form of "$DIR" & "; fi; mkdir -p \"\$HOME/Library/Logs\"; cd \"\$D\" && /bin/bash ./start_mac.command >> \"$LOG\" 2>&1 || { tail -n 15 \"$LOG\" >&2; exit 1; }"
	try
		do shell script sh
	on error errMsg
		display dialog "Cell Analyzer could not be started:" & return & return & errMsg & return & return & "Full log: ~/Library/Logs/cell_analyzer_launcher.log" buttons {"OK"} default button 1 with icon stop
	end try
end run
APPLESCRIPT
if osacompile -o "$APP" "$TMP_SCPT" 2>/dev/null; then
  xattr -dr com.apple.quarantine "$APP" 2>/dev/null
  echo "  ✓ created: $APP"
  APP_OK=1
else
  echo "  ⚠ could not create the app (osacompile failed) -- use start_mac.command instead."
fi
rm -f "$TMP_SCPT"

OLD_PY="$(find_env_python "$OLD_ENV_NAME")"
if [ -n "$OLD_PY" ]; then
  echo ""
  echo "  Note: the old environment '$OLD_ENV_NAME' (from before the rename) is no longer needed."
  echo "  To free disk space:  \"$CONDA\" env remove -n $OLD_ENV_NAME"
fi

echo ""
echo "  ✓ Done."
if [ -n "$APP_OK" ]; then
  echo "  Start the program by double-clicking 'Cell Analyzer.app' in this folder"
  echo "  (tip: drag it to the Dock). start_mac.command works too."
else
  echo "  Start the program with start_mac.command."
fi
pause_if_terminal
