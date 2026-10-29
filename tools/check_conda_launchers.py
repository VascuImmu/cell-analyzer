"""Run by install_windows.bat with the BASE conda python.

Miniforge 26.9.4-0 for Windows ships a broken record of the package 'conda-launchers'
(its file list in conda-meta is empty). conda 26.9 needs that list to find cli-64.exe, so creating
any environment fails with
    FileNotFoundError: Could not find 'share/conda-launchers/cli-64.exe' in the conda-launchers package
Exit code 1 = the record is broken (the installer then reinstalls conda-launchers), 0 = fine / not present.
See https://github.com/conda-forge/miniforge/pull/914
"""
import glob
import json
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else sys.prefix
records = glob.glob(os.path.join(root, "conda-meta", "conda-launchers-*.json"))
if not records:
    sys.exit(0)
try:
    with open(records[0], encoding="utf-8") as f:
        rec = json.load(f)
except Exception:
    sys.exit(1)
paths = (rec.get("paths_data") or {}).get("paths") or []
if any(str(p.get("_path", "")).replace("\\", "/").endswith("cli-64.exe") for p in paths):
    sys.exit(0)
print("  conda-launchers record is incomplete (known Miniforge 26.9.4 bug)")
sys.exit(1)
