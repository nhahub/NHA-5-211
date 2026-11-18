"""Launcher script for the SMART IDS v3 Streamlit SOC Dashboard."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    app_file = root / "app.py"
    env = os.environ.copy()
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_file), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd, cwd=str(root), env=env))


if __name__ == "__main__":
    main()
