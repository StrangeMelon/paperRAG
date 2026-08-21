#!/usr/bin/env python3
"""Launch the Paper RAG Streamlit dashboard."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding exported variables."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--address", default="127.0.0.1")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    _load_dotenv(project_root / ".env")
    app_path = project_root / "src" / "paper_rag" / "dashboard" / "app.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.port),
        "--server.address",
        args.address,
        "--server.headless",
        "true",
    ]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
