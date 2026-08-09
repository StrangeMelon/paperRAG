"""Real concurrency acceptance entrypoint for MCP runtime."""

from __future__ import annotations

import json
import subprocess
import sys


def test_accept_mcp_runtime_script_runs_real_threads() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/accept_mcp_runtime.py"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["status"] == "accepted"
    assert payload["distinct_worker_threads"] == 2
    assert payload["event_loop_heartbeats"] >= 3
    assert payload["timeout_held_capacity"] is True
