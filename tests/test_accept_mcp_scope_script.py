"""Real acceptance entrypoint for strict MCP paper scope."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_accept_mcp_scope_script_uses_real_sqlite(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/accept_mcp_scope.py",
            "--workdir",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {
        "accepted_scope": ["paper:public", "paper:owned"],
        "atomic_failure_ids": ["paper:missing", "paper:pending", "paper:private"],
        "database_exists": True,
        "status": "accepted",
    }
