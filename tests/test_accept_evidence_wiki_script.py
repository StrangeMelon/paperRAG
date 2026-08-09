"""Real SQLite acceptance for evidence Wiki enrichment."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_accept_evidence_wiki_uses_real_sqlite(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "scripts/accept_evidence_wiki.py", "--workdir", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {
        "entry": {
            "definition": "A state space sequence model used as background.",
            "name": "State Space Model",
        },
        "status": "accepted",
    }
