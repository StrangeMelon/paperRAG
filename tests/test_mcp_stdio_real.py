"""No-mock stdio acceptance through langchain-mcp-adapters."""

from __future__ import annotations

import json
import subprocess
import sys


def test_langchain_adapter_discovers_tool_and_receives_scope_error() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/accept_mcp_stdio.py", "--invalid-scope"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["status"] == "accepted"
    assert payload["tools"] == ["paper_retrieve_evidence"]
    assert payload["invalid_scope_error"] is True
    assert payload["public_arguments"] == [
        "include_wiki",
        "max_evidence",
        "paper_ids",
        "query",
        "wiki_max_entries",
    ]
