#!/usr/bin/env python3
"""No-mock stdio acceptance through langchain-mcp-adapters."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any


async def _accept(
    *,
    invalid_scope: bool,
    query: str | None,
    paper_ids: list[str] | None,
    server_timeout_sec: float,
) -> dict[str, Any]:
    from dotenv import load_dotenv
    from langchain_mcp_adapters.client import MultiServerMCPClient

    load_dotenv(override=False)
    client = MultiServerMCPClient(
        {
            "paper-rag": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [
                    "-m",
                    "paper_rag.mcp.server",
                    "--retrieval-timeout-sec",
                    str(server_timeout_sec),
                ],
            }
        }
    )
    tools = await client.get_tools(server_name="paper-rag")
    names = [tool.name for tool in tools]
    assert names == ["paper_retrieve_evidence"], names
    tool = tools[0]
    result: dict[str, Any] = {
        "status": "accepted",
        "tools": names,
        "public_arguments": sorted(tool.args),
    }

    if invalid_scope:
        try:
            await tool.ainvoke(
                {
                    "query": "scope validation acceptance",
                    "paper_ids": ["paper:definitely-missing"],
                }
            )
        except Exception as exc:
            assert "invalid_paper_scope" in str(exc), str(exc)
            result["invalid_scope_error"] = True
        else:
            raise AssertionError("invalid paper scope unexpectedly succeeded")

    if query:
        raw = await tool.ainvoke({"query": query, "paper_ids": paper_ids})
        payload = json.loads(raw) if isinstance(raw, str) else raw
        assert set(payload) in (
            {"decision", "retrieval_id", "evidence", "wiki"},
            {"decision", "retrieval_id", "evidence"},
        )
        result["call"] = payload
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--invalid-scope", action="store_true")
    parser.add_argument("--query")
    parser.add_argument("--paper-id", action="append", dest="paper_ids")
    parser.add_argument("--server-timeout-sec", type=float, default=90)
    args = parser.parse_args(argv)
    result = asyncio.run(
        _accept(
            invalid_scope=args.invalid_scope,
            query=args.query,
            paper_ids=args.paper_ids,
            server_timeout_sec=args.server_timeout_sec,
        )
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
