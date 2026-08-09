"""Async MCP runtime admission and timeout contracts."""

from __future__ import annotations

import asyncio
import time

import pytest

from paper_rag.mcp.errors import RetrievalBusyError, RetrievalTimeoutError
from paper_rag.mcp.runtime import McpRuntime


def test_runtime_runs_sync_work_off_event_loop() -> None:
    async def scenario() -> None:
        runtime = McpRuntime(max_running=1, max_queued=1, admission_timeout=1)
        started = time.perf_counter()
        result = await asyncio.gather(
            runtime.run_sync(lambda: (time.sleep(0.05), "ok")[1]),
            asyncio.sleep(0),
        )
        assert result[0] == "ok"
        assert time.perf_counter() - started < 1

    asyncio.run(scenario())


def test_runtime_rejects_over_capacity() -> None:
    async def scenario() -> None:
        runtime = McpRuntime(max_running=1, max_queued=0, admission_timeout=0.01)
        first = asyncio.create_task(runtime.run_sync(lambda: time.sleep(0.1)))
        await asyncio.sleep(0.01)
        with pytest.raises(RetrievalBusyError):
            await runtime.run_sync(lambda: None)
        await first

    asyncio.run(scenario())


def test_runtime_timeout_maps_without_releasing_before_worker_finishes() -> None:
    async def scenario() -> None:
        runtime = McpRuntime(max_running=1, max_queued=0, admission_timeout=0.01)
        with pytest.raises(RetrievalTimeoutError):
            await runtime.run_sync(lambda: time.sleep(0.1), timeout=0.01)
        with pytest.raises(RetrievalBusyError):
            await runtime.run_sync(lambda: None)
        await asyncio.sleep(0.12)
        assert await runtime.run_sync(lambda: "released") == "released"

    asyncio.run(scenario())
