#!/usr/bin/env python3
"""Real thread and admission acceptance for the MCP runtime."""

from __future__ import annotations

import asyncio
import json
import threading
import time

from paper_rag.mcp.errors import RetrievalBusyError, RetrievalTimeoutError
from paper_rag.mcp.runtime import McpRuntime


async def _run() -> dict:
    runtime = McpRuntime(max_running=2, max_queued=1, admission_timeout=0.2, thread_tokens=2)
    barrier = threading.Barrier(2)

    def work() -> int:
        barrier.wait(timeout=2)
        time.sleep(0.1)
        return threading.get_ident()

    started = time.perf_counter()
    tasks = [asyncio.create_task(runtime.run_sync(work)) for _ in range(2)]
    heartbeats = 0
    while not all(task.done() for task in tasks):
        await asyncio.sleep(0.02)
        heartbeats += 1
    worker_ids = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - started
    assert elapsed < 0.3, f"workers did not run concurrently: {elapsed:.3f}s"
    assert len(set(worker_ids)) == 2

    timeout_runtime = McpRuntime(
        max_running=1,
        max_queued=0,
        admission_timeout=0.02,
        thread_tokens=1,
    )
    try:
        await timeout_runtime.run_sync(lambda: time.sleep(0.15), timeout=0.02)
    except RetrievalTimeoutError:
        pass
    else:
        raise AssertionError("slow worker did not time out")

    try:
        await timeout_runtime.run_sync(lambda: None)
    except RetrievalBusyError:
        timeout_held_capacity = True
    else:
        raise AssertionError("timed-out worker released capacity before it finished")
    await asyncio.sleep(0.16)
    assert await timeout_runtime.run_sync(lambda: "released") == "released"

    return {
        "status": "accepted",
        "distinct_worker_threads": len(set(worker_ids)),
        "event_loop_heartbeats": heartbeats,
        "parallel_elapsed_ms": round(elapsed * 1000),
        "timeout_held_capacity": timeout_held_capacity,
    }


def main() -> int:
    print(json.dumps(asyncio.run(_run()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
