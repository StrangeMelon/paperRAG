"""Async admission and bounded thread execution for MCP requests."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import partial
from typing import Any

import anyio

from .errors import RetrievalBusyError, RetrievalTimeoutError


class McpRuntime:
    def __init__(
        self,
        *,
        max_running: int = 2,
        max_queued: int = 8,
        admission_timeout: float = 2.0,
        thread_tokens: int = 8,
    ) -> None:
        if max_running < 1 or max_queued < 0 or thread_tokens < 1:
            raise ValueError("runtime capacities are invalid")
        self._running = asyncio.Semaphore(max_running)
        self._max_queued = max_queued
        self._queued = 0
        self._queue_lock = asyncio.Lock()
        self._admission_timeout = admission_timeout
        self._thread_limiter = anyio.CapacityLimiter(thread_tokens)

    async def _acquire(self) -> None:
        async with self._queue_lock:
            if self._running.locked() and self._queued >= self._max_queued:
                raise RetrievalBusyError(retry_after=1)
            if self._running.locked():
                self._queued += 1
        try:
            await asyncio.wait_for(self._running.acquire(), self._admission_timeout)
        except asyncio.TimeoutError as exc:
            raise RetrievalBusyError(retry_after=1) from exc
        finally:
            async with self._queue_lock:
                if self._queued:
                    self._queued -= 1

    def _release(self) -> None:
        self._running.release()

    async def run_sync(
        self,
        function: Callable[..., Any],
        *args: Any,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Any:
        await self._acquire()
        worker = asyncio.create_task(
            anyio.to_thread.run_sync(
                partial(function, *args, **kwargs),
                limiter=self._thread_limiter,
            )
        )
        try:
            if timeout is None:
                result = await worker
            else:
                result = await asyncio.wait_for(asyncio.shield(worker), timeout)
        except asyncio.TimeoutError as exc:
            worker.add_done_callback(lambda _: self._release())
            raise RetrievalTimeoutError() from exc
        except BaseException:
            self._release()
            raise
        else:
            self._release()
            return result
