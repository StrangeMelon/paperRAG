"""Synchronous process-level resource guards."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

from .errors import RetrievalBusyError

_GPU_COMPONENTS = {"embedding", "reranker", "vision", "mineru"}
_ACTIVE_GUARDS: ResourceGuards | None = None
_ACTIVE_LOCK = threading.Lock()


class ResourceGuards:
    def __init__(self, limits: dict[str, int]) -> None:
        if any(value < 1 for value in limits.values()):
            raise ValueError("resource limits must be positive")
        self._semaphores = {
            name: threading.BoundedSemaphore(value) for name, value in limits.items()
        }

    @contextmanager
    def hold(self, name: str, *, timeout: float | None = None) -> Iterator[None]:
        if name not in self._semaphores:
            raise KeyError(name)
        names = ["gpu_total", name] if name in _GPU_COMPONENTS else [name]
        acquired: list[str] = []
        deadline = None if timeout is None else time.monotonic() + timeout
        try:
            for resource in names:
                if resource not in self._semaphores:
                    raise KeyError(resource)
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                if not self._semaphores[resource].acquire(timeout=remaining):
                    raise RetrievalBusyError(retry_after=1)
                acquired.append(resource)
            yield
        finally:
            for resource in reversed(acquired):
                self._semaphores[resource].release()


def configure_resource_guards(limits: dict[str, int]) -> ResourceGuards:
    global _ACTIVE_GUARDS
    guards = ResourceGuards(limits)
    with _ACTIVE_LOCK:
        _ACTIVE_GUARDS = guards
    return guards


@contextmanager
def hold_resource(name: str, *, timeout: float | None = None) -> Iterator[None]:
    guards = _ACTIVE_GUARDS
    if guards is None:
        yield
        return
    with guards.hold(name, timeout=timeout):
        yield


__all__ = ["ResourceGuards", "configure_resource_guards", "hold_resource"]
