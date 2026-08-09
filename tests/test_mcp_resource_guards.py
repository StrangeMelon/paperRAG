"""Process-level resource semaphore contracts."""

from __future__ import annotations

import threading

import pytest

from paper_rag.mcp.errors import RetrievalBusyError
from paper_rag.mcp.resource_guards import ResourceGuards


def test_component_guard_limits_concurrent_execution() -> None:
    guards = ResourceGuards({"gpu_total": 1, "embedding": 1})
    entered = threading.Event()
    release = threading.Event()

    def holder() -> None:
        with guards.hold("embedding"):
            entered.set()
            release.wait(timeout=2)

    thread = threading.Thread(target=holder)
    thread.start()
    assert entered.wait(timeout=1)
    with pytest.raises(RetrievalBusyError), guards.hold("embedding", timeout=0.01):
        pass
    release.set()
    thread.join(timeout=2)


def test_unknown_resource_is_rejected() -> None:
    guards = ResourceGuards({"gpu_total": 1})

    with pytest.raises(KeyError), guards.hold("missing"):
        pass
