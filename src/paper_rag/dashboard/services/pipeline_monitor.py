"""Persistent history for ingestion and retrieval pipeline timings only."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .jsonl_store import JsonlStore


class PipelineMonitorStore:
    def __init__(self, path: str | Path) -> None:
        self._store = JsonlStore(path, id_field="run_id")

    def append(self, record: dict[str, Any]) -> None:
        self._store.append(record)

    def get(self, run_id: str) -> dict[str, Any] | None:
        return self._store.get(run_id)

    def list(self, *, pipeline: str = "all", limit: int | None = None) -> list[dict[str, Any]]:
        records = self._store.load()
        if pipeline != "all":
            records = [item for item in records if item.get("pipeline") == pipeline]
        return records[:limit] if limit else records

    def delete(self, run_id: str) -> bool:
        return self._store.delete(run_id)


def _default_store() -> PipelineMonitorStore:
    from ... import config as cfg

    return PipelineMonitorStore(
        Path(cfg.load().paths.data_root) / "dashboard" / "pipeline_monitor.jsonl"
    )


def record_ingestion_run(
    *, paper_id: str, status: str, timings_seconds: dict[str, float], metadata: dict[str, Any]
) -> dict[str, Any]:
    record = {
        "run_id": f"ingest-{uuid4().hex[:12]}",
        "pipeline": "ingestion",
        "paper_id": paper_id,
        "status": status,
        "created_at": datetime.now(UTC).isoformat(),
        "timings_ms": {
            key: round(float(value) * 1000, 1) for key, value in timings_seconds.items()
        },
        "metadata": metadata,
    }
    if not _monitor_disabled():
        _default_store().append(record)
    return record


def record_retrieval_run(
    *, query: str, status: str, timings_ms: dict[str, float], metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    record = {
        "run_id": f"retrieve-{uuid4().hex[:12]}",
        "pipeline": "retrieval",
        "query": query,
        "status": status,
        "created_at": datetime.now(UTC).isoformat(),
        "timings_ms": {key: round(float(value), 1) for key, value in timings_ms.items()},
        "metadata": metadata or {},
    }
    if not _monitor_disabled():
        _default_store().append(record)
    return record


def _monitor_disabled() -> bool:
    return os.environ.get("PAPER_RAG_DISABLE_PIPELINE_MONITOR", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
