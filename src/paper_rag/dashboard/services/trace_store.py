"""Persistent query trace history used by the dashboard."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .jsonl_store import JsonlStore


class QueryTraceStore:
    def __init__(self, path: str | Path) -> None:
        self._store = JsonlStore(path, id_field="trace_id")

    def append(self, record: dict[str, Any]) -> None:
        self._store.append(record)

    def get(self, trace_id: str) -> dict[str, Any] | None:
        return self._store.get(trace_id)

    def list(
        self,
        *,
        keyword: str | None = None,
        status: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        records = self._store.load()
        if keyword:
            needle = keyword.casefold()
            records = [
                item
                for item in records
                if needle in str(item.get("query", "")).casefold()
                or needle in str(item.get("trace_id", "")).casefold()
            ]
        if status and status != "all":
            records = [item for item in records if item.get("status") == status]
        if start_date:
            records = [item for item in records if _record_date(item) >= start_date]
        if end_date:
            records = [item for item in records if _record_date(item) <= end_date]
        return records[:limit] if limit else records

    def delete(self, trace_id: str) -> bool:
        return self._store.delete(trace_id)

    def clear(self) -> int:
        return self._store.clear()

    def export_json(self, **filters: Any) -> str:
        return json.dumps(self.list(**filters), ensure_ascii=False, indent=2, default=str)


def _record_date(record: dict[str, Any]) -> date:
    raw = str(record.get("created_at", ""))
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
        return date.min
