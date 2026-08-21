"""Small durable JSONL store shared by dashboard histories."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any


class JsonlStore:
    def __init__(self, path: str | Path, *, id_field: str) -> None:
        self.path = Path(path)
        self.id_field = id_field
        self._lock = Lock()

    def append(self, record: dict[str, Any]) -> None:
        if not record.get(self.id_field):
            raise ValueError(f"record requires {self.id_field}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict) and record.get(self.id_field):
                    records.append(record)
        records.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        return records

    def get(self, record_id: str) -> dict[str, Any] | None:
        return next((item for item in self.load() if item.get(self.id_field) == record_id), None)

    def delete(self, record_id: str) -> bool:
        records = self.load()
        remaining = [item for item in records if item.get(self.id_field) != record_id]
        if len(records) == len(remaining):
            return False
        self._rewrite(remaining)
        return True

    def clear(self) -> int:
        records = self.load()
        self._rewrite([])
        return len(records)

    def _rewrite(self, records: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("w", encoding="utf-8") as handle:
            for record in reversed(records):
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
