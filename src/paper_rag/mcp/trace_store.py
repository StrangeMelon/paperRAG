"""Thread-safe in-process TTL/LRU retrieval trace store."""

from __future__ import annotations

import copy
import re
import time
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from threading import RLock
from typing import Any

from ..rag.evidence_retrieval import Principal, RetrievalExecution
from .errors import PermissionDeniedError, RetrievalExpiredError

_SENSITIVE_KEY = re.compile(
    r"(?:api[_-]?key|secret|password|prompt|source[_-]?path|asset[_-]?path)", re.I
)


def _scrub(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _scrub(item)
            for key, item in value.items()
            if not _SENSITIVE_KEY.search(str(key))
        }
    if isinstance(value, list):
        return [_scrub(item) for item in value]
    if isinstance(value, tuple):
        return [_scrub(item) for item in value]
    return value


class RetrievalTraceStore:
    def __init__(
        self,
        *,
        ttl_sec: float = 1800,
        max_entries: int = 1000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_sec <= 0 or max_entries < 1:
            raise ValueError("trace ttl and capacity must be positive")
        self.ttl_sec = ttl_sec
        self.max_entries = max_entries
        self._clock = clock
        self._records: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._lock = RLock()

    def _purge(self) -> None:
        now = self._clock()
        expired = [key for key, (expires, _) in self._records.items() if expires <= now]
        for key in expired:
            self._records.pop(key, None)

    def put(self, execution: RetrievalExecution, principal: Principal) -> None:
        now = datetime.now(UTC)
        expires_at = now + timedelta(seconds=self.ttl_sec)
        trace = _scrub(execution.trace)
        record = {
            "retrieval_id": execution.retrieval_id,
            "tenant_id": principal.tenant_id,
            "user_id": principal.user_id,
            "query": trace.get("query", ""),
            "paper_scope": trace.get("paper_scope"),
            "intent": trace.get("intent", {}),
            "rewrites": trace.get("rewrites", []),
            "iterations": trace.get("iters", []),
            "candidate_scores": trace.get("candidate_scores", []),
            "abstain": trace.get("abstain", {}),
            "evidence_chunk_ids": execution.allowed_chunk_ids,
            "allowed_chunk_ids": execution.allowed_chunk_ids,
            "wiki_matches": trace.get("wiki_entries", trace.get("wiki_context", {})),
            "timings": trace.get("timings", {}),
            "degraded_components": trace.get("degraded_components", []),
            "trace": trace,
            "created_at": now.isoformat(),
            "expires_at": expires_at.isoformat(),
        }
        with self._lock:
            self._purge()
            self._records[execution.retrieval_id] = (self._clock() + self.ttl_sec, record)
            self._records.move_to_end(execution.retrieval_id)
            while len(self._records) > self.max_entries:
                self._records.popitem(last=False)

    def get(self, retrieval_id: str, principal: Principal) -> dict[str, Any]:
        with self._lock:
            self._purge()
            item = self._records.get(retrieval_id)
            if item is None:
                raise RetrievalExpiredError(retrieval_id)
            _, record = item
            if record["tenant_id"] != principal.tenant_id or (
                not principal.is_admin and record["user_id"] != principal.user_id
            ):
                raise PermissionDeniedError()
            self._records.move_to_end(retrieval_id)
            return copy.deepcopy(record)

    def purge_expired(self) -> int:
        with self._lock:
            before = len(self._records)
            self._purge()
            return before - len(self._records)
