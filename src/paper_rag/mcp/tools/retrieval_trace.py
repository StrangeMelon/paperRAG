"""Admin-only retrieval trace adapter."""

from __future__ import annotations

from typing import Any

from ...rag.evidence_retrieval import Principal
from ..trace_store import RetrievalTraceStore


async def paper_get_retrieval_trace(
    retrieval_id: str,
    *,
    trace_store: RetrievalTraceStore,
    principal: Principal,
) -> dict[str, Any]:
    return trace_store.get(retrieval_id, principal)


__all__ = ["paper_get_retrieval_trace"]
