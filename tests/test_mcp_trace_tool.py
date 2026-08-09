"""Admin trace tool adapter contract."""

from __future__ import annotations

import asyncio

from paper_rag.mcp.tools.retrieval_trace import paper_get_retrieval_trace
from paper_rag.rag.evidence_retrieval import Principal


def test_trace_tool_reads_store_with_principal() -> None:
    calls: list[tuple[str, Principal]] = []

    class Store:
        def get(self, retrieval_id, principal):
            calls.append((retrieval_id, principal))
            return {"retrieval_id": retrieval_id, "allowed_chunk_ids": ["c1"]}

    principal = Principal(tenant_id="tenant-a", user_id="admin", is_admin=True)
    result = asyncio.run(paper_get_retrieval_trace("r1", trace_store=Store(), principal=principal))

    assert result == {"retrieval_id": "r1", "allowed_chunk_ids": ["c1"]}
    assert calls == [("r1", principal)]
