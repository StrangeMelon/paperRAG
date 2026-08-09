"""Thread-safe TTL/LRU retrieval trace store contracts."""

from __future__ import annotations

import pytest

from paper_rag.mcp.errors import PermissionDeniedError, RetrievalExpiredError
from paper_rag.mcp.trace_store import RetrievalTraceStore
from paper_rag.rag.evidence_retrieval import Principal, RetrievalExecution


def _execution(retrieval_id: str = "r1") -> RetrievalExecution:
    return RetrievalExecution(
        retrieval_id=retrieval_id,
        public_decision="no_evidence",
        internal_decision="no_evidence",
        candidate_chunks=[],
        evidence_chunks=[],
        wiki_entries=[],
        allowed_chunk_ids=["c1"],
        trace={
            "query": "question",
            "paper_scope": ["paper:1"],
            "intent": {"intent": "factual"},
            "rewrites": [],
            "iters": [],
            "abstain": {"decision": "no_evidence"},
            "source_path": "/private/paper.pdf",
            "api_key": "must-not-store",
        },
    )


def test_put_get_binds_trace_to_tenant_and_scrubs_sensitive_fields() -> None:
    store = RetrievalTraceStore(ttl_sec=30, max_entries=10)
    principal = Principal(tenant_id="tenant-a", user_id="user-a")

    store.put(_execution(), principal)
    record = store.get("r1", principal)

    assert record["retrieval_id"] == "r1"
    assert record["tenant_id"] == "tenant-a"
    assert record["allowed_chunk_ids"] == ["c1"]
    assert "source_path" not in str(record)
    assert "api_key" not in str(record)

    with pytest.raises(PermissionDeniedError):
        store.get("r1", Principal(tenant_id="tenant-b", user_id="user-a"))


def test_expired_trace_is_unreadable() -> None:
    now = [100.0]
    store = RetrievalTraceStore(ttl_sec=10, max_entries=10, clock=lambda: now[0])
    principal = Principal(tenant_id="tenant-a", user_id="user-a")
    store.put(_execution(), principal)

    now[0] = 111.0
    with pytest.raises(RetrievalExpiredError):
        store.get("r1", principal)


def test_lru_eviction_keeps_newest_records() -> None:
    store = RetrievalTraceStore(ttl_sec=30, max_entries=2)
    principal = Principal(tenant_id="tenant-a", user_id="user-a")
    store.put(_execution("r1"), principal)
    store.put(_execution("r2"), principal)
    store.put(_execution("r3"), principal)

    with pytest.raises(RetrievalExpiredError):
        store.get("r1", principal)
    assert store.get("r2", principal)["retrieval_id"] == "r2"
    assert store.get("r3", principal)["retrieval_id"] == "r3"
