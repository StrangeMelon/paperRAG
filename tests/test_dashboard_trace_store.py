"""Dashboard query trace persistence contracts."""

from __future__ import annotations

import json


def _record(trace_id: str, query: str, *, status: str = "ok", created_at: str) -> dict:
    return {
        "trace_id": trace_id,
        "query": query,
        "mode": "agentic",
        "status": status,
        "created_at": created_at,
        "latency_ms": 42,
        "answer": "answer",
        "citations": ["c1"],
        "chunks": [{"chunk_id": "c1", "text": "context"}],
        "trace": {"stopped_by": "answered"},
    }


def test_trace_store_persists_filters_deletes_and_exports(tmp_path) -> None:
    from paper_rag.dashboard.services.trace_store import QueryTraceStore

    path = tmp_path / "query_traces.jsonl"
    store = QueryTraceStore(path)
    store.append(_record("t1", "Graph retrieval", created_at="2026-08-12T10:00:00+00:00"))
    store.append(
        _record(
            "t2",
            "Vision table",
            status="error",
            created_at="2026-08-13T10:00:00+00:00",
        )
    )

    assert [item["trace_id"] for item in store.list(keyword="graph")] == ["t1"]
    assert [item["trace_id"] for item in store.list(status="error")] == ["t2"]
    assert store.get("t1")["query"] == "Graph retrieval"

    exported = json.loads(store.export_json())
    assert [item["trace_id"] for item in exported] == ["t2", "t1"]

    assert store.delete("t2") is True
    assert store.delete("missing") is False
    assert [item["trace_id"] for item in store.list()] == ["t1"]


def test_trace_store_skips_malformed_lines(tmp_path) -> None:
    from paper_rag.dashboard.services.trace_store import QueryTraceStore

    path = tmp_path / "query_traces.jsonl"
    path.write_text('{"trace_id":"ok","created_at":"2026-08-13T00:00:00Z"}\nnot-json\n')

    assert [item["trace_id"] for item in QueryTraceStore(path).list()] == ["ok"]
