"""MCP evidence adapter public response contract."""

from __future__ import annotations

import asyncio
import json

import pytest

from paper_rag.mcp.tools.retrieve_evidence import _MAX_PUBLIC_RESPONSE_CHARS, build_public_response
from paper_rag.rag.evidence_retrieval import Principal, RetrievalExecution


def _chunk(chunk_id: str = "c1") -> dict:
    return {
        "chunk_id": chunk_id,
        "paper_id": "paper:1",
        "title": "Paper One",
        "section": "Method",
        "page": 6,
        "modality": "text",
        "text": "Graph-Mamba uses selective state spaces.",
        "score_rerank": 0.99,
        "source_path": "/private/path/paper.pdf",
    }


def _execution(**overrides) -> RetrievalExecution:
    values = {
        "retrieval_id": "r_public",
        "public_decision": "confident",
        "internal_decision": "confident",
        "candidate_chunks": [_chunk()],
        "evidence_chunks": [_chunk()],
        "wiki_entries": [{"name": "State Space Model", "definition": "Background"}],
        "allowed_chunk_ids": ["c1"],
        "trace": {"intent": {"intent": "factual"}, "candidate_scores": [0.99]},
    }
    values.update(overrides)
    return RetrievalExecution(**values)


def test_confident_response_is_minimal_and_strips_internal_fields() -> None:
    result = build_public_response(_execution())

    assert set(result) == {"decision", "retrieval_id", "evidence", "wiki"}
    assert result["decision"] == "confident"
    assert result["evidence"] == [
        {
            "citation": "[chunk:c1]",
            "paper_id": "paper:1",
            "title": "Paper One",
            "section": "Method",
            "page": 6,
            "modality": "text",
            "text": "Graph-Mamba uses selective state spaces.",
        }
    ]
    assert result["wiki"] == [{"name": "State Space Model", "definition": "Background"}]
    assert "score_rerank" not in str(result)
    assert "source_path" not in str(result)
    assert "candidate_scores" not in str(result)


def test_weak_response_keeps_same_public_shape() -> None:
    result = build_public_response(_execution(public_decision="weak_evidence"))

    assert set(result) == {"decision", "retrieval_id", "evidence", "wiki"}
    assert result["decision"] == "weak_evidence"


def test_no_evidence_response_omits_wiki_and_internal_candidates() -> None:
    result = build_public_response(
        _execution(
            public_decision="no_evidence",
            internal_decision="no_evidence",
            candidate_chunks=[_chunk()],
            evidence_chunks=[],
            wiki_entries=[{"name": "Should Not Leak", "definition": "x"}],
            allowed_chunk_ids=[],
        )
    )

    assert result == {"decision": "no_evidence", "retrieval_id": "r_public", "evidence": []}


def test_async_tool_runs_domain_service_and_persists_trace(monkeypatch) -> None:
    from paper_rag.mcp.tools import retrieve_evidence as adapter

    execution = _execution()
    calls: list[dict] = []

    class FakeRuntime:
        async def run_sync(self, function, *args, **kwargs):
            calls.append({"function": function, "args": args, "kwargs": kwargs})
            return execution

    class FakeTraceStore:
        def put(self, value, principal):
            calls.append({"trace": value, "principal": principal})

    principal = Principal(tenant_id="tenant-a", user_id="user-a")
    result = asyncio.run(
        adapter.paper_retrieve_evidence(
            {
                "query": "What is RAG?",
                "max_evidence": 2,
                "include_wiki": True,
                "wiki_max_entries": 1,
            },
            runtime=FakeRuntime(),
            trace_store=FakeTraceStore(),
            principal=principal,
        )
    )

    assert result["decision"] == "confident"
    assert calls[0]["args"][0] == "What is RAG?"
    assert calls[0]["kwargs"]["max_evidence"] == 2
    assert calls[0]["kwargs"]["principal"] == principal
    assert calls[1]["trace"] is execution


def test_async_tool_rejects_invalid_input_before_runtime() -> None:
    from paper_rag.mcp.tools import retrieve_evidence as adapter

    class ExplodingRuntime:
        async def run_sync(self, *args, **kwargs):
            raise AssertionError("runtime must not be called")

    with pytest.raises(ValueError):
        asyncio.run(
            adapter.paper_retrieve_evidence(
                {"query": "   "},
                runtime=ExplodingRuntime(),
                trace_store=object(),
                principal=Principal(tenant_id="tenant-a", user_id="user-a"),
            )
        )


def test_public_response_budget_drops_wiki_before_evidence_and_clips_text() -> None:
    chunks = [_chunk(f"c{index}") | {"text": "x" * 12000} for index in range(2)]
    result = build_public_response(
        _execution(
            candidate_chunks=chunks,
            evidence_chunks=chunks,
            allowed_chunk_ids=["c0", "c1"],
            wiki_entries=[
                {"name": f"Wiki {index}", "definition": "w" * 4000} for index in range(3)
            ],
        )
    )

    assert len(json.dumps(result, ensure_ascii=False)) <= _MAX_PUBLIC_RESPONSE_CHARS
    assert len(result["evidence"]) >= 1
    assert result["wiki"] == []
    assert result["evidence"][0]["text"].endswith("...")
