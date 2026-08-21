"""Dashboard QA mode adaptation and persistence tests."""

from __future__ import annotations


def test_query_service_runs_agentic_and_persists_normalized_trace(tmp_path) -> None:
    from paper_rag.dashboard.services.query_service import QueryService
    from paper_rag.dashboard.services.trace_store import QueryTraceStore

    def agentic(question: str, *, paper_ids=None):
        return {
            "answer": "grounded answer",
            "citations": ["c1"],
            "chunks": [{"chunk_id": "c1", "paper_id": "p1", "text": "ctx"}],
            "evidence_chunks": [{"chunk_id": "c1", "paper_id": "p1", "text": "ctx"}],
            "trace": {
                "trace_id": "trace-agentic",
                "intent": {"intent": "factual"},
                "iters": [{"query": question, "n_retrieved": 1}],
                "abstain": {"decision": "confident"},
            },
        }

    store = QueryTraceStore(tmp_path / "traces.jsonl")
    result = QueryService(store=store, agentic_answer=agentic).run(
        "What is RAG?", mode="agentic", paper_ids=["p1"]
    )

    assert result["trace_id"] == "trace-agentic"
    assert result["intent"] == "factual"
    assert result["abstain"] == "confident"
    assert store.get("trace-agentic")["answer"] == "grounded answer"


def test_query_service_supports_simple_and_stream_modes(tmp_path) -> None:
    from paper_rag.dashboard.services.query_service import QueryService
    from paper_rag.dashboard.services.trace_store import QueryTraceStore

    simple = lambda question, **kwargs: {  # noqa: E731
        "answer": "simple",
        "citations": [],
        "chunks": [],
    }

    def stream(question: str, *, paper_ids=None):
        yield {"event": "intent", "data": {"intent": "explore"}}
        yield {"event": "answer_chunk", "data": {"text": "stream "}}
        yield {"event": "answer_chunk", "data": {"text": "answer"}}
        yield {"event": "done", "data": {"citations": ["c2"], "abstain": {"decision": "weak"}}}

    service = QueryService(
        store=QueryTraceStore(tmp_path / "traces.jsonl"),
        simple_answer=simple,
        stream_answer=stream,
    )

    assert service.run("q", mode="simple", top_k=4)["answer"] == "simple"
    streamed = service.run("q", mode="stream")
    assert streamed["answer"] == "stream answer"
    assert streamed["intent"] == "explore"
    assert streamed["abstain"] == "weak"
