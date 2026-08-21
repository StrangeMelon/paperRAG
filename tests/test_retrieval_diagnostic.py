"""Interactive retrieval diagnostic contracts."""

from __future__ import annotations


def test_run_retrieval_diagnostic_returns_stage_results_and_total(monkeypatch) -> None:
    from paper_rag.dashboard.services.retrieval_diagnostic import run_retrieval_diagnostic

    def fake_retrieve(query, paper_ids, top_k, *, timings=None, diagnostics=None, **kwargs):
        assert query == "原问题"
        assert timings is not None
        assert diagnostics is not None
        timings.update(
            {
                "query_rewrite_ms": 1.0,
                "dense_ms": 2.0,
                "sparse_ms": 3.0,
                "rrf_ms": 4.0,
                "rerank_ms": 5.0,
                "diversify_ms": 6.0,
            }
        )
        diagnostics.update(
            {
                "rewrite": {"dense_queries": ["改写问题"], "bm25_query": "关键词"},
                "dense": [
                    {"chunk_id": "d2", "score": 0.42},
                    {"chunk_id": "d1", "score": 0.91},
                ],
                "sparse": [{"chunk_id": "s1", "score_bm25": 8.2}],
                "rrf": [{"chunk_id": "r1", "score_rrf": 0.12}],
                "rerank": [{"chunk_id": "r1", "score_rerank": 0.88}],
                "diversify": [{"chunk_id": "r1", "score_rerank": 0.88}],
            }
        )
        return [{"chunk_id": "r1"}], {"dense_queries": ["改写问题"]}, timings

    monkeypatch.setattr(
        "paper_rag.dashboard.services.retrieval_diagnostic.retrieve_round_with_rewrite",
        fake_retrieve,
    )

    result = run_retrieval_diagnostic("原问题", top_k=4)

    assert result["query"] == "原问题"
    assert result["timings_ms"]["retrieval_total_ms"] == 21.0
    assert result["stages"]["Query Rewrite"]["rewritten_queries"] == ["改写问题"]
    assert [item["chunk_id"] for item in result["stages"]["Dense"]["items"]] == ["d1", "d2"]
    assert result["stages"]["Sparse"]["items"][0]["score_bm25"] == 8.2
    assert result["stages"]["RRF"]["items"][0]["score_rrf"] == 0.12
    assert result["stages"]["Diversify"]["items"][0]["chunk_id"] == "r1"


def test_run_retrieval_diagnostic_rejects_blank_query() -> None:
    from paper_rag.dashboard.services.retrieval_diagnostic import run_retrieval_diagnostic

    try:
        run_retrieval_diagnostic("  ")
    except ValueError as exc:
        assert "query" in str(exc)
    else:  # pragma: no cover - assertion is the behavior contract
        raise AssertionError("blank query should be rejected")
