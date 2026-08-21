from __future__ import annotations


def test_retrieve_round_can_skip_query_rewrite(monkeypatch) -> None:
    from paper_rag.retrieve import pipeline

    called = False

    def rewrite(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("rewrite should not be called")

    monkeypatch.setattr(pipeline, "hybrid_search", lambda query, **kwargs: [])
    result = pipeline.retrieve_round(
        "q",
        None,
        2,
        rewrite_enabled=False,
        wiki_context={},
    )
    assert result == []
    assert called is False
