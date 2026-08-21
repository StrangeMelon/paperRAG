"""SQLite-backed retrieval diagnostic history contracts."""

from __future__ import annotations

from sqlmodel import SQLModel, create_engine


def _result(query: str, score: float = 0.9) -> dict:
    return {
        "query": query,
        "timings_ms": {
            "query_rewrite_ms": 1.0,
            "dense_ms": 2.0,
            "sparse_ms": 3.0,
            "rrf_ms": 4.0,
            "rerank_ms": 5.0,
            "diversify_ms": 6.0,
            "retrieval_total_ms": 21.0,
        },
        "stages": {
            "Query Rewrite": {
                "timing_ms": 1.0,
                "rewritten_queries": [f"{query} rewritten"],
                "bm25_query": f"{query} keyword",
            },
            "Dense": {
                "timing_ms": 2.0,
                "items": [
                    {
                        "chunk_id": "c1",
                        "paper_id": "p1",
                        "score": score,
                        "text": "x" * 700,
                    }
                ],
            },
            "Sparse": {"timing_ms": 3.0, "items": []},
            "RRF": {"timing_ms": 4.0, "items": []},
            "Rerank": {"timing_ms": 5.0, "items": []},
            "Diversify": {"timing_ms": 6.0, "items": []},
        },
        "chunks": [{"chunk_id": "c1"}],
    }


def _store(tmp_path):
    from paper_rag.dashboard.services.retrieval_history import RetrievalHistoryStore

    engine = create_engine(f"sqlite:///{tmp_path / 'history.sqlite'}")
    SQLModel.metadata.create_all(engine)
    return RetrievalHistoryStore(engine=engine)


def test_history_persists_and_restores_complete_diagnostic(tmp_path) -> None:
    store = _store(tmp_path)

    run_id = store.save(_result("Ed25519"), paper_ids=["p1"], top_k=8)
    restored = store.get(run_id)

    assert restored is not None
    assert restored["query"] == "Ed25519"
    assert restored["paper_ids"] == ["p1"]
    assert restored["top_k"] == 8
    assert restored["timings_ms"]["retrieval_total_ms"] == 21.0
    assert restored["stages"]["Query Rewrite"]["rewritten_queries"] == ["Ed25519 rewritten"]
    dense = restored["stages"]["Dense"]["items"][0]
    assert dense["rank"] == 1
    assert dense["score"] == 0.9
    assert len(dense["text"]) == 500


def test_history_lists_newest_filters_paginates_and_deletes(tmp_path) -> None:
    store = _store(tmp_path)
    first = store.save(_result("first query"), paper_ids=None, top_k=4)
    second = store.save(_result("Ed25519 algorithm"), paper_ids=["p1"], top_k=8)
    third = store.save(_result("third query"), paper_ids=None, top_k=6)

    assert [item["run_id"] for item in store.list(limit=2)] == [third, second]
    assert [item["run_id"] for item in store.list(query="ed25519")] == [second]
    assert [item["run_id"] for item in store.list(limit=1, offset=1)] == [second]
    assert store.count() == 3
    assert store.count(query="Ed25519") == 1
    assert store.delete(second) is True
    assert store.get(second) is None
    assert store.delete(second) is False
    assert store.count() == 2
    assert first in {item["run_id"] for item in store.list(limit=10)}
