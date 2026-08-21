"""Tests for the retrieval-only Golden Set and runner."""

from __future__ import annotations

import json
import threading
import time

import pytest


def _write(path, cases):
    path.write_text(
        json.dumps(
            {
                "schema_version": "retrieval.v1",
                "name": "test",
                "corpus": {"selection": "all_indexed"},
                "test_cases": cases,
            }
        ),
        encoding="utf-8",
    )


def test_retrieval_golden_set_rejects_empty_cases(tmp_path) -> None:
    from paper_rag.evaluation.retrieval import RetrievalGoldenSetError, load_retrieval_golden_set

    path = tmp_path / "retrieval.json"
    _write(path, [])
    with pytest.raises(RetrievalGoldenSetError, match="non-empty"):
        load_retrieval_golden_set(path)


def test_retrieval_golden_set_rejects_legacy_expected_sources(tmp_path) -> None:
    from paper_rag.evaluation.retrieval import RetrievalGoldenSetError, load_retrieval_golden_set

    path = tmp_path / "retrieval.json"
    _write(
        path,
        [
            {
                "id": "q1",
                "query": "q",
                "expected_chunk_ids": ["c1"],
                "expected_sources": ["p1"],
                "reference_answer": "answer",
                "search_scope": None,
            }
        ],
    )
    with pytest.raises(RetrievalGoldenSetError, match="expected_paper_ids"):
        load_retrieval_golden_set(path)


def test_retrieval_golden_set_accepts_scoped_case(tmp_path) -> None:
    from paper_rag.evaluation.retrieval import load_retrieval_golden_set

    path = tmp_path / "retrieval.json"
    _write(
        path,
        [
            {
                "id": "q1",
                "query": "q",
                "expected_chunk_ids": ["c1"],
                "expected_paper_ids": ["p1"],
                "reference_answer": "answer",
                "search_scope": {"paper_ids": ["p1", "p2"]},
            }
        ],
    )
    cases, digest = load_retrieval_golden_set(path)
    assert cases[0].search_scope == ["p1", "p2"]
    assert len(digest) == 64


def test_retrieval_runner_records_metadata_and_metrics(monkeypatch, tmp_path) -> None:
    from paper_rag.evaluation.retrieval import CorpusSnapshot, RetrievalEvalRunner

    path = tmp_path / "retrieval.json"
    _write(
        path,
        [
            {
                "id": "q1",
                "query": "q",
                "expected_chunk_ids": ["c1", "c2"],
                "expected_paper_ids": ["p1"],
                "reference_answer": "answer",
                "search_scope": None,
            }
        ],
    )

    class Paper:
        def __init__(self, paper_id):
            self.paper_id = paper_id
            self.status = "done"

    monkeypatch.setattr(
        "paper_rag.evaluation.retrieval.list_papers_by_status", lambda status: [Paper("p1")]
    )
    monkeypatch.setattr(
        "paper_rag.evaluation.retrieval.get_papers_by_ids",
        lambda ids: [Paper(item) for item in ids],
    )
    monkeypatch.setattr(
        "paper_rag.evaluation.retrieval.get_chunk",
        lambda chunk_id: type("Chunk", (), {"paper_id": "p1"})(),
    )

    seen = {}

    def retrieve(query, *, paper_ids, top_k, rewrite_enabled):
        seen.update(paper_ids=paper_ids, top_k=top_k, rewrite_enabled=rewrite_enabled)
        return [
            {"chunk_id": "c2", "paper_id": "p1"},
            {"chunk_id": "c3", "paper_id": "p1"},
        ]

    report = RetrievalEvalRunner(retrieve, top_k=5, rewrite_enabled=False).run(path)
    assert seen == {"paper_ids": ["p1"], "top_k": 5, "rewrite_enabled": False}
    assert report["evaluation"]["mode"] == "retrieval"
    assert report["evaluation"]["query_rewrite"] is False
    assert report["evaluation"]["max_concurrency"] == 8
    assert report["evaluation"]["corpus_paper_count"] == 1
    assert report["aggregate_metrics"] == {
        "hit_rate": 1.0,
        "mrr": 1.0 / 1.0,
        "recall": 0.5,
        "paper_hit_rate": 1.0,
    }


def test_retrieval_runner_parallelizes_rewritten_cases_and_preserves_order(
    monkeypatch, tmp_path
) -> None:
    from paper_rag.evaluation.retrieval import RetrievalEvalRunner

    path = tmp_path / "retrieval.json"
    _write(
        path,
        [
            {
                "id": "q1",
                "query": "first",
                "expected_chunk_ids": ["c1"],
                "expected_paper_ids": ["p1"],
                "reference_answer": "answer",
            },
            {
                "id": "q2",
                "query": "second",
                "expected_chunk_ids": ["c2"],
                "expected_paper_ids": ["p1"],
                "reference_answer": "answer",
            },
        ],
    )

    class Paper:
        def __init__(self, paper_id):
            self.paper_id = paper_id
            self.status = "done"

    monkeypatch.setattr(
        "paper_rag.evaluation.retrieval.list_papers_by_status", lambda status: [Paper("p1")]
    )
    monkeypatch.setattr(
        "paper_rag.evaluation.retrieval.get_papers_by_ids",
        lambda ids: [Paper(item) for item in ids],
    )
    monkeypatch.setattr(
        "paper_rag.evaluation.retrieval.get_chunk",
        lambda chunk_id: type("Chunk", (), {"paper_id": "p1"})(),
    )

    lock = threading.Lock()
    active = 0
    max_active = 0

    def retrieve(query, *, paper_ids, top_k, rewrite_enabled):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        chunk_id = "c1" if query == "first" else "c2"
        return [{"chunk_id": chunk_id, "paper_id": "p1"}]

    report = RetrievalEvalRunner(retrieve, max_concurrency=2).run(path)

    assert max_active == 2
    assert [item["id"] for item in report["query_results"]] == ["q1", "q2"]


def test_retrieval_runner_parallelizes_complete_cases_without_rewrite(
    monkeypatch, tmp_path
) -> None:
    from paper_rag.evaluation.retrieval import RetrievalEvalRunner

    path = tmp_path / "retrieval.json"
    _write(
        path,
        [
            {
                "id": f"q{index}",
                "query": f"query-{index}",
                "expected_chunk_ids": [f"c{index}"],
                "expected_paper_ids": ["p1"],
                "reference_answer": "answer",
            }
            for index in range(2)
        ],
    )

    class Paper:
        def __init__(self, paper_id):
            self.paper_id = paper_id
            self.status = "done"

    monkeypatch.setattr(
        "paper_rag.evaluation.retrieval.list_papers_by_status", lambda status: [Paper("p1")]
    )
    monkeypatch.setattr(
        "paper_rag.evaluation.retrieval.get_papers_by_ids",
        lambda ids: [Paper(item) for item in ids],
    )
    monkeypatch.setattr(
        "paper_rag.evaluation.retrieval.get_chunk",
        lambda chunk_id: type("Chunk", (), {"paper_id": "p1"})(),
    )

    barrier = threading.Barrier(2, timeout=2)
    parallel_flags: list[bool] = []

    def retrieve(query, *, paper_ids, top_k, rewrite_enabled, evaluation_parallel):
        parallel_flags.append(evaluation_parallel)
        barrier.wait()
        return [{"chunk_id": query.replace("query-", "c"), "paper_id": "p1"}]

    report = RetrievalEvalRunner(retrieve, rewrite_enabled=False, max_concurrency=2).run(path)

    assert parallel_flags == [True, True]
    assert [item["status"] for item in report["query_results"]] == ["ok", "ok"]
