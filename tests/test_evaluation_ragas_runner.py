"""RAGAS-only runner and report contracts."""

from __future__ import annotations

import json
import threading

import pytest


def test_ragas_runner_calls_qa_once_and_reports_metric_coverage(tmp_path) -> None:
    from paper_rag.evaluation.ragas_runner import RagasEvalRunner
    from paper_rag.evaluation.ragas_schema import (
        RagasCorpusSnapshot,
        RagasMetricObservation,
        RagasSampleEvaluation,
    )

    path = tmp_path / "ragas.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "ragas-eval.v1",
                "corpus": {"selection": "all_indexed"},
                "test_cases": [
                    {
                        "id": "one",
                        "query": "q1",
                        "paper_ids": ["p1"],
                        "reference_answer": "a1",
                        "reference_chunk_ids": ["c1"],
                        "expected_abstain": False,
                        "tags": ["factual"],
                    },
                    {
                        "id": "two",
                        "query": "q2",
                        "paper_ids": ["p1"],
                        "reference_answer": "a2",
                        "reference_chunk_ids": ["c1"],
                        "expected_abstain": False,
                        "tags": ["factual"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    calls: list[dict] = []

    def answer_fn(
        query: str,
        *,
        paper_ids=None,
        top_k_override=None,
        query_rewrite_enabled=True,
        evaluation_parallel=False,
    ):
        calls.append(
            {
                "query": query,
                "paper_ids": paper_ids,
                "top_k_override": top_k_override,
                "query_rewrite_enabled": query_rewrite_enabled,
                "evaluation_parallel": evaluation_parallel,
            }
        )
        return {
            "answer": f"answer {query}",
            "chunks": [{"chunk_id": "candidate", "paper_id": "p1", "text": "candidate"}],
            "evidence_chunks": [
                {"chunk_id": "evidence", "paper_id": "p1", "text": "evidence context"}
            ],
            "citations": ["evidence"],
            "trace": {"abstain": {"decision": "confident"}},
        }

    class Evaluator:
        metrics = ["faithfulness"]

        def evaluate_batch(self, samples):
            return [
                RagasSampleEvaluation(
                    sample_id=samples[0].id,
                    observations={"faithfulness": RagasMetricObservation.ok(0.8, latency_ms=1.0)},
                ),
                RagasSampleEvaluation(
                    sample_id=samples[1].id,
                    observations={
                        "faithfulness": RagasMetricObservation.failure(
                            "error", "RuntimeError", "offline", eligible=True
                        )
                    },
                ),
            ]

    runner = RagasEvalRunner(
        answer_fn,
        Evaluator(),
        top_k=12,
        query_rewrite=False,
        corpus_inspector=lambda cases: RagasCorpusSnapshot(
            selection="all_indexed", paper_ids=("p1",), manifest_sha256="manifest"
        ),
    )

    report = runner.run(path).to_dict()

    assert calls == [
        {
            "query": "q1",
            "paper_ids": ["p1"],
            "top_k_override": 12,
            "query_rewrite_enabled": False,
            "evaluation_parallel": False,
        },
        {
            "query": "q2",
            "paper_ids": ["p1"],
            "top_k_override": 12,
            "query_rewrite_enabled": False,
            "evaluation_parallel": False,
        },
    ]
    assert report["query_results"][0]["retrieved_chunk_ids"] == ["evidence"]
    assert report["schema_version"] == "ragas-report.v1"
    assert report["aggregate_metrics"]["faithfulness"] == {
        "mean": 0.8,
        "count": 1,
        "eligible_count": 2,
        "coverage": 0.5,
    }
    assert report["query_results"][1]["status"] == "error"
    assert report["evaluation"]["top_k"] == 12
    assert report["evaluation"]["query_rewrite"] is False
    assert report["evaluation"]["max_concurrency"] == 1


def test_ragas_runner_rejects_invalid_top_k() -> None:
    from paper_rag.evaluation.ragas_runner import RagasEvalRunner

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        RagasEvalRunner(lambda query, **kwargs: {}, object(), top_k=0)


def test_ragas_runner_collects_qa_cases_concurrently_and_preserves_order(tmp_path) -> None:
    from paper_rag.evaluation.ragas_runner import RagasEvalRunner
    from paper_rag.evaluation.ragas_schema import RagasCorpusSnapshot

    path = tmp_path / "ragas.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "ragas-eval.v1",
                "corpus": {"selection": "all_indexed"},
                "test_cases": [
                    {
                        "id": case_id,
                        "query": query,
                        "paper_ids": ["p1"],
                        "reference_answer": "answer",
                        "reference_chunk_ids": ["c1"],
                        "expected_abstain": False,
                        "tags": ["concurrency"],
                    }
                    for case_id, query in (("one", "q1"), ("two", "q2"))
                ],
            }
        ),
        encoding="utf-8",
    )
    lock = threading.Lock()
    both_started = threading.Event()
    active = 0
    peak = 0
    parallel_flags: list[bool] = []

    def answer_fn(query: str, *, evaluation_parallel: bool, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            parallel_flags.append(evaluation_parallel)
            if active == 2:
                both_started.set()
        started_together = both_started.wait(timeout=1)
        with lock:
            active -= 1
        assert started_together
        return {
            "answer": f"answer {query}",
            "chunks": [{"chunk_id": query, "text": query}],
            "citations": [query],
            "trace": {"abstain": {"decision": "confident"}},
        }

    class Evaluator:
        metrics = []

        @staticmethod
        def evaluate_batch(samples):
            return []

    report = RagasEvalRunner(
        answer_fn,
        Evaluator(),
        max_concurrency=2,
        corpus_inspector=lambda cases: RagasCorpusSnapshot(
            selection="all_indexed", paper_ids=("p1",), manifest_sha256="manifest"
        ),
    ).run(path)

    assert peak == 2
    assert parallel_flags == [True, True]
    assert [result.id for result in report.query_results] == ["one", "two"]
    assert report.max_concurrency == 2


def test_ragas_runner_rejects_invalid_concurrency() -> None:
    from paper_rag.evaluation.ragas_runner import RagasEvalRunner

    with pytest.raises(ValueError, match="max_concurrency must be between 1 and 16"):
        RagasEvalRunner(lambda query, **kwargs: {}, object(), max_concurrency=0)


def test_ragas_runner_counts_qa_errors_in_metric_coverage(tmp_path) -> None:
    from paper_rag.evaluation.ragas import RagasEvaluator
    from paper_rag.evaluation.ragas_runner import RagasEvalRunner
    from paper_rag.evaluation.ragas_schema import RagasCorpusSnapshot

    path = tmp_path / "ragas.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "ragas-eval.v1",
                "corpus": {"selection": "all_indexed"},
                "test_cases": [
                    {
                        "id": "broken",
                        "query": "q",
                        "paper_ids": ["p1"],
                        "reference_answer": "a",
                        "reference_chunk_ids": ["c1"],
                        "expected_abstain": False,
                        "tags": ["factual"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def broken_answer(query: str, *, paper_ids=None):
        raise RuntimeError("qa offline")

    evaluator = RagasEvaluator(
        ["faithfulness"],
        settings=type("Settings", (), {"max_concurrency": 1})(),
        metric_instances={"faithfulness": object()},
    )
    runner = RagasEvalRunner(
        broken_answer,
        evaluator,
        corpus_inspector=lambda cases: RagasCorpusSnapshot(
            selection="all_indexed", paper_ids=("p1",), manifest_sha256="manifest"
        ),
    )

    report = runner.run(path).to_dict()

    assert report["query_results"][0]["status"] == "qa_error"
    assert report["aggregate_metrics"]["faithfulness"]["eligible_count"] == 1
    assert report["aggregate_metrics"]["faithfulness"]["coverage"] == 0.0


def test_ragas_runner_validates_evaluator_before_calling_qa(tmp_path) -> None:
    from paper_rag.evaluation.ragas_runner import RagasEvalRunner
    from paper_rag.evaluation.ragas_schema import RagasCorpusSnapshot

    path = tmp_path / "ragas.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "ragas-eval.v1",
                "corpus": {"selection": "all_indexed"},
                "test_cases": [
                    {
                        "id": "one",
                        "query": "q",
                        "paper_ids": ["p1"],
                        "reference_answer": "a",
                        "reference_chunk_ids": ["c1"],
                        "expected_abstain": False,
                        "tags": ["factual"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    qa_calls: list[str] = []

    class Evaluator:
        metrics = ["faithfulness"]

        def validate_runtime(self):
            raise ValueError("judge config missing")

    runner = RagasEvalRunner(
        lambda query, *, paper_ids=None: qa_calls.append(query),
        Evaluator(),
        corpus_inspector=lambda cases: RagasCorpusSnapshot(
            selection="all_indexed", paper_ids=("p1",), manifest_sha256="manifest"
        ),
    )

    with pytest.raises(ValueError, match="judge config missing"):
        runner.run(path)

    assert qa_calls == []
