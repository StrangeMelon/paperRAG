"""Unit tests for Golden Set loading and batch evaluation."""

from __future__ import annotations

import json


def test_runner_evaluates_cases_and_aggregates_metrics(tmp_path) -> None:
    from paper_rag.evaluation.custom import CustomEvaluator
    from paper_rag.evaluation.runner import EvalRunner

    test_set = tmp_path / "golden.json"
    test_set.write_text(
        json.dumps(
            {
                "version": "1.0",
                "test_cases": [
                    {"id": "a", "query": "q1", "expected_chunk_ids": ["c1"]},
                    {"id": "b", "query": "q2", "expected_chunk_ids": ["missing"]},
                ],
            }
        ),
        encoding="utf-8",
    )

    def answer_fn(query: str, *, paper_ids=None):
        return {
            "answer": f"answer {query}",
            "citations": ["c1"] if query == "q1" else [],
            "chunks": [{"chunk_id": "c1", "paper_id": "p1", "text": "ctx"}]
            if query == "q1"
            else [],
            "trace": {"trace_id": "trace-1", "abstain": {"decision": "confident"}},
        }

    report = EvalRunner(
        answer_fn=answer_fn, evaluator=CustomEvaluator(metrics=["hit_rate", "mrr"])
    ).run(test_set)

    assert report.query_count == 2
    assert report.aggregate_metrics == {"hit_rate": 0.5, "mrr": 0.5}
    assert report.query_results[0].status == "ok"
    assert report.query_results[1].status == "ok"


def test_runner_records_answer_and_evaluator_errors(tmp_path) -> None:
    from paper_rag.evaluation.custom import CustomEvaluator
    from paper_rag.evaluation.runner import EvalRunner

    test_set = tmp_path / "golden.json"
    test_set.write_text(json.dumps({"test_cases": [{"id": "a", "query": "q"}]}), encoding="utf-8")

    def broken_answer(query: str, *, paper_ids=None):
        raise RuntimeError("llm unavailable")

    report = EvalRunner(answer_fn=broken_answer, evaluator=CustomEvaluator()).run(test_set)

    assert report.query_results[0].status == "error"
    assert "llm unavailable" in report.query_results[0].errors[0]["message"]
