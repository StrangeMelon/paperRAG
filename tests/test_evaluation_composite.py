"""Unit tests for evaluator composition."""

from __future__ import annotations


class _StubEvaluator:
    def __init__(self, values: dict[str, float], error: Exception | None = None) -> None:
        self.values = values
        self.error = error

    def evaluate(self, **kwargs) -> dict[str, float]:
        if self.error:
            raise self.error
        return self.values


def test_composite_namespaces_metrics_and_keeps_partial_results() -> None:
    from paper_rag.evaluation.composite import CompositeEvaluator

    evaluator = CompositeEvaluator(
        [
            ("custom", _StubEvaluator({"mrr": 0.5})),
            ("ragas", _StubEvaluator({"faithfulness": 0.9})),
            ("broken", _StubEvaluator({}, RuntimeError("offline"))),
        ]
    )

    result = evaluator.evaluate("q", [{"chunk_id": "c1"}], generated_answer="a")

    assert result.metrics == {"custom.mrr": 0.5, "ragas.faithfulness": 0.9}
    assert result.errors[0]["backend"] == "broken"


def test_composite_fails_when_all_backends_fail() -> None:
    from paper_rag.evaluation.composite import CompositeEvaluator

    evaluator = CompositeEvaluator([("custom", _StubEvaluator({}, ValueError("bad")))])

    result = evaluator.evaluate("q", [])

    assert result.metrics == {}
    assert result.failed is True
