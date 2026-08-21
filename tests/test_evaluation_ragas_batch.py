"""Batch scoring contracts for the isolated RAGAS evaluator."""

from __future__ import annotations

from types import SimpleNamespace


class _MetricResult:
    def __init__(self, value: float) -> None:
        self.value = value


class _Metric:
    def __init__(self, value: float = 0.8, error: Exception | None = None) -> None:
        self.value = value
        self.error = error
        self.calls: list[dict] = []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return _MetricResult(self.value)


def _sample(**overrides):
    from paper_rag.evaluation.ragas_schema import RagasSample

    values = {
        "id": "s1",
        "query": "What is RAG?",
        "response": "RAG retrieves evidence.",
        "retrieved_contexts": ["RAG combines retrieval and generation."],
        "retrieved_chunk_ids": ["c1"],
        "citations": ["c1"],
        "reference": "RAG retrieves evidence before generation.",
        "reference_chunk_ids": ["c1"],
        "expected_abstain": False,
        "actual_abstain": "confident",
        "tags": ["factual"],
    }
    values.update(overrides)
    return RagasSample(**values)


def _settings():
    return SimpleNamespace(max_concurrency=2, api_key=None, embedding_api_key=None)


def test_batch_maps_each_metric_to_the_required_ragas_fields() -> None:
    from paper_rag.evaluation.ragas import RagasEvaluator

    metrics = {
        "faithfulness": _Metric(0.9),
        "answer_relevancy": _Metric(0.8),
        "context_precision": _Metric(0.7),
        "context_recall": _Metric(0.6),
        "answer_correctness": _Metric(0.5),
    }
    evaluator = RagasEvaluator(
        metrics=list(metrics), settings=_settings(), metric_instances=metrics
    )

    result = evaluator.evaluate_batch([_sample()])[0]

    assert result.values == {
        "faithfulness": 0.9,
        "answer_relevancy": 0.8,
        "context_precision": 0.7,
        "context_recall": 0.6,
        "answer_correctness": 0.5,
    }
    assert set(metrics["faithfulness"].calls[0]) == {
        "user_input",
        "response",
        "retrieved_contexts",
    }
    assert set(metrics["answer_relevancy"].calls[0]) == {"user_input", "response"}
    assert set(metrics["context_precision"].calls[0]) == {
        "user_input",
        "reference",
        "retrieved_contexts",
    }
    assert set(metrics["answer_correctness"].calls[0]) == {
        "user_input",
        "response",
        "reference",
    }


def test_batch_keeps_successful_metrics_when_another_metric_fails() -> None:
    from paper_rag.evaluation.ragas import RagasEvaluator

    evaluator = RagasEvaluator(
        metrics=["faithfulness", "context_recall"],
        settings=_settings(),
        metric_instances={
            "faithfulness": _Metric(0.75),
            "context_recall": _Metric(error=RuntimeError("judge offline")),
        },
    )

    result = evaluator.evaluate_batch([_sample()])[0]

    assert result.values == {"faithfulness": 0.75}
    assert result.observations["context_recall"].status == "error"
    assert "judge offline" in result.observations["context_recall"].error["message"]
    assert result.status == "partial"


def test_batch_marks_missing_context_as_eligible_failure() -> None:
    from paper_rag.evaluation.ragas import RagasEvaluator

    evaluator = RagasEvaluator(
        metrics=["faithfulness", "answer_relevancy"],
        settings=_settings(),
        metric_instances={
            "faithfulness": _Metric(0.9),
            "answer_relevancy": _Metric(0.8),
        },
    )

    result = evaluator.evaluate_batch([_sample(retrieved_contexts=[])])[0]

    assert result.observations["faithfulness"].status == "missing_input"
    assert result.observations["faithfulness"].eligible is True
    assert result.observations["answer_relevancy"].status == "ok"


def test_batch_marks_expected_abstain_as_not_applicable() -> None:
    from paper_rag.evaluation.ragas import RagasEvaluator

    metric = _Metric(0.9)
    evaluator = RagasEvaluator(
        metrics=["faithfulness"],
        settings=_settings(),
        metric_instances={"faithfulness": metric},
    )

    result = evaluator.evaluate_batch(
        [_sample(expected_abstain=True, response="No evidence", retrieved_contexts=[])]
    )[0]

    assert result.observations["faithfulness"].status == "not_applicable"
    assert result.observations["faithfulness"].eligible is False
    assert metric.calls == []


def test_batch_rejects_non_finite_metric_values() -> None:
    from paper_rag.evaluation.ragas import RagasEvaluator

    evaluator = RagasEvaluator(
        metrics=["faithfulness"],
        settings=_settings(),
        metric_instances={"faithfulness": _Metric(float("nan"))},
    )

    result = evaluator.evaluate_batch([_sample()])[0]

    assert result.observations["faithfulness"].status == "error"
    assert "invalid score" in result.observations["faithfulness"].error["message"]
