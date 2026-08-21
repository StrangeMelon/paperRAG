"""Import and object-construction contract for the optional evaluation extra."""

from __future__ import annotations

import pytest


def test_evaluation_extra_constructs_all_ragas_metrics_without_network() -> None:
    ragas = pytest.importorskip("ragas")
    instructor = pytest.importorskip("instructor")

    from paper_rag import config as cfg
    from paper_rag.evaluation.ragas import RagasEvaluator

    settings = cfg.load().evaluation.ragas.model_copy(
        update={
            "base_url": "https://example.invalid/v1",
            "api_key": "test-key",
            "judge_model": "judge-model",
            "embedding_model": "embedding-model",
            "max_retries": 0,
        }
    )
    evaluator = RagasEvaluator(sorted(RagasEvaluator.SUPPORTED_METRICS), settings=settings)

    metrics = evaluator._build_metric_instances()

    assert ragas.__version__ == "0.4.3"
    assert hasattr(instructor, "from_openai")
    assert {name: type(metric).__name__ for name, metric in metrics.items()} == {
        "answer_correctness": "AnswerCorrectness",
        "answer_relevancy": "AnswerRelevancy",
        "context_precision": "ContextPrecisionWithReference",
        "context_recall": "ContextRecall",
        "faithfulness": "Faithfulness",
    }
