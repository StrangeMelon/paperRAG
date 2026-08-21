"""Unit tests for the optional RAGAS adapter."""

from __future__ import annotations

from unittest.mock import Mock

import pytest


def test_ragas_extracts_context_text_from_project_chunks() -> None:
    from paper_rag.evaluation.ragas import RagasEvaluator

    evaluator = RagasEvaluator(metrics=["faithfulness"])

    assert evaluator._extract_contexts(
        [{"text": "a"}, {"content": "b"}, {"page_content": "c"}, "d"]
    ) == ["a", "b", "c", "d"]


def test_ragas_requires_an_answer() -> None:
    from paper_rag.evaluation.ragas import RagasEvaluator

    with pytest.raises(ValueError, match="generated_answer"):
        RagasEvaluator(metrics=["faithfulness"]).evaluate("q", [{"text": "ctx"}])


def test_ragas_delegates_to_backend_and_returns_selected_metrics() -> None:
    from paper_rag.evaluation.ragas import RagasEvaluator

    evaluator = RagasEvaluator(metrics=["faithfulness", "answer_relevancy"])
    evaluator._run_ragas = Mock(  # type: ignore[method-assign]
        return_value={"faithfulness": 0.9, "answer_relevancy": 0.8}
    )

    result = evaluator.evaluate(
        query="q",
        retrieved_chunks=[{"text": "ctx"}],
        generated_answer="answer",
    )

    assert result == {"faithfulness": 0.9, "answer_relevancy": 0.8}
    evaluator._run_ragas.assert_called_once_with("q", ["ctx"], "answer", None)


def test_ragas_reports_missing_optional_dependency_clearly(monkeypatch) -> None:
    from paper_rag.evaluation import ragas

    monkeypatch.setattr(ragas, "_import_ragas", lambda: (_ for _ in ()).throw(ImportError("ragas")))
    evaluator = ragas.RagasEvaluator(metrics=["faithfulness"])

    with pytest.raises(ImportError, match="ragas"):
        evaluator._run_ragas("q", ["ctx"], "answer", None)
