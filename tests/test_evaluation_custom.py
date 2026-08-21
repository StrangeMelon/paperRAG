"""Unit tests for deterministic evaluation metrics."""

from __future__ import annotations

import pytest


def _chunk(chunk_id: str, paper_id: str = "p1") -> dict:
    return {"chunk_id": chunk_id, "paper_id": paper_id, "text": f"text {chunk_id}"}


def test_custom_evaluator_computes_retrieval_metrics() -> None:
    from paper_rag.evaluation.custom import CustomEvaluator

    evaluator = CustomEvaluator(metrics=["hit_rate", "mrr", "recall", "paper_hit_rate"])
    result = evaluator.evaluate(
        query="q",
        retrieved_chunks=[_chunk("c1", "p1"), _chunk("c2", "p2"), _chunk("c3", "p2")],
        ground_truth={"chunk_ids": ["c2", "c9"], "paper_ids": ["p2"]},
    )

    assert result == {
        "hit_rate": 1.0,
        "mrr": 0.5,
        "recall": 0.5,
        "paper_hit_rate": 1.0,
    }


def test_custom_evaluator_supports_citations_and_abstain_metrics() -> None:
    from paper_rag.evaluation.custom import CustomEvaluator

    evaluator = CustomEvaluator(
        metrics=["citation_precision", "citation_recall", "abstain_accuracy"]
    )
    result = evaluator.evaluate(
        query="q",
        retrieved_chunks=[_chunk("c1"), _chunk("c2")],
        generated_answer="answer",
        ground_truth={
            "chunk_ids": ["c1", "c2"],
            "citations": ["c1", "missing"],
            "expected_abstain": False,
        },
        citations=["c1", "missing"],
        abstain_decision="confident",
    )

    assert result["citation_precision"] == 0.5
    assert result["citation_recall"] == 0.5
    assert result["abstain_accuracy"] == 1.0


def test_custom_evaluator_empty_retrieval_is_a_zero_score_not_an_exception() -> None:
    from paper_rag.evaluation.custom import CustomEvaluator

    result = CustomEvaluator(metrics=["hit_rate", "mrr", "recall"]).evaluate(
        query="q",
        retrieved_chunks=[],
        ground_truth={"chunk_ids": ["expected"]},
    )

    assert result == {"hit_rate": 0.0, "mrr": 0.0, "recall": 0.0}


def test_custom_evaluator_rejects_unknown_metrics() -> None:
    from paper_rag.evaluation.custom import CustomEvaluator

    with pytest.raises(ValueError, match="Unsupported custom metrics"):
        CustomEvaluator(metrics=["faithfulness"])
