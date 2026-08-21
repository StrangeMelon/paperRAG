"""Configuration contract tests for evaluation settings."""

from __future__ import annotations


def test_default_config_contains_disabled_custom_evaluation() -> None:
    from paper_rag import config

    config.load.cache_clear()
    loaded = config.load()

    assert loaded.evaluation.enabled is False
    assert loaded.evaluation.provider == "custom"
    assert loaded.evaluation.backends == ["custom"]
    assert "hit_rate" in loaded.evaluation.metrics
    assert loaded.evaluation.max_concurrency == 8


def test_default_config_contains_isolated_ragas_settings(monkeypatch) -> None:
    from paper_rag import config

    monkeypatch.delenv("RAGAS_MODEL", raising=False)
    monkeypatch.delenv("RAGAS_EMBEDDING_MODEL", raising=False)
    config.load.cache_clear()
    ragas = config.load().evaluation.ragas
    config.load.cache_clear()

    assert ragas.golden_set == "tests/fixtures/evaluation/ragas_golden.json"
    assert ragas.metrics == [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    assert ragas.max_concurrency == 4
    assert ragas.timeout_sec == 120
    assert ragas.judge_model is None
    assert ragas.embedding_model is None
