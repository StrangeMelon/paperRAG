"""Configuration validation for RAGAS model construction."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_ragas_model_factory_fails_before_import_when_judge_config_is_missing() -> None:
    from paper_rag.evaluation.ragas_models import build_ragas_models

    settings = SimpleNamespace(base_url=None, api_key=None, judge_model=None)

    with pytest.raises(ValueError, match="base_url, api_key, judge_model"):
        build_ragas_models(settings, require_embeddings=False)


def test_ragas_model_factory_requires_embedding_model_only_for_embedding_metrics() -> None:
    from paper_rag.evaluation.ragas_models import build_ragas_models

    settings = SimpleNamespace(
        base_url="https://example.test/v1",
        api_key="secret",
        judge_model="judge",
        embedding_model=None,
    )

    with pytest.raises(ValueError, match="embedding_model"):
        build_ragas_models(settings, require_embeddings=True)
