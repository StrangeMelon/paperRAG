"""Lazy model construction for the isolated RAGAS evaluation path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RagasModelBundle:
    judge: Any
    embeddings: Any | None


def build_ragas_models(settings: Any, *, require_embeddings: bool) -> RagasModelBundle:
    missing = [
        name
        for name in ("base_url", "api_key", "judge_model")
        if not str(getattr(settings, name, "") or "").strip()
    ]
    if missing:
        raise ValueError(f"RAGAS judge configuration is incomplete: {', '.join(missing)}")
    if require_embeddings and not str(getattr(settings, "embedding_model", "") or "").strip():
        raise ValueError("RAGAS embedding_model is required by the selected metrics")

    try:
        from openai import AsyncOpenAI
        from ragas.embeddings.base import embedding_factory
        from ragas.llms import llm_factory
    except ImportError as exc:
        raise ImportError("RAGAS evaluation requires the optional 'evaluation' extra") from exc

    judge_client = AsyncOpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=settings.timeout_sec,
        max_retries=settings.max_retries,
    )
    judge = llm_factory(
        settings.judge_model,
        provider="openai",
        client=judge_client,
        **dict(settings.judge_options),
    )
    embeddings = None
    if require_embeddings:
        embedding_client = AsyncOpenAI(
            api_key=settings.embedding_api_key or settings.api_key,
            base_url=settings.embedding_base_url or settings.base_url,
            timeout=settings.timeout_sec,
            max_retries=settings.max_retries,
        )
        embeddings = embedding_factory(
            "openai",
            model=settings.embedding_model,
            client=embedding_client,
            interface="modern",
        )
    return RagasModelBundle(judge=judge, embeddings=embeddings)
