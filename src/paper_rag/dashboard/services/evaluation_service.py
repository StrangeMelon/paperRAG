"""Evaluation history persistence and execution service."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from ...evaluation.composite import CompositeEvaluator
from ...evaluation.custom import CustomEvaluator
from ...evaluation.ragas import RagasEvaluator
from ...evaluation.retrieval import RETRIEVAL_METRICS, RetrievalEvalRunner
from ...evaluation.runner import EvalRunner
from .jsonl_store import JsonlStore


class EvaluationHistory:
    def __init__(self, path: str | Path) -> None:
        self._store = JsonlStore(path, id_field="run_id")

    def append(self, record: dict[str, Any]) -> None:
        self._store.append(record)

    def list(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        records = self._store.load()
        return records[:limit] if limit else records

    def get(self, run_id: str) -> dict[str, Any] | None:
        return self._store.get(run_id)


class EvaluationService:
    def __init__(self, history: EvaluationHistory) -> None:
        self.history = history

    def run(
        self,
        *,
        test_set: str | Path,
        backend: str,
        custom_metrics: list[str],
        ragas_metrics: list[str],
        mode: str = "qa",
        top_k: int = 8,
        query_rewrite: bool = True,
        max_concurrency: int | None = None,
    ) -> dict[str, Any]:
        if mode == "retrieval":
            if backend != "custom":
                raise ValueError("retrieval mode only supports the custom backend")
            from ... import config as cfg
            from ...retrieve.pipeline import retrieve_round

            report = RetrievalEvalRunner(
                retrieve_round,
                top_k=top_k,
                rewrite_enabled=query_rewrite,
                max_concurrency=cfg.load().evaluation.max_concurrency,
                metrics=tuple(custom_metrics) if custom_metrics else RETRIEVAL_METRICS,
            ).run(test_set)
            record = {
                "run_id": uuid4().hex[:12],
                "created_at": datetime.now(UTC).isoformat(),
                "backend": "custom",
                **report,
            }
            self.history.append(record)
            return record
        if mode != "qa":
            raise ValueError(f"unsupported evaluation mode: {mode}")
        if backend == "ragas":
            from ... import config as cfg
            from ...evaluation.ragas_runner import RagasEvalRunner
            from ...rag.qa_agentic import answer

            settings = cfg.load().evaluation.ragas
            resolved_concurrency = (
                settings.max_concurrency if max_concurrency is None else max_concurrency
            )
            settings = settings.model_copy(update={"max_concurrency": resolved_concurrency})
            evaluator = RagasEvaluator(ragas_metrics or settings.metrics, settings=settings)
            record = (
                RagasEvalRunner(
                    answer,
                    evaluator,
                    top_k=top_k,
                    query_rewrite=query_rewrite,
                    max_concurrency=resolved_concurrency,
                )
                .run(test_set)
                .to_dict()
            )
            self.history.append(record)
            return record
        evaluator = _build_evaluator(backend, custom_metrics, ragas_metrics)
        from ...rag.qa_agentic import answer

        report = EvalRunner(answer_fn=answer, evaluator=evaluator).run(test_set)
        record = {
            "run_id": uuid4().hex[:12],
            "created_at": datetime.now(UTC).isoformat(),
            "backend": backend,
            **report.to_dict(),
        }
        self.history.append(record)
        return record


def _build_evaluator(backend: str, custom_metrics: list[str], ragas_metrics: list[str]):
    if backend == "custom":
        return CustomEvaluator(custom_metrics)
    if backend == "ragas":
        return RagasEvaluator(ragas_metrics)
    if backend == "composite":
        return CompositeEvaluator(
            [
                ("custom", CustomEvaluator(custom_metrics)),
                ("ragas", RagasEvaluator(ragas_metrics)),
            ]
        )
    raise ValueError(f"unsupported evaluation backend: {backend}")
