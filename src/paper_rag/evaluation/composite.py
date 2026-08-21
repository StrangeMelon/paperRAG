"""Composition of independent evaluation backends."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .base import BaseEvaluator


@dataclass
class EvaluationResult:
    metrics: dict[str, float] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)
    failed: bool = False


class CompositeEvaluator(BaseEvaluator):
    def __init__(self, evaluators: Iterable[tuple[str, BaseEvaluator]]) -> None:
        self.evaluators = list(evaluators)
        if not self.evaluators:
            raise ValueError("CompositeEvaluator requires at least one evaluator")

    def evaluate(
        self,
        query: str,
        retrieved_chunks: list[dict],
        generated_answer: str | None = None,
        ground_truth: dict | None = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        metrics: dict[str, float] = {}
        errors: list[dict[str, str]] = []
        for backend, evaluator in self.evaluators:
            try:
                values = evaluator.evaluate(
                    query=query,
                    retrieved_chunks=retrieved_chunks,
                    generated_answer=generated_answer,
                    ground_truth=ground_truth,
                    **kwargs,
                )
                metrics.update({f"{backend}.{key}": float(value) for key, value in values.items()})
            except Exception as exc:
                errors.append({"backend": backend, "type": type(exc).__name__, "message": str(exc)})
        return EvaluationResult(metrics=metrics, errors=errors, failed=not metrics and bool(errors))
