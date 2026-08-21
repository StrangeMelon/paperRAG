"""RAGAS 0.4 adapter isolated from the deterministic Custom evaluators."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Sequence
from typing import Any

from .base import BaseEvaluator
from .ragas_models import build_ragas_models
from .ragas_schema import (
    RagasMetricObservation,
    RagasSample,
    RagasSampleEvaluation,
)


def _import_ragas() -> Any:
    try:
        import ragas
    except ImportError as exc:
        raise ImportError("RAGAS evaluation requires the optional 'evaluation' extra") from exc
    return ragas


def extract_contexts(chunks: list[Any]) -> list[str]:
    """Read non-empty text from the chunk shapes produced by the QA pipeline."""
    contexts: list[str] = []
    for chunk in chunks:
        if isinstance(chunk, str):
            text = chunk
        elif isinstance(chunk, dict):
            text = str(chunk.get("text") or chunk.get("content") or chunk.get("page_content") or "")
        else:
            text = str(getattr(chunk, "text", chunk))
        if text.strip():
            contexts.append(text)
    return contexts


class RagasEvaluator(BaseEvaluator):
    SUPPORTED_METRICS = {
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_correctness",
    }
    _EMBEDDING_METRICS = {"answer_relevancy", "answer_correctness"}
    _CONTEXT_METRICS = {"faithfulness", "context_precision", "context_recall"}
    _REFERENCE_METRICS = {"context_precision", "context_recall", "answer_correctness"}

    def __init__(
        self,
        metrics: Sequence[str] | None = None,
        *,
        settings: Any | None = None,
        metric_instances: dict[str, Any] | None = None,
    ) -> None:
        self.metrics = [
            str(item).strip().lower()
            for item in (metrics or ("faithfulness", "answer_relevancy", "context_precision"))
        ]
        unsupported = sorted(set(self.metrics) - self.SUPPORTED_METRICS)
        if unsupported:
            raise ValueError(f"Unsupported ragas metrics: {', '.join(unsupported)}")
        if len(self.metrics) != len(set(self.metrics)):
            raise ValueError("RAGAS metrics must not contain duplicates")
        if settings is None:
            from .. import config as cfg

            settings = cfg.load().evaluation.ragas
        self.settings = settings
        self._metric_instances = metric_instances

    def evaluate(
        self,
        query: str,
        retrieved_chunks: list[dict],
        generated_answer: str | None = None,
        ground_truth: dict | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        self.validate_query(query)
        self.validate_chunks(retrieved_chunks)
        if not generated_answer or not generated_answer.strip():
            raise ValueError("RagasEvaluator requires a non-empty generated_answer")
        return self._run_ragas(
            query, self._extract_contexts(retrieved_chunks), generated_answer, ground_truth
        )

    def _extract_contexts(self, chunks: list[Any]) -> list[str]:
        return extract_contexts(chunks)

    def _run_ragas(
        self, query: str, contexts: list[str], answer: str, ground_truth: dict | None
    ) -> dict[str, float]:
        _import_ragas()
        truth = ground_truth or {}
        sample = RagasSample(
            id="single",
            query=query,
            response=answer,
            retrieved_contexts=contexts,
            retrieved_chunk_ids=[],
            citations=[],
            reference=truth.get("reference_answer"),
            reference_chunk_ids=[],
            expected_abstain=False,
            actual_abstain=None,
            tags=[],
        )
        result = self.evaluate_batch([sample])[0]
        if not result.values:
            messages = [
                item.error["message"]
                for item in result.observations.values()
                if item.error is not None
            ]
            raise RuntimeError("; ".join(messages) or "RAGAS produced no metrics")
        return result.values

    def evaluate_batch(self, samples: list[RagasSample]) -> list[RagasSampleEvaluation]:
        if not isinstance(samples, list):
            raise ValueError("samples must be a list")
        if not samples:
            return []
        return asyncio.run(self._aevaluate_batch(samples))

    def validate_runtime(self) -> None:
        """Validate optional dependencies and model configuration before QA calls."""
        if self._metric_instances is None:
            self._metric_instances = self._build_metric_instances()

    async def _aevaluate_batch(self, samples: list[RagasSample]) -> list[RagasSampleEvaluation]:
        metric_instances = self._metric_instances or self._build_metric_instances()
        missing = sorted(set(self.metrics) - set(metric_instances))
        if missing:
            raise ValueError(f"Missing RAGAS metric instances: {', '.join(missing)}")
        semaphore = asyncio.Semaphore(int(self.settings.max_concurrency))
        results = [
            RagasSampleEvaluation(sample_id=sample.id, observations={}) for sample in samples
        ]
        tasks: list[asyncio.Task[tuple[int, str, RagasMetricObservation]]] = []

        for index, sample in enumerate(samples):
            for metric_name in self.metrics:
                ineligible = self._ineligible_observation(metric_name, sample)
                if ineligible is not None:
                    results[index].observations[metric_name] = ineligible
                    continue
                tasks.append(
                    asyncio.create_task(
                        self._score_metric(
                            index,
                            metric_name,
                            metric_instances[metric_name],
                            sample,
                            semaphore,
                        )
                    )
                )

        if tasks:
            for index, metric_name, observation in await asyncio.gather(*tasks):
                results[index].observations[metric_name] = observation
        return results

    def _build_metric_instances(self) -> dict[str, Any]:
        _import_ragas()
        try:
            from ragas.metrics.collections import (
                AnswerCorrectness,
                AnswerRelevancy,
                ContextPrecisionWithReference,
                ContextRecall,
                Faithfulness,
            )
        except ImportError as exc:
            raise ImportError(
                "Installed RAGAS version does not provide the 0.4 metrics API"
            ) from exc

        require_embeddings = bool(set(self.metrics) & self._EMBEDDING_METRICS)
        bundle = build_ragas_models(self.settings, require_embeddings=require_embeddings)
        factories = {
            "faithfulness": lambda: Faithfulness(llm=bundle.judge),
            "answer_relevancy": lambda: AnswerRelevancy(
                llm=bundle.judge, embeddings=bundle.embeddings
            ),
            "context_precision": lambda: ContextPrecisionWithReference(llm=bundle.judge),
            "context_recall": lambda: ContextRecall(llm=bundle.judge),
            "answer_correctness": lambda: AnswerCorrectness(
                llm=bundle.judge, embeddings=bundle.embeddings
            ),
        }
        return {name: factories[name]() for name in self.metrics}

    def _ineligible_observation(
        self, metric_name: str, sample: RagasSample
    ) -> RagasMetricObservation | None:
        if sample.expected_abstain:
            return RagasMetricObservation.not_applicable()
        if not sample.response.strip():
            return RagasMetricObservation.failure(
                "missing_input", "MissingInput", "response is empty", eligible=True
            )
        if metric_name in self._CONTEXT_METRICS and not sample.retrieved_contexts:
            return RagasMetricObservation.failure(
                "missing_input", "MissingInput", "retrieved_contexts is empty", eligible=True
            )
        if metric_name in self._REFERENCE_METRICS and not (sample.reference or "").strip():
            return RagasMetricObservation.failure(
                "missing_input", "MissingInput", "reference is empty", eligible=True
            )
        return None

    async def _score_metric(
        self,
        index: int,
        metric_name: str,
        metric: Any,
        sample: RagasSample,
        semaphore: asyncio.Semaphore,
    ) -> tuple[int, str, RagasMetricObservation]:
        kwargs = self._metric_kwargs(metric_name, sample)
        started = time.perf_counter()
        try:
            async with semaphore:
                result = await metric.ascore(**kwargs)
            value = float(result.value)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"metric returned an invalid score: {value}")
            observation = RagasMetricObservation.ok(
                value, latency_ms=(time.perf_counter() - started) * 1000
            )
        except Exception as exc:
            observation = RagasMetricObservation.failure(
                "error",
                type(exc).__name__,
                self._sanitise_error(str(exc)),
                eligible=True,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        return index, metric_name, observation

    @staticmethod
    def _metric_kwargs(metric_name: str, sample: RagasSample) -> dict[str, Any]:
        if metric_name == "faithfulness":
            return {
                "user_input": sample.query,
                "response": sample.response,
                "retrieved_contexts": sample.retrieved_contexts,
            }
        if metric_name == "answer_relevancy":
            return {"user_input": sample.query, "response": sample.response}
        if metric_name in {"context_precision", "context_recall"}:
            return {
                "user_input": sample.query,
                "reference": sample.reference,
                "retrieved_contexts": sample.retrieved_contexts,
            }
        if metric_name == "answer_correctness":
            return {
                "user_input": sample.query,
                "response": sample.response,
                "reference": sample.reference,
            }
        raise ValueError(f"Unsupported ragas metric: {metric_name}")

    def _sanitise_error(self, message: str) -> str:
        result = message
        for name in ("api_key", "embedding_api_key"):
            secret = getattr(self.settings, name, None)
            if secret:
                result = result.replace(str(secret), "***")
        return result
