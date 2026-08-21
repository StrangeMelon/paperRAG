"""Batch runner owned by the RAGAS evaluation path."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..store.sqlite_store import get_chunk, get_papers_by_ids, list_papers_by_status
from .ragas import RagasEvaluator, extract_contexts
from .ragas_schema import (
    RagasCase,
    RagasCorpusSnapshot,
    RagasGoldenSetError,
    RagasMetricObservation,
    RagasQueryResult,
    RagasReport,
    RagasSample,
    RagasSampleEvaluation,
    load_ragas_golden_set,
)


def inspect_ragas_corpus(cases: list[RagasCase]) -> RagasCorpusSnapshot:
    indexed = list_papers_by_status("done")
    paper_ids = tuple(sorted(str(paper.paper_id) for paper in indexed))
    if not paper_ids:
        raise RagasGoldenSetError("all_indexed contains no papers with status='done'")
    available = set(paper_ids)
    requested = {paper_id for case in cases for paper_id in case.paper_ids}
    missing = sorted(requested - available)
    if missing:
        raise RagasGoldenSetError(
            f"RAGAS Golden Set references unavailable papers: {', '.join(missing)}"
        )
    papers = get_papers_by_ids(sorted(requested))
    if {paper.paper_id for paper in papers} != requested:
        raise RagasGoldenSetError("RAGAS Golden Set paper validation was incomplete")
    for case in cases:
        for chunk_id in case.reference_chunk_ids:
            chunk = get_chunk(chunk_id)
            if chunk is None:
                raise RagasGoldenSetError(f"{case.id}: reference chunk does not exist: {chunk_id}")
            if chunk.paper_id not in case.paper_ids:
                raise RagasGoldenSetError(
                    f"{case.id}: reference chunk {chunk_id} belongs to {chunk.paper_id}"
                )
    manifest = hashlib.sha256("\n".join(paper_ids).encode()).hexdigest()
    return RagasCorpusSnapshot("all_indexed", paper_ids, manifest)


class RagasEvalRunner:
    def __init__(
        self,
        answer_fn: Callable[..., dict[str, Any]],
        evaluator: RagasEvaluator,
        *,
        top_k: int | None = None,
        query_rewrite: bool = True,
        max_concurrency: int | None = None,
        corpus_inspector: Callable[[list[RagasCase]], RagasCorpusSnapshot] = inspect_ragas_corpus,
    ) -> None:
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be at least 1")
        if max_concurrency is None:
            settings = getattr(evaluator, "settings", None)
            max_concurrency = int(getattr(settings, "max_concurrency", 1))
        if not 1 <= max_concurrency <= 16:
            raise ValueError("max_concurrency must be between 1 and 16")
        self.answer_fn = answer_fn
        self.evaluator = evaluator
        self.top_k = top_k
        self.query_rewrite = query_rewrite
        self.max_concurrency = max_concurrency
        self.corpus_inspector = corpus_inspector

    def run(self, test_set_path: str | Path) -> RagasReport:
        cases, golden_sha256 = load_ragas_golden_set(test_set_path)
        snapshot = self.corpus_inspector(cases)
        validate_runtime = getattr(self.evaluator, "validate_runtime", None)
        if callable(validate_runtime):
            validate_runtime()
        started = time.perf_counter()
        query_results: list[RagasQueryResult] = []
        samples: list[RagasSample] = []

        if len(cases) > 1 and self.max_concurrency > 1:
            workers = min(self.max_concurrency, len(cases))
            with ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="ragas-qa",
            ) as executor:
                collected = list(
                    executor.map(lambda case: self._collect_case(case, snapshot), cases)
                )
        else:
            collected = [self._collect_case(case, snapshot) for case in cases]

        for result, sample in collected:
            query_results.append(result)
            if sample is not None:
                samples.append(sample)

        evaluations = self.evaluator.evaluate_batch(samples)
        evaluated_ids = {evaluation.sample_id for evaluation in evaluations}
        for result in query_results:
            if result.id in evaluated_ids or result.status != "qa_error":
                continue
            evaluations.append(
                RagasSampleEvaluation(
                    sample_id=result.id,
                    observations={
                        metric: RagasMetricObservation.failure(
                            "qa_error",
                            "QAError",
                            "QA answer collection failed",
                            eligible=True,
                        )
                        for metric in self.evaluator.metrics
                    },
                )
            )
        self._merge_evaluations(query_results, evaluations)
        aggregate = self._aggregate(evaluations, self.evaluator.metrics)
        tags_by_id = {result.id: result.tags for result in query_results}
        tags = sorted({tag for result in query_results for tag in result.tags})
        tag_metrics = {
            tag: self._aggregate(
                [
                    evaluation
                    for evaluation in evaluations
                    if tag in tags_by_id[evaluation.sample_id]
                ],
                self.evaluator.metrics,
            )
            for tag in tags
        }
        settings = getattr(self.evaluator, "settings", None)
        try:
            ragas_version = version("ragas")
        except PackageNotFoundError:
            ragas_version = "not-installed"
        return RagasReport(
            run_id=uuid4().hex[:12],
            created_at=datetime.now(UTC).isoformat(),
            test_set=str(test_set_path),
            golden_set_sha256=golden_sha256,
            corpus=snapshot,
            ragas_version=ragas_version,
            judge_model=getattr(settings, "judge_model", None),
            embedding_model=getattr(settings, "embedding_model", None),
            top_k=self.top_k,
            query_rewrite=self.query_rewrite,
            max_concurrency=self.max_concurrency,
            aggregate_metrics=aggregate,
            tag_metrics=tag_metrics,
            query_results=query_results,
            total_elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    def _collect_case(
        self, case: RagasCase, snapshot: RagasCorpusSnapshot
    ) -> tuple[RagasQueryResult, RagasSample | None]:
        started = time.perf_counter()
        result = RagasQueryResult(
            id=case.id,
            query=case.query,
            reference=case.reference_answer,
            reference_chunk_ids=case.reference_chunk_ids,
            expected_abstain=case.expected_abstain,
            tags=case.tags,
        )
        try:
            output = self.answer_fn(
                case.query,
                paper_ids=case.paper_ids or list(snapshot.paper_ids),
                top_k_override=self.top_k,
                query_rewrite_enabled=self.query_rewrite,
                evaluation_parallel=self.max_concurrency > 1,
            )
            candidate_chunks = list(output.get("chunks") or [])
            result.response = str(output.get("answer") or "")
            result.citations = [str(item) for item in output.get("citations", [])]
            trace = output.get("trace") or {}
            result.actual_abstain = (trace.get("abstain") or {}).get("decision")
            if result.actual_abstain in {"no_evidence", "no_chunks"}:
                chunks = []
            elif "evidence_chunks" in output:
                chunks = list(output.get("evidence_chunks") or [])
            else:
                chunks = candidate_chunks
            result.retrieved_chunk_ids = [
                str(chunk.get("chunk_id", chunk.get("id", ""))) for chunk in chunks
            ]
            contexts = extract_contexts(chunks)
            sample = RagasSample(
                id=case.id,
                query=case.query,
                response=result.response,
                retrieved_contexts=contexts,
                retrieved_chunk_ids=result.retrieved_chunk_ids,
                citations=result.citations,
                reference=case.reference_answer,
                reference_chunk_ids=case.reference_chunk_ids,
                expected_abstain=case.expected_abstain,
                actual_abstain=result.actual_abstain,
                tags=case.tags,
            )
        except Exception as exc:
            result.status = "qa_error"
            result.errors.append(
                {
                    "stage": "qa",
                    "type": type(exc).__name__,
                    "message": self._sanitise_error(str(exc)),
                }
            )
            sample = None
        result.qa_latency_ms = (time.perf_counter() - started) * 1000
        return result, sample

    def _sanitise_error(self, message: str) -> str:
        settings = getattr(self.evaluator, "settings", None)
        result = message
        for name in ("api_key", "embedding_api_key"):
            secret = getattr(settings, name, None)
            if secret:
                result = result.replace(str(secret), "***")
        return result

    @staticmethod
    def _merge_evaluations(
        query_results: list[RagasQueryResult],
        evaluations: list[RagasSampleEvaluation],
    ) -> None:
        by_id = {item.id: item for item in query_results}
        for evaluation in evaluations:
            result = by_id[evaluation.sample_id]
            result.metrics = evaluation.values
            result.metric_details = {
                name: observation.to_dict() for name, observation in evaluation.observations.items()
            }
            result.errors.extend(
                {"stage": "ragas", "metric": name, **observation.error}
                for name, observation in evaluation.observations.items()
                if observation.error is not None and observation.status == "error"
            )
            if result.status != "qa_error":
                result.status = evaluation.status
                result.ragas_latency_ms = sum(
                    observation.latency_ms for observation in evaluation.observations.values()
                )

    @staticmethod
    def _aggregate(
        evaluations: list[RagasSampleEvaluation], metrics: list[str]
    ) -> dict[str, dict[str, float | int]]:
        aggregate: dict[str, dict[str, float | int]] = {}
        for metric in metrics:
            observations = [
                item.observations[metric]
                for item in evaluations
                if metric in item.observations and item.observations[metric].eligible
            ]
            values = [
                item.value
                for item in observations
                if item.status == "ok" and item.value is not None
            ]
            eligible_count = len(observations)
            aggregate[metric] = {
                "mean": sum(values) / len(values) if values else 0.0,
                "count": len(values),
                "eligible_count": eligible_count,
                "coverage": len(values) / eligible_count if eligible_count else 0.0,
            }
        return aggregate
