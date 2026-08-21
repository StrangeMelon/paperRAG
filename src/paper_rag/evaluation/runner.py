"""Golden Set batch runner for the public QA contract."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import BaseEvaluator
from .composite import EvaluationResult


@dataclass
class GoldenCase:
    id: str
    query: str
    expected_chunk_ids: list[str] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)
    reference_answer: str | None = None
    expected_abstain: bool | None = None
    paper_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass
class QueryResult:
    id: str
    query: str
    status: str = "ok"
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    answer: str | None = None
    citations: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class EvalReport:
    evaluator_name: str
    test_set_path: str
    query_results: list[QueryResult]
    aggregate_metrics: dict[str, float]
    total_elapsed_ms: float

    @property
    def query_count(self) -> int:
        return len(self.query_results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator": self.evaluator_name,
            "test_set": self.test_set_path,
            "query_count": self.query_count,
            "aggregate_metrics": self.aggregate_metrics,
            "total_elapsed_ms": round(self.total_elapsed_ms, 1),
            "query_results": [result.__dict__ for result in self.query_results],
        }


def load_golden_set(path: str | Path) -> list[GoldenCase]:
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("test_cases"), list):
        raise ValueError("Golden Set must contain a test_cases list")
    return [
        GoldenCase(
            id=str(item.get("id", index)),
            query=str(item["query"]),
            expected_chunk_ids=[str(x) for x in item.get("expected_chunk_ids", [])],
            expected_sources=[str(x) for x in item.get("expected_sources", [])],
            reference_answer=item.get("reference_answer"),
            expected_abstain=item.get("expected_abstain"),
            paper_ids=[str(x) for x in item.get("paper_ids", [])],
            tags=[str(x) for x in item.get("tags", [])],
        )
        for index, item in enumerate(data["test_cases"])
    ]


class EvalRunner:
    def __init__(self, answer_fn: Callable[..., dict[str, Any]], evaluator: BaseEvaluator) -> None:
        self.answer_fn = answer_fn
        self.evaluator = evaluator

    def run(self, test_set_path: str | Path, *, paper_ids: list[str] | None = None) -> EvalReport:
        cases = load_golden_set(test_set_path)
        started = time.perf_counter()
        results = [self._run_case(case, paper_ids=paper_ids) for case in cases]
        all_metrics = sorted({key for result in results for key in result.metrics})
        aggregate = {
            key: sum(result.metrics[key] for result in results if key in result.metrics)
            / sum(key in result.metrics for result in results)
            for key in all_metrics
        }
        return EvalReport(
            type(self.evaluator).__name__,
            str(test_set_path),
            results,
            aggregate,
            (time.perf_counter() - started) * 1000,
        )

    def _run_case(self, case: GoldenCase, *, paper_ids: list[str] | None) -> QueryResult:
        started = time.perf_counter()
        result = QueryResult(id=case.id, query=case.query)
        try:
            output = self.answer_fn(case.query, paper_ids=case.paper_ids or paper_ids)
            chunks = output.get("chunks") or []
            result.answer = output.get("answer")
            result.citations = [str(item) for item in output.get("citations", [])]
            result.retrieved_chunk_ids = [
                str(chunk.get("chunk_id", chunk.get("id", ""))) for chunk in chunks
            ]
            trace = output.get("trace") or {}
            abstain = (trace.get("abstain") or {}).get("decision")
            ground_truth = {
                "chunk_ids": case.expected_chunk_ids,
                "paper_ids": case.expected_sources,
                "reference_answer": case.reference_answer,
                "expected_abstain": case.expected_abstain,
            }
            evaluated = self.evaluator.evaluate(
                case.query,
                chunks,
                result.answer,
                ground_truth,
                citations=result.citations,
                abstain_decision=abstain,
            )
            if isinstance(evaluated, EvaluationResult):
                result.metrics, result.errors = evaluated.metrics, evaluated.errors
                result.status = "error" if evaluated.failed else "ok"
            else:
                result.metrics = evaluated
        except Exception as exc:
            result.status = "error"
            result.errors.append({"type": type(exc).__name__, "message": str(exc)})
        result.latency_ms = (time.perf_counter() - started) * 1000
        return result
