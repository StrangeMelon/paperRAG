"""Strict, deterministic Custom evaluation for the retrieval pipeline."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..store.sqlite_store import get_chunk, get_papers_by_ids, list_papers_by_status
from .custom import CustomEvaluator

RETRIEVAL_SCHEMA_VERSION = "retrieval.v1"
RETRIEVAL_METRICS = ("hit_rate", "mrr", "recall", "paper_hit_rate")


@dataclass(frozen=True)
class RetrievalCase:
    id: str
    query: str
    expected_chunk_ids: list[str]
    expected_paper_ids: list[str]
    search_scope: list[str] | None
    reference_answer: str
    tags: list[str] = field(default_factory=list)
    notes: str | None = None


@dataclass(frozen=True)
class CorpusSnapshot:
    selection: str
    paper_ids: tuple[str, ...]
    manifest_sha256: str


class RetrievalGoldenSetError(ValueError):
    """Raised when the Golden Set or current SQLite index is invalid."""


def _sha256_ids(ids: list[str] | tuple[str, ...]) -> str:
    payload = "\n".join(ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        with Path(path).open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalGoldenSetError(f"cannot read retrieval Golden Set: {path}") from exc
    if not isinstance(data, dict):
        raise RetrievalGoldenSetError("retrieval Golden Set root must be an object")
    return data


def load_retrieval_golden_set(path: str | Path) -> tuple[list[RetrievalCase], str]:
    """Load and structurally validate the retrieval-only Golden Set."""
    raw = _read_json(path)
    if raw.get("schema_version") != RETRIEVAL_SCHEMA_VERSION:
        raise RetrievalGoldenSetError(f"schema_version must be {RETRIEVAL_SCHEMA_VERSION!r}")
    corpus = raw.get("corpus")
    if not isinstance(corpus, dict) or corpus.get("selection") != "all_indexed":
        raise RetrievalGoldenSetError("corpus.selection must be 'all_indexed'")
    items = raw.get("test_cases")
    if not isinstance(items, list) or not items:
        raise RetrievalGoldenSetError("test_cases must be a non-empty list")

    cases: list[RetrievalCase] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise RetrievalGoldenSetError(f"test_cases[{index}] must be an object")
        case_id = str(item.get("id", "")).strip()
        query = str(item.get("query", "")).strip()
        if not case_id or case_id in seen_ids:
            raise RetrievalGoldenSetError(f"test_cases[{index}] has a missing or duplicate id")
        seen_ids.add(case_id)
        if not query:
            raise RetrievalGoldenSetError(f"{case_id}: query must not be empty")
        chunks = _string_list(item.get("expected_chunk_ids"), f"{case_id}.expected_chunk_ids")
        papers = _string_list(item.get("expected_paper_ids"), f"{case_id}.expected_paper_ids")
        if not chunks:
            raise RetrievalGoldenSetError(f"{case_id}: expected_chunk_ids must not be empty")
        if not papers:
            raise RetrievalGoldenSetError(f"{case_id}: expected_paper_ids must not be empty")
        reference_answer = str(item.get("reference_answer", "")).strip()
        if not reference_answer:
            raise RetrievalGoldenSetError(f"{case_id}: reference_answer must not be empty")
        if len(chunks) != len(set(chunks)):
            raise RetrievalGoldenSetError(f"{case_id}: expected_chunk_ids contains duplicates")
        if len(papers) != len(set(papers)):
            raise RetrievalGoldenSetError(f"{case_id}: expected_paper_ids contains duplicates")
        scope = item.get("search_scope")
        if scope is None:
            scope_ids = None
        elif isinstance(scope, dict):
            scope_ids = _string_list(scope.get("paper_ids"), f"{case_id}.search_scope.paper_ids")
            if not scope_ids:
                raise RetrievalGoldenSetError(
                    f"{case_id}: search_scope.paper_ids must not be empty"
                )
            if not set(papers).issubset(scope_ids):
                raise RetrievalGoldenSetError(
                    f"{case_id}: expected_paper_ids must be within search_scope.paper_ids"
                )
        else:
            raise RetrievalGoldenSetError(f"{case_id}: search_scope must be null or an object")
        tags = item.get("tags", [])
        if not isinstance(tags, list) or any(not str(tag).strip() for tag in tags):
            raise RetrievalGoldenSetError(f"{case_id}: tags must be a list of non-empty strings")
        cases.append(
            RetrievalCase(
                id=case_id,
                query=query,
                expected_chunk_ids=chunks,
                expected_paper_ids=papers,
                search_scope=scope_ids,
                reference_answer=reference_answer,
                tags=[str(tag) for tag in tags],
                notes=str(item["notes"]) if item.get("notes") is not None else None,
            )
        )
    return cases, hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_corpus() -> CorpusSnapshot:
    papers = list_papers_by_status("done")
    ids = tuple(sorted(str(paper.paper_id) for paper in papers))
    if not ids:
        raise RetrievalGoldenSetError("all_indexed contains no papers with status='done'")
    return CorpusSnapshot("all_indexed", ids, _sha256_ids(ids))


def validate_retrieval_cases(cases: list[RetrievalCase], snapshot: CorpusSnapshot) -> None:
    """Validate Golden Set references against one fixed SQLite corpus snapshot."""
    available = set(snapshot.paper_ids)
    referenced = {paper for case in cases for paper in case.expected_paper_ids}
    scoped = {paper for case in cases for paper in (case.search_scope or [])}
    missing = sorted((referenced | scoped) - available)
    if missing:
        raise RetrievalGoldenSetError(
            f"Golden Set references unavailable papers: {', '.join(missing)}"
        )
    all_papers = get_papers_by_ids(sorted(referenced | scoped))
    by_id = {paper.paper_id: paper for paper in all_papers}
    incomplete = sorted(paper_id for paper_id, paper in by_id.items() if paper.status != "done")
    if incomplete:
        raise RetrievalGoldenSetError(
            f"Golden Set references non-done papers: {', '.join(incomplete)}"
        )

    for case in cases:
        for chunk_id in case.expected_chunk_ids:
            chunk = get_chunk(chunk_id)
            if chunk is None:
                raise RetrievalGoldenSetError(
                    f"{case.id}: expected chunk does not exist: {chunk_id}"
                )
            if chunk.paper_id not in case.expected_paper_ids:
                raise RetrievalGoldenSetError(
                    f"{case.id}: chunk {chunk_id} belongs to {chunk.paper_id}, "
                    f"not expected_paper_ids"
                )


class RetrievalEvalRunner:
    """Run retrieval-only cases against a fixed corpus snapshot."""

    def __init__(
        self,
        retrieve_fn: Callable[..., list[dict]],
        *,
        top_k: int = 8,
        rewrite_enabled: bool = True,
        max_concurrency: int = 8,
        metrics: tuple[str, ...] = RETRIEVAL_METRICS,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        unsupported = set(metrics) - set(RETRIEVAL_METRICS)
        if unsupported:
            raise ValueError(f"unsupported retrieval metrics: {', '.join(sorted(unsupported))}")
        self.retrieve_fn = retrieve_fn
        self.top_k = top_k
        self.rewrite_enabled = rewrite_enabled
        self.max_concurrency = max_concurrency
        self.evaluator = CustomEvaluator(metrics=metrics)

    def run(self, test_set_path: str | Path) -> dict[str, Any]:
        cases, golden_sha256 = load_retrieval_golden_set(test_set_path)
        snapshot = snapshot_corpus()
        validate_retrieval_cases(cases, snapshot)
        started = time.perf_counter()
        # A worker owns one complete case: rewrite, dense/sparse retrieval, rerank, and scoring.
        # Query Rewrite must not be the condition that decides whether evaluation is concurrent.
        if len(cases) > 1 and self.max_concurrency > 1:
            workers = min(self.max_concurrency, len(cases))
            with ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="retrieval-eval"
            ) as executor:
                # executor.map preserves Golden Set order while allowing retrieval calls to overlap.
                results = list(executor.map(lambda case: self._run_case(case, snapshot), cases))
        else:
            results = [self._run_case(case, snapshot) for case in cases]

        successful = [result for result in results if result["status"] == "ok"]
        aggregate = {
            metric: sum(result["metrics"][metric] for result in successful) / len(successful)
            for metric in RETRIEVAL_METRICS
            if successful
        }
        return {
            "evaluation": {
                "mode": "retrieval",
                "top_k": self.top_k,
                "query_rewrite": self.rewrite_enabled,
                "max_concurrency": self.max_concurrency,
                "golden_set": str(test_set_path),
                "golden_set_sha256": golden_sha256,
                "corpus_selection": snapshot.selection,
                "corpus_paper_count": len(snapshot.paper_ids),
                "corpus_manifest_sha256": snapshot.manifest_sha256,
                "golden_case_count": len(cases),
                "valid_case_count": len(cases),
                "successful_case_count": len(successful),
            },
            "aggregate_metrics": aggregate,
            "query_results": results,
            "total_elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    def _run_case(self, case: RetrievalCase, snapshot: CorpusSnapshot) -> dict[str, Any]:
        case_started = time.perf_counter()
        try:
            retrieve_kwargs = {
                "paper_ids": case.search_scope or list(snapshot.paper_ids),
                "top_k": self.top_k,
                "rewrite_enabled": self.rewrite_enabled,
                "evaluation_parallel": True,
            }
            try:
                chunks = self.retrieve_fn(case.query, **retrieve_kwargs)
            except TypeError as exc:
                # Keep injected legacy retrieve functions usable in focused tests/tools.
                if "evaluation_parallel" not in str(exc):
                    raise
                retrieve_kwargs.pop("evaluation_parallel")
                chunks = self.retrieve_fn(case.query, **retrieve_kwargs)
            chunks = list(chunks or [])
            retrieved_ids = [str(chunk.get("chunk_id", "")) for chunk in chunks]
            retrieved_papers = sorted(
                {str(chunk["paper_id"]) for chunk in chunks if chunk.get("paper_id")}
            )
            metrics = self.evaluator.evaluate(
                case.query,
                chunks,
                ground_truth={
                    "chunk_ids": case.expected_chunk_ids,
                    "paper_ids": case.expected_paper_ids,
                },
            )
            return {
                "id": case.id,
                "query": case.query,
                "search_scope": case.search_scope,
                "expected_chunk_ids": case.expected_chunk_ids,
                "expected_paper_ids": case.expected_paper_ids,
                "reference_answer": case.reference_answer,
                "retrieved_chunk_ids": retrieved_ids,
                "retrieved_paper_ids": retrieved_papers,
                "metrics": metrics,
                "latency_ms": round((time.perf_counter() - case_started) * 1000, 1),
                "status": "ok",
                "errors": [],
                "tags": case.tags,
            }
        except Exception as exc:
            return {
                "id": case.id,
                "query": case.query,
                "search_scope": case.search_scope,
                "expected_chunk_ids": case.expected_chunk_ids,
                "expected_paper_ids": case.expected_paper_ids,
                "reference_answer": case.reference_answer,
                "retrieved_chunk_ids": [],
                "retrieved_paper_ids": [],
                "metrics": {},
                "latency_ms": round((time.perf_counter() - case_started) * 1000, 1),
                "status": "error",
                "errors": [{"type": type(exc).__name__, "message": str(exc)}],
                "tags": case.tags,
            }


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or any(not str(item).strip() for item in value):
        raise RetrievalGoldenSetError(f"{field} must be a non-empty list of strings")
    return [str(item) for item in value]


__all__ = [
    "RETRIEVAL_METRICS",
    "RETRIEVAL_SCHEMA_VERSION",
    "CorpusSnapshot",
    "RetrievalCase",
    "RetrievalEvalRunner",
    "RetrievalGoldenSetError",
    "load_retrieval_golden_set",
    "snapshot_corpus",
    "validate_retrieval_cases",
]
