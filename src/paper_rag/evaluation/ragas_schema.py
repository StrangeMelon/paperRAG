"""Data contracts owned exclusively by the RAGAS evaluation path."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RAGAS_GOLDEN_SCHEMA_VERSION = "ragas-eval.v1"
RAGAS_REPORT_SCHEMA_VERSION = "ragas-report.v1"


class RagasGoldenSetError(ValueError):
    """Raised when a RAGAS-only Golden Set is structurally invalid."""


@dataclass(frozen=True)
class RagasCase:
    id: str
    query: str
    paper_ids: list[str]
    reference_answer: str | None
    reference_chunk_ids: list[str]
    expected_abstain: bool
    tags: list[str]
    notes: str | None = None


@dataclass(frozen=True)
class RagasSample:
    id: str
    query: str
    response: str
    retrieved_contexts: list[str]
    retrieved_chunk_ids: list[str]
    citations: list[str]
    reference: str | None
    reference_chunk_ids: list[str]
    expected_abstain: bool
    actual_abstain: str | None
    tags: list[str]


@dataclass(frozen=True)
class RagasCorpusSnapshot:
    selection: str
    paper_ids: tuple[str, ...]
    manifest_sha256: str


@dataclass
class RagasMetricObservation:
    status: str
    eligible: bool
    value: float | None = None
    latency_ms: float = 0.0
    error: dict[str, str] | None = None

    @classmethod
    def ok(cls, value: float, *, latency_ms: float) -> RagasMetricObservation:
        return cls(status="ok", eligible=True, value=value, latency_ms=latency_ms)

    @classmethod
    def failure(
        cls,
        status: str,
        error_type: str,
        message: str,
        *,
        eligible: bool,
        latency_ms: float = 0.0,
    ) -> RagasMetricObservation:
        return cls(
            status=status,
            eligible=eligible,
            latency_ms=latency_ms,
            error={"type": error_type, "message": message},
        )

    @classmethod
    def not_applicable(cls) -> RagasMetricObservation:
        return cls(status="not_applicable", eligible=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "eligible": self.eligible,
            "value": self.value,
            "latency_ms": round(self.latency_ms, 1),
            "error": self.error,
        }


@dataclass
class RagasSampleEvaluation:
    sample_id: str
    observations: dict[str, RagasMetricObservation]

    @property
    def values(self) -> dict[str, float]:
        return {
            name: observation.value
            for name, observation in self.observations.items()
            if observation.status == "ok" and observation.value is not None
        }

    @property
    def status(self) -> str:
        statuses = {item.status for item in self.observations.values()}
        if statuses and statuses == {"not_applicable"}:
            return "not_applicable"
        if statuses and statuses == {"ok"}:
            return "ok"
        if "ok" in statuses:
            return "partial"
        return "error"


@dataclass
class RagasQueryResult:
    id: str
    query: str
    response: str = ""
    reference: str | None = None
    reference_chunk_ids: list[str] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    expected_abstain: bool = False
    actual_abstain: str | None = None
    tags: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    metric_details: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)
    status: str = "ok"
    qa_latency_ms: float = 0.0
    ragas_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "response": self.response,
            "reference": self.reference,
            "reference_chunk_ids": self.reference_chunk_ids,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "citations": self.citations,
            "expected_abstain": self.expected_abstain,
            "actual_abstain": self.actual_abstain,
            "tags": self.tags,
            "metrics": self.metrics,
            "metric_details": self.metric_details,
            "errors": self.errors,
            "status": self.status,
            "qa_latency_ms": round(self.qa_latency_ms, 1),
            "ragas_latency_ms": round(self.ragas_latency_ms, 1),
        }


@dataclass
class RagasReport:
    run_id: str
    created_at: str
    test_set: str
    golden_set_sha256: str
    corpus: RagasCorpusSnapshot
    ragas_version: str
    judge_model: str | None
    embedding_model: str | None
    top_k: int | None
    query_rewrite: bool
    max_concurrency: int
    aggregate_metrics: dict[str, dict[str, float | int]]
    tag_metrics: dict[str, dict[str, dict[str, float | int]]]
    query_results: list[RagasQueryResult]
    total_elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RAGAS_REPORT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "backend": "ragas",
            "test_set": self.test_set,
            "evaluation": {
                "mode": "ragas",
                "ragas_version": self.ragas_version,
                "judge_model": self.judge_model,
                "embedding_model": self.embedding_model,
                "top_k": self.top_k,
                "query_rewrite": self.query_rewrite,
                "max_concurrency": self.max_concurrency,
                "golden_set_sha256": self.golden_set_sha256,
                "corpus_selection": self.corpus.selection,
                "corpus_paper_count": len(self.corpus.paper_ids),
                "corpus_manifest_sha256": self.corpus.manifest_sha256,
            },
            "aggregate_metrics": self.aggregate_metrics,
            "tag_metrics": self.tag_metrics,
            "query_results": [item.to_dict() for item in self.query_results],
            "total_elapsed_ms": round(self.total_elapsed_ms, 1),
        }


def _string_list(value: Any, field_name: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list):
        raise RagasGoldenSetError(f"{field_name} must be a list")
    result = [str(item).strip() for item in value]
    if any(not item for item in result):
        raise RagasGoldenSetError(f"{field_name} must contain non-empty strings")
    if not allow_empty and not result:
        raise RagasGoldenSetError(f"{field_name} must not be empty")
    if len(result) != len(set(result)):
        raise RagasGoldenSetError(f"{field_name} contains duplicates")
    return result


def load_ragas_golden_set(path: str | Path) -> tuple[list[RagasCase], str]:
    source = Path(path)
    try:
        raw_bytes = source.read_bytes()
        data = json.loads(raw_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise RagasGoldenSetError(f"cannot read RAGAS Golden Set: {source}") from exc
    if not isinstance(data, dict):
        raise RagasGoldenSetError("RAGAS Golden Set root must be an object")
    if data.get("schema_version") != RAGAS_GOLDEN_SCHEMA_VERSION:
        raise RagasGoldenSetError(f"schema_version must be {RAGAS_GOLDEN_SCHEMA_VERSION!r}")
    corpus = data.get("corpus")
    if not isinstance(corpus, dict) or corpus.get("selection") != "all_indexed":
        raise RagasGoldenSetError("corpus.selection must be 'all_indexed'")
    items = data.get("test_cases")
    if not isinstance(items, list) or not items:
        raise RagasGoldenSetError("test_cases must be a non-empty list")

    cases: list[RagasCase] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise RagasGoldenSetError(f"test_cases[{index}] must be an object")
        case_id = str(item.get("id", "")).strip()
        query = str(item.get("query", "")).strip()
        if not case_id or case_id in seen_ids:
            raise RagasGoldenSetError(f"test_cases[{index}] has a missing or duplicate id")
        seen_ids.add(case_id)
        if not query:
            raise RagasGoldenSetError(f"{case_id}.query must not be empty")
        expected_abstain = item.get("expected_abstain")
        if not isinstance(expected_abstain, bool):
            raise RagasGoldenSetError(f"{case_id}.expected_abstain must be a boolean")
        paper_ids = _string_list(
            item.get("paper_ids", []), f"{case_id}.paper_ids", allow_empty=expected_abstain
        )
        chunk_ids = _string_list(
            item.get("reference_chunk_ids", []),
            f"{case_id}.reference_chunk_ids",
            allow_empty=expected_abstain,
        )
        reference = item.get("reference_answer")
        reference = str(reference).strip() if reference is not None else None
        if not expected_abstain and not reference:
            raise RagasGoldenSetError(f"{case_id}.reference_answer must not be empty")
        tags = _string_list(item.get("tags", []), f"{case_id}.tags")
        cases.append(
            RagasCase(
                id=case_id,
                query=query,
                paper_ids=paper_ids,
                reference_answer=reference or None,
                reference_chunk_ids=chunk_ids,
                expected_abstain=expected_abstain,
                tags=tags,
                notes=str(item["notes"]) if item.get("notes") is not None else None,
            )
        )
    return cases, hashlib.sha256(raw_bytes).hexdigest()
