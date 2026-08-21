"""SQLite persistence for interactive retrieval diagnostic runs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import Engine, func
from sqlmodel import Field, Session, SQLModel, select

_STAGE_NAMES = ("Dense", "Sparse", "RRF", "Rerank", "Diversify")
_TEXT_SNAPSHOT_LIMIT = 500


class RetrievalDiagnosticRun(SQLModel, table=True):
    __tablename__ = "retrieval_diagnostic_runs"

    run_id: str = Field(primary_key=True)
    query: str = Field(index=True)
    paper_ids_json: str = "[]"
    top_k: int = 8
    status: str = Field(default="ok", index=True)
    total_ms: float = 0.0
    timings_json: str = "{}"
    rewrite_json: str = "{}"
    final_chunks_json: str = "[]"
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC), index=True)


class RetrievalDiagnosticStageResult(SQLModel, table=True):
    __tablename__ = "retrieval_diagnostic_stage_results"

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(
        foreign_key="retrieval_diagnostic_runs.run_id",
        index=True,
    )
    stage: str = Field(index=True)
    rank: int
    chunk_id: str | None = Field(default=None, index=True)
    paper_id: str | None = Field(default=None, index=True)
    score: float | None = None
    timing_ms: float = 0.0
    payload_json: str = "{}"


class RetrievalHistoryStore:
    def __init__(self, *, engine: Engine | None = None) -> None:
        if engine is None:
            from ...store.sqlite_store import get_engine

            engine = get_engine()
        self.engine = engine
        SQLModel.metadata.create_all(
            self.engine,
            tables=[
                RetrievalDiagnosticRun.__table__,
                RetrievalDiagnosticStageResult.__table__,
            ],
        )

    def save(
        self,
        result: dict[str, Any],
        *,
        paper_ids: list[str] | None,
        top_k: int,
    ) -> str:
        run_id = str(result.get("run_id") or f"diagnostic-{uuid4().hex[:12]}")
        timings = result.get("timings_ms") or {}
        stages = result.get("stages") or {}
        rewrite = stages.get("Query Rewrite") or {}
        created_at = _parse_datetime(result.get("created_at"))
        run = RetrievalDiagnosticRun(
            run_id=run_id,
            query=str(result.get("query") or ""),
            paper_ids_json=_dump(paper_ids or []),
            top_k=int(top_k),
            status=str(result.get("status") or "ok"),
            total_ms=float(timings.get("retrieval_total_ms") or 0.0),
            timings_json=_dump(timings),
            rewrite_json=_dump(rewrite),
            final_chunks_json=_dump(result.get("chunks") or []),
            error=str(result.get("error")) if result.get("error") else None,
            created_at=created_at,
        )
        with Session(self.engine) as session:
            session.add(run)
            for stage_name in _STAGE_NAMES:
                stage = stages.get(stage_name) or {}
                timing_ms = float(stage.get("timing_ms") or 0.0)
                for rank, item in enumerate(stage.get("items") or [], 1):
                    payload = _snapshot_item(item, rank)
                    session.add(
                        RetrievalDiagnosticStageResult(
                            run_id=run_id,
                            stage=stage_name,
                            rank=rank,
                            chunk_id=_optional_string(item.get("chunk_id")),
                            paper_id=_optional_string(item.get("paper_id")),
                            score=_stage_score(stage_name, item),
                            timing_ms=timing_ms,
                            payload_json=_dump(payload),
                        )
                    )
            session.commit()
        return run_id

    def get(self, run_id: str) -> dict[str, Any] | None:
        with Session(self.engine) as session:
            run = session.get(RetrievalDiagnosticRun, run_id)
            if run is None:
                return None
            rows = list(
                session.exec(
                    select(RetrievalDiagnosticStageResult)
                    .where(RetrievalDiagnosticStageResult.run_id == run_id)
                    .order_by(
                        RetrievalDiagnosticStageResult.stage,
                        RetrievalDiagnosticStageResult.rank,
                    )
                )
            )
        stages: dict[str, dict[str, Any]] = {
            "Query Rewrite": _load_object(run.rewrite_json),
        }
        for stage_name in _STAGE_NAMES:
            stage_rows = [row for row in rows if row.stage == stage_name]
            stages[stage_name] = {
                "timing_ms": stage_rows[0].timing_ms if stage_rows else _timing(run, stage_name),
                "items": [_load_object(row.payload_json) for row in stage_rows],
            }
        return {
            "run_id": run.run_id,
            "query": run.query,
            "paper_ids": _load_list(run.paper_ids_json),
            "top_k": run.top_k,
            "status": run.status,
            "created_at": run.created_at.isoformat(),
            "timings_ms": _load_object(run.timings_json),
            "stages": stages,
            "chunks": _load_list(run.final_chunks_json),
            "error": run.error,
        }

    def list(
        self,
        *,
        query: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        statement = select(RetrievalDiagnosticRun)
        if query and query.strip():
            statement = statement.where(
                RetrievalDiagnosticRun.query.contains(query.strip(), autoescape=True)
            )
        statement = statement.order_by(RetrievalDiagnosticRun.created_at.desc()).offset(offset)
        if limit > 0:
            statement = statement.limit(limit)
        with Session(self.engine) as session:
            rows = list(session.exec(statement))
        return [_summary(row) for row in rows]

    def count(self, *, query: str | None = None) -> int:
        statement = select(func.count()).select_from(RetrievalDiagnosticRun)
        if query and query.strip():
            statement = statement.where(
                RetrievalDiagnosticRun.query.contains(query.strip(), autoescape=True)
            )
        with Session(self.engine) as session:
            return int(session.exec(statement).one())

    def delete(self, run_id: str) -> bool:
        with Session(self.engine) as session:
            run = session.get(RetrievalDiagnosticRun, run_id)
            if run is None:
                return False
            stage_rows = session.exec(
                select(RetrievalDiagnosticStageResult).where(
                    RetrievalDiagnosticStageResult.run_id == run_id
                )
            ).all()
            for row in stage_rows:
                session.delete(row)
            session.delete(run)
            session.commit()
        return True


def _summary(run: RetrievalDiagnosticRun) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "query": run.query,
        "paper_ids": _load_list(run.paper_ids_json),
        "top_k": run.top_k,
        "status": run.status,
        "total_ms": run.total_ms,
        "created_at": run.created_at.isoformat(),
    }


def _snapshot_item(item: dict[str, Any], rank: int) -> dict[str, Any]:
    payload = dict(item)
    text = str(payload.get("text") or payload.get("context_text") or "")
    payload["text"] = text[:_TEXT_SNAPSHOT_LIMIT]
    payload.pop("context_text", None)
    payload["rank"] = rank
    return payload


def _stage_score(stage_name: str, item: dict[str, Any]) -> float | None:
    key = {
        "Dense": "score",
        "Sparse": "score_bm25",
        "RRF": "score_rrf",
        "Rerank": "score_rerank",
        "Diversify": "score_rerank",
    }[stage_name]
    value = item.get(key)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _timing(run: RetrievalDiagnosticRun, stage_name: str) -> float:
    key = {
        "Dense": "dense_ms",
        "Sparse": "sparse_ms",
        "RRF": "rrf_ms",
        "Rerank": "rerank_ms",
        "Diversify": "diversify_ms",
    }[stage_name]
    return float(_load_object(run.timings_json).get(key) or 0.0)


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if value:
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(UTC)


def _optional_string(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _load_object(value: str) -> dict[str, Any]:
    loaded = json.loads(value or "{}")
    return loaded if isinstance(loaded, dict) else {}


def _load_list(value: str) -> list[Any]:
    loaded = json.loads(value or "[]")
    return loaded if isinstance(loaded, list) else []
