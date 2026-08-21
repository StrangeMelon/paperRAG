"""Read and mutation operations used by the data browser."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlalchemy import func
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from ...store.sqlite_store import Chunk, IngestRun, Paper, Section, get_engine


class DashboardDataService:
    def __init__(
        self,
        *,
        engine: Engine | None = None,
        wiki_loader: Callable[[], list[Any]] | None = None,
        qdrant_delete: Callable[[str], Any] | None = None,
        fts_sync: Callable[[str], Any] | None = None,
        wiki_remove: Callable[[str], int] | None = None,
    ) -> None:
        self.engine = engine or get_engine()
        self._wiki_loader = wiki_loader or _load_wiki_entries
        self._qdrant_delete = qdrant_delete or _delete_qdrant
        self._fts_sync = fts_sync or _sync_fts
        self._wiki_remove = wiki_remove or _remove_wiki_links

    def summary(self) -> dict[str, int]:
        with Session(self.engine) as session:
            papers = int(session.exec(select(func.count()).select_from(Paper)).one())
            chunks = int(session.exec(select(func.count()).select_from(Chunk)).one())
            visual = int(
                session.exec(
                    select(func.count())
                    .select_from(Chunk)
                    .where(Chunk.modality.in_(["figure", "table", "formula"]))
                ).one()
            )
        return {
            "papers": papers,
            "chunks": chunks,
            "visual_chunks": visual,
            "wiki_entries": len(self.list_wiki_entries()),
        }

    def list_papers(self, *, keyword: str | None = None) -> list[dict[str, Any]]:
        with Session(self.engine) as session:
            statement = select(Paper).order_by(Paper.updated_at.desc())
            if keyword:
                statement = statement.where(Paper.title.contains(keyword))
            papers = list(session.exec(statement))
            counts = dict(
                session.exec(select(Chunk.paper_id, func.count()).group_by(Chunk.paper_id)).all()
            )
        return [
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "authors": _json_list(paper.authors_json),
                "year": paper.year,
                "venue": paper.venue,
                "status": paper.status,
                "parsed_with": paper.parsed_with,
                "error": paper.error,
                "updated_at": paper.updated_at.isoformat() if paper.updated_at else None,
                "chunk_count": int(counts.get(paper.paper_id, 0)),
            }
            for paper in papers
        ]

    def get_paper_detail(self, paper_id: str) -> dict[str, Any] | None:
        with Session(self.engine) as session:
            paper = session.get(Paper, paper_id)
            if paper is None:
                return None
            chunks = list(
                session.exec(
                    select(Chunk)
                    .where(Chunk.paper_id == paper_id)
                    .order_by(Chunk.section_idx, Chunk.page, Chunk.chunk_id)
                )
            )
            runs = list(
                session.exec(
                    select(IngestRun)
                    .where(IngestRun.paper_id == paper_id)
                    .order_by(IngestRun.started_at.desc())
                )
            )
        return {
            "paper": next(item for item in self.list_papers() if item["paper_id"] == paper_id),
            "chunks": [_chunk_dict(chunk) for chunk in chunks],
            "ingest_runs": [
                {
                    "step": run.step,
                    "status": run.status,
                    "started_at": run.started_at.isoformat(),
                    "finished_at": run.finished_at.isoformat() if run.finished_at else None,
                    "error": run.error,
                }
                for run in runs
            ],
        }

    def list_wiki_entries(self) -> list[dict[str, Any]]:
        try:
            entries = self._wiki_loader()
        except Exception:
            return []
        return [
            entry.model_dump(mode="json") if hasattr(entry, "model_dump") else dict(entry)
            for entry in entries
        ]

    def preview_delete(self, paper_id: str) -> dict[str, Any] | None:
        with Session(self.engine) as session:
            paper = session.get(Paper, paper_id)
            if paper is None:
                return None
            sections = list(session.exec(select(Section).where(Section.paper_id == paper_id)))
            chunks = list(session.exec(select(Chunk).where(Chunk.paper_id == paper_id)))
        assets = sorted(
            {
                str(path)
                for chunk in chunks
                if chunk.asset_path and (path := Path(chunk.asset_path)).exists()
            }
        )
        wiki_links = sum(
            paper_id in (entry.get("key_papers") or []) for entry in self.list_wiki_entries()
        )
        return {
            "paper_id": paper_id,
            "title": paper.title,
            "sections": len(sections),
            "chunks": len(chunks),
            "assets": assets,
            "wiki_links": wiki_links,
        }

    def delete_paper(self, paper_id: str, *, delete_assets: bool = False) -> dict[str, Any]:
        preview = self.preview_delete(paper_id)
        if preview is None:
            return {"paper_id": paper_id, "deleted": False, "errors": ["paper not found"]}
        errors: list[str] = []
        for name, operation in (
            ("qdrant", self._qdrant_delete),
            ("wiki", self._wiki_remove),
        ):
            try:
                operation(paper_id)
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        if errors:
            return {"paper_id": paper_id, "deleted": False, "errors": errors}

        with Session(self.engine) as session:
            for model in (IngestRun, Chunk, Section):
                rows = list(session.exec(select(model).where(model.paper_id == paper_id)))
                for row in rows:
                    session.delete(row)
            paper = session.get(Paper, paper_id)
            if paper is not None:
                session.delete(paper)
            session.commit()
        try:
            self._fts_sync(paper_id)
        except Exception as exc:
            errors.append(f"fts5: {exc}")
        deleted_assets = 0
        if delete_assets:
            for raw_path in preview["assets"]:
                try:
                    Path(raw_path).unlink(missing_ok=True)
                    deleted_assets += 1
                except OSError as exc:
                    errors.append(f"asset {raw_path}: {exc}")
        return {
            "paper_id": paper_id,
            "deleted": True,
            "deleted_assets": deleted_assets,
            "errors": errors,
        }


def _chunk_dict(chunk: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "paper_id": chunk.paper_id,
        "section": chunk.section,
        "page": chunk.page,
        "modality": chunk.modality,
        "text": chunk.text,
        "context_text": chunk.context_text,
        "asset_path": chunk.asset_path,
        "asset_rel_path": chunk.asset_rel_path,
        "metadata": _json_dict(chunk.metadata_json),
    }


def _json_list(value: str) -> list[Any]:
    try:
        loaded = json.loads(value or "[]")
    except json.JSONDecodeError:
        return []
    return loaded if isinstance(loaded, list) else []


def _json_dict(value: str) -> dict[str, Any]:
    try:
        loaded = json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _load_wiki_entries() -> list[Any]:
    from ...wiki.store import list_entries

    return list_entries()


def _delete_qdrant(paper_id: str) -> None:
    from ...store.qdrant_store import delete_chunks_for_paper

    delete_chunks_for_paper(paper_id)


def _sync_fts(paper_id: str) -> None:
    from ...retrieve.fts5 import sync_paper

    sync_paper(paper_id)


def _remove_wiki_links(paper_id: str) -> int:
    from sqlmodel import Session, select

    from ...wiki.store import WikiEntryEvidenceRow, WikiEntryPaperRow, _engine

    removed = 0
    with Session(_engine()) as session:
        paper_rows = list(
            session.exec(select(WikiEntryPaperRow).where(WikiEntryPaperRow.paper_id == paper_id))
        )
        evidence_rows = list(
            session.exec(
                select(WikiEntryEvidenceRow).where(WikiEntryEvidenceRow.paper_id == paper_id)
            )
        )
        for row in [*paper_rows, *evidence_rows]:
            session.delete(row)
            removed += 1
        session.commit()
    return removed
