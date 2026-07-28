"""SQLite 论文元数据存储的行为契约测试。"""

from __future__ import annotations

import importlib
import json
import sqlite3
from pathlib import Path
from types import ModuleType, SimpleNamespace

from sqlalchemy import inspect


def _store_module() -> ModuleType:
    return importlib.import_module("paper_rag.store.sqlite_store")


def _isolated_store(
    monkeypatch,
    tmp_path: Path,
) -> tuple[ModuleType, Path]:
    store = _store_module()
    database_path = tmp_path / "papers.sqlite"
    config = SimpleNamespace(
        paths=SimpleNamespace(sqlite_path=str(database_path))
    )

    monkeypatch.setattr(store.cfg, "load", lambda: config)
    monkeypatch.setattr(store, "_ENGINE", None)

    return store, database_path


def test_get_engine_initializes_sqlite_database(
    monkeypatch,
    tmp_path: Path,
) -> None:
    store, database_path = _isolated_store(monkeypatch, tmp_path)

    first_engine = store.get_engine()
    second_engine = store.get_engine()

    assert first_engine is second_engine
    assert database_path.is_file()
    assert "paper" in inspect(first_engine).get_table_names()

    with first_engine.connect() as connection:
        journal_mode = connection.exec_driver_sql(
            "PRAGMA journal_mode"
        ).scalar_one()
        busy_timeout = connection.exec_driver_sql(
            "PRAGMA busy_timeout"
        ).scalar_one()
        foreign_keys = connection.exec_driver_sql(
            "PRAGMA foreign_keys"
        ).scalar_one()

    assert str(journal_mode).lower() == "wal"
    assert busy_timeout == 5000
    assert foreign_keys == 1

# 测试论文插入功能
def test_upsert_paper_inserts_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    store, _ = _isolated_store(monkeypatch, tmp_path)
    metadata = {
        "paper_id": "arxiv:2310.11511",
        "title": "Self-RAG: Learning to Retrieve",
        "authors": ["Akari Asai", "Zhiqing Sun"],
        "year": 2023,
        "venue": "ICLR",
        "arxiv_id": "2310.11511",
        "abstract": "A retrieval-augmented generation paper.",
        "extra": {"arxiv_version": "v2"},
    }

    store.upsert_paper(metadata)
    paper = store.get_paper("arxiv:2310.11511")

    assert paper is not None
    assert paper.title == "Self-RAG: Learning to Retrieve"
    assert json.loads(paper.authors_json) == [
        "Akari Asai",
        "Zhiqing Sun",
    ]
    assert paper.year == 2023
    assert paper.arxiv_version == "v2"
    assert paper.title_norm == "selfraglearningtoretrieve"
    assert paper.status == "created"
    assert paper.user_id == "system"

# 测试论文的更新功能
def test_upsert_paper_updates_existing_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    store, _ = _isolated_store(monkeypatch, tmp_path)
    paper_id = "doi:10.1000/example"

    store.upsert_paper(
        {
            "paper_id": paper_id,
            "title": "Original Title",
            "authors": ["Alice"],
            "year": 2024,
        }
    )
    original = store.get_paper(paper_id)
    assert original is not None
    created_at = original.created_at

    store.upsert_paper(
        {
            "paper_id": paper_id,
            "title": "Updated Title",
            "authors": ["Alice", "Bob"],
            "year": 2026,
        },
        status="fetched",
    )
    updated = store.get_paper(paper_id)

    assert updated is not None
    assert updated.title == "Updated Title"
    assert json.loads(updated.authors_json) == ["Alice", "Bob"]
    assert updated.year == 2026
    assert updated.status == "fetched"
    assert updated.created_at == created_at
    assert updated.updated_at >= created_at

# 测试 set_status 方法的功能
def test_set_status_updates_existing_paper_and_ignores_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    store, _ = _isolated_store(monkeypatch, tmp_path)
    paper_id = "paper:status-test"
    store.upsert_paper({"paper_id": paper_id, "title": "Status Test"})

    store.set_status(
        paper_id,
        "parsed",
        parsed_with="mineru",
    )
    parsed = store.get_paper(paper_id)

    assert parsed is not None
    assert parsed.status == "parsed"
    assert parsed.parsed_with == "mineru"

    store.set_status(
        paper_id,
        "failed",
        error="parser failed",
    )
    failed = store.get_paper(paper_id)

    assert failed is not None
    assert failed.status == "failed"
    assert failed.error == "parser failed"

    store.set_status("paper:missing", "failed", error="not found")
    assert store.get_paper("paper:missing") is None


def test_record_and_finish_ingest_step(
      monkeypatch,
      tmp_path: Path,
  ) -> None:
      from sqlmodel import Session

      store, _ = _isolated_store(monkeypatch, tmp_path)

      run_id = store.record_ingest_step(
          "paper:ingest-run",
          "parsed",
      )

      assert isinstance(run_id, int)

      store.finish_ingest_step(
          run_id,
          status="error",
          error="parser failed",
      )

      with Session(store.get_engine()) as session:
          ingest_run = session.get(store.IngestRun, run_id)

      assert ingest_run is not None
      assert ingest_run.paper_id == "paper:ingest-run"
      assert ingest_run.step == "parsed"
      assert ingest_run.status == "error"
      assert ingest_run.error == "parser failed"
      assert ingest_run.started_at is not None
      assert ingest_run.finished_at is not None

      store.finish_ingest_step(999_999)


def test_find_existing_paper_uses_identifier_priority(
    monkeypatch,
    tmp_path: Path,
) -> None:
    store, _ = _isolated_store(monkeypatch, tmp_path)

    store.upsert_paper(
        {
            "paper_id": "paper:doi",
            "title": "Paper Found By DOI",
            "doi": "10.1000/doi-paper",
        }
    )
    store.upsert_paper(
        {
            "paper_id": "paper:arxiv",
            "title": "Paper Found By arXiv",
            "arxiv_id": "2310.12345",
        }
    )
    store.upsert_paper(
        {
            "paper_id": "paper:title",
            "title": "Retrieval Augmented Generation",
        }
    )

    found_by_doi = store.find_existing_paper(
        doi="10.1000/doi-paper",
        arxiv_id="2310.12345",
    )
    found_by_arxiv = store.find_existing_paper(
        arxiv_id="2310.12345",
    )
    found_by_title = store.find_existing_paper(
        title_norm="retrievalaugmentedgeneration",
    )
    missing = store.find_existing_paper(
        doi="10.1000/missing",
        arxiv_id="9999.99999",
        title_norm="missingtitle",
    )

    assert found_by_doi is not None
    assert found_by_doi.paper_id == "paper:doi"

    assert found_by_arxiv is not None
    assert found_by_arxiv.paper_id == "paper:arxiv"

    assert found_by_title is not None
    assert found_by_title.paper_id == "paper:title"
    assert missing is None


def test_sections_and_chunks_round_trip(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from sqlmodel import Session

    store, _ = _isolated_store(monkeypatch, tmp_path)
    paper_id = "paper:chunk-round-trip"

    sections = [
        {
            "section_id": "section:introduction",
            "paper_id": paper_id,
            "idx": 0,
            "name": "Introduction",
            "page_start": 1,
            "page_end": 2,
        }
    ]
    chunks = [
        {
            "chunk_id": "chunk:introduction:0",
            "paper_id": paper_id,
            "section_id": "section:introduction",
            "section": "Introduction",
            "section_idx": 0,
            "modality": "figure",
            "page": 2,
            "text": "Figure 1 shows the architecture.",
            "context_text": (
                "[Title: Example] [Section: Introduction]\n"
                "Figure 1 shows the architecture."
            ),
            "title": "Example",
            "source_path": "/papers/example.pdf",
            "asset_path": "/parsed/example/figure-1.png",
            "asset_rel_path": "figure-1.png",
            "char_start": 100,
            "char_end": 132,
            "raw_snippet": "Figure 1 shows the architecture.",
            "neighbors": ["chunk:introduction:1"],
            "metadata": {
                "figure_label": "Figure 1",
                "visual_summary": "A RAG architecture diagram.",
            },
            "embedding": [0.1, 0.2],
        }
    ]

    store.upsert_sections_and_chunks(
        paper_id,
        sections,
        chunks,
    )

    stored_chunks = store.list_chunks_for_papers([paper_id])
    stored_chunk = store.get_chunk("chunk:introduction:0")

    assert len(stored_chunks) == 1
    assert stored_chunk is not None
    assert stored_chunk.modality == "figure"
    assert stored_chunk.section == "Introduction"
    assert stored_chunk.asset_rel_path == "figure-1.png"
    assert json.loads(stored_chunk.neighbors_json) == [
        "chunk:introduction:1"
    ]
    assert json.loads(stored_chunk.metadata_json) == {
        "figure_label": "Figure 1",
        "visual_summary": "A RAG architecture diagram.",
    }
    assert "embedding" not in stored_chunk.model_dump()

    with Session(store.get_engine()) as session:
        stored_section = session.get(
            store.Section,
            "section:introduction",
        )

    assert stored_section is not None
    assert stored_section.name == "Introduction"
    assert stored_section.page_start == 1
    assert stored_section.page_end == 2


def test_upsert_replaces_stale_sections_and_chunks(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from sqlmodel import Session

    store, _ = _isolated_store(monkeypatch, tmp_path)
    paper_id = "paper:replacement"

    store.upsert_sections_and_chunks(
        paper_id,
        sections=[
            {
                "section_id": "section:old",
                "paper_id": paper_id,
                "idx": 0,
                "name": "Old Section",
            },
            {
                "section_id": "section:keep",
                "paper_id": paper_id,
                "idx": 1,
                "name": "Original Name",
            },
        ],
        chunks=[
            {
                "chunk_id": "chunk:old",
                "paper_id": paper_id,
                "text": "Stale text",
            },
            {
                "chunk_id": "chunk:keep",
                "paper_id": paper_id,
                "text": "Original text",
            },
        ],
    )

    store.upsert_sections_and_chunks(
        paper_id,
        sections=[
            {
                "section_id": "section:keep",
                "paper_id": paper_id,
                "idx": 0,
                "name": "Updated Name",
            }
        ],
        chunks=[
            {
                "chunk_id": "chunk:keep",
                "paper_id": paper_id,
                "text": "Updated text",
                "metadata": {"revision": 2},
            }
        ],
    )

    assert store.get_chunk("chunk:old") is None

    retained_chunk = store.get_chunk("chunk:keep")
    assert retained_chunk is not None
    assert retained_chunk.text == "Updated text"
    assert json.loads(retained_chunk.metadata_json) == {
        "revision": 2
    }

    with Session(store.get_engine()) as session:
        removed_section = session.get(store.Section, "section:old")
        retained_section = session.get(store.Section, "section:keep")

    assert removed_section is None
    assert retained_section is not None
    assert retained_section.name == "Updated Name"


def test_get_engine_migrates_legacy_chunk_columns(
    monkeypatch,
    tmp_path: Path,
) -> None:
    store, database_path = _isolated_store(monkeypatch, tmp_path)

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE chunk (
                chunk_id TEXT PRIMARY KEY,
                paper_id TEXT NOT NULL,
                section_id TEXT,
                modality TEXT NOT NULL,
                page INTEGER,
                text TEXT NOT NULL,
                context_text TEXT NOT NULL,
                neighbors_json TEXT NOT NULL
            )
            """
        )

    engine = store.get_engine()

    with engine.connect() as connection:
        rows = connection.exec_driver_sql(
            "PRAGMA table_info(chunk)"
        ).fetchall()

    column_names = {row[1] for row in rows}
    expected_migrated_columns = {
        "section",
        "section_idx",
        "title",
        "source_path",
        "asset_path",
        "asset_rel_path",
        "char_start",
        "char_end",
        "raw_snippet",
        "metadata_json",
    }

    assert expected_migrated_columns <= column_names
