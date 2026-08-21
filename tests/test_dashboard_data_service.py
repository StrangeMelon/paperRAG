"""Dashboard paper browser and deletion boundary tests."""

from __future__ import annotations

from sqlmodel import Session, SQLModel, create_engine

from paper_rag.store.sqlite_store import Chunk, Paper, Section


def _engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'dashboard.sqlite'}")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Paper(paper_id="p1", title="Paper One", status="done"))
        session.add(Section(section_id="p1:s1", paper_id="p1", idx=0, name="Method"))
        session.add(
            Chunk(
                chunk_id="c1",
                paper_id="p1",
                section_id="p1:s1",
                section="Method",
                modality="figure",
                text="figure context",
                asset_path=str(tmp_path / "figure.png"),
            )
        )
        session.commit()
    (tmp_path / "figure.png").write_bytes(b"image")
    return engine


def test_data_service_lists_summary_papers_chunks_and_wiki(tmp_path) -> None:
    from paper_rag.dashboard.services.data_service import DashboardDataService

    service = DashboardDataService(
        engine=_engine(tmp_path),
        wiki_loader=lambda: [
            {
                "entry_id": "concept:rag",
                "name": "RAG",
                "key_papers": ["p1"],
                "evidence_chunks": ["c1"],
            }
        ],
    )

    assert service.summary()["papers"] == 1
    assert service.summary()["chunks"] == 1
    assert service.summary()["visual_chunks"] == 1
    assert service.list_papers()[0]["title"] == "Paper One"
    assert service.get_paper_detail("p1")["chunks"][0]["chunk_id"] == "c1"
    assert service.list_wiki_entries()[0]["name"] == "RAG"


def test_delete_preview_is_explicit_and_delete_calls_every_backend(tmp_path) -> None:
    from paper_rag.dashboard.services.data_service import DashboardDataService

    calls: list[tuple[str, str]] = []
    service = DashboardDataService(
        engine=_engine(tmp_path),
        qdrant_delete=lambda paper_id: calls.append(("qdrant", paper_id)),
        fts_sync=lambda paper_id: calls.append(("fts", paper_id)),
        wiki_remove=lambda paper_id: calls.append(("wiki", paper_id)) or 1,
    )

    preview = service.preview_delete("p1")
    assert preview == {
        "paper_id": "p1",
        "title": "Paper One",
        "sections": 1,
        "chunks": 1,
        "assets": [str(tmp_path / "figure.png")],
        "wiki_links": 0,
    }

    result = service.delete_paper("p1", delete_assets=True)
    assert result["deleted"] is True
    assert calls == [("qdrant", "p1"), ("wiki", "p1"), ("fts", "p1")]
    assert service.list_papers() == []
    assert not (tmp_path / "figure.png").exists()
