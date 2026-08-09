"""存储初始化入口的行为契约测试。"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType, SimpleNamespace

from sqlalchemy import inspect


class _FakeQdrantClient:
    def __init__(
        self,
        existing_collections: set[str] | None = None,
    ) -> None:
        self.existing_collections = set(existing_collections or set())
        self.created_collections: list[dict] = []

    def get_collections(self):
        return SimpleNamespace(
            collections=[SimpleNamespace(name=name) for name in self.existing_collections]
        )

    def create_collection(self, **kwargs) -> None:
        self.created_collections.append(kwargs)
        self.existing_collections.add(kwargs["collection_name"])


def _init_store_module() -> ModuleType:
    return importlib.import_module("scripts.init_store")


def _config(sqlite_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        paths=SimpleNamespace(
            sqlite_path=str(sqlite_path),
        ),
        embedding=SimpleNamespace(dim=1024),
        qdrant=SimpleNamespace(
            collection_chunks="test_paper_chunks",
            collection_wiki="test_wiki_entries",
            distance="Cosine",
        ),
    )


def test_init_qdrant_creates_missing_collections(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _init_store_module()
    client = _FakeQdrantClient()
    close_calls: list[str] = []

    from paper_rag.store import qdrant_store

    monkeypatch.setattr(
        module.cfg,
        "load",
        lambda: _config(tmp_path / "papers.sqlite"),
    )
    monkeypatch.setattr(
        qdrant_store,
        "get_client",
        lambda: client,
    )
    monkeypatch.setattr(
        qdrant_store,
        "close_client",
        lambda: close_calls.append("closed"),
    )

    module.init_qdrant()

    created_names = {call["collection_name"] for call in client.created_collections}
    assert created_names == {
        "test_paper_chunks",
        "test_wiki_entries",
    }

    for call in client.created_collections:
        assert call["vectors_config"].size == 1024
        assert str(call["vectors_config"].distance).lower().endswith("cosine")

    assert close_calls == ["closed"]


def test_init_qdrant_keeps_existing_collections(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _init_store_module()
    client = _FakeQdrantClient(
        {
            "test_paper_chunks",
            "test_wiki_entries",
        }
    )
    close_calls: list[str] = []

    from paper_rag.store import qdrant_store

    monkeypatch.setattr(
        module.cfg,
        "load",
        lambda: _config(tmp_path / "papers.sqlite"),
    )
    monkeypatch.setattr(
        qdrant_store,
        "get_client",
        lambda: client,
    )
    monkeypatch.setattr(
        qdrant_store,
        "close_client",
        lambda: close_calls.append("closed"),
    )

    module.init_qdrant()

    assert client.created_collections == []
    assert close_calls == ["closed"]


def test_init_sqlite_creates_real_database(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _init_store_module()
    database_path = tmp_path / "index" / "papers.sqlite"

    from paper_rag.store import sqlite_store

    monkeypatch.setattr(
        module.cfg,
        "load",
        lambda: _config(database_path),
    )
    monkeypatch.setattr(sqlite_store, "_ENGINE", None)

    module.init_sqlite()

    engine = sqlite_store.get_engine()

    assert database_path.is_file()
    assert {
        "paper",
        "section",
        "chunk",
        "ingest_runs",
    } <= set(inspect(engine).get_table_names())

    engine.dispose()
    sqlite_store._ENGINE = None


def test_init_sqlite_creates_wiki_tables(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """wiki 8 张表必须由 init_store 显式建出, 不能靠 QA 首次访问懒建。"""
    module = _init_store_module()
    from sqlalchemy import inspect

    from paper_rag.store import sqlite_store

    monkeypatch.setattr(
        module.cfg,
        "load",
        lambda: _config(tmp_path / "papers.sqlite"),
    )
    monkeypatch.setattr(
        sqlite_store.cfg,
        "load",
        lambda: _config(tmp_path / "papers.sqlite"),
    )
    sqlite_store._ENGINE = None

    module.init_sqlite()

    engine = sqlite_store.get_engine()
    assert {
        "wiki_entries",
        "wiki_labels",
        "wiki_entry_papers",
        "wiki_entry_evidence",
        "wiki_versions",
        "wiki_jobs",
        "wiki_review_queue",
        "wiki_usage",
    } <= set(inspect(engine).get_table_names())

    engine.dispose()
    sqlite_store._ENGINE = None


def test_main_initializes_components_in_order(
    monkeypatch,
) -> None:
    module = _init_store_module()
    calls: list[str] = []

    monkeypatch.setattr(
        module.cfg,
        "load",
        lambda: calls.append("config"),
    )
    monkeypatch.setattr(
        module,
        "ensure_dirs",
        lambda: calls.append("directories"),
    )
    monkeypatch.setattr(
        module,
        "init_sqlite",
        lambda: calls.append("sqlite"),
    )
    monkeypatch.setattr(
        module,
        "init_qdrant",
        lambda: calls.append("qdrant"),
    )

    module.main()

    assert calls == [
        "config",
        "directories",
        "sqlite",
        "qdrant",
    ]
