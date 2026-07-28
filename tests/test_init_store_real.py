"""存储初始化入口的真实集成测试, 不使用 mock。"""

from __future__ import annotations

import importlib
from pathlib import Path

from sqlalchemy import inspect


def test_real_init_store_is_idempotent() -> None:
    init_store = importlib.import_module("scripts.init_store")

    from paper_rag import config as cfg
    from paper_rag.store import qdrant_store, sqlite_store

    config = cfg.load()

    print("[1/4] 第一次执行真实初始化")
    init_store.main()

    print("[2/4] 第二次执行真实初始化")
    init_store.main()

    print("[3/4] 检查真实 SQLite")
    database_path = Path(config.paths.sqlite_path)
    assert database_path.is_file()

    engine = sqlite_store.get_engine()
    table_names = set(inspect(engine).get_table_names())

    assert {
        "paper",
        "section",
        "chunk",
        "ingest_runs",
    } <= table_names

    print("[4/4] 检查真实 Qdrant collections")
    client = qdrant_store.get_client()

    try:
        collection_names = {
            collection.name
            for collection in client.get_collections().collections
        }
    finally:
        qdrant_store.close_client()

    assert config.qdrant.collection_chunks in collection_names
    assert config.qdrant.collection_wiki in collection_names

    print("真实存储初始化集成测试通过")
