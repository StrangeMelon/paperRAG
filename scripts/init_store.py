"""初始化运行目录、SQLite 表和 Qdrant collections。"""

from __future__ import annotations

from paper_rag import config as cfg
from paper_rag.utils.logger import get_logger
from paper_rag.utils.paths import ensure_dirs

log = get_logger(__name__)


def init_qdrant() -> None:
    """幂等创建项目需要的 Qdrant collections。"""
    from qdrant_client.http import models as qdrant_models

    from paper_rag.store import qdrant_store

    config = cfg.load()
    client = qdrant_store.get_client()
    distance_name = config.qdrant.distance.lower()

    if distance_name == "cosine":
        distance = qdrant_models.Distance.COSINE
    elif distance_name == "dot":
        distance = qdrant_models.Distance.DOT
    else:
        raise ValueError(f"unsupported Qdrant distance: {config.qdrant.distance}")

    try:
        collection_names = (
            config.qdrant.collection_chunks,
            config.qdrant.collection_wiki,
        )
        existing_names = {collection.name for collection in (client.get_collections().collections)}

        for collection_name in collection_names:
            if collection_name in existing_names:
                log.info(f"Qdrant collection already exists: {collection_name}")
                continue

            client.create_collection(
                collection_name=collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=config.embedding.dim,
                    distance=distance,
                ),
            )
            existing_names.add(collection_name)

            log.info(
                f"created Qdrant collection: "
                f"{collection_name} "
                f"(dim={config.embedding.dim}, "
                f"distance={config.qdrant.distance})"
            )

        client.create_payload_index(
            collection_name=config.qdrant.collection_chunks,
            field_name="paper_id",
            field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            wait=True,
        )
        log.info(
            f"ensured Qdrant payload index: {config.qdrant.collection_chunks}.paper_id (keyword)"
        )
    finally:
        qdrant_store.close_client()


def init_sqlite() -> None:
    """创建 SQLite 文件和 SQLModel 表结构(含 wiki 的 8 张表)。"""
    from sqlmodel import SQLModel

    from paper_rag.store import sqlite_store

    # wiki 的表定义分散在各自模块里, 必须先 import 让它们注册进
    # SQLModel.metadata, 否则 create_all 建不出 wiki 表, 只能等 worker/QA
    # 首次访问时懒建(初始化脚本应当一次把库建全)。
    from paper_rag.wiki import queue as _wiki_queue
    from paper_rag.wiki import review_queue as _wiki_review
    from paper_rag.wiki import store as _wiki_store
    from paper_rag.wiki import usage as _wiki_usage

    config = cfg.load()
    engine = sqlite_store.get_engine()
    # engine 可能在 wiki 模块 import 之前就已缓存(create_all 已跑过), 幂等补建
    SQLModel.metadata.create_all(engine)

    log.info(f"SQLite ready at {config.paths.sqlite_path} (dialect={engine.dialect.name})")


def main() -> None:
    """按照依赖顺序初始化全部存储资源。"""
    log.info("[1/3] loading configuration and directories")
    cfg.load()
    ensure_dirs()

    log.info("[2/3] initializing SQLite")
    init_sqlite()

    log.info("[3/3] initializing Qdrant")
    init_qdrant()

    log.info("storage initialization completed")


if __name__ == "__main__":
    main()
