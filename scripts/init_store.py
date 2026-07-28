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
        raise ValueError(
            "unsupported Qdrant distance: "
            f"{config.qdrant.distance}"
        )

    try:
        collection_names = (
            config.qdrant.collection_chunks,
            config.qdrant.collection_wiki,
        )
        existing_names = {
            collection.name
            for collection in (
                client.get_collections().collections
            )
        }

        for collection_name in collection_names:
            if collection_name in existing_names:
                log.info(
                    f"Qdrant collection already exists: "
                    f"{collection_name}"
                )
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
    finally:
        qdrant_store.close_client()


def init_sqlite() -> None:
    """创建 SQLite 文件和 SQLModel 表结构。"""
    from paper_rag.store import sqlite_store

    config = cfg.load()
    engine = sqlite_store.get_engine()

    log.info(
        f"SQLite ready at {config.paths.sqlite_path} "
        f"(dialect={engine.dialect.name})"
    )


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
