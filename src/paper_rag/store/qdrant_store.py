"""Qdrant 向量存储适配器。"""

from __future__ import annotations

import atexit
import hashlib
from collections.abc import Iterable
from threading import Lock
from typing import Any

from .. import config as cfg
from ..utils.logger import get_logger

log = get_logger(__name__)

_CLIENT: Any | None = None
_ATEXIT_REGISTERED = False
_CLIENT_LOCK = Lock()


def close_client() -> None:
    """关闭已缓存的 Qdrant 客户端。"""
    global _CLIENT

    client = _CLIENT
    _CLIENT = None

    close = getattr(client, "close", None)

    if callable(close):
        try:
            close()
        except Exception as error:
            log.debug(f"qdrant close skipped: {error}")


def get_client() -> Any:
    """按配置创建并缓存 Qdrant server 或 embedded 客户端。"""
    global _CLIENT, _ATEXIT_REGISTERED

    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                from qdrant_client import QdrantClient

                qdrant_config = cfg.load().qdrant
                local_path = getattr(qdrant_config, "local_path", None)
                url = qdrant_config.url or ""

                if local_path:
                    log.info(f"qdrant client in embedded mode at {local_path}")
                    _CLIENT = QdrantClient(path=local_path)
                elif url.startswith(("file://", "local://")):
                    path = url.split("://", 1)[1]
                    log.info(f"qdrant client in embedded mode at {path}")
                    _CLIENT = QdrantClient(path=path)
                else:
                    log.info(f"qdrant client in server mode at {url}")
                    _CLIENT = QdrantClient(url=url)

                if not _ATEXIT_REGISTERED:
                    atexit.register(close_client)
                    _ATEXIT_REGISTERED = True

    return _CLIENT


def _stable_point_id(chunk_id: str) -> int:
    """将字符串 chunk ID 稳定映射为 Qdrant 接受的整数 point ID。"""
    digest = hashlib.sha1(
        chunk_id.encode("utf-8")
    ).hexdigest()

    return int(digest[:16], 16)


def upsert_chunks(
    items: Iterable[dict[str, Any]],
    vectors: list[list[float]],
) -> int:
    """将 chunk payload 与对应向量写入 Qdrant。"""
    from qdrant_client.http import models as qdrant_models

    item_list = list(items)

    if len(item_list) != len(vectors):
        raise ValueError(
            f"items({len(item_list)}) and "
            f"vectors({len(vectors)}) must align"
        )

    client = get_client()
    collection_name = cfg.load().qdrant.collection_chunks
    points = []

    for item, vector in zip(item_list, vectors, strict=True):
        point_id = _stable_point_id(item["chunk_id"])
        payload = {
            key: value
            for key, value in item.items()
            if key != "vector"
        }
        points.append(
            qdrant_models.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
        )

    client.upsert(
        collection_name=collection_name,
        points=points,
        wait=True,
    )
    log.info(
        f"qdrant upsert {len(points)} points into {collection_name}"
    )

    return len(points)


def delete_chunks_for_paper(paper_id: str) -> None:
    """删除一篇论文在 Qdrant 中的全部 chunk。"""
    from qdrant_client.http import models as qdrant_models

    try:
        client = get_client()
        collection_name = cfg.load().qdrant.collection_chunks
        paper_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="paper_id",
                    match=qdrant_models.MatchValue(
                        value=paper_id
                    ),
                )
            ]
        )
        selector = qdrant_models.FilterSelector(
            filter=paper_filter
        )

        client.delete(
            collection_name=collection_name,
            points_selector=selector,
            wait=True,
        )
        log.info(
            f"qdrant deleted paper_id={paper_id} "
            f"from {collection_name}"
        )
    except Exception as error:
        log.warning(
            f"qdrant delete degraded for {paper_id}: "
            f"{type(error).__name__}: {error}"
        )


def search(
    query_vector: list[float],
    top_k: int = 8,
    paper_ids: list[str] | None = None,
    modality: str | None = None,
    raise_on_error: bool = False,
) -> list[dict[str, Any]]:
    """执行带 metadata 过滤条件的向量检索。"""
    from qdrant_client.http import models as qdrant_models

    conditions: list[Any] = []

    if paper_ids:
        conditions.append(
            qdrant_models.FieldCondition(
                key="paper_id",
                match=qdrant_models.MatchAny(any=paper_ids),
            )
        )

    if modality:
        conditions.append(
            qdrant_models.FieldCondition(
                key="modality",
                match=qdrant_models.MatchValue(
                    value=modality
                ),
            )
        )

    query_filter = (
        qdrant_models.Filter(must=conditions)
        if conditions
        else None
    )

    try:
        client = get_client()
        collection_name = cfg.load().qdrant.collection_chunks

        if hasattr(client, "query_points"):
            query_result = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
            hits = (
                query_result.points
                if hasattr(query_result, "points")
                else query_result
            )
        else:
            hits = client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
    except Exception as error:
        if raise_on_error:
            raise
        log.warning(
            "qdrant search degraded, returning empty result: "
            f"{type(error).__name__}: {error}"
        )
        return []

    results: list[dict[str, Any]] = []

    for hit in hits:
        result = dict(hit.payload or {})
        result["score"] = float(hit.score)
        results.append(result)

    return results
