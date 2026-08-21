"""Qdrant 向量存储适配器。"""

from __future__ import annotations

import atexit
from collections.abc import Iterable
from threading import Lock
from typing import Any

from .. import config as cfg
from ..utils.logger import get_logger
from .incremental import stable_point_id

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


def _stable_point_id(paper_id: str, chunk_id: str) -> str:
    """兼容旧测试的内部别名; 新 Point ID 使用 paper-scoped UUIDv5。"""
    return stable_point_id(paper_id, chunk_id)


def upsert_chunks(
    items: Iterable[dict[str, Any]],
    vectors: list[list[float]],
) -> int:
    """将 chunk payload 与对应向量写入 Qdrant。"""
    from qdrant_client.http import models as qdrant_models

    item_list = list(items)

    if len(item_list) != len(vectors):
        raise ValueError(f"items({len(item_list)}) and vectors({len(vectors)}) must align")
    if not item_list:
        return 0

    client = get_client()
    collection_name = cfg.load().qdrant.collection_chunks
    points = []

    for item, vector in zip(item_list, vectors, strict=True):
        point_id = stable_point_id(item["paper_id"], item["chunk_id"])
        payload = {key: value for key, value in item.items() if key != "vector"}
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
    log.info(f"qdrant upsert {len(points)} points into {collection_name}")

    return len(points)


def list_chunks_for_paper(
    paper_id: str,
    *,
    page_size: int = 256,
) -> list[dict[str, Any]]:
    """按 paper_id 分页读取 Qdrant 旧 chunk 快照, 不读取向量。"""
    from qdrant_client.http import models as qdrant_models

    if page_size <= 0:
        raise ValueError("page_size must be positive")

    client = get_client()
    collection_name = cfg.load().qdrant.collection_chunks
    query_filter = qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="paper_id",
                match=qdrant_models.MatchValue(value=paper_id),
            )
        ]
    )
    fields = [
        "paper_id",
        "chunk_id",
        "content_id",
        "embedding_version",
        "payload_fingerprint",
    ]
    points: list[dict[str, Any]] = []
    offset = None

    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=query_filter,
            limit=page_size,
            offset=offset,
            with_payload=fields,
            with_vectors=False,
        )
        for record in records:
            payload = dict(record.payload or {})
            payload["point_id"] = str(record.id)
            points.append(payload)
        if next_offset is None:
            break
        offset = next_offset

    return points


def overwrite_chunk_payload(item: dict[str, Any]) -> None:
    """完整覆盖单个 chunk payload, 删除新快照中不存在的旧字段。"""
    client = get_client()
    client.overwrite_payload(
        collection_name=cfg.load().qdrant.collection_chunks,
        points=[stable_point_id(item["paper_id"], item["chunk_id"])],
        payload={key: value for key, value in item.items() if key != "vector"},
        wait=True,
    )


def delete_points(point_ids: Iterable[str]) -> int:
    """按精确 Point ID 删除旧 chunk。"""
    ids = list(dict.fromkeys(str(point_id) for point_id in point_ids))
    if not ids:
        return 0
    client = get_client()
    client.delete(
        collection_name=cfg.load().qdrant.collection_chunks,
        points_selector=ids,
        wait=True,
    )
    return len(ids)


def delete_chunks_for_paper(paper_id: str) -> None:
    """删除一篇论文在 Qdrant 中的全部 chunk。"""
    from qdrant_client.http import models as qdrant_models

    client = get_client()
    collection_name = cfg.load().qdrant.collection_chunks
    paper_filter = qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="paper_id",
                match=qdrant_models.MatchValue(value=paper_id),
            )
        ]
    )
    selector = qdrant_models.FilterSelector(filter=paper_filter)

    client.delete(
        collection_name=collection_name,
        points_selector=selector,
        wait=True,
    )
    log.info(f"qdrant deleted paper_id={paper_id} from {collection_name}")


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
                match=qdrant_models.MatchValue(value=modality),
            )
        )

    query_filter = qdrant_models.Filter(must=conditions) if conditions else None

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
            hits = query_result.points if hasattr(query_result, "points") else query_result
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
            f"qdrant search degraded, returning empty result: {type(error).__name__}: {error}"
        )
        return []

    results: list[dict[str, Any]] = []

    for hit in hits:
        result = dict(hit.payload or {})
        result["score"] = float(hit.score)
        results.append(result)

    return results
