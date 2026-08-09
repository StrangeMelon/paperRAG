"""稠密检索: 查询经 BGE-M3 编码后在 paper_chunks 向量集合中带过滤检索。

查询侧与文档侧共用同一编码器(同一向量空间); 查询编码裸文本, 不加文档侧的
上下文前缀。降级语义继承 qdrant_store.search(出错返回空列表), 本层保持薄。
"""

from __future__ import annotations

from ..embed import bge_m3
from ..store import qdrant_store


def retrieve(
    query: str,
    top_k: int = 8,
    paper_ids: list[str] | None = None,
    modality: str | None = None,
) -> list[dict]:
    qvec = bge_m3.encode_one(query)
    return qdrant_store.search(
        qvec,
        top_k=top_k,
        paper_ids=paper_ids,
        modality=modality,
        raise_on_error=True,
    )
