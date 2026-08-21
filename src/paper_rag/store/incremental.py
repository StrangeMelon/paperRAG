"""论文 Chunk 增量同步的纯逻辑契约。

本模块不依赖 Qdrant 客户端或 embedding 模型, 负责生成可持久化指纹,
确定性 Point ID 和新旧快照差量计划。存储适配器只负责执行计划。
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any

_POINT_NAMESPACE = uuid.UUID("7e7c5b2b-3b7d-4f59-9f3e-82a4c8baf6d1")
_FINGERPRINT_FIELDS = {"content_id", "embedding_version", "payload_fingerprint"}


@dataclass
class IncrementalPlan:
    """Qdrant 单篇论文同步计划。"""

    vector_updates: list[dict[str, Any]] = field(default_factory=list)
    payload_updates: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    delete_ids: set[str] = field(default_factory=set)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _payload_for_fingerprint(item: dict[str, Any]) -> dict[str, Any]:
    """返回不含向量和派生指纹的业务 payload。"""
    return {
        key: value
        for key, value in item.items()
        if key != "vector" and key not in _FINGERPRINT_FIELDS
    }


def build_chunk_fingerprints(
    item: dict[str, Any],
    *,
    embedding_version: str,
) -> dict[str, Any]:
    """复制 chunk 并附加内容、embedding 和 payload 指纹。"""
    context_text = item.get("context_text")
    if not isinstance(context_text, str):
        raise ValueError("chunk context_text must be a string")
    if not embedding_version:
        raise ValueError("embedding_version must not be empty")

    prepared = dict(item)
    prepared["content_id"] = hashlib.sha256(context_text.encode("utf-8")).hexdigest()
    prepared["embedding_version"] = embedding_version
    prepared["payload_fingerprint"] = _sha256_json(_payload_for_fingerprint(prepared))
    return prepared


def stable_point_id(paper_id: str, chunk_id: str) -> str:
    """生成跨进程、跨重跑稳定的 UUIDv5 Point ID。"""
    if not paper_id or not chunk_id:
        raise ValueError("paper_id and chunk_id must not be empty")
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{paper_id}\0{chunk_id}"))


def plan_incremental_update(
    chunks: list[dict[str, Any]],
    old_points: list[dict[str, Any]],
) -> IncrementalPlan:
    """根据新 chunk 快照和 Qdrant 旧快照生成幂等差量计划。"""
    new_by_point: dict[str, dict[str, Any]] = {}
    for item in chunks:
        point_id = stable_point_id(str(item.get("paper_id") or ""), str(item.get("chunk_id") or ""))
        if point_id in new_by_point:
            raise ValueError(f"duplicate point id: {point_id}")
        new_by_point[point_id] = item

    old_by_point = {str(point["point_id"]): point for point in old_points}
    plan = IncrementalPlan(delete_ids=set(old_by_point) - set(new_by_point))

    for point_id, item in new_by_point.items():
        old = old_by_point.get(point_id)
        if old is None:
            plan.vector_updates.append(item)
            continue

        if (
            not old.get("content_id")
            or not old.get("embedding_version")
            or not old.get("payload_fingerprint")
            or old.get("content_id") != item.get("content_id")
            or old.get("embedding_version") != item.get("embedding_version")
        ):
            plan.vector_updates.append(item)
        elif old.get("payload_fingerprint") != item.get("payload_fingerprint"):
            plan.payload_updates.append(item)
        else:
            plan.skipped.append(item)

    return plan
