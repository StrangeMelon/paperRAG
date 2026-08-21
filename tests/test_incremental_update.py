"""Qdrant 增量更新的纯逻辑契约测试。

这些测试先锁定三件事: 指纹输入、确定性 Point ID 和差量分类。
不触达真实 Qdrant 或 embedding 模型。
"""

from __future__ import annotations

import pytest

from paper_rag.store.incremental import (
    build_chunk_fingerprints,
    plan_incremental_update,
    stable_point_id,
)


def _chunk(**overrides) -> dict:
    item = {
        "chunk_id": "chunk-1",
        "paper_id": "paper-1",
        "context_text": "[Title: T]\ncontent",
        "text": "content",
        "section": "Introduction",
        "metadata": {"chunk_ordinal": 0},
    }
    item.update(overrides)
    return item


def test_fingerprints_hash_actual_embedding_input_and_payload() -> None:
    prepared = build_chunk_fingerprints(_chunk(), embedding_version="bge-m3:v1")

    assert prepared["content_id"]
    assert prepared["embedding_version"] == "bge-m3:v1"
    assert prepared["payload_fingerprint"]

    changed_context = build_chunk_fingerprints(
        _chunk(context_text="[Title: T]\nchanged"), embedding_version="bge-m3:v1"
    )
    assert changed_context["content_id"] != prepared["content_id"]
    assert changed_context["payload_fingerprint"] != prepared["payload_fingerprint"]


def test_payload_fingerprint_changes_without_content_change() -> None:
    original = build_chunk_fingerprints(_chunk(), embedding_version="bge-m3:v1")
    changed = build_chunk_fingerprints(_chunk(page=9), embedding_version="bge-m3:v1")

    assert changed["content_id"] == original["content_id"]
    assert changed["payload_fingerprint"] != original["payload_fingerprint"]


def test_stable_point_id_is_uuid_and_includes_paper_scope() -> None:
    first = stable_point_id("paper-1", "chunk-1")

    assert first == stable_point_id("paper-1", "chunk-1")
    assert first != stable_point_id("paper-2", "chunk-1")
    assert len(first) == 36
    assert first.count("-") == 4


def test_plan_classifies_vector_payload_skip_and_delete() -> None:
    unchanged = build_chunk_fingerprints(_chunk(), embedding_version="bge-m3:v1")
    payload_changed = build_chunk_fingerprints(_chunk(page=7), embedding_version="bge-m3:v1")
    vector_changed = build_chunk_fingerprints(
        _chunk(chunk_id="chunk-vector", context_text="[Title: T]\nnew"),
        embedding_version="bge-m3:v1",
    )
    added = build_chunk_fingerprints(_chunk(chunk_id="chunk-new"), embedding_version="bge-m3:v1")

    old_points = [
        {"point_id": stable_point_id("paper-1", "chunk-1"), **unchanged},
        {"point_id": stable_point_id("paper-1", "chunk-delete"), "chunk_id": "chunk-delete"},
    ]
    plan = plan_incremental_update([payload_changed, vector_changed, added], old_points)

    assert [item["chunk_id"] for item in plan.payload_updates] == ["chunk-1"]
    assert [item["chunk_id"] for item in plan.vector_updates] == ["chunk-vector", "chunk-new"]
    assert plan.skipped == []
    assert plan.delete_ids == {stable_point_id("paper-1", "chunk-delete")}


def test_plan_rejects_duplicate_new_point_ids() -> None:
    first = _chunk()
    second = _chunk(context_text="other")
    second["chunk_id"] = first["chunk_id"]
    prepared = [
        build_chunk_fingerprints(first, embedding_version="bge-m3:v1"),
        build_chunk_fingerprints(second, embedding_version="bge-m3:v1"),
    ]

    with pytest.raises(ValueError, match="duplicate point id"):
        plan_incremental_update(prepared, [])
