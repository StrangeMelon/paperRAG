"""vision/schema.py 与 vision 配置的边界契约。

本模块只钉纯数据结构与配置键, 不触网、不读图片。
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from paper_rag import config as cfg
from paper_rag.vision import schema as sch


def test_status_constants_cover_six_states():
    assert sch.STATUS_OK == "ok"
    assert sch.STATUS_FALLBACK == "fallback"
    assert sch.STATUS_CACHED == "cached"
    assert sch.STATUS_SKIPPED == "skipped"
    assert sch.STATUS_FAILED == "failed"
    assert sch.STATUS_UNAVAILABLE == "unavailable"


def test_prompt_version_bumped_to_v2_for_bilingual_prompt():
    # 双语提示词与中文标签是新行为, 旧 v1 英文缓存必须自然失效。
    assert sch.PROMPT_VERSION == "v2"


def test_request_carries_language_and_defaults_to_none():
    req = sch.VisualSummaryRequest(
        paper_id="p1",
        chunk_id="c1",
        modality="figure",
        asset_path=Path("/tmp/x.png"),
    )
    assert req.language is None
    assert req.caption == ""
    assert req.surrounding_context == ""
    assert req.prompt_version == "v2"


def test_request_accepts_zh_language():
    req = sch.VisualSummaryRequest(
        paper_id="p1",
        chunk_id="c1",
        modality="table",
        asset_path=Path("/tmp/x.png"),
        caption="表 1 实验结果",
        surrounding_context="上下文",
        language="zh",
    )
    assert req.language == "zh"
    assert req.modality == "table"


def test_request_is_frozen():
    req = sch.VisualSummaryRequest(
        paper_id="p1", chunk_id="c1", modality="figure", asset_path=Path("/tmp/x.png")
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        req.language = "zh"  # type: ignore[misc]


def test_result_defaults_are_empty_and_warnings_isolated():
    a = sch.VisualSummaryResult(status=sch.STATUS_OK)
    b = sch.VisualSummaryResult(status=sch.STATUS_FAILED)
    a.warnings.append("w")
    assert b.warnings == []  # default_factory 而非共享可变默认值
    assert a.summary == ""
    assert a.provider is None
    assert a.raw is None


def test_vision_config_has_temperature_and_extra_body():
    v = cfg.load().vision
    # 智谱 OpenAI 兼容口径: temperature 区间为 (0,1), 0 不适用, 故不写死 0。
    assert 0 < v.temperature < 1
    # GLM-4.6V 默认思考, reasoning 会吃光 max_tokens 致 content 空/截断
    # (2026-08-05 真实复现), 故默认关思考; 换非思考模型时可清空该键。
    assert v.extra_body == {"thinking": {"type": "disabled"}}
