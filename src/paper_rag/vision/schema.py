"""视觉摘要的共享数据结构与状态常量。

纯数据模块, 不含 IO。相对基准的唯一扩展是 ``VisualSummaryRequest.language``:
语言由 builder 的 ``read_language`` 一路传到提示词路由与缓存键, 是中文论文
增强不退化成英文摘要的枢纽。``PROMPT_VERSION`` 随双语提示词升到 v2, 让旧的
英文 v1 缓存自然失效。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STATUS_OK = "ok"
STATUS_FALLBACK = "fallback"
STATUS_CACHED = "cached"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"
STATUS_UNAVAILABLE = "unavailable"

PROMPT_VERSION = "v2"


@dataclass(frozen=True)
class VisualSummaryRequest:
    """一次图片摘要请求的全部输入。

    frozen 是刻意的: 请求对象会被哈希进缓存键, 不允许中途改写。
    """

    paper_id: str
    chunk_id: str
    modality: str
    asset_path: Path
    caption: str = ""
    surrounding_context: str = ""
    model: str | None = None
    language: str | None = None  # zh | en | None(不猜, 走基准英文行为)
    prompt_version: str = PROMPT_VERSION


@dataclass
class VisualSummaryResult:
    """一次摘要的结果与记账信息。

    可变 dataclass: 编排层会就地改写 ``status``(命中缓存改 cached、主路失败
    但兜底成功改 fallback)并回填 ``cache_key``。
    """

    status: str
    summary: str = ""
    provider: str | None = None
    model: str | None = None
    raw: dict[str, Any] | None = None
    error: str | None = None
    cache_key: str | None = None
    warnings: list[str] = field(default_factory=list)
