"""论文采集流程使用的公共领域模型"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class PaperMeta(BaseModel):
    """由论文采集源生成的标准化元数据"""

    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    abstract: str | None = None
    language: Literal["zh", "en"] | None = None
    urls: list[str] = Field(default_factory=list)
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source: str = "unknown"
    extra: dict[str, Any] = Field(default_factory=dict)


class FetchResult(BaseModel):
    """一次论文采集操作的标准输出。"""

    meta: PaperMeta
    pdf_path: str
