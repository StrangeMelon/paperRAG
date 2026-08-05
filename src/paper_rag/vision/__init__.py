"""视觉增强: 给 figure/table chunk 补一段模型生成的文字摘要。

OCR 模式下 MinerU 把论文表格整体渲染成图片, 图表内部的数据(趋势、坐标轴、
对比数值)在纯文本链路里完全不可检索——本包把图片交给视觉模型, 把摘要追加进
chunk 的 text 与 context_text, 使图表同时进入稠密向量与 FTS5/BM25 词面索引。

对外只暴露编排入口与数据结构, 重型依赖(openai / transformers / PIL)全部在
函数内惰性导入。
"""

from __future__ import annotations

from .schema import (
    PROMPT_VERSION,
    STATUS_CACHED,
    STATUS_FAILED,
    STATUS_FALLBACK,
    STATUS_OK,
    STATUS_SKIPPED,
    STATUS_UNAVAILABLE,
    VisualSummaryRequest,
    VisualSummaryResult,
)

__all__ = [
    "PROMPT_VERSION",
    "STATUS_CACHED",
    "STATUS_FAILED",
    "STATUS_FALLBACK",
    "STATUS_OK",
    "STATUS_SKIPPED",
    "STATUS_UNAVAILABLE",
    "VisualSummaryRequest",
    "VisualSummaryResult",
    "enrich_chunks",
]


def enrich_chunks(*args, **kwargs):
    """惰性转发到 :mod:`paper_rag.vision.enrich`, 避免包导入即拉起配置。"""
    from .enrich import enrich_chunks as _impl

    return _impl(*args, **kwargs)
