"""上下文前缀: 为 chunk 的嵌入输入注入论文标题与章节的全局上下文。

`context_text = with_context(chunk.text, ...)` 是 BGE-M3 的嵌入输入(裸 text 走
BM25), 前缀直接塑造稠密向量。与基准的差异(2026-08-01 已确认):
- 语言路由: zh 论文用 `chunk.context_prefix_zh` 中文模板, en/None 用基准英文
  模板, 让中文论文的嵌入输入保持单语。
- 空值省略: title/section 为空时整段移除渲染出的 `[标签: ]` 空架子(基准把空串
  填入模板, 无标题论文的每个 chunk 都会嵌入死标签噪声); 两者都空时直接返回
  原文。段移除按 `[...: ]` 形态匹配, 自定义模板若不用该形态则空值退化为基准的
  空串填入行为。
"""

from __future__ import annotations

import re

from .. import config as cfg

# 空值渲染出的标签段: "[Title: ]" / "[标题: ]", 冒号(半角/全角)后无内容
_EMPTY_SEGMENT_RE = re.compile(r"\[[^\[\]]*?[:：]\s*\]")  # noqa: RUF001
# 段移除后残留的行尾/串首空白
_DANGLING_SPACE_RE = re.compile(r"[ \t]+(?=\n)|\A[ \t]+")


def with_context(text: str, *, title: str, section: str, language: str | None = None) -> str:
    c = cfg.load().chunk
    template = c.context_prefix_zh if language == "zh" else c.context_prefix  # 这里的设计意图是让中文论文的 embedding 输入保持中文上下文, 而不是给中文正文强行加英文标签
    prefix = template.format(title=title or "", section=section or "")
    prefix = _EMPTY_SEGMENT_RE.sub("", prefix)
    prefix = _DANGLING_SPACE_RE.sub("", prefix)
    if not prefix.strip():
        return text
    return f"{prefix}{text}"
