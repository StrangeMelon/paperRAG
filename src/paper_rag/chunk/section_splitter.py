"""章节切分器: 把 paper.md 全文切分为按文档顺序排列的章节。

切片 1 只实现 markdown 标题路径(`#{1,4} Title`)与无标题时的 Body 兜底。
纯文本标题(PyMuPDF 降级产物)在切片 2 实现, 中文纯文本规则在切片 5。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADER_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$", re.MULTILINE)


@dataclass
class RawSection:
    """一个章节: 名称、层级与正文在原文中的字符区间。"""

    idx: int
    name: str
    level: int
    start: int
    end: int
    body: str


@dataclass
class _Header:
    """标题行本身在原文中的区间; source 供后续切片的去重规则区分来源。"""

    start: int
    end: int
    name: str
    level: int
    source: str # 表示标题的来源, 是从 markdown 解析的, 还是从纯文本解析的


def split_sections(md: str, *, language: str | None = None) -> list[RawSection]:
    """把 markdown 全文切分为章节列表。

    language 是领域语言提示, 取值 zh | en | None, None 表示双语规则全开。
    markdown 标题路径语言中立, 该参数要到切片 5 才参与纯文本标题的路由。
    """
    headers = _collect_headers(md)
    if not headers:
        return [RawSection(idx=0, name="Body", level=1, start=0, end=len(md), body=md.strip())]

    sections: list[RawSection] = []
    for i, h in enumerate(headers):
        start = h.end
        end = headers[i + 1].start if i + 1 < len(headers) else len(md)
        sections.append(
            RawSection(
                idx=i,
                name=h.name,
                level=h.level,
                start=start,
                end=end,
                body=md[start:end].strip(),
            )
        )
    return sections


# 获取每个标题的名称、层级、在原文中的字符区间, 以及来源(用于后续切片的去重规则)
def _collect_headers(md: str) -> list[_Header]:
    headers: list[_Header] = []
    for m in _HEADER_RE.finditer(md):
        name = m.group(2)
        if _valid_markdown_heading(name):
            headers.append(_Header(m.start(), m.end(), name, len(m.group(1)), "markdown"))
    return headers


# 验证 markdown 标题名称是否有效, 仅包含空格或标点符号的标题不算有效标题
def _valid_markdown_heading(name: str) -> bool:
    if not name:
        return False
    return bool(re.search(r"[A-Za-z0-9\u4e00-\u9fff]", name))
