"""章节切分器: 把 paper.md 全文切分为按文档顺序排列的章节。

切片 1 实现 markdown 标题路径(`#{1,4} Title`)与无标题时的 Body 兜底。
切片 2 实现英文纯文本标题(PyMuPDF 降级产物)的四种形态与两个守卫。
标题清洗与描述性合法性在切片 3, markdown 优先级去重与 References
尾部过滤在切片 4, 中文纯文本规则在切片 5。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADER_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$", re.MULTILINE)

# 行内 Abstract: "Abstract— 正文" / "Abstract: 正文" / "Abstract- 正文"。
# 标题只吞掉 Abstract 和分隔符, 同一行剩余文字留给正文
_ABSTRACT_INLINE_RE = re.compile(r"^(abstract)\s*[-—–:]\s*", re.IGNORECASE)  # noqa: RUF001
# 孤立编号行: "1." / "2.1" / "IV." 独占一行, 标题在下一行
_STANDALONE_NUM_RE = re.compile(r"^(?:\d+(?:\.\d+)*|[IVX]+)\.?$")
# 行内编号标题: "2. Related Work" / "2.1 Evaluation", 编号与标题同行
_INLINE_NUM_TITLE_RE = re.compile(r"^((?:\d+(?:\.\d+)*|[IVX]+)\.?)\s+(.+?)\s*$")
_PAGE_MARKER_RE = re.compile(r"<!--\s*page\s+\d+\s*-->", re.IGNORECASE)
# 表格/图片标注行: 其后的孤立短语大概率是表头单元格而不是章节标题
_TABLE_CONTEXT_RE = re.compile(r"^(?:table|fig\.|figure)\b", re.IGNORECASE)

# 英文规范章节标题白名单(小写比较); 描述性标题的启发式判定在切片 3
_CANONICAL_HEADINGS = {
    "abstract",
    "introduction",
    "background",
    "preliminaries",
    "related work",
    "motivation",
    "method",
    "methods",
    "methodology",
    "approach",
    "model",
    "architecture",
    "implementation",
    "experiments",
    "experiment",
    "experimental setup",
    "evaluation",
    "results",
    "discussion",
    "analysis",
    "ablation study",
    "ablations",
    "case study",
    "conclusion",
    "conclusions",
    "future work",
    "limitations",
    "acknowledgments",
    "acknowledgements",
    "references",
    "bibliography",
    "appendix",
    "ethics statement",
    "broader impact",
}

# Abstract 与 References 在 PyMuPDF 提取中经常紧贴上一段落, 永远按标题处理
_GUARD_BYPASS = {"abstract", "references", "bibliography"}


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
    source: str  # 表示标题的来源, 是从 markdown 解析的, 还是从纯文本解析的


@dataclass
class _Line:
    """一行文本及其在原文中的字符区间(end 不含换行符)。"""

    start: int
    end: int
    text: str


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
    headers = _markdown_headers(md)
    headers.extend(_plain_headers(md))
    headers.sort(key=lambda h: (h.start, h.end))
    return _dedupe_overlaps(headers)


def _markdown_headers(md: str) -> list[_Header]:
    headers: list[_Header] = []
    for m in _HEADER_RE.finditer(md):
        name = m.group(2)
        if _valid_markdown_heading(name):
            headers.append(_Header(m.start(), m.end(), name, len(m.group(1)), "markdown"))
    return headers


def _plain_headers(md: str) -> list[_Header]:
    """逐行扫描英文纯文本标题; 每行按固定顺序尝试四种形态, 命中即停。

    孤立编号形态会吞掉下一行标题, 但扫描仍逐行前进, 标题行自身还会命中
    裸规范标题形态产生重叠, 由 _dedupe_overlaps 统一丢弃。
    """
    lines = _lines(md)
    headers: list[_Header] = []
    for i in range(len(lines)):
        if not lines[i].text.strip():
            continue
        for matcher in (
            _match_inline_abstract,
            _match_standalone_number,
            _match_inline_numbered,
            _match_bare_canonical,
        ):
            header = matcher(lines, i)
            if header is not None:
                headers.append(header)
                break
    return headers


def _match_inline_abstract(lines: list[_Line], i: int) -> _Header | None:
    line = lines[i]
    m = _ABSTRACT_INLINE_RE.match(line.text.strip())
    if not m:
        return None
    # 标题区间只覆盖 "Abstract" 与分隔符, 行内剩余文字属于正文
    lead = len(line.text) - len(line.text.lstrip())
    return _Header(line.start, line.start + lead + m.end(), m.group(1), 1, "plain")


def _match_standalone_number(lines: list[_Line], i: int) -> _Header | None:
    line = lines[i]
    stripped = line.text.strip()
    if not _STANDALONE_NUM_RE.match(stripped):
        return None
    if i + 1 >= len(lines) or not _paragraph_boundary_before(lines, i):
        return None
    title = lines[i + 1].text.strip()
    if not _is_canonical_heading(title):
        return None
    # 编号行与下一行标题合并为一个标题, 区间从编号行起点到标题行终点
    return _Header(line.start, lines[i + 1].end, title, _level_from_number(stripped), "plain")


def _match_inline_numbered(lines: list[_Line], i: int) -> _Header | None:
    line = lines[i]
    m = _INLINE_NUM_TITLE_RE.match(line.text.strip())
    if not m:
        return None
    if not _is_canonical_heading(m.group(2)) or not _paragraph_boundary_before(lines, i):
        return None
    return _Header(line.start, line.end, m.group(2), _level_from_number(m.group(1)), "plain")


def _match_bare_canonical(lines: list[_Line], i: int) -> _Header | None:
    line = lines[i]
    stripped = line.text.strip()
    if not _is_canonical_heading(stripped):
        return None
    if stripped.lower() not in _GUARD_BYPASS and (
        not _paragraph_boundary_before(lines, i) or _table_context_before(lines, i)
    ):
        return None
    return _Header(line.start, line.end, stripped, 1, "plain")


def _lines(md: str) -> list[_Line]:
    lines: list[_Line] = []
    pos = 0
    for raw in md.split("\n"):
        lines.append(_Line(start=pos, end=pos + len(raw), text=raw))
        pos += len(raw) + 1
    return lines


def _paragraph_boundary_before(lines: list[_Line], i: int) -> bool:
    """上一行是空行、页标记或孤立编号行时, 当前行位于段落边界。"""
    if i == 0:
        return True
    prev = lines[i - 1].text.strip()
    if not prev or _PAGE_MARKER_RE.fullmatch(prev):
        return True
    return bool(_STANDALONE_NUM_RE.match(prev))


def _table_context_before(lines: list[_Line], i: int) -> bool:
    """最近 2 个非空行(跳过页标记)里有表格/图片标注时, 判定处于表格上下文。"""
    checked = 0
    j = i - 1
    while j >= 0 and checked < 2:
        text = lines[j].text.strip()
        j -= 1
        if not text or _PAGE_MARKER_RE.fullmatch(text):
            continue
        if _TABLE_CONTEXT_RE.match(text):
            return True
        checked += 1
    return False


def _is_canonical_heading(name: str) -> bool:
    return name.strip().lower() in _CANONICAL_HEADINGS


def _level_from_number(number: str) -> int:
    """只用编号 token 计算层级: "1." 一级, "2.1" 二级, 罗马数字一级。

    基准实现把整行传进来, "2. Related Work" 的点被误计入层级; 重建版修正。
    """
    token = number.strip().rstrip(".")
    if re.fullmatch(r"[IVX]+", token):
        return 1
    return token.count(".") + 1


def _dedupe_overlaps(headers: list[_Header]) -> list[_Header]:
    """按 (start, end) 排序后的最小重叠去重: 起点落入前一标题区间的丢弃。"""
    kept: list[_Header] = []
    for h in headers:
        if kept and h.start < kept[-1].end:
            continue
        kept.append(h)
    return kept


# 验证 markdown 标题名称是否有效, 仅包含空格或标点符号的标题不算有效标题
def _valid_markdown_heading(name: str) -> bool:
    if not name:
        return False
    return bool(re.search(r"[A-Za-z0-9\u4e00-\u9fff]", name))
