"""章节切分器: 把 paper.md 全文切分为按文档顺序排列的章节。

切片 1 实现 markdown 标题路径(`#{1,4} Title`)与无标题时的 Body 兜底。
切片 2 实现英文纯文本标题(PyMuPDF 降级产物)的四种形态与两个守卫。
切片 3 实现标题清洗、描述性合法性判定、first-abstract 守卫与层级封顶。
markdown 优先级去重与 References 尾部过滤在切片 4, 中文纯文本规则在切片 5。
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
# 标题名开头的编号前缀, 清洗时剥掉(层级计算不依赖这里, 见 _level_from_number)
_NUMBER_PREFIX_RE = re.compile(r"^(?:\d+(?:\.\d+)*|[IVX]+)\.?\s+")
_PAGE_MARKER_RE = re.compile(r"<!--\s*page\s+\d+\s*-->", re.IGNORECASE)
# 表格/图片标注行: 其后的孤立短语大概率是表头单元格而不是章节标题
_TABLE_CONTEXT_RE = re.compile(r"^(?:table|fig\.|figure)\b", re.IGNORECASE)

# 清洗时从标题两端剥掉的空白与标点(含 em dash 与 en dash)
_STRIP_PUNCT = " \t:.-—–"  # noqa: RUF001

# 图表/算法标注开头的标题一票否决, 在白名单与描述性规则之前生效
_BAD_HEADING_PREFIXES = ("fig.", "figure ", "table ", "algorithm ")

# 描述性标题启发式的领域关键词(小写子串匹配), 面向 RAG 论文语料
_DESCRIPTIVE_KEYWORDS = (
    "retrieval",
    "generation",
    "hallucination",
    "mitigation",
    "prompt",
    "tuning",
    "decoding",
    "faithfulness",
    "fine-tuning",
    "finetuning",
    "query",
    "confidence",
    "flare",
    "evaluation",
    "setup",
    "results",
    "analysis",
    "dataset",
    "training",
    "method",
)

# 英文规范章节标题白名单(小写比较)
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
        name = _clean_heading(m.group(2))
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
        stripped = lines[i].text.strip()
        # 空行跳过; # 开头的行属于 markdown 语法域, 由 markdown 路径全权处理,
        # 否则清洗剥掉 # 后会命中裸规范标题, 产生与 markdown 标题竞争的重复项
        if not stripped or stripped.startswith("#"):
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
    title = _clean_heading(lines[i + 1].text)
    if not _valid_heading_name(title, allow_descriptive=True):
        return None
    # first-abstract 守卫: 首个 Abstract 之前的描述性标题是封面/作者区噪声;
    # 白名单标题豁免, 因为正文首节 "1. Introduction" 之前未必有 Abstract
    if not _is_canonical_heading(title) and _before_first_abstract(lines, i):
        return None
    # 编号行与下一行标题合并为一个标题, 区间从编号行起点到标题行终点
    return _Header(line.start, lines[i + 1].end, title, _level_from_number(stripped), "plain")


def _match_inline_numbered(lines: list[_Line], i: int) -> _Header | None:
    line = lines[i]
    m = _INLINE_NUM_TITLE_RE.match(line.text.strip())
    if not m:
        return None
    title = _clean_heading(m.group(2))
    if not _valid_heading_name(title, allow_descriptive=True):
        return None
    if not _paragraph_boundary_before(lines, i):
        return None
    return _Header(line.start, line.end, title, _level_from_number(m.group(1)), "plain")


def _match_bare_canonical(lines: list[_Line], i: int) -> _Header | None:
    line = lines[i]
    name = _clean_heading(line.text)
    if not _valid_heading_name(name):
        return None
    if name.lower() not in _GUARD_BYPASS and (
        not _paragraph_boundary_before(lines, i) or _table_context_before(lines, i)
    ):
        return None
    return _Header(line.start, line.end, name, 1, "plain")


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


def _before_first_abstract(lines: list[_Line], i: int) -> bool:
    """第 i 行之前尚未出现任何 Abstract 标题(裸标题或行内形态)时为 True。"""
    for line in lines[:i]:
        text = line.text.strip()
        if _clean_heading(text).lower() == "abstract" or _ABSTRACT_INLINE_RE.match(text):
            return False
    return True


def _clean_heading(value: str) -> str:
    """标题名清洗: 去 # 前缀与编号前缀, 压缩内部空白, 剥两端标点。"""
    value = value.strip()
    value = re.sub(r"^#+\s*", "", value)
    value = _NUMBER_PREFIX_RE.sub("", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(_STRIP_PUNCT)


def _valid_heading_name(name: str, *, allow_descriptive: bool = False) -> bool:
    """标题名合法性总入口: 黑名单前缀 → 白名单 → (可选)描述性启发式。

    描述性启发式仅对编号形态开启: 词数 2-12、长度 3-120、无括号、不以 - 结尾,
    且满足"字母大写比例 >= 0.85"或"首字母大写 + 含描述性关键词"之一。
    """
    if not name:
        return False
    low = name.lower()
    if low.startswith(_BAD_HEADING_PREFIXES):
        return False
    if _is_canonical_heading(name):
        return True
    if not allow_descriptive:
        return False
    words = len(name.split())
    if len(name) > 120 or words > 12:
        return False
    if len(name) < 3 or words < 2:
        return False
    if re.search(r"[\[\]{}()]", name):
        return False
    if name.endswith("-"):
        return False
    letters = [c for c in name if c.isalpha()]
    if not letters:
        return False
    if sum(c.isupper() for c in letters) / len(letters) >= 0.85:
        return True
    return name[0].isupper() and any(keyword in low for keyword in _DESCRIPTIVE_KEYWORDS)


def _is_canonical_heading(name: str) -> bool:
    return name.strip().lower() in _CANONICAL_HEADINGS


def _level_from_number(number: str) -> int:
    """只用编号 token 计算层级: "1." 一级, "2.1" 二级, 罗马数字一级, 封顶 4 级。

    基准实现把整行传进来, "2. Related Work" 的点被误计入层级; 重建版修正。
    """
    token = number.strip().rstrip(".")
    if re.fullmatch(r"[IVX]+", token):
        return 1
    return min(token.count(".") + 1, 4)


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
