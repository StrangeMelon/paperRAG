"""章节切分器: 把 paper.md 全文切分为按文档顺序排列的章节。

切片 1 实现 markdown 标题路径(`#{1,4} Title`)与无标题时的 Body 兜底。
切片 2 实现英文纯文本标题(PyMuPDF 降级产物)的四种形态与两个守卫。
切片 3 实现标题清洗、描述性合法性判定、first-abstract 守卫与层级封顶。
切片 4 实现 markdown 优先级去重与 References 尾部过滤。
切片 5 实现中文纯文本标题(白名单/编号/行内摘要)与 zh/en/None 语言路由。
切片 5b 补充中文阿拉伯点分编号(1. / 1.1 / 2.3.1)与量词/列表句守卫。
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
# 表格/图片标注行: 其后的孤立短语大概率是表头单元格而不是章节标题;
# 中文分支要求编号跟随, 避免 "表示学习" 这类正常词开头的行被误判为标注
_TABLE_CONTEXT_RE = re.compile(
    r"^(?:table|fig\.|figure)\b|^(?:图|表|算法)\s*[0-9一二三四五六七八九十]",
    re.IGNORECASE,
)

# 清洗时从标题两端剥掉的空白与标点(含 em/en dash 与全角冒号、句号)
_STRIP_PUNCT = " \t:.-—–：。"  # noqa: RUF001

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

# 规范标题的前缀匹配: "Appendix A" / "Appendix B: Proofs" 也算规范标题
_CANONICAL_PREFIXES = ("appendix ",)

# ---- 中文纯文本标题(切片 5) ----

# 行内摘要: "摘要" 后跟全角或半角冒号, 标题只吞掉 "摘要" 与分隔符
_ZH_ABSTRACT_INLINE_RE = re.compile(r"^摘\s*要\s*[:：]\s*")  # noqa: RUF001
# 中文编号标题五形态: 一、 / (一) / 第X章 / 1、 / 阿拉伯点分(1. 1.1 2.3.1);
# (一) 取 2 级, 点分层级按段数由 _level_from_number 计算, 其余 1 级。
# 点分编号后的空格可省略("1.1实验设置" 也常见), 但纯数字必须带点,
# 避免把 "2023 年…" 这类年份行当成编号
_ZH_NUM_TITLE_RE = re.compile(
    r"^(?:[一二三四五六七八九十]+、"
    r"|（(?P<paren>[一二三四五六七八九十]+)）"  # noqa: RUF001
    r"|第[一二三四五六七八九十百0-9]+章"
    r"|[0-9]+、"
    r"|(?P<arabic>[0-9]+(?:\.[0-9]+)+\.?|[0-9]+\.))"
    r"\s*(?P<title>.+?)\s*$"
)
# 标题名开头的中文编号前缀, 清洗时剥掉("# 一、引言" 这类 markdown 中文标题);
# 点分分支覆盖英文前缀正则不管的无空格形态("# 1.1实验设置");
# 末分支剥中文期刊常见的 "纯数字直贴标题"("# 1综合能源服务系统物理架构"),
# 限 1-2 位数字, 避免剥掉 "2023年…" 这类年份开头的真标题
_ZH_NUMBER_PREFIX_RE = re.compile(
    r"^(?:[一二三四五六七八九十]+、"
    r"|（[一二三四五六七八九十]+）"  # noqa: RUF001
    r"|第[一二三四五六七八九十百0-9]+章"
    r"|[0-9]+、"
    r"|[0-9]+(?:\.[0-9]+)+\.?"
    r"|[0-9]+\."
    r"|[0-9]{1,2}(?=[\u4e00-\u9fff]))\s*"
)
# 图/表/算法 + 编号开头是图表标注, 一票否决; 必须带编号, 不误伤 "表示学习"
_ZH_BAD_PREFIX_RE = re.compile(r"^(?:图|表|算法)\s*[0-9一二三四五六七八九十]")
# 点分编号与小数在语法上无法区分("3.5 倍的提升" vs "3.5 实验设置"),
# 标题部分以计量单位字符开头的一票否决
_ZH_UNIT_PREFIX_RE = re.compile(r"^[倍%‰]")
# 真标题不含句读; 逗号/分号/句中句号说明是编号列表句或叙述句。
# 问号/叹号不在其列: 设问式标题("为什么需要检索增强?")是合法标题
_ZH_SENTENCE_PUNCT_RE = re.compile(r"[，。；,;]")  # noqa: RUF001
# 附录标题: "附录" / "附录A" / "附录 一" 等短编号; 中文没有空格分隔,
# 叙述句 "附录中给出…" 的误报面比英文更大, 故要求编号短促、整行匹配
_ZH_APPENDIX_RE = re.compile(r"^附\s*录\s*[A-Za-z0-9一二三四五六七八九十]{0,3}$")

# 中文规范章节标题白名单(白名单比较前先压掉内部空格, 兼容 "摘 要" 排版)
_ZH_CANONICAL_HEADINGS = {
    "摘要",
    "引言",
    "绪论",
    "前言",
    "背景",
    "研究背景",
    "相关工作",
    "研究现状",
    "方法",
    "研究方法",
    "模型",
    "系统设计",
    "实验",
    "实验设置",
    "实验结果",
    "实验与分析",
    "结果",
    "结果与分析",
    "讨论",
    "分析",
    "结论",
    "总结",
    "结论与展望",
    "展望",
    "致谢",
    "参考文献",
    "附录",
}

# 摘要与参考文献在中文扫描件里同样常紧贴上一段落, 跳过边界/表格守卫
_ZH_GUARD_BYPASS = {"摘要", "参考文献"}

# 尾部过滤的触发标题(中英共用一个过滤器)
_REFERENCE_TAIL_TRIGGERS = {"references", "参考文献"}


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

    language 是领域语言提示: "en" 只启用英文纯文本规则, "zh" 只启用中文,
    None 表示双语规则全开。markdown 标题路径始终语言中立。
    """
    headers = _collect_headers(md, language)
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


# 获取每个标题的名称、层级、在原文中的字符区间, 以及来源(供去重规则区分优先级)
def _collect_headers(md: str, language: str | None) -> list[_Header]:
    headers = _markdown_headers(md)
    headers.extend(_plain_headers(md, language))
    headers.sort(key=lambda h: (h.start, h.end))
    return _filter_reference_tail(_dedupe_headers(headers))


def _markdown_headers(md: str) -> list[_Header]:
    headers: list[_Header] = []
    for m in _HEADER_RE.finditer(md):
        name = _clean_heading(m.group(2))
        if _valid_markdown_heading(name):
            headers.append(_Header(m.start(), m.end(), name, len(m.group(1)), "markdown"))
    return headers


def _plain_headers(md: str, language: str | None) -> list[_Header]:
    """逐行扫描纯文本标题; 按语言路由选择形态列表, 每行按固定顺序尝试, 命中即停。

    孤立编号形态会吞掉下一行标题, 但扫描仍逐行前进, 标题行自身还会命中
    裸规范标题形态产生重叠, 由 _dedupe_headers 统一丢弃。
    """
    lines = _lines(md)
    matchers = _matchers_for(language)
    headers: list[_Header] = []
    for i in range(len(lines)):
        stripped = lines[i].text.strip()
        # 空行跳过; # 开头的行属于 markdown 语法域, 由 markdown 路径全权处理,
        # 否则清洗剥掉 # 后会命中裸规范标题, 产生与 markdown 标题竞争的重复项
        if not stripped or stripped.startswith("#"):
            continue
        for matcher in matchers:
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
    """英文编号形态不要求段落边界: 真实 PyMuPDF 产物整篇无空行, 标题紧贴上一段;
    误报由描述性合法性(白名单/大写比例/关键词)兜住, 与基准一致。"""
    line = lines[i]
    stripped = line.text.strip()
    if not _STANDALONE_NUM_RE.match(stripped):
        return None
    if i + 1 >= len(lines):
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


def _match_zh_inline_abstract(lines: list[_Line], i: int) -> _Header | None:
    line = lines[i]
    m = _ZH_ABSTRACT_INLINE_RE.match(line.text.strip())
    if not m:
        return None
    # 标题区间只覆盖 "摘要" 与分隔符; 名称统一存规范形式(压掉 "摘 要" 的空格)
    lead = len(line.text) - len(line.text.lstrip())
    return _Header(line.start, line.start + lead + m.end(), "摘要", 1, "plain")


def _match_zh_numbered(lines: list[_Line], i: int) -> _Header | None:
    """中文编号形态保留段落边界守卫: 中文合法性没有大写比例可用, 判别力弱于
    英文, 去守卫误报面过大; 中文主路径是 MinerU(markdown 标题), 不受影响。"""
    line = lines[i]
    m = _ZH_NUM_TITLE_RE.match(line.text.strip())
    if not m:
        return None
    title = _clean_heading(m.group("title"))
    if not _valid_zh_heading_name(title):
        return None
    if not _paragraph_boundary_before(lines, i):
        return None
    if m.group("paren"):
        level = 2
    elif m.group("arabic"):
        level = _level_from_number(m.group("arabic"))
    else:
        level = 1
    return _Header(line.start, line.end, title, level, "plain")


def _match_zh_bare_canonical(lines: list[_Line], i: int) -> _Header | None:
    line = lines[i]
    # 白名单比较与存储都用压掉内部空格的规范形式("摘 要" -> "摘要")
    name = _clean_heading(line.text).replace(" ", "")
    if not _is_zh_canonical(name):
        return None
    if name not in _ZH_GUARD_BYPASS and (
        not _paragraph_boundary_before(lines, i) or _table_context_before(lines, i)
    ):
        return None
    return _Header(line.start, line.end, name, 1, "plain")


_EN_MATCHERS = (
    _match_inline_abstract,
    _match_standalone_number,
    _match_inline_numbered,
    _match_bare_canonical,
)
_ZH_MATCHERS = (
    _match_zh_inline_abstract,
    _match_zh_numbered,
    _match_zh_bare_canonical,
)


def _matchers_for(language: str | None) -> tuple:
    """语言路由: en/zh 只启用本语言的纯文本规则, 其他值(含 None)双语全开。"""
    if language == "en":
        return _EN_MATCHERS
    if language == "zh":
        return _ZH_MATCHERS
    return _EN_MATCHERS + _ZH_MATCHERS


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
    """标题名清洗: 去 # 前缀与中英编号前缀, 压缩内部空白, 剥两端标点。"""
    value = value.strip()
    value = re.sub(r"^#+\s*", "", value)
    value = _NUMBER_PREFIX_RE.sub("", value)
    value = _ZH_NUMBER_PREFIX_RE.sub("", value)
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
    low = name.strip().lower()
    return low in _CANONICAL_HEADINGS or low.startswith(_CANONICAL_PREFIXES)


def _valid_zh_heading_name(name: str) -> bool:
    """编号形态的中文标题合法性: 黑名单前缀 → 白名单 → 2-30 字符 + 含中文。

    中文不按空格分词, 描述性判定用字符数而不是词数; 也没有大写比例可用,
    以 "必须含中文字符" 排除纯数字/纯符号行。单位字符开头(小数量词)与
    含句读(编号列表句/叙述句)的标题一票否决。
    """
    if not name:
        return False
    if _ZH_BAD_PREFIX_RE.match(name):
        return False
    if _ZH_UNIT_PREFIX_RE.match(name):
        return False
    if _is_zh_canonical(name):
        return True
    if not 2 <= len(name) <= 30:
        return False
    if re.search(r"[\[\]{}()（）]", name):  # noqa: RUF001
        return False
    if _ZH_SENTENCE_PUNCT_RE.search(name):
        return False
    return bool(re.search(r"[\u4e00-\u9fff]", name))


def _is_zh_canonical(name: str) -> bool:
    compact = name.replace(" ", "")
    return compact in _ZH_CANONICAL_HEADINGS or bool(_ZH_APPENDIX_RE.match(compact))


def _level_from_number(number: str) -> int:
    """只用编号 token 计算层级: "1." 一级, "2.1" 二级, 罗马数字一级, 封顶 4 级。

    基准实现把整行传进来, "2. Related Work" 的点被误计入层级; 重建版修正。
    """
    token = number.strip().rstrip(".")
    if re.fullmatch(r"[IVX]+", token):
        return 1
    return min(token.count(".") + 1, 4)


def _dedupe_headers(headers: list[_Header]) -> list[_Header]:
    """重叠去重: 默认先到先得, 但 markdown 标题可顶替已保留的纯文本标题。

    典型冲突是孤立编号行吞掉下一行的 markdown 标题: 纯文本标题起点更早先被
    保留, 而 markdown 标记比纯文本猜测可信, 名称与层级都应以 markdown 为准。
    基准还有同起点同名去重分支; 重建版的纯文本扫描跳过 # 开头的行, 两个来源
    不可能同起点, 同一行也只会命中一种形态, 该分支不可达, 故不保留。
    """
    kept: list[_Header] = []
    for h in headers:
        if kept and h.start < kept[-1].end:
            if h.source == "markdown" and kept[-1].source != "markdown":
                kept[-1] = h
            continue
        kept.append(h)
    return kept


def _filter_reference_tail(headers: list[_Header]) -> list[_Header]:
    """References / 参考文献 之后的标题一律丢弃, 仅放行附录标题并以之解除过滤。

    参考文献条目("编号 + 首字母大写标题" 或 "1、中文标题")与行内编号标题
    形态无法区分, 只能按位置过滤; 附录出现后恢复正常识别。中英共用一个过滤器,
    混排文档里任一语言的 References 标题都能触发。
    """
    kept: list[_Header] = []
    in_references = False
    for h in headers:
        if in_references and not _is_appendix_heading(h.name):
            continue
        kept.append(h)
        if h.name.lower() in _REFERENCE_TAIL_TRIGGERS:
            in_references = True
        elif _is_appendix_heading(h.name):
            in_references = False
    return kept


def _is_appendix_heading(name: str) -> bool:
    return name.lower().startswith("appendix") or bool(_ZH_APPENDIX_RE.match(name))


# 验证 markdown 标题名称是否有效, 仅包含空格或标点符号的标题不算有效标题
def _valid_markdown_heading(name: str) -> bool:
    if not name:
        return False
    return bool(re.search(r"[A-Za-z0-9\u4e00-\u9fff]", name))
