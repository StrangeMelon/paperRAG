"""章节切分器的行为契约测试。

切片 1: markdown 标题主路径(`#{1,4} Title`) + 无标题兜底。
切片 2: 英文纯文本标题(PyMuPDF 降级产物)四形态与守卫。
切片 3: 标题清洗、描述性合法性、层级边界与 first-abstract 守卫。
切片 4: markdown 优先级去重、References 尾部过滤与英文集成用例。
切片 5: 中文纯文本标题扩展与 zh/en/None 语言路由。
切片 5b: 中文阿拉伯点分编号(1. / 1.1 / 2.3.1)与量词/列表句守卫。
输入是解析层产出的 `parsed/<paper_id>/paper.md` 全文; 输出是按文档顺序排列的
`RawSection` 列表。

接口约定(切块层已确认方案, 2026-08-01):

    split_sections(md: str, *, language: str | None = None) -> list[RawSection]

`language` 是仅限关键字的领域语言提示(`zh | en | None`), markdown 标题路径
语言中立; 纯文本标题的语言路由规则见切片 5 区块。
"""

from __future__ import annotations

import importlib
from types import ModuleType

import pytest


def _splitter_module() -> ModuleType:
    return importlib.import_module("paper_rag.chunk.section_splitter")


def test_markdown_headings_split_in_document_order() -> None:
    mod = _splitter_module()
    md = (
        "Paper Title Line\n"
        "\n"
        "# Introduction\n"
        "\n"
        "Intro paragraph.\n"
        "\n"
        "## Approach\n"
        "\n"
        "Approach body text.\n"
        "\n"
        "#### Details\n"
        "\n"
        "Detail body text.\n"
    )

    sections = mod.split_sections(md)

    assert [s.name for s in sections] == ["Introduction", "Approach", "Details"]
    # 层级来自 # 号数量
    assert [s.level for s in sections] == [1, 2, 4]
    # idx 按文档顺序从 0 递增
    assert [s.idx for s in sections] == [0, 1, 2]
    assert sections[0].body == "Intro paragraph."
    assert sections[1].body == "Approach body text."
    assert sections[2].body == "Detail body text."
    # 首个标题之前的导言(标题行、作者等)不属于任何章节
    assert all("Paper Title Line" not in s.body for s in sections)
    # start/end 是正文在原文中的字符偏移, 与 body 一致(body 两端 strip)
    for s in sections:
        assert md[s.start : s.end].strip() == s.body
    assert sections[-1].end == len(md)


def test_section_body_keeps_page_markers_and_markdown() -> None:
    mod = _splitter_module()
    md = (
        "# Method\n"
        "\n"
        "First paragraph.\n"
        "\n"
        "<!-- page 2 -->\n"
        "\n"
        "Second paragraph.\n"
        "\n"
        "![fig](figures/f1.png)\n"
    )

    sections = mod.split_sections(md)

    assert len(sections) == 1
    # 页标记与图片等 markdown 元素原样保留在 body 中, 页码归属交给下游 builder
    assert "<!-- page 2 -->" in sections[0].body
    assert "![fig](figures/f1.png)" in sections[0].body


def test_no_heading_falls_back_to_single_body_section() -> None:
    mod = _splitter_module()
    md = "Plain text without any heading.\n\nSecond paragraph.\n"

    sections = mod.split_sections(md)

    assert len(sections) == 1
    only = sections[0]
    assert only.name == "Body"
    assert only.level == 1
    assert only.idx == 0
    # 兜底章节覆盖全文
    assert (only.start, only.end) == (0, len(md))
    assert only.body == md.strip()


def test_symbol_only_markdown_heading_is_ignored() -> None:
    mod = _splitter_module()
    # 不含字母 / 数字 / 中文的标题行不是有效标题, 整篇回退到 Body 兜底
    md = "# ***\n\nSome text after a decorative line.\n"

    sections = mod.split_sections(md)

    assert [s.name for s in sections] == ["Body"]


def test_heading_deeper_than_four_levels_is_not_a_heading() -> None:
    mod = _splitter_module()
    # 标题层级只认 1-4 个 # 号
    md = "##### Too Deep\n\nBody text.\n"

    sections = mod.split_sections(md)

    assert [s.name for s in sections] == ["Body"]


def test_chinese_markdown_heading_is_recognized() -> None:
    mod = _splitter_module()
    md = (
        "# 摘要\n"
        "\n"
        "本文提出了一种综合能源服务的信用评价方法。\n"
        "\n"
        "# 结论\n"
        "\n"
        "实验验证了方法的有效性。\n"
    )

    sections = mod.split_sections(md)

    # markdown 标题路径语言中立: 中文标题与英文标题同样切分
    assert [s.name for s in sections] == ["摘要", "结论"]
    assert [s.level for s in sections] == [1, 1]


@pytest.mark.parametrize("language", ["zh", "en", None])
def test_language_hint_does_not_change_markdown_path(language: str | None) -> None:
    mod = _splitter_module()
    md = "# Introduction\n\nIntro text.\n\n## Results\n\nResult text.\n"

    default_names = [s.name for s in mod.split_sections(md)]
    hinted_names = [s.name for s in mod.split_sections(md, language=language)]

    assert hinted_names == default_names == ["Introduction", "Results"]


def test_language_hint_is_keyword_only() -> None:
    mod = _splitter_module()

    # 领域语言提示必须显式命名传入, 防止和未来的位置参数混淆
    with pytest.raises(TypeError):
        mod.split_sections("# Introduction\n\ntext\n", "zh")


# ---------------------------------------------------------------------------
# 切片 2: 英文纯文本标题(PyMuPDF 降级产物)
#
# 四种形态: 行内 Abstract / 孤立编号行+标题行 / 行内编号标题 / 裸规范标题。
# 守卫: 段落边界、Table 上下文。first-abstract 守卫与描述性标题合法性耦合,
# 一并放到切片 3; 本切片的合法性判定只需规范标题白名单。
# 孤立编号形态与裸规范标题会在同一个标题行上重叠, 因此本切片需要最小的
# 重叠去重(后一个标题落在前一个标题区间内时丢弃); markdown 优先级与
# References 尾部过滤仍在切片 4。
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sep", ["—", ":", "-"])  # — = em dash 破折号
def test_inline_abstract_starts_abstract_section(sep: str) -> None:
    mod = _splitter_module()
    md = (
        "Great Paper Title\n"
        "Alice and Bob\n"
        "\n"
        f"Abstract{sep} We study retrieval methods.\n"
        "\n"
        "More abstract text.\n"
    )

    sections = mod.split_sections(md)

    assert [s.name for s in sections] == ["Abstract"]
    assert sections[0].level == 1
    # 标题只吞掉 "Abstract" 与分隔符, 同一行剩余文字属于正文
    assert sections[0].body == "We study retrieval methods.\n\nMore abstract text."


def test_standalone_number_line_followed_by_title_line() -> None:
    mod = _splitter_module()
    md = "Some preamble text.\n\n1.\nIntroduction\n\nIntro body.\n\n2.1\nEvaluation\n\nEval body.\n"

    sections = mod.split_sections(md)

    # 编号行与下一行标题合并为一个标题, 编号行不进入任何 body;
    # 标题行自身也会命中裸规范标题规则, 依赖最小重叠去重只留一个
    assert [s.name for s in sections] == ["Introduction", "Evaluation"]
    assert [s.level for s in sections] == [1, 2]
    assert sections[0].body == "Intro body."
    assert sections[1].body == "Eval body."
    for s in sections:
        assert md[s.start : s.end].strip() == s.body


def test_inline_numbered_title_line() -> None:
    mod = _splitter_module()
    md = "2. Related Work\n\nPrior systems exist.\n\n2.1 Evaluation\n\nSetup details.\n"

    sections = mod.split_sections(md)

    assert [s.name for s in sections] == ["Related Work", "Evaluation"]
    # 层级只看编号本身: "2." 一级, "2.1" 二级。基准把整行传给层级计算,
    # "2. Related Work" 被误判为二级; 重建版修正为只用编号计算
    assert [s.level for s in sections] == [1, 2]
    assert sections[0].body == "Prior systems exist."
    assert sections[1].body == "Setup details."


def test_bare_canonical_heading_requires_paragraph_boundary() -> None:
    mod = _splitter_module()
    # 前一行是空行, 构成段落边界, 是标题
    with_boundary = "Preamble text.\n\nMethod\n\nMethod body.\n"
    sections = mod.split_sections(with_boundary)
    assert [s.name for s in sections] == ["Method"]
    assert sections[0].body == "Method body."

    # 紧跟在普通文字后面, 只是断行句子的一部分, 不是标题
    without_boundary = "We call this the\nMethod\nof our system.\n"
    assert [s.name for s in mod.split_sections(without_boundary)] == ["Body"]


def test_page_marker_counts_as_paragraph_boundary() -> None:
    mod = _splitter_module()
    md = "Earlier paragraph text.\n<!-- page 3 -->\nConclusion\n\nFinal remarks.\n"

    sections = mod.split_sections(md)

    # 页标记等价于段落边界: 标题常出现在换页后第一行
    assert [s.name for s in sections] == ["Conclusion"]
    assert sections[0].body == "Final remarks."


def test_abstract_and_references_ignore_boundary_guards() -> None:
    mod = _splitter_module()
    # PyMuPDF 提取常把 References 紧贴上一段落, 这两个词永远按标题处理
    md = "The sentence runs into\nReferences\n[1] Someone. 2024.\n"
    sections = mod.split_sections(md)
    assert [s.name for s in sections] == ["References"]
    assert sections[0].body == "[1] Someone. 2024."

    md2 = "Great Paper Title\nAbstract\nWe study retrieval methods.\n"
    sections2 = mod.split_sections(md2)
    assert [s.name for s in sections2] == ["Abstract"]
    assert sections2[0].body == "We study retrieval methods."


def test_table_context_blocks_bare_canonical_heading() -> None:
    mod = _splitter_module()
    md = "Table 2: Ablation results on the dev set\n\nResults\n\nActual paragraph text.\n"

    # "Results" 出现在表格标注附近, 大概率是表头单元格而非章节标题
    assert [s.name for s in mod.split_sections(md)] == ["Body"]


def test_titlecase_non_canonical_line_is_not_a_heading() -> None:
    mod = _splitter_module()
    # 首字母大写但不在规范白名单里的孤立行不是标题(描述性规则在切片 3)
    md = "Our Proposed System\n\nWe describe the system here.\n"

    assert [s.name for s in mod.split_sections(md)] == ["Body"]


def test_markdown_and_plain_headers_merge_in_document_order() -> None:
    mod = _splitter_module()
    md = (
        "# Introduction\n"
        "\n"
        "Intro body.\n"
        "\n"
        "3. Experiments\n"
        "\n"
        "Experiment body.\n"
        "\n"
        "# Conclusion\n"
        "\n"
        "Concluding body.\n"
    )

    sections = mod.split_sections(md)

    # 两个来源的标题合并后仍按文档顺序排列
    assert [s.name for s in sections] == ["Introduction", "Experiments", "Conclusion"]
    assert [s.level for s in sections] == [1, 1, 1]
    assert [s.idx for s in sections] == [0, 1, 2]
    assert [s.body for s in sections] == ["Intro body.", "Experiment body.", "Concluding body."]


# ---------------------------------------------------------------------------
# 切片 3: 标题清洗、描述性合法性、层级边界与 first-abstract 守卫
#
# 清洗: 去掉编号前缀、压缩内部连续空白、剥掉尾部的 : . - 破折号等标点;
# markdown 标题名同样清洗, 但层级仍由 # 号数量决定。
# 描述性合法性(仅编号形态启用): 词数 2-12、长度 3-120、无括号、不以 - 结尾,
# 且满足"首字母大写 + 含描述性关键词"或"字母大写比例 >= 0.85"之一;
# 图表/算法前缀一票否决。
# first-abstract 守卫(仅孤立编号形态): 首个 Abstract 之前的描述性标题视为
# 封面/作者区噪声; 白名单标题豁免(切片 2 的孤立编号用例即豁免路径的回归)。
# 层级边界: 罗马数字固定一级, 深层编号封顶 4 级。
# ---------------------------------------------------------------------------


def test_markdown_heading_number_prefix_and_trailing_punct_are_cleaned() -> None:
    mod = _splitter_module()
    md = "# 1. Introduction:\n\nIntro body.\n\n## 2.1 Evaluation Setup —\n\nSetup body.\n"

    sections = mod.split_sections(md)

    # 名称清洗掉编号前缀与尾部标点; 层级仍来自 # 号数量而不是编号
    assert [s.name for s in sections] == ["Introduction", "Evaluation Setup"]
    assert [s.level for s in sections] == [1, 2]


def test_heading_internal_whitespace_is_collapsed() -> None:
    mod = _splitter_module()
    md = "# Related   Work\n\nBody text.\n"

    sections = mod.split_sections(md)

    assert [s.name for s in sections] == ["Related Work"]


def test_bare_canonical_with_trailing_period_is_cleaned() -> None:
    mod = _splitter_module()
    md = "Preamble text.\n\nConclusion.\n\nFinal body.\n"

    sections = mod.split_sections(md)

    # 裸规范标题先清洗再查白名单: "Conclusion." 也能命中
    assert [s.name for s in sections] == ["Conclusion"]
    assert sections[0].body == "Final body."


def test_numbered_descriptive_title_with_keyword_is_heading() -> None:
    mod = _splitter_module()
    md = "Preamble text.\n\n3. Retrieval Augmented Generation\n\nWe describe the pipeline.\n"

    sections = mod.split_sections(md)

    # 非白名单标题走描述性规则: 首字母大写 + 含描述性关键词(retrieval/generation)
    assert [s.name for s in sections] == ["Retrieval Augmented Generation"]
    assert [s.level for s in sections] == [1]
    assert sections[0].body == "We describe the pipeline."


def test_numbered_all_caps_title_is_heading() -> None:
    mod = _splitter_module()
    md = "Preamble text.\n\n4. DESIGN AND SCOPE\n\nCaps body.\n"

    sections = mod.split_sections(md)

    # 描述性规则的另一分支: 字母大写比例 >= 0.85, 不要求命中关键词
    assert [s.name for s in sections] == ["DESIGN AND SCOPE"]


def test_numbered_titlecase_without_keyword_is_rejected() -> None:
    mod = _splitter_module()
    # 首字母大写但既不在白名单、也不含描述性关键词、大写比例不足
    md = "Preamble text.\n\n3. Something Wonderful Here\n\nPlain body.\n"

    assert [s.name for s in mod.split_sections(md)] == ["Body"]


def test_numbered_title_with_brackets_or_figure_prefix_is_rejected() -> None:
    mod = _splitter_module()
    # 括号是描述性标题的黑名单字符
    md1 = "Preamble text.\n\n3. Results (preliminary)\n\nBody text.\n"
    assert [s.name for s in mod.split_sections(md1)] == ["Body"]

    # 图表/算法前缀黑名单在白名单和描述性规则之前生效
    md2 = "Preamble text.\n\n3. Figure 2 shows the pipeline\n\nBody text.\n"
    assert [s.name for s in mod.split_sections(md2)] == ["Body"]


def test_descriptive_standalone_number_blocked_before_first_abstract() -> None:
    mod = _splitter_module()
    # 首个 Abstract 之前: 孤立编号 + 描述性标题是封面/作者区噪声
    before = "1.\nRetrieval Augmented Generation\n\nFront matter noise.\n"
    assert [s.name for s in mod.split_sections(before)] == ["Body"]

    # Abstract 出现之后, 同样的形态是真正的章节标题
    after = "Abstract\nWe study retrieval.\n\n1.\nRetrieval Augmented Generation\n\nSection body.\n"
    sections = mod.split_sections(after)
    assert [s.name for s in sections] == ["Abstract", "Retrieval Augmented Generation"]
    assert sections[1].body == "Section body."


def test_deep_numbering_level_is_capped_at_four() -> None:
    mod = _splitter_module()
    md = "Preamble text.\n\n2.1.3.4.5 Evaluation Setup\n\nDeep body.\n"

    sections = mod.split_sections(md)

    assert [s.name for s in sections] == ["Evaluation Setup"]
    # 编号有 5 段, 层级封顶为 4
    assert [s.level for s in sections] == [4]


def test_roman_numeral_gives_level_one_for_descriptive_title() -> None:
    mod = _splitter_module()
    md = "Preamble text.\n\nIV. Evaluation Setup Details\n\nRoman body.\n"

    sections = mod.split_sections(md)

    assert [s.name for s in sections] == ["Evaluation Setup Details"]
    # 罗马数字编号不数点, 固定一级
    assert [s.level for s in sections] == [1]


# ---------------------------------------------------------------------------
# 切片 4: markdown 优先级去重 + References 尾部过滤 + 英文集成用例
#
# markdown 优先级: 标题重叠时不再一律"先到先得", markdown 来源的标题可以顶替
# 已保留的纯文本标题(MinerU 混排输出中 markdown 标记比纯文本猜测可信)。
# References 尾部过滤: References 节之后的"标题"多为参考文献条目噪声, 一律丢弃,
# 仅放行 Appendix* 标题并以之解除过滤; 为此 Appendix A / Appendix B 这类带编号
# 的附录标题也要算进规范白名单(前缀匹配)。
# ---------------------------------------------------------------------------


def test_markdown_heading_replaces_overlapping_plain_header() -> None:
    mod = _splitter_module()
    md = "Abstract\n\nSome overview text.\n\nIV.\n## Experimental Setup\n\nSetup details here.\n"

    sections = mod.split_sections(md)

    # 孤立编号行吞掉下一行的 markdown 标题形成重叠; markdown 来源更可信,
    # 顶替纯文本标题, 层级取 ## 的 2 级而不是罗马数字的 1 级
    assert [s.name for s in sections] == ["Abstract", "Experimental Setup"]
    assert [s.level for s in sections] == [1, 2]
    # 顶替后标题区间只剩 markdown 行, 编号行退回上一节正文(与基准一致的取舍)
    assert "IV." in sections[0].body
    assert sections[1].body == "Setup details here."


def test_citation_like_numbered_lines_after_references_are_dropped() -> None:
    mod = _splitter_module()
    md = (
        "Abstract\n"
        "\n"
        "We study retrieval augmented generation.\n"
        "\n"
        "1. Introduction\n"
        "\n"
        "Intro body.\n"
        "\n"
        "References\n"
        "\n"
        "1. Retrieval Augmented Generation Survey\n"
        "\n"
        "2. Prompt Tuning Methods Overview\n"
    )

    sections = mod.split_sections(md)

    # 参考文献条目 "编号 + 首字母大写标题" 与行内编号标题形态无法区分,
    # 只能靠位置过滤: References 之后的标题一律不再开新节
    assert [s.name for s in sections] == ["Abstract", "Introduction", "References"]
    assert "Retrieval Augmented Generation Survey" in sections[2].body
    assert "Prompt Tuning Methods Overview" in sections[2].body


def test_canonical_heading_after_references_is_dropped() -> None:
    mod = _splitter_module()
    md = (
        "Preamble text.\n"
        "\n"
        "Conclusion\n"
        "\n"
        "Final remarks.\n"
        "\n"
        "References\n"
        "\n"
        "Discussion\n"
        "\n"
        "These notes follow the bibliography.\n"
    )

    sections = mod.split_sections(md)

    # 白名单标题也不豁免: References 之后只有 Appendix* 能开新节
    assert [s.name for s in sections] == ["Conclusion", "References"]
    assert "Discussion" in sections[1].body


def test_appendix_heading_survives_reference_tail() -> None:
    mod = _splitter_module()
    md = (
        "Abstract\n"
        "\n"
        "We analyze retrieval methods.\n"
        "\n"
        "References\n"
        "\n"
        "1. Retrieval Augmented Generation Survey\n"
        "\n"
        "Appendix A\n"
        "\n"
        "Extra material body.\n"
    )

    sections = mod.split_sections(md)

    # "Appendix A" 靠白名单前缀匹配(appendix + 空格)成为规范标题, 并穿过尾部过滤
    assert [s.name for s in sections] == ["Abstract", "References", "Appendix A"]
    assert sections[2].level == 1
    assert sections[2].body == "Extra material body."


def test_appendix_resets_reference_tail_filter() -> None:
    mod = _splitter_module()
    md = (
        "References\n"
        "\n"
        "1. Retrieval Augmented Generation Survey\n"
        "\n"
        "Appendix A\n"
        "\n"
        "Supplementary material body.\n"
        "\n"
        "Acknowledgments\n"
        "\n"
        "Thanks to the reviewers.\n"
    )

    sections = mod.split_sections(md)

    # Appendix 解除过滤: 其后的正常标题重新生效。
    # 注意附录正文首行不能以 "Appendix " 开头, 否则会被前缀白名单误判成
    # 标题 —— 这是与基准一致的已知误报面, 留待后续切片评估是否收紧
    assert [s.name for s in sections] == ["References", "Appendix A", "Acknowledgments"]
    assert sections[2].body == "Thanks to the reviewers."


def test_english_plain_text_integration() -> None:
    mod = _splitter_module()
    md = (
        "FLARE: Active Retrieval Augmented Generation\n"
        "\n"
        "Jane Doe, John Smith\n"
        "\n"
        "<!-- page 1 -->\n"
        "\n"
        "Abstract— We propose an active retrieval augmented generation method.\n"
        "\n"
        "1.\n"
        "Introduction\n"
        "\n"
        "Long-form generation often hallucinates facts.\n"
        "\n"
        "2. Related Work\n"
        "\n"
        "Prior systems retrieve only once per query.\n"
        "\n"
        "2.1 Query Formulation Methods\n"
        "\n"
        "The model decides when and what to retrieve.\n"
        "\n"
        "Table 1: Main results on all datasets\n"
        "\n"
        "Results\n"
        "\n"
        "3. Conclusion\n"
        "\n"
        "We presented FLARE and its evaluation.\n"
        "\n"
        "References\n"
        "\n"
        "1. Retrieval Augmented Generation Survey\n"
        "\n"
        "Appendix A\n"
        "\n"
        "Additional prompt details.\n"
    )

    sections = mod.split_sections(md)

    # 英文纯文本链路集成: 行内 Abstract / 孤立编号 / 行内编号(含子节) /
    # Table 上下文守卫 / References 尾部过滤 / Appendix 放行, 一篇全覆盖
    assert [s.name for s in sections] == [
        "Abstract",
        "Introduction",
        "Related Work",
        "Query Formulation Methods",
        "Conclusion",
        "References",
        "Appendix A",
    ]
    assert [s.level for s in sections] == [1, 1, 1, 2, 1, 1, 1]
    assert [s.idx for s in sections] == list(range(7))
    # 标题行与作者行不属于任何章节
    assert all("Jane Doe" not in s.body for s in sections)
    # 行内 Abstract 的同一行剩余文字进入正文
    assert sections[0].body == "We propose an active retrieval augmented generation method."
    # 表格标注与被守卫拦下的 "Results" 留在上一节正文里
    assert "Table 1: Main results on all datasets" in sections[3].body
    assert "Results" in sections[3].body
    # 参考文献条目与附录正文各归其位
    assert "Retrieval Augmented Generation Survey" in sections[5].body
    assert sections[6].body == "Additional prompt details."
    for s in sections:
        assert md[s.start : s.end].strip() == s.body


# ---------------------------------------------------------------------------
# 切片 5: 中文纯文本标题扩展与 zh/en/None 语言路由
#
# 中文规范白名单(摘要/引言/相关工作/结论/参考文献/附录…), 白名单比较前压掉
# 内部空格("摘 要" 这类 OCR 排版空格)。中文编号四形态: 一、 / (一) / 第X章
# / 1、, 层级 (一) 为 2 其余为 1。行内 "摘要" 冒号切分同英文 Abstract。
# 合法性按 2-30 字符数(中文不按空格分词), 必须含中文字符; 图/表/算法+编号
# 前缀一票否决(必须带编号, 不误伤 "表示学习" 这类正常词)。
# 参考文献→附录 尾部过滤与英文共用一个过滤器; 附录标题要求 "附录+短编号",
# 叙述句 "附录中给出…" 不算(比英文 Appendix 前缀规则更紧, 中文没有空格分隔,
# 误报面更大, 故从一开始就收紧)。
# 路由: language="en" 只启用英文纯文本规则, "zh" 只启用中文, None 双语全开;
# markdown 标题路径始终语言中立。
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sep", ["：", ":"])  # noqa: RUF001  全角/半角冒号
def test_zh_inline_abstract_splits_section(sep: str) -> None:
    mod = _splitter_module()
    md = f"中文论文标题\n作者甲 作者乙\n\n摘要{sep}本文提出一种检索增强方法。\n\n更多摘要内容。\n"

    sections = mod.split_sections(md, language="zh")

    assert [s.name for s in sections] == ["摘要"]
    assert sections[0].level == 1
    # 标题只吞掉 "摘要" 与分隔符, 同一行剩余文字属于正文
    assert sections[0].body == "本文提出一种检索增强方法。\n\n更多摘要内容。"


def test_zh_spaced_heading_is_normalized() -> None:
    mod = _splitter_module()
    # OCR 常把标题字符用空格隔开排版: "摘 要" / "结 论"
    md = "摘 要：本文研究中文问答。\n\n正文内容。\n\n结 论\n\n方法有效。\n"  # noqa: RUF001

    sections = mod.split_sections(md, language="zh")

    # 名称压掉内部空格后再查白名单, 存储的也是规范形式
    assert [s.name for s in sections] == ["摘要", "结论"]


def test_zh_numbered_heading_four_forms() -> None:
    mod = _splitter_module()
    md = (
        "摘要：本文研究检索增强生成。\n"  # noqa: RUF001
        "\n"
        "一、引言\n"
        "\n"
        "引言正文。\n"
        "\n"
        "第三章 实验分析\n"
        "\n"
        "实验正文。\n"
        "\n"
        "（一）数据集构建\n"  # noqa: RUF001
        "\n"
        "数据集正文。\n"
        "\n"
        "1、结果讨论\n"
        "\n"
        "讨论正文。\n"
    )

    sections = mod.split_sections(md, language="zh")

    assert [s.name for s in sections] == ["摘要", "引言", "实验分析", "数据集构建", "结果讨论"]
    # (一) 是子级编号取 2 级, 其余形态 1 级
    assert [s.level for s in sections] == [1, 1, 1, 2, 1]
    assert sections[3].body == "数据集正文。"


def test_zh_numbered_title_char_count_bounds() -> None:
    mod = _splitter_module()
    # 标题字符数下限 2: 单字标题不合法
    too_short = "前置说明。\n\n一、法\n\n正文内容。\n"
    assert [s.name for s in mod.split_sections(too_short, language="zh")] == ["Body"]

    # 上限 30: 整句式的长"标题"不合法
    long_title = (
        "一、" + "基于大规模预训练语言模型的检索增强生成方法在开放域问答任务上的实验研究" + "\n"
    )
    too_long = f"前置说明。\n\n{long_title}\n正文内容。\n"
    assert [s.name for s in mod.split_sections(too_long, language="zh")] == ["Body"]


def test_zh_figure_table_prefix_rejected_but_normal_word_kept() -> None:
    mod = _splitter_module()
    # 图/表/算法 + 编号开头是图表标注, 一票否决
    md1 = "前置说明。\n\n二、表 2 主要实验结果\n\n正文内容。\n"
    assert [s.name for s in mod.split_sections(md1, language="zh")] == ["Body"]

    # 黑名单要求编号跟随: "表示学习方法" 以 "表" 开头但不是图表标注
    md2 = "前置说明。\n\n二、表示学习方法\n\n正文内容。\n"
    sections = mod.split_sections(md2, language="zh")
    assert [s.name for s in sections] == ["表示学习方法"]


def test_zh_table_context_blocks_bare_canonical() -> None:
    mod = _splitter_module()
    # "结果" 出现在表格标注附近, 大概率是表头单元格而非章节标题
    md = "表 1 各方法在数据集上的对比\n\n结果\n\n真正的正文段落。\n"

    assert [s.name for s in mod.split_sections(md, language="zh")] == ["Body"]


def test_zh_reference_tail_filter_and_appendix() -> None:
    mod = _splitter_module()
    md = (
        "摘要：本文研究检索增强生成。\n"  # noqa: RUF001
        "\n"
        "一、结论\n"
        "\n"
        "结论正文。\n"
        "\n"
        "参考文献\n"
        "\n"
        "1、检索增强生成研究综述\n"
        "\n"
        "附录A\n"
        "\n"
        "补充实验细节。\n"
    )

    sections = mod.split_sections(md, language="zh")

    # 参考文献之后的条目噪声被过滤, 附录A 放行并解除过滤
    assert [s.name for s in sections] == ["摘要", "结论", "参考文献", "附录A"]
    assert "检索增强生成研究综述" in sections[2].body
    assert sections[3].body == "补充实验细节。"


def test_zh_appendix_narrative_sentence_is_not_heading() -> None:
    mod = _splitter_module()
    # 中文附录规则要求 "附录 + 短编号", 段首叙述句不会像英文 Appendix 那样误报
    md = "前置说明。\n\n附录中给出了全部定理的证明。\n\n后续内容。\n"

    assert [s.name for s in mod.split_sections(md, language="zh")] == ["Body"]


def test_language_routing_selects_rule_sets() -> None:
    mod = _splitter_module()
    md = (
        "Abstract— We study bilingual retrieval.\n"
        "\n"
        "一、引言\n"
        "\n"
        "中文正文。\n"
        "\n"
        "2. Related Work\n"
        "\n"
        "English body.\n"
    )

    # None: 双语规则全开, 中英标题都识别
    both = [s.name for s in mod.split_sections(md)]
    assert both == ["Abstract", "引言", "Related Work"]

    # en: 中文纯文本规则关闭
    en_only = [s.name for s in mod.split_sections(md, language="en")]
    assert en_only == ["Abstract", "Related Work"]

    # zh: 英文纯文本规则关闭
    zh_only = [s.name for s in mod.split_sections(md, language="zh")]
    assert zh_only == ["引言"]


def test_zh_markdown_heading_number_prefix_is_cleaned() -> None:
    mod = _splitter_module()
    # markdown 路径语言中立, 但清洗要认识中文编号前缀
    md = "# 一、引言\n\n引言正文。\n\n## （一）数据集\n\n数据集正文。\n"  # noqa: RUF001

    sections = mod.split_sections(md)

    assert [s.name for s in sections] == ["引言", "数据集"]
    # 层级仍由 # 号数量决定
    assert [s.level for s in sections] == [1, 2]


def test_zh_integration_mini_paper() -> None:
    mod = _splitter_module()
    md = (
        "基于检索增强生成的中文问答系统研究\n"
        "\n"
        "张三 李四\n"
        "\n"
        "摘 要：本文提出一种检索增强的问答方法。\n"  # noqa: RUF001
        "\n"
        "一、引言\n"
        "\n"
        "大模型在开放域问答中容易产生幻觉。\n"
        "\n"
        "二、相关工作\n"
        "\n"
        "（一）检索增强生成\n"  # noqa: RUF001
        "\n"
        "已有工作在生成前只检索一次。\n"
        "\n"
        "表 1 各方法在数据集上的对比\n"
        "\n"
        "结果\n"
        "\n"
        "三、结论\n"
        "\n"
        "实验验证了方法的有效性。\n"
        "\n"
        "参考文献\n"
        "\n"
        "1、检索增强生成研究综述\n"
        "\n"
        "附录A\n"
        "\n"
        "补充实验细节。\n"
    )

    for language in ("zh", None):
        sections = mod.split_sections(md, language=language)

        # 中文纯文本链路集成: 行内摘要 / 三种编号 / 表格守卫 / 尾部过滤 / 附录放行
        assert [s.name for s in sections] == [
            "摘要",
            "引言",
            "相关工作",
            "检索增强生成",
            "结论",
            "参考文献",
            "附录A",
        ]
        assert [s.level for s in sections] == [1, 1, 1, 2, 1, 1, 1]
        assert [s.idx for s in sections] == list(range(7))
        # 标题行与作者行不属于任何章节
        assert all("张三" not in s.body for s in sections)
        # 行内摘要的同一行剩余文字进入正文
        assert sections[0].body == "本文提出一种检索增强的问答方法。"
        # 表格标注与被守卫拦下的 "结果" 留在上一节正文里
        assert "表 1 各方法在数据集上的对比" in sections[3].body
        assert "结果" in sections[3].body
        # 参考文献条目与附录正文各归其位
        assert "检索增强生成研究综述" in sections[5].body
        assert sections[6].body == "补充实验细节。"
        for s in sections:
            assert md[s.start : s.end].strip() == s.body


# ---------------------------------------------------------------------------
# 切片 5b: 中文阿拉伯点分编号(1. / 1.1 / 2.3.1)
#
# 中文理工科学报的主流编号是阿拉伯点分("1.1 实验设置", GB/T 7713 风格),
# 切片 5 的四形态未覆盖: en 规则能匹配编号但中文标题过不了英文合法性,
# zh 规则不认该形态, 结果整行落入上一节正文。本切片补第五形态:
# 层级由点分段数决定(1. 一级, 1.1 二级, 2.3.1 三级), 编号后空格可省略
# ("1.1实验设置" 也常见)。两类该形态特有的误报面加守卫: 小数量词
# ("3.5 倍的提升")用单位字符黑名单(倍/%/‰)拦截; 编号列表句用句读
# (逗号/分号/句中句号)一票否决。问号/叹号不否决, 设问式标题是合法标题。
# ---------------------------------------------------------------------------


def test_zh_arabic_dotted_heading_three_depths() -> None:
    mod = _splitter_module()
    md = (
        "摘要：本文研究检索增强生成。\n"  # noqa: RUF001
        "\n"
        "1. 引言\n"
        "\n"
        "引言正文。\n"
        "\n"
        "1.1 研究背景与动机\n"
        "\n"
        "背景正文。\n"
        "\n"
        "2.3.1 数据集构建\n"
        "\n"
        "数据集正文。\n"
    )

    for language in ("zh", None):
        sections = mod.split_sections(md, language=language)
        assert [s.name for s in sections] == ["摘要", "引言", "研究背景与动机", "数据集构建"]
        # 点分形态的层级由段数决定, 与英文编号同一套 _level_from_number
        assert [s.level for s in sections] == [1, 1, 2, 3]
        assert sections[2].body == "背景正文。"


def test_zh_arabic_dotted_no_space_and_markdown_cleaning() -> None:
    mod = _splitter_module()
    # 中文排版常省略编号后的空格
    md1 = "前置说明。\n\n1.1实验设置\n\n正文内容。\n"
    sections = mod.split_sections(md1, language="zh")
    assert [s.name for s in sections] == ["实验设置"]
    assert sections[0].level == 2

    # markdown 路径语言中立, 清洗同样要剥掉无空格的点分编号前缀
    md2 = "# 1.1实验设置\n\n正文内容。\n"
    sections = mod.split_sections(md2)
    assert [s.name for s in sections] == ["实验设置"]
    # markdown 标题层级仍由 # 号数量决定
    assert sections[0].level == 1


def test_zh_decimal_quantity_and_list_item_not_heading() -> None:
    mod = _splitter_module()
    # 小数量词: 段首 "3.5 倍…" 与章节号在语法上无法区分, 用单位字符黑名单拦截
    md1 = "前置说明。\n\n3.5 倍的性能提升来自检索模块\n\n后续内容。\n"
    assert [s.name for s in mod.split_sections(md1, language="zh")] == ["Body"]

    # 编号列表句: 真标题不含句读, 含逗号/分号的行一票否决
    md2 = "前置说明。\n\n2. 其次，构造负样本用于训练；\n\n后续内容。\n"  # noqa: RUF001
    assert [s.name for s in mod.split_sections(md2, language="zh")] == ["Body"]


def test_zh_arabic_dotted_english_line_not_captured_by_zh_rule() -> None:
    mod = _splitter_module()
    # 点分编号 + 英文标题仍归英文规则: zh 路由下不识别, None 下由 en 规则命中
    md = "前置说明。\n\n2.1 Experimental Setup\n\nBody text.\n"
    assert [s.name for s in mod.split_sections(md, language="zh")] == ["Body"]
    assert [s.name for s in mod.split_sections(md)] == ["Experimental Setup"]


def test_zh_markdown_digit_attached_prefix_is_cleaned() -> None:
    mod = _splitter_module()
    # 真实中文期刊 MinerU 产物: 纯数字编号与标题无分隔("# 1综合能源服务系统物理架构"),
    # 英文前缀正则要求空格管不到; 限 1-2 位数字, 年份开头的标题("2023年…")不剥
    md = "# 1综合能源服务系统物理架构\n\n正文一。\n\n# 2023年度电力行业回顾\n\n正文二。\n"
    sections = mod.split_sections(md, language="zh")
    assert [s.name for s in sections] == ["综合能源服务系统物理架构", "2023年度电力行业回顾"]


# ---------------------------------------------------------------------------
# 真实验收回归: PyMuPDF 密排版面(无空行)
#
# 真实 PyMuPDF 产物整篇没有空行, 标题行紧贴上一段末行。切片 2 给英文编号
# 形态加的段落边界守卫比基准更严, 在真实文件上把 "1. Introduction" 等全部
# 拦掉, 整篇只剩 Abstract/References。修正为与基准一致: 英文编号形态不要求
# 段落边界, 由描述性合法性(白名单/大写比例/关键词)兜住误报。
# 中文编号形态保留边界守卫: 中文合法性没有大写比例可用, 判别力更弱,
# 去掉守卫误报面过大; 中文主路径是 MinerU(有 markdown 标题), 不受影响。
# ---------------------------------------------------------------------------


def test_dense_pymupdf_numbered_headings_without_blank_lines() -> None:
    mod = _splitter_module()
    # 版面取自真实 Graph-Mamba PyMuPDF 产物: 编号标题上一行是正文, 无空行
    md = (
        "Correspondence to: Bo Wang <bo@example.com>.\n"
        "1. Introduction\n"
        "Graph modeling has been widely used.\n"
        "ing its efficiency in long-range graph datasets.\n"
        "2. Related Work\n"
        "2.1. Graph Transformers\n"
        "Prior systems exist.\n"
    )
    sections = mod.split_sections(md, language="en")

    # "2.1. Graph Transformers" 无白名单/关键词命中, 被合法性拒绝留在正文里
    assert [s.name for s in sections] == ["Introduction", "Related Work"]
    assert "2.1. Graph Transformers" in sections[1].body


def test_dense_standalone_number_heading_without_blank_lines() -> None:
    mod = _splitter_module()
    # 孤立编号行同样可能紧贴上一段(换页/换栏处), 不要求段落边界。
    # 层级必须来自编号行("2.1" 二级): 若守卫拦下编号形态, 标题行会经
    # 裸规范标题侧门被识别成一级, 名字碰巧一样但层级错误
    md = "The previous paragraph ends here.\n2.1\nExperimental Setup\nSetup body.\n"
    sections = mod.split_sections(md, language="en")

    assert [s.name for s in sections] == ["Experimental Setup"]
    assert sections[0].level == 2
    assert sections[0].body == "Setup body."


def test_zh_numbered_heading_still_requires_boundary() -> None:
    mod = _splitter_module()
    # 中文编号形态保留边界守卫: 密排文本里断行产生的行首枚举不是标题
    md = "实验对比了三种方法\n一、基于稀疏检索的方法与\n二、基于稠密检索的方法。\n"

    assert [s.name for s in mod.split_sections(md, language="zh")] == ["Body"]
