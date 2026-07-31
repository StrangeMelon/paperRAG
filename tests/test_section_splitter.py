"""章节切分器的行为契约测试。

切片 1: markdown 标题主路径 + 无标题兜底。
输入是解析层产出的 `parsed/<paper_id>/paper.md` 全文; 输出是按文档顺序排列的
`RawSection` 列表。本切片只覆盖 MinerU 产出的 markdown 标题(`#{1,4} Title`),
纯文本标题(PyMuPDF 降级产物)留给切片 2, 中文纯文本规则留给切片 5。

接口约定(切块层已确认方案, 2026-08-01):

    split_sections(md: str, *, language: str | None = None) -> list[RawSection]

`language` 是仅限关键字的领域语言提示(`zh | en | None`), markdown 标题路径
语言中立, 本切片仅验证参数存在且不改变 markdown 路径行为。
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
