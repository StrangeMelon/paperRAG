"""多模态抽取器 extract_figures/tables/formulas 的行为契约测试。

切片 0: 图片抽取(![alt](path) 语法、空 alt、偏移不变量、多图顺序)。
切片 1: 表格抽取(管道表、公式残片守卫、strip 后 span 对齐修正)。
切片 2: 公式抽取($$...$$ 展示块, 含 DOTALL 跨行)。
切片 3: 语言路由(zh 前缀 图:/表:/公式:/上下文:/路径:, en/None 用基准英文)。

接口约定(2026-08-01 确认):

    extract_figures(body: str, *, language: str | None = None) -> list[MMChunk]
    extract_tables(body: str, *, language: str | None = None) -> list[MMChunk]
    extract_formulas(body: str, *, language: str | None = None) -> list[MMChunk]

不变量强化(基准表格块 span 含尾随空白但 raw 是 strip 后文本, 不可回切):
每个 MMChunk 满足 body[char_start:char_end] == raw。
"""

from __future__ import annotations

import pytest

from paper_rag.chunk.multimodal_chunker import (
    extract_figures,
    extract_formulas,
    extract_tables,
)


def _assert_raw_invariant(body: str, chunks) -> None:
    for c in chunks:
        assert body[c.char_start : c.char_end] == c.raw, f"raw 不可回切: {c.raw[:40]!r}"


# ---------------------------------------------------------------------------
# 切片 0: 图片抽取
# ---------------------------------------------------------------------------


def test_figure_basic_en() -> None:
    body = "See ![Overview diagram](figures/a.jpg) here."
    chunks = extract_figures(body)

    assert len(chunks) == 1
    c = chunks[0]
    assert c.modality == "figure"
    assert c.raw == "![Overview diagram](figures/a.jpg)"
    assert c.asset_rel_path == "figures/a.jpg"
    assert c.text == "Figure: Overview diagram\nContext: See here.\nPath: figures/a.jpg"
    assert c.alt == "Overview diagram"
    assert c.context == "See here."
    _assert_raw_invariant(body, chunks)


def test_figure_empty_alt_keeps_context() -> None:
    body = "前一段说明文字。\n\n![](figures/b.jpg)\n\n图1系统架构示意\n"
    chunks = extract_figures(body, language="zh")

    assert len(chunks) == 1
    c = chunks[0]
    assert c.alt == ""
    assert "图1系统架构示意" in c.context  # 图注在邻近上下文里
    _assert_raw_invariant(body, chunks)


def test_multiple_figures_in_order() -> None:
    body = "![](f/1.jpg)\n\nmiddle text\n\n![](f/2.jpg)\n"
    chunks = extract_figures(body)
    assert [c.asset_rel_path for c in chunks] == ["f/1.jpg", "f/2.jpg"]
    assert chunks[0].char_start < chunks[1].char_start


# ---------------------------------------------------------------------------
# 切片 1: 表格抽取
# ---------------------------------------------------------------------------

TABLE_BODY = (
    "Results are shown below.\n"
    "\n"
    "| Method | Score |\n"
    "| GraphGPS | 0.65 |\n"
    "| Graph-Mamba | 0.68 |\n"
    "\n"
    "Discussion follows.\n"
)


def test_table_basic_en() -> None:
    chunks = extract_tables(TABLE_BODY)

    assert len(chunks) == 1
    c = chunks[0]
    assert c.modality == "table"
    assert c.raw.startswith("| Method | Score |")
    assert c.raw.endswith("| Graph-Mamba | 0.68 |")
    assert c.text.startswith("Table:\n| Method | Score |")
    assert "Context: " in c.text
    _assert_raw_invariant(TABLE_BODY, chunks)  # 基准在此漂移(span 含尾随换行)


def test_table_artifact_single_cell_rejected() -> None:
    body = "text\n\n|Q|\n|Q|\n\nmore text\n"
    assert extract_tables(body) == []


def test_table_needs_two_rows() -> None:
    body = "text\n\n| a | b |\n\nmore text\n"
    assert extract_tables(body) == []


# ---------------------------------------------------------------------------
# 切片 2: 公式抽取
# ---------------------------------------------------------------------------


def test_formula_basic_en() -> None:
    body = "Energy is defined as $$E = mc^2$$ in physics."
    chunks = extract_formulas(body)

    assert len(chunks) == 1
    c = chunks[0]
    assert c.modality == "formula"
    assert c.raw == "$$E = mc^2$$"
    assert c.text.startswith("Formula: E = mc^2\nContext: ")
    _assert_raw_invariant(body, chunks)


def test_formula_multiline_dotall() -> None:
    body = "Loss:\n$$\nL = \\sum_i w_i\n$$\nas above.\n"
    chunks = extract_formulas(body)
    assert len(chunks) == 1
    assert chunks[0].text.startswith("Formula: L = \\sum_i w_i")
    _assert_raw_invariant(body, chunks)


# ---------------------------------------------------------------------------
# 切片 3: 语言路由
# ---------------------------------------------------------------------------


def test_zh_prefixes_route_all_three() -> None:
    fig = extract_figures("说明文字 ![架构图](f/a.jpg) 后文", language="zh")[0]
    assert fig.text == "图: 架构图\n上下文: 说明文字 后文\n路径: f/a.jpg"

    tab = extract_tables("前文\n\n| 方法 | 得分 |\n| 甲 | 1 |\n\n后文\n", language="zh")[0]
    assert tab.text.startswith("表:\n| 方法 | 得分 |")
    assert "\n上下文: " in tab.text

    formula = extract_formulas("定义 $$y = kx$$ 如上", language="zh")[0]
    assert formula.text.startswith("公式: y = kx\n上下文: ")


def test_en_and_none_use_baseline_prefixes() -> None:
    body = "text ![alt](f/a.jpg) tail"
    assert extract_figures(body, language="en")[0].text == extract_figures(body)[0].text
    assert extract_figures(body)[0].text.startswith("Figure: ")


def test_language_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        extract_figures("![a](b.jpg)", "zh")  # type: ignore[misc]
