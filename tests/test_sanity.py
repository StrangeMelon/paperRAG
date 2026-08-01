"""章节完整性打分器 grade_sections 的行为契约测试。

切片 0: 英文基准保真(四值阶梯、大小写不敏感、子串匹配)。
切片 1: 中文关键词表(zh 路由; 真实中文期刊节名形态必须判 complete)。
切片 2: 语言路由(zh/en/None=双表并集; 关键词表互不越界)。

接口约定(2026-08-01 确认):

    grade_sections(section_names: list[str], *, language: str | None = None) -> str
        -> "complete" | "partial" | "minimal" | "broken"

输出标签与基准四值完全一致(`parsed_with = "{parser}+{quality}"` 契约不能动)。
基准缺陷: 关键词表全英文, 完美解析的中文论文会被判 broken 而遭误过滤。
"""

from __future__ import annotations

from paper_rag.chunk.sanity import grade_sections

# ---------------------------------------------------------------------------
# 切片 0: 英文基准保真
# ---------------------------------------------------------------------------


def test_en_complete_all_four_areas() -> None:
    names = ["Abstract", "Introduction", "Method", "Experiments", "Conclusion"]
    assert grade_sections(names, language="en") == "complete"


def test_en_partial_missing_conclusion_but_has_experiment() -> None:
    names = ["Introduction", "Approach", "Evaluation"]
    assert grade_sections(names, language="en") == "partial"


def test_en_minimal_only_intro() -> None:
    assert grade_sections(["Introduction"], language="en") == "minimal"


def test_en_broken_no_canonical_area() -> None:
    assert grade_sections(["Acknowledgements", "References"], language="en") == "broken"


def test_en_case_insensitive_and_substring() -> None:
    # 基准同款: 小写后子串匹配, "Experimental Setup" 命中 experiment 区
    names = ["INTRODUCTION", "Model Architecture", "Experimental Setup", "Future Work"]
    assert grade_sections(names, language="en") == "complete"


def test_empty_sections_is_broken() -> None:
    assert grade_sections([], language="en") == "broken"
    assert grade_sections([], language="zh") == "broken"
    assert grade_sections([]) == "broken"


# ---------------------------------------------------------------------------
# 切片 1: 中文关键词表
# ---------------------------------------------------------------------------


def test_zh_complete_canonical_names() -> None:
    names = ["摘要", "引言", "方法", "实验", "结论"]
    assert grade_sections(names, language="zh") == "complete"


def test_zh_complete_real_journal_style() -> None:
    # 真实中文期刊节名形态(综合能源期刊论文): 区域词内嵌在长节名里
    names = ["引言", "综合能源服务区块链网络架构", "交互模型", "算例分析", "结论"]
    assert grade_sections(names, language="zh") == "complete"


def test_zh_partial_missing_experiment_and_conclusion_areas() -> None:
    assert grade_sections(["引言", "系统模型", "总结"], language="zh") == "partial"


def test_zh_minimal_only_intro_area() -> None:
    assert grade_sections(["摘要"], language="zh") == "minimal"


def test_zh_broken_no_canonical_area() -> None:
    assert grade_sections(["致谢", "参考文献"], language="zh") == "broken"


# ---------------------------------------------------------------------------
# 切片 2: 语言路由
# ---------------------------------------------------------------------------


def test_zh_names_under_en_route_are_broken() -> None:
    """en 路由只查英文表——这就是基准对中文论文的真实行为(本课修复的动机)。"""
    names = ["引言", "方法", "实验", "结论"]
    assert grade_sections(names, language="en") == "broken"


def test_en_names_under_zh_route_are_broken() -> None:
    names = ["Introduction", "Method", "Experiments", "Conclusion"]
    assert grade_sections(names, language="zh") == "broken"


def test_none_route_unions_both_tables() -> None:
    # 中英混合节名在 None 路由下互补命中四区
    names = ["Introduction", "系统架构", "Experiments", "展望"]
    assert grade_sections(names) == "complete"


def test_language_is_keyword_only() -> None:
    import pytest

    with pytest.raises(TypeError):
        grade_sections(["Introduction"], "zh")  # type: ignore[misc]
