"""上下文前缀的行为契约测试。

切片 0: 默认英文模板(en/None 路由一致), 前缀在前、正文在后。
切片 1: zh 语言路由到 `chunk.context_prefix_zh` 中文模板(新增配置键)。
切片 2: 空 title/section 省略对应标签段, 都空时直接返回原文(基准会留 [Title: ] 死架子)。
切片 3: 值含花括号安全(str.format 只解释模板里的占位符)。

接口约定(切块层已确认方案, 2026-08-01):

    with_context(text: str, *, title: str, section: str,
                 language: str | None = None) -> str

`context_text` 是 BGE-M3 的嵌入输入; zh 模板让中文论文的嵌入输入保持单语。
"""

from __future__ import annotations

import importlib
from types import ModuleType

import paper_rag.config as config


def _mod() -> ModuleType:
    return importlib.import_module("paper_rag.chunk.contextual")


# ---------------------------------------------------------------------------
# 切片 0: 默认英文模板
# ---------------------------------------------------------------------------


def test_default_template_prefixes_title_and_section() -> None:
    out = _mod().with_context("Body text.", title="Self-RAG", section="Introduction")
    assert out == "[Title: Self-RAG] [Section: Introduction]\nBody text."


def test_language_none_and_en_share_default_template() -> None:
    mod = _mod()
    none_out = mod.with_context("Body.", title="T", section="S")
    en_out = mod.with_context("Body.", title="T", section="S", language="en")
    assert none_out == en_out == "[Title: T] [Section: S]\nBody."


# ---------------------------------------------------------------------------
# 切片 1: zh 语言路由
# ---------------------------------------------------------------------------


def test_zh_language_routes_to_chinese_template() -> None:
    out = _mod().with_context("区块链正文。", title="综合能源服务", section="摘要", language="zh")
    assert out == "[标题: 综合能源服务] [章节: 摘要]\n区块链正文。"


def test_zh_template_config_key_exists() -> None:
    conf = config.load()
    assert "{title}" in conf.chunk.context_prefix_zh
    assert "{section}" in conf.chunk.context_prefix_zh
    assert "标题" in conf.chunk.context_prefix_zh


# ---------------------------------------------------------------------------
# 切片 2: 空值省略标签段
# ---------------------------------------------------------------------------


def test_empty_title_drops_title_segment() -> None:
    out = _mod().with_context("Body.", title="", section="Methods")
    assert out == "[Section: Methods]\nBody."


def test_empty_section_drops_section_segment() -> None:
    out = _mod().with_context("Body.", title="Self-RAG", section="")
    assert out == "[Title: Self-RAG]\nBody."


def test_both_empty_returns_text_unchanged() -> None:
    assert _mod().with_context("Body.", title="", section="") == "Body."


def test_zh_empty_title_drops_segment_too() -> None:
    out = _mod().with_context("正文。", title="", section="结论", language="zh")
    assert out == "[章节: 结论]\n正文。"


# ---------------------------------------------------------------------------
# 切片 3: 花括号安全
# ---------------------------------------------------------------------------


def test_braces_in_values_are_literal() -> None:
    out = _mod().with_context("Body.", title="A {b} c", section="S {d}")
    assert out == "[Title: A {b} c] [Section: S {d}]\nBody."
