"""PyMuPDF 兜底解析器的真实 PDF 行为测试。"""

from __future__ import annotations

import importlib  # 根据字符串动态导入模块
from pathlib import Path  # 面向对象的处理文件路径
from types import ModuleType  # 表示 "一个Python 模块对象" 的类型

import pytest

from paper_rag.utils.ids import to_safe_dirname


def _parser_module() -> ModuleType:
    try:
        return importlib.import_module("paper_rag.parse.fallback_pymupdf")  # 相当于动态的执行 import paper_rag.parse.fallback_pymupdf
    except ModuleNotFoundError as exc:
        if exc.name != "paper_rag.parse.fallback_pymupdf":  # fallback_pymupdf 存在, 但它内部导入的其他模块不存在, 直接原样抛出, 暴露真正的错误
            raise
        pytest.fail(    # fallback_pymupdf 本身不存在, 转换成容易理解的 pytest 失败提示
            "尚未实现 paper_rag.parse.fallback_pymupdf.parse_pdf",
            pytrace=False,
        )

# 延迟加载可选的依赖
def _fitz_module() -> ModuleType:
    try:
        import fitz
    except ImportError:
        pytest.fail(
            "PyMuPDF 未安装; 请执行: uv sync --extra dev --extra ingest",
            pytrace=False,
        )
    return fitz # 这里的 fitz 是模块对象, 不是类或 PDF 文档实例


# 创建真实的测试样本, 创建3页PDF, 前两页有内容, 最后一页空
def _create_three_page_pdf(fitz: ModuleType, pdf_path: Path) -> None:
    document = fitz.open()
    try:
        first_page = document.new_page()
        first_page.insert_text(
            (72, 72),
            "First page: retrieval augmented generation.",
            fontsize=12,
        )

        second_page = document.new_page()
        second_page.insert_text(
            (72, 72),
            "Second page: dense and sparse retrieval.",
            fontsize=12,
        )

        document.new_page() # 第三页空白. 这是一个很重要的测试设计: 解析器不能因为页面为空就把这一页丢掉, 仍然必须产生 <!-- page 3 -->
        document.save(str(pdf_path))
    finally:
        document.close()


def test_parse_pdf_writes_page_marked_markdown_idempotently(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fitz = _fitz_module()
    module = _parser_module()
    source_pdf = tmp_path / "source.pdf"    # 测试的pdf路径
    _create_three_page_pdf(fitz, source_pdf)    # 生成一份测试 PDF, 3页, 前两页有内容, 第三页空白

    parsed_root = tmp_path / "parsed"   # 解析结果的根目录. 在项目中实际在./data/parsed中
    paper_id = "arxiv:2310.12345"   # 生成的测试论文ID, 仅用于测试, 不会实际访问arxiv.org
    expected_dir = parsed_root / to_safe_dirname(paper_id)  # 期望的解析结果目录, 在项目中实际在./data/parsed/arxiv_2310.12345中
    monkeypatch.setattr(
        module,
        "parsed_dir",
        lambda received_id: parsed_root / to_safe_dirname(received_id),
    )

    first_result = module.parse_pdf(paper_id, source_pdf)   # 这里把 Path 形式的 PDF 路径传给解析器

    markdown_path = expected_dir / "paper.md"
    markdown = markdown_path.read_text(encoding="utf-8")

    assert first_result == expected_dir
    assert sorted(path.name for path in expected_dir.iterdir()) == [
        "paper.md"
    ]
    assert markdown.count("<!-- page ") == 3
    assert "<!-- page 1 -->" in markdown
    assert "<!-- page 2 -->" in markdown
    assert "<!-- page 3 -->" in markdown
    assert "First page: retrieval augmented generation." in markdown
    assert "Second page: dense and sparse retrieval." in markdown
    assert markdown.index("<!-- page 1 -->") < markdown.index(  # 页码标记要出现在该页的正文之前
        "First page: retrieval augmented generation."
    )
    assert markdown.index("<!-- page 2 -->") < markdown.index(
        "Second page: dense and sparse retrieval."
    )
    assert markdown.index("<!-- page 1 -->") < markdown.index(
        "<!-- page 2 -->"
    ) < markdown.index("<!-- page 3 -->")

    second_result = module.parse_pdf(paper_id, str(source_pdf)) # 这里把 str 形式的 PDF 路径传给解析器

    assert second_result == first_result
    assert markdown_path.read_text(encoding="utf-8") == markdown
