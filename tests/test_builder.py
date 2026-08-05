"""切块组装器 build_chunks 的行为契约测试。

切片 0: 基本组装(sections/chunks 字段、确定性 ID、schema 与基准一致)。
切片 1: 语言贯通(language.json 读取 zh/en/缺失/损坏, 传给 splitter/chunker/contextual)。
切片 2: 页码归属(<!-- page N --> 回扫、无标记 page=None、多块跨页)。
切片 3: 偏移精确化(全局不变量 md[abs_start:abs_end] == chunk.text,
        含基准会漂移的"节头多空行"场景)。
切片 4: 参考文献打标(References/参考文献 节的块 metadata["is_references"]=True,
        普通块不带该键, 与基准 schema 逐键一致)。
切片 5: 多模态块与 layout 增强(figure/table/formula 组装、raw_snippet 回切
        不变量、asset_path 解析、图块自身页码、图注注入与表重定型、降级)。

接口约定(与基准一致, 2026-08-01 确认):

    build_chunks(paper_id: str, parsed_dir: Path, *, title: str)
        -> tuple[list[dict], list[dict]]

`language` 不是参数: builder 从 parsed_dir/language.json 读 document_language
(zh/en, 缺失或损坏 -> None), 是把语言贯通到全链的唯一枢纽。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import paper_rag.config as config
from paper_rag.chunk.builder import build_chunks


@pytest.fixture(autouse=True)
def _no_real_vision(monkeypatch):
    """钉死 vision 钩子为恒等: 本文件只测 builder 本身。

    生产配置 vision.enabled=true 且凭据在 .env 时, 缺省构造会让纯逻辑测试
    真打视觉 API(2026-08-05 实翻车); 钩子契约由 test_builder_vision_handoff 专测。
    """
    import paper_rag.vision.enrich as en

    monkeypatch.setattr(en, "enrich_chunks", lambda paper_id, chunks, **kw: chunks)


def _write_parsed(tmp_path: Path, md: str, language: str | None = None) -> Path:
    (tmp_path / "paper.md").write_text(md, encoding="utf-8")
    if language is not None:
        (tmp_path / "language.json").write_text(
            json.dumps({"document_language": language}), encoding="utf-8"
        )
    return tmp_path


def _patch_config(monkeypatch, *, target_tokens: int, overlap_tokens: int = 0) -> None:
    conf = config.load()
    conf.chunk.text.target_tokens = target_tokens
    conf.chunk.text.overlap_tokens = overlap_tokens
    monkeypatch.setattr(config, "load", lambda path=None: conf)


EN_MD = (
    "# Introduction\n\nIntro paragraph body text.\n\n## Methods\n\nMethods paragraph body text.\n"
)


# ---------------------------------------------------------------------------
# 切片 0: 基本组装
# ---------------------------------------------------------------------------


def test_sections_and_chunks_schema(tmp_path: Path) -> None:
    parsed = _write_parsed(tmp_path, EN_MD)
    sections, chunks = build_chunks("p1", parsed, title="Demo Paper")

    assert [s["name"] for s in sections] == ["Introduction", "Methods"]
    assert [s["idx"] for s in sections] == [0, 1]
    for s in sections:
        assert s["paper_id"] == "p1"
        expected = hashlib.sha1(f"p1::sec::{s['idx']}".encode()).hexdigest()[:16]
        assert s["section_id"] == expected

    assert len(chunks) == 2
    intro = chunks[0]
    assert intro["chunk_id"] == hashlib.sha1(b"p1::0::text::0").hexdigest()[:20]
    assert intro["section_id"] == sections[0]["section_id"]
    assert intro["section"] == "Introduction"
    assert intro["section_idx"] == 0
    assert intro["modality"] == "text"
    assert intro["text"] == "Intro paragraph body text."
    assert intro["context_text"].startswith("[Title: Demo Paper] [Section: Introduction]\n")
    assert intro["title"] == "Demo Paper"
    assert intro["source_path"] == str((parsed / "paper.md").resolve())
    assert intro["metadata"] == {"section_level": 1, "chunk_ordinal": 0}
    assert intro["neighbors"] == []
    assert intro["page"] is None  # 无标记的 md


# ---------------------------------------------------------------------------
# 切片 1: 语言贯通
# ---------------------------------------------------------------------------

ZH_MD = "# 引言\n\n中文正文第一段内容。\n\n# 结论\n\n中文收尾段落内容。\n"


def test_zh_language_flows_to_context_text(tmp_path: Path) -> None:
    parsed = _write_parsed(tmp_path, ZH_MD, language="zh")
    _, chunks = build_chunks("p1", parsed, title="中文论文")

    assert chunks, "中文产物切出空结果"
    for c in chunks:
        assert c["context_text"].startswith("[标题: 中文论文] [章节: ")


def test_missing_language_json_defaults_to_english_template(tmp_path: Path) -> None:
    parsed = _write_parsed(tmp_path, EN_MD)  # 不写 language.json
    _, chunks = build_chunks("p1", parsed, title="T")
    assert chunks[0]["context_text"].startswith("[Title: T] ")


def test_corrupt_language_json_degrades_to_none(tmp_path: Path) -> None:
    parsed = _write_parsed(tmp_path, EN_MD)
    (parsed / "language.json").write_text("{not json", encoding="utf-8")
    sections, chunks = build_chunks("p1", parsed, title="T")  # 不抛错
    assert sections and chunks
    assert chunks[0]["context_text"].startswith("[Title: T] ")


def test_unknown_language_value_treated_as_none(tmp_path: Path) -> None:
    parsed = _write_parsed(tmp_path, EN_MD, language="fr")
    _, chunks = build_chunks("p1", parsed, title="T")
    assert chunks[0]["context_text"].startswith("[Title: T] ")


# ---------------------------------------------------------------------------
# 切片 2: 页码归属
# ---------------------------------------------------------------------------

PAGED_MD = (
    "<!-- page 1 -->\n"
    "\n"
    "# Introduction\n"
    "\n"
    "First page paragraph text.\n"
    "\n"
    "<!-- page 2 -->\n"
    "\n"
    "Second page paragraph text.\n"
    "\n"
    "## Methods\n"
    "\n"
    "Third paragraph on page two.\n"
)


def test_page_attribution_follows_markers(tmp_path: Path, monkeypatch) -> None:
    _patch_config(monkeypatch, target_tokens=8)  # 极小目标逼出多块
    parsed = _write_parsed(tmp_path, PAGED_MD)
    _, chunks = build_chunks("p1", parsed, title="T")

    intro_chunks = [c for c in chunks if c["section"] == "Introduction"]
    methods_chunks = [c for c in chunks if c["section"] == "Methods"]
    # 标记被空行包围自成段落, 极小 target 下独立成块(真实 500 目标会并入邻块):
    # [第一页段落, <!-- page 2 --> 标记块, 第二页段落]
    assert [c["page"] for c in intro_chunks] == [1, 2, 2]
    assert intro_chunks[0]["text"] == "First page paragraph text."
    assert intro_chunks[-1]["text"] == "Second page paragraph text."
    assert methods_chunks[0]["page"] == 2


# ---------------------------------------------------------------------------
# 切片 3: 偏移精确化
# ---------------------------------------------------------------------------


def test_global_offset_invariant_md_slice_equals_text(tmp_path: Path, monkeypatch) -> None:
    _patch_config(monkeypatch, target_tokens=8)
    parsed = _write_parsed(tmp_path, PAGED_MD)
    md = PAGED_MD
    _, chunks = build_chunks("p1", parsed, title="T")

    assert chunks
    for c in chunks:
        assert md[c["char_start"] : c["char_end"]] == c["text"], f"偏移漂移: {c['text'][:30]!r}"


def test_offset_invariant_survives_extra_blank_lines(tmp_path: Path) -> None:
    """节头多空行时基准的 sec.start + 相对偏移会漂移; 重建版用 body 真实起点。"""
    md = "# Introduction\n\n\n\n\nIntro paragraph body text.\n"
    parsed = _write_parsed(tmp_path, md)
    _, chunks = build_chunks("p1", parsed, title="T")

    assert len(chunks) == 1
    c = chunks[0]
    assert md[c["char_start"] : c["char_end"]] == c["text"] == "Intro paragraph body text."


# ---------------------------------------------------------------------------
# 切片 4: 参考文献打标(2026-08-01 sanity 课确认: 保留块 + metadata 标记)
# ---------------------------------------------------------------------------


def test_references_section_chunks_are_flagged(tmp_path: Path) -> None:
    md = (
        "# Introduction\n\nIntro paragraph body text.\n\n"
        "# References\n\n[1] Some Author. Some paper title. 2024.\n"
    )
    parsed = _write_parsed(tmp_path, md)
    _, chunks = build_chunks("p1", parsed, title="T")

    refs = [c for c in chunks if c["section"] == "References"]
    others = [c for c in chunks if c["section"] != "References"]
    assert refs and others
    for c in refs:
        assert c["metadata"]["is_references"] is True
    for c in others:
        assert "is_references" not in c["metadata"]  # 普通块 schema 与基准逐键一致


def test_zh_references_section_chunks_are_flagged(tmp_path: Path) -> None:
    md = "# 引言\n\n中文正文第一段内容。\n\n# 参考文献\n\n[1] 作者. 论文标题. 2024.\n"
    parsed = _write_parsed(tmp_path, md, language="zh")
    _, chunks = build_chunks("p1", parsed, title="中文论文")

    refs = [c for c in chunks if c["section"] == "参考文献"]
    assert refs
    assert all(c["metadata"]["is_references"] is True for c in refs)


# ---------------------------------------------------------------------------
# 切片 5: 多模态块与 layout 增强(2026-08-01 multimodal 课确认)
# ---------------------------------------------------------------------------

MM_MD = (
    "# Results\n"
    "\n"
    "Intro sentence for results.\n"
    "\n"
    "![Arch overview](figures/h1.jpg)\n"
    "\n"
    "| Method | Score |\n"
    "| A | 1 |\n"
    "\n"
    "$$y = kx$$\n"
)


def _write_layout(parsed: Path, blocks: list) -> None:
    (parsed / "layout.json").write_text(json.dumps(blocks, ensure_ascii=False), encoding="utf-8")


def test_mm_chunks_schema_and_global_invariant(tmp_path: Path) -> None:
    parsed = _write_parsed(tmp_path, MM_MD)
    _, chunks = build_chunks("p1", parsed, title="T")

    mm_chunks = [c for c in chunks if c["modality"] != "text"]
    assert sorted(c["modality"] for c in mm_chunks) == ["figure", "formula", "table"]
    for c in mm_chunks:
        assert MM_MD[c["char_start"] : c["char_end"]] == c["raw_snippet"], (
            f"raw_snippet 不可回切: {c['modality']}"
        )
        assert c["metadata"]["element_type"] == c["modality"]
        assert c["context_text"].startswith("[Title: T] [Section: Results]")

    fig = next(c for c in mm_chunks if c["modality"] == "figure")
    assert fig["chunk_id"] == hashlib.sha1(b"p1::0::figure::0").hexdigest()[:20]
    assert fig["asset_rel_path"] == "figures/h1.jpg"
    assert fig["asset_path"] is None  # 文件不存在
    assert fig["page"] is None  # 无标记也无 layout
    assert fig["text"].startswith("Figure: Arch overview\nContext: ")


def test_mm_asset_path_resolves_when_file_exists(tmp_path: Path) -> None:
    parsed = _write_parsed(tmp_path, "# S\n\n![](figures/h1.jpg)\n")
    (parsed / "figures").mkdir()
    (parsed / "figures/h1.jpg").write_bytes(b"jpg")
    _, chunks = build_chunks("p1", parsed, title="T")

    fig = next(c for c in chunks if c["modality"] == "figure")
    assert fig["asset_path"] == str((parsed / "figures/h1.jpg").resolve())


def test_layout_gives_figure_page_and_caption(tmp_path: Path) -> None:
    """图块页码用自身 page_idx+1(md 无标记也有页), 图注补上空 alt 的语义。"""
    parsed = _write_parsed(tmp_path, "# 架构\n\n![](figures/h1.jpg)\n", language="zh")
    _write_layout(
        parsed,
        [
            {
                "type": "image",
                "img_path": "images/h1.jpg",
                "page_idx": 4,
                "img_caption": ["图1 测试架构"],
            }
        ],
    )
    _, chunks = build_chunks("p1", parsed, title="中文论文")

    fig = next(c for c in chunks if c["modality"] == "figure")
    assert fig["page"] == 5
    assert fig["text"].startswith("图: 图1 测试架构\n上下文: ")


def test_layout_retypes_table_image(tmp_path: Path) -> None:
    """配对到 layout table 块的图片重定型为 table, id 命名空间保持抽取器 kind。"""
    parsed = _write_parsed(tmp_path, "# 实验\n\n![](figures/h2.jpg)\n", language="zh")
    _write_layout(
        parsed,
        [
            {
                "type": "table",
                "img_path": "images/h2.jpg",
                "page_idx": 8,
                "table_caption": ["表1 对比数据"],
            }
        ],
    )
    _, chunks = build_chunks("p1", parsed, title="中文论文")

    tab = next(c for c in chunks if c["modality"] == "table")
    assert tab["page"] == 9
    assert tab["text"].startswith("表:\n表1 对比数据\n上下文: ")
    assert tab["metadata"]["element_type"] == "table"
    assert tab["chunk_id"] == hashlib.sha1(b"p1::0::figure::0").hexdigest()[:20]


def test_corrupt_layout_degrades_to_baseline(tmp_path: Path) -> None:
    parsed = _write_parsed(tmp_path, "# S\n\n![alt text](figures/h1.jpg)\n")
    (parsed / "layout.json").write_text("{broken", encoding="utf-8")
    _, chunks = build_chunks("p1", parsed, title="T")  # 不抛错

    fig = next(c for c in chunks if c["modality"] == "figure")
    assert fig["page"] is None
    assert fig["text"].startswith("Figure: alt text\nContext: ")
