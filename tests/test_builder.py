"""切块组装器 build_chunks 的行为契约测试。

切片 0: 基本组装(sections/chunks 字段、确定性 ID、schema 与基准一致)。
切片 1: 语言贯通(language.json 读取 zh/en/缺失/损坏, 传给 splitter/chunker/contextual)。
切片 2: 页码归属(<!-- page N --> 回扫、无标记 page=None、多块跨页)。
切片 3: 偏移精确化(全局不变量 md[abs_start:abs_end] == chunk.text,
        含基准会漂移的"节头多空行"场景)。

接口约定(与基准一致, 2026-08-01 确认):

    build_chunks(paper_id: str, parsed_dir: Path, *, title: str)
        -> tuple[list[dict], list[dict]]

`language` 不是参数: builder 从 parsed_dir/language.json 读 document_language
(zh/en, 缺失或损坏 -> None), 是把语言贯通到全链的唯一枢纽。多模态块
(figure/table/formula)在 multimodal_chunker 课接入, 本课只交文本主路径。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import paper_rag.config as config
from paper_rag.chunk.builder import build_chunks


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
