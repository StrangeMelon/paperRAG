"""builder → vision 的交接契约: metadata 正门补料 + 语言贯通。

不触网: 用 monkeypatch 换掉 enrich_chunks, 只证 builder 交出的原料与语言正确。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_rag.chunk import builder


def _write_parsed(tmp_path: Path, *, language: str, md: str, layout: list) -> Path:
    d = tmp_path / "parsed"
    (d / "images").mkdir(parents=True)
    (d / "images" / "fig1.jpg").write_bytes(b"\x89PNG-FIG")
    (d / "paper.md").write_text(md, encoding="utf-8")
    (d / "layout.json").write_text(json.dumps(layout, ensure_ascii=False), encoding="utf-8")
    (d / "language.json").write_text(json.dumps({"document_language": language}), encoding="utf-8")
    return d


_ZH_MD = """# 摘要

本文提出一种新方法。

# 实验

我们在多个数据集上评测了模型性能, 结果见下图。

![](images/fig1.jpg)

后续讨论了误差来源。
"""

_ZH_LAYOUT = [
    {
        "type": "image",
        "page_idx": 3,
        "img_path": "images/fig1.jpg",
        "img_caption": ["图 2 不同模型的准确率对比"],
    }
]


@pytest.fixture
def captured(monkeypatch):
    """拦截 enrich_chunks, 记录 builder 实际传入的参数。"""
    seen: dict = {}

    def fake_enrich(paper_id, chunks, **kwargs):
        seen["paper_id"] = paper_id
        seen["kwargs"] = kwargs
        seen["chunks"] = chunks
        return chunks

    import paper_rag.vision.enrich as en

    monkeypatch.setattr(en, "enrich_chunks", fake_enrich)
    return seen


def test_language_is_passed_to_enrich(tmp_path, captured):
    parsed = _write_parsed(tmp_path, language="zh", md=_ZH_MD, layout=_ZH_LAYOUT)
    builder.build_chunks("p1", parsed, title="中文论文")
    assert captured["kwargs"].get("language") == "zh"


def test_english_language_is_passed_to_enrich(tmp_path, captured):
    parsed = _write_parsed(tmp_path, language="en", md=_ZH_MD, layout=_ZH_LAYOUT)
    builder.build_chunks("p1", parsed, title="Paper")
    assert captured["kwargs"].get("language") == "en"


def test_figure_metadata_carries_caption_and_context(tmp_path, captured):
    parsed = _write_parsed(tmp_path, language="zh", md=_ZH_MD, layout=_ZH_LAYOUT)
    _, chunks = builder.build_chunks("p1", parsed, title="中文论文")
    figs = [c for c in chunks if c["modality"] == "figure"]
    assert figs, "should produce a figure chunk"
    meta = figs[0]["metadata"]
    # 正门原料: layout 图注不再依赖文本反解
    assert "准确率对比" in meta["caption"]
    assert meta["surrounding_context"]
    assert "数据集" in meta["surrounding_context"] or "误差" in meta["surrounding_context"]


def test_text_chunks_have_no_vision_metadata(tmp_path, captured):
    parsed = _write_parsed(tmp_path, language="zh", md=_ZH_MD, layout=_ZH_LAYOUT)
    _, chunks = builder.build_chunks("p1", parsed, title="中文论文")
    for c in chunks:
        if c["modality"] == "text":
            assert "caption" not in c["metadata"]
            assert "surrounding_context" not in c["metadata"]


def test_caption_absent_layout_still_sets_metadata_keys(tmp_path, captured):
    # layout 缺失时 caption 可为空, 但键必须存在, 让 vision 侧走文本反解兜底。
    parsed = _write_parsed(tmp_path, language="zh", md=_ZH_MD, layout=[])
    _, chunks = builder.build_chunks("p1", parsed, title="中文论文")
    figs = [c for c in chunks if c["modality"] == "figure"]
    assert figs
    assert "caption" in figs[0]["metadata"]
    assert "surrounding_context" in figs[0]["metadata"]


def test_enrich_failure_does_not_break_build(tmp_path, monkeypatch):
    import paper_rag.vision.enrich as en

    def boom(paper_id, chunks, **kwargs):
        raise RuntimeError("vision exploded")

    monkeypatch.setattr(en, "enrich_chunks", boom)
    parsed = _write_parsed(tmp_path, language="zh", md=_ZH_MD, layout=_ZH_LAYOUT)
    sections, chunks = builder.build_chunks("p1", parsed, title="中文论文")
    assert sections and chunks  # fail-open: 整篇照常入库
