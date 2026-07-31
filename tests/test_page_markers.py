"""页码标记注入的行为契约测试(方案 A, 2026-08-01 已确认)。

MinerU 路径的 paper.md 没有 `<!-- page N -->` 标记, 基准 builder 因此把 MinerU
论文所有 chunk 记为 page=None。重建版新增纯函数:

    inject_page_markers(md: str, layout: object) -> str

按 layout.json(MinerU content_list, 块含 type/text/page_idx, 0 基)在页码跳变处
用块文本前缀顺序对齐定位 md 偏移, 在所在行行首插入 `<!-- page N -->`
(N = page_idx + 1, 与 PyMuPDF 兜底一致的 1 基)。定位失败跳过该块、同页后续块
兜底; 整页不可定位则该页无标记(优雅降级, 不抛错)。锚定块最短长度双档:
含 CJK 的块 >= 2 字符, 纯 ASCII 块 >= 4 字符(页码 "1"/"12" 不做锚点)。

切片 0: 正常对齐、0 基转 1 基、行首插入(markdown 标题不被拆开)。
切片 1: 降级路径(定位失败跳过、同页兜底、短 ASCII 块不锚定、非 list 布局原样返回)。
切片 2: 内容保真(剥掉标记即还原原文)与 MinerU 标准化接入。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from paper_rag.parse.page_markers import inject_page_markers

# ---------------------------------------------------------------------------
# 切片 0: 正常对齐
# ---------------------------------------------------------------------------

MD = "# 引言\n\n第一页正文内容开头。\n\n第二页的内容从这里开始。\n"


def test_markers_injected_at_page_transitions_one_based() -> None:
    blocks = [
        {"type": "text", "text": "引言", "page_idx": 0},
        {"type": "text", "text": "第一页正文内容开头。", "page_idx": 0},
        {"type": "text", "text": "第二页的内容从这里开始。", "page_idx": 1},
    ]
    out = inject_page_markers(MD, blocks)
    # 0 基 page_idx 转 1 基页码; 标题的标记插在行首, "# 引言" 不被拆开
    assert out.startswith("<!-- page 1 -->\n\n# 引言\n")
    assert "<!-- page 2 -->\n\n第二页的内容从这里开始。" in out


def test_page_idx_jump_keeps_one_based_numbering() -> None:
    blocks = [
        {"type": "text", "text": "第一页正文内容开头。", "page_idx": 0},
        {"type": "text", "text": "第二页的内容从这里开始。", "page_idx": 4},
    ]
    out = inject_page_markers(MD, blocks)
    assert "<!-- page 5 -->\n\n第二页的内容从这里开始。" in out
    assert "page 2" not in out


def test_image_blocks_without_text_are_not_anchors() -> None:
    blocks = [
        {"type": "text", "text": "第一页正文内容开头。", "page_idx": 0},
        {"type": "image", "img_path": "figures/x.jpg", "page_idx": 1},
        {"type": "text", "text": "第二页的内容从这里开始。", "page_idx": 1},
    ]
    out = inject_page_markers(MD, blocks)
    assert "<!-- page 2 -->\n\n第二页的内容从这里开始。" in out


# ---------------------------------------------------------------------------
# 切片 1: 降级路径
# ---------------------------------------------------------------------------


def test_unlocatable_page_is_skipped_gracefully() -> None:
    blocks = [
        {"type": "text", "text": "第一页正文内容开头。", "page_idx": 0},
        {"type": "text", "text": "这段文本不在正文里", "page_idx": 1},
    ]
    out = inject_page_markers(MD, blocks)
    assert "<!-- page 1 -->" in out
    assert "page 2" not in out


def test_unlocatable_block_falls_back_to_next_block_same_page() -> None:
    blocks = [
        {"type": "text", "text": "第一页正文内容开头。", "page_idx": 0},
        {"type": "text", "text": "这段文本不在正文里", "page_idx": 1},
        {"type": "text", "text": "第二页的内容从这里开始。", "page_idx": 1},
    ]
    out = inject_page_markers(MD, blocks)
    assert "<!-- page 2 -->\n\n第二页的内容从这里开始。" in out


def test_short_ascii_block_is_not_an_anchor() -> None:
    md = "Version 1 of the method.\n\nSecond page content starts here.\n"
    blocks = [
        {"type": "text", "text": "Version 1 of the method.", "page_idx": 0},
        {"type": "text", "text": "1", "page_idx": 1},  # 页脚页码, 不可作锚点
    ]
    out = inject_page_markers(md, blocks)
    assert "page 2" not in out
    assert out.startswith("<!-- page 1 -->\n\n")


def test_cjk_two_char_block_is_a_valid_anchor() -> None:
    md = "# 摘要\n\n概述内容。\n\n# 结论\n\n收尾内容。\n"
    blocks = [
        {"type": "text", "text": "摘要", "page_idx": 0},
        {"type": "text", "text": "结论", "page_idx": 1},
    ]
    out = inject_page_markers(md, blocks)
    assert "<!-- page 2 -->\n\n# 结论\n" in out


def test_non_list_layout_returns_md_unchanged() -> None:
    assert inject_page_markers(MD, {"pdf_info": []}) == MD  # middle.json 形态
    assert inject_page_markers(MD, None) == MD
    assert inject_page_markers(MD, []) == MD


def test_malformed_blocks_are_skipped() -> None:
    blocks = [
        "not-a-dict",
        {"type": "text", "page_idx": 0},  # 无 text
        {"type": "text", "text": "第一页正文内容开头。"},  # 无 page_idx
        {"type": "text", "text": "第二页的内容从这里开始。", "page_idx": 1},
    ]
    out = inject_page_markers(MD, blocks)
    assert "<!-- page 2 -->\n\n第二页的内容从这里开始。" in out
    assert "page 1" not in out


# ---------------------------------------------------------------------------
# 切片 2: 内容保真与 MinerU 标准化接入
# ---------------------------------------------------------------------------


def test_stripping_markers_restores_original_md() -> None:
    blocks = [
        {"type": "text", "text": "引言", "page_idx": 0},
        {"type": "text", "text": "第二页的内容从这里开始。", "page_idx": 1},
    ]
    out = inject_page_markers(MD, blocks)
    assert re.sub(r"<!-- page \d+ -->\n\n", "", out) == MD


def test_normalize_into_injects_markers_from_content_list(tmp_path: Path) -> None:
    from paper_rag.parse.mineru_local import _normalize_into

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    source_md = raw_dir / "paper.md"
    source_md.write_text(MD, encoding="utf-8")
    blocks = [
        {"type": "text", "text": "引言", "page_idx": 0},
        {"type": "text", "text": "第二页的内容从这里开始。", "page_idx": 1},
    ]
    (raw_dir / "paper_content_list.json").write_text(
        json.dumps(blocks, ensure_ascii=False), encoding="utf-8"
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _normalize_into(out_dir, source_md, None)

    normalized = (out_dir / "paper.md").read_text(encoding="utf-8")
    assert normalized.startswith("<!-- page 1 -->\n\n# 引言\n")
    assert "<!-- page 2 -->\n\n第二页的内容从这里开始。" in normalized
    assert json.loads((out_dir / "layout.json").read_text(encoding="utf-8")) == blocks


def test_normalize_into_middle_json_shape_leaves_md_unmarked(tmp_path: Path) -> None:
    from paper_rag.parse.mineru_local import _normalize_into

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    source_md = raw_dir / "paper.md"
    source_md.write_text(MD, encoding="utf-8")
    (raw_dir / "paper_middle.json").write_text(json.dumps({"pdf_info": []}), encoding="utf-8")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _normalize_into(out_dir, source_md, None)

    normalized = (out_dir / "paper.md").read_text(encoding="utf-8")
    assert normalized == MD
    assert "page" not in normalized
