"""页码标记注入: 给 MinerU 路径的 paper.md 补上 `<!-- page N -->` 标记。

PyMuPDF 兜底解析逐页产出、天然带标记; MinerU 的 markdown 没有页概念, 基准
builder 只靠标记归属页码, 导致 MinerU 论文所有 chunk `page=None`。方案 A
(2026-08-01 已确认): 按 layout.json(MinerU content_list, 块含 type/text/
page_idx, 0 基)在页码跳变处用块文本前缀顺序对齐定位 md 偏移, 在所在行行首
插入 `<!-- page N -->`(N = page_idx + 1, 与 PyMuPDF 一致的 1 基)。

降级规则(不抛错):
- 布局不是 list(middle.json 形态)或块缺 text/page_idx: 跳过, md 原样返回;
- 块文本在 md 中定位失败: 跳过该块, 同页后续块兜底; 整页无法定位则该页无标记;
- 锚定块最短长度双档: 含 CJK 的块 >= 2 字符(中文短标题 "引言/摘要" 可锚定),
  纯 ASCII 块 >= 4 字符(页脚页码 "1"/"12" 不做锚点, 避免假匹配拖走游标)。
"""

from __future__ import annotations

# 与 chunk 层回退计数同源的 CJK 判定区间(部首到统一表意、兼容表意、全角形式)
_CJK_RANGES = ((0x2E80, 0x9FFF), (0xF900, 0xFAFF), (0xFF00, 0xFFEF))

_PREFIX_LEN = 20
_MIN_ANCHOR_CJK = 2
_MIN_ANCHOR_ASCII = 4


def _contains_cjk(s: str) -> bool:
    return any(lo <= ord(ch) <= hi for ch in s for lo, hi in _CJK_RANGES)


def inject_page_markers(md: str, layout: object) -> str:
    """按 content_list 布局给 md 注入 1 基页码标记; 无法处理时原样返回。"""
    if not isinstance(layout, list) or not md:
        return md

    insertions: list[tuple[int, int]] = []  # (行首偏移, 1 基页码)
    seen_pages: set[int] = set()
    cursor = 0
    for block in layout:
        if not isinstance(block, dict):
            continue
        page_idx = block.get("page_idx")
        text = block.get("text")
        if not isinstance(page_idx, int) or page_idx < 0 or not isinstance(text, str):
            continue
        prefix = text.strip()[:_PREFIX_LEN]
        min_len = _MIN_ANCHOR_CJK if _contains_cjk(prefix) else _MIN_ANCHOR_ASCII
        if len(prefix) < min_len:
            continue
        pos = md.find(prefix, cursor)
        if pos == -1:
            continue  # 定位失败: 不动游标, 同页后续块仍有机会兜底
        cursor = pos + len(prefix)
        if page_idx not in seen_pages:
            seen_pages.add(page_idx)
            line_start = md.rfind("\n", 0, pos) + 1
            insertions.append((line_start, page_idx + 1))

    out = md
    for offset, page in sorted(insertions, reverse=True):
        out = f"{out[:offset]}<!-- page {page} -->\n\n{out[offset:]}"
    return out
