"""多模态抽取器: 从章节正文抽出图/表/公式三类特殊 chunk。

与基准一致的识别方式(markdown 正则, MinerU 与 PyMuPDF 产物通用, 后者召回
偏低——真实测量为零, 见 demo): 图片 `![alt](path)`、连续管道表行、`$$...$$`
展示公式。与基准的差异(2026-08-01 已确认):
- 嵌入文本前缀按语言路由: zh 用 图:/表:/公式:/上下文:/路径:, en/None 用基准
  英文(Figure:/Table:/Formula:/Context:/Path:); 三个 compose_* 助手公开导出,
  builder 拿 layout.json 图注重组文本时复用同一模板。
- 偏移精确化: 表格块的 span 与 strip 后的 raw 对齐(基准 span 含尾随空白,
  raw 不可回切); 不变量 body[char_start:char_end] == raw 对三类块都成立。
- MMChunk 额外携带 alt 与 context 原料字段, 供 builder 在拿到更好的语义源
  (layout 图注)时重组嵌入文本。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_FIGURE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)")
_TABLE_BLOCK_RE = re.compile(r"((?:^\|.*\|\s*$\n?)+)", re.MULTILINE)
_FORMULA_BLOCK_RE = re.compile(r"\$\$(?P<body>.+?)\$\$", re.DOTALL)

_LABELS_EN = {
    "figure": "Figure",
    "table": "Table",
    "formula": "Formula",
    "context": "Context",
    "path": "Path",
}
_LABELS_ZH = {"figure": "图", "table": "表", "formula": "公式", "context": "上下文", "path": "路径"}


def _labels(language: str | None) -> dict[str, str]:
    return _LABELS_ZH if language == "zh" else _LABELS_EN


@dataclass
class MMChunk:
    text: str  # 进嵌入的描述文本
    modality: str  # figure | table | formula
    raw: str  # 原始 markdown 片段, body[char_start:char_end] == raw
    char_start: int
    char_end: int
    asset_rel_path: str | None = None
    alt: str = ""  # 图片 alt 原料(真实 MinerU 产物几乎全空)
    context: str = ""  # 前后 ±240 字符上下文原料


def compose_figure_text(
    caption: str, context: str, path: str, *, language: str | None = None
) -> str:
    lb = _labels(language)
    return f"{lb['figure']}: {caption}\n{lb['context']}: {context}\n{lb['path']}: {path}"


def compose_table_text(content: str, context: str, *, language: str | None = None) -> str:
    lb = _labels(language)
    return f"{lb['table']}:\n{content}\n{lb['context']}: {context}"


def compose_formula_text(latex: str, context: str, *, language: str | None = None) -> str:
    lb = _labels(language)
    return f"{lb['formula']}: {latex}\n{lb['context']}: {context}"


def extract_figures(body: str, *, language: str | None = None) -> list[MMChunk]:
    out: list[MMChunk] = []
    for m in _FIGURE_RE.finditer(body):
        alt = m.group("alt").strip()
        path = m.group("path").strip()
        context = _surrounding_text(body, m.start(), m.end())
        out.append(
            MMChunk(
                text=compose_figure_text(alt, context, path, language=language),
                modality="figure",
                raw=m.group(0),
                char_start=m.start(),
                char_end=m.end(),
                asset_rel_path=path,
                alt=alt,
                context=context,
            )
        )
    return out


def extract_tables(body: str, *, language: str | None = None) -> list[MMChunk]:
    out: list[MMChunk] = []
    for m in _TABLE_BLOCK_RE.finditer(body):
        block_raw = m.group(1)
        block = block_raw.strip()
        if not _looks_like_table(block):
            continue
        # span 与 strip 后的 raw 对齐(基准 span 含尾随换行, raw 不可回切)
        start = m.start(1) + (len(block_raw) - len(block_raw.lstrip()))
        end = start + len(block)
        context = _surrounding_text(body, m.start(), m.end())
        out.append(
            MMChunk(
                text=compose_table_text(block, context, language=language),
                modality="table",
                raw=block,
                char_start=start,
                char_end=end,
                context=context,
            )
        )
    return out


def _looks_like_table(block: str) -> bool:
    """拦掉公式残片, 如两行重复的单单元格 ``|Q|``。"""
    rows = [line.strip() for line in block.splitlines() if line.strip()]
    valid_rows = [
        row
        for row in rows
        if row.startswith("|") and row.endswith("|") and len(_table_cells(row)) >= 2
    ]
    return len(valid_rows) >= 2


def _table_cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip("|").split("|")]


def extract_formulas(body: str, *, language: str | None = None) -> list[MMChunk]:
    out: list[MMChunk] = []
    for m in _FORMULA_BLOCK_RE.finditer(body):
        latex = m.group("body").strip()
        context = _surrounding_text(body, m.start(), m.end())
        out.append(
            MMChunk(
                text=compose_formula_text(latex, context, language=language),
                modality="formula",
                raw=m.group(0),
                char_start=m.start(),
                char_end=m.end(),
                context=context,
            )
        )
    return out


def _surrounding_text(body: str, start: int, end: int, span: int = 240) -> str:
    left = max(0, start - span)
    right = min(len(body), end + span)
    snippet = body[left:start] + " " + body[end:right]
    return re.sub(r"\s+", " ", snippet).strip()
