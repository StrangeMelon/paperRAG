"""切块组装器: build_chunks 把解析产物组装为可入库的 sections 与 chunks。

输入 paper_id、解析目录与标题; 输出 (sections, chunks) 两组字典, 供 SQLite +
Qdrant 入库。与基准的差异(2026-08-01 已确认):
- 语言贯通: 从 parsed_dir/language.json 读 document_language(zh/en, 缺失或
  损坏降级 None, 不终止流程), 传给 split_sections / chunk_text / with_context;
  builder 是全链唯一的语言枢纽, 公开签名与基准一致(language 不是参数)。
- 偏移精确化: 用 md.index(body, sec.start) 求 body 真实起点做绝对偏移基准
  (基准的 sec.start + 相对偏移在节头有多余空行时整体漂移), 全局不变量
  md[char_start:char_end] == chunk["text"], 页码归属与溯源逐字节精确。
- 页码标记保留在 chunk 文本里(基准同款, sanity 课再评估清洗)。
- 多模态块(figure/table/formula)与 vision enrich 在 multimodal_chunker 课接入,
  当前只组装文本主路径。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from ..utils.logger import get_logger
from .contextual import with_context
from .section_splitter import split_sections
from .text_chunker import chunk_text

log = get_logger("chunk.builder")


def _chunk_id(paper_id: str, section_idx: int, kind: str, ord_: int) -> str:
    base = f"{paper_id}::{section_idx}::{kind}::{ord_}"
    return hashlib.sha1(base.encode()).hexdigest()[:20]


def _section_id(paper_id: str, idx: int) -> str:
    return hashlib.sha1(f"{paper_id}::sec::{idx}".encode()).hexdigest()[:16]


_PAGE_RE = re.compile(r"<!--\s*page\s+(\d+)\s*-->")


def _page_for_offset(text: str, offset: int) -> int | None:
    page = None
    for m in _PAGE_RE.finditer(text):
        if m.start() > offset:
            break
        page = int(m.group(1))
    return page


def _read_language(parsed_dir: Path) -> str | None:
    """读解析层的语言标注; 缺失、损坏或域外取值一律降级 None。"""
    language_file = parsed_dir / "language.json"
    if not language_file.is_file():
        return None
    try:
        value = json.loads(language_file.read_text(encoding="utf-8")).get("document_language")
    except (OSError, json.JSONDecodeError, AttributeError):
        log.warning(f"language.json unreadable, fallback to None: {language_file}")
        return None
    return value if value in ("zh", "en") else None


def build_chunks(paper_id: str, parsed_dir: Path, *, title: str) -> tuple[list[dict], list[dict]]:
    md_path = parsed_dir / "paper.md"
    md = md_path.read_text(encoding="utf-8")
    source_path = str(md_path.resolve())
    language = _read_language(parsed_dir)

    sections: list[dict] = []
    chunks: list[dict] = []

    for raw_sec in split_sections(md, language=language):
        sec_id = _section_id(paper_id, raw_sec.idx)
        sections.append(
            {
                "section_id": sec_id,
                "paper_id": paper_id,
                "idx": raw_sec.idx,
                "name": raw_sec.name,
            }
        )

        # body 是 strip 过的切片, 真实起点可能晚于 sec.start(节头空行)
        body_base = md.index(raw_sec.body, raw_sec.start) if raw_sec.body else raw_sec.start
        for i, tc in enumerate(chunk_text(raw_sec.body, language=language)):
            abs_start = body_base + tc.char_start
            abs_end = body_base + tc.char_end
            chunks.append(
                {
                    "chunk_id": _chunk_id(paper_id, raw_sec.idx, "text", i),
                    "paper_id": paper_id,
                    "section_id": sec_id,
                    "section": raw_sec.name,
                    "section_idx": raw_sec.idx,
                    "modality": "text",
                    "page": _page_for_offset(md, abs_start),
                    "text": tc.text,
                    "context_text": with_context(
                        tc.text, title=title, section=raw_sec.name, language=language
                    ),
                    "title": title,
                    "source_path": source_path,
                    "char_start": abs_start,
                    "char_end": abs_end,
                    "metadata": {
                        "section_level": raw_sec.level,
                        "chunk_ordinal": i,
                    },
                    "neighbors": [],
                }
            )

        # 多模态块(figure/table/formula)在 multimodal_chunker 课接入

    log.info(f"built {len(sections)} sections, {len(chunks)} chunks for {paper_id}")
    return sections, chunks
