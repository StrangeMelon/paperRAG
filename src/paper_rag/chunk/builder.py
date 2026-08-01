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
- 参考文献类节的 chunk 打 metadata["is_references"]=True(基准无此标记), 块照常
  入库, 检索层将来可据此降权/过滤(2026-08-01 sanity 课确认)。
- layout 增强(2026-08-01 multimodal 课确认, 基准无): 读 parsed_dir/layout.json
  (MinerU content_list), 按图片 basename 配对——图块页码优先用自身 page_idx+1
  (标记回扫只作兜底, 修复纯图表页 ±1 误差); img_caption 注入图块嵌入文本
  (真实产物 alt 全空, 图注才是图的语义本体); 配对到 layout table 块的图片
  重定型为 modality="table" 并用 table_caption。layout 缺失/损坏/异形一律
  优雅降级回基准行为。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from ..utils.logger import get_logger
from . import multimodal_chunker as mm
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

# 参考文献类节名(splitter 清洗后的整名): 其 chunk 打 is_references 标记照常入库,
# 检索层将来可据此降权/过滤, 且不丢"本文引用了哪些工作"类问题的证据(2026-08-01 确认)
_REFERENCES_SECTION_NAMES = {"references", "bibliography", "参考文献"}


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


def _resolve_asset_path(parsed_dir: Path, rel_path: str | None) -> str | None:
    if not rel_path:
        return None
    p = Path(rel_path)
    if p.is_absolute():
        return str(p) if p.exists() else None
    candidate = (parsed_dir / p).resolve()
    return str(candidate) if candidate.exists() else None


def _load_layout_assets(parsed_dir: Path) -> dict[str, dict]:
    """按图片 basename 索引 layout.json 的 image/table 块: 页码、图注与真实类型。

    md 里图片路径是 figures/<hash>.jpg, layout 里是 images/<hash>.jpg,
    basename 同哈希可精确配对。缺失/损坏/非 content_list 形态一律返回空表,
    builder 降级回基准行为(页码靠标记回扫、语义靠 alt+上下文)。
    """
    layout_file = parsed_dir / "layout.json"
    if not layout_file.is_file():
        return {}
    try:
        blocks = json.loads(layout_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log.warning(f"layout.json unreadable, mm enrichment skipped: {layout_file}")
        return {}
    if not isinstance(blocks, list):
        return {}

    assets: dict[str, dict] = {}
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") not in ("image", "table"):
            continue
        name = Path(str(block.get("img_path") or "")).name
        if not name:
            continue
        try:
            page = int(block["page_idx"]) + 1
        except (KeyError, TypeError, ValueError):
            page = None
        caption_lines = block.get("img_caption" if block["type"] == "image" else "table_caption")
        if isinstance(caption_lines, str):
            caption_lines = [caption_lines]
        caption = " ".join(
            s.strip() for s in (caption_lines or []) if isinstance(s, str) and s.strip()
        )
        assets[name] = {"kind": block["type"], "page": page, "caption": caption}
    return assets


def build_chunks(paper_id: str, parsed_dir: Path, *, title: str) -> tuple[list[dict], list[dict]]:
    md_path = parsed_dir / "paper.md"
    md = md_path.read_text(encoding="utf-8")
    source_path = str(md_path.resolve())
    language = _read_language(parsed_dir)
    layout_assets = _load_layout_assets(parsed_dir)

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
        is_references = raw_sec.name.strip().lower() in _REFERENCES_SECTION_NAMES
        for i, tc in enumerate(chunk_text(raw_sec.body, language=language)):
            abs_start = body_base + tc.char_start
            abs_end = body_base + tc.char_end
            metadata: dict = {
                "section_level": raw_sec.level,
                "chunk_ordinal": i,
            }
            if is_references:
                metadata["is_references"] = True
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
                    "metadata": metadata,
                    "neighbors": [],
                }
            )

        # 多模态块: figure/table/formula(chunk_id 命名空间用抽取器 kind,
        # 即使图片被 layout 重定型为 table 也不与真正的管道表块撞 id)
        for kind, items in (
            ("figure", mm.extract_figures(raw_sec.body, language=language)),
            ("table", mm.extract_tables(raw_sec.body, language=language)),
            ("formula", mm.extract_formulas(raw_sec.body, language=language)),
        ):
            for j, mmc in enumerate(items):
                abs_start = body_base + mmc.char_start
                abs_end = body_base + mmc.char_end
                modality = mmc.modality
                text = mmc.text
                page = _page_for_offset(md, abs_start)

                asset_name = Path(mmc.asset_rel_path).name if mmc.asset_rel_path else ""
                asset = layout_assets.get(asset_name)
                if asset is not None:
                    if asset["page"] is not None:
                        page = asset["page"]  # 图块自身页码比标记回扫准(纯图表页无标记)
                    caption = " ".join(s for s in (mmc.alt, asset["caption"]) if s)
                    if asset["kind"] == "table":
                        modality = "table"
                        text = mm.compose_table_text(caption, mmc.context, language=language)
                    elif caption:
                        text = mm.compose_figure_text(
                            caption, mmc.context, mmc.asset_rel_path or "", language=language
                        )

                mm_metadata: dict = {
                    "section_level": raw_sec.level,
                    "chunk_ordinal": j,
                    "element_type": modality,
                }
                if is_references:
                    mm_metadata["is_references"] = True
                chunks.append(
                    {
                        "chunk_id": _chunk_id(paper_id, raw_sec.idx, kind, j),
                        "paper_id": paper_id,
                        "section_id": sec_id,
                        "section": raw_sec.name,
                        "section_idx": raw_sec.idx,
                        "modality": modality,
                        "page": page,
                        "text": text,
                        "context_text": with_context(
                            text, title=title, section=raw_sec.name, language=language
                        ),
                        "title": title,
                        "source_path": source_path,
                        "asset_rel_path": mmc.asset_rel_path,
                        "asset_path": _resolve_asset_path(parsed_dir, mmc.asset_rel_path),
                        "char_start": abs_start,
                        "char_end": abs_end,
                        "raw_snippet": mmc.raw,
                        "metadata": mm_metadata,
                        "neighbors": [],
                    }
                )

    try:
        from paper_rag.vision.enrich import enrich_chunks

        chunks = enrich_chunks(paper_id, chunks)
    except Exception as exc:
        log.warning(f"visual enrichment skipped for {paper_id}: {exc}")

    log.info(f"built {len(sections)} sections, {len(chunks)} chunks for {paper_id}")
    return sections, chunks
