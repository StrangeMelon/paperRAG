"""编排层: 一篇已入库论文 -> 概念抽取 -> 三级解析 -> 词条建/补 -> 镜像。

由 scripts/wiki_worker.py 消费 wiki_jobs 时调用(语言随任务显式传入);
force=True 供 backfill 绕过 kill switch 使用。

关键决策:
- 质量门槛在此兜底: parsed_with 命中黑名单(mineru+broken 等)或文本块数
  低于 min_chunks 的文档直接 skipped 带原因——正式 PDF 目录混有非论文文件
  是已确认事实, 垃圾词条会污染 QA 消费端;
- match 分支下, 即使 flow.patch_entry 被 self_eval 拦下, 论文/证据/双语标签
  的机械关联仍直接落库(add_key_papers/add_evidence/add_labels)——
  "24h 锁与门槛只限制昂贵的内容改写, 不限制关系记录" 的另一半保障;
- Qdrant 镜像失败非致命: 脏标存续, worker 的补偿轮会重试;
- 单概念异常不中断整篇: 计入 dropped, 报告如实返回。
"""

from __future__ import annotations

from typing import Any

from .. import config as cfg
from ..utils.logger import get_logger
from . import concept_extractor, flow, normalize
from . import store as wstore
from .schema import WikiLabel

log = get_logger(__name__)


def _load_paper(paper_id: str) -> dict[str, Any] | None:
    from sqlmodel import Session

    from ..store.sqlite_store import Paper, get_engine

    with Session(get_engine()) as s:
        p = s.get(Paper, paper_id)
        if p is None:
            return None
        return {"paper_id": p.paper_id, "title": p.title, "parsed_with": p.parsed_with}


def _load_text_chunks(paper_id: str) -> list[dict[str, Any]]:
    from sqlmodel import Session, select

    from ..store.sqlite_store import Chunk, get_engine

    with Session(get_engine()) as s:
        rows = list(s.exec(select(Chunk).where(Chunk.paper_id == paper_id)))
    return [
        {"chunk_id": r.chunk_id, "section": r.section, "text": r.text}
        for r in rows
        if r.modality == "text"
    ]


def _mirror(entry) -> None:
    from ..embed import bge_m3

    vec = bge_m3.encode_one(f"{entry.name}\n{entry.definition}")
    wstore.mirror_entry(entry, vec)


def _enqueue_review(
    *,
    concept: dict[str, Any],
    paper_id: str,
    candidates: list[dict[str, Any]],
    reason: str,
) -> None:
    from . import review_queue

    review_queue.enqueue(
        "resolve_review",
        concept=concept.get("name"),
        paper_id=paper_id,
        reason=reason,
        payload={"concept": concept, "candidates": candidates},
    )


def _mechanical_labels(concept: dict[str, Any], paper_id: str) -> list[WikiLabel]:
    """抽取器给出的双语标签 + surface_name, 机械并入词条(去重在 store 层)。"""
    out: list[WikiLabel] = []
    for text in [concept.get("surface_name"), *concept.get("labels_zh", [])]:
        if text:
            out.append(
                WikiLabel(text=text, language="zh", kind="translation", source_paper_id=paper_id)
            )
    for text in concept.get("labels_en", []):
        if text:
            out.append(
                WikiLabel(text=text, language="en", kind="variant", source_paper_id=paper_id)
            )
    return out


def _handle_match(
    entry_id: str,
    concept: dict[str, Any],
    paper: dict[str, Any],
    language: str | None,
    evidence_chunks: list[dict[str, Any]],
) -> bool:
    """返回是否成功 patch。机械关联(论文/证据/标签)无条件落库。"""
    paper_id = paper["paper_id"]
    wstore.add_key_papers(entry_id, [paper_id])
    wstore.add_evidence(
        entry_id,
        [{"chunk_id": c["chunk_id"], "paper_id": paper_id} for c in evidence_chunks],
    )
    labels = _mechanical_labels(concept, paper_id)
    if labels:
        wstore.add_labels(entry_id, labels)

    existing = wstore.get_entry(entry_id)
    if existing is None:
        return False
    patched = flow.patch_entry(
        existing=existing,
        paper_id=paper_id,
        paper_title=paper.get("title") or "",
        language=language,
        chunks=evidence_chunks,
    )
    if patched is None:
        return False
    saved = wstore.upsert_entry(patched, reason=f"patched from {paper_id}")
    try:
        _mirror(saved)
    except Exception as e:
        log.warning(f"qdrant mirror pending for {saved.entry_id}: {e}")
    return True


def on_paper_indexed(
    paper_id: str,
    *,
    language: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """为一篇已入库论文运行 wiki 更新, 返回小报告。所有失败路径均有诚实信号。"""
    conf = cfg.load().wiki
    if not conf.enabled and not force:
        return {"skipped": "wiki_disabled"}

    paper = _load_paper(paper_id)
    if paper is None:
        return {"error": f"paper not found: {paper_id}"}

    parsed_with = paper.get("parsed_with") or ""
    if parsed_with in conf.quality_gate.skip_parsed_with:
        return {"skipped": f"parsed_with={parsed_with}"}

    chunks = _load_text_chunks(paper_id)
    if len(chunks) < conf.quality_gate.min_chunks:
        return {"skipped": f"chunks={len(chunks)}<{conf.quality_gate.min_chunks}"}

    concepts = concept_extractor.extract_concepts(
        title=paper.get("title") or "", chunks=chunks, language=language
    )

    chunk_by_id = {c["chunk_id"]: c for c in chunks}
    created = patched = review = dropped = 0
    for concept in concepts:
        try:
            evidence = [
                chunk_by_id[cid] for cid in concept["evidence_chunk_ids"] if cid in chunk_by_id
            ] or chunks[:3]
            hint = " ".join((c.get("text") or "")[:150] for c in evidence[:2])
            resolved = normalize.resolve_concept(
                concept["name"], language=language, definition_hint=hint
            )
            if resolved["decision"] == "match":
                if _handle_match(resolved["entry_id"], concept, paper, language, evidence):
                    patched += 1
            elif resolved["decision"] == "novel":
                entry = flow.create_entry(
                    name=concept["name"],
                    category=concept["category"],
                    language=language,
                    paper_id=paper_id,
                    paper_title=paper.get("title") or "",
                    chunks=evidence,
                    labels_zh=[concept["surface_name"], *concept["labels_zh"]]
                    if language == "zh"
                    else concept["labels_zh"],
                    labels_en=concept["labels_en"],
                )
                if entry is None:
                    dropped += 1
                    continue
                saved = wstore.upsert_entry(entry, reason=f"created from {paper_id}")
                try:
                    _mirror(saved)
                except Exception as e:
                    log.warning(f"qdrant mirror pending for {saved.entry_id}: {e}")
                created += 1
            else:  # review
                try:
                    _enqueue_review(
                        concept=concept,
                        paper_id=paper_id,
                        candidates=resolved.get("candidates") or [],
                        reason=resolved.get("reason") or "resolve_review",
                    )
                except Exception as e:
                    log.warning(f"review enqueue failed (non-fatal): {e}")
                review += 1
        except Exception as e:
            dropped += 1
            log.warning(f"wiki concept {concept.get('name')!r} failed for {paper_id}: {e}")

    log.info(
        f"wiki update for {paper_id}: created={created} patched={patched} "
        f"review={review} dropped={dropped}"
    )
    return {
        "paper_id": paper_id,
        "created": created,
        "patched": patched,
        "review": review,
        "dropped": dropped,
    }
