"""入库流水线: 采集产物 -> 解析 -> 切块 -> 嵌入 -> SQLite/Qdrant 索引的唯一入口。

状态机(任一步失败转 failed 并记录 error):
    created -> fetched -> parsed -> chunked -> embedded -> indexed -> done

每步在 ingest_runs 表记 start/finish/error(_step 包装器), 便于事后诊断。
跨源去重: 入库前按 DOI > arxiv_id > 规范化标题 探测, 命中返回 merged_into
并跳过; 已 done 的论文跳过; force=True 强制重建。Qdrant 写入是替换语义
(先删该论文旧点再插新点), 重跑不残留脏向量。

与基准的差异(2026-08-01 已确认):
- 语言贯通: 用 chunk.builder.read_language 读解析层 language.json, 同一语言值
  喂给 grade_sections(基准不传语言, 中文论文会被英文关键词表降级误判)与
  元数据卡片模板——与 chunk 层同源, 不分叉。
- 元数据卡片按语言路由: zh 用中文模板(论文元数据记录。/标题:/作者:/...),
  en/None 用基准英文; _title_aliases 缩写词逻辑保持基准(中文标题优雅空集)。
- 真空守卫前移: 基准把 "chunks 为空则 failed" 放在插入元数据卡片之后, 卡片
  必然存在使守卫成为死代码; 重建版在插卡之前检查 build_chunks 的真实产物。
- wiki 入队钩子改为持久化队列: 只做一次幂等 INSERT(paper_id + 内容指纹 + 语言),
  LLM 成本全部在独立 wiki_worker 进程, 批量入库与 wiki 建设解耦;
  try/except 非致命, 失败只记 warning 不影响入库结果。
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Any

from .. import config as cfg
from ..chunk.builder import build_chunks, read_language
from ..embed import bge_m3
from ..ingest.dedup import normalize_title
from ..ingest.schema import FetchResult
from ..parse.dispatcher import parse_pdf
from ..utils.logger import get_logger
from . import qdrant_store, sqlite_store
from .incremental import build_chunk_fingerprints, plan_incremental_update

log = get_logger("store.ingest")


_TITLE_ALIAS_SPLIT_RE = re.compile(r":|\s+for\s+|\s+with\s+|\s+via\s+", re.IGNORECASE)
_TITLE_WORD_RE = re.compile(r"[A-Za-z]+")
_ACRONYM_STOPWORDS = {"a", "an", "and", "of", "the", "to", "in", "on"}

# 元数据卡片文案按语言路由; 键序即卡片行序
_CARD_LABELS = {
    "en": {
        "header": "Paper metadata record.",
        "purpose": (
            "Retrieve this record for questions about this paper as a whole, including its "
            "contribution, method, evaluation, datasets, experiments, results, limitations, "
            "figures, tables, and formulas."
        ),
        "title": "Title",
        "paper_id": "Paper id",
        "arxiv_id": "arXiv id",
        "aliases": "Title aliases and acronyms",
        "authors": "Authors",
        "year": "Year",
        "venue": "Venue",
        "doi": "DOI",
        "abstract": "Abstract",
    },
    "zh": {
        "header": "论文元数据记录。",
        "purpose": (
            "当问题涉及这篇论文整体时检索本记录, "
            "包括其贡献、方法、评测、数据集、实验、结果、局限、图、表与公式。"
        ),
        "title": "标题",
        "paper_id": "论文 id",
        "arxiv_id": "arXiv id",
        "aliases": "标题别名与缩写",
        "authors": "作者",
        "year": "年份",
        "venue": "发表于",
        "doi": "DOI",
        "abstract": "摘要",
    },
}


def _step(paper_id: str, name: str, fn) -> Any:
    """执行一个流水线步骤, 在 ingest_runs 记 start/finish, 失败上抛。"""
    run_id = sqlite_store.record_ingest_step(paper_id, name)
    try:
        out = fn()
    except Exception as e:
        sqlite_store.finish_ingest_step(run_id, status="error", error=str(e))
        sqlite_store.set_status(paper_id, "failed", error=f"{name}: {e}")
        raise
    sqlite_store.finish_ingest_step(run_id, status="ok")
    return out


def _resolve_dedup(meta) -> str | None:
    """若同一篇论文已以其他 id 存在, 返回既有 paper_id。"""
    title_norm = normalize_title(meta.title) if meta.title else None
    existing = sqlite_store.find_existing_paper(
        doi=meta.doi,
        arxiv_id=meta.arxiv_id,
        title_norm=title_norm,
    )
    if existing and existing.paper_id != meta.paper_id:
        return existing.paper_id
    return None


def _title_aliases(title: str) -> list[str]:
    """从标题主短语提取轻量缩写词, 供整篇论文检索(中文标题优雅空集)。"""
    aliases: list[str] = []
    phrase = _TITLE_ALIAS_SPLIT_RE.split(title, maxsplit=1)[0]
    words = [w for w in _TITLE_WORD_RE.findall(phrase) if w.lower() not in _ACRONYM_STOPWORDS]
    acronym = "".join(w[0].upper() for w in words)
    if 2 <= len(acronym) <= 8 and acronym not in title:
        aliases.append(acronym)
    return aliases


def _paper_metadata_chunk(meta, *, language: str | None = None) -> dict[str, Any]:
    """整篇论文的"名片" chunk: 标题/摘要/别名可召回(modality="metadata")。

    正文块擅长回答细节, 但"这篇论文在哪些数据集上评测"类问题往往先用标题、
    缩写或 arXiv 号锁定论文——本卡片给稠密/稀疏检索一个整篇目标, 不参与
    答案生成的证据边界之外的任何环节。
    """
    lb = _CARD_LABELS["zh" if language == "zh" else "en"]
    chunk_id = hashlib.sha1(f"{meta.paper_id}::paper-metadata".encode()).hexdigest()[:20]
    aliases = _title_aliases(meta.title or "")
    lines = [
        lb["header"],
        lb["purpose"],
        f"{lb['title']}: {meta.title}",
        f"{lb['paper_id']}: {meta.paper_id}",
    ]
    if meta.arxiv_id:
        lines.append(f"{lb['arxiv_id']}: {meta.arxiv_id}")
    if aliases:
        lines.append(f"{lb['aliases']}: " + ", ".join(aliases))
    if meta.authors:
        lines.append(f"{lb['authors']}: " + ", ".join(meta.authors[:12]))
    if meta.year:
        lines.append(f"{lb['year']}: {meta.year}")
    if meta.venue:
        lines.append(f"{lb['venue']}: {meta.venue}")
    if meta.doi:
        lines.append(f"{lb['doi']}: {meta.doi}")
    if meta.abstract:
        lines.append(f"{lb['abstract']}: " + " ".join(meta.abstract.split()))

    text = "\n".join(lines)
    return {
        "chunk_id": chunk_id,
        "paper_id": meta.paper_id,
        "section_id": None,
        "section": "Paper Metadata",
        "section_idx": -1,
        "modality": "metadata",
        "page": None,
        "text": text,
        "context_text": text,
        "title": meta.title,
        "source_path": None,
        "metadata": {
            "element_type": "paper_metadata",
            "source": meta.source,
            "aliases": aliases,
        },
        "neighbors": [],
    }


def ingest(
    result: FetchResult,
    *,
    force: bool = False,
    timings: dict[str, float] | None = None,
) -> dict[str, Any]:
    total_started = time.perf_counter()
    timings = timings if timings is not None else {}
    paper_id = result.meta.paper_id

    # 跨源去重探测(同 paper_id 不触发)
    merged_into = _resolve_dedup(result.meta)
    if merged_into and not force:
        log.info(f"dedup: {paper_id} already exists as {merged_into}, skip")
        return {
            "paper_id": paper_id,
            "status": "skipped",
            "merged_into": merged_into,
            "reason": "dedup",
        }

    if not force:
        existing = sqlite_store.get_paper(paper_id)
        if existing and existing.status == "done":
            log.info(f"{paper_id} already done, skip")
            return {"paper_id": paper_id, "status": "skipped", "reason": "done"}

    sqlite_store.upsert_paper(result.meta.model_dump(mode="json"), status="fetched")

    parse_started = time.perf_counter()
    parsed, parser_name = _step(
        paper_id,
        "parse",
        lambda: parse_pdf(paper_id, result.pdf_path),
    )
    timings["parse_seconds"] = time.perf_counter() - parse_started
    sqlite_store.set_status(paper_id, "parsed", parsed_with=parser_name)
    language = read_language(Path(parsed))

    chunk_started = time.perf_counter()
    chunk_timings: dict[str, float] = {}
    build_kwargs = {"title": result.meta.title}

    def _build_chunks_with_timing():
        try:
            return build_chunks(
                paper_id,
                Path(parsed),
                **build_kwargs,
                timings=chunk_timings,
            )
        except TypeError as exc:
            if "timings" not in str(exc):
                raise
            return build_chunks(paper_id, Path(parsed), **build_kwargs)

    sections, chunks = _step(
        paper_id,
        "chunk",
        _build_chunks_with_timing,
    )
    build_seconds = time.perf_counter() - chunk_started
    vision_seconds = chunk_timings.get("vision_seconds", 0.0)
    timings["vision_seconds"] = vision_seconds
    timings["chunk_seconds"] = max(0.0, build_seconds - vision_seconds)
    # 真空守卫在插卡之前: 元数据卡片必然存在, 放在插卡后是死代码(基准缺陷)
    if not chunks:
        sqlite_store.set_status(
            paper_id, "failed", error="chunk: empty (parser produced no chunks)"
        )
        return {"paper_id": paper_id, "status": "failed", "reason": "no_chunks"}
    chunks.insert(0, _paper_metadata_chunk(result.meta, language=language))
    sqlite_started = time.perf_counter()
    embedding_version = _embedding_version()
    chunks = [
        build_chunk_fingerprints(chunk, embedding_version=embedding_version) for chunk in chunks
    ]
    sqlite_store.upsert_sections_and_chunks(paper_id, sections, chunks)
    timings["sqlite_seconds"] = time.perf_counter() - sqlite_started

    # 章节完整性打分拼进 parsed_with(如 "mineru+broken"), 供日后过滤坏解析
    try:
        from ..chunk.sanity import grade_sections

        quality = grade_sections([sec.get("name", "") for sec in sections], language=language)
        sqlite_store.set_status(
            paper_id,
            "parsed",
            parsed_with=f"{parser_name}+{quality}",
        )
        log.info(f"section quality for {paper_id}: {quality}")
    except Exception as e:
        log.warning(f"section grading skipped: {e}")
    sqlite_store.set_status(paper_id, "chunked")

    incremental_started = time.perf_counter()
    snapshot_started = time.perf_counter()
    old_points = _step(
        paper_id,
        "qdrant_snapshot",
        lambda: qdrant_store.list_chunks_for_paper(paper_id),
    )
    timings["qdrant_snapshot_seconds"] = time.perf_counter() - snapshot_started
    plan_started = time.perf_counter()
    plan = plan_incremental_update(chunks, old_points)
    timings["incremental_plan_seconds"] = time.perf_counter() - plan_started

    vector_started = time.perf_counter()
    vectors = _step(
        paper_id,
        "embed",
        lambda: bge_m3.encode([c["context_text"] for c in plan.vector_updates]),
    )
    timings["embedding_seconds"] = time.perf_counter() - vector_started
    sqlite_store.set_status(paper_id, "embedded")

    index_started = time.perf_counter()
    index_timings: dict[str, float] = {}
    _step(
        paper_id,
        "index",
        lambda: _index_chunks(paper_id, plan, vectors, timings=index_timings),
    )
    timings["index_seconds"] = time.perf_counter() - index_started
    timings.update(index_timings)
    timings["incremental_update_seconds"] = time.perf_counter() - incremental_started
    sqlite_store.set_status(paper_id, "indexed")
    sqlite_store.set_status(paper_id, "done")

    # wiki 持久化入队(非阻塞): 语言与内容指纹显式随任务传递, worker 异步消费。
    # force 重建 -> chunk 集合变化 -> 指纹变化 -> 自然产生新任务(幂等键失配)。
    wiki_started = time.perf_counter()
    try:
        from ..wiki.queue import submit_paper_indexed

        wiki_report = submit_paper_indexed(
            paper_id,
            language=language,
            content_fingerprint=_content_fingerprint(chunks),
        )
    except Exception as e:
        log.warning(f"wiki enqueue failed (non-fatal): {e}")
        wiki_report = {"error": str(e)}
    timings["wiki_enqueue_seconds"] = time.perf_counter() - wiki_started

    timings["total_seconds"] = time.perf_counter() - total_started
    out_incremental = {
        "vector_updates": len(plan.vector_updates),
        "payload_updates": len(plan.payload_updates),
        "skipped": len(plan.skipped),
        "deleted": len(plan.delete_ids),
    }
    try:
        from ..dashboard.services.pipeline_monitor import record_ingestion_run

        record_ingestion_run(
            paper_id=paper_id,
            status="done",
            timings_seconds=timings,
            metadata={"chunks": len(chunks), "incremental": out_incremental},
        )
    except Exception as exc:
        log.warning(f"pipeline monitor write skipped (non-fatal): {exc}")
    return {
        "paper_id": paper_id,
        "status": "done",
        "chunks": len(chunks),
        "wiki": wiki_report,
        "incremental": out_incremental,
        "timings": timings or {},
    }


def _embedding_version() -> str:
    """读取显式 embedding 版本, 兼容尚未加入 version 字段的测试配置。"""
    embedding = cfg.load().embedding
    return str(getattr(embedding, "version", "bge-m3:v1"))


def _content_fingerprint(chunks: list[dict]) -> str:
    """排序后 chunk_id 集合的 sha1: 与内容切块一一对应, 与入库顺序无关。"""
    ids = sorted(f"{c.get('chunk_id') or ''}\t{c.get('content_id') or ''}" for c in chunks)
    return hashlib.sha1("\n".join(ids).encode("utf-8")).hexdigest()


def _index_chunks(
    paper_id: str,
    plan,
    vectors: list[list[float]],
    *,
    timings: dict[str, float] | None = None,
) -> int:
    """按差量计划同步 Qdrant, 先写新数据再删除旧 Point。"""
    qdrant_started = time.perf_counter()
    written = qdrant_store.upsert_chunks(plan.vector_updates, vectors)
    for item in plan.payload_updates:
        qdrant_store.overwrite_chunk_payload(item)
    deleted = qdrant_store.delete_points(plan.delete_ids)
    if timings is not None:
        timings["qdrant_write_seconds"] = time.perf_counter() - qdrant_started
    fts_started = time.perf_counter()
    _sync_fts5_nonfatal(paper_id)
    if timings is not None:
        timings["fts5_sync_seconds"] = time.perf_counter() - fts_started
    return written + len(plan.payload_updates) + deleted


def _sync_fts5_nonfatal(paper_id: str) -> None:
    """FTS5 稀疏索引单篇增量同步(ADR-0001 规模修订: 10^6 chunks 下不能依赖
    分钟级的全量自愈)。失败不致命——打 warning, search 行数自愈兜底。"""
    try:
        from ..retrieve import fts5

        fts5.sync_paper(paper_id)
    except Exception as e:
        log.warning(f"fts5 sync skipped (non-fatal): {e}")
