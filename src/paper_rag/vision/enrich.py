"""视觉增强编排: 给 figure/table chunk 补摘要, 且绝不打断 ingest。

语言贯通是本模块的中文扩展主线: ``language`` 由 builder 的 ``read_language``
传入 → 提示词路由(api/local)→ 缓存键 → 追加行标签。四处任缺一处, 中文论文
就会得到英文摘要或脏缓存, 而摘要要同时进 BGE-M3 与 FTS5/BM25, 英文摘要在
中文语料的词面侧等于失明。

取料走"metadata 正门 + 文本反解兜底"两级(基准留了正门但无生产方写入):
builder 已把 layout 图注与邻近上下文写进 metadata; 反解仅用于非 builder 直出
的 chunk, 且按 zh/en 前缀路由并修掉基准表块"同行取 caption"恒空的缺陷。

失败语义一律 fail-open: 单图 skipped/failed 只写 metadata 记账, chunk 原文
逐字保留, 论文照常入库。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from .. import config as cfg
from ..utils.logger import get_logger
from .cache import VisionSummaryCache
from .schema import (
    STATUS_CACHED,
    STATUS_FAILED,
    STATUS_FALLBACK,
    STATUS_OK,
    STATUS_SKIPPED,
    STATUS_UNAVAILABLE,
    VisualSummaryRequest,
    VisualSummaryResult,
)

log = get_logger("vision.enrich")

SummarizerFn = Callable[[VisualSummaryRequest], VisualSummaryResult]


@dataclass
class _EnrichmentPlan:
    """A visual chunk's result slot; only the main thread commits it."""

    chunk: dict
    request: VisualSummaryRequest | None = None
    cache_key: str | None = None
    result: VisualSummaryResult | None = None
    status: str | None = None
    error: str = ""
    future: Future[tuple[VisualSummaryResult, str]] | None = None


# 只有带图片资产的 figure/table 进视觉模型; formula 是 LaTeX 文本, 无 asset。
_VISUAL_MODALITIES = {"figure", "table"}

_SUMMARY_LABEL_ZH = "视觉摘要"
_SUMMARY_LABEL_EN = "Visual summary"

# 反解兜底: 与 multimodal_chunker 的 compose_* 模板前缀对齐(半/全角冒号均认)。
_CAPTION_PREFIXES = ("figure", "table", "图", "表")
_CONTEXT_PREFIXES = ("context", "上下文")
_COLON = "[:：]"  # noqa: RUF001 — 全角冒号是中文 PDF 的真实形态, 即业务数据
_CONTEXT_CHARS = 500


def _summary_label(language: str | None) -> str:
    return _SUMMARY_LABEL_ZH if language == "zh" else _SUMMARY_LABEL_EN


def _split_prefixed(line: str, prefixes: tuple[str, ...]) -> tuple[str, str] | None:
    """把 ``前缀: 值`` 拆成 (前缀, 值); 不匹配返回 None。"""
    for prefix in prefixes:
        m = re.match(rf"^\s*{re.escape(prefix)}\s*{_COLON}(?P<value>.*)$", line, re.IGNORECASE)
        if m:
            return prefix, m.group("value").strip()
    return None


def _caption_from_text(text: str) -> str:
    """反解图注; 表块的 ``表:`` 后值在下一行(基准同行取值恒空, 此处修复)。"""
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        hit = _split_prefixed(line, _CAPTION_PREFIXES)
        if hit is None:
            continue
        _, value = hit
        if value:
            return value
        for nxt in lines[idx + 1 :]:
            stripped = nxt.strip()
            if not stripped:
                continue
            if _split_prefixed(stripped, _CONTEXT_PREFIXES):
                break
            return stripped
        return ""
    return ""


def _context_from_text(text: str) -> str:
    for line in text.splitlines():
        hit = _split_prefixed(line, _CONTEXT_PREFIXES)
        if hit is not None and hit[1]:
            return hit[1]
    return text[:_CONTEXT_CHARS]


def request_from_chunk(
    chunk: dict,
    *,
    model: str | None = None,
    language: str | None = None,
) -> VisualSummaryRequest:
    """从 chunk 组装请求: metadata 原料优先, 文本反解兜底。"""
    metadata = chunk.get("metadata") or {}
    text = str(chunk.get("text") or "")
    return VisualSummaryRequest(
        paper_id=str(chunk.get("paper_id") or ""),
        chunk_id=str(chunk.get("chunk_id") or ""),
        modality=str(chunk.get("modality") or ""),
        asset_path=Path(str(chunk.get("asset_path"))),
        caption=str(metadata.get("caption") or _caption_from_text(text)),
        surrounding_context=str(metadata.get("surrounding_context") or _context_from_text(text)),
        model=model,
        language=language,
    )


def _append_summary(chunk: dict, summary: str, language: str | None) -> None:
    """把摘要追加进 text 与 context_text; 重复 enrich 不重复追加。"""
    line = f"{_summary_label(language)}: {summary}"
    for field in ("text", "context_text"):
        current = str(chunk.get(field) or "")
        if line in current:
            continue
        chunk[field] = f"{current}\n{line}" if current else line


def _record(chunk: dict, status: str, result: VisualSummaryResult | None, error: str = "") -> None:
    metadata = chunk.setdefault("metadata", {})
    metadata["visual_summary_status"] = status
    if result is not None:
        if result.summary:
            metadata["visual_summary"] = result.summary
        # 失败路径也记 provider/model, 否则排障时无法区分 api 还是 local 出错
        if result.provider:
            metadata["visual_summary_provider"] = result.provider
        if result.model:
            metadata["visual_summary_model"] = result.model
    message = error or (result.error if result is not None else "")
    if message:
        metadata["visual_summary_error"] = message


def _default_summarizer(vision) -> SummarizerFn | None:
    if not (vision.base_url and vision.api_key and vision.model):
        return None
    from .api import OpenAIVisionSummarizer

    return OpenAIVisionSummarizer(
        base_url=vision.base_url,
        api_key=vision.api_key,
        model=vision.model,
        timeout_sec=vision.timeout_sec,
        temperature=vision.temperature,
        extra_body=vision.extra_body,
    ).summarize


def _default_fallback(vision) -> SummarizerFn | None:
    if not vision.fallback_local:
        return None
    from .local import LocalVisionSummarizer

    return LocalVisionSummarizer(vision.local_model).summarize


def enrich_chunks(
    paper_id: str,
    chunks: list[dict],
    *,
    summarizer: SummarizerFn | None = None,
    fallback_summarizer: SummarizerFn | None = None,
    cache: VisionSummaryCache | None = None,
    cache_enabled: bool | None = None,
    language: str | None = None,
    max_image_bytes: int | None = None,
    max_images_per_paper: int | None = None,
    max_concurrency: int | None = None,
) -> list[dict]:
    """就地给本篇的 figure/table chunk 追加视觉摘要, 返回同一列表。

        ``summarizer`` 未注入时按配置构造; 配置不全则整体跳过(chunk 原样返回)。
    并发线程只请求摘要, 不修改 chunk 也不写缓存; 全部结果就绪后由
    主线程按输入 chunk 顺序追加摘要、记录状态并写缓存。
    """
    vision = cfg.load().vision
    if summarizer is None:
        if not vision.enabled:
            return chunks
        summarizer = _default_summarizer(vision)
        if summarizer is None:
            log.warning("vision.enabled 但 base_url/api_key/model 不全, 跳过视觉增强")
            return chunks
        if fallback_summarizer is None:
            fallback_summarizer = _default_fallback(vision)

    use_cache = vision.cache if cache_enabled is None else cache_enabled
    if use_cache and cache is None:
        cache = VisionSummaryCache(vision.cache_dir)
    limit = vision.max_images_per_paper if max_images_per_paper is None else max_images_per_paper
    size_cap = vision.max_image_bytes if max_image_bytes is None else max_image_bytes
    concurrency = (
        vision.max_concurrency if max_concurrency is None else max(1, int(max_concurrency))
    )

    processed = 0
    plans: list[_EnrichmentPlan] = []
    for chunk in chunks:
        if chunk.get("modality") not in _VISUAL_MODALITIES:
            continue
        if str(chunk.get("paper_id") or "") != paper_id:
            continue

        plan = _EnrichmentPlan(chunk=chunk)
        plans.append(plan)
        asset = chunk.get("asset_path")
        if not asset:
            plan.status = STATUS_SKIPPED
            plan.error = "missing asset_path"
            continue
        path = Path(str(asset))
        if not path.exists():
            plan.status = STATUS_SKIPPED
            plan.error = f"asset not found: {path}"
            continue
        if processed >= limit:
            plan.status = STATUS_SKIPPED
            plan.error = f"max_images_per_paper={limit} reached"
            continue
        if path.stat().st_size > size_cap:
            plan.status = STATUS_SKIPPED
            plan.error = f"image exceeds max_image_bytes={size_cap}"
            continue

        request = request_from_chunk(chunk, model=vision.model, language=language)
        plan.request = request
        processed += 1

        key = cache.key_for(request) if cache is not None and use_cache else None
        plan.cache_key = key
        if key is not None:
            hit = cache.read(key)
            if hit is not None and hit.summary:
                plan.result = hit
                plan.status = STATUS_CACHED
                plan.request = None
                continue

    pending = [plan for plan in plans if plan.request is not None and plan.result is None]
    fallback_lock = Lock() if fallback_summarizer is not None else None
    workers = min(concurrency, len(pending))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vision") as executor:
            for plan in pending:
                plan.future = executor.submit(
                    _summarize_request,
                    summarizer,
                    fallback_summarizer,
                    plan.request,
                    fallback_lock,
                )
            # Resolve slots in input order. Workers may finish in any order, but do not mutate state.
            for plan in pending:
                plan.result, plan.status = plan.future.result()
    else:
        for plan in pending:
            plan.result, plan.status = _summarize_request(
                summarizer,
                fallback_summarizer,
                plan.request,
                fallback_lock,
            )

    # The only mutation/cache-write phase. plans preserves the original visual chunk order.
    for plan in plans:
        chunk, result, status = plan.chunk, plan.result, plan.status
        if status == STATUS_SKIPPED:
            _record(chunk, status, None, plan.error)
            continue
        if result is None or status is None:
            _record(
                chunk,
                STATUS_FAILED,
                result,
                "vision worker returned no result",
            )
            continue
        if status == STATUS_CACHED:
            _append_summary(chunk, result.summary, language)
            _record(chunk, status, result)
            continue
        if result.status not in (STATUS_OK,) or not result.summary:
            _record(chunk, result.status if result.status != STATUS_OK else STATUS_FAILED, result)
            continue

        _append_summary(chunk, result.summary, language)
        _record(chunk, status, result)
        if plan.cache_key is not None and cache is not None:
            cache.write(plan.cache_key, result)
    return chunks


def _summarize_request(
    summarizer: SummarizerFn,
    fallback_summarizer: SummarizerFn | None,
    request: VisualSummaryRequest,
    fallback_lock: Lock | None,
) -> tuple[VisualSummaryResult, str]:
    """工作线程的纯计算边界: 只返回结果, 不修改 chunk/缓存。"""
    result = _invoke(summarizer, request)
    status = STATUS_OK
    if result.status != STATUS_OK and fallback_summarizer is not None:
        if fallback_lock is None:
            fallback = _invoke(fallback_summarizer, request)
        else:
            # Local fallback owns one GPU model and must not run concurrently.
            with fallback_lock:
                fallback = _invoke(fallback_summarizer, request)
        if fallback.status == STATUS_OK and fallback.summary:
            result, status = fallback, STATUS_FALLBACK
    return result, status


def _invoke(fn: SummarizerFn, request: VisualSummaryRequest) -> VisualSummaryResult:
    """summarizer 自身抛错也不得穿透到 ingest。"""
    try:
        return fn(request)
    except Exception as exc:
        return VisualSummaryResult(status=STATUS_FAILED, error=str(exc))


__all__ = [
    "STATUS_UNAVAILABLE",
    "SummarizerFn",
    "enrich_chunks",
    "request_from_chunk",
]
