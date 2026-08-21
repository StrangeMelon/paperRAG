"""retrieve 层组合入口: rewrite -> 多查询池化 hybrid -> rerank -> 论文多样化。

供 QA 三条路径(agentic/stream/simple)复用的"一轮检索", 集中在这里避免每个
调用点复制粘贴组装逻辑(改精排窗口/换模型只动一处)。

组装比朴素串联多四个心思(与基准一致):
1. 多查询池化: query rewrite 产出的多个查询各跑一遍 hybrid, 按 chunk_id
   去重, 同块保留 score_rrf 最高的版本;
2. 模态线索: 从问题嗅探 公式/图/表 意图(线索表含中文), 追加定向检索轮;
3. 精排窗口 top_k*3(多查询池比单轮更杂, 窗口放宽);
4. 论文多样化: 单篇最多 2 块, 溢出块垫底补位, 防单篇霸榜。

相对基准的过渡偏离(P6 收尾课确认): rag.query_rewrite 尚未重建(P7)时,
默认回退恒等改写并打 warning——与 wiki/vision 钩子同款诚实信号; P7 落地后
默认导入自动恢复, 无需回改。
"""

from __future__ import annotations

import time

from .. import config as cfg
from ..utils.logger import get_logger
from .hybrid import hybrid_search
from .reference_policy import apply_reference_ranking, detect_reference_intent
from .rerank import rerank as _rerank

log = get_logger("retrieve.pipeline")

_MODALITY_HINTS = {
    "formula": (
        "formula",
        "equation",
        "latex",
        "derive",
        "derivation",
        "公式",
        "方程",
        "推导",
    ),
    "figure": (
        "figure",
        "fig.",
        "diagram",
        "plot",
        "image",
        "图",
        "图像",
        "图表",
        "示意图",
    ),
    "table": (
        "table",
        "tab.",
        "表",
        "表格",
        "对比表",
    ),
}
_MAX_CHUNKS_PER_PAPER = 50


def infer_modalities(query: str) -> list[str]:
    """从用户问题里嗅探模态检索线索(中英线索表)。"""
    q = query.lower()
    return [modality for modality, hints in _MODALITY_HINTS.items() if any(h in q for h in hints)]


def _identity_rewrite(query: str, wiki_context: dict | None = None) -> dict:
    """P7 重建 rag.query_rewrite 前的恒等改写回退。"""
    return {"dense_queries": [query]}


def retrieve_round_with_rewrite(
    query: str,
    paper_ids: list[str] | None,
    top_k: int,
    *,
    rewrite_fn=None,
    rewrite_enabled: bool = True,
    reference_intent: bool | None = None,
    wiki_context: dict | None = None,
    timings: dict[str, float] | None = None,
    diagnostics: dict[str, object] | None = None,
    evaluation_parallel: bool = False,
) -> tuple[list[dict], dict] | tuple[list[dict], dict, dict[str, float]]:
    """一轮检索。返回 (重排后的 chunks, 改写载荷)。

    ``rewrite_fn`` 可注入(测试免打补丁); 默认用 rag.query_rewrite.rewrite,
    该模块未重建时回退恒等改写(warning 是诚实信号)。
    """
    if not rewrite_enabled:
        rewrite_fn = _identity_rewrite
    elif rewrite_fn is None:
        try:
            from ..rag.query_rewrite import rewrite as rewrite_fn  # 局部导入避免环
        except Exception as e:
            log.warning(f"query_rewrite unavailable, using identity rewrite: {e}")
            rewrite_fn = _identity_rewrite

    rewrite_started = time.perf_counter()
    try:
        rw = rewrite_fn(query, wiki_context=wiki_context)
    except TypeError as exc:
        if "wiki_context" not in str(exc):
            raise
        rw = rewrite_fn(query)
    if timings is not None:
        timings["query_rewrite_ms"] = round((time.perf_counter() - rewrite_started) * 1000, 1)
    if diagnostics is not None:
        diagnostics["rewrite"] = dict(rw)
    reference_cfg = cfg.load().retrieve.references
    resolved_reference_intent = (
        detect_reference_intent(query) if reference_intent is None else reference_intent
    )
    modalities = infer_modalities(query)
    pooled: dict[str, dict] = {}
    hybrid_started = time.perf_counter()
    for q in rw["dense_queries"]:
        search_specs = [(None, top_k)]
        search_specs.extend((modality, top_k) for modality in modalities)
        for modality, k in search_specs:
            hybrid_timings: dict[str, float] = {}
            hybrid_diagnostics: dict[str, list[dict]] = {}
            hybrid_kwargs = {
                "top_k": k,
                "paper_ids": paper_ids,
                "modality": modality,
            }
            if evaluation_parallel:
                hybrid_kwargs["allow_concurrent"] = True
            if rw.get("bm25_query"):
                hybrid_kwargs["sparse_query"] = rw["bm25_query"]
            if timings is None:
                hits = _hybrid_search_compat(q, hybrid_kwargs)
            else:
                diagnostic_kwargs = {
                    **hybrid_kwargs,
                    "timings": hybrid_timings,
                }
                if diagnostics is not None:
                    diagnostic_kwargs["diagnostics"] = hybrid_diagnostics
                hits = _hybrid_search_compat(q, diagnostic_kwargs)
                for name, value in hybrid_timings.items():
                    timings[name] = timings.get(name, 0.0) + float(value)
                if diagnostics is not None:
                    for stage in ("dense", "sparse", "rrf"):
                        diagnostics.setdefault(stage, [])
                        diagnostics[stage].extend(hybrid_diagnostics.get(stage, []))
            for hit in hits:
                cid = hit.get("chunk_id")
                if not cid:
                    continue
                if cid not in pooled or hit.get("score_rrf", 0) > pooled[cid].get("score_rrf", 0):
                    pooled[cid] = hit
    if timings is not None:
        timings["hybrid_ms"] = round((time.perf_counter() - hybrid_started) * 1000, 1)
    candidates = list(pooled.values())
    candidates = apply_reference_ranking(
        candidates,
        reference_intent=resolved_reference_intent,
        penalty=reference_cfg.penalty,
        enabled=reference_cfg.enabled,
        legacy_section_fallback=reference_cfg.legacy_section_fallback,
    )
    candidates = candidates[: top_k * 3]
    rerank_started = time.perf_counter()
    try:
        ranked = _rerank(query, candidates, top_k=top_k * 3, allow_concurrent=evaluation_parallel)
    except TypeError as exc:
        if "allow_concurrent" not in str(exc):
            raise
        ranked = _rerank(query, candidates, top_k=top_k * 3)
    ranked = apply_reference_ranking(
        ranked,
        reference_intent=resolved_reference_intent,
        penalty=reference_cfg.penalty,
        enabled=reference_cfg.enabled,
        legacy_section_fallback=reference_cfg.legacy_section_fallback,
    )
    if timings is not None:
        timings["rerank_ms"] = round((time.perf_counter() - rerank_started) * 1000, 1)
    diversify_started = time.perf_counter()
    selected = _diversify_by_paper(ranked, top_k=top_k)
    if diagnostics is not None:
        diagnostics["rerank"] = [dict(item) for item in ranked]
        diagnostics["diversify"] = [dict(item) for item in selected]
        diagnostics["reference_policy"] = {
            "enabled": reference_cfg.enabled,
            "intent": resolved_reference_intent,
            "penalty": reference_cfg.penalty,
            "penalized_chunks": sum(bool(item.get("reference_penalized")) for item in ranked),
        }
    if timings is not None:
        timings["diversify_ms"] = round((time.perf_counter() - diversify_started) * 1000, 1)
        return selected, rw, timings
    return selected, rw


def retrieve_round(
    query: str,
    paper_ids: list[str] | None,
    top_k: int,
    *,
    wiki_context: dict | None = None,
    timings: dict[str, float] | None = None,
    rewrite_enabled: bool = True,
    reference_intent: bool | None = None,
    evaluation_parallel: bool = False,
) -> list[dict]:
    """丢弃改写载荷的便捷封装。"""
    result = retrieve_round_with_rewrite(
        query,
        paper_ids,
        top_k,
        wiki_context=wiki_context,
        timings=timings,
        rewrite_enabled=rewrite_enabled,
        reference_intent=reference_intent,
        evaluation_parallel=evaluation_parallel,
    )
    return result[0] if timings is None else result


def _hybrid_search_compat(query: str, kwargs: dict) -> list[dict]:
    """Call hybrid search while retaining supported optional diagnostic arguments."""
    attempted = dict(kwargs)
    optional = ("sparse_query", "diagnostics", "timings", "allow_concurrent")
    while True:
        try:
            return hybrid_search(query, **attempted)
        except TypeError as exc:
            message = str(exc)
            unsupported = next(
                (name for name in optional if name in attempted and name in message), None
            )
            if unsupported is None:
                raise
            attempted.pop(unsupported)


def _diversify_by_paper(chunks: list[dict], *, top_k: int) -> list[dict]:
    """单篇限额内保留强结果, 防一篇论文占满窗口; 溢出块垫底补位。"""
    selected: list[dict] = []
    overflow: list[dict] = []
    counts: dict[str, int] = {}
    for chunk in chunks:
        paper_id = chunk.get("paper_id")
        if paper_id and counts.get(paper_id, 0) >= _MAX_CHUNKS_PER_PAPER:
            overflow.append(chunk)
            continue
        selected.append(chunk)
        if paper_id:
            counts[paper_id] = counts.get(paper_id, 0) + 1
        if len(selected) >= top_k:
            return selected[:top_k]

    for chunk in overflow:
        selected.append(chunk)
        if len(selected) >= top_k:
            break
    return selected[:top_k]


__all__ = ["infer_modalities", "retrieve_round", "retrieve_round_with_rewrite"]
