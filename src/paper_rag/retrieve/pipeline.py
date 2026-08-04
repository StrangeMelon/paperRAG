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

from ..utils.logger import get_logger
from .hybrid import hybrid_search
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
_MAX_CHUNKS_PER_PAPER = 2


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
    wiki_context: dict | None = None,
) -> tuple[list[dict], dict]:
    """一轮检索。返回 (重排后的 chunks, 改写载荷)。

    ``rewrite_fn`` 可注入(测试免打补丁); 默认用 rag.query_rewrite.rewrite,
    该模块未重建时回退恒等改写(warning 是诚实信号)。
    """
    if rewrite_fn is None:
        try:
            from ..rag.query_rewrite import rewrite as rewrite_fn  # 局部导入避免环
        except Exception as e:
            log.warning(f"query_rewrite unavailable, using identity rewrite: {e}")
            rewrite_fn = _identity_rewrite

    try:
        rw = rewrite_fn(query, wiki_context=wiki_context)
    except TypeError as exc:
        if "wiki_context" not in str(exc):
            raise
        rw = rewrite_fn(query)
    modalities = infer_modalities(query)
    pooled: dict[str, dict] = {}
    for q in rw["dense_queries"]:
        search_specs = [(None, top_k)]
        search_specs.extend((modality, top_k) for modality in modalities)
        for modality, k in search_specs:
            hits = hybrid_search(q, top_k=k, paper_ids=paper_ids, modality=modality)
            for hit in hits:
                cid = hit.get("chunk_id")
                if not cid:
                    continue
                if cid not in pooled or hit.get("score_rrf", 0) > pooled[cid].get("score_rrf", 0):
                    pooled[cid] = hit
    candidates = list(pooled.values())
    candidates.sort(key=lambda x: x.get("score_rrf", 0), reverse=True)
    candidates = candidates[: top_k * 3]
    ranked = _rerank(query, candidates, top_k=top_k * 3)
    return _diversify_by_paper(ranked, top_k=top_k), rw


def retrieve_round(
    query: str,
    paper_ids: list[str] | None,
    top_k: int,
    *,
    wiki_context: dict | None = None,
) -> list[dict]:
    """丢弃改写载荷的便捷封装。"""
    chunks, _ = retrieve_round_with_rewrite(query, paper_ids, top_k, wiki_context=wiki_context)
    return chunks


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
